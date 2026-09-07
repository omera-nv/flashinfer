"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""FlashInfer GDP decode vs vLLM's GDN kernel driven by the same expansion.

GDP decode is GDN on a sequence n_h times longer: gate neutral on all but the
first micro-step of a token, query zero on all but the last. Both columns do
that expansion, then call a GDN kernel:

  flashinfer   gated_delta_product_mtp
  vllm         fused_sigmoid_gating_delta_rule_update (its spec-decode path)

Both write a per-token state snapshot and skip the intermediates. vLLM gets
T+1 writes to our T, because its read slot doubles as a write target.

Run --check first; it reports relative error, since these outputs are ~1e-3 and
an absolute tolerance would pass on uncorrelated results.

Examples
--------
python benchmarks/bench_gdp_decode.py --check
python benchmarks/bench_gdp_decode.py --num-householder 2 3 --head-size 64 128 \
    --draft-len 1 2 4 --batch-size 32 128 512 --iters 200 --cold-l2
"""

import argparse
import inspect

import numpy as np
import torch

from flashinfer.gdn_product import GATE_NEUTRAL_A_SENTINEL, gated_delta_product_mtp
from flashinfer.testing import bench_gpu_time

try:
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm import _custom_ops as vllm_ops
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        layer_norm_fwd,
    )

    HAS_VLLM = True
except ImportError:  # pragma: no cover - benchmark-only dependency
    HAS_VLLM = False

# vLLM's hand-written CUDA path (csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu).
# It fuses qkv unpack + decode + gated RMSNorm + output gate into ONE launch, so it
# is only comparable against flashinfer's kernel PLUS a separate norm -- which is
# what the `--fused` columns do. Requires vLLM's C extension to be built; vLLM
# itself falls back to the triton path when it is not.
HAS_VLLM_FUSED = HAS_VLLM and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
# The wrapper's signature moves between vLLM versions (output_gate_activation is
# present on some, absent on others) and the INSTALLED vllm need not match the
# checkout you are reading. Filter kwargs against the runtime signature.
_FUSED_KW = (
    set(inspect.signature(vllm_ops.fused_gdn_decode_post_conv_mtp).parameters)
    if HAS_VLLM_FUSED
    else set()
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _make_inputs(B, T, n_h, HQ, HV, K, dtype, device, seed):
    """Dense decode inputs plus a state pool.

    ``a``/``b`` are the model dtype (raw logits the kernels convert internally),
    A_log/dt_bias fp32, everything ~0.1 to keep the gate in range.

    Pool rows, disjoint: 0 unused (it is a skip sentinel), then one initial
    state per request, then one snapshot per real token.
    """
    torch.manual_seed(seed)
    with device:
        return dict(
            q=torch.randn(B, T, HQ, K, dtype=dtype) * 0.1,
            k=torch.randn(B, T, n_h, HQ, K, dtype=dtype) * 0.1,
            v=torch.randn(B, T, n_h, HV, K, dtype=dtype) * 0.1,
            a=torch.randn(B, T, HV, dtype=dtype) * 0.1,
            b=torch.randn(B, T, n_h, HV, dtype=dtype) * 0.1,
            A_log=torch.randn(HV, dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(HV, dtype=torch.float32) * 0.1,
            pool=torch.randn(1 + B + B * T, HV, K, K, dtype=torch.float32) * 0.01,
            initial_idx=torch.arange(1, 1 + B, dtype=torch.int32),
            ssm_idx=torch.arange(1 + B, 1 + B + B * T, dtype=torch.int32).reshape(B, T),
            # output gate for the fused comparison: ONE tensor, so both sides
            # gate the same values. Drawn per real token; the vllm side expands
            # it, and the intermediate micro-steps it invents are discarded.
            gate=torch.randn(B, T, HV, K, dtype=dtype) * 0.1,
            norm_weight=torch.ones(K, dtype=dtype),
        )


def _repeat_to_hv(x, HQ, HV):
    """vLLM's kernel needs one head count. Timed, since a vLLM caller pays this
    every step; FlashInfer is GQA-native. --num-q-heads == --num-v-heads
    removes the confound.
    """
    return x if HQ == HV else x.repeat_interleave(HV // HQ, dim=-2)


def _flashinfer_runner(t, B, T, n_h, HQ, HV, K, dtype, device):
    """Preallocated expansion scratch, refilled per call -- as serving does."""
    TN = T * n_h
    out = torch.empty(B, T, HV, K, dtype=dtype, device=device)
    exp_q = torch.empty(B, TN, HQ, K, dtype=dtype, device=device) if n_h > 1 else None
    exp_a = torch.empty(B, TN, HV, dtype=dtype, device=device) if n_h > 1 else None
    exp_o = torch.empty(B, TN, HV, K, dtype=dtype, device=device) if n_h > 1 else None
    mib = sum(
        x.numel() * x.element_size() for x in (exp_q, exp_a, exp_o) if x is not None
    ) / (1024**2)

    def run():
        gated_delta_product_mtp(
            t["q"],
            t["k"],
            t["v"],
            t["pool"],
            t["initial_idx"],
            t["A_log"],
            t["a"],
            t["dt_bias"],
            t["b"],
            scale=K**-0.5,
            output=out,
            ssm_state_indices=t["ssm_idx"],
            disable_state_update=False,
            expanded_q=exp_q,
            expanded_a=exp_a,
            expanded_output=exp_o,
        )
        return out

    return run, mib


def _vllm_runner(t, B, T, n_h, HQ, HV, K, dtype, device):
    TN = T * n_h
    HE = HV if HQ != HV else HQ
    cu = torch.arange(0, B * TN + 1, TN, dtype=torch.int32, device=device)
    e_q = torch.zeros(B, TN, HE, K, dtype=dtype, device=device)
    e_a = torch.empty(B, TN, HV, dtype=dtype, device=device)

    idx = torch.zeros(B, TN, dtype=torch.int32, device=device)
    idx[:, 0] = t["initial_idx"]
    idx[:, n_h - 1 :: n_h] = t["ssm_idx"]
    accepted = torch.ones(B, dtype=torch.int32, device=device)

    def run():
        # expansion charged here, exactly as it is charged to flashinfer
        e_q.zero_()
        e_q[:, n_h - 1 :: n_h] = _repeat_to_hv(t["q"], HQ, HV)
        e_a.fill_(GATE_NEUTRAL_A_SENTINEL)
        e_a[:, ::n_h] = t["a"]
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=t["A_log"],
            a=e_a.reshape(1, -1, HV),
            b=t["b"].reshape(1, -1, HV),
            dt_bias=t["dt_bias"],
            q=e_q.reshape(1, -1, HE, K),
            k=_repeat_to_hv(t["k"].flatten(1, 2), HQ, HV).reshape(1, -1, HE, K),
            v=t["v"].flatten(1, 2).reshape(1, -1, HV, K),
            scale=K**-0.5,
            initial_state=t["pool"],
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=idx,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        return o.reshape(B, TN, HV, K)[:, n_h - 1 :: n_h]

    return run, 0.0


def _norm_gate(x, gate, out, weight, eps):
    """Gated RMSNorm + output gate, called the way vLLM's non-fused path calls it
    (qwen_gdn_linear_attn.py:1833). This is the work their fused kernel absorbs,
    so flashinfer has to pay it explicitly for the comparison to mean anything."""
    x2, g2, o2 = (t.reshape(-1, t.shape[-1]) for t in (x, gate, out))
    layer_norm_fwd(
        x2,
        weight,
        None,
        eps,
        z=g2,
        out=o2,
        group_size=x.shape[-1],
        # The fused kernel sums squares over the RAW decode output and applies
        # the gate afterwards (fused_gdn_decode_kernel.cu:345-362) -- that is
        # norm(x) * silu(z). False would fold the gate into the sum of squares
        # and normalise by a different quantity.
        norm_before_gate=True,
        is_rms_norm=True,
        activation="silu",
    )


def _flashinfer_fused_runner(t, B, T, n_h, HQ, HV, K, dtype, device):
    """flashinfer decode + the norm/gate their CUDA kernel fuses in."""
    inner, mib = _flashinfer_runner(t, B, T, n_h, HQ, HV, K, dtype, device)
    final = torch.empty(B, T, HV, K, dtype=dtype, device=device)

    def run():
        _norm_gate(inner(), t["gate"], final, t["norm_weight"], 1e-5)
        return final

    return run, mib


def _vllm_fused_runner(t, B, T, n_h, HQ, HV, K, dtype, device):
    """vLLM's fused CUDA kernel: unpack + decode + norm + gate, one launch.

    Takes mixed_qkv packed as [tokens, 2*key_dim + value_dim] post-conv, and a
    1D state_indices -- one slot per REQUEST, with num_accepted_tokens driving
    rollback, rather than the per-token scatter the other paths use.
    """
    TN = T * n_h
    kd, vd = HQ * K, HV * K
    mixed_qkv = torch.cat(
        [
            t["q"].new_zeros(B * TN, kd),
            _repeat_to_hv(t["k"].flatten(1, 2), HQ, HQ).reshape(B * TN, kd),
            t["v"].flatten(1, 2).reshape(B * TN, vd),
        ],
        dim=-1,
    ).contiguous()
    mixed_qkv[:, :kd] = 0
    mixed_qkv[:, :kd].view(B, TN, HQ, K)[:, n_h - 1 :: n_h] = t["q"]
    a = torch.full((B * TN, HV), GATE_NEUTRAL_A_SENTINEL, dtype=dtype, device=device)
    a.view(B, TN, HV)[:, ::n_h] = t["a"]
    cu = torch.arange(0, B * TN + 1, TN, dtype=torch.int32, device=device)
    # [N, S], not 1D: the kernel wants one column per spec token
    # (tests/kernels/mamba/test_gdn_fused_mtp.py builds torch.ones(1, SPEC_TOKENS)).
    state_idx_2d = t["initial_idx"][:, None].expand(B, TN).contiguous()
    out = torch.empty(B * TN, HV, K, dtype=dtype, device=device)
    # same gate as flashinfer at the real-token rows; intermediates are dropped
    gate = torch.zeros(B, TN, HV, K, dtype=dtype, device=device)
    gate[:, n_h - 1 :: n_h] = t["gate"]
    gate = gate.reshape(B * TN, HV, K)

    def run():
        # the vllm._custom_ops wrapper, not torch.ops._C directly: the raw op
        # takes 14 args and no output_gate_activation, which the wrapper applies.
        kw = dict(
            mixed_qkv=mixed_qkv,
            a=a,
            b=t["b"].reshape(B * TN, HV),
            A_log=t["A_log"],
            dt_bias=t["dt_bias"],
            state_indices=state_idx_2d,
            cu_seqlens=cu,
            num_accepted_tokens=torch.full((B,), TN, dtype=torch.int32, device=device),
            state=t["pool"],
            output_gate=gate,
            norm_weight=t["norm_weight"],
            out=out,
            scale=K**-0.5,
            norm_eps=1e-5,
            output_gate_activation="silu",
        )
        vllm_ops.fused_gdn_decode_post_conv_mtp(
            **{k: v for k, v in kw.items() if k in _FUSED_KW}
        )
        return out.view(B, TN, HV, K)[:, n_h - 1 :: n_h]

    return run, 0.0


def _runners(args, B, T, n_h, K, dtype):
    """Build one runner per comparable column."""
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = torch.device("cuda")
    t = _make_inputs(B, T, n_h, HQ, HV, K, dtype, device, args.seed)
    # snapshot BEFORE the availability probe below, which mutates the pool
    t["pool_pristine"] = t["pool"].clone()
    r = {"flashinfer": _flashinfer_runner(t, B, T, n_h, HQ, HV, K, dtype, device)}
    if HAS_VLLM:
        r["vllm"] = _vllm_runner(t, B, T, n_h, HQ, HV, K, dtype, device)
    if HAS_VLLM_FUSED:
        # The fused kernel's supported HV/H ratios are version-dependent (0.28.0
        # allows 1 only; {1,2,3,4,8} landed later), and it validates shapes
        # internally. Probe once rather than encode a version's rules.
        fused = _vllm_fused_runner(t, B, T, n_h, HQ, HV, K, dtype, device)
        try:
            fused[0]()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if _runners.warned is None:
                _runners.warned = True
                print(f"NOTE: vllm fused kernel unavailable for this shape -- {exc}")
        else:
            # end-to-end pair: both sides produce the final layer output
            r["flashinfer+norm"] = _flashinfer_fused_runner(
                t, B, T, n_h, HQ, HV, K, dtype, device
            )
            r["vllm-fused"] = fused
    return t, r


_runners.warned = None


def _configs(args):
    for dt in args.dtype:
        for K in args.head_size:
            for n_h in args.num_householder:
                for B in args.batch_size:
                    for T in args.draft_len:
                        yield dt, K, n_h, B, T


# Which columns are comparable with which. The fused pair includes the gated
# RMSNorm and output gate; the plain pair does not, so they are NOT comparable
# across groups -- only within one.
_CHECK_GROUPS = [
    ("decode only", "flashinfer", ["vllm"]),
    ("+ norm/gate", "flashinfer+norm", ["vllm-fused"]),
]


def _check(args):
    print("correctness check -- relative L2, within comparable groups")
    for dt, K, n_h, B, T in _configs(args):
        t, r = _runners(args, B, T, n_h, K, DTYPES[dt])
        for label, ref_name, others in _CHECK_GROUPS:
            present = [n for n in others if n in r]
            if ref_name not in r or not present:
                continue
            # Every runner writes state into the SHARED pool -- flashinfer
            # scatters snapshots, vllm-fused runs inplace_final_state on the
            # rows it reads, and the availability probe already ran one of them.
            # Restore the pool before each call so they all see one initial
            # state; otherwise `got` is computed from whatever `ref` left behind.
            t["pool"].copy_(t["pool_pristine"])
            ref = r[ref_name][0]().float().clone()
            cols = []
            for name in present:
                t["pool"].copy_(t["pool_pristine"])
                got = r[name][0]().float()
                if got.shape != ref.shape:
                    cols.append(f"{name}=SHAPE{tuple(got.shape)}")
                    continue
                rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                cols.append(
                    f"{name}={rel:.4f}" + ("" if rel < 0.02 else " <-- MISMATCH")
                )
            print(
                f"  {dt:9s} D={K:3d} n_h={n_h} B={B:4d} T={T} "
                f"[{label} vs {ref_name}]: " + "  ".join(cols)
            )


def main():
    p = argparse.ArgumentParser(
        description="FlashInfer GDP decode vs vLLM's GDN kernel + the same expansion"
    )
    p.add_argument("--num-householder", type=int, nargs="+", default=[3])
    p.add_argument("--head-size", type=int, nargs="+", choices=[64, 128], default=[128])
    p.add_argument(
        "--draft-len",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="T = num_spec + 1, real tokens per step",
    )
    p.add_argument("--batch-size", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=["bfloat16"])
    p.add_argument("--num-q-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--cold-l2", action="store_true")
    p.add_argument(
        "--check", action="store_true", help="verify the two agree, then exit"
    )
    args = p.parse_args()

    if not HAS_VLLM:
        print(
            "WARNING: vllm not importable -- flashinfer column only.\n"
            "         needs `vllm.third_party.flash_linear_attention`.\n"
        )
    if args.num_q_heads != args.num_v_heads:
        print(
            f"NOTE: GQA {args.num_q_heads}->{args.num_v_heads}. vLLM's kernel needs "
            "one head count, so its q/k repeat IS timed (a real per-step cost for a "
            f"vLLM caller; FlashInfer is GQA-native). --num-q-heads "
            f"{args.num_v_heads} removes the confound.\n"
        )

    if args.check:
        _check(args)
        return

    print(
        f"HQ={args.num_q_heads} HV={args.num_v_heads} graph={args.cuda_graph} "
        f"cold_l2={args.cold_l2} iters={args.iters}"
    )
    names = ["flashinfer"] + (["vllm"] if HAS_VLLM else [])
    head = f"{'dtype':>9} {'D':>4} {'n_h':>4} {'batch':>6} {'T':>3}"
    for n in names:
        head += f" {n + '(ms)':>16}"
    print(head + f" {'speedup':>9} {'scratch MiB':>12}")

    for dt, K, n_h, B, T in _configs(args):
        try:
            t, r = _runners(args, B, T, n_h, K, DTYPES[dt])
            got = {
                n: (
                    float(
                        np.median(
                            bench_gpu_time(
                                fn,
                                dry_run_iters=args.warmup,
                                repeat_iters=args.iters,
                                use_cuda_graph=args.cuda_graph,
                                cold_l2_cache=args.cold_l2,
                            )
                        )
                    ),
                    mib,
                )
                for n, (fn, mib) in r.items()
            }
        except Exception as exc:  # keep the sweep going (OOM, unsupported config)
            print(
                f"{dt:>9} {K:4d} {n_h:4d} {B:6d} {T:3d}   "
                f"FAILED: {type(exc).__name__}: {exc}"
            )
            continue
        line = f"{dt:>9} {K:4d} {n_h:4d} {B:6d} {T:3d}"
        for n in names:
            line += f" {got[n][0]:16.4f}"
        line += (
            f" {got['vllm'][0] / got['flashinfer'][0]:8.2f}x"
            if HAS_VLLM
            else f" {'-':>9}"
        )
        print(line + f" {got['flashinfer'][1]:12.1f}")


if __name__ == "__main__":
    main()
