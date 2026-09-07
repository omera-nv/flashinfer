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

Neither project ships a DeltaProduct decode kernel -- vLLM has no
``num_householder`` anywhere -- so both columns are the same trick: expand one
real token into ``n_h`` micro-steps (gate neutral on all but the first, query
zero on all but the last) and hand the result to a GDN kernel.

  flashinfer   gated_delta_product_mtp -> gated_delta_rule_mtp
  vllm         the same expansion -> fused_sigmoid_gating_delta_rule_update,
               which is what vLLM's own speculative-decode path calls
               (qwen_gdn_linear_attn.py ~1381)

Both write a per-token state snapshot and skip the intermediate micro-steps via
a slot their scatters guard on, so this compares the same work rather than two
different amounts of it.

One asymmetry worth knowing: vLLM overloads ``ssm_state_indices`` for BOTH the
initial-state read (at ``num_accepted_tokens - 1``) and the per-token write,
while FlashInfer takes separate ``initial_state_indices`` and
``ssm_state_indices``. The read slot therefore doubles as a write target on the
vLLM side, giving it T+1 writes to our T -- small at these T, but real.

Run --check first. A latency comparison between kernels that disagree
numerically measures nothing; it reports RELATIVE error, because these outputs
are order 1e-3 and an absolute tolerance passes even when they are
uncorrelated.

Examples
--------
python benchmarks/bench_gdp_decode.py --check
python benchmarks/bench_gdp_decode.py --num-householder 2 3 --head-size 64 128 \
    --draft-len 1 2 4 --batch-size 32 128 512 --iters 200 --cold-l2
"""

import argparse

import numpy as np
import torch

from flashinfer.gdn_product import GATE_NEUTRAL_A_SENTINEL, gated_delta_product_mtp
from flashinfer.testing import bench_gpu_time

try:
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    HAS_VLLM = True
except ImportError:  # pragma: no cover - benchmark-only dependency
    HAS_VLLM = False

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _make_inputs(B, T, n_h, HQ, HV, K, dtype, device, seed):
    """Dense decode inputs plus a state pool.

    Magnitudes and dtypes follow tests/gdn/test_decode_delta_product.py: ``a``
    and ``b`` are the MODEL dtype (raw logits the kernels convert internally),
    only A_log/dt_bias are fp32, everything scaled to ~0.1 so the gate stays in
    a sane range.

    Pool layout, all disjoint: row 0 unused (0 is a skip sentinel on both
    sides), then one initial state per request, then one snapshot per real
    token.
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
        )


def _repeat_to_hv(x, HQ, HV):
    """vLLM's kernel needs one head count; its layer repeats q/k for GQA.

    Timed, because a vLLM caller pays it every step -- FlashInfer is GQA-native.
    Pass --num-q-heads == --num-v-heads to remove the confound.
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


def _runners(args, B, T, n_h, K, dtype):
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = torch.device("cuda")
    t = _make_inputs(B, T, n_h, HQ, HV, K, dtype, device, args.seed)
    r = {"flashinfer": _flashinfer_runner(t, B, T, n_h, HQ, HV, K, dtype, device)}
    if HAS_VLLM:
        r["vllm"] = _vllm_runner(t, B, T, n_h, HQ, HV, K, dtype, device)
    return r


def _configs(args):
    for dt in args.dtype:
        for K in args.head_size:
            for n_h in args.num_householder:
                for B in args.batch_size:
                    for T in args.draft_len:
                        yield dt, K, n_h, B, T


def _check(args):
    print("correctness check -- relative L2 vs flashinfer")
    for dt, K, n_h, B, T in _configs(args):
        r = _runners(args, B, T, n_h, K, DTYPES[dt])
        ref = r["flashinfer"][0]().float().clone()
        cols = []
        for name, (fn, _) in r.items():
            if name == "flashinfer":
                continue
            got = fn().float()
            if got.shape != ref.shape:
                cols.append(f"{name}=SHAPE{tuple(got.shape)}")
                continue
            rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            cols.append(f"{name}={rel:.4f}" + ("" if rel < 0.02 else " <-- MISMATCH"))
        print(
            f"  {dt:9s} D={K:3d} n_h={n_h} B={B:4d} T={T}: "
            + ("  ".join(cols) or "(vllm unavailable)")
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
            r = _runners(args, B, T, n_h, K, DTYPES[dt])
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
