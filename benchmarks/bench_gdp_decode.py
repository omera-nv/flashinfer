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

"""Microbenchmark: FlashInfer vs FLA for GDN / GDP decode.

Two comparisons, selected with ``--compare``:

``gdn`` -- the prior question: is FlashInfer's decode kernel worth using at all?
    flashinfer   gated_delta_rule_mtp, called directly
    fla          fused_recurrent_gated_delta_rule, called directly
  No DeltaProduct, no expansion on either side. If FLA wins here, the GDP
  wrapper should be built on FLA's kernel and FlashInfer's adds nothing.

``gdp`` -- the product question, at a real num_householder:
    flashinfer      our host expansion over gated_delta_rule_mtp
    fla-recurrent   the SAME expansion over FLA's fused-recurrent GDN kernel
    fla-chunk       FLA's own chunk_gated_delta_product -- the only GDP entry
                    point FLA ships. Chunked/training kernel, so at decode-sized
                    T this is the "use FLA as-is" number.

Fairness rules, applied to every variant:
  * scratch buffers are allocated once, outside the timed region;
  * per-step host work (expansion, GQA repeat, gate/beta math for fla-chunk) is
    INSIDE it, because serving pays it every step;
  * state writes are off on both sides unless --state-mode scatter, which no FLA
    entry point can do at all (see the note it prints).

head_size accepts 64 and 128 -- the sizes gated_delta_rule_mtp validates
(gdn_decode.py asserts `K in (64, 128)`). Pass several to sweep them; the
vec_size the kernel picks is K // 32, so 64 runs at vec_size=2 and 128 at 4.

Run --check first. A latency comparison between kernels that disagree
numerically measures nothing, and the check reports RELATIVE error because these
outputs are order 1e-3.

Examples
--------
python benchmarks/bench_gdp_decode.py --check
python benchmarks/bench_gdp_decode.py --compare gdn --batch-size 32 128 512
python benchmarks/bench_gdp_decode.py --compare gdp --num-householder 3 \
    --dtype bfloat16 float16 --batch-size 32 128 512 --draft-len 1 2 4 --iters 100
"""

import argparse
import json

import numpy as np
import torch

from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.gdn_product import GATE_NEUTRAL_A_SENTINEL, gated_delta_product_mtp
from flashinfer.testing import bench_gpu_time

try:
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    HAS_FLA = True
except ImportError:  # pragma: no cover - benchmark-only dependency
    HAS_FLA = False

# vLLM's vendored FLA, extracted for benchmarking. This is the only FLA-family kernel
# that can do per-token state scatter, so it is the ONLY real baseline for
# --state-mode scatter; upstream FLA cannot do that work at all.
import os
import sys

_FLA_VLLM_PATH = os.environ.get("FLA_VLLM_PATH")
if (
    _FLA_VLLM_PATH is not None
    and os.path.isdir(_FLA_VLLM_PATH)
    and _FLA_VLLM_PATH not in sys.path
):
    sys.path.insert(0, _FLA_VLLM_PATH)
try:
    from fla_vllm_ops import fused_sigmoid_gating_delta_rule_update

    HAS_FLA_VLLM = True
except ImportError:
    HAS_FLA_VLLM = False

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _gdp_flops(num_tokens, num_householder, num_heads, head_size):
    """Recurrence FLOPs for ``num_tokens`` real tokens at ``n_h`` householders.

    Per micro-step: one state read/apply plus one rank-1 update, both
    ``head_size**2`` per head. The trailing ``2`` is the query readout, which
    happens once per real token, not once per micro-step.
    """
    return (4 * num_householder + 2) * num_tokens * num_heads * head_size**2


def _make_inputs(B, T, n_h, HQ, HV, K, V, dtype, device, seed=0):
    """Dense decode inputs plus a partitioned state pool.

    Magnitudes and dtypes follow tests/gdn/test_decode_delta_product.py: ``a``
    and ``b`` are the MODEL dtype (raw logits the kernel converts internally),
    only A_log/dt_bias are fp32, and everything is scaled to ~0.1 so the gate
    stays in a sane range.
    """
    torch.manual_seed(seed)
    with device:
        q = torch.randn(B, T, HQ, K, dtype=dtype) * 0.1
        k = torch.randn(B, T, n_h, HQ, K, dtype=dtype) * 0.1
        v = torch.randn(B, T, n_h, HV, V, dtype=dtype) * 0.1
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=dtype) * 0.1
        b = torch.randn(B, T, n_h, HV, dtype=dtype) * 0.1
        # row 0 unused (0 is a sentinel elsewhere in flashinfer); then initial
        # states, per-real-token snapshots, and scratch -- all disjoint.
        pool = torch.randn(1 + B + B * T + B, HV, V, K, dtype=torch.float32) * 0.01
        initial_idx = torch.arange(1, 1 + B, dtype=torch.int32)
        ssm_idx = torch.arange(1 + B, 1 + B + B * T, dtype=torch.int32).reshape(B, T)
        scratch_idx = torch.arange(1 + B + B * T, 1 + B + B * T + B, dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        dt_bias=dt_bias,
        a=a,
        b=b,
        pool=pool,
        initial_idx=initial_idx,
        ssm_idx=ssm_idx,
        scratch_idx=scratch_idx,
    )


def _repeat_to_hv(x, HQ, HV):
    """FLA needs a single head count; its layer repeats q/k for GQA.

    Timed, because an FLA caller pays it every step -- our kernels are
    GQA-native. Use --num-q-heads == --num-v-heads to remove the confound.
    """
    return x if HQ == HV else x.repeat_interleave(HV // HQ, dim=-2)


def _expand_into(dst_q, dst_a, t, n_h, HQ, HV):
    """Host expansion mirroring gated_delta_product_mtp, for FLA's GDN kernel.

    One real token becomes n_h micro-steps: the gate is neutral on all but the
    first, the query zero on all but the last. k/v/b are already per householder
    and only need flattening (a view, so free).

    Writes into PREALLOCATED buffers and is called INSIDE the timed region,
    because that is the deal our wrapper gets: handed preallocated scratch,
    refilled every call. Serving refills every step too.
    """
    dst_q.zero_()
    dst_q[:, n_h - 1 :: n_h] = _repeat_to_hv(t["q"], HQ, HV)
    dst_a.fill_(GATE_NEUTRAL_A_SENTINEL)
    dst_a[:, ::n_h] = t["a"]


def _fla_state(t):
    """Our pool is [pool, HV, V, K]; FLA wants [N, H, K, V] (state_v_first=False)."""
    return t["pool"][t["initial_idx"].long()].transpose(-1, -2).contiguous()


def _gdn_runners(args, B, T, dtype, K):
    """Raw GDN decode: no DeltaProduct, no expansion on either side."""
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = torch.device("cuda")
    t = _make_inputs(B, T, 1, HQ, HV, K, K, dtype, device, args.seed)
    q, k, v, b = t["q"], t["k"].squeeze(2), t["v"].squeeze(2), t["b"].squeeze(2)
    out = torch.empty(B, T, HV, K, dtype=dtype, device=device)
    st = _fla_state(t)
    runners = {}
    # honour --state-mode here too; hardcoding disable_state_update=True made
    # the scatter comparison measure us skipping the work fla-vllm was doing
    scatter = args.state_mode == "scatter" and T >= 2

    def run_fi():
        gated_delta_rule_mtp(
            q,
            k,
            v,
            t["pool"],
            t["initial_idx"],
            t["A_log"],
            t["a"],
            t["dt_bias"],
            b,
            scale=K**-0.5,
            output=out,
            disable_state_update=not scatter,
            ssm_state_indices=t["ssm_idx"] if scatter else None,
        )
        return out

    runners["flashinfer"] = (run_fi, 0.0)
    if HAS_FLA:

        def run_fla():
            o, _ = fused_recurrent_gated_delta_rule(
                q=_repeat_to_hv(q, HQ, HV),
                k=_repeat_to_hv(k, HQ, HV),
                v=v,
                beta=b,
                A_log=t["A_log"],
                g=t["a"],
                dt_bias=t["dt_bias"],
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                scale=K**-0.5,
                initial_state=st,
                output_final_state=False,
            )
            return o

        runners["fla"] = (run_fla, 0.0)
    _add_fla_vllm(runners, args, t, B, T, 1, dtype, args.state_mode == "scatter", K)
    return t, runners


def _gdp_runners(args, B, T, n_h, dtype, K):
    """GDP decode: ours, the same expansion on FLA's GDN kernel, and FLA's GDP."""
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = torch.device("cuda")
    t = _make_inputs(B, T, n_h, HQ, HV, K, K, dtype, device, args.seed)
    TN = T * n_h
    runners = {}

    out = torch.empty(B, T, HV, K, dtype=dtype, device=device)
    exp_q = torch.empty(B, TN, HQ, K, dtype=dtype, device=device) if n_h > 1 else None
    exp_a = torch.empty(B, TN, HV, dtype=dtype, device=device) if n_h > 1 else None
    exp_out = torch.empty(B, TN, HV, K, dtype=dtype, device=device) if n_h > 1 else None
    scratch_mib = sum(
        x.numel() * x.element_size() for x in (exp_q, exp_a, exp_out) if x is not None
    ) / (1024**2)

    # "scatter" writes a per-REAL-token snapshot (what spec-decode rollback
    # needs). No FLA entry point can do this, so the FLA columns become a floor
    # rather than a baseline. Needs an expanded T of at least 2.
    scatter = args.state_mode == "scatter" and TN >= 2

    def run_ours():
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
            disable_state_update=not scatter,
            ssm_state_indices=t["ssm_idx"] if scatter else None,
            scratch_state_indices=t["scratch_idx"] if scatter else None,
            expanded_q=exp_q,
            expanded_a=exp_a,
            expanded_output=exp_out,
        )
        return out

    runners["flashinfer"] = (run_ours, scratch_mib)
    if not HAS_FLA:
        return t, runners

    st = _fla_state(t)
    HE = HV if HQ != HV else HQ
    e_q = torch.zeros(B, TN, HE, K, dtype=dtype, device=device)
    e_a = torch.empty(B, TN, HV, dtype=dtype, device=device)

    def run_fla_recurrent():
        _expand_into(e_q, e_a, t, n_h, HQ, HV)
        o, _ = fused_recurrent_gated_delta_rule(
            # FLA names the raw pre-activation `g`, not `a`; `a=` lands in
            # **kwargs and is silently ignored.
            q=e_q,
            k=_repeat_to_hv(t["k"].flatten(1, 2), HQ, HV),
            v=t["v"].flatten(1, 2),
            beta=t["b"].flatten(1, 2),
            A_log=t["A_log"],
            g=e_a,
            dt_bias=t["dt_bias"],
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            scale=K**-0.5,
            initial_state=st,
            output_final_state=False,
        )
        return o[:, n_h - 1 :: n_h]

    runners["fla-recurrent"] = (run_fla_recurrent, 0.0)

    # q/g at REAL length, k/v/beta at EXPANDED length (fla/ops/gated_delta_
    # product/chunk.py:340). No fused gating here, so a caller computes g/beta on
    # the host every step; our kernel derives both internally. Timed, not hoisted.
    def run_fla_chunk():
        g = (
            -torch.nn.functional.softplus(t["a"].float() + t["dt_bias"])
            * t["A_log"].exp()
        )
        o, _ = chunk_gated_delta_product(
            q=_repeat_to_hv(t["q"], HQ, HV),
            k=_repeat_to_hv(t["k"].flatten(1, 2), HQ, HV),
            v=t["v"].flatten(1, 2),
            g=g,
            beta=torch.sigmoid(t["b"].float()).flatten(1, 2),
            num_householder=n_h,
            scale=K**-0.5,
            initial_state=st,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return o

    runners["fla-chunk"] = (run_fla_chunk, 0.0)
    _add_fla_vllm(runners, args, t, B, T, n_h, dtype, scatter, K)
    return t, runners


def _add_fla_vllm(runners, args, t, B, T, n_h, dtype, scatter, K):
    """vLLM's spec-decode GDN kernel, driven through the same expansion as ours.

    vLLM has NO DeltaProduct support, so this is our host expansion feeding
    their GDN kernel -- the same trick as fla-recurrent, with vLLM's kernel
    instead of upstream FLA's. There is no vLLM GDP implementation to compare against.

    Varlen: q/k/v are [1, total_tokens, H, D] with cu_seqlens marking request
    boundaries, one "sequence" per request's T*n_h micro-steps.

    vLLM ALWAYS calls this with ssm_state_indices + the pool +
    inplace_final_state=True (qwen_gdn_linear_attn.py ~1381); the no-indices
    path faults. So this exists only under --state-mode scatter, which is also
    the only mode where it is the comparison that matters.
    """
    if not HAS_FLA_VLLM or not scatter:
        return
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = t["q"].device
    TN = T * n_h
    pool_kv = t["pool"].transpose(-1, -2).contiguous()
    cu = torch.arange(0, B * TN + 1, TN, dtype=torch.int32, device=device)
    e_q = torch.zeros(B, TN, HV, K, dtype=dtype, device=device)
    e_a = torch.empty(B, TN, HV, dtype=dtype, device=device)

    # Two index sets -- the gap between them IS the value of the scatter guard,
    # measured rather than inferred:
    #
    #  scratch -- intermediates point at a throwaway pool row, so all TN
    #    snapshots get written. This is what OUR wrapper is forced to do:
    #    gdn_decode_mtp.py:740 scatters unconditionally, with no sentinel
    #    (even though the same file guards its request-level index at :325).
    #
    #  guard -- intermediates get 0, which vLLM's kernel SKIPS
    #    (`if final_state_idx > 0`, fused_recurrent.py:158). Only the n_h-th row
    #    of each token is written, so (n_h-1)/n_h of the state traffic vanishes.
    #    Pool row 0 is reserved-unused here, so 0 is a safe sentinel.
    idx_scratch = t["scratch_idx"][:, None].expand(B, TN).contiguous().to(torch.int32)
    idx_scratch[:, n_h - 1 :: n_h] = t["ssm_idx"]
    idx_guard = torch.zeros(B, TN, dtype=torch.int32, device=device)
    idx_guard[:, n_h - 1 :: n_h] = t["ssm_idx"]

    def _mk(idx):
        def run():
            _expand_into(e_q, e_a, t, n_h, HQ, HV)
            ret = fused_sigmoid_gating_delta_rule_update(
                A_log=t["A_log"],
                a=e_a.reshape(1, -1, HV),
                b=t["b"].reshape(1, -1, HV),
                dt_bias=t["dt_bias"],
                q=e_q.reshape(1, -1, HV, K),
                k=_repeat_to_hv(t["k"].flatten(1, 2), HQ, HV).reshape(1, -1, HV, K),
                v=t["v"].flatten(1, 2).reshape(1, -1, HV, K),
                scale=K**-0.5,
                initial_state=pool_kv,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )
            o = ret[0] if isinstance(ret, tuple) else ret
            # [1, B*TN, HV, K] -> one row per REAL token, as every variant does
            return o.reshape(B, TN, HV, K)[:, n_h - 1 :: n_h]

        return run

    runners["fla-vllm"] = (_mk(idx_scratch), 0.0)
    if n_h > 1:
        runners["fla-vllm-guard"] = (_mk(idx_guard), 0.0)


def _runners(args, mode, B, T, n_h, dtype, K):
    if mode == "gdn":
        return _gdn_runners(args, B, T, dtype, K)
    return _gdp_runners(args, B, T, n_h, dtype, K)


def _check(args):
    """Confirm the variants agree before any timing is believed.

    RELATIVE error, not absolute. These outputs are order 1e-3, so an absolute
    tolerance like the test suite's atol=1e-2 passes even when the results are
    uncorrelated -- which is how a wrong-output bug hid here once already.
    """
    print("correctness check -- relative L2 vs flashinfer")
    cases = [
        ("gdn", 4, 2, 1),
        ("gdn", 8, 4, 1),
        ("gdp", 4, 1, 2),
        ("gdp", 4, 2, 3),
        ("gdp", 8, 4, 3),
    ]
    for dt_name in args.dtype:
        for K in args.head_size:
            for mode, B, T, n_h in cases:
                _check_one(args, mode, dt_name, K, B, T, n_h)


def _check_one(args, mode, dt_name, K, B, T, n_h):
    """Relative L2 of every variant against flashinfer, for one config."""
    _, runners = _runners(args, mode, B, T, n_h, DTYPES[dt_name], K)
    ref = runners["flashinfer"][0]().float().clone()
    cols = []
    for name, (fn, _) in runners.items():
        if name == "flashinfer":
            continue
        got = fn().float()
        if got.shape != ref.shape:
            cols.append(f"{name}=SHAPE{tuple(got.shape)}")
            continue
        rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        cols.append(f"{name}={rel:.4f}" + ("" if rel < 0.02 else " <-- MISMATCH"))
    print(
        f"  {dt_name:9s} D={K:3d} {mode} B={B:3d} T={T} n_h={n_h}: " + "  ".join(cols)
    )


def _bench(args, runners, num_tokens, n_h, K):
    res = {}
    for name, (fn, scratch_mib) in runners.items():
        ms = float(
            np.median(
                bench_gpu_time(
                    fn,
                    dry_run_iters=args.warmup,
                    repeat_iters=args.iters,
                    use_cuda_graph=args.cuda_graph,
                    cold_l2_cache=args.cold_l2,
                )
            )
        )
        res[name] = dict(
            latency_ms=ms,
            tokens_per_s=num_tokens * 1e3 / ms,
            tflops=_gdp_flops(num_tokens, n_h, args.num_v_heads, K) / ms / 1e9,
            scratch_mib=scratch_mib,
        )
    return res


def _sweep(args, mode, record, baseline):
    if mode == "gdn":
        names = ["flashinfer"] + (["fla"] if HAS_FLA else [])
        nhs = [1]
    else:
        names = ["flashinfer"] + (["fla-recurrent", "fla-chunk"] if HAS_FLA else [])
        nhs = args.num_householder
    if HAS_FLA_VLLM and args.state_mode == "scatter":
        names.append("fla-vllm")
        if mode == "gdp" and any(n > 1 for n in nhs):
            names.append("fla-vllm-guard")
    print(f"\n=== {mode.upper()}: " + " vs ".join(names) + " ===")
    head = f"{'dtype':>9} {'D':>4} {'batch':>6} {'T':>3} {'n_h':>4}"
    for n in names:
        head += f" {n + '(ms)':>17}"
    head += f" {'speedup':>9} {'scratch MiB':>12}"
    if baseline is not None:
        head += f" {'vs base':>9}"
    print(head)
    regressions = []
    for dt_name in args.dtype:
        for K in args.head_size:
            for B in args.batch_size:
                for T in args.draft_len:
                    for n_h in nhs:
                        _sweep_row(
                            args,
                            mode,
                            names,
                            dt_name,
                            K,
                            B,
                            T,
                            n_h,
                            record,
                            baseline,
                            regressions,
                        )
    for key, base, ours, pct in regressions:
        print(f"  REGRESSION {key}: {base:.4f} -> {ours:.4f} ms ({pct:+.1f}%)")
    return regressions


def _sweep_row(args, mode, names, dt_name, K, B, T, n_h, record, baseline, regressions):
    """One table row: run every variant at this config and print the comparison."""
    try:
        _, runners = _runners(args, mode, B, T, n_h, DTYPES[dt_name], K)
        r = _bench(args, runners, B * T, n_h, K)
    except Exception as exc:  # keep the sweep going (OOM, unsupported config)
        print(
            f"{dt_name:>9} {K:4d} {B:6d} {T:3d} {n_h:4d}   "
            f"FAILED: {type(exc).__name__}: {exc}"
        )
        return
    line = f"{dt_name:>9} {K:4d} {B:6d} {T:3d} {n_h:4d}"
    for n in names:
        line += f" {r[n]['latency_ms']:17.4f}"
    # In scatter mode only fla-vllm does the same work; the upstream-FLA
    # columns write no state and would flatter us.
    speedup_against = (
        ["fla-vllm"]
        if (args.state_mode == "scatter" and "fla-vllm" in names)
        else [n for n in names if n != "flashinfer"]
    )
    others = [r[n]["latency_ms"] for n in speedup_against if n in r]
    line += (
        f" {min(others) / r['flashinfer']['latency_ms']:8.2f}x"
        if others
        else f" {'-':>9}"
    )
    line += f" {r['flashinfer']['scratch_mib']:12.1f}"
    # Record/compare only the flashinfer column: this guard is about OUR kernel
    # not regressing, and the FLA columns move with unrelated triton churn.
    key = f"{mode}|{dt_name}|{K}|{B}|{T}|{n_h}"
    ours = r["flashinfer"]["latency_ms"]
    record[key] = ours
    if baseline is not None:
        base = baseline.get(key)
        if base is None:
            line += f" {'(new)':>9}"
        else:
            pct = (ours - base) / base * 100.0
            line += f" {pct:+8.1f}%"
            if pct > args.regression_pct:
                regressions.append((key, base, ours, pct))
    print(line)


def main():
    p = argparse.ArgumentParser(description="FlashInfer vs FLA: GDN / GDP decode")
    p.add_argument(
        "--compare", nargs="+", choices=["gdn", "gdp"], default=["gdn", "gdp"]
    )
    p.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=["bfloat16"])
    p.add_argument("--batch-size", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument(
        "--draft-len",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="T = num_spec + 1, real tokens per step",
    )
    p.add_argument("--num-householder", type=int, nargs="+", default=[3])
    p.add_argument("--num-q-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument(
        "--head-size",
        type=int,
        nargs="+",
        choices=[64, 128],
        default=[128],
        help="head sizes to sweep; 64 and 128 are what gated_delta_rule_mtp "
        "validates. vec_size is K // 32, so 64 runs at 2 and 128 at 4.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--cold-l2", action="store_true")
    p.add_argument(
        "--state-mode",
        choices=["none", "scatter"],
        default="none",
        help="none: no state writes (fair vs FLA). scatter: per-token snapshots "
        "for spec-decode rollback (no FLA equivalent)",
    )
    p.add_argument(
        "--check", action="store_true", help="verify the variants agree, then exit"
    )
    p.add_argument(
        "--save-baseline",
        metavar="PATH",
        help="write this run's flashinfer latencies to PATH for later comparison",
    )
    p.add_argument(
        "--compare-baseline",
        metavar="PATH",
        help="compare against a saved baseline; exits 1 if any config regresses "
        "by more than --regression-pct. Use this to prove a kernel change left "
        "head_size=128 performance alone.",
    )
    p.add_argument(
        "--regression-pct",
        type=float,
        default=5.0,
        help="percent slowdown that counts as a regression (default: 5)",
    )
    args = p.parse_args()

    if not HAS_FLA:
        print("WARNING: flash-linear-attention not installed -- flashinfer only.")
        print("         pip install flash-linear-attention\n")
    if args.num_q_heads != args.num_v_heads:
        print(
            f"NOTE: GQA {args.num_q_heads}->{args.num_v_heads}. FLA needs one head "
            "count, so its q/k repeat IS timed (a real per-step cost for an FLA "
            f"caller; ours are GQA-native). --num-q-heads {args.num_v_heads} "
            "removes the confound."
        )

    if args.state_mode == "scatter":
        print(
            "NOTE: state=scatter. Read the columns as:\n"
            "  flashinfer      writes all T*n_h snapshots (no sentinel exists)\n"
            "  fla-vllm        same waste, on vLLM's kernel -> LIKE-FOR-LIKE, and\n"
            "                  what the speedup column compares against\n"
            "  fla-vllm-guard  same kernel with 0-sentinel intermediates, which it\n"
            "                  SKIPS -> the gap vs fla-vllm is what a scatter guard\n"
            "                  in gdn_decode_mtp.py:740 would be worth to us\n"
            "  fla / fla-recurrent / fla-chunk  write NO state -> a floor, not a\n"
            "                  baseline; upstream FLA cannot do this work at all"
        )

    if args.check:
        _check(args)
        return

    print(
        f"HQ={args.num_q_heads} HV={args.num_v_heads} D={args.head_size} "
        f"graph={args.cuda_graph} cold_l2={args.cold_l2} state={args.state_mode} "
        f"iters={args.iters}"
    )
    baseline = None
    if args.compare_baseline:
        with open(args.compare_baseline) as f:
            baseline = json.load(f)
        print(
            f"comparing against {args.compare_baseline} "
            f"({len(baseline)} configs, regression threshold "
            f"{args.regression_pct:g}%)"
        )

    record, regressions = {}, []
    for mode in args.compare:
        regressions += _sweep(args, mode, record, baseline)

    if args.save_baseline:
        with open(args.save_baseline, "w") as f:
            json.dump(record, f, indent=1, sort_keys=True)
        print(f"\nwrote {len(record)} configs to {args.save_baseline}")
    if baseline is not None:
        missing = sorted(set(baseline) - set(record))
        if missing:
            print(f"\n{len(missing)} baseline configs not re-run: {missing[:4]}...")
        if regressions:
            print(
                f"\nFAIL: {len(regressions)} regression(s) over "
                f"{args.regression_pct:g}%"
            )
            raise SystemExit(1)
        print(f"\nOK: no config regressed by more than {args.regression_pct:g}%")


if __name__ == "__main__":
    main()
