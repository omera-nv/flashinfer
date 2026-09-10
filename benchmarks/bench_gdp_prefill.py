"""End-to-end microbenchmark for Gated DeltaProduct prefill.

Columns
-------
  flashinfer   chunk_gated_delta_product
  vllm         chunk_gated_delta_rule (vLLM's flash-linear-attention fork) on
               the n_h-expanded sequence

GDP is GDN on a sequence n_h times longer: the forget gate applies on the FIRST
micro-step of each token (neutral elsewhere), the query on the LAST, and the
output is read back every n_h-th row.  FLA has no delta-product op, so that
expansion is charged to the vllm column.  FlashInfer's kernel is n_h-aware and
indexes q/gate/output per real token, so it needs no expansion at all -- the
vllm column is therefore paying for something flashinfer genuinely avoids, not
for bookkeeping both sides do.

Gate conventions differ and are NOT interchangeable: flashinfer's ``g`` is the
linear-space forget gate alpha in (0, 1) ("all ones" = no decay), FLA's is in
log space.  Feeding one to the other yields NaN, so both runners are driven
from the SAME alpha and the vllm side takes its log.

Run --check first: it reports relative L2 error against flashinfer.

Examples
--------
python benchmarks/bench_gdp_prefill.py --check
python benchmarks/bench_gdp_prefill.py --num-householder 3 --head-size 128 \
    --v-head-size 64 --batch-size 1 4 --seq-len 4096 --iters 50
"""

import argparse

import numpy as np
import torch

from flashinfer.gdn_product import chunk_gated_delta_product
from flashinfer.testing import bench_gpu_time

try:
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    HAS_VLLM = True
except ImportError:  # pragma: no cover - benchmark-only dependency
    HAS_VLLM = False

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _gdp_flops(total_tokens, n_h, num_heads, K, V):
    """Dominant state-update + readout terms; K*V because the state is [V, K]."""
    return (4 * n_h + 2) * total_tokens * num_heads * K * V


def _make_inputs(B, L, n_h, HQ, HV, K, V, dtype, device, seed):
    """Equal-length packed sequences, one state per sequence.

    ``alpha`` is the linear-space forget gate both sides are driven from; k is
    L2-normalised because the delta rule expects unit keys.
    """
    torch.manual_seed(seed)
    T = B * L
    with device:
        k = torch.randn(T, n_h, HQ, K, dtype=torch.float32)
        return dict(
            q=torch.randn(T, HQ, K, dtype=dtype) * 0.1,
            k=torch.nn.functional.normalize(k, p=2.0, dim=-1).to(dtype),
            v=torch.randn(T, n_h, HV, V, dtype=dtype) * 0.1,
            # keep alpha well inside (0, 1): 1.0 is neutral, 0.0 would zero the
            # state and hide any gate handling difference between the two.
            alpha=torch.rand(T, HV, dtype=torch.float32) * 0.5 + 0.25,
            beta=torch.rand(T, n_h, HV, dtype=torch.float32).sigmoid(),
            cu_seqlens=torch.arange(0, T + 1, L, dtype=torch.int64),
        )


def _repeat_to_hv(x, HQ, HV):
    """vLLM's kernel needs one head count. Timed, since a vLLM caller pays this
    every call; FlashInfer is GQA-native. --num-q-heads == --num-v-heads
    removes the confound.
    """
    return x if HQ == HV else x.repeat_interleave(HV // HQ, dim=-2)


def _flashinfer_runner(t, B, L, n_h, HQ, HV, K, V, dtype, device):
    T = B * L
    HO = max(HQ, HV)
    out = torch.empty(T, HO, V, dtype=dtype, device=device)
    state = torch.empty(B, HO, V, K, dtype=torch.float32, device=device)

    # GDP no longer needs expansion scratch: the kernel indexes q/gate/output
    # per real token, so the only n_h-scaled buffers are k/v/beta, which the
    # caller already owns.
    scratch = 0

    def run():
        chunk_gated_delta_product(
            t["q"],
            t["k"],
            t["v"],
            t["alpha"],
            t["beta"],
            scale=K**-0.5,
            output_final_state=True,
            cu_seqlens=t["cu_seqlens"],
            output=out,
            output_state=state,
        )
        return out

    return run, scratch / (1024**2)


def _vllm_runner(t, B, L, n_h, HQ, HV, K, V, dtype, device):
    """FLA chunked GDN over the n_h-expanded sequence.

    Scratch is preallocated (flashinfer's is too); the expansion WORK -- the
    GQA repeat and the two scatters -- is inside the timed region.
    """
    T = B * L
    TE = T * n_h
    q_exp = torch.zeros(1, TE, HV, K, dtype=dtype, device=device)
    g_exp = torch.zeros(1, TE, HV, dtype=torch.float32, device=device)  # log space
    cu_exp = (t["cu_seqlens"] * n_h).to(torch.int32)
    scratch = sum(x.numel() * x.element_size() for x in (q_exp, g_exp))

    def run():
        # gate on the first micro-step of each token, neutral (log 1 = 0)
        # elsewhere; query on the last.  Both scatters hit the same slots every
        # call, so the zero fill outside stays valid.
        g_exp[0, 0::n_h] = t["alpha"].log()
        q_exp[0, n_h - 1 :: n_h] = _repeat_to_hv(t["q"], HQ, HV)
        o, _ = chunk_gated_delta_rule(
            q_exp,
            _repeat_to_hv(t["k"].reshape(TE, HQ, K), HQ, HV).reshape(1, TE, HV, K),
            t["v"].reshape(1, TE, HV, V),
            g_exp,
            t["beta"].reshape(1, TE, HV),
            scale=K**-0.5,
            output_final_state=True,
            cu_seqlens=cu_exp,
        )
        return o[0, n_h - 1 :: n_h]

    return run, scratch / (1024**2)


def _runners(args, B, L, n_h, K, V, dtype):
    HQ, HV = args.num_q_heads, args.num_v_heads
    device = torch.device("cuda")
    t = _make_inputs(B, L, n_h, HQ, HV, K, V, dtype, device, args.seed)
    r = {"flashinfer": _flashinfer_runner(t, B, L, n_h, HQ, HV, K, V, dtype, device)}
    if HAS_VLLM:
        r["vllm"] = _vllm_runner(t, B, L, n_h, HQ, HV, K, V, dtype, device)
    return t, r


def _configs(args):
    for dt in args.dtype:
        for K in args.head_size:
            for n_h in args.num_householder:
                for B in args.batch_size:
                    for L in args.seq_len:
                        yield dt, K, args.v_head_size or K, n_h, B, L


def _check(args):
    print("correctness check -- relative L2 vs flashinfer")
    for dt, K, V, n_h, B, L in _configs(args):
        _, r = _runners(args, B, L, n_h, K, V, DTYPES[dt])
        if "vllm" not in r:
            print("  vllm unavailable -- nothing to compare")
            return
        ref = r["flashinfer"][0]().float().clone()
        got = r["vllm"][0]().float()
        if got.shape != ref.shape:
            verdict = f"SHAPE {tuple(got.shape)} vs {tuple(ref.shape)}"
        else:
            rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            verdict = f"{rel:.4f}" + ("" if rel < 0.02 else "  <-- MISMATCH")
        print(f"  {dt:9s} K={K:3d} V={V:3d} n_h={n_h} B={B:4d} L={L:6d}: {verdict}")


def main():
    p = argparse.ArgumentParser(
        description="FlashInfer GDP prefill vs vLLM's FLA fork on the expanded sequence"
    )
    p.add_argument("--batch-size", type=int, nargs="+", default=[1, 8])
    p.add_argument("--seq-len", type=int, nargs="+", default=[128, 512])
    p.add_argument("--num-householder", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--head-size", type=int, nargs="+", choices=[64, 128], default=[128])
    p.add_argument(
        "--v-head-size",
        type=int,
        choices=[64, 128],
        default=None,
        help="V head size when it differs from K (e.g. K=128, V=64). "
        "Defaults to the K head size.",
    )
    p.add_argument("--num-q-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=["bfloat16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
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
            f"NOTE: GQA {args.num_q_heads}->{args.num_v_heads}. FLA needs one head "
            "count, so its q/k repeat IS timed (a real per-call cost for a vLLM "
            f"caller; FlashInfer is GQA-native). --num-q-heads {args.num_v_heads} "
            "removes the confound.\n"
        )

    if args.check:
        _check(args)
        return

    print(
        f"HQ={args.num_q_heads} HV={args.num_v_heads} graph={args.cuda_graph} "
        f"cold_l2={args.cold_l2} iters={args.iters}"
    )
    names = ["flashinfer"] + (["vllm"] if HAS_VLLM else [])
    head = f"{'dtype':>9} {'K':>4} {'V':>4} {'n_h':>4} {'batch':>6} {'seq':>7}"
    for n in names:
        head += f" {n + '(ms)':>15}"
    print(head + f" {'speedup':>9} {'tokens/s':>12} {'TFLOP/s':>9} {'scratch MiB':>12}")

    for dt, K, V, n_h, B, L in _configs(args):
        try:
            _, r = _runners(args, B, L, n_h, K, V, DTYPES[dt])
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
                f"{dt:>9} {K:4d} {V:4d} {n_h:4d} {B:6d} {L:7d}   "
                f"FAILED: {type(exc).__name__}: {exc}"
            )
            continue
        ms = got["flashinfer"][0]
        tok = B * L * 1e3 / ms
        tflops = _gdp_flops(B * L, n_h, args.num_v_heads, K, V) / ms / 1e9
        line = f"{dt:>9} {K:4d} {V:4d} {n_h:4d} {B:6d} {L:7d}"
        for n in names:
            line += f" {got[n][0]:15.4f}"
        line += f" {got['vllm'][0] / ms:8.2f}x" if HAS_VLLM else f" {'-':>9}"
        print(line + f" {tok:12.0f} {tflops:9.2f} {got['flashinfer'][1]:12.1f}")


if __name__ == "__main__":
    main()
