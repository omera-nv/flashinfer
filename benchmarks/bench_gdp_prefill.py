"""End-to-end microbenchmark for Gated DeltaProduct prefill.

Examples
--------
python benchmarks/bench_gdp_prefill.py --batch-size 1 8 --seq-len 128 512
python benchmarks/bench_gdp_prefill.py --batch-size 1 --seq-len 512 --cuda-graph
"""

import argparse

import numpy as np
import torch

from flashinfer.gdn_product import chunk_gated_delta_product
from flashinfer.testing import bench_gpu_time


def _gdp_flops(total_tokens, num_householder, num_heads, head_size):
    return (4 * num_householder + 2) * total_tokens * num_heads * head_size**2


def _benchmark_case(args, batch_size, seq_len, num_householder):
    B, L, N = batch_size, seq_len, num_householder
    HQ, HV, D = args.num_q_heads, args.num_v_heads, args.head_size
    total_tokens = B * L
    dtype = torch.bfloat16
    device = torch.device("cuda")

    q = torch.randn(total_tokens, HQ, D, dtype=dtype, device=device)
    k = torch.randn(total_tokens, N, HQ, D, dtype=dtype, device=device)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_tokens, N, HV, D, dtype=dtype, device=device)
    g = torch.rand(total_tokens, HV, dtype=torch.float32, device=device)
    beta = torch.rand(total_tokens, N, HV, dtype=torch.float32, device=device)
    cu_seqlens = torch.arange(
        0, total_tokens + 1, L, dtype=torch.int64, device=device
    )
    output = torch.empty(total_tokens, HV, D, dtype=dtype, device=device)
    output_state = torch.empty(B, HV, D, D, dtype=torch.float32, device=device)

    expanded_q = expanded_g = expanded_output = expanded_cu_seqlens = None
    scratch_bytes = 0
    if N > 1:
        expanded_q = torch.empty(total_tokens * N, HQ, D, dtype=dtype, device=device)
        expanded_g = torch.empty(
            total_tokens * N, HV, dtype=torch.float32, device=device
        )
        expanded_output = torch.empty(
            total_tokens * N, HV, D, dtype=dtype, device=device
        )
        expanded_cu_seqlens = torch.empty_like(cu_seqlens)
        scratch_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                expanded_q,
                expanded_g,
                expanded_output,
                expanded_cu_seqlens,
            )
        )

    def run():
        chunk_gated_delta_product(
            q,
            k,
            v,
            g,
            beta,
            scale=D**-0.5,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            output=output,
            output_state=output_state,
            expanded_q=expanded_q,
            expanded_g=expanded_g,
            expanded_output=expanded_output,
            expanded_cu_seqlens=expanded_cu_seqlens,
        )

    times_ms = bench_gpu_time(
        run,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        use_cuda_graph=args.cuda_graph,
        cold_l2_cache=args.cold_l2,
    )
    median_ms = float(np.median(times_ms))
    flops = _gdp_flops(total_tokens, N, HV, D)
    return {
        "latency_ms": median_ms,
        "tokens_per_s": total_tokens * 1e3 / median_ms,
        "tflops": flops / median_ms / 1e9,
        "scratch_mib": scratch_bytes / (1024**2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDP prefill")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--num-householder", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    args = parser.parse_args()

    print(
        f"GDP prefill: HQ={args.num_q_heads}, HV={args.num_v_heads}, "
        f"D={args.head_size}, graph={args.cuda_graph}, cold_l2={args.cold_l2}"
    )
    print(
        f"{'batch':>7} {'seq':>6} {'N':>3} {'latency(ms)':>13} "
        f"{'tokens/s':>13} {'eff TFLOPS':>11} {'scratch MiB':>12}"
    )
    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            for num_householder in args.num_householder:
                result = _benchmark_case(args, batch_size, seq_len, num_householder)
                print(
                    f"{batch_size:7d} {seq_len:6d} {num_householder:3d} "
                    f"{result['latency_ms']:13.3f} {result['tokens_per_s']:13.0f} "
                    f"{result['tflops']:11.2f} {result['scratch_mib']:12.1f}"
                )


if __name__ == "__main__":
    main()
