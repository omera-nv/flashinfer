"""Per-kernel breakdown for one config of bench_gdp_decode.

Reuses that script's runners so the thing profiled is the thing benchmarked.
Prints, per column: launch count, total device time, and the kernels sorted by
cost -- which separates "our kernel is slower" from "we issue more launches".

    python benchmarks/profile_gdp_decode.py --num-householder 3 --head-size 128 \
        --batch-size 128 --draft-len 4

For instruction-level detail on the single hot kernel, follow up with:

    ncu --set full --kernel-name-base demangled \
        --launch-skip 8 --launch-count 1 \
        python benchmarks/profile_gdp_decode.py <same args>
"""

import argparse
import importlib.util
import pathlib
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import flashinfer.gdn_kernels.gdn_decode_mtp as _mtp

_ORIG_MTP_CONFIG = _mtp.get_mtp_config


def _force_tile_v(tile_v):
    """Override the tile_v that get_mtp_config picks, leaving the rest alone.

    tile_v sets how much of V one CTA covers, so it decides the grid: CTAs ~
    B * HV * (V // tile_v). It is host-side, so this isolates the effect of the
    decomposition without touching the kernel.
    """
    if tile_v is None:
        _mtp.get_mtp_config = _ORIG_MTP_CONFIG
        return

    def forced(*a, **k):
        _, vec, ilp, smem = _ORIG_MTP_CONFIG(*a, **k)
        return tile_v, vec, ilp, smem

    _mtp.get_mtp_config = forced


_BENCH = pathlib.Path(__file__).with_name("bench_gdp_decode.py")
_spec = importlib.util.spec_from_file_location("bench_gdp_decode", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _profile(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_time_total > 0]
    launches = sum(e.count for e in ev) / iters
    total = sum(e.device_time_total for e in ev) / iters
    main = max(ev, key=lambda x: x.device_time_total)
    print(f"\n{name}: {launches:.0f} launches/call, {total:.1f} us/call")
    for e in sorted(ev, key=lambda x: -x.device_time_total):
        print(
            f"  {e.count / iters:5.1f}x {e.device_time_total / iters:9.1f} us"
            f"  {100 * e.device_time_total / iters / total:5.1f}%  {e.key[:60]}"
        )
    print(f"RESULT\t{name}\t{main.device_time_total / iters:.3f}\t{total:.3f}")
    return main.device_time_total / iters


def _sweep(a):
    """One subprocess per (D, B, T, tile_v).

    vllm is re-profiled in every child. Its config does not depend on tile_v, so
    the spread of its time WITHIN one (D, B, T) group is this setup's drift.
    Across groups it varies by design -- comparing there would be meaningless.
    """
    rows, vllm_by_cfg, failures = {}, {}, []
    for D in a.head_size:
        for B in a.batch_size:
            for T in a.draft_len:
                for tv in a.force_tile_v:
                    if D % tv:
                        continue
                    proc = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--num-householder",
                            str(a.num_householder),
                            "--head-size",
                            str(D),
                            "--batch-size",
                            str(B),
                            "--draft-len",
                            str(T),
                            "--force-tile-v",
                            str(tv),
                            "--iters",
                            str(a.iters),
                            "--warmup",
                            str(a.warmup),
                            "--dtype",
                            a.dtype[0],
                            "--num-q-heads",
                            str(a.num_q_heads),
                            "--num-v-heads",
                            str(a.num_v_heads),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    hits = 0
                    for line in proc.stdout.splitlines():
                        if not line.startswith("RESULT\t"):
                            continue
                        _, name, kern, _tot = line.split("\t")
                        if name.startswith("vllm"):
                            vllm_by_cfg.setdefault((D, B, T), []).append(float(kern))
                        elif f"tile_v={tv}" in name and "heuristic" not in name:
                            rows[(D, B, T, tv)] = float(kern)
                            hits += 1
                    if not hits:
                        tail = (proc.stdout + proc.stderr).strip().splitlines()
                        failures.append(
                            (D, B, T, tv, tail[-1][:90] if tail else "no output")
                        )
                    print(f"  ran D={D} B={B} T={T} tile_v={tv}", flush=True)

    print(
        f"\n{'D':>4} {'batch':>6} {'T':>3} | "
        + " ".join(f"tile_v={tv:<3}" for tv in a.force_tile_v)
        + " |    best      vllm  speedup"
    )
    for D, B, T in sorted({k[:3] for k in rows}):
        got = {
            tv: rows[(D, B, T, tv)] for tv in a.force_tile_v if (D, B, T, tv) in rows
        }
        best = min(got, key=got.get)
        cells = " ".join(
            f"{got[tv]:10.1f}" if tv in got else f"{'-':>10}" for tv in a.force_tile_v
        )
        # vllm from THIS config only, never a cross-config average
        v = vllm_by_cfg.get((D, B, T))
        vcol = f"{sorted(v)[len(v) // 2]:9.1f}" if v else f"{'-':>9}"
        spd = f"{sorted(v)[len(v) // 2] / got[best]:7.2f}x" if v else f"{'-':>8}"
        print(f"{D:>4} {B:>6} {T:>3} | {cells} | tile_v={best:<3} {vcol} {spd}")

    worst = 0.0
    for _cfg, times in vllm_by_cfg.items():
        if len(times) > 1:
            worst = max(worst, (max(times) - min(times)) / min(times) * 100)
    print(f"\ndrift control: worst vllm spread within a single config: {worst:.2f}%")
    if worst > 2:
        print("  -> children of the same config disagree; raise --iters")
    for D, B, T, tv, why in failures:
        print(f"  FAILED D={D} B={B} T={T} tile_v={tv}: {why}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-householder", type=int, default=3)
    p.add_argument("--head-size", type=int, nargs="+", choices=[64, 128], default=[128])
    p.add_argument("--draft-len", type=int, nargs="+", default=[4])
    p.add_argument("--batch-size", type=int, nargs="+", default=[128])
    p.add_argument("--dtype", choices=list(bench.DTYPES), default="bfloat16")
    p.add_argument("--num-q-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="grid over --batch-size x --draft-len x --head-size x --force-tile-v, "
        "one SUBPROCESS per point, then a winner-per-config table. Subprocesses "
        "because an in-process sweep drifts ~15%% across a long run; separate "
        "ones measured 0.0%%.",
    )
    p.add_argument(
        "--force-tile-v",
        type=int,
        nargs="+",
        metavar="N",
        help="also profile flashinfer at these tile_v values; the heuristic's "
        "own choice is always profiled first. V must be divisible by N.",
    )
    a = p.parse_args()

    # bench._runners reads these off the namespace
    a.dtype = [a.dtype] if isinstance(a.dtype, str) else a.dtype
    if a.sweep:
        if not a.force_tile_v:
            p.error("--sweep needs --force-tile-v with the values to compare")
        _sweep(a)
        return
    a.head_size, a.batch_size, a.draft_len = (
        a.head_size[0],
        a.batch_size[0],
        a.draft_len[0],
    )
    ns = argparse.Namespace(
        num_q_heads=a.num_q_heads, num_v_heads=a.num_v_heads, seed=a.seed
    )
    _, runners = bench._runners(
        ns,
        a.batch_size,
        a.draft_len,
        a.num_householder,
        a.head_size,
        bench.DTYPES[a.dtype[0]],
    )
    print(
        f"D={a.head_size} n_h={a.num_householder} B={a.batch_size} T={a.draft_len} "
        f"{a.dtype[0]}  (vllm {'available' if bench.HAS_VLLM else 'NOT importable'})"
    )
    for name, (fn, _) in runners.items():
        if name == "flashinfer":
            continue
        _profile(name, fn, a.warmup, a.iters)

    fi = runners["flashinfer"][0]
    seen = {}
    for tile_v in [None] + list(a.force_tile_v or []):
        if tile_v is not None and a.head_size % tile_v:
            print(f"\nskipping tile_v={tile_v}: V={a.head_size} not divisible by it")
            continue
        _force_tile_v(tile_v)
        chosen = _ORIG_MTP_CONFIG(
            a.batch_size,
            a.draft_len * a.num_householder,
            num_v_heads=a.num_v_heads,
            v_dim=a.head_size,
        )[0]
        tv = chosen if tile_v is None else tile_v
        ctas = a.batch_size * a.num_v_heads * (a.head_size // tv)
        label = f"flashinfer tile_v={tv}" + (" (heuristic)" if tile_v is None else "")
        seen.setdefault(tv, []).append(
            _profile(f"{label}  ~{ctas} CTAs", fi, a.warmup, a.iters)
        )
    _force_tile_v(None)

    # The heuristic's own tile_v may also appear in --force-tile-v. Those two
    # runs are the SAME configuration, so any gap between them is this setup's
    # noise floor -- drift across a long sweep, mostly. A tile_v effect smaller
    # than that floor is not measurable here.
    for tv, times in seen.items():
        if len(times) > 1:
            spread = (max(times) - min(times)) / min(times) * 100
            print(
                f"\nnoise floor: tile_v={tv} profiled {len(times)}x "
                f"(same config) spread {spread:.1f}% "
                f"[{', '.join(f'{t:.1f}' for t in times)} us]"
            )
            if spread > 5:
                print(
                    "  -> larger than any plausible tile_v effect. Raise --iters, "
                    "or profile one tile_v per process invocation."
                )


if __name__ == "__main__":
    main()
