"""Read the token index map directly, and bisect NV vs the O path.

q = k = one-hot  ->  Q K^T = I, A_inv = I  (inverse is trivial)
v[t] = (t+1) * e_0  ->  all value mass in v-column 0, magnitude labels the token

Then:  O[t, 0] must be t+1        (any token mixing is readable as a wrong label)
       S[k, 0] must be k+1        (state is K^T @ NV, same labelling)

Part 2 sets q = 0: O is then trivially 0, so a correct state proves NV is fine
and puts the bug downstream in GEMM 6 / the O epilogue.
"""

import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

DEV = torch.device("cuda")
T = 64


def run(dk, dv, zero_q=False, dtype=torch.float16):
    q = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    k = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    v = torch.zeros(T, 1, dv, dtype=dtype, device=DEV)
    for t in range(T):
        if not zero_q:
            q[t, 0, t % dk] = 1.0
        k[t, 0, t % dk] = 1.0
        v[t, 0, 0] = float(t + 1)  # label = token index + 1
    o = torch.full((T, 1, dv), float("nan"), dtype=dtype, device=DEV)
    st = torch.full((1, 1, dv, dk), float("nan"), dtype=torch.float32, device=DEV)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        None,
        torch.ones(T, 1, dtype=torch.float32, device=DEV),
        1.0,
        None,
        True,
        torch.tensor([0, T], dtype=torch.int64, device=DEV),
        True,
        output=o,
        output_state=st,
        use_cp=False,
    )
    torch.cuda.synchronize()
    return o[:, 0, :].float(), st[0, 0].float()


def show(tag, got, expect, n=32):
    g = [int(round(x)) for x in got[:n].tolist()]
    e = [int(round(x)) for x in expect[:n].tolist()]
    print(f"  {tag}")
    print(f"    expect[0:{n}] = {e}")
    print(f"    got   [0:{n}] = {g}")
    bad = [(i, e[i], g[i]) for i in range(n) if e[i] != g[i]]
    print(f"    mismatches (idx, expect, got): {bad[:14]}")


for dk, dv in [(128, 128), (64, 64)]:
    print(f"\n=== DK={dk} DV={dv} : labelled V, q=k=one-hot ===")
    O, S = run(dk, dv)
    show(
        "O[:, 0]  (token t should carry label t+1)",
        O[:, 0],
        torch.arange(1, T + 1, device=DEV),
    )
    show(
        "S[:, 0]  (state row k should carry label k+1)",
        S[:, 0],
        torch.arange(1, min(dv, T) + 1, device=DEV),
        n=min(dv, T),
    )
    print(f"    O mass outside column 0: {O[:, 1:].abs().max():.3e} (expect 0)")

    print(f"\n=== DK={dk} DV={dv} : q = 0 -> isolates NV via the state ===")
    O0, S0 = run(dk, dv, zero_q=True)
    print(f"    O max|.| = {O0.abs().max():.3e} (expect 0)")
    show(
        "S[:, 0] with q=0 (must still be k+1: state does not depend on q)",
        S0[:, 0],
        torch.arange(1, min(dv, T) + 1, device=DEV),
        n=min(dv, T),
    )
