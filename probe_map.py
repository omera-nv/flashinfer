"""Full corruption map: where does each token's value actually land?

v[t] = (t+1) * e_0  -- one labelled scalar per token, all in v-column 0.
Expected O: nonzero only at (token t, v-col 0) with value t+1.

Every nonzero is printed as  O[token, vcol] = label  ->  "token `label-1` landed here".
That is the complete (source -> destination) map of the corruption.

State slice fixed: kernel state is [DV, DK], labels live along S[0, :].
"""

import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

DEV = torch.device("cuda")


def run(dk, dv, T, dtype=torch.float16):
    q = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    k = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    v = torch.zeros(T, 1, dv, dtype=dtype, device=DEV)
    for t in range(T):
        q[t, 0, t % dk] = 1.0
        k[t, 0, t % dk] = 1.0
        v[t, 0, 0] = float(t + 1)
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


for dk, dv, T in [(128, 128, 64), (64, 64, 64), (64, 64, 16)]:
    O, S = run(dk, dv, T)
    print(f"\n{'=' * 70}\n=== DK={dk} DV={dv} T={T} ===")

    nz = (O.abs() > 1e-3).nonzero().tolist()
    print(f"O nonzeros: {len(nz)} (expect {T}, all at vcol 0)")
    wrong = [
        (t, c, int(round(float(O[t, c]))))
        for t, c in nz
        if c != 0 or int(round(float(O[t, c]))) != t + 1
    ]
    print(f"  misplaced: {len(wrong)}")
    for t, c, lab in wrong[:40]:
        print(
            f"    O[tok={t:2d}, vcol={c:3d}] = {lab:3d}"
            f"   <- value of token {lab - 1}"
            f"{'   (right token, wrong vcol)' if lab - 1 == t else ''}"
        )
    missing = [t for t in range(T) if abs(float(O[t, 0]) - (t + 1)) > 1e-3]
    print(f"  tokens with wrong/absent value at vcol 0: {missing}")

    # state: [DV, DK]; labels live along S[0, :]
    exp_s = torch.arange(1, min(dk, T) + 1, device=DEV, dtype=torch.float32)
    got_s = S[0, : min(dk, T)]
    bad_s = [
        (i, int(exp_s[i]), int(round(float(got_s[i]))))
        for i in range(len(exp_s))
        if abs(got_s[i] - exp_s[i]) > 1e-3
    ]
    print(f"S[0, :] mismatches: {len(bad_s)}  first 16: {bad_s[:16]}")
    print(f"S mass outside row 0: {S[1:].abs().max():.3e} (expect 0)")
