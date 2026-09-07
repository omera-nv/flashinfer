"""Localise the DV=64 failure: does it break at T=1, or only as the tile fills?

Same one-hot construction as probe_onehot.py (O must equal I exactly).
T=1 exercises only the V load, a trivial 1x1 inverse, and the O store.
If T=1 is already wrong, the bug is in data movement, not the chunk algebra.
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
        v[t, 0, t % dv] = 1.0
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


print("T sweep -- O must equal eye(T, dv) exactly\n")
print(
    f"{'DK':>4} {'DV':>4} {'T':>4} {'O max|err|':>12} {'O diag mean':>12} {'#rows ok':>9}"
)
for dk, dv in [(128, 128), (64, 64), (128, 64)]:
    for T in (1, 2, 4, 8, 16, 32, 64):
        O, S = run(dk, dv, T)
        exp = torch.eye(T, dv, device=DEV)
        m = min(T, dv)
        i = torch.arange(m, device=DEV)
        rows_ok = int(((O - exp).abs().max(dim=1).values < 1e-3).sum())
        print(
            f"{dk:>4} {dv:>4} {T:>4} {(O - exp).abs().max():>12.3e} "
            f"{O[i, i].mean():>12.4f} {rows_ok:>6}/{T}"
        )
    print()

print("\nActual O[0:12, 0:12] at DK=DV=64 (expect identity):")
O, S = run(64, 64, 64)
torch.set_printoptions(linewidth=200, precision=2, sci_mode=False)
print(O[0:12, 0:12].cpu())
print("\nrow norms O[0:16]:", [round(float(x), 3) for x in O[0:16].norm(dim=1).cpu()])
print(
    "col norms O[:,0:16]:", [round(float(x), 3) for x in O[:, 0:16].norm(dim=0).cpu()]
)
print("\nnonzero (row, col, val) in O[0:8]:")
nz = (O[0:8].abs() > 1e-3).nonzero()
for r, c in nz.tolist()[:24]:
    print(f"   O[{r:2d},{c:3d}] = {float(O[r, c]):.3f}   (expected 1.0 at col {r})")
