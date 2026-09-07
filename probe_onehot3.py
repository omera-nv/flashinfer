"""Is the DV=64 failure deterministic (addressing) or not (race)?

Runs the identical one-hot case repeatedly. A pure addressing/layout bug is
bit-reproducible; a pipeline/barrier bug is not.
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
    return o[:, 0, :].float().clone(), st[0, 0].float().clone()


print("=== determinism: 5 identical runs, DK=DV=64, T=64 ===")
base_o, base_s = run(64, 64, 64)
for i in range(4):
    o, s = run(64, 64, 64)
    print(
        f"  run {i + 1}: O identical={torch.equal(o, base_o)}  "
        f"S identical={torch.equal(s, base_s)}  "
        f"O diff elems={(o != base_o).sum().item()}"
    )

print("\n=== onset: exact T at which DV=64 breaks ===")
print(f"{'T':>4} {'O max|err|':>12} {'#rows ok':>10}  first bad rows")
for T in range(1, 17):
    O, _ = run(64, 64, T)
    exp = torch.eye(T, 64, device=DEV)
    bad = ((O - exp).abs().max(dim=1).values >= 1e-3).nonzero().flatten().tolist()
    print(f"{T:>4} {(O - exp).abs().max():>12.3e} {T - len(bad):>7}/{T}  {bad[:8]}")

print("\n=== does head count matter? (T=64, DK=DV=64) ===")
for h in (1, 2, 4, 8):
    q = torch.zeros(64, h, 64, dtype=torch.float16, device=DEV)
    k = torch.zeros(64, h, 64, dtype=torch.float16, device=DEV)
    v = torch.zeros(64, h, 64, dtype=torch.float16, device=DEV)
    for t in range(64):
        q[t, :, t] = 1.0
        k[t, :, t] = 1.0
        v[t, :, t] = 1.0
    o = torch.full((64, h, 64), float("nan"), dtype=torch.float16, device=DEV)
    st = torch.full((1, h, 64, 64), float("nan"), dtype=torch.float32, device=DEV)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        None,
        torch.ones(64, h, dtype=torch.float32, device=DEV),
        1.0,
        None,
        True,
        torch.tensor([0, 64], dtype=torch.int64, device=DEV),
        True,
        output=o,
        output_state=st,
        use_cp=False,
    )
    torch.cuda.synchronize()
    exp = torch.eye(64, device=DEV)
    errs = [float((o[:, i, :].float() - exp).abs().max()) for i in range(h)]
    print(f"  heads={h}: per-head max|err| = {[round(e, 3) for e in errs]}")
