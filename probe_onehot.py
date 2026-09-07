"""One-hot structural probe for the SM100 head_size=64 / rectangular failure.

With K = Q = I, V = I, beta = 1, alpha = 1, scale = 1 the chunked algebra
collapses to IKK = I -> T = I -> NV = V = I, so:

    O      must equal I  (exactly, [T, H, DV])
    state  must equal I  ([H, V, K]: I in the left DV x DV block)

Any deviation is a direct readout of how indices are being permuted.
"""

import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

DEV = torch.device("cuda")
T = 64


def probe(dk, dv, dtype=torch.float16):
    q = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    k = torch.zeros(T, 1, dk, dtype=dtype, device=DEV)
    v = torch.zeros(T, 1, dv, dtype=dtype, device=DEV)
    for t in range(T):
        q[t, 0, t % dk] = 1.0
        k[t, 0, t % dk] = 1.0
        v[t, 0, t % dv] = 1.0
    beta = torch.ones(T, 1, dtype=torch.float32, device=DEV)
    cu = torch.tensor([0, T], dtype=torch.int64, device=DEV)

    o = torch.full((T, 1, dv), float("nan"), dtype=dtype, device=DEV)
    st = torch.full((1, 1, dv, dk), float("nan"), dtype=torch.float32, device=DEV)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        None,
        beta,
        1.0,
        None,
        True,
        cu,
        True,
        output=o,
        output_state=st,
        use_cp=False,
    )
    torch.cuda.synchronize()

    O = o[:, 0, :].float()  # [T, DV], expect I
    S = st[0, 0].float()  # [DV, DK], expect I in left block
    exp_O = torch.eye(T, dv, device=DEV)
    # Only T tokens arrive, so the state has ones on the diagonal for
    # i < min(T, dk, dv) and zeros beyond -- eye(dv, dk) would wrongly expect
    # ones past the last token when dk and dv both exceed T.
    exp_S = torch.zeros(dv, dk, device=DEV)
    n_diag = min(T, dk, dv)
    idx = torch.arange(n_diag, device=DEV)
    exp_S[idx, idx] = 1.0

    def report(name, A, exp):
        # A may be non-square, so zero the diagonal in place rather than
        # subtracting a torch.diag() that would not broadcast.
        m = min(A.shape[0], A.shape[1])
        idx = torch.arange(m, device=A.device)
        diag = A[idx, idx]
        off = A.clone()
        off[idx, idx] = 0.0
        print(
            f"  {name} shape={tuple(A.shape)} max|err|={(A - exp).abs().max():.3e}  "
            f"diag mean={diag.mean():.4f}  offdiag max={off.abs().max():.4f}"
        )
        perm = A.argmax(dim=1)
        wrong = [(i, int(perm[i])) for i in range(m) if int(perm[i]) != i]
        print(
            f"  {name} row -> argmax mismatches: {len(wrong)}/{m}"
            f"   first 12: {wrong[:12]}"
        )

    print(f"\n=== DK={dk} DV={dv} ===")
    report("O", O, exp_O)
    report("S", S, exp_S)


probe(128, 128)  # control: known good
probe(64, 64)  # broken square
probe(128, 64)  # rectangular
