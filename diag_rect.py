"""Diagnose the rectangular-state (DK=128, DV=64) failure on SM100.

Run from the repo root:  python diag_rect.py
"""

import random
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from tests.gdn.conftest import gen_qkv
from tests.gdn.reference_delta_rule import blockwise_delta_rule, exclusive_cumsum

DEV = torch.device("cuda")


def build(dk, dv, seq_lens=(64,), h=1, dtype=torch.float16):
    random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    with DEV:
        q, k, v = gen_qkv(list(seq_lens), h, h, h, dk, dtype, head_size_v=dv)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu = torch.tensor(exclusive_cumsum(list(seq_lens)), dtype=torch.int64)
        beta = torch.rand(sum(seq_lens), h)
    return q, k, v, cu, beta


def run(dk, dv, seq_lens=(64,), h=1, store_state=True):
    q, k, v, cu, beta = build(dk, dv, seq_lens, h)
    T = sum(seq_lens)
    o = torch.full((T, h, dv), float("nan"), dtype=q.dtype, device=DEV)
    st = None
    if store_state:
        st = torch.full(
            (len(seq_lens), h, dv, dk), float("nan"), dtype=torch.float32, device=DEV
        )
    chunk_gated_delta_rule(
        q,
        k,
        v,
        None,
        beta,
        dk**-0.5,
        None,
        store_state,
        cu,
        True,
        output=o,
        output_state=st,
        use_cp=False,
    )
    torch.cuda.synchronize()
    ref_o, ref_st = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        list(seq_lens),
        scale_factor=dk**-0.5,
        beta=beta,
        state_dtype=torch.float32,
    )
    return o, st, ref_o.to(q.dtype), ref_st


def summarize(tag, o, ref_o, st, ref_st):
    e = (o.float() - ref_o.float()).abs()
    print(f"\n=== {tag} ===")
    print(
        f"  output  max|err|={e.max():.3e}   bad={(e > 2e-3).sum().item()}/{e.numel()}"
    )
    if e.max() > 2e-3:
        bad_tok = (e > 2e-3).any(-1).any(-1).nonzero().flatten().tolist()
        bad_v = (e > 2e-3).any(0).any(0).nonzero().flatten().tolist()
        print(
            f"  bad token rows : {bad_tok[:8]}{'...' if len(bad_tok) > 8 else ''} "
            f"(n={len(bad_tok)}, of {o.shape[0]})"
        )
        print(
            f"  bad v columns  : {bad_v[:8]}{'...' if len(bad_v) > 8 else ''} "
            f"(n={len(bad_v)}, of {o.shape[-1]})"
        )
    if st is not None:
        se = (st.transpose(-1, -2).float() - ref_st.float()).abs()
        print(
            f"  state   max|err|={se.max():.3e}  bad={(se > 1e-3).sum().item()}/{se.numel()}"
        )
        bad_k = (se > 1e-3).any(0).any(0).any(-1).nonzero().flatten().tolist()
        print(
            f"  bad state K rows: n={len(bad_k)} of {se.shape[-2]}"
            f"  first={bad_k[:6]} last={bad_k[-6:] if bad_k else []}"
        )
    print(f"  output has NaN: {bool(o.isnan().any())}")


# 1. controls: square configs must still be clean (my diff is a no-op there)
for d in (64, 128):
    o, st, ro, rst = run(d, d)
    summarize(f"CONTROL square DK=DV={d}", o, ro, st, rst)

# 2. the failing rectangular case, WITH final-state store
o, st, ro, rst = run(128, 64)
summarize("RECT DK=128 DV=64  (output_final_state=True)", o, ro, st, rst)

# 3. same, WITHOUT final-state store -> isolates the state path
o2, _, ro2, _ = run(128, 64, store_state=False)
summarize("RECT DK=128 DV=64  (output_final_state=False)", o2, ro2, None, None)

same = torch.equal(o, o2)
print(f"\n>>> output identical with/without state store: {same}")
print(">>> if False, the state path is corrupting the output accumulator")
