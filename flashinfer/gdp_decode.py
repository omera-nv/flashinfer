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

Gated DeltaProduct (arXiv:2502.10297) -- decode API layer.
"""

from typing import Optional, Tuple

import torch

from .gdn_decode import gated_delta_rule_mtp


GATE_NEUTRAL_A_SENTINEL = -1.0e4


def gated_delta_product_mtp(
    q: torch.Tensor,  # [B, T,      num_q_heads, K]
    k: torch.Tensor,  # [B, T, n_h, num_k_heads, K]
    v: torch.Tensor,  # [B, T, n_h, num_v_heads, V]
    initial_state: torch.Tensor,  # [pool_size, HV, V, K] fp32 -- the state POOL
    initial_state_indices: torch.Tensor,  # [B] read slot per batch row
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T,      HV]  decay logits, ONE per real token
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, T, n_h, HV]  update-gate logits, per householder
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,  # [B, T, HV, V]
    ssm_state_indices: Optional[torch.Tensor] = None,  # [B, T] per-token scatter
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,  # [B]
    # expansion scratch -- see chunk_gated_delta_product
    expanded_q: Optional[torch.Tensor] = None,  # [B, T*n_h, num_q_heads, K]
    expanded_a: Optional[torch.Tensor] = None,  # [B, T*n_h, HV]
    expanded_output: Optional[torch.Tensor] = None,  # [B, T*n_h, HV, V]
    expanded_ssm_state_indices: Optional[torch.Tensor] = None,  # [B, T*n_h] int32
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated DeltaProduct decode / MTP.

    GDP decode is :func:`flashinfer.gdn_decode.gated_delta_rule_mtp` with
    ``T -> T * num_householder``: one real token becomes ``n_h`` micro-steps.
    With speculative decoding on top, ``T`` is already ``num_spec + 1``, so the
    expanded axis is ``n_h * (num_spec + 1)``.

    **The gate is fused**, unlike the prefill kernel. Prefill takes ``g``
    directly; here the kernel derives alpha from ``A_log``/``a``/``dt_bias``.
    Neutralising the gate on micro-steps ``1..n_h-1`` therefore happens through
    ``a``, using :data:`GATE_NEUTRAL_A_SENTINEL` -- not by writing 1.0 anywhere.

    Parameters
    ----------
    k, v, b : torch.Tensor
        Carry a householder axis at dim 2. ``q`` and ``a`` do not: one query and
        one gate per REAL token.
    ssm_state_indices : torch.Tensor, optional
        ``[B, T]`` int32, one pool slot per REAL token, as for GDN MTP. The
        wrapper expands this to ``[B, T*n_h]``, giving micro-steps ``1..n_h-1``
        a negative slot -- which the kernel's scatter skips -- and routing only
        the last micro-step of each token to the caller's slot.
    expanded_* : torch.Tensor, optional
        Scratch for the expansion; required for CUDA graph capture. See
        :func:`chunk_gated_delta_product` for why.

    Returns
    -------
    ``(output, initial_state)``, matching ``gated_delta_rule_mtp``. ``output``
    is ``[B, T, HV, V]`` -- one row per REAL token.
    """
    if k.dim() != 5 or v.dim() != 5:
        raise ValueError(
            f"k/v must carry a householder axis [B, T, n_h, H, D]; "
            f"got k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )
    num_householder = k.size(2)
    if v.size(2) != num_householder:
        raise ValueError(
            f"k/v householder counts differ: {num_householder} vs {v.size(2)}"
        )
    if b.dim() != 4 or b.size(2) != num_householder:
        raise ValueError(
            f"b must be [B, T, n_h, HV] with n_h={num_householder}; "
            f"got {tuple(b.shape)}"
        )
    if a.dim() != 3:
        raise ValueError(
            f"a is one decay logit per REAL token, expected [B, T, HV]; "
            f"got {tuple(a.shape)}"
        )

    # n_h == 1 is plain GDN MTP. Delegate so this path stays bit-identical.
    if num_householder == 1:
        return gated_delta_rule_mtp(
            q,
            k.squeeze(2),
            v.squeeze(2),
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b.squeeze(2),
            scale=scale,
            output=output,
            ssm_state_indices=ssm_state_indices,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
            output_state_indices=output_state_indices,
        )

    # GDP = GDN with a sequence n_h times longer
    k = torch.flatten(k, start_dim=1, end_dim=2)
    v = torch.flatten(v, start_dim=1, end_dim=2)
    b = torch.flatten(b, start_dim=1, end_dim=2)

    if expanded_q is None:
        expanded_q = torch.empty(
            q.size(0),
            q.size(1) * num_householder,
            *q.shape[2:],
            dtype=q.dtype,
            device=q.device,
        )
    elif expanded_q.shape != (q.size(0), q.size(1) * num_householder, *q.shape[2:]):
        raise ValueError("expanded_q shape must be [B, T*n_h, num_q_heads,  D]")
    elif expanded_q.dtype != q.dtype:
        raise ValueError(
            f"expanded_q.dtype and q.dtype must match, got {expanded_q.dtype} != {q.dtype}"
        )

    # Micro-steps 0..n_h-2 of every token must read a ZERO query -- only the
    # last one produces a kept output row. Cleared unconditionally because a
    # caller-supplied buffer may be dirty (CUDA-graph scratch is reused);
    # empty + zero_ is a single pass, same cost as torch.zeros.
    expanded_q.zero_()
    expanded_q[:, num_householder - 1 :: num_householder] = q

    if expanded_a is None:
        expanded_a = torch.empty(
            (a.size(0), a.size(1) * num_householder, *a.shape[2:]),
            dtype=a.dtype,
            device=a.device,
        )
    elif expanded_a.shape != (a.size(0), a.size(1) * num_householder, *a.shape[2:]):
        raise ValueError("expanded_a shape must be [B, T*n_h, num_sab_heads]")
    elif expanded_a.dtype != a.dtype:
        raise ValueError(f"expanded_a dtype must match a dtype ({a.dtype})")
    # Micro-steps 1..n_h-1 must carry the neutral sentinel so alpha == 1 there.
    # This one is load-bearing, not hygiene: a dirty buffer gives those steps
    # random gates, which corrupts the STATE rather than merely the discarded
    # output rows, and goes NaN at larger n_h.
    expanded_a.fill_(GATE_NEUTRAL_A_SENTINEL)
    expanded_a[:, ::num_householder] = a

    _o_shape = (
        expanded_q.size(0),
        expanded_q.size(1),
        max(q.size(2), v.size(2)),
        v.size(3),
    )
    if expanded_output is None:
        expanded_output = torch.empty(
            *_o_shape,
            # kernel hardcodes bf16...
            dtype=torch.bfloat16,
            device=output.device if output is not None else q.device,
        )
    elif expanded_output.shape != _o_shape:
        raise ValueError(
            f"expanded_output shape must be {_o_shape} [B, T*n_h, num_o_heads, V]"
        )

    if ssm_state_indices is not None:
        if expanded_ssm_state_indices is None:
            expanded_ssm_state_indices = torch.empty(
                ssm_state_indices.size(0),
                ssm_state_indices.size(1) * num_householder,
                dtype=ssm_state_indices.dtype,
                device=ssm_state_indices.device,
            )
        elif expanded_ssm_state_indices.shape != (
            ssm_state_indices.size(0),
            ssm_state_indices.size(1) * num_householder,
        ):
            raise ValueError("expanded_ssm_state_indices shape must be [B, T*n_h]")
        elif expanded_ssm_state_indices.dtype != ssm_state_indices.dtype:
            raise ValueError(
                f"expanded_ssm_state_indices dtype must be {ssm_state_indices.dtype}"
            )

        # negative slot indices signal to skip writing
        expanded_ssm_state_indices[:] = -1
        expanded_ssm_state_indices[:, num_householder - 1 :: num_householder] = (
            ssm_state_indices
        )
    else:
        expanded_ssm_state_indices = None

    if initial_state_indices.data_ptr() % 16 != 0:
        # GDN kernel requires 16-byte alignment on the index tensor
        initial_state_indices = initial_state_indices.clone()

    _, state = gated_delta_rule_mtp(
        expanded_q,
        k,
        v,
        initial_state,
        initial_state_indices,
        A_log,
        expanded_a,
        dt_bias,
        b,
        scale=scale,
        output=expanded_output,
        ssm_state_indices=expanded_ssm_state_indices,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
        output_state_indices=output_state_indices,
    )

    if output is not None:
        output[:] = expanded_output[:, num_householder - 1 :: num_householder]
    else:
        output = expanded_output[:, num_householder - 1 :: num_householder].clone()

    return output.to(q.dtype), state
