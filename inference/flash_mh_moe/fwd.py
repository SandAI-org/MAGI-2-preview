# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Flash-MH-MoE forward — fused Triton kernel with deterministic scatter."""

from __future__ import annotations

import os

import torch


def _moe_backend() -> str:
    """``MAGI2_MOE_BACKEND``: ``triton`` (fused kernel) or ``grouped_mm`` (torch grouped GEMM).

    Default is ``triton`` on Hopper and ``grouped_mm`` on Blackwell, where Triton 3.5
    cannot lower the fused kernel for sm_103.
    """
    backend = os.environ.get("MAGI2_MOE_BACKEND")
    if backend:
        return backend.strip().lower()
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        return "grouped_mm"
    return "triton"


def _is_deterministic() -> bool:
    return os.environ.get("MAGI2_DETERMINISTIC", "0") == "1"


def flash_mh_moe_fwd(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    sorted_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flash-MH-MoE forward: gather → gate/up GEMM → SwiGLU7 → down GEMM → scatter.

    When ``MAGI2_DETERMINISTIC=1``, uses sequential scatter for bit-exact reproducibility.
    """
    extra = {}
    if _moe_backend() == "grouped_mm":
        from .grouped_mm import mh_moe_fwd_grouped_mm as mh_moe_fwd_func

        extra["sorted_order"] = sorted_order
    else:
        from .triton import mh_moe_fwd_func

    return mh_moe_fwd_func(
        x=x,
        gather_ids=gather_ids,
        probs=probs,
        expert_offsets=expert_offsets,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        deterministic=_is_deterministic(),
        **extra,
    )
