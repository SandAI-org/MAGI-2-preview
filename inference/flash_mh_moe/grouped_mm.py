# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Flash-MH-MoE forward on torch grouped GEMM (cuBLAS/CUTLASS via ``torch._grouped_mm``).

Used where the fused Triton kernel cannot be compiled: Triton 3.5's Blackwell lowering
fails on sm_103 (tcgen05 selection / tensor-memory budget). Same math as the Triton
kernel: gather -> gate/up GEMM -> SwiGLU7 -> down GEMM -> prob-weighted combine.

Everything around the three GEMMs is memory-bound glue, so it is kept to one pass each:
flat ``index_select`` gather, a Triton SwiGLU7 (torch.compile with dynamic shapes was
2.6x slower than the same kernel at a static shape), the group chunks (cuBLAS refuses
>= 1024 groups) run gate/up/act/down end to end so only the small down-GEMM output is
copied, and the top-k combine reads the expert rows through the inverse routing
permutation (fp32 accumulate, no atomics, deterministic) instead of a bf16 scatter_add.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .triton import mh_moe_fwd as _triton_fwd
from .triton.mh_moe_fwd import _scatter_back


def _const(c):
    return float(getattr(c, "value", c))


SWIGLU7_ALPHA = _const(_triton_fwd.SWIGLU7_ALPHA)
SWIGLU7_BIAS = _const(_triton_fwd.SWIGLU7_BIAS)
SWIGLU7_LIMIT = _const(_triton_fwd.SWIGLU7_LIMIT)

# ``MAGI2_MOE_SCATTER``: ``inv`` (default: fused prob-weighted combine through the inverse
# routing permutation, fp32 accumulate, deterministic), ``bf16`` (sorted scatter_add in
# bf16) or ``fp32`` (index_add_ into an fp32 buffer).
_SCATTER = os.environ.get("MAGI2_MOE_SCATTER", "inv").strip().lower()

# cuBLAS grouped GEMM refuses more than 1024 groups per call ("Can't process more than 1024
# groups"). EP=2 puts 6 heads x 256 experts = 1536 groups on a rank, so slice by group.
_MAX_GROUPS = int(os.environ.get("MAGI2_GMM_MAX_GROUPS", 768))  # cuBLAS limit is < 1024


def _wslice(W, g0: int, g1: int) -> torch.Tensor:
    """Expert weights for groups [g0, g1): a slice of the bf16 tensor, or a bf16 dequant of
    NVFP4-packed experts (``nvfp4_pack.FP4Experts``, ``MAGI2_EXPERT_QUANT=fp4``)."""
    if isinstance(W, torch.Tensor):
        return W[g0:g1]
    return W.dequant(g0, g1)

_ACT_BLOCK = 4096


@triton.jit
def _swiglu7_kernel(g_ptr, u_ptr, o_ptr, n, ALPHA, LIMIT, BIAS, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    y = g * tl.sigmoid(ALPHA * g) * (u + BIAS)
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


def _swiglu7(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU7, fp32 math, one pass; shape-agnostic (T varies per call)."""
    out = torch.empty_like(gate)
    n = gate.numel()
    if n:
        _swiglu7_kernel[(triton.cdiv(n, _ACT_BLOCK),)](
            gate, up, out, n, SWIGLU7_ALPHA, SWIGLU7_LIMIT, SWIGLU7_BIAS,
            BLOCK=_ACT_BLOCK, num_warps=4,
        )
    return out


@triton.jit
def _combine_kernel(y_ptr, p_ptr, inv_ptr, o_ptr, S, H, D: tl.constexpr, K: tl.constexpr):
    # Output row r = s*H + h gathers its K routed rows from positions h*S*K + s*K + k of the
    # pre-sort flat order; ``inv`` maps those to rows of the expert-sorted ``y``.
    r = tl.program_id(0).to(tl.int64)
    s = r // H
    h = r % H
    base = (h * S + s) * K
    d = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for k in tl.static_range(K):
        src = tl.load(inv_ptr + base + k).to(tl.int64)
        p = tl.load(p_ptr + src)
        acc += tl.load(y_ptr + src * D + d).to(tl.float32) * p
    tl.store(o_ptr + r * D + d, acc.to(o_ptr.dtype.element_ty))


def _combine_inv(out: torch.Tensor, probs: torch.Tensor, sorted_order: torch.Tensor,
                 S: int, H: int, K: int, D: int, dtype: torch.dtype) -> torch.Tensor:
    T = out.size(0)
    inv = torch.empty_like(sorted_order)
    inv[sorted_order] = torch.arange(T, device=out.device, dtype=sorted_order.dtype)
    y = torch.empty(S * H, D, device=out.device, dtype=dtype)
    _combine_kernel[(S * H,)](out, probs.float(), inv, y, S, H, D=D, K=K, num_warps=2)
    return y.view(S, H, D)


def _group_row_ranges(expert_offsets: torch.Tensor, G: int) -> list[tuple[int, int, int, int]]:
    ends = expert_offsets[1:].tolist()  # one sync per call; the offsets are already materialised
    chunks = []
    row0 = 0
    for g0 in range(0, G, _MAX_GROUPS):
        g1 = min(g0 + _MAX_GROUPS, G)
        row1 = ends[g1 - 1]
        chunks.append((g0, g1, row0, row1))
        row0 = row1
    return chunks


def mh_moe_fwd_grouped_mm(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    deterministic: bool = False,
    sorted_order: torch.Tensor | None = None,
) -> torch.Tensor:
    T = gather_ids.size(0)
    if T == 0:
        return torch.zeros_like(x)
    S, H, D = x.shape
    G = W_gate.size(0)
    experts_per_head = G // H

    counts = expert_offsets.diff()
    head_of_expert = torch.arange(G, device=x.device) // experts_per_head
    head_ids = torch.repeat_interleave(head_of_expert, counts, output_size=T)  # (T,)
    flat_idx = gather_ids.long() * H + head_ids
    x_sorted = x.view(S * H, D).index_select(0, flat_idx)  # (T, D) in expert order
    offs = expert_offsets[1:].to(torch.int32)  # cumulative group ends

    if G <= _MAX_GROUPS:
        gate = torch._grouped_mm(x_sorted, _wslice(W_gate, 0, G), offs=offs)
        up = torch._grouped_mm(x_sorted, _wslice(W_up, 0, G), offs=offs)
        del x_sorted
        hidden = _swiglu7(gate, up)
        del gate, up
        out = torch._grouped_mm(hidden, _wslice(W_down, 0, G), offs=offs)
        del hidden
    else:
        # Run each group chunk end to end: the (rows, d_expert) gate/up/hidden tensors stay
        # chunk-local and only the (rows, d_head) down output is copied into place.
        out = torch.empty(T, D, dtype=x.dtype, device=x.device)
        for g0, g1, row0, row1 in _group_row_ranges(expert_offsets, G):
            if row1 <= row0:
                continue
            xs = x_sorted[row0:row1]
            o = (offs[g0:g1] - row0).contiguous()
            gate = torch._grouped_mm(xs, _wslice(W_gate, g0, g1), offs=o)
            up = torch._grouped_mm(xs, _wslice(W_up, g0, g1), offs=o)
            hidden = _swiglu7(gate, up)
            del gate, up
            out[row0:row1] = torch._grouped_mm(hidden, _wslice(W_down, g0, g1), offs=o)
            del hidden
        del x_sorted

    if _SCATTER == "inv" and sorted_order is not None:
        K = T // (S * H)
        return _combine_inv(out, probs, sorted_order, S, H, K, D, x.dtype)
    if _SCATTER == "fp32" and not deterministic:
        out = out.float() * probs[:, None]
        y = torch.zeros(S * H, D, device=x.device, dtype=torch.float32)
        y.index_add_(0, flat_idx, out)
        return y.to(x.dtype).view(S, H, D)
    out = out * probs[:, None].to(out.dtype)
    y = torch.zeros(S * H, D, device=x.device, dtype=out.dtype)
    y.scatter_add_(0, flat_idx.unsqueeze(-1).expand_as(out), out)
    return y.view(S, H, D)
