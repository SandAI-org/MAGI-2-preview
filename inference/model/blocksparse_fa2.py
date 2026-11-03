# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Range-sparse attention on FlashAttention-2 varlen for GPUs without MagiAttention FFA.

Takes the same (q_ranges, k_ranges) pair list the refiner builds for MagiAttention
(all pairs full attention). Each distinct query segment attends to the concatenation
of every k range paired with it, in ONE varlen call, so softmax is exact and no LSE
merging is needed. Keys/values are gathered per chunk of segments to bound memory.

Why: MagiAttention's FA4 (Blackwell) backend represents the mask as a dense
(n_ranges x seqlen_q) table; at 1080p the refiner has ~380k block-window ranges over
~390k tokens, i.e. a 1.5 TB table. The FFA kernel itself is sm_90a only.

Cost at 1080p (per rank, 8 heads): ~51M gathered kv rows per layer call.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 8


class _Plan:
    __slots__ = ("q_idx", "k_idx", "chunks", "n_q")

    def __init__(self, q_idx, k_idx, chunks, n_q):
        self.q_idx = q_idx  # (total_q,) int64: query rows of all used segments, in segment order
        self.k_idx = k_idx  # (total_k,) int64: kv rows, concatenated per segment
        self.chunks = chunks  # list of (q_lo, q_hi, k_lo, k_hi, cu_q, cu_k, max_q, max_k)
        self.n_q = n_q


def _kv_budget() -> int:
    return int(os.environ.get("MAGI2_BSA_KV_BUDGET", 2_000_000))


@torch.no_grad()
def _build_plan(q_ranges: torch.Tensor, k_ranges: torch.Tensor, n_q: int) -> _Plan:
    dev = q_ranges.device
    qs, qe = q_ranges[:, 0].long(), q_ranges[:, 1].long()
    ks, ke = k_ranges[:, 0].long(), k_ranges[:, 1].long()
    keep = (qe > qs) & (ke > ks)
    qs, qe, ks, ke = qs[keep], qe[keep], ks[keep], ke[keep]

    # Split every pair over the non-overlapping query segments induced by all q boundaries.
    bounds = torch.unique(torch.cat([qs, qe]))
    seg_lo = torch.searchsorted(bounds, qs)
    seg_hi = torch.searchsorted(bounds, qe)
    n_per_pair = seg_hi - seg_lo
    pair_rep = torch.repeat_interleave(torch.arange(qs.numel(), device=dev), n_per_pair)
    local = torch.arange(pair_rep.numel(), device=dev) - torch.repeat_interleave(
        torch.cumsum(n_per_pair, 0) - n_per_pair, n_per_pair
    )
    seg = seg_lo[pair_rep] + local
    order = torch.argsort(seg, stable=True)
    seg, e_ks, e_ke = seg[order], ks[pair_rep][order], ke[pair_rep][order]

    # Gather index for kv rows, entry by entry.
    k_len = e_ke - e_ks
    e_rep = torch.repeat_interleave(torch.arange(k_len.numel(), device=dev), k_len)
    k_idx = e_ks[e_rep] + (
        torch.arange(e_rep.numel(), device=dev)
        - torch.repeat_interleave(torch.cumsum(k_len, 0) - k_len, k_len)
    )

    # Per used segment: q length and k length.
    used, seg_inv = torch.unique_consecutive(seg, return_inverse=True)
    seg_klen = torch.zeros(used.numel(), dtype=torch.long, device=dev).index_add_(0, seg_inv, k_len)
    seg_qs, seg_qe = bounds[used], bounds[used + 1]
    seg_qlen = seg_qe - seg_qs
    q_rep = torch.repeat_interleave(torch.arange(used.numel(), device=dev), seg_qlen)
    q_idx = seg_qs[q_rep] + (
        torch.arange(q_rep.numel(), device=dev)
        - torch.repeat_interleave(torch.cumsum(seg_qlen, 0) - seg_qlen, seg_qlen)
    )

    # Chunk segments so each chunk gathers at most `budget` kv rows (one segment may exceed it).
    budget = _kv_budget()
    qlen_c, klen_c = seg_qlen.tolist(), seg_klen.tolist()
    chunks, lo, acc = [], 0, 0
    q_off, k_off = 0, 0
    cq, ck = [0], [0]
    for i, kl in enumerate(klen_c):
        if acc + kl > budget and i > lo:
            chunks.append((lo, i, q_off, k_off, cq, ck))
            q_off += sum(qlen_c[lo:i]); k_off += acc
            lo, acc, cq, ck = i, 0, [0], [0]
        acc += kl
        cq.append(cq[-1] + qlen_c[i]); ck.append(ck[-1] + kl)
    chunks.append((lo, len(klen_c), q_off, k_off, cq, ck))
    out = []
    for lo, hi, q_off, k_off, cq, ck in chunks:
        out.append((
            q_off, q_off + cq[-1], k_off, k_off + ck[-1],
            torch.tensor(cq, dtype=torch.int32, device=dev), torch.tensor(ck, dtype=torch.int32, device=dev),
            max(qlen_c[lo:hi]), max(klen_c[lo:hi]),
        ))
    return _Plan(q_idx, k_idx, out, n_q)


def _plan_for(q_ranges: torch.Tensor, k_ranges: torch.Tensor, n_q: int) -> _Plan:
    key = (q_ranges.data_ptr(), k_ranges.data_ptr(), tuple(q_ranges.shape), tuple(k_ranges.shape), n_q)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(q_ranges, k_ranges, n_q)
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        _PLAN_CACHE[key] = plan
    return plan


@torch.no_grad()
def range_sparse_attn_fa2(
    query: torch.Tensor,  # (S, H, D)
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    softmax_scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flash_attn import flash_attn_varlen_func

    plan = _plan_for(q_ranges, k_ranges, query.shape[0])
    output = torch.zeros_like(query)
    lse = torch.full((query.shape[0], query.shape[1]), -float("inf"), dtype=torch.float32, device=query.device)
    # MAGI2_BSA_KERNEL=fa4: per-chunk MagiAttention FA4 (3x FA2 on B300). Each chunk has only
    # ~150 segments, so FA4's (2*n_ranges+1 x chunk_seqlen) func table stays small.
    use_fa4 = os.environ.get("MAGI2_BSA_KERNEL", "fa2").lower() == "fa4"
    if use_fa4:
        from magi_attention.api import flex_flash_attn_func
    for q_lo, q_hi, k_lo, k_hi, cu_q, cu_k, max_q, max_k in plan.chunks:
        qi = plan.q_idx[q_lo:q_hi]
        ki = plan.k_idx[k_lo:k_hi]
        if use_fa4:
            qr = torch.stack([cu_q[:-1], cu_q[1:]], dim=1).contiguous()
            kr = torch.stack([cu_k[:-1], cu_k[1:]], dim=1).contiguous()
            amap = torch.zeros(qr.shape[0], dtype=torch.int32, device=query.device)
            out, meta = flex_flash_attn_func(query[qi], key[ki], value[ki], qr, kr, amap,
                                             **({"softmax_scale": softmax_scale} if softmax_scale else {}))
            output[qi] = out
            lse[qi] = meta.lse
            continue
        out, cur_lse, _ = flash_attn_varlen_func(
            query[qi], key[ki], value[ki], cu_q, cu_k, max_q, max_k,
            softmax_scale=softmax_scale, return_attn_probs=True,
        )
        output[qi] = out
        if cur_lse.dim() == 2 and cur_lse.shape[0] == query.shape[1]:  # (H, total_q)
            lse[qi] = cur_lse.t()
        else:
            lse[qi] = cur_lse.reshape(query.shape[1], -1).t()
    return output, lse
