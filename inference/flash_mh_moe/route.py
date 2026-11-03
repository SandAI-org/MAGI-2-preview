# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, TypeAlias

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

TopKScoreFunc: TypeAlias = Literal["softmax", "sigmoid"]


# ═══════════════════════════════════════════════════════════════════════════
#  Fused top-k probabilities and indices computation
# ═══════════════════════════════════════════════════════════════════════════


def _compute_topk_probs_and_indices_torch_impl(
    router_logits: torch.Tensor,
    top_k: int,
    score_func: TopKScoreFunc = "softmax",
    expert_bias: torch.Tensor | None = None,
    num_virtual_group: int = 1,
    topk_virtual_group: int = 0,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
    use_score_matrix_for_topk_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    H, S, E = router_logits.shape
    K = top_k

    # ── Gating scores ──

    match score_func:
        case "softmax":
            router_scores = F.softmax(router_logits, dim=-1)
        case "sigmoid":
            router_scores = F.sigmoid(router_logits)
        case _:
            raise ValueError(f"Unknown score_func: {score_func}")

    # ── Aux-free bias for topk selection (not for actual probs) ──

    topk_scores = router_scores
    if expert_bias is not None:
        bias = expert_bias.view(H, 1, E)  # [H, E] -> [H, 1, E]
        topk_scores = topk_scores + bias

    # ── TopK per head per token (with optional virtual group filtering) ──

    if num_virtual_group > 1:
        assert (
            topk_virtual_group > 0 and topk_virtual_group <= num_virtual_group
        ), f"topk_virtual_group ({topk_virtual_group}) must be in (0, num_virtual_group ({num_virtual_group})]"
        assert (
            E % num_virtual_group == 0
        ), f"num_experts ({E}) must be divisible by num_virtual_group ({num_virtual_group})"
        experts_per_group = E // num_virtual_group

        # Score each group by sum of its top-2 expert scores
        # [H, S, E] -> [H, S, num_virtual_group, experts_per_group]
        # -> topk(2, dim=-1) -> [H, S, num_virtual_group, 2]
        # -> sum(dim=-1) -> [H, S, num_virtual_group]
        group_scores = (
            topk_scores.view(H, S, num_virtual_group, experts_per_group)
            .topk(min(2, experts_per_group), dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top-k virtual groups
        # [H, S, num_virtual_group] -> topk(k, dim=-1) -> [H, S, topk_virtual_group]
        _, group_idx = torch.topk(
            group_scores,
            k=topk_virtual_group,
            dim=-1,
            sorted=False,
        )

        # Build a mask to select experts from the top-k virtual groups
        # group_mask: [H, S, num_virtual_group]
        # score_mask: [H, S, num_virtual_group, 1] -> expand(dim=-1) -> [H, S, E]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(H, S, num_virtual_group, experts_per_group)
            .reshape(H, S, E)
        )

        # Mask out scores not in the top-k virtual groups before selecting top-k experts
        # topk_scores: [H, S, E] -> masked_fill(~score_mask, 0.0) -> [H, S, E]
        # -> topk(k, dim=-1) -> topk_probs: [H, S, K]
        scores_for_choice = topk_scores.masked_fill(~score_mask.bool(), 0.0)
        _, topk_indices = torch.topk(scores_for_choice, k=K, dim=-1, sorted=False)
    else:
        _, topk_indices = torch.topk(topk_scores, K, dim=-1)  # [H, S, K]

    if use_score_matrix_for_topk_probs:
        # NOTE: this version is the original one that computes global_score_matrix first
        # and apply normalization on the global score matrix, then gather the topk_probs from the global score matrix,
        # which is mathematically equivalent to the performance-better version below
        # that directly gather the topk_probs from the router_scores and apply normalization on the gathered topk_probs.
        # however, due to normalization is applied on different sets of values (global_score_matrix vs topk_probs),
        # the final topk_probs after normalization might be numerically different.
        # So we keep this old branch here for exactly reproducing the old results if needed,
        # and the new branch below for better performance.
        global_routing_matrix = torch.zeros(
            H,
            S,
            E,
            device=topk_indices.device,
            dtype=torch.int32,
        ).scatter_(-1, topk_indices, 1)

        global_score_matrix = global_routing_matrix * router_scores

        # ── Normalize ──
        if route_norm:
            global_score_matrix = F.normalize(
                global_score_matrix, p=1, dim=-1, eps=norm_eps
            )

        topk_probs = global_score_matrix.gather(-1, topk_indices)
    else:
        # NOTE: topk_indices are based on topk_scores
        # (biased if expert_bias is provided and might be affected by virtual group masking),
        # and we're supposed to gather the final probs from router_scores (without bias or virtual group masking)
        topk_probs = router_scores.gather(-1, topk_indices)  # [H, S, K]

        # ── Normalize ──

        if route_norm:
            topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=norm_eps)

    return (
        topk_probs,
        topk_indices,
    )


score_func_name2id_map = {
    "softmax": 0,
    "sigmoid": 1,
}


@triton.jit
def _route_sigmoid_topk_kernel(l_ptr, b_ptr, p_ptr, i_ptr, S, eps, E: tl.constexpr, K: tl.constexpr,
                               HAS_BIAS: tl.constexpr, NORM: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)  # row of the (H*S, E) logits
    h = r // S
    e = tl.arange(0, E)
    score = tl.sigmoid(tl.load(l_ptr + r * E + e))
    sel = score
    if HAS_BIAS:
        sel = score + tl.load(b_ptr + h * E + e)
    total = 0.0
    for k in tl.static_range(K):
        m = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == m, e, E), axis=0)  # lowest index among ties
        p = tl.sum(tl.where(e == idx, score, 0.0), axis=0)
        tl.store(i_ptr + r * K + k, idx.to(tl.int64))
        tl.store(p_ptr + r * K + k, p)
        total += p
        sel = tl.where(e == idx, float("-inf"), sel)
    if NORM:
        inv = 1.0 / tl.maximum(total, eps)
        for k in tl.static_range(K):
            p = tl.load(p_ptr + r * K + k)
            tl.store(p_ptr + r * K + k, p * inv)


def _route_kernel_enabled() -> bool:
    v = os.environ.get("MAGI2_ROUTE_KERNEL")
    if v:
        return v.strip().lower() == "triton"
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _compute_topk_sigmoid_triton(router_logits, top_k, expert_bias, route_norm, norm_eps):
    """One pass per (head, token) row: sigmoid, +bias for selection, top-k (descending, ties to
    the lowest index), probs from the unbiased scores, L1-normalised. Same outputs as the torch
    path except for exact ties, which torch.topk breaks arbitrarily."""
    H, S, E = router_logits.shape
    logits = router_logits.contiguous().view(H * S, E)
    probs = torch.empty(H, S, top_k, device=logits.device, dtype=torch.float32)
    indices = torch.empty(H, S, top_k, device=logits.device, dtype=torch.int64)
    bias = expert_bias.contiguous().float() if expert_bias is not None else logits
    _route_sigmoid_topk_kernel[(H * S,)](
        logits, bias, probs, indices, S, float(norm_eps), E=E, K=top_k,
        HAS_BIAS=expert_bias is not None, NORM=route_norm, num_warps=2,
    )
    return probs, indices


def compute_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    score_func: TopKScoreFunc = "softmax",
    expert_bias: torch.Tensor | None = None,
    num_virtual_group: int = 1,
    topk_virtual_group: int = 0,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-k probabilities and indices from router logits.

    Args:
        router_logits: Tensor of shape [H, S, E], the router logits for each head, token, and expert.
        top_k: The number of top probabilities and indices to compute.
        score_func: The function to convert logits to scores, either "softmax" or "sigmoid".
        expert_bias: Optional tensor of shape [H, E] to be added to the router scores before top-k selection,
            for biasing certain experts.
        num_virtual_group: If >1, the experts are divided into this many virtual groups
            for a two-stage top-k selection (first select top-k virtual groups,
            then select top-k experts within those groups).
        topk_virtual_group: If num_virtual_group > 1,
            the number of top virtual groups to select before selecting top-k experts.
        route_norm: Whether to normalize the top-k probabilities to sum to 1. Defaults to ``True``.
        norm_eps: the epsilon used for route_norm. Defaults to ``1e-12``.


    Returns:
        A tuple of (topk_probs, topk_indices):
            - topk_probs: Tensor of shape [H, S, top_k], the top-k probabilities for each head and token.
            - topk_indices: Tensor of shape [H, S, top_k],
                the indices of the top-k experts for each head and token.
    """
    if (
        score_func == "sigmoid"
        and num_virtual_group <= 1
        and router_logits.is_cuda
        and router_logits.dtype == torch.float32
        and (router_logits.size(-1) & (router_logits.size(-1) - 1)) == 0
        and _route_kernel_enabled()
    ):
        return _compute_topk_sigmoid_triton(router_logits, top_k, expert_bias, route_norm, norm_eps)
    return _compute_topk_probs_and_indices_torch_impl(
        router_logits=router_logits,
        top_k=top_k,
        score_func=score_func,
        expert_bias=expert_bias,
        num_virtual_group=num_virtual_group,
        topk_virtual_group=topk_virtual_group,
        route_norm=route_norm,
        norm_eps=norm_eps,
    )



# ═══════════════════════════════════════════════════════════════════════════
#  Mapping topk probs/indices to sorted gather ids and probs and offsets
#  prepared as meta inputs for flash_mh_moe kernels
# ═══════════════════════════════════════════════════════════════════════════


def flash_mh_moe_global_sort(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    return_order: bool = False,
):
    """
    Map per-head expert ids to global [0, H*E) and argsort into CSR format.
    With ``return_order`` the stable argsort permutation (T,) is returned as a 4th value.

    Args:
        topk_probs (torch.Tensor): Routing probabilities with shape ``(H, S, K)``.
        topk_indices (torch.Tensor): Per-head expert indices in ``[0, E)`` with shape ``(H, S, K)``.
        num_experts (int): E, number of experts per head.

    Returns:
        gather_ids (torch.Tensor): Token indices to gather from ``[S]`` with shape ``(T,)`` (int32).
        probs_sorted (torch.Tensor): Routing probabilities sorted by expert with shape ``(T,)``.
        expert_offsets (torch.Tensor): CSR offsets with shape ``(H*E+1,)`` (long).
    """
    H, S, K = topk_indices.shape
    E = num_experts
    device = topk_indices.device

    head_offset = torch.arange(H, device=device).view(H, 1, 1) * E
    global_indices = topk_indices + head_offset

    flat_indices = global_indices.reshape(-1)
    flat_probs = topk_probs.reshape(-1)
    flat_token_ids = (
        torch.arange(S, device=device).view(1, S, 1).expand(H, S, K).reshape(-1)
    )

    sorted_order = flat_indices.argsort(stable=True)
    gather_ids = flat_token_ids[sorted_order].to(torch.int32)
    probs_sorted = flat_probs[sorted_order].float()

    expert_counts = torch.zeros(H * E, device=device, dtype=torch.long)
    expert_counts.scatter_add_(
        0,
        flat_indices[sorted_order],
        torch.ones_like(sorted_order, dtype=torch.long),
    )
    expert_offsets = torch.zeros(H * E + 1, device=device, dtype=torch.long)
    expert_offsets[1:] = expert_counts.cumsum(0)

    if return_order:
        return gather_ids, probs_sorted, expert_offsets, sorted_order
    return gather_ids, probs_sorted, expert_offsets
