# Copyright (c) 2025 SandAI. All Rights Reserved.
# Apache-2.0.
"""Minimal logging helpers for MAGI-2 inference."""

import os

import torch
import torch.distributed as dist


def print_rank_0(message, *args, **kwargs):
    """Print only on rank 0 (or when not distributed)."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return
    elif int(os.getenv("RANK", "0")) != 0:
        return
    print(message, *args, **kwargs)


def print_mem_info_rank_0(prefix: str = ""):
    """Print GPU memory stats on rank 0."""
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print_rank_0(f"{prefix} GPU mem: {alloc:.1f}GB alloc, {reserved:.1f}GB reserved")
