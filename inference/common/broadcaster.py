# Copyright (c) 2026 SandAI. All Rights Reserved.
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

from typing import Optional

import torch
import torch.distributed as dist

_DTYPE_TABLE = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)


def _dtype_to_index(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TABLE.index(dtype)
    except ValueError as exc:
        raise ValueError(f"Unsupported tensor dtype for broadcast: {dtype}") from exc


class DistBroadcaster:
    src_rank: int
    group: dist.ProcessGroup
    dtype: Optional[torch.dtype] = None

    def __init__(self, src_rank: int, group: dist.ProcessGroup, dtype: Optional[torch.dtype] = None):
        assert torch.distributed.is_available() and torch.distributed.is_initialized()
        assert group is not None
        self.src_rank = src_rank
        self.group = group
        self.dtype = dtype

    @property
    def is_src_rank(self) -> bool:
        return self.src_rank == dist.get_rank()

    def broadcast(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if dist.get_world_size(self.group) == 1:
            assert tensor is not None, "tensor must not be None when broadcasting to a single rank"
            return tensor
        device = torch.device("cuda", torch.cuda.current_device())
        if self.is_src_rank:
            assert tensor is not None, "tensor must not be None when broadcasting from a source rank"
            if self.dtype is not None:
                assert tensor.dtype == self.dtype, f"dtype mismatch: {tensor.dtype} != {self.dtype}"
            tensor = tensor.to(device=device)

        if self.dtype is None:
            if self.is_src_rank:
                dtype_index = torch.tensor([_dtype_to_index(tensor.dtype)], dtype=torch.int64, device=device)
            else:
                dtype_index = torch.empty(1, dtype=torch.int64, device=device)
            dist.broadcast(dtype_index, src=self.src_rank, group=self.group)
            self.dtype = _DTYPE_TABLE[int(dtype_index.item())]

        if self.is_src_rank:
            ndim = torch.tensor([tensor.ndim], dtype=torch.int64, device=device)
        else:
            ndim = torch.empty(1, dtype=torch.int64, device=device)
        dist.broadcast(ndim, src=self.src_rank, group=self.group)

        if self.is_src_rank:
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.int64, device=device)
        else:
            shape_tensor = torch.empty(int(ndim.item()), dtype=torch.int64, device=device)
        dist.broadcast(shape_tensor, src=self.src_rank, group=self.group)

        if not self.is_src_rank:
            shape = tuple(int(dim) for dim in shape_tensor.tolist())
            tensor = torch.empty(shape, dtype=self.dtype, device=device)
        dist.broadcast(tensor, src=self.src_rank, group=self.group)
        return tensor
