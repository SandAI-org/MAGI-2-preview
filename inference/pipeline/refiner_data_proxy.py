# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

import torch
from einops import rearrange
from torch.nn import functional as F
from unfoldNd import UnfoldNd

from inference.common.magi2_config import Magi2RefinerDataProxyConfig
from inference.model.magi2_refiner import calc_local_attn_ffa_handler


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


@dataclass
class VarlenHandler(object):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class WindowLocalAttnHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: Optional[float] = None
    bwd_q_ranges: Optional[torch.Tensor] = None
    bwd_k_ranges: Optional[torch.Tensor] = None
    bwd_attn_type_map: Optional[torch.Tensor] = None
    auto_range_merge: bool = False
    sparse_load: bool = False


@dataclass(frozen=True)
class BlockLocalAttnRanges:
    fwd_q_ranges: torch.Tensor
    fwd_k_ranges: torch.Tensor
    bwd_q_ranges: torch.Tensor
    bwd_k_ranges: torch.Tensor
    max_q_len: int


def _max_range_len(ranges: torch.Tensor) -> int:
    if ranges.numel() == 0:
        return 0
    lengths = ranges[:, 1].to(torch.int64) - ranges[:, 0].to(torch.int64)
    return int(lengths.max().item())


def _cat_ranges(*ranges: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    valid_ranges = [r for r in ranges if r is not None and r.numel() > 0]
    if valid_ranges:
        return torch.cat(valid_ranges, dim=0).to(dtype=torch.int32).contiguous()
    return torch.empty((0, 2), device=device, dtype=torch.int32)


class BlockLocalAttn:
    def __init__(self, block_ranges_tensor: torch.Tensor, win_size: int) -> None:
        self.block_ranges_tensor = block_ranges_tensor.to(
            dtype=torch.int32
        ).contiguous()
        self.win_size = int(win_size)

    @property
    def num_blocks(self) -> int:
        return int(self.block_ranges_tensor.shape[0])

    @property
    def device(self) -> torch.device:
        return self.block_ranges_tensor.device

    def build_ranges(self) -> BlockLocalAttnRanges:
        if self.num_blocks == 0:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        window_width = min(self.num_blocks, max(1, self.win_size))
        q_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int32)
        left_width = window_width // 2
        window_start = (q_block - left_width).clamp(
            min=0, max=self.num_blocks - window_width
        )
        window_end = window_start + window_width
        q_ranges = self.block_ranges_tensor[q_block.long()].contiguous()
        k_ranges = torch.stack(
            [
                self.block_ranges_tensor[window_start.long(), 0],
                self.block_ranges_tensor[(window_end - 1).long(), 1],
            ],
            dim=1,
        ).contiguous()

        k_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int32)
        first_q_block = torch.searchsorted(
            window_end, k_block, right=True, out_int32=True
        )
        last_q_block = (
            torch.searchsorted(window_start, k_block, right=True, out_int32=True) - 1
        )
        bwd_q_ranges = torch.stack(
            [
                self.block_ranges_tensor[first_q_block.long(), 0],
                self.block_ranges_tensor[last_q_block.long(), 1],
            ],
            dim=1,
        ).contiguous()
        bwd_k_ranges = self.block_ranges_tensor[k_block.long()].contiguous()
        return BlockLocalAttnRanges(
            q_ranges, k_ranges, bwd_q_ranges, bwd_k_ranges, _max_range_len(q_ranges)
        )


class BlockGridLocalAttn:
    def __init__(
        self,
        block_ranges_tensor: torch.Tensor,
        block_grid_shape: tuple[int, int, int],
        radius: tuple[int, int, int],
    ) -> None:
        self.block_ranges_tensor = block_ranges_tensor.to(
            dtype=torch.int32
        ).contiguous()
        self.block_grid_shape = tuple(int(dim) for dim in block_grid_shape)
        self.radius = tuple(int(r) for r in radius)
        expected_num_blocks = (
            self.block_grid_shape[0]
            * self.block_grid_shape[1]
            * self.block_grid_shape[2]
        )
        if expected_num_blocks != int(self.block_ranges_tensor.shape[0]):
            raise ValueError(
                "block_grid_shape product must match block count, got "
                f"{expected_num_blocks=} num_blocks={self.block_ranges_tensor.shape[0]}"
            )

    @property
    def num_blocks(self) -> int:
        return int(self.block_ranges_tensor.shape[0])

    @property
    def device(self) -> torch.device:
        return self.block_ranges_tensor.device

    def build_ranges(self) -> BlockLocalAttnRanges:
        if self.num_blocks == 0:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        nt, nh, nw = self.block_grid_shape
        rt, rh, rw = self.radius
        q_block = torch.arange(self.num_blocks, device=self.device, dtype=torch.int64)
        q_t = q_block // (nh * nw)
        q_h = (q_block // nw) % nh
        q_w = q_block % nw
        q_ranges_by_block = self.block_ranges_tensor[q_block]
        q_ranges_parts: list[torch.Tensor] = []
        k_ranges_parts: list[torch.Tensor] = []
        for dt in range(-rt, rt + 1):
            kt = q_t + dt
            valid_t = (kt >= 0) & (kt < nt)
            if not bool(valid_t.any()):
                continue
            for dh in range(-rh, rh + 1):
                kh = q_h + dh
                valid_th = valid_t & (kh >= 0) & (kh < nh)
                if not bool(valid_th.any()):
                    continue
                for dw in range(-rw, rw + 1):
                    kw = q_w + dw
                    valid = valid_th & (kw >= 0) & (kw < nw)
                    if not bool(valid.any()):
                        continue
                    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                    k_block = kt * (nh * nw) + kh * nw + kw
                    q_ranges_parts.append(q_ranges_by_block[valid_idx])
                    k_ranges_parts.append(self.block_ranges_tensor[k_block[valid_idx]])
        if not q_ranges_parts:
            empty = torch.empty((0, 2), device=self.device, dtype=torch.int32)
            return BlockLocalAttnRanges(empty, empty, empty, empty, 0)
        q_ranges = torch.cat(q_ranges_parts, dim=0).to(dtype=torch.int32).contiguous()
        k_ranges = torch.cat(k_ranges_parts, dim=0).to(dtype=torch.int32).contiguous()
        return BlockLocalAttnRanges(
            q_ranges, k_ranges, q_ranges, k_ranges, _max_range_len(q_ranges)
        )


def get_coords(
    shape: list[int],
    ref_feat_shape: list[int],
    offset_thw: list[int] = [0, 0, 0],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Generate feature grid coordinates with original and reference size info
    Args:
        feat_shape: [T, H, W] original feature map shape
        ref_feat_shape: [T_ref, H_ref, W_ref] reference feature map shape
        device: Device for coordinate tensors
    Returns:
        coords: Tensor of shape (T*H*W, 9), containing (t, h, w, T, H, W, ref_T, ref_H, ref_W)
    """
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    # Generate index ranges
    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    # Create 3D meshgrid (T, H, W)
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_rng, height_rng, width_rng, indexing="ij"
    )

    # Stack and flatten
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)

    # Construct and expand size metadata
    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype
    )
    meta_expanded = meta.expand(coords_flat.size(0), -1)

    # Concatenate and return
    return torch.cat([coords_flat, meta_expanded], dim=-1)


@dataclass
class SingleData:
    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    ref_audio_feat: torch.Tensor
    ref_audio_feat_len: int
    ref_video_feat: torch.Tensor
    ref_video_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    text_offset: int
    coords_style: Literal["v1", "v2"] = "v1"

    def __post_init__(self):
        self.video_token_num = self.video_x_t.shape[0]

        self.ref_video_token_num = self.ref_video_feat.shape[0]
        self.ref_video_feat = self.ref_video_feat[: self.ref_video_feat_len]

        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.ref_audio_feat = self.ref_audio_feat[: self.ref_audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]

        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]
        self.txt_channel = self.txt_feat.shape[-1]

    @property
    def device(self):
        return self.video_x_t.device

    @property
    def default_dtype(self):
        return self.video_x_t.dtype

    @property
    def total_token_num(self):
        return (
            self.video_token_num
            + self.audio_feat_len
            + self.txt_feat_len
            + self.ref_audio_feat_len
            + self.ref_video_feat_len
        )

    @property
    def token_sequence(self):
        tensors_to_concat = [
            self.video_x_t,
            self.audio_x_t,
            self.txt_feat,
            self.ref_audio_feat,
            self.ref_video_feat,
        ]
        max_channel = max(tensor.shape[-1] for tensor in tensors_to_concat)

        padded_tensors = [
            F.pad(t, (0, max_channel - t.shape[-1])) for t in tensors_to_concat
        ]
        ret_val = torch.cat(padded_tensors, dim=0)
        return ret_val

    @property
    def modality_mapping(self):
        v_map = torch.full(
            (self.video_token_num,),
            Modality.VIDEO,
            dtype=torch.int64,
            device=self.device,
        )
        a_map = torch.full(
            (self.audio_feat_len,),
            Modality.AUDIO,
            dtype=torch.int64,
            device=self.device,
        )
        t_map = torch.full(
            (self.txt_feat_len,), Modality.TEXT, dtype=torch.int64, device=self.device
        )
        r_audio_map = torch.full(
            (self.ref_audio_feat_len,),
            Modality.AUDIO,
            dtype=torch.int64,
            device=self.device,
        )

        r_video_map = torch.full(
            (self.ref_video_feat_len,),
            Modality.VIDEO,
            dtype=torch.int64,
            device=self.device,
        )
        modality_mapping = torch.cat(
            [v_map, a_map, t_map, r_audio_map, r_video_map], dim=0
        )
        return modality_mapping

    def default_coords(self, shape, ref_feat_shape, offset_thw=[0, 0, 0]):
        return get_coords(
            shape=shape,
            ref_feat_shape=ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
        )

    @property
    def coords_mapping(self):
        if self.spatial_rope_interpolation == "inter":
            video_ref_feat_shape = (self.t // self.t_patch_size, 32, 32)
        else:
            video_ref_feat_shape = (
                self.t // self.t_patch_size,
                self.h // self.patch_size,
                self.w // self.patch_size,
            )

        video_coords = self.default_coords(
            shape=(
                self.t // self.t_patch_size,
                self.h // self.patch_size,
                self.w // self.patch_size,
            ),
            ref_feat_shape=video_ref_feat_shape,
        )

        # Dynamically compute ref_video spatial dimensions from ref_video_feat_len
        # ref_video_feat_len = (H * W) // 4, assuming H = W (square)
        ref_video_spatial_size = (
            int(math.ceil(math.sqrt(self.ref_video_feat_len)))
            if self.ref_video_feat_len > 0
            else 10
        )

        magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self.default_coords(
            shape=(self.audio_feat_len, 1, 1),
            ref_feat_shape=(magic_audio_ref_t // self.t_patch_size, 1, 1),
        )

        text_coords = self.default_coords(
            shape=(self.txt_feat_len, 1, 1),
            ref_feat_shape=(1, 1, 1),
            offset_thw=[-self.txt_feat_len, 0, 0],
        )

        # Interpolation needs at least two reference positions whenever there is
        # more than one reference-audio token.
        ref_audio_ref_t = math.ceil(((self.ref_audio_feat_len - 1) // 8 + 1) / self.t_patch_size)
        if self.ref_audio_feat_len > 1:
            ref_audio_ref_t = max(ref_audio_ref_t, 2)
        ref_audio_coords = self.default_coords(
            shape=(self.ref_audio_feat_len, 1, 1),
            ref_feat_shape=(ref_audio_ref_t, 1, 1),
            offset_thw=[2 * self.audio_feat_len, 0, 0],
        )

        ref_video_coords = self.default_coords(
            shape=(1, ref_video_spatial_size, ref_video_spatial_size),
            ref_feat_shape=(1, ref_video_spatial_size, ref_video_spatial_size),
            offset_thw=[1000, 0, 0],
        )[: self.ref_video_feat_len]

        coords_mapping = torch.cat(
            [
                video_coords,
                audio_coords,
                text_coords,
                ref_audio_coords,
                ref_video_coords,
            ],
            dim=0,
        )
        return coords_mapping

    def unpack_token_sequence(self, token_sequence):
        video_x_t = token_sequence[: self.video_token_num, : self.video_channel]
        video_x_t = rearrange(
            video_x_t,
            "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
            H=self.h // self.patch_size,
            W=self.w // self.patch_size,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()

        audio_x_t = token_sequence[
            self.video_token_num : self.video_token_num + self.audio_feat_len,
            : self.audio_channel,
        ]
        return video_x_t, audio_x_t


@dataclass
class SimplePackedData:
    items: list[SingleData]

    @property
    def token_sequence(self):
        return torch.cat([item.token_sequence for item in self.items], dim=0)

    @property
    def modality_mapping(self):
        return torch.cat([item.modality_mapping for item in self.items], dim=0)

    @property
    def coords_mapping(self):
        return torch.cat([item.coords_mapping for item in self.items], dim=0)

    @property
    def total_token_num(self):
        return sum([item.total_token_num for item in self.items])

    def __getitem__(self, index):
        return self.items[index]

    @property
    def cu_seqlen(self):
        cu_seqlen = torch.cumsum(
            torch.tensor([item.total_token_num for item in self.items]), dim=0
        )
        cu_seqlen = torch.nn.functional.pad(cu_seqlen, (1, 0))
        return cu_seqlen

    @property
    def max_seqlen(self):
        return max([item.total_token_num for item in self.items])

    def unpack_token_sequence(self, token_sequence):
        video_x_t_list = []
        audio_x_t_list = []

        token_sequence_list = torch.split(
            token_sequence, [item.total_token_num for item in self.items], dim=0
        )
        for item, token_sequence in zip(self.items, token_sequence_list):
            video_x_t, audio_x_t = item.unpack_token_sequence(token_sequence)
            video_x_t_list.append(video_x_t)
            audio_x_t_list.append(audio_x_t)
        return torch.stack(video_x_t_list, dim=0), torch.stack(audio_x_t_list, dim=0)


class Magi2RefinerDataProxy:
    def __init__(self, config: Magi2RefinerDataProxyConfig):
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self.frame_receptive_field = config.frame_receptive_field
        self.spatial_rope_interpolation = config.spatial_rope_interpolation
        self.text_offset = config.text_offset
        self.unfold = UnfoldNd(
            kernel_size=(self.t_patch_size, self.patch_size, self.patch_size),
            stride=(self.t_patch_size, self.patch_size, self.patch_size),
        )
        self.coords_style = config.coords_style
        self.attn_config = dict(config.attn_config or {})
        attn_mode = str(self.attn_config.get("mode", "dense")).lower()
        if attn_mode == "local":
            attn_mode = "window"
        if attn_mode not in {"dense", "window"}:
            raise ValueError(f"Unsupported refiner attention mode: {attn_mode!r}")
        self.use_window_attn = attn_mode == "window"
        self.window_attn_config = dict(self.attn_config.get("window", {}))

        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs):
        """
        Save intermediate data for the process_output stage
        Supports keyword argument calls: saved_for_output(a=1, b=2)
        Can be called multiple times to accumulate data

        Args:
            **kwargs: Key-value pairs to save
        """
        # Update dict directly, supports accumulation across calls
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str):
        """
        Get saved data
        """
        return self._saved_data[key]

    def img2tokens(self, x_t: torch.Tensor):
        x_t_unfolded = self.unfold(x_t)
        x_t = rearrange(
            x_t_unfolded, "N col_dim num_tokens -> N num_tokens col_dim"
        ).contiguous()
        return x_t

    @staticmethod
    def _dim_block_sizes(
        dim_size: int, block_size: int, device: torch.device
    ) -> torch.Tensor:
        sizes = torch.full(
            (math.ceil(dim_size / block_size),),
            block_size,
            device=device,
            dtype=torch.int32,
        )
        sizes[-1] = dim_size - block_size * (sizes.numel() - 1)
        return sizes

    @classmethod
    def _build_video_block_ranges(
        cls,
        patch_nums: tuple[int, int, int],
        block_sizes: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        t_num, h_num, w_num = patch_nums
        bs_t, bs_h, bs_w = block_sizes
        t_sizes = cls._dim_block_sizes(t_num, bs_t, device)
        h_sizes = cls._dim_block_sizes(h_num, bs_h, device)
        w_sizes = cls._dim_block_sizes(w_num, bs_w, device)
        block_sizes_tensor = (
            t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
        ).flatten()
        ends = torch.cumsum(block_sizes_tensor, dim=0)
        starts = ends - block_sizes_tensor
        return torch.stack([starts, ends], dim=1).contiguous()

    @classmethod
    def _build_block_order(
        cls,
        *,
        patch_nums: tuple[int, int, int],
        block_sizes: tuple[int, int, int],
        valid_token_num: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        video_token_num = math.prod(patch_nums)
        video_token_indices = torch.arange(
            video_token_num, device=device, dtype=torch.int32
        ).view(*patch_nums)
        padded_nums = tuple(
            math.ceil(patch_num / block_size) * block_size
            for patch_num, block_size in zip(patch_nums, block_sizes)
        )
        padded_video_token_indices = torch.full(
            padded_nums, -1, dtype=torch.int32, device=device
        )
        padded_video_token_indices[
            : patch_nums[0], : patch_nums[1], : patch_nums[2]
        ] = video_token_indices
        bs_t, bs_h, bs_w = block_sizes
        padded_video_token_indices = (
            padded_video_token_indices.view(
                padded_nums[0] // bs_t,
                bs_t,
                padded_nums[1] // bs_h,
                bs_h,
                padded_nums[2] // bs_w,
                bs_w,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1)
        )
        valid_mask = padded_video_token_indices >= 0
        video_order = padded_video_token_indices[valid_mask]
        tail_indices = torch.arange(
            video_token_num, valid_token_num, device=device, dtype=torch.int32
        )
        token_order = torch.cat([video_order, tail_indices], dim=0)
        token_restore_order = torch.empty_like(token_order)
        token_restore_order[token_order] = torch.arange(
            token_order.numel(), device=device, dtype=torch.int32
        )
        block_grid_shape = (
            math.ceil(patch_nums[0] / bs_t),
            math.ceil(patch_nums[1] / bs_h),
            math.ceil(patch_nums[2] / bs_w),
        )
        video_block_ranges = cls._build_video_block_ranges(
            patch_nums, block_sizes, device
        )
        return token_order, token_restore_order, video_block_ranges, block_grid_shape

    @staticmethod
    def _dense_context_q_ranges(
        q_range: tuple[int, int], block_size: int, device: torch.device
    ) -> torch.Tensor:
        q_start, q_end = q_range
        if q_end <= q_start:
            return torch.empty((0, 2), device=device, dtype=torch.int32)
        q_starts = torch.arange(
            q_start, q_end, max(1, block_size), device=device, dtype=torch.int32
        )
        q_ends = torch.clamp(q_starts + max(1, block_size), max=q_end)
        return torch.stack((q_starts, q_ends), dim=1).contiguous()

    def _build_window_local_attn_handler(
        self, item: SingleData, total_token_num: int, device: torch.device
    ) -> WindowLocalAttnHandler:
        window_cfg = {**self.attn_config, **self.window_attn_config}
        level = str(window_cfg.get("level", "block")).lower()
        if level not in {"block", "frame"}:
            raise ValueError(f"Unsupported window attention level: {level!r}")
        patch_nums = (
            item.t // item.t_patch_size,
            item.h // item.patch_size,
            item.w // item.patch_size,
        )
        token_per_frame = patch_nums[1] * patch_nums[2]
        video_token_num = math.prod(patch_nums)
        tail_range = (video_token_num, total_token_num)
        video_range = (0, video_token_num)

        if level == "block":
            block_sizes = (
                int(
                    window_cfg.get(
                        "block_t_size", self.attn_config.get("block_t_size", 1)
                    )
                ),
                int(
                    window_cfg.get("block_size", self.attn_config.get("block_size", 1))
                ),
                int(
                    window_cfg.get("block_size", self.attn_config.get("block_size", 1))
                ),
            )
            _, _, video_block_ranges, block_grid_shape = self._build_block_order(
                patch_nums=patch_nums,
                block_sizes=block_sizes,
                valid_token_num=total_token_num,
                device=device,
            )
            block_mode = str(
                window_cfg.get(
                    "block_mode", window_cfg.get("window_block_mode", "scan")
                )
            ).lower()
            if block_mode == "grid":
                window_ranges = BlockGridLocalAttn(
                    block_ranges_tensor=video_block_ranges,
                    block_grid_shape=block_grid_shape,
                    radius=(
                        int(window_cfg.get("block_t_radius", 1)),
                        int(window_cfg.get("block_h_radius", 1)),
                        int(window_cfg.get("block_w_radius", 1)),
                    ),
                ).build_ranges()
            elif block_mode == "scan":
                window_ranges = BlockLocalAttn(
                    block_ranges_tensor=video_block_ranges,
                    win_size=int(
                        window_cfg.get("win_size", self.attn_config.get("win_size", 1))
                    ),
                ).build_ranges()
            else:
                raise ValueError(f"Unsupported block window mode: {block_mode!r}")
            video_dense_q_ranges = video_block_ranges
            dense_context_q_block_size = max(
                block_sizes[0] * block_sizes[1] * block_sizes[2], 1
            )
        else:
            frame_idx = torch.arange(patch_nums[0], device=device, dtype=torch.int32)
            frame_ranges = torch.stack(
                (frame_idx * token_per_frame, (frame_idx + 1) * token_per_frame), dim=1
            ).contiguous()
            radius = int(
                window_cfg.get("frame_receptive_field", self.frame_receptive_field)
            )
            if radius < 0:
                raise ValueError(
                    "frame-level window attention requires frame_receptive_field >= 0"
                )
            fwd_start_frame = (frame_idx - radius).clamp(min=0)
            fwd_end_frame = (frame_idx + radius + 1).clamp(max=patch_nums[0])
            fwd_k_ranges = torch.stack(
                (fwd_start_frame * token_per_frame, fwd_end_frame * token_per_frame),
                dim=1,
            )
            window_ranges = BlockLocalAttnRanges(
                frame_ranges, fwd_k_ranges, fwd_k_ranges, frame_ranges, token_per_frame
            )
            video_dense_q_ranges = frame_ranges
            dense_context_q_block_size = max(token_per_frame, 1)

        dense_q_ranges = torch.empty((0, 2), device=device, dtype=torch.int32)
        dense_k_ranges = torch.empty((0, 2), device=device, dtype=torch.int32)
        if tail_range[0] < tail_range[1]:
            q_parts = [
                video_dense_q_ranges,
                self._dense_context_q_ranges(
                    tail_range, dense_context_q_block_size, device
                ),
                self._dense_context_q_ranges(
                    tail_range, dense_context_q_block_size, device
                ),
            ]
            k_parts = [
                torch.tensor(tail_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(video_dense_q_ranges.shape[0], 2),
                torch.tensor(video_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(q_parts[1].shape[0], 2),
                torch.tensor(tail_range, device=device, dtype=torch.int32)
                .view(1, 2)
                .expand(q_parts[2].shape[0], 2),
            ]
            dense_q_ranges = torch.cat(q_parts, dim=0).contiguous()
            dense_k_ranges = torch.cat(k_parts, dim=0).contiguous()

        q_ranges = _cat_ranges(
            dense_q_ranges, window_ranges.fwd_q_ranges, device=device
        )
        k_ranges = _cat_ranges(
            dense_k_ranges, window_ranges.fwd_k_ranges, device=device
        )
        return WindowLocalAttnHandler(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            max_seqlen_q=max(_max_range_len(q_ranges), window_ranges.max_q_len),
            max_seqlen_k=total_token_num,
            attn_type_map=torch.zeros(
                [q_ranges.shape[0]], device=device, dtype=torch.int32
            ),
            bwd_q_ranges=_cat_ranges(
                dense_q_ranges, window_ranges.bwd_q_ranges, device=device
            ),
            bwd_k_ranges=_cat_ranges(
                dense_k_ranges, window_ranges.bwd_k_ranges, device=device
            ),
            bwd_attn_type_map=torch.zeros(
                [q_ranges.shape[0]], device=device, dtype=torch.int32
            ),
            auto_range_merge=bool(window_cfg.get("auto_range_merge", False)),
            sparse_load=bool(window_cfg.get("sparse_load", False)),
        )

    def process_input(self, transported_data):
        # init img2col module

        batch_size, input_video_channel, t, h, w = transported_data.x_t.shape
        # 1. Process video features, keep batch dimension
        x_t = self.img2tokens(transported_data.x_t)

        ref_video_feat = self.img2tokens(transported_data.ref_video_feat)

        # 2. Process audio features, keep batch dimension
        # Assume transported_data.audio_x_t shape is already (N, num_tokens, col_dim)
        audio_x_t = transported_data.audio_x_t.contiguous()

        # If there's reference audio, concatenate along token dim (dim=1)

        ref_audio_feat = transported_data.ref_audio_feat.contiguous()

        # Assume text_in shape is (N, num_tokens, col_dim)
        text_in = transported_data.txt_feat.contiguous()

        simple_packed_data = SimplePackedData(items=[])
        for i in range(batch_size):
            single_data = SingleData(
                video_x_t=x_t[i],
                audio_x_t=audio_x_t[i],
                audio_feat_len=transported_data.audio_feat_len[i],
                ref_audio_feat=ref_audio_feat[i],
                ref_audio_feat_len=transported_data.ref_audio_feat_len[i],
                txt_feat=text_in[i],
                txt_feat_len=transported_data.txt_feat_len[i],
                t=t,
                h=h,
                w=w,
                patch_size=self.patch_size,
                t_patch_size=self.t_patch_size,
                spatial_rope_interpolation=self.spatial_rope_interpolation,
                text_offset=self.text_offset,
                ref_video_feat=ref_video_feat[i],
                ref_video_feat_len=transported_data.ref_video_feat_len[i],
                coords_style=self.coords_style,
            )
            simple_packed_data.items.append(single_data)

        varlen_handler = VarlenHandler(
            cu_seqlens_q=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            cu_seqlens_k=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            max_seqlen_q=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
            max_seqlen_k=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
        )

        x = simple_packed_data.token_sequence
        coords_mapping = simple_packed_data.coords_mapping
        modality_mapping = simple_packed_data.modality_mapping
        token_restore_order = None

        if self.use_window_attn:
            assert batch_size == 1, "window attention only supports batch size 1"
            item = simple_packed_data[0]
            local_attn_handler = self._build_window_local_attn_handler(
                item, int(simple_packed_data.total_token_num), x.device
            )
            level = str(self.window_attn_config.get("level", "block")).lower()
            if level == "block":
                window_cfg = {**self.attn_config, **self.window_attn_config}
                block_sizes = (
                    int(
                        window_cfg.get(
                            "block_t_size", self.attn_config.get("block_t_size", 1)
                        )
                    ),
                    int(
                        window_cfg.get(
                            "block_size", self.attn_config.get("block_size", 1)
                        )
                    ),
                    int(
                        window_cfg.get(
                            "block_size", self.attn_config.get("block_size", 1)
                        )
                    ),
                )
                patch_nums = (
                    item.t // item.t_patch_size,
                    item.h // item.patch_size,
                    item.w // item.patch_size,
                )
                token_order, token_restore_order, _, _ = self._build_block_order(
                    patch_nums=patch_nums,
                    block_sizes=block_sizes,
                    valid_token_num=int(simple_packed_data.total_token_num),
                    device=x.device,
                )
                x = x[token_order]
                coords_mapping = coords_mapping[token_order]
                modality_mapping = modality_mapping[token_order]
        elif self.frame_receptive_field != -1:
            assert batch_size == 1, "local attention only supports batch size 1"
            local_attn_handler = calc_local_attn_ffa_handler(
                num_video_tokens=simple_packed_data[0].video_token_num,
                num_audio_and_txt_tokens=simple_packed_data[0].audio_feat_len
                + simple_packed_data[0].txt_feat_len
                + simple_packed_data[0].ref_audio_feat_len,
                num_frames=t,
                frame_receptive_field=self.frame_receptive_field,
            )
            if isinstance(local_attn_handler.max_seqlen_k, torch.Tensor):
                local_attn_handler.max_seqlen_k = local_attn_handler.max_seqlen_k.item()
            if isinstance(local_attn_handler.max_seqlen_q, torch.Tensor):
                local_attn_handler.max_seqlen_q = local_attn_handler.max_seqlen_q.item()
        else:
            local_attn_handler = None

        self.saved_for_output(
            simple_packed_data=simple_packed_data,
            input_video_channel=input_video_channel,
            token_restore_order=token_restore_order,
        )

        return (x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler)

    def process_output(self, x: torch.Tensor):
        token_restore_order = self.get_saved_data("token_restore_order")
        if token_restore_order is not None:
            x = x[token_restore_order]

        simple_packed_data: SimplePackedData = self.get_saved_data("simple_packed_data")
        return simple_packed_data.unpack_token_sequence(x)
