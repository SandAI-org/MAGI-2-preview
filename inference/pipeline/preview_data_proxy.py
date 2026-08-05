from __future__ import annotations
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

import torch
from dataclasses import dataclass
@dataclass
class ModelInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]
    t: torch.Tensor
    ref_audio_feat: Optional[torch.Tensor] = None
    ref_audio_feat_len: torch.Tensor | list[int] = None
    ref_video_feat: Optional[torch.Tensor] = None
    ref_video_feat_len: torch.Tensor | list[int] = None
    per_token_video_t: Optional[torch.Tensor] = None
    per_token_audio_t: Optional[torch.Tensor] = None
    # Conditioning images are packed as (B, M, C, 1, H, W) with one (H, W) pair
    # per image in ref_image_feat_len, plus one special token embedding each.
    ref_image_feat: Optional[torch.Tensor] = None
    ref_image_feat_len: Optional[torch.Tensor] = None
    ref_image_special_token_embedding: Optional[torch.Tensor] = None


@dataclass
class CFGConfig:
    use_cfg_trick: bool = False
    cfg_trick_start_frame: int = 0
    cfg_trick_value: float = 0.0
    use_dynamic_cfg: bool = False
    dynamic_cfg_start_t: int = 0
    dynamic_cfg_cutoff_value: float = 0.0
    video_txt_guidance_scale: float = 0.0
    audio_txt_guidance_scale: float = 0.0
    use_ref_for_uncond: bool = False
    use_skimmed_cfg_linear: bool = False
    skimmed_cfg_scale: float = 5.0
    cfg_rescale: float = 0.0


@dataclass
class SamplerInput:
    video_t_list: list[torch.Tensor]
    audio_t_list: list[torch.Tensor]
    latent: torch.Tensor  # B, C, T, H, W
    audio_latent: torch.Tensor  # B, L, C
    txt_feat: torch.Tensor  # B, L, D
    null_txt_feat: torch.Tensor  # B, L, C
    ref_audio_feat: torch.Tensor  # B, L, C
    ref_video_feat: Optional[torch.Tensor]  # B, C, 1, H, W
    video_scheduler: SchedulerMixin
    audio_scheduler: SchedulerMixin
    cfg_config: CFGConfig
    ref_image_feat: Optional[torch.Tensor] = None  # B, M, C, 1, H, W
    ref_image_feat_len: Optional[torch.Tensor] = None  # B, M, 2 holding (H, W)
    ref_image_special_token_embedding: Optional[torch.Tensor] = None  # B, M, D


import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn import functional as F
from unfoldNd import UnfoldNd

from inference.common.magi2_config import DataProxyConfig
from inference.infra.distributed import psm
from inference.model.magi2_preview import (
    Modality,
    VarlenHandler,
    get_coords,
    sinusoidal_embedding_1d,
)


def _to_int(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _pad_cat(
    tensors: list[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if not tensors:
        return torch.empty(0, 0, device=device, dtype=dtype)
    max_channel = max(t.shape[-1] for t in tensors)
    return torch.cat(
        [
            F.pad(t.to(device=device, dtype=dtype), (0, max_channel - t.shape[-1]))
            for t in tensors
        ],
        dim=0,
    )


def _segment_paint(
    values: list[int],
    seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
    output_size: int,
) -> torch.Tensor:
    mapping = torch.empty(output_size, dtype=dtype, device=device)
    offset = 0
    for value, seqlen in zip(values, seqlens):
        mapping[offset : offset + seqlen] = int(value)
        offset += seqlen
    return mapping


def _seqlens2cu_seqlens(
    seqlens: list[int], device: torch.device | None = None
) -> torch.Tensor:
    seqlens_tensor = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return F.pad(torch.cumsum(seqlens_tensor, dim=0), (1, 0))


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _len_to_list(value: torch.Tensor) -> list[int]:
    """Flatten a packed ref-length entry (e.g. [H, W]) back into a python list."""
    return [int(v) for v in value.detach().to(torch.long).reshape(-1).tolist()]


@dataclass
class SingleData:
    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    diffusion_t: Optional[torch.Tensor] = None
    per_token_video_t: Optional[torch.Tensor] = None
    per_token_audio_t: Optional[torch.Tensor] = None
    time_channel_dim: int = 0
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0
    time_pos_fps: float = 3.125
    ref_image_feats: Optional[list[torch.Tensor]] = None
    ref_image_feat_lens: Optional[list[list[int]]] = None
    ref_image_special_tokens: Optional[list[torch.Tensor]] = None

    def __post_init__(self) -> None:
        self.video_token_num = self.video_x_t.shape[0]
        self.origin_audio_feat_len = self.audio_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]
        if self.per_token_audio_t is not None:
            self.per_token_audio_t = self.per_token_audio_t[: self.audio_feat_len]

        self.ref_image_feats = self.ref_image_feats or []
        self.ref_image_feat_lens = self.ref_image_feat_lens or []
        self.ref_image_special_tokens = self.ref_image_special_tokens or []
        # feat_lens holds the patch-grid (H, W) per image, so their product is the
        # token count that survived img2tokens for that image.
        self.ref_image_token_nums: list[int] = [
            int(math.prod(feat_len)) for feat_len in self.ref_image_feat_lens
        ]
        self.ref_image_feats = [
            feat[:num] for feat, num in zip(self.ref_image_feats, self.ref_image_token_nums)
        ]
        self.num_ref_images = len(self.ref_image_feats)
        self.total_ref_image_feat_len = sum(self.ref_image_token_nums)

        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.video_x_t.device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.video_x_t.dtype

    @property
    def add_time_token(self) -> bool:
        return self.diffusion_t is not None

    @property
    def total_token_num(self) -> int:
        total = self.video_token_num + self.audio_feat_len + self.txt_feat_len
        # Each conditioning image also contributes one leading special token.
        total += self.total_ref_image_feat_len + self.num_ref_images
        return total + (1 if self.add_time_token else 0)

    @property
    def feat_to_cat(self) -> list[torch.Tensor]:
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        for k in range(self.num_ref_images):
            tensors.append(self.ref_image_special_tokens[k])
            tensors.append(self.ref_image_feats[k])
        if self.add_time_token:
            tensors.append(
                self.diffusion_t.to(
                    device=self.device, dtype=self.default_dtype
                ).reshape(1, 1)
            )
        return tensors

    @property
    def token_sequence(self) -> torch.Tensor:
        return _pad_cat(self.feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_map_seqlens(self) -> tuple[list[int], list[int]]:
        seqlens = [self.video_token_num, self.audio_feat_len, self.txt_feat_len]
        maps = [Modality.VIDEO, Modality.AUDIO, Modality.TEXT]
        for k in range(self.num_ref_images):
            # The special token comes from the text encoder, so it routes as text
            # while the image patches themselves route as video.
            seqlens.append(1)
            maps.append(Modality.TEXT)
            seqlens.append(self.ref_image_token_nums[k])
            maps.append(Modality.VIDEO)
        if self.add_time_token:
            seqlens.append(1)
            maps.append(Modality.TIME)
        return seqlens, maps

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens, maps = self.modality_map_seqlens
        return _segment_paint(
            maps,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    def _default_coords(
        self,
        shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        offset_thw: tuple[int, int, int] = (0, 0, 0),
        time_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return get_coords(
            shape,
            ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
            time_positions=time_positions,
        )

    @property
    def coords_to_cat(self) -> list[torch.Tensor]:
        t_steps = self.t // self.t_patch_size
        h_steps = self.h // self.patch_size
        w_steps = self.w // self.patch_size

        if self.spatial_rope_interpolation == "inter":
            video_h_ref, video_w_ref = 32, 32
        else:
            video_h_ref, video_w_ref = h_steps, w_steps

        video_coords = self._default_coords(
            (t_steps, h_steps, w_steps), (t_steps, video_h_ref, video_w_ref)
        )
        magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self._default_coords(
            (self.audio_feat_len, 1, 1),
            (magic_audio_ref_t // self.t_patch_size, 1, 1),
        )

        coords = [
            video_coords,
            audio_coords,
            self._default_coords(
                (self.txt_feat_len, 1, 1),
                (1, 1, 1),
                offset_thw=(-self.txt_feat_len, 0, 0),
            ),
        ]

        for k in range(self.num_ref_images):
            token_len = self.ref_image_token_nums[k]
            if len(self.ref_image_feat_lens[k]) >= 2:
                h_i = int(self.ref_image_feat_lens[k][0])
                w_i = int(self.ref_image_feat_lens[k][1])
            else:
                h_i = w_i = int(math.ceil(math.sqrt(token_len)))
            # Images sit past the generated clip on the time axis, one latent step
            # apart, with a gap of 1 so they never alias the last video frame.
            t_off = t_steps + 2 + k
            coords.append(
                torch.tensor(
                    [[t_off, -1, -1, 1, h_i, w_i, 1, h_i, w_i]],
                    device=self.device,
                    dtype=self.default_dtype,
                )
            )
            coords.append(
                self._default_coords(
                    (1, h_i, w_i), (1, h_i, w_i), offset_thw=(t_off, 0, 0)
                )[:token_len]
            )

        if self.add_time_token:
            coords.append(self._default_coords((1, 1, 1), (1, 1, 1))[:1])
        return coords

    @property
    def coords_mapping(self) -> torch.Tensor:
        return torch.cat(self.coords_to_cat, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        if self.time_channel_dim == 0:
            return torch.empty(self.total_token_num, 0, device=self.device)
        assert self.per_token_video_t is not None and self.per_token_audio_t is not None
        parts = [
            self.per_token_video_t.squeeze(-1),
            self.per_token_audio_t.squeeze(-1),
            torch.zeros(self.txt_feat_len, device=self.device),
        ]
        for k in range(self.num_ref_images):
            # Images are clean conditioning, so their diffusion time is 0.
            parts.append(torch.zeros(1, device=self.device))
            parts.append(torch.zeros(self.ref_image_token_nums[k], device=self.device))
        if self.add_time_token:
            parts.append(self.diffusion_t.reshape(1).to(self.device))
        raw_t = torch.cat(parts, dim=0)
        if self.time_channel_dim == 1:
            return raw_t.unsqueeze(-1)
        return sinusoidal_embedding_1d(self.time_channel_dim, raw_t)


@dataclass
class SimplePackedData:
    items: list[SingleData]

    def __post_init__(self) -> None:
        if len(self.items) == 0:
            raise ValueError("SimplePackedData must contain at least one item.")
        self._total_token_num_list = [item.total_token_num for item in self.items]
        self._total_token_num_sum = sum(self._total_token_num_list)
        self._total_token_num_max = max(self._total_token_num_list)
        self._total_token_cu_seqlens = _seqlens2cu_seqlens(self._total_token_num_list)

    @property
    def device(self) -> torch.device:
        return self.items[0].device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.items[0].default_dtype

    @property
    def token_sequence(self) -> torch.Tensor:
        feat_to_cat = list(chain.from_iterable(item.feat_to_cat for item in self.items))
        return _pad_cat(feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens: list[int] = []
        maps: list[int] = []
        for item in self.items:
            item_seqlens, item_maps = item.modality_map_seqlens
            seqlens.extend(item_seqlens)
            maps.extend(item_maps)
        return _segment_paint(
            maps,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    @property
    def coords_mapping(self) -> torch.Tensor:
        coords = list(chain.from_iterable(item.coords_to_cat for item in self.items))
        return torch.cat(coords, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        return torch.cat([item.time_token_sequence for item in self.items], dim=0)

    @property
    def total_token_num(self) -> int:
        return self._total_token_num_sum

    @property
    def cu_seqlen(self) -> torch.Tensor:
        return self._total_token_cu_seqlens.clone()

    @property
    def max_seqlen(self) -> int:
        return self._total_token_num_max

    def __getitem__(self, index: int) -> SingleData:
        return self.items[index]

    def depack_token_sequence(
        self, token_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        videos, audios = [], []
        for item, token_slice in zip(
            self.items,
            torch.split(token_sequence, [i.total_token_num for i in self.items], dim=0),
        ):
            video_flat = token_slice[: item.video_token_num, : item.video_channel]
            c_out = item.video_channel // (
                item.t_patch_size * item.patch_size * item.patch_size
            )
            video = rearrange(
                video_flat,
                "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
                T=item.t // item.t_patch_size,
                H=item.h // item.patch_size,
                W=item.w // item.patch_size,
                pT=item.t_patch_size,
                pH=item.patch_size,
                pW=item.patch_size,
                C=c_out,
            ).contiguous()
            audio = torch.zeros(
                item.origin_audio_feat_len,
                item.audio_channel,
                device=token_sequence.device,
                dtype=token_sequence.dtype,
            )
            audio[: item.audio_feat_len] = token_slice[
                item.video_token_num : item.video_token_num + item.audio_feat_len,
                : item.audio_channel,
            ]
            videos.append(video)
            audios.append(audio)
        return torch.stack(videos, dim=0), torch.stack(audios, dim=0)


class Magi2DataProxy:
    def __init__(self, config: DataProxyConfig) -> None:
        self.config = config
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self.unfold = UnfoldNd(
            kernel_size=(self.t_patch_size, self.patch_size, self.patch_size),
            stride=(self.t_patch_size, self.patch_size, self.patch_size),
        )
        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs: Any) -> None:
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str) -> Any:
        return self._saved_data[key]

    def _reduce_max_token_num_for_ep_cp(self, total_token_num: int) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return total_token_num

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        total = torch.tensor([total_token_num], device=device, dtype=torch.int64)
        max_sizes = [total]

        ep_size = psm.get_world_size("ep")
        if ep_size > 1:
            max_size_in_ep = total.clone()
            dist.all_reduce(
                max_size_in_ep, op=dist.ReduceOp.MAX, group=psm.get_parallel_group("ep")
            )
            max_sizes.append(max_size_in_ep)

        cp_size = psm.get_world_size("cp")
        if cp_size > 1:
            max_size_in_cp = total.clone()
            dist.all_reduce(
                max_size_in_cp, op=dist.ReduceOp.MAX, group=psm.get_parallel_group("cp")
            )
            max_sizes.append(max_size_in_cp)

        return max(int(max_size.item()) for max_size in max_sizes)

    def _pad_for_ep_cp(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        time_token_sequence: torch.Tensor,
        varlen_handler: VarlenHandler,
        align_to: int = 48,
    ):
        cp_size = max(1, psm.get_world_size("cp"))
        ep_size = max(1, psm.get_world_size("ep"))
        if cp_size <= 1 and ep_size <= 1:
            return (
                x,
                coords_mapping,
                modality_mapping,
                time_token_sequence,
                varlen_handler,
                0,
            )

        total_token_num = x.shape[0]
        target_size = self._reduce_max_token_num_for_ep_cp(total_token_num)
        padded_size = _ceil_div(target_size, align_to) * align_to
        pad_size = padded_size - total_token_num
        if pad_size <= 0:
            return (
                x,
                coords_mapping,
                modality_mapping,
                time_token_sequence,
                varlen_handler,
                0,
            )

        x = F.pad(x, (0, 0, 0, pad_size), value=0)
        pad_coords = get_coords(
            shape=(pad_size, 1, 1),
            ref_feat_shape=(pad_size, 1, 1),
            offset_thw=(0, 0, 0),
            device=x.device,
            dtype=torch.float32,
        )
        coords_mapping = torch.cat([coords_mapping, pad_coords], dim=0)
        modality_mapping = F.pad(
            modality_mapping, (0, pad_size), value=int(Modality.TEXT)
        )
        if time_token_sequence.shape[-1] == 0:
            time_token_sequence = time_token_sequence.new_zeros(padded_size, 0)
        else:
            time_token_sequence = F.pad(
                time_token_sequence, (0, 0, 0, pad_size), value=0
            )
        varlen_handler = VarlenHandler(
            cu_seqlens_q=torch.cat(
                [
                    varlen_handler.cu_seqlens_q,
                    torch.tensor([padded_size], device=x.device, dtype=torch.int32),
                ]
            ),
            cu_seqlens_k=torch.cat(
                [
                    varlen_handler.cu_seqlens_k,
                    torch.tensor([padded_size], device=x.device, dtype=torch.int32),
                ]
            ),
            max_seqlen_q=max(varlen_handler.max_seqlen_q, pad_size),
            max_seqlen_k=max(varlen_handler.max_seqlen_k, pad_size),
        )
        return (
            x,
            coords_mapping,
            modality_mapping,
            time_token_sequence,
            varlen_handler,
            pad_size,
        )

    def img2tokens(self, x_t: torch.Tensor) -> torch.Tensor:
        kernel_size = (self.t_patch_size, self.patch_size, self.patch_size)
        if not all(s >= k for s, k in zip(x_t.shape[2:], kernel_size)):
            return torch.empty(
                x_t.shape[0],
                0,
                x_t.shape[1] * math.prod(kernel_size),
                device=x_t.device,
                dtype=x_t.dtype,
            )
        x_t = self.unfold(x_t)
        return rearrange(
            x_t, "N col_dim num_tokens -> N num_tokens col_dim"
        ).contiguous()

    def process_input(
        self, data: ModelInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler, torch.Tensor]:
        batch_size, input_video_channel, t, h, w = data.x_t.shape
        video_tokens = self.img2tokens(data.x_t)
        audio_tokens = data.audio_x_t.contiguous()
        text_tokens = data.txt_feat.contiguous()

        ptv_tokens = None
        pta = None
        if self.config.time_channel_dim > 0 and data.per_token_video_t is not None:
            ptv_tokens = self.img2tokens(data.per_token_video_t)[:, :, :1]
            pta = data.per_token_audio_t

        ref_image_data = data.ref_image_feat
        ref_image_feat_len = data.ref_image_feat_len
        ref_image_special_token_embedding = data.ref_image_special_token_embedding
        has_ref_image = (
            ref_image_data is not None
            and ref_image_data.ndim >= 5
            and ref_image_feat_len is not None
            and ref_image_feat_len.ndim >= 2
            and ref_image_special_token_embedding is not None
            and ref_image_special_token_embedding.ndim >= 3
        )
        num_ref_image = ref_image_data.shape[1] if has_ref_image else 0
        ref_device, ref_dtype = text_tokens.device, text_tokens.dtype

        items = []
        for i in range(batch_size):
            ref_image_feats_i: list[torch.Tensor] = []
            ref_image_feat_lens_i: list[list[int]] = []
            ref_image_special_tokens_i: list[torch.Tensor] = []
            for k in range(num_ref_image):
                # img2tokens works on a batch, so the single image is given a
                # batch axis and then squeezed back out.
                feat = self.img2tokens(ref_image_data[i, k].unsqueeze(0)).squeeze(0)
                ref_image_feats_i.append(feat)
                ref_image_feat_lens_i.append(_len_to_list(ref_image_feat_len[i, k]))
                ref_image_special_tokens_i.append(
                    ref_image_special_token_embedding[i, k]
                    .to(device=ref_device, dtype=ref_dtype)
                    .unsqueeze(0)
                )

            items.append(
                SingleData(
                    video_x_t=video_tokens[i],
                    audio_x_t=audio_tokens[i],
                    audio_feat_len=_to_int(data.audio_feat_len[i]),
                    txt_feat=text_tokens[i],
                    txt_feat_len=_to_int(data.txt_feat_len[i]),
                    ref_image_feats=ref_image_feats_i,
                    ref_image_feat_lens=ref_image_feat_lens_i,
                    ref_image_special_tokens=ref_image_special_tokens_i,
                    t=t,
                    h=h,
                    w=w,
                    patch_size=self.patch_size,
                    t_patch_size=self.t_patch_size,
                    spatial_rope_interpolation=self.config.spatial_rope_interpolation,
                    diffusion_t=data.t[i] if self.config.add_time_token else None,
                    per_token_video_t=ptv_tokens[i] if ptv_tokens is not None else None,
                    per_token_audio_t=pta[i] if pta is not None else None,
                    time_channel_dim=self.config.time_channel_dim,
                    time_pos_fps=self.config.time_pos_fps,
                    vae_first_latent_is_image=self.config.vae_first_latent_is_image,
                    video_fps=self.config.video_fps,
                )
            )

        packed = SimplePackedData(items)
        varlen_handler = VarlenHandler(
            cu_seqlens_q=packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32),
            cu_seqlens_k=packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32),
            max_seqlen_q=packed.max_seqlen,
            max_seqlen_k=packed.max_seqlen,
        )
        (
            token_sequence,
            coords_mapping,
            modality_mapping,
            time_token_sequence,
            varlen_handler,
            pad_size,
        ) = self._pad_for_ep_cp(
            packed.token_sequence,
            packed.coords_mapping,
            packed.modality_mapping,
            packed.time_token_sequence,
            varlen_handler,
        )

        self.saved_for_output(
            simple_packed_data=packed,
            input_video_channel=input_video_channel,
            pad_size=pad_size,
        )
        return (
            token_sequence,
            coords_mapping,
            modality_mapping,
            varlen_handler,
            time_token_sequence,
        )

    def process_output(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed: SimplePackedData = self.get_saved_data("simple_packed_data")
        pad_size = self.get_saved_data("pad_size")
        if pad_size > 0:
            x = x[:-pad_size]
        x_video, x_audio = packed.depack_token_sequence(x)
        return x_video, x_audio
