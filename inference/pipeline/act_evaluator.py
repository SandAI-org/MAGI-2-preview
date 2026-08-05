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

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

from inference.common import DistBroadcaster, OffloadMode, OffloadUtils
from inference.common.magi2_config import EvaluationConfig
from inference.infra.distributed import psm
from inference.model.turbo_vaed import TurboVAED, get_turbo_vaed
from inference.model.vae2_2 import Wan2_2_VAE, get_vae2_2
from inference.utils import print_mem_info_rank_0, print_rank_0
from inference.pipeline.audio_decoder import SAAudioFeatureExtractor, resample_audio_sinc



def _parity_dump(name, tensor):
    d = os.environ.get("PARITY_LATENT_DIR")
    if not d or not psm.is_group_first_rank("cp"):
        return
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.pt")
    torch.save(tensor.detach().cpu(), p)
    print_rank_0(f"[magi2 parity] saved {name} shape={tuple(tensor.shape)} -> {p}")


from .preview_data_proxy import Magi2DataProxy
from .sampler import Magi2PreviewSampler, FlowUniPCMultistepScheduler
from .preview_data_proxy import CFGConfig, SamplerInput


@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor
    ref_audio_feat: torch.Tensor
    ref_audio_feat_len: torch.Tensor
    ref_video_feat: "Optional[torch.Tensor]" = None
    ref_video_feat_len: "Optional[torch.Tensor]" = None



EvalTaskType = Literal["image2video", "text2video"]


def resizecrop(image: Image.Image, th: int, tw: int):
    w, h = image.size
    if w == tw and h == th:
        return image
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    return image.crop((left, top, right, bottom))


class Magi2ActEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        config: EvaluationConfig,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        magi2_refiner: Optional[torch.nn.Module] = None,
    ):
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        self.model = model
        self.model.eval()
        self.config = config
        self.device = device
        self.dtype = weight_dtype

        self._text_enc_mode = OffloadMode.resolve_from_env("text_enc", OffloadMode.CPU)
        self._preview_mode = OffloadMode.resolve_from_env("magi2_preview", OffloadMode.ROUNDTRIP)
        self._refiner_mode = OffloadMode.resolve_from_env("magi2_refiner", OffloadMode.ROUNDTRIP)
        self._vae_mode = OffloadMode.resolve_from_env("vae", OffloadMode.GPU)
        OffloadUtils.setup(self.model, self._preview_mode, self.device)

        self.data_proxy = Magi2DataProxy(config.data_proxy_config)
        self.vae_stride = config.vae_stride
        self.z_dim = config.z_dim
        self.patch_size = config.patch_size
        self.fps = float(config.fps)
        self.shift = config.shift

        cached_vae_dir = config.vae_model_path
        vae_ckpt_path = os.path.join(cached_vae_dir, "Wan2.2_VAE.pth")
        self.vae: Wan2_2_VAE = get_vae2_2(
            vae_ckpt_path, self.device, weight_dtype=torch.float32
        )
        if config.use_turbo_vae and psm.is_group_first_rank("cp"):
            self.turbo_vae: TurboVAED = get_turbo_vaed(
                config.student_config,
                config.student_ckpt,
                self.device,
                weight_dtype=weight_dtype,
            )
        OffloadUtils.setup(self.vae, self._vae_mode, self.device)
        if config.use_turbo_vae and psm.is_group_first_rank("cp"):
            OffloadUtils.setup(self.turbo_vae, self._vae_mode, self.device)
        self.video_processor = VideoProcessor(vae_scale_factor=2 * self.vae_stride[1])
        self.txt_model_path = config.txt_model_path

        self.audio_latent_dim = 64
        self.audio_vae = SAAudioFeatureExtractor(config.audio_model_path) if psm.is_group_first_rank("cp") else None

        self.video_scheduler = FlowUniPCMultistepScheduler()
        self.audio_scheduler = FlowUniPCMultistepScheduler()
        self.cfg_config = CFGConfig(
            use_cfg_trick=config.use_cfg_trick,
            cfg_trick_start_frame=config.cfg_trick_start_frame,
            cfg_trick_value=config.cfg_trick_value,
            use_dynamic_cfg=config.use_dynamic_cfg,
            dynamic_cfg_start_t=config.dynamic_cfg_start_t,
            dynamic_cfg_cutoff_value=config.dynamic_cfg_cutoff_value,
            video_txt_guidance_scale=config.video_txt_guidance_scale,
            audio_txt_guidance_scale=config.audio_txt_guidance_scale,
            use_skimmed_cfg_linear=config.use_skimmed_cfg_linear,
            skimmed_cfg_scale=config.skimmed_cfg_scale,
            cfg_rescale=config.cfg_rescale,
        )
        if self.cfg_config.use_skimmed_cfg_linear or self.cfg_config.cfg_rescale > 0:
            print_rank_0(
                "[magi2 cfg] "
                f"use_skimmed_cfg_linear={self.cfg_config.use_skimmed_cfg_linear}, "
                f"skimmed_cfg_scale={self.cfg_config.skimmed_cfg_scale}, "
                f"cfg_rescale={self.cfg_config.cfg_rescale}"
            )
        self.sampler = Magi2PreviewSampler(self.model, self.data_proxy, self.device, self.dtype)
        print_mem_info_rank_0("[magi2] after init")

        self.text_embedding_broadcaster = DistBroadcaster(
            src_rank=psm.get_global_ranks("cp")[0],
            group=psm.get_parallel_group("cp"),
            dtype=self.dtype,
        )
        self.image_broadcaster = DistBroadcaster(
            src_rank=psm.get_global_ranks("cp")[0],
            group=psm.get_parallel_group("cp"),
            dtype=torch.float32,
        )
        if self.text_embedding_broadcaster.is_src_rank:
            initialize_text_encoder(
                self.txt_model_path,
                self.device,
                self.dtype,
                txt_encoder_type=self.config.txt_encoder_type,
                text_enc_mode=self._text_enc_mode,
            )

        if config.use_magi2_refiner and magi2_refiner is not None:
            self.magi2_refiner = magi2_refiner
            self.magi2_refiner.eval()
            OffloadUtils.setup(self.magi2_refiner, self._refiner_mode, self.device)
            self.magi2_refiner_data_proxy = None
            if (
                hasattr(config, "magi2_refiner_data_proxy_config")
                and config.magi2_refiner_data_proxy_config is not None
            ):
                from .refiner_data_proxy import MMActDataProxy

                self.magi2_refiner_data_proxy = MMActDataProxy(config.magi2_refiner_data_proxy_config)
            self.magi2_refiner_sigmas = ZeroSNRDDPMDiscretization()(
                1000, do_append_zero=False, flip=True
            )

        print_rank_0(
            f"[magi2] offload config: text_enc={self._text_enc_mode.value} "
            f"preview={self._preview_mode.value} refiner={self._refiner_mode.value} "
            f"vae={self._vae_mode.value}"
        )

    def _resolve_lengths(self, seconds: float) -> tuple[int, int]:
        seconds = float(seconds)
        video_frame_length = int(round(seconds * self.fps * 2))
        video_latent_length = (video_frame_length - 1) // self.vae_stride[0] + 1
        audio_latent_fps = float(self.config.data_proxy_config.audio_latent_fps)
        audio_latent_length = int(round(seconds * audio_latent_fps))
        return video_latent_length, audio_latent_length

    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        if self.text_embedding_broadcaster.is_src_rank:
            embedding = get_text_encoder_embedding(prompt, self.dtype, self._text_enc_mode)
        else:
            embedding = None
        return self.text_embedding_broadcaster.broadcast(embedding)

    def get_null_feature(self) -> torch.Tensor:
        return torch.zeros(1, 0, 5120, dtype=self.dtype, device=self.device)

    def encode_image(
        self, image: str | Image.Image, height: int, width: int
    ) -> torch.Tensor:
        if self.image_broadcaster.is_src_rank:
            image = load_image(image)
            image = resizecrop(image, height, width)
            image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=self.device, dtype=self.dtype).unsqueeze(2)
            image = image[:, :3]
            OffloadUtils.maybe_load(self.vae, self._vae_mode, self.device)
            latent = self.vae.encode(image.float())
            OffloadUtils.maybe_offload(self.vae, self._vae_mode)
        else:
            latent = None
        return self.image_broadcaster.broadcast(latent)

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        image: Optional[str | Image.Image],
        eval_task_type: EvalTaskType,
        seconds: float,
        preview_width: int,
        preview_height: int,
        br_num_inference_steps: int,
        refiner_width: Optional[int] = None,
        refiner_height: Optional[int] = None,
        magi2_refiner_num_inference_steps: Optional[int] = None,
        persona_path: Optional[str] = None,
    ):

        OffloadUtils.maybe_load(self.model, self._preview_mode, self.device)

        self.latent_height = (
            preview_height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        )
        self.latent_width = (
            preview_width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        )
        height = self.latent_height * self.vae_stride[1]
        width = self.latent_width * self.vae_stride[2]

        video_latent_length, audio_latent_length = self._resolve_lengths(seconds)
        if eval_task_type == "text2video":
            image = None

        first_frame_feat = (
            self.encode_image(image, height, width) if image is not None else None
        )
        _parity_dump("post_encode", first_frame_feat)
        ref_video_feat = (
            self.encode_image(persona_path, 320, 320)
            if persona_path is not None
            else None
        )

        if self._text_enc_mode == OffloadMode.ROUNDTRIP:
            OffloadUtils.offload(self.model)
            torch.cuda.empty_cache()

        context = self.get_text_embedding(prompt)
        context_null = self.get_null_feature()

        if self._text_enc_mode == OffloadMode.ROUNDTRIP:
            torch.cuda.empty_cache()
            self.model.to(self.device)

        latent = torch.randn(
            1,
            self.z_dim,
            video_latent_length,
            self.latent_height,
            self.latent_width,
            dtype=torch.float32,
            device=self.device,
        )

        # The audio modality carries no user-supplied signal: MAGI-2 ships without an
        # audio encoder, so these tokens are pure noise placeholders. They still take
        # part in attention and in the MoE routing, and the draw itself advances the
        # RNG, so both the count and the ordering here are load-bearing.
        audio_latent = torch.randn(
            1,
            audio_latent_length,
            self.audio_latent_dim,
            dtype=torch.float32,
            device=self.device,
        )


        ref_audio_feat = torch.zeros(
            1, 0, self.audio_latent_dim, dtype=torch.float32, device=self.device
        )

        self.video_scheduler.set_timesteps(
            br_num_inference_steps, device=self.device, shift=self.shift
        )
        self.audio_scheduler.set_timesteps(
            br_num_inference_steps, device=self.device, shift=self.shift
        )

        sampler_input = SamplerInput(
            video_t_list=self.video_scheduler.timesteps,
            audio_t_list=self.audio_scheduler.timesteps,
            latent=latent,
            audio_latent=audio_latent,
            txt_feat=context,
            null_txt_feat=context_null,
            ref_audio_feat=ref_audio_feat,
            ref_video_feat=ref_video_feat,
            first_frame_feat=first_frame_feat,
            video_scheduler=self.video_scheduler,
            audio_scheduler=self.audio_scheduler,
            cfg_config=self.cfg_config,
        )

        torch.cuda.reset_peak_memory_stats()
        print_mem_info_rank_0("[magi2] before preview denoise")
        latent, latent_audio = self.sampler.sample(sampler_input)
        print_mem_info_rank_0("[magi2] after preview denoise")
        _parity_dump("post_preview", latent)

        OffloadUtils.maybe_offload(self.model, self._preview_mode)

        if first_frame_feat is not None:
            latent[:, :, :1] = first_frame_feat[:, :, :1]

        use_magi2_refiner = all(
            [
                hasattr(self, "magi2_refiner") and self.magi2_refiner is not None,
                refiner_width is not None,
                refiner_height is not None,
                magi2_refiner_num_inference_steps is not None,
            ]
        )
        if use_magi2_refiner:
            latent, _refiner_audio = self.evaluate_magi2_refiner_with_latent(
                context=context,
                context_null=context_null,
                image=image,
                latent_video=latent,
                latent_audio=latent_audio,
                refiner_width=refiner_width,
                refiner_height=refiner_height,
                magi2_refiner_num_inference_steps=magi2_refiner_num_inference_steps,
            )
            _parity_dump("post_refiner", latent)

        result = self.post_process(latent, latent_audio)
        return result

    def _forward_magi2_refiner(
        self,
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        txt_feat: torch.Tensor,
        ref_audio_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.magi2_refiner is not None and self.magi2_refiner_data_proxy is not None
        eval_input = EvalInput(
            x_t=latent_video,
            audio_x_t=latent_audio,
            audio_feat_len=torch.tensor([latent_audio.shape[1]]),
            txt_feat=txt_feat.to(torch.float32),
            txt_feat_len=torch.tensor([txt_feat.shape[1]]),
            ref_audio_feat=ref_audio_feat.to(torch.float32),
            ref_audio_feat_len=torch.tensor([ref_audio_feat.shape[1]]),
            ref_video_feat=torch.empty_like(latent_video),
            ref_video_feat_len=torch.tensor([0]),
        )
        model_input = self.magi2_refiner_data_proxy.process_input(eval_input)
        model_output = self.magi2_refiner(*model_input)
        x_video, x_audio = self.magi2_refiner_data_proxy.process_output(model_output)
        return x_video, x_audio

    @torch.inference_mode()
    def evaluate_magi2_refiner_with_latent(
        self,
        context: torch.Tensor,
        context_null: torch.Tensor,
        image: Optional[str | Image.Image],
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        refiner_width: int,
        refiner_height: int,
        magi2_refiner_num_inference_steps: int,
    ) -> torch.Tensor:
        assert self.magi2_refiner is not None

        refiner_latent_height = (
            refiner_height // self.config.magi2_refiner_vae_stride[1] // self.config.magi2_refiner_patch_size[1]
        )
        refiner_latent_height *= self.config.magi2_refiner_patch_size[1]
        refiner_latent_width = (
            refiner_width // self.config.magi2_refiner_vae_stride[2] // self.config.magi2_refiner_patch_size[2]
        )
        refiner_latent_width *= self.config.magi2_refiner_patch_size[2]
        refiner_height = refiner_latent_height * self.config.magi2_refiner_vae_stride[1]
        refiner_width = refiner_latent_width * self.config.magi2_refiner_vae_stride[2]
        print_rank_0(
            "[magi2 refiner] ENTER_REFINER_SAMPLING: "
            f"preview_latent_shape={tuple(latent_video.shape)} "
            f"target_latent_size={refiner_latent_width}x{refiner_latent_height} "
            f"target_resolution={refiner_width}x{refiner_height} "
            f"steps={magi2_refiner_num_inference_steps}"
        )

        first_frame_feat = (
            self.encode_image(image, refiner_height, refiner_width) if image is not None else None
        )
        latent_video = torch.nn.functional.interpolate(
            latent_video,
            size=(latent_video.shape[2] * 2 - 1, refiner_latent_height, refiner_latent_width),
            mode="trilinear",
            align_corners=True,
        )
        refiner_latent_frames = latent_video.shape[2]
        min_latent_frames = self.config.magi2_refiner_min_latent_frames
        if min_latent_frames > refiner_latent_frames:
            latent_video = torch.cat(
                [
                    latent_video,
                    latent_video[:, :, -1:].repeat(
                        1, 1, min_latent_frames - refiner_latent_frames, 1, 1
                    ),
                ],
                dim=2,
            )
            print_rank_0(
                f"[magi2 refiner] temporal pad {refiner_latent_frames} "
                f"-> {latent_video.shape[2]}"
            )
        if self.config.magi2_refiner_noise_value != 0:
            noise = torch.randn_like(latent_video, device=latent_video.device)
            sigmas = self.magi2_refiner_sigmas.to(latent_video.device)
            sigma = sigmas[self.config.magi2_refiner_noise_value]
            latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5

        audio_noise_scale = self.config.magi2_refiner_audio_noise_scale
        magi2_refiner_audio_latent = latent_audio
        if audio_noise_scale < 0:
            # No audio tokens at all: the refiner is asked to improve the video and
            # nothing else, which is what the shipping configuration wants.
            magi2_refiner_audio_latent = torch.zeros(
                1, 0, self.audio_latent_dim, dtype=torch.float32, device=self.device
            )
        if latent_audio.numel() > 0 and audio_noise_scale > 0:
            magi2_refiner_audio_latent = torch.randn_like(
                latent_audio, device=latent_audio.device
            ) * audio_noise_scale + latent_audio * (1 - audio_noise_scale)
        magi2_refiner_ref_audio_feat = torch.zeros(
            1, 0, self.audio_latent_dim, dtype=torch.float32, device=self.device
        )

        video_scheduler = FlowUniPCMultistepScheduler()
        magi2_refiner_shift = (
            self.config.magi2_refiner_shift if self.config.magi2_refiner_shift is not None else self.shift
        )
        video_scheduler.set_timesteps(
            magi2_refiner_num_inference_steps, device=self.device, shift=magi2_refiner_shift
        )
        audio_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler.set_timesteps(
            magi2_refiner_num_inference_steps, device=self.device, shift=magi2_refiner_shift
        )
        video_cfg = (
            torch.tensor(self.config.magi2_refiner_video_txt_guidance_scale, device=self.device)
            .expand(1, 1, latent_video.shape[2], 1, 1)
            .clone()
        )
        if self.config.use_cfg_trick:
            video_cfg[:, :, : self.config.cfg_trick_start_frame] = min(
                self.config.cfg_trick_value, self.config.magi2_refiner_video_txt_guidance_scale
            )
        use_uncond_only = not torch.any(video_cfg != 0).item()
        audio_cfg = self.config.magi2_refiner_audio_txt_guidance_scale

        OffloadUtils.maybe_load(self.magi2_refiner, self._refiner_mode, self.device)
        torch.cuda.reset_peak_memory_stats()
        print_mem_info_rank_0("[magi2] before refiner denoise")

        for step_idx, t in enumerate(video_scheduler.timesteps, start=1):
            if first_frame_feat is not None:
                latent_video[:, :, :1] = first_frame_feat[:, :, :1]

            if use_uncond_only:
                v_cfg_video, v_cfg_audio = self._forward_magi2_refiner(
                    latent_video=latent_video,
                    latent_audio=magi2_refiner_audio_latent,
                    txt_feat=context_null,
                    ref_audio_feat=magi2_refiner_ref_audio_feat,
                )
            else:
                model_video, model_audio = self._forward_magi2_refiner(
                    latent_video=latent_video,
                    latent_audio=magi2_refiner_audio_latent,
                    txt_feat=context,
                    ref_audio_feat=magi2_refiner_ref_audio_feat,
                )
                uncond_video, uncond_audio = self._forward_magi2_refiner(
                    latent_video=latent_video,
                    latent_audio=magi2_refiner_audio_latent,
                    txt_feat=context_null,
                    ref_audio_feat=magi2_refiner_ref_audio_feat,
                )
                v_cfg_video = uncond_video + video_cfg * (model_video - uncond_video)
                v_cfg_audio = uncond_audio + audio_cfg * (model_audio - uncond_audio)
            latent_video = video_scheduler.step(
                v_cfg_video, t, latent_video, return_dict=False
            )[0]

            if v_cfg_audio.numel() > 0 and audio_noise_scale > 0:
                magi2_refiner_audio_latent = audio_scheduler.step(
                    v_cfg_audio, t, magi2_refiner_audio_latent, return_dict=False
                )[0]

            if first_frame_feat is not None:
                latent_video[:, :, :1] = first_frame_feat[:, :, :1]

            print_rank_0(
                f"[magi2 refiner] REFINER_STEP_DONE {step_idx}/{len(video_scheduler.timesteps)}"
            )

        print_mem_info_rank_0("[magi2] after refiner denoise")

        OffloadUtils.maybe_offload(self.magi2_refiner, self._refiner_mode)
        if latent_video.shape[2] > refiner_latent_frames:
            latent_video = latent_video[:, :, :refiner_latent_frames].contiguous()
        return latent_video, magi2_refiner_audio_latent

    def decode_audio(self, audio_latent: torch.Tensor):
        waveform = self.audio_vae.decode(audio_latent.T)
        audio_np = waveform.squeeze(0).T.cpu().numpy()
        return resample_audio_sinc(audio_np, 441 / 512)

    def decode_video(self, latent: torch.Tensor):
        z = latent.squeeze(0).to(self.dtype)
        if z.dim() == 4:
            z = z.unsqueeze(0)
        OffloadUtils.maybe_load(self.vae, self._vae_mode, self.device)
        if hasattr(self, "turbo_vae"):
            OffloadUtils.maybe_load(self.turbo_vae, self._vae_mode, self.device)
        _parity_dump("pre_vae", z)

        decoded = (
            self.turbo_vae.decode(z).float()
            if psm.is_group_first_rank("cp")
            else None
        )
        if decoded is not None:
            _parity_dump("post_vae", decoded)
        OffloadUtils.maybe_offload(self.vae, self._vae_mode)
        if hasattr(self, "turbo_vae"):
            OffloadUtils.maybe_offload(self.turbo_vae, self._vae_mode)
        if decoded is None:
            return None
        videos = decoded if isinstance(decoded, list) else [decoded]
        result = []
        for video in videos:
            video = video.float().mul(0.5).add(0.5).clamp(0, 1)
            if video.ndim == 5:
                video = (
                    video.cpu().permute(0, 2, 3, 4, 1).mul(255).numpy().astype(np.uint8)
                )
                result.extend(list(video))
            else:
                video = (
                    video.cpu().permute(1, 2, 3, 0).mul(255).numpy().astype(np.uint8)
                )
                result.append(video)
        return result

    def post_process(self, latent_video: torch.Tensor, latent_audio: Optional[torch.Tensor] = None):
        torch.cuda.empty_cache()
        latent_save_dir = os.environ.get("MAGI2_SAVE_LATENT_PATH")
        # Every rank reaches this with the same filename, so without the leader
        # check they overwrite each other and the file is whatever won the race.
        if latent_save_dir and psm.is_group_first_rank("cp"):
            if not hasattr(self, "_latent_save_counter"):
                self._latent_save_counter = 0
            os.makedirs(latent_save_dir, exist_ok=True)
            save_path = os.path.join(latent_save_dir, f"latent_{self._latent_save_counter}.pt")
            torch.save(latent_video.cpu(), save_path)
            print_rank_0(f"[magi2] Saved post-refiner latent to {save_path}, shape={tuple(latent_video.shape)}")
            self._latent_save_counter += 1
        torch.cuda.reset_peak_memory_stats()
        print_mem_info_rank_0("[magi2] before vae decode")
        videos_np = self.decode_video(latent_video)
        print_mem_info_rank_0("[magi2] after vae decode")

        # One rank per cp group materializes the result; on the others it would
        # only be a redundant CPU copy of the same video.
        if not psm.is_group_first_rank("cp"):
            return None, None

        video_np = None if videos_np is None else videos_np[0]

        audio_np = None
        if latent_audio is not None and latent_audio.numel() > 0:
            latent_audio_sq = latent_audio.squeeze(0)
            audio_np = self.decode_audio(latent_audio_sq)

        return video_np, audio_np


# -----------------------------------------------------------------------------
# Text encoder cache
# -----------------------------------------------------------------------------

from typing import Literal, Optional

import torch

from inference.model.qwen35 import Qwen35TextEncoder


_text_encoder_cache: Optional[Qwen35TextEncoder] = None


@torch.inference_mode()
def get_text_encoder_embedding(prompt: str, weight_dtype: torch.dtype, mode: OffloadMode) -> torch.Tensor:
    assert _text_encoder_cache is not None, "Text encoder not initialized"
    device = torch.cuda.current_device()
    OffloadUtils.maybe_load(_text_encoder_cache, mode, device)
    embedding = _text_encoder_cache.encode(prompt)
    OffloadUtils.maybe_offload(_text_encoder_cache, mode)
    return embedding.to(weight_dtype).to(device)


def initialize_text_encoder(
    model_path: str,
    device: str,
    weight_dtype: torch.dtype,
    skip_layer: int = 2,
    txt_encoder_type: Literal["qwen35"] = "qwen35",
    text_enc_mode: OffloadMode = OffloadMode.ROUNDTRIP,
) -> None:
    if txt_encoder_type != "qwen35":
        raise ValueError(f"Invalid text encoder type: {txt_encoder_type}")

    text_encoder = Qwen35TextEncoder(
        model_path=model_path,
        device=text_enc_mode.resolve_device(device),
        precision=weight_dtype,
        skip_layer=skip_layer,
    )
    OffloadUtils.setup(text_encoder, text_enc_mode, device)

    global _text_encoder_cache
    _text_encoder_cache = text_encoder


# -----------------------------------------------------------------------------
# Refiner noise schedule
# -----------------------------------------------------------------------------

import torch


class ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        num_timesteps: int = 1000,
        shift_scale: float = 1.0,
        keep_start: bool = False,
        post_shift: bool = False,
    ):
        if keep_start and not post_shift:
            linear_start = linear_start / (
                shift_scale + (1 - shift_scale) * linear_start
            )
        self.num_timesteps = num_timesteps
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64
            )
            ** 2
        )
        alphas = 1.0 - betas.numpy()
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (
                shift_scale + (1 - shift_scale) * self.alphas_cumprod
            )
        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        device: str = "cpu",
        flip: bool = False,
    ):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = (
            torch.cat([sigmas, sigmas.new_zeros([1])]) if do_append_zero else sigmas
        )
        return sigmas if not flip else torch.flip(sigmas, (0,))

    def get_sigmas(self, n: int, device: str = "cpu"):
        if n < self.num_timesteps:
            timesteps = np.linspace(
                self.num_timesteps - 1, 0, n, endpoint=False
            ).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"n must be <= {self.num_timesteps}, got {n}")

        alphas_cumprod = torch.tensor(
            alphas_cumprod, dtype=torch.float32, device=device
        )
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_t = alphas_cumprod_sqrt[-1].clone()
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_t
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (
            alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_t
        )
        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2
                / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5
        return torch.flip(alphas_cumprod_sqrt, (0,))
