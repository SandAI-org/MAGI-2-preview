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
import torch.nn.functional as F
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from inference.infra.distributed import psm

from .preview_data_proxy import CFGConfig, Magi2DataProxy, ModelInput, SamplerInput




def _pad_or_trim(tensor, target_size: int, dim: int, pad_value: float = 0.0):
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding_tuple), 'constant', pad_value), current_size
    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], current_size

class Magi2PreviewSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        data_proxy: Magi2DataProxy,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.data_proxy = data_proxy
        self.device = device
        self.dtype = dtype

    def forward(self, model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = self.data_proxy.process_input(model_input)
        self.model.to(self.device)
        model_output = self.model(*model_input)
        return self.data_proxy.process_output(model_output)

    def sample(
        self, sampler_input: SamplerInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_cfgs, audio_cfgs = self.precompute_cfg(
            sampler_input.video_t_list,
            sampler_input.latent.shape[2],
            sampler_input.cfg_config,
        )

        latent = sampler_input.latent.clone()
        audio_latent = sampler_input.audio_latent.clone()

        progress = tqdm(
            zip(sampler_input.video_t_list, video_cfgs, audio_cfgs),
            total=len(sampler_input.video_t_list),
            disable=psm.get_global_rank() != 0,
        )
        for t, video_cfg, audio_cfg in progress:
            if sampler_input.first_frame_feat is not None:
                latent[:, :, :1] = sampler_input.first_frame_feat[:, :, :1]

            model_input = self.prepare_model_input(
                latent=latent,
                audio_latent=audio_latent,
                txt_feat=sampler_input.txt_feat,
                null_txt_feat=sampler_input.null_txt_feat,
                ref_audio_feat=sampler_input.ref_audio_feat,
                ref_video_feat=sampler_input.ref_video_feat,
                t=t,
                has_first_frame=sampler_input.first_frame_feat is not None,
            )
            model_pred = self.forward(model_input)

            latent, audio_latent, v_cfg_video, v_cfg_audio = self.step(
                model_pred,
                latent,
                audio_latent,
                video_cfg,
                audio_cfg,
                sampler_input.video_scheduler,
                sampler_input.audio_scheduler,
                t,
                cfg_config=sampler_input.cfg_config,
            )

            if sampler_input.first_frame_feat is not None:
                latent[:, :, :1] = sampler_input.first_frame_feat[:, :, :1]


        return latent, audio_latent

    def prepare_model_input(
        self,
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        txt_feat: torch.Tensor,
        null_txt_feat: torch.Tensor,
        ref_audio_feat: Optional[torch.Tensor] = None,
        ref_video_feat: Optional[torch.Tensor] = None,
        t: torch.Tensor | None = None,
        has_first_frame: bool = False,
    ) -> ModelInput:
        audio_latent_len = audio_latent.shape[1]
        ref_audio_feat_len = ref_audio_feat.shape[1]
        txt_feat_len = txt_feat.shape[1]
        null_txt_feat_len = null_txt_feat.shape[1]
        target_length = max(txt_feat_len, null_txt_feat_len)

        txt_feat, _ = _pad_or_trim(txt_feat, target_length, 1)
        null_txt_feat, _ = _pad_or_trim(null_txt_feat, target_length, 1)

        if ref_video_feat is None:
            ref_video_feat = torch.empty_like(latent)
            ref_video_feat_len = 0
        else:
            ref_video_feat_len = (
                ref_video_feat.shape[3] * ref_video_feat.shape[4]
            ) // 4
        ref_video_feat = torch.cat(
            [ref_video_feat, torch.zeros_like(ref_video_feat)], dim=0
        )
        ref_video_feat_len = torch.tensor(
            [ref_video_feat_len, ref_video_feat_len], device=latent.device
        )

        if t is None:
            t_value = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)
        elif isinstance(t, torch.Tensor):
            t_value = t.reshape(-1)[0].to(device=latent.device, dtype=latent.dtype)
        else:
            t_value = torch.tensor(float(t), device=latent.device, dtype=latent.dtype)
        t_batch = torch.stack([t_value, t_value])
        t_normalized = t_batch / 1000.0

        batch_cfg = 2
        _, _, video_t, video_h, video_w = latent.shape
        audio_t = audio_latent.shape[1]
        per_token_video_t = (
            t_normalized.view(-1, 1, 1, 1, 1)
            .expand(batch_cfg, 1, video_t, video_h, video_w)
            .clone()
        )
        if has_first_frame:
            per_token_video_t[:, :, 0, :, :] = 0.0
        per_token_audio_t = (
            t_normalized.view(-1, 1, 1).expand(batch_cfg, audio_t, 1).clone()
        )

        return ModelInput(
            x_t=torch.cat([latent, latent], dim=0),
            audio_x_t=torch.cat([audio_latent, audio_latent], dim=0),
            audio_feat_len=torch.tensor(
                [audio_latent_len, audio_latent_len], device=latent.device
            ),
            txt_feat=torch.cat([txt_feat, null_txt_feat], dim=0),
            txt_feat_len=torch.tensor(
                [txt_feat_len, null_txt_feat_len], device=latent.device
            ),
            t=t_normalized,
            per_token_video_t=per_token_video_t,
            per_token_audio_t=per_token_audio_t,
            ref_audio_feat=torch.cat(
                [ref_audio_feat, torch.zeros_like(ref_audio_feat)], dim=0
            ),
            ref_audio_feat_len=torch.tensor(
                [ref_audio_feat_len, ref_audio_feat_len], device=latent.device
            ),
            ref_video_feat=ref_video_feat,
            ref_video_feat_len=ref_video_feat_len,
        )

    def precompute_cfg(
        self, t_list: list[torch.Tensor], latent_length: int, cfg_config: CFGConfig
    ) -> tuple[list[torch.Tensor], list[float]]:
        all_video_cfgs = []
        all_audio_cfgs = []

        if cfg_config.use_cfg_trick:
            video_cfg_base = (
                torch.tensor(cfg_config.video_txt_guidance_scale, device=self.device)
                .expand(1, 1, latent_length, 1, 1)
                .clone()
            )
            video_cfg_base[:, :, : cfg_config.cfg_trick_start_frame] = min(
                cfg_config.cfg_trick_value, cfg_config.video_txt_guidance_scale
            )
        else:
            video_cfg_base = torch.tensor(
                [cfg_config.video_txt_guidance_scale], device=self.device
            )

        for t in t_list:
            video_cfg = video_cfg_base.clone()
            if cfg_config.use_dynamic_cfg and t < cfg_config.dynamic_cfg_start_t:
                video_cfg[video_cfg > cfg_config.dynamic_cfg_cutoff_value] = (
                    cfg_config.dynamic_cfg_cutoff_value
                )
            all_video_cfgs.append(video_cfg)
            all_audio_cfgs.append(cfg_config.audio_txt_guidance_scale)

        return all_video_cfgs, all_audio_cfgs

    @staticmethod
    def _cfg_scale_tensor(
        guidance_scale: torch.Tensor | float, like: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(guidance_scale, torch.Tensor):
            return guidance_scale.to(device=like.device, dtype=like.dtype)
        return torch.tensor(float(guidance_scale), device=like.device, dtype=like.dtype)

    @classmethod
    def _get_skimming_mask(
        cls,
        x_orig: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        x_orig = x_orig.to(device=cond.device, dtype=cond.dtype)
        denoised = x_orig - (
            (x_orig - uncond) + guidance_scale * ((x_orig - cond) - (x_orig - uncond))
        )
        matching_pred_signs = (cond - uncond).sign() == cond.sign()
        matching_diff_after = (
            cond.sign()
            == (cond * guidance_scale - uncond * (guidance_scale - 1)).sign()
        )
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        return matching_pred_signs & matching_diff_after & deviation_influence

    @classmethod
    def _apply_skimmed_cfg_linear(
        cls,
        x_orig: Optional[torch.Tensor],
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
        cfg_config: Optional[CFGConfig],
    ) -> torch.Tensor:
        if (
            cfg_config is None
            or not cfg_config.use_skimmed_cfg_linear
            or x_orig is None
            or not torch.any(uncond).item()
        ):
            return uncond

        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        valid_scale = guidance_scale > 1
        if not torch.any(valid_scale).item():
            return uncond

        scale_delta = torch.where(
            valid_scale, guidance_scale - 1, torch.ones_like(guidance_scale)
        )
        fallback_weight = (cfg_config.skimmed_cfg_scale - 1) / scale_delta

        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, cond, uncond, guidance_scale)
        uncond = torch.where(skim_mask & valid_scale, target_uncond, uncond)

        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, uncond, cond, guidance_scale)
        return torch.where(skim_mask & valid_scale, target_uncond, uncond)

    @staticmethod
    def _cfg_rescale_dims(tensor: torch.Tensor) -> tuple[int, ...]:
        if tensor.ndim <= 1:
            return ()
        if tensor.ndim == 2:
            return (1,)
        if tensor.ndim == 3:
            return (1,)
        return tuple(range(2, tensor.ndim))

    @classmethod
    def _apply_cfg_rescale(
        cls, pos: torch.Tensor, cfg: torch.Tensor, cfg_config: Optional[CFGConfig]
    ) -> torch.Tensor:
        if cfg_config is None or cfg_config.cfg_rescale <= 0 or cfg.numel() == 0:
            return cfg

        spatial_dims = cls._cfg_rescale_dims(cfg)
        if not spatial_dims:
            return cfg

        pos_std = pos.float().std(dim=spatial_dims, keepdim=True, unbiased=False)
        cfg_std = (
            cfg.float()
            .std(dim=spatial_dims, keepdim=True, unbiased=False)
            .clamp_min(1e-6)
        )
        factor = cfg_config.cfg_rescale * (pos_std / cfg_std) + (
            1 - cfg_config.cfg_rescale
        )
        return cfg * factor.to(device=cfg.device, dtype=cfg.dtype)

    def cfg_velocity(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        video_txt_guidance_scale: torch.Tensor,
        audio_txt_guidance_scale: torch.Tensor,
        cfg_config: Optional[CFGConfig] = None,
        latent: Optional[torch.Tensor] = None,
        audio_latent: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cond_uncond_video = model_output[0]
        cond_uncond_audio = model_output[1]

        if cond_uncond_video.numel() > 0:
            cond_video, uncond_video = cond_uncond_video[0:1], cond_uncond_video[1:2]
            uncond_video = self._apply_skimmed_cfg_linear(
                latent, cond_video, uncond_video, video_txt_guidance_scale, cfg_config
            )
            cfg_video = uncond_video + video_txt_guidance_scale * (
                cond_video - uncond_video
            )
            cfg_video = self._apply_cfg_rescale(cond_video, cfg_video, cfg_config)
        else:
            cfg_video = None

        if cond_uncond_audio.numel() > 0:
            cond_audio, uncond_audio = cond_uncond_audio[0:1], cond_uncond_audio[1:2]
            uncond_audio = self._apply_skimmed_cfg_linear(
                audio_latent,
                cond_audio,
                uncond_audio,
                audio_txt_guidance_scale,
                cfg_config,
            )
            cfg_audio = uncond_audio + audio_txt_guidance_scale * (
                cond_audio - uncond_audio
            )
            cfg_audio = self._apply_cfg_rescale(cond_audio, cfg_audio, cfg_config)
        else:
            cfg_audio = None

        return cfg_video, cfg_audio

    def step(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        video_txt_guidance_scale: torch.Tensor,
        audio_txt_guidance_scale: torch.Tensor,
        video_scheduler: SchedulerMixin,
        audio_scheduler: SchedulerMixin,
        t: torch.Tensor,
        cfg_config: Optional[CFGConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cfg_video, cfg_audio = self.cfg_velocity(
            model_output,
            video_txt_guidance_scale,
            audio_txt_guidance_scale,
            cfg_config,
            latent,
            audio_latent,
        )
        if cfg_video is not None:
            latent = video_scheduler.step(cfg_video, t, latent, return_dict=False)[0]
        if cfg_audio is not None:
            audio_latent = audio_scheduler.step(
                cfg_audio, t, audio_latent, return_dict=False
            )[0]
        return latent, audio_latent, cfg_video, cfg_audio


# -----------------------------------------------------------------------------
# Flow UniPC scheduler
# -----------------------------------------------------------------------------

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `UniPCMultistepScheduler` is a training-free framework designed for the fast sampling of diffusion models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, default `2`):
            The UniPC order which can be any positive integer. The effective order of accuracy is `solver_order + 1`
            due to the UniC. It is recommended to use `solver_order=2` for guided sampling, and `solver_order=3` for
            unconditional sampling.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler, which predicts
            the flow of the diffusion process.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and `predict_x0=True`.
        predict_x0 (`bool`, defaults to `True`):
            Whether to use the updating algorithm on the predicted x0.
        solver_type (`str`, default `bh2`):
            Solver type for UniPC. It is recommended to use `bh1` for unconditional sampling when steps < 10, and `bh2`
            otherwise.
        lower_order_final (`bool`, default `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        disable_corrector (`list`, default `[]`):
            Decides which step to disable the corrector to mitigate the misalignment between `epsilon_theta(x_t, c)`
            and `epsilon_theta(x_t^c, c)` which can influence convergence for a large guidance scale. Corrector is
            usually disabled during the first few steps.
        solver_p (`SchedulerMixin`, default `None`):
            Any other scheduler that if specified, the algorithm becomes `solver_p + UniC`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}"
                )

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[
            ::-1
        ].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(
                self.sigma_max, self.sigma_min, num_inference_steps + 1
            ).copy()[:-1]  # type: ignore

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)  # type: ignore
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # type: ignore

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)  # pyright: ignore

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.int64
        )

        self.num_inference_steps = len(timesteps)  # type: ignore

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.set_timesteps
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(
        self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated "
                "and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: Optional[torch.Tensor] = None,
        order: Optional[int] = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError(" missing `order` as a required keyward argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated "
                "and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )  # pyright: ignore
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s: Optional[List[Any]] = []
        for i in range(1, order):
            si = self.step_index - i  # pyright: ignore
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)  # type: ignore

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:  # type: ignore
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)  # type: ignore
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)  # pyright: ignore
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)  # pyright: ignore
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: Optional[int] = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            this_timestep (`int`):
                The current timestep `t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The `p` of UniC-p at this step. The effective order of accuracy should be `order + 1`.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError(" missing`last_sample` as a required keyward argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError(" missing`this_sample` as a required keyward argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing`order` as a required keyward argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated "
                "and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )  # pyright: ignore
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s: Optional[List[Any]] = []
        for i in range(1, order):
            si = self.step_index - (i + 1)  # pyright: ignore
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)  # type: ignore

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:  # type: ignore
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator=None,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep UniPC.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:  # type: ignore
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None  # pyright: ignore
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep  # pyright: ignore

        if self.config.lower_order_final:
            this_order = min(
                self.config.solver_order, len(self.timesteps) - self.step_index
            )  # pyright: ignore
        else:
            this_order = self.config.solver_order

        self.this_order = min(
            this_order, self.lower_order_nums + 1
        )  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1  # pyright: ignore

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def step_ddim(
        # https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
        self,
        velocity: torch.FloatTensor,
        t: int,
        curr_state: torch.FloatTensor,
        prev_state: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep UniPC.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        device = curr_state.device
        curr_t = self.sigmas[t]
        prev_t = self.sigmas[t + 1]
        variance_noise = randn_tensor(
            curr_state.shape, generator=generator, device=device, dtype=curr_state.dtype
        )
        cur_clean_ = curr_state - curr_t * velocity
        prev_state = prev_t * variance_noise + (1 - prev_t) * cur_clean_

        return prev_state

    def step_sde(
        # https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
        self,
        velocity: torch.FloatTensor,
        t: int,
        curr_state: torch.FloatTensor,
        noise_theta: float = 1.0,
        prev_state: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            velocity (`torch.FloatTensor`): (B, C, T, H, W)
                The direct output from learned flow model.
            timestep (`float`): (B, )
                The current discrete timestep in the diffusion chain.
            curr_state (`torch.FloatTensor`): (B, C, T, H, W)
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        device = curr_state.device
        curr_t = self.sigmas[t]
        prev_t = self.sigmas[t + 1]
        cos = torch.cos(torch.tensor(noise_theta) * torch.pi / 2).to(
            device
        )  # degenerates to standard flow matching when noise_theta=0
        sin = torch.sin(torch.tensor(noise_theta) * torch.pi / 2).to(device)
        prev_sample_mean = (1 - prev_t + prev_t * cos) * (
            curr_state - curr_t * velocity
        ) + prev_t * cos * velocity
        std_dev_t = prev_t * sin
        std_dev_t = torch.ones((1, 1)).to(curr_state) * std_dev_t
        if prev_state is None:
            variance_noise = randn_tensor(
                curr_state.shape,
                generator=generator,
                device=device,
                dtype=curr_state.dtype,
            )
            prev_state = prev_sample_mean + std_dev_t * variance_noise
        else:
            prev_state = prev_sample_mean + (prev_state - prev_sample_mean.detach())

        return prev_state

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32
            )
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps


# -----------------------------------------------------------------------------
# Flow UniPC scheduler
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
