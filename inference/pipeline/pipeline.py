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

"""MAGI-2 Pipeline: orchestrates model loading and inference."""

import os
import subprocess
from typing import Optional

import torch

from inference.common.magi2_config import Magi2Config
from inference.utils import print_rank_0


class Magi2Pipeline:
    """High-level pipeline for MAGI-2 inference."""

    def __init__(self, config: Magi2Config):
        self.config = config
        self.device = f"cuda:{torch.cuda.current_device()}"

        self._load_models()

    def _load_models(self):
        """Load DiT model and initialize evaluator."""
        from inference.infra.checkpoint.load_checkpoint import load_magi2_model

        print_rank_0("[magi2] Loading preview model...")
        model = load_magi2_model(self.config)
        self.model = model

        magi2_refiner = None
        if self.config.evaluation_config.use_magi2_refiner:
            print_rank_0("[magi2] Loading magi2_refiner model...")
            from inference.infra.checkpoint.load_checkpoint import load_magi2_refiner

            magi2_refiner = load_magi2_refiner(self.config)

        self.evaluator = Magi2Evaluator(
            model=model,
            config=self.config.evaluation_config,
            device=self.device,
            magi2_refiner=magi2_refiner,
        )

    def run_offline(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        save_path_prefix: str = "output",
        seconds: float = 5.0,
        preview_width: int = 448,
        preview_height: int = 256,
        refiner_width: Optional[int] = None,
        refiner_height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        magi2_refiner_num_inference_steps: Optional[int] = None,
    ):
        """Run a single offline inference job."""
        self.evaluator.run(
            prompt=prompt,
            image_path=image_path,
            save_path_prefix=save_path_prefix,
            seconds=seconds,
            preview_width=preview_width,
            preview_height=preview_height,
            refiner_width=refiner_width,
            refiner_height=refiner_height,
            output_width=output_width,
            output_height=output_height,
            num_inference_steps=num_inference_steps,
            magi2_refiner_num_inference_steps=magi2_refiner_num_inference_steps,
        )


# -----------------------------------------------------------------------------
# Evaluator facade
# -----------------------------------------------------------------------------

"""MAGI-2 Evaluator — wraps Magi2ActEvaluator for standalone offline inference."""

from typing import Optional

import numpy as np

from inference.common.magi2_config import EvaluationConfig
from inference.infra.distributed import psm

from .act_evaluator import Magi2ActEvaluator


class Magi2Evaluator:
    """Evaluator that runs the full MAGI-2 inference pipeline on each rank."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: EvaluationConfig,
        device: str = "cuda",
        magi2_refiner: Optional[torch.nn.Module] = None,
    ):
        self.act_evaluator = Magi2ActEvaluator(
            model=model,
            config=config,
            device=device,
            weight_dtype=torch.bfloat16,
            magi2_refiner=magi2_refiner,
        )
        self.config = config
        self.device = device
        print_rank_0("[magi2] Evaluator initialized (real weights)")

    def run(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        save_path_prefix: str = "output",
        seconds: float = 5.0,
        preview_width: int = 448,
        preview_height: int = 256,
        refiner_width: Optional[int] = None,
        refiner_height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        magi2_refiner_num_inference_steps: Optional[int] = None,
    ):
        """Run full inference: text encode -> denoise -> VAE decode -> save."""
        eval_task_type = "text2video"
        if image_path is not None:
            eval_task_type = "image2video"

        steps = num_inference_steps or self.config.num_inference_steps or 30
        refiner_steps = magi2_refiner_num_inference_steps or self.config.magi2_refiner_num_inference_steps

        video_np, audio_np = self.act_evaluator.evaluate(
            prompt=prompt,
            image=image_path,
            eval_task_type=eval_task_type,
            seconds=seconds,
            preview_width=preview_width,
            preview_height=preview_height,
            br_num_inference_steps=steps,
            refiner_width=refiner_width,
            refiner_height=refiner_height,
            magi2_refiner_num_inference_steps=refiner_steps if refiner_width else None,
        )

        rank = (
            psm.get_global_rank()
            if psm.is_initialized()
            else int(os.environ.get("RANK", "0"))
        )
        if video_np is not None:
            if output_width is not None and output_height is not None:
                video_np = self._resize_video(video_np, output_width, output_height)
            output_fps = float(self.config.fps) * (2 if refiner_width else 1)
            self._save_video(video_np, save_path_prefix, seconds, output_fps=output_fps)
            if audio_np is not None:
                self._mux_audio(save_path_prefix, audio_np, self.act_evaluator.audio_vae.sample_rate)
            print_rank_0(f"[magi2] Video saved: {save_path_prefix}.mp4")
        else:
            print_rank_0(f"[magi2] Rank {rank}: not leader, skip save")

    @staticmethod
    def _resize_video(
        video_np: np.ndarray, width: int, height: int, mode: str = "bilinear"
    ) -> np.ndarray:
        """Scale every frame to (height, width)."""
        frames = torch.from_numpy(video_np)
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        # interpolate resizes the trailing two dims only, so moving colour to the
        # front keeps this a per-frame 2D resize instead of one across time.
        frames = frames.permute(3, 0, 1, 2)
        frames = torch.nn.functional.interpolate(
            frames,
            size=(height, width),
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
        )
        return (frames.permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().numpy()

    @staticmethod
    def _save_video(
        video_np: np.ndarray,
        save_path_prefix: str,
        seconds: float,
        output_fps: Optional[float] = None,
    ):
        import imageio

        os.makedirs(os.path.dirname(save_path_prefix) or ".", exist_ok=True)
        video_path = f"{save_path_prefix}.mp4"

        if video_np.ndim != 4:
            raise ValueError(
                f"expected video array of shape (T, H, W, C), got {video_np.shape}"
            )

        fps = output_fps if output_fps else (video_np.shape[0] / seconds if video_np.shape[0] > 1 else 24)
        imageio.mimwrite(video_path, video_np, fps=fps, quality=8, output_params=["-loglevel", "error"])

    @staticmethod
    def _mux_audio(save_path_prefix: str, audio_np, sample_rate: int):
        import soundfile as sf

        video_path = f"{save_path_prefix}.mp4"
        audio_path = f"{save_path_prefix}_audio.wav"
        sf.write(audio_path, audio_np, sample_rate)
        muxed_path = f"{save_path_prefix}_av.mp4"
        cmd = [
            "ffmpeg", "-i", video_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac",
            "-y", muxed_path, "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True)
            os.replace(muxed_path, video_path)
            os.remove(audio_path)
        except subprocess.CalledProcessError as e:
            print_rank_0(f"[magi2] ffmpeg audio mux failed: {e}")
