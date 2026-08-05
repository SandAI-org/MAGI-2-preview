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

import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.signal import resample

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SA_RESCALE_MEAN = [0.0]
SA_RESCALE_STD = [1.0]


def resample_audio_sinc(audio: np.ndarray, time_stretching: float):
    new_length = int(audio.shape[0] * time_stretching)
    return resample(audio, new_length)


def adaptive_volume_scaling(
    audio_samples: np.ndarray, target_rms_db: float = -31, max_gain_db: float = 0
):
    rms = np.sqrt(np.mean(audio_samples**2))
    if rms < 1e-10:
        return audio_samples
    peak = np.max(np.abs(audio_samples))
    current_rms_db = 20 * np.log10(rms + 1e-10)
    desired_gain_db = target_rms_db - current_rms_db
    actual_gain_db = min(desired_gain_db, max_gain_db)
    scaled = audio_samples * (10 ** (actual_gain_db / 20))
    if np.max(np.abs(scaled)) > 1.0:
        scaled = scaled / np.max(np.abs(scaled))
    return scaled


class SAAudioFeatureExtractor:
    """Stable Audio Open VAE for encoding/decoding audio latents."""

    def __init__(self, model_path: str):

        self.device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.vae_model, self.sample_rate, self.downsampling_ratio = self._get_vae_only(
            model_path
        )
        self.rescale_mean = torch.tensor(
            SA_RESCALE_MEAN, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)
        self.rescale_std = torch.tensor(
            SA_RESCALE_STD, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)

    @property
    def latent_fps(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    def _get_vae_only(self, model_path):
        """Load only the VAE/pretransform part of Stable Audio Open."""
        from inference.model.sa_audio_vae import create_model_from_config

        model_config_path = os.path.join(model_path, "model_config.json")
        with open(model_config_path) as f:
            full_config = json.load(f)

        vae_config = full_config["model"]["pretransform"]["config"]
        sample_rate = full_config["sample_rate"]
        downsampling_ratio = vae_config["downsampling_ratio"]

        autoencoder_config = {
            "model_type": "autoencoder",
            "sample_rate": sample_rate,
            "model": vae_config,
        }

        vae_model = create_model_from_config(autoencoder_config)
        weights_path = Path(model_path) / "model.safetensors"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weight file does not exist: {weights_path}")

        full_state_dict = load_file(weights_path, device=self.device)

        vae_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith("pretransform.model."):
                vae_key = key[len("pretransform.model."):]
                vae_state_dict[vae_key] = value

        vae_model.load_state_dict(vae_state_dict)
        vae_model.to(self.device).eval()

        print(f"[magi2] Stable Audio VAE loaded (sr={sample_rate}, ds_ratio={downsampling_ratio})")
        return vae_model, sample_rate, downsampling_ratio

    def decode(self, latents):
        latents = latents * self.rescale_std + self.rescale_mean
        with torch.no_grad():
            return self.vae_model.decode(latents)

    def encode(self, waveform):
        with torch.no_grad():
            latents = self.vae_model.encode(waveform)
        return (latents - self.rescale_mean) / self.rescale_std
