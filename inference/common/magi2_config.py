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

"""MAGI-2 configuration — self-contained standalone version."""

import json
import math
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.utils import env_is_true


def _env_float(name: str, default: float | str) -> float:
    return float(os.environ.get(name, default))


def _cfg_rescale_default() -> float:
    return float(os.environ.get("CFG_RESCALE", 0.0))


CKPT_ROOT_ENV = "MAGI2_CKPT_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def ckpt_root() -> str:
    """Directory holding downloaded model weights.

    Defaults to ``<repo>/ckpt`` so a fresh clone works once weights are placed
    there; set MAGI2_CKPT_ROOT to point somewhere else.
    """
    return os.environ.get(CKPT_ROOT_ENV) or str(_REPO_ROOT / "ckpt")


def expand_ckpt_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return os.path.expanduser(
        os.path.expandvars(value.replace("${" + CKPT_ROOT_ENV + "}", ckpt_root()))
    )


# --------------------------------------------------------------------------
# Engine configuration
# --------------------------------------------------------------------------


class EngineConfig(BaseModel):
    seed: int = 1234
    load: Optional[str] = None
    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    distributed_timeout_minutes: int = 30
    cp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    cp_strategy: Literal["none", "cp_ulysses", "cp_shuffle_overlap", "dffa"] = "none"

    @field_validator("load", mode="after")
    @classmethod
    def expand_load_path(cls, value: Optional[str]) -> Optional[str]:
        return expand_ckpt_path(value)


# --------------------------------------------------------------------------
# Architecture configs
# --------------------------------------------------------------------------


class AttentionGatingConfig(BaseModel):
    enable: bool = True


class AttentionSinksConfig(BaseModel):
    enable: bool = True
    sink_token_num: int = 1


class MHCConfig(BaseModel):
    enable: bool = True
    num_stream: int = 4
    alpha_init: float = 0.01


class MoEConfig(BaseModel):
    num_experts: int = 256
    top_k: int = 6
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 4.90
    moe_layers: list[int] = Field(default_factory=lambda: list(range(2, 38)))
    expert_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 1280
    modality_specific_expert_intermediate_size: int = 1280
    num_heads: int = 12
    split_merge_intermediate_size: Optional[int] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_query_groups: int = 24
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    params_dtype: torch.dtype | str = torch.bfloat16
    intermediate_factor: int = 4
    mm_layers: list[int] = Field(default_factory=lambda: [0, 1, 38, 39])
    layer_activation_types: dict[int, str] = Field(default_factory=dict)
    activation_type: str = "swiglu7"
    attn_softcap: float = -1.0
    attn_gating: AttentionGatingConfig = Field(default_factory=AttentionGatingConfig)
    attn_sinks: AttentionSinksConfig = Field(default_factory=AttentionSinksConfig)
    mhc_config: MHCConfig = Field(default_factory=MHCConfig)
    moe_config: MoEConfig = Field(default_factory=MoEConfig)

    num_heads_q: int = 0
    num_heads_kv: int = 0

    @field_serializer("params_dtype")
    def serialize_dtype(self, value: torch.dtype | str) -> str:
        return str(value)

    @field_validator("params_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        if isinstance(value, torch.dtype):
            return value
        if value in ("torch.float32", "float32"):
            return torch.float32
        if value in ("torch.float16", "float16"):
            return torch.float16
        if value in ("torch.bfloat16", "bfloat16"):
            return torch.bfloat16
        raise ValueError(f"Unknown torch.dtype string: {value!r}")


# --------------------------------------------------------------------------
# Data proxy configs
# --------------------------------------------------------------------------


class DataProxyConfig(BaseModel):
    t_patch_size: int = 1
    patch_size: int = 1
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    add_time_token: bool = False
    time_channel_dim: int = 64
    time_aligned_rope: bool = False
    audio_latent_fps: float = 25.0
    time_pos_fps: float = 3.125
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0


class Magi2RefinerDataProxyConfig(BaseModel):
    """magi2_refiner data proxy configuration."""

    t_patch_size: int = 1
    patch_size: int = 2
    frame_receptive_field: int = 11
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    text_offset: int = 0
    coords_style: Literal["v1", "v2"] = "v1"
    attn_config: dict = Field(default_factory=dict)
    magi2_refiner_condition_input: Literal["none", "concat_x1"] = "none"


class Magi2RefinerModelConfig(BaseModel):
    """magi2_refiner architecture config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    num_layers: int = 30
    hidden_size: int = 4096
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    checkpoint_qk_layernorm_rope: bool = False
    params_dtype: torch.dtype | str = torch.bfloat16
    mm_layers: list[int] = Field(default_factory=lambda: [0, 1, 28, 29])
    local_attn_layers: list[int] = Field(default_factory=lambda: list(range(30)))
    enable_attn_gating: bool = True
    activation_type: str = "swiglu7"
    layer_activation_types: dict = Field(default_factory=dict)
    post_norm_layers: list[int] = Field(default_factory=list)

    num_heads_q: int = 0
    num_heads_kv: int = 0

    @field_serializer("params_dtype")
    def serialize_dtype(self, value: torch.dtype | str) -> str:
        return str(value)

    @field_validator("params_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        if isinstance(value, torch.dtype):
            return value
        if value in ("torch.float32", "float32"):
            return torch.float32
        if value in ("torch.float16", "float16"):
            return torch.float16
        if value in ("torch.bfloat16", "bfloat16"):
            return torch.bfloat16
        raise ValueError(f"Unknown torch.dtype string: {value!r}")


# --------------------------------------------------------------------------
# Evaluation config
# --------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    data_proxy_config: DataProxyConfig = Field(default_factory=DataProxyConfig)
    fps: Union[int, float] = 12.5
    shift: float = 5.0
    vae_model_path: str = "${MAGI2_CKPT_ROOT}/vae"
    txt_model_path: str = "${MAGI2_CKPT_ROOT}/text_encoder"
    audio_model_path: str = Field(default="${MAGI2_CKPT_ROOT}/stable-audio-open-1.0", validate_default=True)
    vae_stride: Tuple[int, int, int] = (8, 16, 16)
    patch_size: Tuple[int, int, int] = (1, 1, 1)
    z_dim: int = 48
    txt_encoder_type: Literal["qwen35"] = "qwen35"
    video_txt_guidance_scale: float = Field(
        default_factory=lambda: float(os.environ.get("VG", 5.0))
    )
    audio_txt_guidance_scale: float = Field(
        default_factory=lambda: float(os.environ.get("AG", 5.0))
    )
    use_cfg_trick: bool = True
    cfg_trick_start_frame: int = 13
    cfg_trick_value: float = 2.0
    use_dynamic_cfg: bool = False
    dynamic_cfg_start_t: int = 500
    dynamic_cfg_cutoff_value: float = 2.0
    use_negative_prompt: bool = False
    use_ref_for_uncond: bool = False
    use_skimmed_cfg_linear: bool = Field(
        default_factory=lambda: env_is_true("USE_SKIMMED_CFG_LINEAR")
    )
    skimmed_cfg_scale: float = Field(
        default_factory=lambda: _env_float("SKIMMED_CFG_SCALE", 5.0)
    )
    cfg_rescale: float = Field(default_factory=_cfg_rescale_default)
    ref_image_type: Literal["resize", "short_edge_resize_crop"] = "short_edge_resize_crop"

    @field_validator("skimmed_cfg_scale", "cfg_rescale")
    @classmethod
    def validate_nonnegative_finite(cls, value: float, info):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{info.field_name} must be a non-negative finite number, got {value}"
            )
        return value

    @field_validator(
        "vae_model_path",
        "txt_model_path",
        "student_config",
        "student_ckpt",
        "audio_model_path",
        "magi2_refiner_model_path",
        mode="after",
    )
    @classmethod
    def expand_checkpoint_paths(cls, value: str) -> str:
        return expand_ckpt_path(value)


    use_turbo_vae: bool = True
    student_config: str = (
        "${MAGI2_CKPT_ROOT}/turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json"
    )
    student_ckpt: str = "${MAGI2_CKPT_ROOT}/turbo_vae/checkpoint.ckpt"
    num_inference_steps: int = 30
    magi2_refiner_num_inference_steps: int = 5

    use_magi2_refiner: bool = False
    magi2_refiner_model_name: Literal["magi2_refiner"] = "magi2_refiner"
    magi2_refiner_model_path: str = "${MAGI2_CKPT_ROOT}/refiner"
    magi2_refiner_video_txt_guidance_scale: float = 2.0
    magi2_refiner_audio_txt_guidance_scale: float = 5.0
    magi2_refiner_noise_value: int = 220
    magi2_refiner_vae_stride: Tuple[int, int, int] = (4, 16, 16)
    magi2_refiner_patch_size: Tuple[int, int, int] = (1, 1, 1)
    magi2_refiner_shift: Optional[float] = None
    magi2_refiner_audio_noise_scale: float = -1.0
    magi2_refiner_data_proxy_config: Magi2RefinerDataProxyConfig = Field(
        default_factory=lambda: Magi2RefinerDataProxyConfig(
            frame_receptive_field=11,
            t_patch_size=1,
            patch_size=1,
            spatial_rope_interpolation="extra",
            coords_style="v1",
            text_offset=0,
        )
    )


# --------------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------------


class Magi2Config(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_ignore_unknown_args=True, cli_implicit_flags=True
    )

    engine_config: EngineConfig = Field(default_factory=EngineConfig)
    arch_config: ModelConfig = Field(default_factory=ModelConfig)
    magi2_refiner_arch_config: Optional[Magi2RefinerModelConfig] = None
    evaluation_config: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def post_override_config(self):
        self.arch_config.num_heads_q = (
            self.arch_config.hidden_size // self.arch_config.head_dim
        )
        self.arch_config.num_heads_kv = self.arch_config.num_query_groups
        if self.magi2_refiner_arch_config is None:
            if self.evaluation_config.use_magi2_refiner:
                raise ValueError(
                    "magi2_refiner_arch_config must be provided in config when use_magi2_refiner is true"
                )
            return self
        self.magi2_refiner_arch_config.num_heads_q = (
            self.magi2_refiner_arch_config.hidden_size // self.magi2_refiner_arch_config.head_dim
        )
        self.magi2_refiner_arch_config.num_heads_kv = self.magi2_refiner_arch_config.num_query_groups
        return self


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Overlay wins, except that nested objects merge key by key."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_json(config_path: Path) -> dict:
    with config_path.open() as f:
        data = json.load(f)
    base = data.pop("extends", None)
    if base is None:
        return data
    return _deep_merge(_read_config_json(config_path.parent / base), data)


def load_config(config_path: str) -> Magi2Config:
    """Load a Magi2Config from a JSON file.

    A config may name another under "extends" and then carry only what it adds
    or changes. The name is relative to the file doing the extending.
    """
    return Magi2Config(**_read_config_json(Path(config_path)))
