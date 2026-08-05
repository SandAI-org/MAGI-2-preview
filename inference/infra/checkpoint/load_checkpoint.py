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

"""Checkpoint loading for MAGI-2 models."""

import gc

import torch

from inference.common.magi2_config import Magi2Config
from inference.infra.checkpoint.magi2_checkpointing import (
    load_magi2_model_state_dict,
    load_safetensors_dir,
)
from inference.utils import env_is_true, print_rank_0


def load_magi2_model(config: Magi2Config) -> torch.nn.Module:
    """Load the MAGI-2 DiT model from checkpoint."""
    from inference.model.magi2_preview import Transformer

    arch = config.arch_config
    print_rank_0(
        f"[magi2] Building model: layers={arch.num_layers}, "
        f"hidden={arch.hidden_size}, MoE experts={arch.moe_config.num_experts}"
    )

    model = Transformer(arch, ep_size=config.engine_config.ep_size)
    print_rank_0(f"[magi2] Model built, params_dtype={arch.params_dtype}")

    if env_is_true("SKIP_LOAD_MODEL"):
        # Transformer.__init__ already filled in synthetic weights. Shapes,
        # compilation and memory footprint stay real; only the read is skipped.
        print_rank_0("[magi2] SKIP_LOAD_MODEL=1: synthetic weights, no checkpoint read")
    else:
        state_dict = load_magi2_model_state_dict(model, config.engine_config)
        model.load_state_dict(state_dict, strict=True)
        print_rank_0("[magi2] Checkpoint loaded successfully")

        del state_dict
        gc.collect()

    model = model.to(f"cuda:{torch.cuda.current_device()}")
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_magi2_refiner(config: Magi2Config) -> torch.nn.Module:
    """Load the super-resolution model."""
    from inference.model.magi2_refiner import Transformer as Magi2RefinerModel

    refiner_config = config.magi2_refiner_arch_config
    print_rank_0(
        f"[magi2] Building magi2_refiner: layers={refiner_config.num_layers}, "
        f"hidden={refiner_config.hidden_size}"
    )

    model = Magi2RefinerModel(refiner_config)
    if env_is_true("SKIP_LOAD_MODEL"):
        print_rank_0("[magi2] SKIP_LOAD_MODEL=1: synthetic refiner weights, no checkpoint read")
        model.eval()
        return model

    model_path = config.evaluation_config.magi2_refiner_model_path
    print_rank_0(f"[magi2] Loading magi2_refiner checkpoint from {model_path}")

    state_dict = load_safetensors_dir(model_path, desc="Loading magi2_refiner shards")

    model.load_state_dict(state_dict, strict=True)
    print_rank_0("[magi2] magi2_refiner checkpoint loaded successfully")

    del state_dict
    gc.collect()
    model.eval()
    return model
