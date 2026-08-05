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

"""Roundtrip CPU<->GPU offload utilities.

Store a persistent CPU weight backup directly on the module so that offloading
restores param.data pointers (O(1) swap) instead of copying GPU->CPU (O(n)).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

import torch

from .module_state_backup import ModuleStateBackup, ModuleStateUtils


class OffloadMode(str, Enum):
    """Component offload strategy: where model weights reside between compute phases."""

    CPU = "cpu"
    GPU = "gpu"
    ROUNDTRIP = "roundtrip"

    def resolve_device(self, gpu_device: str | torch.device) -> str:
        """Return *gpu_device* for GPU mode, "cpu" otherwise."""
        return str(gpu_device) if self == OffloadMode.GPU else "cpu"

    @classmethod
    def resolve_from_env(cls, component: str, default: OffloadMode) -> OffloadMode:
        """Read a component's mode from MAGI2_<COMPONENT>_OFFLOAD_MODE.

        Components are named after the stage they belong to, so magi2_preview is
        configured by MAGI2_PREVIEW_OFFLOAD_MODE rather than by a doubled prefix.
        """
        env_key = f"MAGI2_{component.upper().removeprefix('MAGI2_')}_OFFLOAD_MODE"
        raw = os.environ.get(env_key, "").strip().lower()
        if not raw:
            return default
        try:
            return cls(raw)
        except ValueError:
            raise ValueError(f"{env_key}={raw!r} is not a valid offload mode, expected one of {[m.value for m in cls]}")


_BACKUP_ATTR = "_roundtrip_cpu_backup"


class OffloadUtils:
    """Stateless utilities for CPU<->GPU roundtrip offload with persistent CPU backup."""

    @staticmethod
    def setup(model: Any, mode: OffloadMode, device: str | torch.device) -> None:
        """Place model on the correct device and create CPU backup if roundtrip."""
        model.to(mode.resolve_device(device))
        if mode == OffloadMode.ROUNDTRIP:
            setattr(model, _BACKUP_ATTR, ModuleStateUtils.backup(model))

    @staticmethod
    def maybe_load(model: Any, mode: OffloadMode, device: str | torch.device) -> None:
        """Load model only when configured for roundtrip offload."""
        if mode == OffloadMode.ROUNDTRIP:
            OffloadUtils.load(model, device)

    @staticmethod
    def maybe_offload(model: Any, mode: OffloadMode) -> None:
        """Offload model only when configured for roundtrip offload."""
        if mode == OffloadMode.ROUNDTRIP:
            OffloadUtils.offload(model)

    @staticmethod
    def load(model: Any, device: str | torch.device) -> None:
        """Load a model configured for roundtrip offload to GPU."""
        if not hasattr(model, _BACKUP_ATTR):
            raise RuntimeError("OffloadUtils.setup() must be called with roundtrip mode before load()")
        model.to(device)

    @staticmethod
    def offload(model: Any) -> None:
        """Restore CPU tensor pointers so GPU weights become unreferenced and GC-able."""
        backup: ModuleStateBackup | None = getattr(model, _BACKUP_ATTR, None)
        if backup is None:
            model.to("cpu")
            return
        ModuleStateUtils.restore(model, backup)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
