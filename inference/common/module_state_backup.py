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

"""Backup/restore module parameter storage for CPU<->GPU roundtrip offload."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

ModuleStateBackup = Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]


class ModuleStateUtils:
    """Stateless utilities for backing up and restoring nn.Module parameter/buffer tensors."""

    @staticmethod
    def backup(model: Any) -> ModuleStateBackup:
        """Capture references to current parameter/buffer tensors (typically on CPU)."""
        module_param_backup: Dict[str, torch.Tensor] = {}
        module_buffer_backup: Dict[str, torch.Tensor] = {}
        other_backup: Dict[str, Any] = {}

        def _save(mod: torch.nn.Module, prefix: str) -> None:
            for name, param in mod.named_parameters():
                if param is not None:
                    module_param_backup[prefix + name] = param.data
            for name, buffer in mod.named_buffers():
                if buffer is not None:
                    module_buffer_backup[prefix + name] = buffer.data

        if isinstance(model, torch.nn.Module):
            _save(model, "")
        else:
            for name, attr_val in model.__dict__.items():
                if isinstance(attr_val, torch.nn.Module):
                    _save(attr_val, name + ".")
                elif isinstance(attr_val, torch.Tensor):
                    other_backup[name] = attr_val

        return module_param_backup, module_buffer_backup, other_backup

    @staticmethod
    def restore(model: Any, backup: ModuleStateBackup) -> None:
        """Restore parameter/buffer tensor references from a prior backup."""
        module_param_backup, module_buffer_backup, other_backup = backup

        def _restore(mod: torch.nn.Module, prefix: str) -> None:
            for name, param in mod.named_parameters():
                full_key = prefix + name
                if full_key in module_param_backup:
                    param.data = module_param_backup[full_key]
            for name, buffer in mod.named_buffers():
                full_key = prefix + name
                if full_key in module_buffer_backup:
                    buffer.data = module_buffer_backup[full_key]

        if isinstance(model, torch.nn.Module):
            _restore(model, "")
        else:
            for name, attr_val in model.__dict__.items():
                if isinstance(attr_val, torch.nn.Module):
                    _restore(attr_val, name + ".")
            for name, val in other_backup.items():
                setattr(model, name, val)
