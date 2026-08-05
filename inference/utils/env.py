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

import os
from typing import Optional


def env_is_true(env_name: str) -> bool:
    return str(os.environ.get(env_name, "0")).lower() in {"1", "true", "yes", "y", "on", "enabled"}


def env_is_false(env_name: str) -> bool:
    return str(os.environ.get(env_name, "0")).lower() in {"0", "false", "no", "n", "off", "disabled"}


def env_to_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def optional_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def optional_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)
