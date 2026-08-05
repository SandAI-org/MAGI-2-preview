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

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Optional

import torch
from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer


def strip_empty(obj):
    if isinstance(obj, dict):
        return {
            key: value
            for key, value in ((key, strip_empty(value)) for key, value in obj.items())
            if value is not None and value != [] and value != {}
        }
    if isinstance(obj, list):
        return [strip_empty(value) for value in obj if value is not None]
    return obj


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _pick(values: dict, keys: Iterable[str]) -> list[tuple[str, Any]]:
    return [
        (key, values[key])
        for key in keys
        if key in values and values[key] not in (None, [], {})
    ]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
    if isinstance(raw_json, str):
        obj = json.loads(raw_json.strip())
    else:
        obj = raw_json

    obj = strip_empty(obj)
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return (
            raw_json
            if isinstance(raw_json, str)
            else json.dumps(raw_json, ensure_ascii=False)
        )

    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines: list[str] = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = []
    for label, key in (
        ("style", "style"),
        ("mood", "mood_atmosphere"),
        ("color", "color_scheme"),
    ):
        value = _one_line(aesthetics.get(key))
        if value:
            aesthetic_parts.append(f"{label}={value}")
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            sid = _one_line(subject.get("subject_id"))
            desc = _one_line(subject.get("description"))
            position = _one_line(subject.get("position"))
            orientation = _one_line(subject.get("orientation"))
            attrs = []
            for key, value in _pick(
                _as_dict(subject.get("visual_attributes")),
                ["gender", "age_group", "ethnicity", "clothing", "appearance_details"],
            ):
                value = _one_line(value)
                if value:
                    attrs.append(f"{key}={value}")
            row = " - " + " | ".join([part for part in [sid, desc] if part])
            extra = "; ".join([part for part in [position, orientation] if part])
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for obj_item in objects:
            if not isinstance(obj_item, dict):
                continue
            oid = _one_line(obj_item.get("object_id"))
            desc = _one_line(obj_item.get("description"))
            shape = _one_line(obj_item.get("shape_and_color"))
            position = _one_line(obj_item.get("position"))
            row = " - " + " | ".join([part for part in [oid, desc] if part])
            details = "; ".join([part for part in [shape, position] if part])
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            timestamp = _one_line(basic.get("timestamp_range"))
            desc = _one_line(basic.get("segment_description"))
            head = f" - {timestamp}" if timestamp else " -"
            if desc:
                head += f" {desc}"
            lines.append(head.rstrip())

            segment_audio = _as_dict(segment.get("audio"))
            for dialogue_line in segment_audio.get("dialogue_lines", []) or []:
                if not isinstance(dialogue_line, dict):
                    continue
                speaker = _one_line(dialogue_line.get("speaker"))
                text = _one_line(dialogue_line.get("text"))
                timestamp = _one_line(dialogue_line.get("timestamp"))
                if text:
                    prefix = (
                        f"   - dialogue {speaker}: " if speaker else "   - dialogue: "
                    )
                    line = prefix + text
                    if timestamp:
                        line += f" ({timestamp})"
                    lines.append(line)

            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                sid = _one_line(alive.get("subject_id"))
                action = _as_dict(alive.get("action"))
                parts = [
                    _one_line(action.get("primary_action")),
                    _one_line(action.get("interaction")),
                    _one_line(action.get("facial_expression")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - action {sid}: " + "; ".join(parts))

            for obj_item in segment.get("objects", []) or []:
                if not isinstance(obj_item, dict):
                    continue
                oid = _one_line(obj_item.get("object_id"))
                state = _as_dict(obj_item.get("dynamic_state"))
                parts = [
                    _one_line(state.get("state_change")),
                    _one_line(state.get("motion_detail")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - objects {oid}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())

    return "\n".join(line for line in lines if line.strip())


class Qwen35TextEncoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        precision: torch.dtype = torch.bfloat16,
        max_length: int = 7000,
        skip_layer: int = 0,
    ):
        self.device = device
        self.max_length = max_length
        self.skip_layer = skip_layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

        cjk_split = pre_tokenizers.Split(
            pattern=Regex(
                r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
            ),
            behavior="isolated",
        )
        original_pre_tokenizer = self.tokenizer.backend_tokenizer.pre_tokenizer
        self.tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [cjk_split, original_pre_tokenizer]
        )

        from transformers import Qwen3_5TextModel

        self.text_model = Qwen3_5TextModel.from_pretrained(
            model_path, torch_dtype=precision
        )
        self.to(device)
        self.text_model.eval()
        self.text_model.requires_grad_(False)

    def to(self, device: str | torch.device):
        self.text_model.to(device)
        self.device = str(device)
        return self

    def _normalize_prompt(self, prompt: str) -> str:
        try:
            parsed_json = json.loads(prompt)
            if isinstance(parsed_json, (dict, list)):
                return json_to_compact_markdown(prompt)
        except (json.JSONDecodeError, TypeError):
            pass
        return prompt

    def get_target_token_indices(
        self, prompt: str, target_str: Optional[str]
    ) -> Optional[list[int]]:
        if not target_str:
            return None
        prompt = self._normalize_prompt(prompt)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )
        offsets = inputs["offset_mapping"][0]
        start_char_idx = prompt.find(target_str)
        if start_char_idx == -1:
            return None
        end_char_idx = start_char_idx + len(target_str)

        indices: list[int] = []
        for index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0 and index != 0:
                continue
            if max(start, start_char_idx) < min(end, end_char_idx):
                indices.append(index)
        return indices

    def get_special_token(
        self, prompt: str, target_strs: list[str], text_feature: torch.Tensor
    ) -> torch.Tensor:
        embeddings = []
        for target_str in target_strs:
            indices = self.get_target_token_indices(prompt, target_str)
            if indices:
                embeddings.append(text_feature[0, indices, :].mean(dim=0).clone())
            else:
                embeddings.append(
                    torch.zeros(
                        text_feature.shape[-1],
                        device=text_feature.device,
                        dtype=text_feature.dtype,
                    )
                )
        return torch.stack(embeddings, dim=0)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        prompt = self._normalize_prompt(prompt)
        device = next(self.text_model.parameters()).device
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        ).to(device)
        outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        if self.skip_layer == 0:
            return outputs.last_hidden_state
        return outputs.hidden_states[-(self.skip_layer + 1)]
