"""Prompt enhancement (PE) for MAGI-2 inference.

Rewrites a short user prompt into the dense structured caption the video model
expects. T2V and I2V use different caption templates / JSON schemas; both render
the model JSON to readable Markdown via ``pe_markdown``.

LLM transport is intentionally thin: subclass :class:`LLMClient` (or replace
the default example client) to call whatever provider you use. The example
client below is only a reference for OpenAI-compatible endpoints.
"""

from __future__ import annotations

import base64
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .pe_markdown import render_markdown
from .pe_schemas import I2V_RESPONSE_SCHEMA, T2V_RESPONSE_SCHEMA

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Example defaults used by :class:`OpenAICompatibleClient`. Leave ``API_KEY``
# empty to skip PE. Replace with your own credentials / endpoint as needed.
DEFAULT_MODEL = "gemini-3.1-pro-preview"
API_KEY = ""
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_OUTPUT_TOKENS = 16384


@dataclass(frozen=True)
class TaskSpec:
    """Per-task PE configuration (template + response schema)."""

    name: str  # "t2v" | "i2v"
    template_path: Path
    response_schema_name: str
    response_schema: dict[str, Any]
    required_top_level_keys: tuple[str, ...]


T2V_TASK = TaskSpec(
    name="t2v",
    template_path=PROMPT_DIR / "t2v.md",
    response_schema_name="T2VEnhanceResult",
    response_schema=T2V_RESPONSE_SCHEMA,
    required_top_level_keys=("global_layer", "dynamic_layer"),
)

I2V_TASK = TaskSpec(
    name="i2v",
    template_path=PROMPT_DIR / "i2v.md",
    response_schema_name="I2VEnhanceResult",
    response_schema=I2V_RESPONSE_SCHEMA,
    required_top_level_keys=(
        "global_caption",
        "reference_bank",
        "shot_timeline",
        "audio_event_timeline",
        "visible_text",
        "generation_requirements",
    ),
)


def to_image_url(image: str | Image.Image) -> str:
    """Turn a PIL image or local path into a data URL; pass http(s)/data URLs through."""
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return image
        path = Path(image).expanduser()
        mime = {
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")
        data = path.read_bytes()
    else:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        mime, data = "image/jpeg", buffer.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class LLMClient(ABC):
    """LLM transport used by :class:`PromptEnhancer`.

    Implement :meth:`chat_completion` for your provider. ``messages`` follow the
    OpenAI chat format (``role`` + ``content``); ``kwargs`` may include
    ``model``, ``max_tokens``, and ``response_format``.
    """

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible example client.

    This is a reference implementation only — swap it for Vertex, Azure, or any
    other backend by providing your own :class:`LLMClient`.
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 180.0,
    ):
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(messages=messages, **kwargs)
        return response.choices[0].message.content or ""


class PromptEnhancer:
    """Run the T2V / I2V PE pipeline against an :class:`LLMClient`."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ):
        self.client = client or OpenAICompatibleClient()
        self.model = model
        self.max_output_tokens = max_output_tokens

    def enhance(self, prompt: str, image: Optional[str | Image.Image] = None) -> str:
        task = I2V_TASK if image is not None else T2V_TASK
        pe_prompt = (
            task.template_path.read_text(encoding="utf-8")
            .replace("{user_input_text}", prompt)
            .replace("{variation_seed}", "")
        )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": pe_prompt}]
        if image is not None:
            user_content.append(
                {"type": "image_url", "image_url": {"url": to_image_url(image)}}
            )

        result_text = self.client.chat_completion(
            messages=[{"role": "user", "content": user_content}],
            model=self.model,
            max_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": task.response_schema_name,
                    "schema": task.response_schema,
                    "strict": True,
                },
            },
        )

        try:
            pe_result = json.loads(result_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse {task.name} PE JSON: {exc}, response: {result_text}"
            ) from exc
        if not isinstance(pe_result, dict):
            raise ValueError(f"{task.name} PE response must be a JSON object")
        missing = [key for key in task.required_top_level_keys if key not in pe_result]
        if missing:
            raise ValueError(
                f"{task.name} PE response missing required fields {missing}: {result_text}"
            )

        return render_markdown(
            data=pe_result,
            drop_empty=True,
            keep_empty_string=False,
            omit_section_leaf_keys=False,
            quote_strings=False,
            collapse_string_whitespace=False,
        )


_default_enhancer: Optional[PromptEnhancer] = None


def enhance_prompt(prompt: str, image: Optional[str | Image.Image]) -> str:
    """Enhance a prompt; I2V when ``image`` is set, otherwise T2V.

    Skips PE and returns ``prompt`` unchanged when ``API_KEY`` is not set.
    """
    if not API_KEY:
        return prompt
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = PromptEnhancer()
    return _default_enhancer.enhance(prompt, image)
