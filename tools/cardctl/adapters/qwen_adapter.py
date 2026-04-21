# tools/cardctl/adapters/qwen_adapter.py
"""Qwen model discovery adapter (DashScope, OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter


class QwenAdapter(OpenAICompatAdapter):
    provider_name = "qwen"
    api_key_env_var = "DASHSCOPE_API_KEY"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("qwen")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        normalized.architecture_family = "qwen"
        normalized.supports_tools = True
        normalized.supports_json_mode = True

        mid = normalized.model_id
        if "vl" in mid:
            normalized.supports_vision = True
        if "coder" in mid:
            normalized.tags.append("code")
        return normalized
