# tools/cardctl/adapters/deepseek_adapter.py
"""DeepSeek model discovery adapter (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter


class DeepSeekAdapter(OpenAICompatAdapter):
    provider_name = "deepseek"
    api_key_env_var = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("deepseek")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        normalized.supports_tools = True
        normalized.supports_json_mode = True
        normalized.architecture_family = "deepseek"

        mid = normalized.model_id
        if "reasoner" in mid:
            normalized.supports_reasoning = True
            normalized.architecture_type = "moe"
            normalized.tags.append("reasoning")
        elif "chat" in mid:
            normalized.architecture_type = "moe"
        return normalized
