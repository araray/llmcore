# tools/cardctl/adapters/moonshot_adapter.py
"""Moonshot (Kimi) model discovery adapter (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter


class MoonshotAdapter(OpenAICompatAdapter):
    provider_name = "moonshot"
    api_key_env_var = "MOONSHOT_API_KEY"
    base_url = "https://api.moonshot.cn/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("moonshot") or mid.startswith("kimi")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        normalized.architecture_family = "moonshot"
        normalized.architecture_type = "transformer"
        normalized.supports_tools = True
        return normalized
