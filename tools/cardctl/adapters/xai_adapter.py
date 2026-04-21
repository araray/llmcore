# tools/cardctl/adapters/xai_adapter.py
"""xAI (Grok) model discovery adapter (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter


class XAIAdapter(OpenAICompatAdapter):
    provider_name = "xai"
    api_key_env_var = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("grok")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        normalized.supports_tools = True
        normalized.supports_vision = True
        normalized.supports_json_mode = True
        normalized.architecture_family = "grok"
        normalized.architecture_type = "transformer"
        return normalized
