# tools/cardctl/adapters/deepseek_adapter.py
"""DeepSeek model discovery adapter (OpenAI-compatible).

Fetches models from the DeepSeek ``/models`` endpoint and enriches them
with V4-specific metadata (MoE architecture, thinking mode, context length).

DeepSeek V4 models (2026-04):
- ``deepseek-v4-flash`` — 284B total / 13B active, 1M context
- ``deepseek-v4-pro`` — 1.6T total / 49B active, 1M context

Legacy aliases (retiring 2026-07-24):
- ``deepseek-chat`` → deepseek-v4-flash (non-thinking)
- ``deepseek-reasoner`` → deepseek-v4-flash (thinking)
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

# V4 model metadata keyed by model_id.
_V4_MODELS: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "context_length": 1_000_000,
        "parameter_count": "284B",
        "active_parameters": "13B",
        "architecture_type": "moe",
    },
    "deepseek-v4-pro": {
        "context_length": 1_000_000,
        "parameter_count": "1.6T",
        "active_parameters": "49B",
        "architecture_type": "moe",
    },
}


class DeepSeekAdapter(OpenAICompatAdapter):
    provider_name = "deepseek"
    api_key_env_var = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("deepseek")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        mid = normalized.model_id

        # All DeepSeek models support these
        normalized.supports_tools = True
        normalized.supports_json_mode = True
        normalized.architecture_family = "deepseek"

        # V4 model enrichments
        v4_meta = _V4_MODELS.get(mid)
        if v4_meta:
            normalized.context_length = v4_meta["context_length"]
            normalized.parameter_count = v4_meta["parameter_count"]
            normalized.active_parameters = v4_meta.get("active_parameters")
            normalized.architecture_type = v4_meta["architecture_type"]
            normalized.supports_reasoning = True
            normalized.architecture_family = "deepseek-v4"
            if "flash" in mid:
                normalized.tags.extend(["v4", "flash", "cost-effective", "million-context"])
            elif "pro" in mid:
                normalized.tags.extend(["v4", "pro", "flagship", "million-context"])
        elif "reasoner" in mid or "r1" in mid:
            normalized.supports_reasoning = True
            normalized.architecture_type = "moe"
            normalized.tags.extend(["reasoning", "legacy"])
        elif "chat" in mid:
            normalized.architecture_type = "moe"
            normalized.tags.append("legacy")
        else:
            normalized.architecture_type = "moe"

        return normalized
