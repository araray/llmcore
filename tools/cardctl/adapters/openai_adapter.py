# tools/cardctl/adapters/openai_adapter.py
"""OpenAI model discovery adapter."""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

# Prefixes for models to exclude from card generation.
_EXCLUDE_PREFIXES = (
    "ft:",          # fine-tuned models
    "dall-e",       # image generation
    "whisper",      # speech-to-text (handled separately)
    "tts",          # text-to-speech (handled separately)
    "babbage",      # legacy completions
    "davinci",      # legacy completions
    "gpt-3.5-turbo-instruct",  # legacy instruct
    "chatgpt-4o-latest",       # ephemeral alias
)

_REASONING_PREFIXES = ("o1", "o3", "o4")


class OpenAIAdapter(OpenAICompatAdapter):
    provider_name = "openai"
    api_key_env_var = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return not any(mid.startswith(p) for p in _EXCLUDE_PREFIXES)

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        mid = normalized.model_id

        # Reasoning models
        if any(mid.startswith(p) for p in _REASONING_PREFIXES):
            normalized.supports_reasoning = True
            normalized.tags.append("reasoning")

        # Most GPT-4+ models support tools and vision
        if "gpt-4" in mid or mid.startswith("gpt-4") or mid.startswith("o"):
            normalized.supports_tools = True
            normalized.supports_json_mode = True
            normalized.supports_structured_output = True
        if "gpt-4o" in mid or "gpt-4.1" in mid or mid.startswith("o"):
            normalized.supports_vision = True

        normalized.architecture_family = "gpt"
        normalized.architecture_type = "transformer"
        return normalized
