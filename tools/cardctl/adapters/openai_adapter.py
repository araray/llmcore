# tools/cardctl/adapters/openai_adapter.py
"""OpenAI model discovery adapter."""

from __future__ import annotations

import re
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

#: Not chat models; they get no chat capabilities.
_NON_CHAT = re.compile(r"(tts|transcribe|whisper|image|embedding|moderation|realtime|sora|dall-e)")


def infer_capabilities(model_id: str) -> dict[str, bool]:
    """What an OpenAI chat model id can do, by family.

    OpenAI's /models states none of this. The rules: the o-series and every
    GPT-5+ model reason (chat-tuned and search variants do not); every GPT-4+
    and o-series chat model takes tools and structured output; GPT-4o, 4.1,
    GPT-5+ and the o-series take images. The old version knew only o1/o3/o4 as
    reasoning models and "gpt-4" substrings for the rest, which is how the
    GPT-5.5/5.6/6.x cards came out claiming no tools, vision or reasoning.
    """
    if _NON_CHAT.search(model_id):
        return {}
    chat_family = bool(re.match(r"(gpt-[4-9]|o\d|chatgpt|codex)", model_id))
    reasoning = bool(re.match(r"(o\d|gpt-[5-9])", model_id)) and not re.search(
        r"(chat-latest|search-api|search-preview|audio)", model_id
    )
    vision = bool(re.match(r"(gpt-4o|gpt-4\.1|gpt-[5-9]|o\d|chatgpt)", model_id)) and not re.search(
        r"(search-api|audio)", model_id
    )
    return {
        "supports_reasoning": reasoning,
        "supports_tools": chat_family,
        "supports_json_mode": chat_family,
        "supports_structured_output": chat_family,
        "supports_vision": vision,
    }


def known_context_window(model_id: str) -> int | None:
    """The context window llmcore's OpenAI provider knows for this id."""
    try:
        from llmcore.providers.openai_provider import _OPENAI_PREFIX_CONTEXT_HEURISTICS
    except Exception:  # pragma: no cover - llmcore is importable here
        return None
    for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
        if model_id.startswith(prefix):
            return ctx
    return None


class OpenAIAdapter(OpenAICompatAdapter):
    provider_name = "openai"
    api_key_env_var = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return not any(mid.startswith(p) for p in _EXCLUDE_PREFIXES)

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        mid = normalized.model_id
        for flag, value in infer_capabilities(mid).items():
            setattr(normalized, flag, value)
        if normalized.supports_reasoning and "reasoning" not in normalized.tags:
            normalized.tags.append("reasoning")
        if normalized.context_length is None:
            normalized.context_length = known_context_window(mid)

        normalized.architecture_family = "gpt"
        normalized.architecture_type = "transformer"
        return normalized
