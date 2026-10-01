# tools/cardctl/adapters/groq_adapter.py
"""Groq adapter.

``openai_compat`` has claimed Groq since it was written, but no adapter was ever
registered, so ``cardctl generate groq`` failed. This closes that.
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter


class GroqAdapter(OpenAICompatAdapter):
    """Cards for Groq's OpenAI-compatible catalog."""

    provider_name = "groq"
    api_key_env_var = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        """Keep chat models; Groq also lists Whisper and guard models.

        These are kept too but typed correctly in :meth:`_enrich_model`, because
        a transcription endpoint is a real capability rather than noise.
        """
        return bool(model.get("id"))

    def _enrich_model(self, model: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        """Type the non-chat families and record Groq's reported limits."""
        model_id = model.model_id.lower()
        if "whisper" in model_id:
            model.model_type = "stt"
            model.supports_transcription = True
            model.supports_streaming = False
        elif "guard" in model_id:
            model.model_type = "moderation"
            model.supports_streaming = False
        elif "tts" in model_id or "playai" in model_id:
            model.model_type = "tts"
            model.supports_speech_synthesis = True
            model.supports_streaming = False
        else:
            model.supports_tools = True
            model.supports_json_mode = True

        ctx = raw.get("context_window")
        if isinstance(ctx, int) and ctx > 0:
            model.context_length = ctx
        max_out = raw.get("max_completion_tokens")
        if isinstance(max_out, int) and max_out > 0:
            model.max_output_tokens = max_out
        model.open_weights = True
        model.tags = [*model.tags, "groq-lpu"]
        return model
