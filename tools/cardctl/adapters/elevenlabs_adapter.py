# tools/cardctl/adapters/elevenlabs_adapter.py
"""ElevenLabs adapter — voice models.

``/v1/models`` lists the TTS families with per-model language coverage and the
character limit per request, which maps onto the card's context field (the unit
is characters rather than tokens, which the description makes explicit so nobody
reads it as a token window).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class ElevenLabsAdapter(BaseAdapter):
    """Cards for ElevenLabs' speech models."""

    provider_name = "elevenlabs"
    api_key_env_var = "ELEVENLABS_API_KEY"
    base_url = "https://api.elevenlabs.io"

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch the advertised model list."""
        self.check_api_key()
        url = f"{self.base_url.rstrip('/')}/v1/models"
        headers = {"xi-api-key": self.get_api_key() or ""}

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        models: list[NormalizedModel] = []
        for raw in payload if isinstance(payload, list) else payload.get("models", []):
            model = self._normalize(raw)
            if model is not None:
                models.append(model)

        logger.info("Fetched %d models from %s", len(models), self.provider_name)
        return models

    def _normalize(self, raw: dict[str, Any]) -> NormalizedModel | None:
        """Turn one ElevenLabs record into a normalized model."""
        model_id = raw.get("model_id")
        if not model_id:
            return None

        languages = [
            lang.get("language_id") for lang in (raw.get("languages") or []) if lang
        ]
        can_tts = bool(raw.get("can_do_text_to_speech"))
        can_sts = bool(raw.get("can_do_voice_conversion"))

        model = NormalizedModel(
            model_id=str(model_id),
            provider=self.provider_name,
            display_name=raw.get("name"),
            description=raw.get("description"),
            model_type="tts" if can_tts else "audio",
            supports_streaming=can_tts,
            supports_audio_output=can_tts,
            supports_audio_input=can_sts,
            supports_speech_synthesis=can_tts,
            owned_by="elevenlabs",
            tags=["audio", "tts" if can_tts else "speech-to-speech"],
            raw_api_data=raw,
        )

        # The limit is CHARACTERS, not tokens. Recorded in the context field
        # because that is where a caller looks for "how much input fits", with
        # the unit stated so it is not mistaken for a token window.
        limit = raw.get("maximum_text_length_per_request")
        if isinstance(limit, int) and limit > 0:
            model.context_length = limit
            model.description = (
                f"{model.description or model.display_name or model_id} "
                f"(context is {limit} characters, not tokens — ElevenLabs bills "
                f"per character)."
            ).strip()
        if languages:
            model.tags = [*model.tags, f"languages:{len(languages)}"]
        if raw.get("requires_alpha_access"):
            model.tags = [*model.tags, "alpha-access"]
            model.status = "preview"
        return model
