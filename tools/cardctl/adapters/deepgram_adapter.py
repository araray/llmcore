# tools/cardctl/adapters/deepgram_adapter.py
"""Deepgram adapter — speech models, not chat.

Deepgram's ``/v1/models`` returns separate ``stt`` and ``tts`` collections with
language and architecture metadata, which maps onto card types directly.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class DeepgramAdapter(BaseAdapter):
    """Cards for Deepgram's speech-to-text and text-to-speech models."""

    provider_name = "deepgram"
    api_key_env_var = "DEEPGRAM_API_KEY"
    base_url = "https://api.deepgram.com"

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch both the STT and TTS collections."""
        self.check_api_key()
        url = f"{self.base_url.rstrip('/')}/v1/models"
        headers = {"Authorization": f"Token {self.get_api_key()}"}

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        # Deepgram lists one record per model *and language*, so `general`
        # appears dozens of times. Writing a card per record means the same file
        # is rewritten repeatedly and whichever variant happens to come last
        # decides its language metadata — so records are grouped by canonical
        # name and their language coverage is merged into one honest card.
        grouped: dict[str, tuple[dict[str, Any], str, set[str]]] = {}
        order: list[str] = []
        for collection, model_type in (("stt", "stt"), ("tts", "tts")):
            for raw in payload.get(collection) or []:
                model_id = raw.get("canonical_name") or raw.get("name")
                if not model_id:
                    continue
                key = str(model_id)
                languages = {
                    str(lang) for lang in (raw.get("languages") or []) if lang
                }
                if key in grouped:
                    grouped[key][2].update(languages)
                    continue
                grouped[key] = (raw, model_type, set(languages))
                order.append(key)

        models: list[NormalizedModel] = []
        for key in order:
            raw, model_type, languages = grouped[key]
            model = self._normalize(raw, model_type, sorted(languages))
            if model is not None:
                models.append(model)

        logger.info(
            "Fetched %d Deepgram models (%d API records, grouped by canonical name).",
            len(models),
            sum(len(payload.get(c) or []) for c in ("stt", "tts")),
        )
        return models

    def _normalize(
        self, raw: dict[str, Any], model_type: str, languages: list[str] | None = None
    ) -> NormalizedModel | None:
        """Turn one Deepgram record into a normalized model.

        *languages* is the merged coverage across every record sharing this
        canonical name, rather than whatever one record happened to list.
        """
        # Deepgram's canonical id is the `canonical_name` (e.g. "nova-3-general");
        # `name` is the short label and `uuid` is an internal handle, so using
        # canonical_name keeps card ids matching what callers actually pass.
        model_id = raw.get("canonical_name") or raw.get("name")
        if not model_id:
            return None

        languages = languages if languages is not None else (raw.get("languages") or [])
        model = NormalizedModel(
            model_id=str(model_id),
            provider=self.provider_name,
            display_name=raw.get("name"),
            model_type=model_type,
            supports_streaming=True,
            owned_by="deepgram",
            tags=["audio", model_type],
            raw_api_data=raw,
        )
        if model_type == "stt":
            model.supports_transcription = True
            model.supports_audio_input = True
        else:
            model.supports_speech_synthesis = True
            model.supports_audio_output = True

        version = raw.get("version")
        if version:
            model.architecture_family = str(raw.get("architecture") or "") or None
            model.raw_api_data["version"] = version
        if languages:
            model.tags = [*model.tags, f"languages:{len(languages)}"]
            model.raw_api_data["languages"] = languages
        return model
