# tools/cardctl/adapters/mistral_adapter.py
"""Mistral AI model discovery adapter.

Uses the native ``GET /v1/models`` endpoint which returns rich metadata
including capabilities, context length, aliases, and deprecation info.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class MistralAdapter(BaseAdapter):
    provider_name = "mistral"
    api_key_env_var = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        result: list[NormalizedModel] = []
        for m in data.get("data", []):
            normalized = self._parse_model(m)
            if normalized:
                result.append(normalized)

        logger.info("Fetched %d models from mistral", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        model_id = m.get("id", "")
        if not model_id:
            return None

        caps = m.get("capabilities", {}) or {}
        deprecation = m.get("deprecation")

        # Determine model type
        model_type = "chat"
        if not caps.get("completion_chat", False) and not caps.get("completion_fim", False):
            if model_id.startswith(("mistral-embed", "codestral-embed")):
                model_type = "embedding"
            elif model_id.startswith("mistral-ocr"):
                model_type = "ocr"
            elif model_id.startswith("mistral-moderation"):
                model_type = "moderation"
        if caps.get("completion_fim", False):
            model_type = "code"

        # TTS / STT detection
        if model_id.startswith("voxtral") and "tts" in model_id:
            model_type = "tts"
        elif model_id.startswith("voxtral") and ("transcribe" in model_id or "realtime" in model_id):
            model_type = "stt"

        normalized = NormalizedModel(
            model_id=model_id,
            provider="mistral",
            display_name=m.get("name"),
            description=m.get("description", ""),
            model_type=model_type,
            context_length=m.get("max_context_length") or 32768,
            supports_streaming=caps.get("completion_chat", False),
            supports_tools=caps.get("function_calling", False),
            supports_vision=caps.get("vision", False),
            supports_audio_input=caps.get("audio", False) or caps.get("audio_transcription", False),
            supports_audio_output=caps.get("audio_generation", False),
            supports_json_mode=caps.get("json_mode", False),
            supports_reasoning=model_id.startswith("magistral"),
            supports_fim=caps.get("completion_fim", False),
            architecture_family="mistral",
            aliases=m.get("aliases", []) or [],
            owned_by=m.get("owned_by", "mistralai"),
            is_deprecated=bool(deprecation),
            deprecation_date=deprecation if isinstance(deprecation, str) else None,
            created_timestamp=m.get("created"),
            raw_api_data=m,
        )

        # Tags
        if normalized.supports_vision:
            normalized.tags.append("multimodal")
        if normalized.supports_audio_input:
            normalized.tags.append("audio")
        if normalized.supports_reasoning:
            normalized.tags.append("reasoning")
        if model_type == "code":
            normalized.tags.extend(["code", "fim"])

        # Extension fields
        ext: dict[str, Any] = {
            "open_weights": False,
            "fill_in_middle": caps.get("completion_fim", False),
            "guardrails": {"system_prompt": caps.get("safe_prompt_injection", False)},
        }
        default_temp = m.get("default_model_temperature")
        if default_temp is not None:
            ext["default_temperature"] = default_temp
        normalized.raw_api_data["_extension"] = ext

        return normalized
