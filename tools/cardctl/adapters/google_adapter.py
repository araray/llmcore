# tools/cardctl/adapters/google_adapter.py
"""Google Gemini model discovery adapter.

Uses the REST API directly (no SDK dependency):
``GET https://generativelanguage.googleapis.com/v1beta/models`` (key in ``x-goog-api-key``)

Returns rich metadata including context lengths and supported generation
methods.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Families served by the Gemini API (2026-11-16 listing). "aqa"
# (generateAnswer) and other legacy PaLM-era entries stay out.
_INCLUDE_PREFIXES = (
    "gemini-", "gemma-", "lyria-", "nano-banana", "deep-research", "antigravity",
)

#: Generation methods that make a model usable from llmcore.
_USABLE_METHODS = frozenset(
    {"generateContent", "embedContent", "bidiGenerateContent", "bidiGenerateMusic", "createDecision"}
)

#: Gemini generations with function calling, vision and JSON mode
#: (gemini-2 onward; Gemini 4 included).
_MODERN_GEMINI = re.compile(r"^gemini-(?:[2-9]|\d{2,})")


class GoogleAdapter(BaseAdapter):
    provider_name = "google"
    api_key_env_var = "GOOGLE_API_KEY"
    base_url = _GEMINI_API_URL

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        # The key goes in a header, not the URL: query strings end up in
        # proxy and server logs.
        headers = {"x-goog-api-key": api_key}
        url = f"{self.base_url}?pageSize=200"
        result: list[NormalizedModel] = []

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            while url:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                for m in data.get("models", []):
                    normalized = self._parse_model(m)
                    if normalized:
                        result.append(normalized)

                # Pagination
                next_token = data.get("nextPageToken")
                if next_token:
                    url = f"{self.base_url}?pageSize=200&pageToken={next_token}"
                else:
                    url = None

        logger.info("Fetched %d models from google", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        """Classify one listing entry from what the API states.

        The listing no longer names ``streamGenerateContent`` although every
        ``generateContent`` model streams, so streaming follows from
        ``generateContent``; reasoning comes from the entry's ``thinking``
        flag. Model type follows the generation method and the family
        (Live, TTS, image, music, decision, transcription).
        """
        full_name = m.get("name", "")
        model_id = full_name.replace("models/", "")
        if not model_id or not model_id.startswith(_INCLUDE_PREFIXES):
            return None

        methods = set(m.get("supportedGenerationMethods") or [])
        if not methods & _USABLE_METHODS:
            return None
        generate = "generateContent" in methods
        live = bool(methods & {"bidiGenerateContent", "bidiGenerateMusic"}) and not generate

        if "createDecision" in methods:
            model_type = "decision"
        elif "embedContent" in methods and not generate:
            model_type = "embedding"
        elif live or model_id.startswith("lyria-") or "native-audio" in model_id \
                or "image" in model_id or "nano-banana" in model_id:
            # Token-billed multimodal LLMs whose output is audio, music or
            # images; their windows are real token budgets stated by the API,
            # unlike a dedicated speech/image endpoint's prompt limit.
            model_type = "multimodal"
        elif "transcribe" in model_id:
            model_type = "stt"
        elif model_id.endswith("-tts") or "-tts-" in model_id:
            model_type = "tts"
        else:
            model_type = "chat"

        conversational = model_type == "chat"
        normalized = NormalizedModel(
            model_id=model_id,
            provider="google",
            display_name=m.get("displayName"),
            description=m.get("description"),
            model_type=model_type,
            context_length=m.get("inputTokenLimit"),
            max_output_tokens=m.get("outputTokenLimit"),
            supports_streaming=generate or live,
            supports_tools=conversational and bool(_MODERN_GEMINI.match(model_id)),
            architecture_family=model_id.split("-", 1)[0],
            architecture_type="transformer",
            raw_api_data=m,
        )
        if "image" in model_id or "nano-banana" in model_id:
            normalized.supports_vision = True  # Nano Banana edits images it is given
            normalized.tags.append("image-output")
        if conversational and (_MODERN_GEMINI.match(model_id) or model_id.startswith("gemma-4")):
            normalized.supports_vision = True
            normalized.supports_json_mode = True
        if m.get("thinking") is True:
            normalized.supports_reasoning = True
            normalized.tags.append("reasoning")
        if live:
            normalized.tags.append("live")
        if model_id.startswith("lyria-"):
            normalized.tags.append("music")
        return normalized
