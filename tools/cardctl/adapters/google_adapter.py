# tools/cardctl/adapters/google_adapter.py
"""Google Gemini model discovery adapter.

Uses the REST API directly (no SDK dependency):
``GET https://generativelanguage.googleapis.com/v1beta/models?key={key}``

Returns rich metadata including context lengths and supported generation
methods.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Prefixes to include (filter out legacy PaLM, embedding-only, etc.)
_INCLUDE_PREFIXES = ("gemini-",)

# Models that support tool use
_TOOL_CAPABLE_PREFIXES = ("gemini-2", "gemini-3")


class GoogleAdapter(BaseAdapter):
    provider_name = "google"
    api_key_env_var = "GOOGLE_API_KEY"
    base_url = _GEMINI_API_URL

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        url = f"{self.base_url}?key={api_key}&pageSize=200"
        result: list[NormalizedModel] = []

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            while url:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                for m in data.get("models", []):
                    normalized = self._parse_model(m)
                    if normalized:
                        result.append(normalized)

                # Pagination
                next_token = data.get("nextPageToken")
                if next_token:
                    url = f"{self.base_url}?key={api_key}&pageSize=200&pageToken={next_token}"
                else:
                    url = None

        logger.info("Fetched %d models from google", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        # name is like "models/gemini-2.5-flash"
        full_name = m.get("name", "")
        model_id = full_name.replace("models/", "")
        if not model_id:
            return None

        # Filter to Gemini models
        if not any(model_id.startswith(p) for p in _INCLUDE_PREFIXES):
            return None

        gen_methods = m.get("supportedGenerationMethods", [])
        is_chat = "generateContent" in gen_methods
        is_embedding = "embedContent" in gen_methods and not is_chat

        if not is_chat and not is_embedding:
            return None

        model_type = "embedding" if is_embedding else "chat"

        normalized = NormalizedModel(
            model_id=model_id,
            provider="google",
            display_name=m.get("displayName"),
            description=m.get("description"),
            model_type=model_type,
            context_length=m.get("inputTokenLimit"),
            max_output_tokens=m.get("outputTokenLimit"),
            supports_streaming="streamGenerateContent" in gen_methods,
            supports_tools=any(model_id.startswith(p) for p in _TOOL_CAPABLE_PREFIXES),
            architecture_family="gemini",
            architecture_type="transformer",
            raw_api_data=m,
        )

        # Gemini 2.0+ generally supports vision
        if model_id.startswith("gemini-2") or model_id.startswith("gemini-3"):
            normalized.supports_vision = True
            normalized.supports_json_mode = True

        # Reasoning models
        if "thinking" in model_id or "think" in m.get("description", "").lower():
            normalized.supports_reasoning = True
            normalized.tags.append("reasoning")

        return normalized
