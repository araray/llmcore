# tools/cardctl/adapters/openai_compat.py
"""Shared adapter for OpenAI-compatible model listing APIs.

Providers that expose ``GET /v1/models`` with the standard OpenAI response
shape can subclass this adapter and override only the provider-specific
configuration and model filtering.

Used by: OpenAI, DeepSeek, xAI, Qwen, Moonshot, Groq, Together.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class OpenAICompatAdapter(BaseAdapter):
    """Adapter for providers using the OpenAI ``/v1/models`` endpoint.

    The OpenAI model list response is minimal::

        {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"}

    No context length, capabilities, or pricing.  Enrichment overlays carry
    the bulk of useful metadata for these providers.

    Subclasses should override:
    - ``provider_name``
    - ``api_key_env_var``
    - ``base_url``
    - ``_include_model()`` to filter out irrelevant models
    - ``_enrich_model()`` for provider-specific field mapping
    """

    models_endpoint: str = "/models"

    def _include_model(self, model: dict[str, Any]) -> bool:
        """Return True if this model should be included in the output.

        Override in subclasses to filter out fine-tunes, image models, etc.
        """
        return True

    def _enrich_model(
        self, normalized: NormalizedModel, raw: dict[str, Any]
    ) -> NormalizedModel:
        """Apply provider-specific enrichments to a normalized model.

        Override in subclasses for provider-specific capability detection.
        """
        return normalized

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        url = f"{self.base_url.rstrip('/')}{self.models_endpoint}"
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        models_list = data.get("data", [])
        result: list[NormalizedModel] = []

        for m in models_list:
            if not self._include_model(m):
                continue

            model_id = m.get("id", "")
            if not model_id:
                continue

            normalized = NormalizedModel(
                model_id=model_id,
                provider=self.provider_name,
                owned_by=m.get("owned_by"),
                created_timestamp=m.get("created"),
                supports_streaming=True,  # Most OpenAI-compat models support streaming
                raw_api_data=m,
            )

            normalized = self._enrich_model(normalized, m)
            result.append(normalized)

        logger.info("Fetched %d models from %s", len(result), self.provider_name)
        return result
