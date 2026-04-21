# tools/cardctl/adapters/poe_adapter.py
"""Poe model discovery adapter.

Uses the Poe bot settings endpoint to discover available models.
The Poe API uses OpenAI-compatible chat completions but has its own
model listing mechanism.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

_POE_MODELS_URL = "https://api.poe.com/v2/bots"


class PoeAdapter(BaseAdapter):
    provider_name = "poe"
    api_key_env_var = "POE_API_KEY"
    base_url = _POE_MODELS_URL

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        headers = {"Authorization": f"Bearer {api_key}"}
        result: list[NormalizedModel] = []

        # Poe also supports OpenAI-compatible /v1/models
        compat_url = "https://api.poe.com/v1/models"

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                resp = await client.get(compat_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models_list = data.get("data", [])
            except Exception as e:
                logger.warning("Poe model listing failed: %s", e)
                return []

        for m in models_list:
            normalized = self._parse_model(m)
            if normalized:
                result.append(normalized)

        logger.info("Fetched %d models from poe", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        model_id = m.get("id", "")
        if not model_id:
            return None

        normalized = NormalizedModel(
            model_id=model_id,
            provider="poe",
            display_name=m.get("name") or model_id,
            model_type="chat",
            context_length=m.get("context_length") or 4096,
            supports_streaming=True,
            owned_by=m.get("owned_by"),
            created_timestamp=m.get("created"),
            raw_api_data=m,
        )

        return normalized
