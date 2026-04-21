# tools/cardctl/adapters/openrouter_adapter.py
"""OpenRouter model discovery adapter.

OpenRouter aggregates models from many providers and exposes a rich
``GET /api/v1/models`` endpoint with pricing, context length, and
capability information.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class OpenRouterAdapter(BaseAdapter):
    provider_name = "openrouter"
    requires_api_key = False  # Public endpoint, key optional
    api_key_env_var = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        url = f"{self.base_url}/models"
        headers: dict[str, str] = {}
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        result: list[NormalizedModel] = []
        for m in data.get("data", []):
            normalized = self._parse_model(m)
            if normalized:
                result.append(normalized)

        logger.info("Fetched %d models from openrouter", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        model_id = m.get("id", "")
        if not model_id:
            return None

        pricing = m.get("pricing", {}) or {}
        ctx = m.get("context_length") or 4096

        # Parse pricing (OpenRouter returns per-token strings)
        input_price = _parse_price(pricing.get("prompt", "0"))
        output_price = _parse_price(pricing.get("completion", "0"))

        normalized = NormalizedModel(
            model_id=model_id,
            provider="openrouter",
            display_name=m.get("name"),
            description=m.get("description"),
            model_type="chat",
            context_length=ctx,
            max_output_tokens=m.get("top_provider", {}).get("max_completion_tokens"),
            supports_streaming=True,
            supports_tools="tool_use" in (m.get("supported_parameters", []) or []),
            supports_json_mode="response_format" in (m.get("supported_parameters", []) or []),
            created_timestamp=m.get("created"),
            raw_api_data=m,
        )

        # Store pricing in raw_api_data for builder to pick up
        if input_price > 0 or output_price > 0:
            normalized.raw_api_data["_pricing"] = {
                "input": input_price,
                "output": output_price,
            }

        return normalized


def _parse_price(price_str: str | float) -> float:
    """Parse OpenRouter per-token price to per-million-tokens USD."""
    try:
        per_token = float(price_str)
        return round(per_token * 1_000_000, 4)
    except (ValueError, TypeError):
        return 0.0
