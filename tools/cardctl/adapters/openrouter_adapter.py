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

        # The listing is paginated (limit <= 1000, offset, total_count) and shows
        # only text-output models unless asked for every modality (468 of 724
        # models on 2026-11-16).
        listed: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            offset = 0
            while True:
                resp = await client.get(
                    url,
                    headers=headers,
                    params={"output_modalities": "all", "limit": 1000, "offset": offset},
                )
                resp.raise_for_status()
                data = resp.json()
                page = data.get("data", []) or []
                listed.extend(page)
                offset += len(page)
                total = data.get("total_count")
                if not page or total is None or offset >= int(total):
                    break

        result: list[NormalizedModel] = []
        for m in listed:
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
        # No fallback: OpenRouter states `context_length` for everything it
        # routes, so an absent one means the route is unusual, not that it
        # holds 128,000 tokens.
        ctx = m.get("context_length") or None

        # Parse pricing (OpenRouter returns per-token strings)
        input_price = _parse_price(pricing.get("prompt", "0"))
        output_price = _parse_price(pricing.get("completion", "0"))

        params = m.get("supported_parameters", []) or []
        architecture = m.get("architecture", {}) or {}
        inputs = architecture.get("input_modalities", []) or []
        outputs = architecture.get("output_modalities", []) or []
        model_type = "chat"
        if "text" not in outputs:
            if "image" in outputs:
                # Called through chat completions with a token prompt, so the
                # stated window is a real token budget: a multimodal model
                # with image output, not a fixed-prompt image endpoint.
                model_type = "multimodal"
            elif "audio" in outputs:
                model_type = "tts"
            elif "video" in outputs:
                model_type = "video-generation"
            elif outputs == ["embeddings"] or "embeddings" in outputs:
                model_type = "embedding"
        normalized = NormalizedModel(
            model_id=model_id,
            provider="openrouter",
            display_name=m.get("name"),
            description=m.get("description"),
            model_type=model_type,
            context_length=ctx,
            max_output_tokens=m.get("top_provider", {}).get("max_completion_tokens"),
            supports_streaming=True,
            # The parameter is "tools"; "tool_use" (checked before) appears on
            # none of the 724 live models, so every card said "no tools".
            supports_tools="tools" in params,
            supports_json_mode="response_format" in params,
            supports_structured_output="structured_outputs" in params,
            supports_reasoning=bool(m.get("reasoning")) or "reasoning" in params,
            supports_vision="image" in inputs,
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
