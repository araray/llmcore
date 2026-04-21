# tools/cardctl/adapters/anthropic_adapter.py
"""Anthropic model discovery adapter.

Uses the native ``GET /v1/models`` endpoint which returns structured
``ModelInfo`` objects with capability flags.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    provider_name = "anthropic"
    api_key_env_var = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        result: list[NormalizedModel] = []
        url = f"{self.base_url}/models?limit=100"

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            while url:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                for m in data.get("data", []):
                    normalized = self._parse_model(m)
                    if normalized:
                        result.append(normalized)

                # Pagination
                if data.get("has_more") and data.get("last_id"):
                    url = f"{self.base_url}/models?limit=100&after_id={data['last_id']}"
                else:
                    url = None

        logger.info("Fetched %d models from anthropic", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        model_id = m.get("id", "")
        if not model_id:
            return None

        caps = m.get("capabilities", {}) or {}

        def _cap_supported(cap_name: str) -> bool:
            cap = caps.get(cap_name, {})
            return cap.get("supported", False) if isinstance(cap, dict) else bool(cap)

        normalized = NormalizedModel(
            model_id=model_id,
            provider="anthropic",
            display_name=m.get("display_name"),
            description=m.get("description"),
            created_timestamp=None,  # Anthropic uses ISO date string
            supports_streaming=True,
            supports_tools=_cap_supported("tool_use"),
            supports_vision=_cap_supported("vision"),
            supports_json_mode=_cap_supported("json_output"),
            supports_structured_output=_cap_supported("json_output"),
            supports_reasoning=_cap_supported("extended_thinking"),
            architecture_family="claude",
            architecture_type="transformer",
            raw_api_data=m,
        )

        # Parse created_at to timestamp
        created_at = m.get("created_at")
        if created_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                normalized.created_timestamp = int(dt.timestamp())
            except Exception:
                pass

        # Extension fields
        ext: dict[str, Any] = {}
        if _cap_supported("extended_thinking"):
            ext["extended_thinking"] = True
        if _cap_supported("prompt_caching"):
            ext["prompt_caching"] = True
        if _cap_supported("batch"):
            ext["batch_api"] = True
        if _cap_supported("citations"):
            ext["citations"] = True
        if _cap_supported("computer_use"):
            ext["computer_use"] = True
        if _cap_supported("pdf_input"):
            ext["pdf_input"] = True
        if ext:
            normalized.raw_api_data["_extension"] = ext

        return normalized
