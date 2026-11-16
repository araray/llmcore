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
            "anthropic-version": "2026-10-21",
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
        """One Models API entry, read with the keys the API actually uses.

        The old version read ``tool_use``, ``vision``, ``json_output`` and
        ``extended_thinking``, none of which exist, so every Anthropic card
        claimed no tools, no vision, no structured output and no reasoning.
        The real keys are ``image_input``, ``pdf_input``,
        ``structured_outputs``, ``thinking{types}``, ``effort{levels}``,
        ``batch``, ``citations``, ``code_execution``, ``context_management``
        and ``server_tools``.
        """
        model_id = m.get("id", "")
        if not model_id:
            return None

        caps = m.get("capabilities", {}) or {}

        def _cap_supported(cap_name: str) -> bool:
            cap = caps.get(cap_name, {})
            return cap.get("supported", False) if isinstance(cap, dict) else bool(cap)

        thinking = caps.get("thinking") or {}
        thinking_types = sorted(
            name
            for name, flag in (thinking.get("types") or {}).items()
            if isinstance(flag, dict) and flag.get("supported")
        )
        efforts = [
            name
            for name in ("low", "medium", "high", "xhigh", "max")
            if isinstance((caps.get("effort") or {}).get(name), dict)
            and caps["effort"][name].get("supported")
        ]
        lifecycle = m.get("lifecycle")

        normalized = NormalizedModel(
            model_id=model_id,
            provider="anthropic",
            display_name=m.get("display_name"),
            description=m.get("description"),
            created_timestamp=None,  # Anthropic uses ISO date string
            context_length=m.get("max_input_tokens") or None,
            max_output_tokens=m.get("max_tokens") or None,
            # Every Claude model on the Messages API takes tools; the Models
            # API has no flag for it because there is no model that does not.
            supports_streaming=True,
            supports_tools=True,
            supports_vision=_cap_supported("image_input"),
            supports_json_mode=_cap_supported("structured_outputs"),
            supports_structured_output=_cap_supported("structured_outputs"),
            supports_reasoning=bool(thinking_types and thinking_types != ["disabled"]),
            is_deprecated=lifecycle == "deprecated",
            deprecation_date=(m.get("retires_at") or "")[:10] or None,
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
        if thinking_types:
            ext["thinking_types"] = thinking_types
        if efforts:
            ext["effort_levels"] = efforts
        for key, name in (("batch", "batch_api"), ("citations", "citations"),
                          ("pdf_input", "pdf_input"), ("code_execution", "code_execution"),
                          ("context_management", "context_management")):
            if _cap_supported(key):
                ext[name] = True
        server_tools = [
            name for name, flag in (caps.get("server_tools") or {}).items()
            if isinstance(flag, dict) and flag.get("supported")
        ]
        if server_tools:
            ext["server_tools"] = sorted(server_tools)
        if lifecycle:
            ext["lifecycle"] = lifecycle
        if ext:
            normalized.raw_api_data["_extension"] = ext

        return normalized
