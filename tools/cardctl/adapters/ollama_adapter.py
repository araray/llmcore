# tools/cardctl/adapters/ollama_adapter.py
"""Ollama model discovery adapter.

Uses the local Ollama API:
- ``GET /api/tags`` for model listing
- ``POST /api/show`` per model for details (family, params, context, quant)

By default targets ``http://localhost:11434``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Known family → architecture_type mapping.
_FAMILY_ARCH_MAP: dict[str, str] = {
    "llama": "dense",
    "gemma": "dense",
    "gemma2": "dense",
    "gemma3": "dense",
    "phi": "dense",
    "phi3": "dense",
    "qwen": "dense",
    "qwen2": "dense",
    "qwen3": "dense",
    "mistral": "dense",
    "command-r": "dense",
    "deepseek": "moe",
    "deepseek2": "moe",
    "mixtral": "moe",
    "starcoder": "dense",
    "codellama": "dense",
}


class OllamaAdapter(BaseAdapter):
    provider_name = "ollama"
    requires_api_key = False
    api_key_env_var = ""
    base_url = _DEFAULT_OLLAMA_URL

    async def fetch_models(self) -> list[NormalizedModel]:
        url = self.base_url.rstrip("/")
        result: list[NormalizedModel] = []

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            # 1. List models
            try:
                resp = await client.get(f"{url}/api/tags")
                resp.raise_for_status()
                models_data = resp.json().get("models", [])
            except httpx.ConnectError:
                logger.warning("Ollama not reachable at %s", url)
                return []

            # 2. Get details per model
            for m in models_data:
                model_name = m.get("name", "") or m.get("model", "")
                if not model_name:
                    continue

                details = await self._show_model(client, url, model_name)
                normalized = self._parse_model(model_name, m, details)
                if normalized:
                    result.append(normalized)

        logger.info("Fetched %d models from ollama", len(result))
        return result

    async def _show_model(
        self, client: httpx.AsyncClient, url: str, model_name: str
    ) -> dict[str, Any]:
        """Fetch model details via POST /api/show."""
        try:
            resp = await client.post(
                f"{url}/api/show",
                json={"name": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("Failed to show model %s: %s", model_name, e)
        return {}

    def _parse_model(
        self, model_name: str, listing: dict[str, Any], details: dict[str, Any]
    ) -> NormalizedModel | None:
        model_info = details.get("model_info", {})
        model_details = details.get("details", {})

        family = model_details.get("family", "").lower()
        param_size = model_details.get("parameter_size", "")
        quant = model_details.get("quantization_level", "")

        # Context length from model_info
        context_length = None
        for key, val in model_info.items():
            if "context_length" in key and isinstance(val, (int, float)):
                context_length = int(val)
                break

        arch_type = _FAMILY_ARCH_MAP.get(family, "dense")

        normalized = NormalizedModel(
            model_id=model_name,
            provider="ollama",
            model_type="chat",
            context_length=context_length,
            supports_streaming=True,
            architecture_family=family or None,
            architecture_type=arch_type,
            parameter_count=param_size or None,
            open_weights=True,
            license=model_details.get("license"),
            tags=["local", "open-source"],
            raw_api_data={"listing": listing, "details": details},
        )

        # Vision detection
        families = model_details.get("families", []) or []
        if "clip" in families or "vision" in family:
            normalized.supports_vision = True
            normalized.tags.append("vision")

        # Tool support for newer models
        if any(f in family for f in ("llama3", "qwen2", "qwen3", "gemma3", "mistral")):
            normalized.supports_tools = True

        if quant:
            normalized.tags.append(f"quant:{quant}")

        return normalized
