# tools/cardctl/adapters/together_adapter.py
"""Together AI adapter.

Like Groq, Together was named in ``openai_compat``'s docstring but never
registered. Together's listing is richer than the bare OpenAI shape — it
reports type, context length and pricing — so those are used where present.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

logger = logging.getLogger(__name__)


class TogetherAdapter(OpenAICompatAdapter):
    """Cards for Together's catalog."""

    provider_name = "together"
    api_key_env_var = "TOGETHER_API_KEY"
    base_url = "https://api.together.xyz/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch Together's catalog.

        Together returns a bare JSON **array**, not the OpenAI ``{"data": [...]}``
        envelope, so the shared implementation cannot be reused directly.
        """
        self.check_api_key()
        url = f"{self.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self.get_api_key()}"}

        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        entries = payload if isinstance(payload, list) else payload.get("data", [])
        result: list[NormalizedModel] = []
        for raw in entries:
            model_id = raw.get("id") or raw.get("name")
            if not model_id:
                continue
            model = NormalizedModel(
                model_id=str(model_id),
                provider=self.provider_name,
                display_name=raw.get("display_name"),
                owned_by=raw.get("organization"),
                created_timestamp=raw.get("created"),
                supports_streaming=True,
                raw_api_data=raw,
            )
            result.append(self._enrich_model(model, raw))

        logger.info("Fetched %d models from %s", len(result), self.provider_name)
        return result

    def _enrich_model(self, model: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        """Map Together's own ``type`` field onto card model types."""
        kind = str(raw.get("type") or "").lower()
        mapping = {
            "chat": "chat",
            "language": "completion",
            "code": "code",
            "embedding": "embedding",
            "rerank": "rerank",
            "moderation": "moderation",
            "image": "image-generation",
            "audio": "audio",
        }
        model.model_type = mapping.get(kind, "chat")
        if model.model_type == "image-generation":
            model.supports_image_generation = True
            model.supports_streaming = False
        elif model.model_type in ("embedding", "rerank", "moderation"):
            model.supports_streaming = False
        else:
            model.supports_tools = True

        ctx = raw.get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            model.context_length = ctx
        model.open_weights = True
        return model
