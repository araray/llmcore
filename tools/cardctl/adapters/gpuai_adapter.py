# tools/cardctl/adapters/gpuai_adapter.py
"""gpu.ai model-discovery adapter for cardctl.

Written to close a gap the coverage audit was already shouting about:
``gpuai`` was registered in ``PROVIDER_MAP`` when the provider landed, but no
adapter came with it, so ``cardctl generate gpuai`` failed and the shipped
cards could only have been produced out-of-band. That audit exists precisely
to catch this, and it did.

gpu.ai's ``GET /v1/models`` is unusually complete for an OpenAI-compatible
catalogue. It states ``modality``, ``context_length``, ``aliases``,
``supported_parameters`` and ``pricing`` inline — which is why this adapter
needs no scraping and no second request.

**Prices arrive in cents, and not always per token.** The four modalities are
priced in four different ways:

=================  ===========================================
modality           pricing keys
=================  ===========================================
``chat``           ``input_per_1m_tokens_cents`` / ``output_…``
``embedding``      ``input_per_1m_tokens_cents`` only
``image``          ``per_mpxl_microcents``
``video``          ``per_video_microcents``, ``per_video_second_cents``
=================  ===========================================

So a per-token rate is only meaningful for two of them, and the other two go
through the unit pricing added in #48 rather than being coerced into a
per-token figure they do not have.

``supported_parameters`` is read as a **real negative**: gpu.ai enumerates what
each endpoint accepts, so a model without ``tools`` in that list does not take
tools. A model that states ``null`` says nothing, and nothing is what gets
recorded.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

#: ``modality`` -> the card's ``model_type``.
MODALITIES: dict[str, str] = {
    "chat": "chat",
    "embedding": "embedding",
    "image": "image-generation",
    "video": "video-generation",
}

#: ``supported_parameters`` entry -> the capability flags it proves.
PARAM_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "tools": ("supports_tools",),
    "tool_choice": ("supports_tools",),
    "response_format": ("supports_json_mode", "supports_structured_output"),
    "reasoning_effort": ("supports_reasoning",),
    "stream": ("supports_streaming",),
}


def _cents_per_million(value: Any) -> float | None:
    """Convert a cents-per-1M-tokens figure to dollars per 1M tokens."""
    if value is None:
        return None
    try:
        return round(float(value) / 100.0, 6)
    except (TypeError, ValueError):
        return None


def _microcents(value: Any) -> float | None:
    """Convert microcents to dollars."""
    if value is None:
        return None
    try:
        return round(float(value) / 100_000_000.0, 8)
    except (TypeError, ValueError):
        return None


class GpuAiAdapter(BaseAdapter):
    """Discovers gpu.ai models from its OpenAI-compatible catalogue."""

    provider_name = "gpuai"
    requires_api_key = True
    api_key_env_var = "GPUAI_API_KEY"
    base_url = "https://api.gpu.ai/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch and normalize the catalogue.

        Raises:
            RuntimeError: If the catalogue cannot be read. A partial card set
                is worse than none: a missing model looks identical to a
                retired one, and regenerating would delete its card.
        """
        self.check_api_key()
        headers = {"Authorization": f"Bearer {self.get_api_key()}"}
        url = f"{self.base_url.rstrip('/')}/models"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"gpu.ai model discovery failed: {exc}") from exc

        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise RuntimeError(
                "gpu.ai returned no `data` list of models; refusing to generate "
                "cards from a shape this adapter does not recognise."
            )

        out: list[NormalizedModel] = []
        for row in rows:
            model = self._normalize(row)
            if model is not None:
                out.append(model)
        logger.info("gpu.ai: normalized %d of %d catalogue entries", len(out), len(rows))
        return out

    # ------------------------------------------------------------------

    def _normalize(self, row: Any) -> NormalizedModel | None:
        if not isinstance(row, dict) or not row.get("id"):
            return None
        modality = str(row.get("modality") or "")
        model_type = MODALITIES.get(modality)
        if model_type is None:
            # A modality this adapter has never seen is skipped rather than
            # guessed into `chat`, which would make it selectable by a router
            # that cannot call it.
            logger.debug("gpu.ai: skipping %s with modality %r", row["id"], modality)
            return None

        model_id = str(row["id"])
        context = row.get("context_length")
        model = NormalizedModel(
            model_id=model_id,
            provider=self.provider_name,
            display_name=str(row.get("name") or model_id),
            model_type=model_type,
            # Absent for the media modalities, and left absent rather than
            # defaulted: an image generator has no token window at all.
            context_length=int(context) if isinstance(context, int) and context > 0 else None,
            status=str(row.get("status") or "active"),
            raw_api_data=dict(row),
        )

        params = row.get("supported_parameters")
        if isinstance(params, list):
            for param in params:
                for flag in PARAM_CAPABILITIES.get(str(param), ()):
                    setattr(model, flag, True)
        model.extra_aliases = [str(a) for a in (row.get("aliases") or []) if a]

        self._attach_pricing(model, row.get("pricing"))
        author = row.get("author") or row.get("owned_by")
        if author:
            model.raw_api_data["_extension"] = {"author": str(author)}
        return model

    def _attach_pricing(self, model: NormalizedModel, pricing: Any) -> None:
        """Stash prices for the builder, in whichever unit this model uses."""
        if not isinstance(pricing, dict):
            return

        token_rates: dict[str, float] = {}
        for key, field in (
            ("input_per_1m_tokens_cents", "input"),
            ("output_per_1m_tokens_cents", "output"),
            ("cached_input_per_1m_tokens_cents", "cached_input"),
        ):
            rate = _cents_per_million(pricing.get(key))
            if rate is not None:
                token_rates[field] = rate
        if token_rates:
            model.raw_api_data["_pricing"] = token_rates

        # Per-unit pricing, for the modalities a per-token rate cannot
        # describe. Recorded under the keyword each unit is billed by rather
        # than flattened, so `get_cost(..., video_seconds=5)` can find it.
        unit_rates: dict[str, float] = {}
        per_mpxl = _microcents(pricing.get("per_mpxl_microcents"))
        if per_mpxl is not None:
            unit_rates["per_megapixel"] = per_mpxl
        per_video = _microcents(pricing.get("per_video_microcents"))
        if per_video is not None:
            unit_rates["per_video"] = per_video
        per_second = _microcents(pricing.get("per_video_second_microcents"))
        if per_second is None:
            cents = pricing.get("per_video_second_cents")
            per_second = None if cents is None else round(float(cents) / 100.0, 6)
        if per_second is not None:
            unit_rates["per_video_second"] = per_second
        if unit_rates:
            model.raw_api_data["_unit_pricing"] = unit_rates
