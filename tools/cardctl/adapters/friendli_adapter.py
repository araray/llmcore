# tools/cardctl/adapters/friendli_adapter.py
"""FriendliAI (Model APIs) model discovery adapter.

Unlike most OpenAI-compatible services, Friendli's ``GET /serverless/v1/models``
returns a *rich* catalog: context length, max completion tokens, per-token
pricing (input / output / cache read / cache write / audio minute), a
``functionality`` capability block, input/output modalities, reasoning support
with the available ``reasoning_options``, the canonical ``base_model`` id from
models.dev, the serving ``mode``, and the deprecation date.

Almost every card field is therefore derived live; the enrichment overlay
(``enrichments/friendli.toml``) only carries curation that the API cannot know
(architecture family/type, display names, extra aliases).

Only the hosted Model APIs catalog is discoverable.  Dedicated Endpoints and
Friendli Container serve a single deployment each and expose no listing
endpoint, so cards for those are hand-written.

Service docs: https://friendli.ai/docs/llms.txt
Catalog:      https://friendli.ai/docs/guides/model-apis/pricing

Canonical provider key: ``friendli`` (default_cards/friendli/).  Also reachable
via the ``friendliai`` alias in the registry.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

logger = logging.getLogger(__name__)

#: ``mode`` values in the catalog mapped to llmcore model-card types.
_MODE_TO_MODEL_TYPE: dict[str, str] = {
    "chat": "chat",
    "completion": "completion",
    "embedding": "embedding",
}

#: Vendor prefix on a Friendli model id mapped to an architecture family.
#: Friendli serves other vendors' open-weight checkpoints, so the family comes
#: from the checkpoint owner rather than from Friendli itself.
_FAMILY_BY_OWNER: dict[str, str] = {
    "deepseek-ai": "DeepSeek",
    "google": "Gemma",
    "meta-llama": "Llama",
    "minimaxai": "MiniMax",
    "mistralai": "Mistral",
    "moonshotai": "Kimi",
    "nvidia": "Nemotron",
    "openai": "GPT-OSS",
    "qwen": "Qwen",
    "zai-org": "GLM",
}


def _price_per_million(value: Any) -> float | None:
    """Convert a Friendli per-token price string to USD per million tokens."""
    if value in (None, ""):
        return None
    try:
        return round(float(value) * 1_000_000, 6)
    except (TypeError, ValueError):
        return None


class FriendliAdapter(OpenAICompatAdapter):
    """Discover the Friendli Model APIs catalog and map it onto model cards."""

    provider_name = "friendli"
    api_key_env_var = "FRIENDLI_TOKEN"
    base_url = "https://api.friendli.ai/serverless/v1"

    def get_api_key(self) -> str | None:
        """Resolve the key, accepting every documented Friendli variable.

        ``FRIENDLI_TOKEN`` is the official SDK's variable and
        ``FRIENDLIAI_API_KEY`` is the spelling used throughout friendli.ai's
        own documentation examples; both are honored, as is the provider-side
        ``FRIENDLI_API_KEY``.
        """
        import os

        key = super().get_api_key()
        if key:
            return key
        for name in ("FRIENDLIAI_API_KEY", "FRIENDLI_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _include_model(self, model: dict[str, Any]) -> bool:
        """Include every catalog entry that carries an id."""
        return bool(model.get("id"))

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        """Map the rich Friendli catalog entry onto the normalized model."""
        functionality = raw.get("functionality") or {}
        inputs = raw.get("input_modalities") or []
        outputs = raw.get("output_modalities") or []

        normalized.display_name = raw.get("name") or normalized.display_name
        normalized.description = raw.get("description")
        normalized.model_type = _MODE_TO_MODEL_TYPE.get(raw.get("mode", "chat"), "chat")
        normalized.context_length = raw.get("context_length")
        normalized.max_output_tokens = raw.get("max_completion_tokens")

        normalized.supports_tools = bool(functionality.get("tool_call"))
        normalized.supports_structured_output = bool(functionality.get("structured_output"))
        # Friendli enforces response_format on every structured-output model,
        # which subsumes plain JSON mode.
        normalized.supports_json_mode = normalized.supports_structured_output
        normalized.supports_reasoning = bool(raw.get("reasoning"))
        normalized.supports_vision = "image" in inputs
        normalized.supports_video_input = "video" in inputs
        normalized.supports_audio_input = "audio" in inputs
        normalized.supports_audio_output = "audio" in outputs

        # Friendli serves open-weight checkpoints hosted on Hugging Face.
        normalized.open_weights = True
        owner = normalized.model_id.split("/", 1)[0].lower() if "/" in normalized.model_id else ""
        family = _FAMILY_BY_OWNER.get(owner)
        if family:
            normalized.architecture_family = family
        normalized.owned_by = (
            normalized.model_id.split("/", 1)[0] if "/" in normalized.model_id else None
        )

        deprecation = raw.get("deprecation_date")
        if deprecation:
            normalized.deprecation_date = str(deprecation)[:10]
            normalized.is_deprecated = True

        # --- Pricing (per-token USD strings -> per-million USD floats) ---
        pricing = raw.get("pricing") or {}
        input_price = _price_per_million(pricing.get("input") or pricing.get("prompt"))
        output_price = _price_per_million(pricing.get("output") or pricing.get("completion"))
        if input_price is not None or output_price is not None:
            block: dict[str, Any] = {
                "input": input_price or 0.0,
                "output": output_price or 0.0,
            }
            cached = _price_per_million(pricing.get("input_cache_read"))
            if cached is not None:
                block["cached_input"] = cached
            normalized.raw_api_data["_pricing"] = block

        # --- Provider extension (Friendli-specific serving metadata) ---
        extension: dict[str, Any] = {"endpoint_type": "serverless"}
        if raw.get("base_model"):
            extension["base_model"] = raw["base_model"]
        if raw.get("mode"):
            extension["mode"] = raw["mode"]
        if raw.get("interleaved") is not None:
            extension["interleaved"] = raw["interleaved"]
        if raw.get("reasoning_options"):
            extension["reasoning_options"] = raw["reasoning_options"]
            efforts = [
                opt.get("values")
                for opt in raw["reasoning_options"]
                if isinstance(opt, dict) and opt.get("type") == "effort"
            ]
            if efforts and efforts[0]:
                extension["reasoning_effort_levels"] = efforts[0]
            extension["reasoning_toggle"] = any(
                isinstance(opt, dict) and opt.get("type") == "toggle"
                for opt in raw["reasoning_options"]
            )
        if functionality:
            extension["functionality"] = functionality
        if raw.get("default_params"):
            extension["default_params"] = raw["default_params"]
        cache_write = _price_per_million(pricing.get("cache_write"))
        if cache_write is not None:
            extension["cache_write_per_million"] = cache_write
        audio_minute = pricing.get("audio_minute")
        if audio_minute not in (None, ""):
            extension["audio_minute_usd"] = audio_minute
        normalized.raw_api_data["_extension"] = extension

        # --- Tags ---
        tags: list[str] = [f"input:{m}" for m in inputs]
        tags += [f"output:{m}" for m in outputs]
        if normalized.supports_reasoning:
            tags.append("reasoning")
        if normalized.supports_tools:
            tags.append("tools")
        if normalized.supports_structured_output:
            tags.append("structured-output")
        if (normalized.context_length or 0) >= 1_000_000:
            tags.append("million-context")
        normalized.tags.extend(tags)

        return normalized
