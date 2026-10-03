# tools/cardctl/core/builder.py
"""Build ModelCard-compatible dicts from NormalizedModel + enrichments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..adapters.base import NormalizedModel
from .common import model_id_to_display_name
from .enrichment import EnrichmentStore, ModelEnrichment


class CardBuilder:
    """Converts :class:`NormalizedModel` instances into card dicts.

    Merges API-derived data with enrichment overlays to produce dicts
    that pass ``ModelCard.model_validate()``.

    Usage::

        store = EnrichmentStore.load("openai")
        builder = CardBuilder("openai", store)
        card = builder.build(normalized_model)
    """

    # Map from provider name → extension field name on ModelCard.
    _EXTENSION_FIELD_MAP: dict[str, str] = {
        "openai": "provider_openai",
        "anthropic": "provider_anthropic",
        "google": "provider_google",
        "ollama": "provider_ollama",
        "deepseek": "provider_deepseek",
        "qwen": "provider_qwen",
        "mistral": "provider_mistral",
        "xai": "provider_xai",
    }

    def __init__(self, provider: str, enrichments: EnrichmentStore):
        self.provider = provider
        self.enrichments = enrichments

    def build(self, model: NormalizedModel) -> dict[str, Any]:
        """Build a card dict from a normalized model + enrichments.

        Args:
            model: Normalized model from a provider adapter.

        Returns:
            Dict suitable for ``ModelCard.model_validate()``.
        """
        enrichment = self.enrichments.get(model.model_id)

        card: dict[str, Any] = {
            "model_id": model.model_id,
            "display_name": self._resolve_display_name(model, enrichment),
            "provider": model.provider,
            "model_type": model.model_type,
            "architecture": self._build_architecture(model, enrichment),
            "context": self._build_context(model, enrichment),
            "capabilities": self._build_capabilities(model),
            "pricing": self._build_pricing(model, enrichment),
            "lifecycle": self._build_lifecycle(model),
            "license": model.license,
            "open_weights": model.open_weights,
            "aliases": self._merge_aliases(model, enrichment),
            "description": enrichment.overrides.get("description", model.description)
            or f"{model.provider.capitalize()} model: {model.model_id}",
            "tags": self._merge_tags(model, enrichment),
            "source": "generated",
        }

        # Provider-specific extension
        ext = self._build_provider_extension(model, enrichment)
        if ext:
            ext_field = self._EXTENSION_FIELD_MAP.get(self.provider)
            if ext_field:
                card[ext_field] = ext
            else:
                card["provider_extension"] = ext

        return card

    # ------------------------------------------------------------------
    # Field builders
    # ------------------------------------------------------------------

    def _resolve_display_name(self, model: NormalizedModel, enrichment: ModelEnrichment) -> str:
        if "display_name" in enrichment.overrides:
            return enrichment.overrides["display_name"]
        if model.display_name:
            return model.display_name
        return model_id_to_display_name(model.model_id)

    def _build_architecture(
        self, model: NormalizedModel, enrichment: ModelEnrichment
    ) -> dict[str, Any] | None:
        arch: dict[str, Any] = {}

        # Start with API data
        if model.architecture_family:
            arch["family"] = model.architecture_family
        if model.parameter_count:
            arch["parameter_count"] = model.parameter_count
        if model.active_parameters:
            arch["active_parameters"] = model.active_parameters
        if model.architecture_type:
            arch["architecture_type"] = model.architecture_type

        # Overlay enrichment (enrichment wins)
        if enrichment.architecture:
            if "family" in enrichment.architecture:
                arch["family"] = enrichment.architecture["family"]
            if "type" in enrichment.architecture:
                arch["architecture_type"] = enrichment.architecture["type"]
            if "parameter_count" in enrichment.architecture:
                arch["parameter_count"] = enrichment.architecture["parameter_count"]
            if "active_parameters" in enrichment.architecture:
                arch["active_parameters"] = enrichment.architecture["active_parameters"]

        return arch if arch else None

    def _build_context(
        self, model: NormalizedModel, enrichment: ModelEnrichment
    ) -> dict[str, Any] | None:
        """Build the context block, or ``None`` when nothing states a window.

        There is deliberately **no default**. The history of this function is
        the argument for that: it defaulted to 4,096, which was far too low
        and produced broken context budgets at runtime, and was then changed
        to 128,000, which is far too low for some models and meaningless for
        others. Both were the same mistake -- answering with a plausible
        number instead of admitting the question had no answer.

        Measured before this change: **986 of 2,279 effective cards (43%)
        claimed exactly 128,000 tokens**, including 51 speech-to-text models
        whose input is audio and 124 image generators that have no token
        window at all. A card that says nothing is read as unknown; a card
        that says 128,000 is believed.
        """
        max_input = (
            enrichment.overrides.get("max_input_tokens")
            or enrichment.overrides.get("context_length")
            or model.context_length
        )
        if not max_input:
            return None
        ctx: dict[str, Any] = {
            "max_input_tokens": max_input,
        }
        max_output = enrichment.overrides.get("max_output_tokens") or model.max_output_tokens
        if max_output:
            ctx["max_output_tokens"] = max_output
        if model.default_output_tokens:
            ctx["default_output_tokens"] = model.default_output_tokens
        return ctx

    def _build_capabilities(self, model: NormalizedModel) -> dict[str, Any]:
        return {
            "streaming": model.supports_streaming,
            "function_calling": model.supports_tools,
            "tool_use": model.supports_tools,
            "json_mode": model.supports_json_mode,
            "structured_output": model.supports_structured_output,
            "vision": model.supports_vision,
            "audio_input": model.supports_audio_input,
            "audio_output": model.supports_audio_output,
            "video_input": model.supports_video_input,
            "reasoning": model.supports_reasoning,
            # Generative media. Without these a media card would claim no
            # capabilities at all, which is worse than having no card: it reads
            # as "this model does nothing" rather than "nobody filled this in".
            "image_generation": model.supports_image_generation,
            "image_edit": model.supports_image_edit,
            "image_upscale": model.supports_image_upscale,
            "video_generation": model.supports_video_generation,
            "video_interpolation": model.supports_video_interpolation,
            "speech_synthesis": model.supports_speech_synthesis,
            "transcription": model.supports_transcription,
            "music_generation": model.supports_music_generation,
            "sfx_generation": model.supports_sfx_generation,
            "voice_design": model.supports_voice_design,
        }

    def _build_pricing(
        self, model: NormalizedModel, enrichment: ModelEnrichment
    ) -> dict[str, Any] | None:
        """Build the pricing block for a card.

        Resolution order:

        1. Enrichment overlay (manually curated ``[pricing]`` TOML).
        2. Live API pricing stashed by the adapter in
           ``raw_api_data["_pricing"]`` (per-million USD).

        Enrichment wins so curated corrections survive regeneration.

        Args:
            model: The normalized model (may carry ``_pricing``).
            enrichment: The resolved enrichment overlay.

        Returns:
            A ``ModelPricing``-compatible dict, or ``None`` when no pricing is
            available (e.g. local/free models).
        """
        source = enrichment.pricing or model.raw_api_data.get("_pricing")
        if not source:
            return None
        per_million: dict[str, Any] = {
            "input": source.get("input", 0),
            "output": source.get("output", 0),
        }
        if source.get("cached_input") is not None:
            per_million["cached_input"] = source["cached_input"]
        return {
            "currency": "USD",
            "per_million_tokens": per_million,
        }

    def _build_lifecycle(self, model: NormalizedModel) -> dict[str, Any]:
        lifecycle: dict[str, Any] = {"status": model.status}
        if model.created_timestamp:
            try:
                dt = datetime.fromtimestamp(model.created_timestamp, tz=timezone.utc)
                lifecycle["release_date"] = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError, OSError):
                pass
        if model.is_deprecated:
            lifecycle["status"] = "deprecated"
        if model.deprecation_date:
            lifecycle["deprecation_date"] = model.deprecation_date
        return lifecycle

    def _merge_aliases(self, model: NormalizedModel, enrichment: ModelEnrichment) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for a in model.aliases + enrichment.extra_aliases:
            if a not in seen and a != model.model_id:
                seen.add(a)
                result.append(a)
        return result

    def _merge_tags(self, model: NormalizedModel, enrichment: ModelEnrichment) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for t in model.tags + enrichment.extra_tags:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _build_provider_extension(
        self, model: NormalizedModel, enrichment: ModelEnrichment
    ) -> dict[str, Any] | None:
        ext: dict[str, Any] = {}

        # Raw API data that adapters flag as extension-worthy
        if model.raw_api_data.get("_extension"):
            ext.update(model.raw_api_data["_extension"])

        # Enrichment overlay
        if enrichment.provider_extension:
            ext.update(enrichment.provider_extension)

        return ext if ext else None
