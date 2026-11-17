# tools/cardctl/adapters/deepinfra_adapter.py
"""DeepInfra model-discovery adapter for cardctl.

DeepInfra publishes a rich, **public** model catalog in the OpenRouter schema
at ``GET https://api.deepinfra.com/openrouter/models``.  Unlike the bare
OpenAI ``/v1/models`` shape, this endpoint exposes per-model:

* ``input_modalities`` / ``output_modalities`` (text, image, audio, video),
* ``supported_features`` (tools, response_format, reasoning, ...),
* ``context_length`` / ``max_output_length``,
* ``pricing`` (per-token ``prompt`` / ``completion`` / ``input_cache_read``),
* ``deprecation_date``, ``quantization`` and ``hugging_face_id``.

This adapter maps that into :class:`NormalizedModel` instances.  Live pricing is
stashed in ``raw_api_data["_pricing"]`` and DeepInfra-specific metadata in
``raw_api_data["_extension"]`` for :class:`~tools.cardctl.core.builder.CardBuilder`
to pick up.

The endpoint is public, so ``cardctl generate deepinfra`` works without a token.
An API key (``DEEPINFRA_TOKEN`` / ``DEEPINFRA_API_KEY``) is sent when available.

Reference: https://docs.deepinfra.com/api-reference/models/openrouter-models
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)


def _per_million(per_token_price: str | float | None) -> float:
    """Convert a per-token price string to USD per 1M tokens.

    Args:
        per_token_price: Price per single token (string or float).

    Returns:
        Price per 1,000,000 tokens, rounded to 6 dp; ``0.0`` on parse failure.
    """
    try:
        return round(float(per_token_price) * 1_000_000, 6)
    except (TypeError, ValueError):
        return 0.0


def _has_feature(features: list[str], *needles: str) -> bool:
    """Return True if any feature string contains any of *needles*.

    Args:
        features: The ``supported_features`` list (case-insensitive match).
        *needles: Substrings to look for.

    Returns:
        True if a match is found.
    """
    lowered = [str(f).lower() for f in (features or [])]
    return any(needle in feat for feat in lowered for needle in needles)


def _flatten_v24(m: dict[str, Any]) -> dict[str, Any]:
    """A schema-2.4 entry in the flat shape :meth:`_parse_model` reads.

    Since schema 2.4 (live 2026-11-16) modalities are objects: the context
    window sits in ``input_modalities[text].supported_inputs.max_context_length``,
    the parameters (``tools``, ``structured_outputs``, ``reasoning``, ...) and
    the max output in ``output_modalities[text]``, and prices are per-modality
    lists. Read flat, every model came out with no context, no tools, no
    reasoning and no price. Flat (pre-2.4) entries pass through unchanged.
    """
    inputs = m.get("input_modalities") or []
    outputs = m.get("output_modalities") or []
    if not any(isinstance(x, dict) for x in [*inputs, *outputs]):
        return m
    flat = dict(m)
    flat["input_modalities"] = [x.get("type") if isinstance(x, dict) else x for x in inputs]
    flat["output_modalities"] = [x.get("type") if isinstance(x, dict) else x for x in outputs]
    pricing: dict[str, Any] = {}
    features: list[str] = []
    for modality in inputs:
        if not isinstance(modality, dict):
            continue
        window = ((modality.get("supported_inputs") or {}).get("max_context_length") or {}).get("value")
        if modality.get("type") == "text" and window:
            flat.setdefault("context_length", int(window))
        for price in modality.get("pricing") or []:
            kind = {"prompt": "prompt", "cached_prompt": "input_cache_read"}.get(price.get("type"))
            if kind and modality.get("type") == "text":
                pricing.setdefault(kind, price.get("cost_usd"))
    for modality in outputs:
        if not isinstance(modality, dict):
            continue
        params = modality.get("supported_parameters") or {}
        features.extend(name for name, spec in params.items() if spec)
        if modality.get("type") == "text":
            if (modality.get("max_length") or {}).get("value"):
                flat.setdefault("max_output_length", int(modality["max_length"]["value"]))
            for price in modality.get("pricing") or []:
                if price.get("type") == "completion":
                    pricing.setdefault("completion", price.get("cost_usd"))
    flat["supported_features"] = list(dict.fromkeys([*(m.get("supported_features") or []), *features]))
    if pricing and not isinstance(m.get("pricing"), dict):
        flat["pricing"] = pricing
    return flat


class DeepInfraAdapter(BaseAdapter):
    """Adapter that discovers DeepInfra models via ``/openrouter/models``."""

    provider_name = "deepinfra"
    requires_api_key = False  # /openrouter/models is a public endpoint.
    api_key_env_var = "DEEPINFRA_TOKEN"
    base_url = "https://api.deepinfra.com"
    models_endpoint = "/openrouter/models"

    def get_api_key(self) -> str | None:
        """Resolve the API key, also honouring the ``DEEPINFRA_API_KEY`` alias.

        Returns:
            The resolved API key, or ``None`` if unset (endpoint is public).
        """
        key = super().get_api_key()
        if key:
            return key
        import os

        return os.environ.get("DEEPINFRA_API_KEY")

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch and normalize the DeepInfra model catalog.

        Returns:
            A list of :class:`NormalizedModel`.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx response.
        """
        url = f"{self.base_url.rstrip('/')}{self.models_endpoint}"
        headers: dict[str, str] = {}
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        result: list[NormalizedModel] = []
        seen: set[str] = set()
        for m in data.get("data", []):
            # The feed repeats models (519 rows, 352 ids on 2026-11-16):
            # one card per id, from its first row.
            if m.get("id") in seen:
                continue
            seen.add(m.get("id"))
            normalized = self._parse_model(m)
            if normalized:
                result.append(normalized)

        logger.info("Fetched %d models from deepinfra", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        """Convert one OpenRouter-format model entry to a NormalizedModel.

        Args:
            m: A single ``data[]`` entry from ``/openrouter/models``.

        Returns:
            A :class:`NormalizedModel`, or ``None`` if the entry has no ID.
        """
        model_id = m.get("id", "")
        if not model_id:
            return None
        m = _flatten_v24(m)

        input_mods = [str(x).lower() for x in (m.get("input_modalities") or [])]
        output_mods = [str(x).lower() for x in (m.get("output_modalities") or [])]
        features = m.get("supported_features") or []

        model_type = self._infer_model_type(input_mods, output_mods)

        supports_vision = "image" in input_mods
        supports_audio_input = "audio" in input_mods
        supports_video_input = "video" in input_mods
        supports_audio_output = "audio" in output_mods
        # "tool_choice" alone does not mean tools (2.4 lists it as "unknown").
        supports_tools = "tools" in features or _has_feature(features, "function")
        supports_json = _has_feature(features, "json", "response_format")
        supports_structured = _has_feature(features, "structured", "json_schema", "response_format")
        supports_reasoning = _has_feature(features, "reasoning", "thinking")

        deprecation_date = m.get("deprecation_date")
        normalized = NormalizedModel(
            model_id=model_id,
            provider="deepinfra",
            display_name=m.get("name"),
            description=m.get("description"),
            model_type=model_type,
            context_length=m.get("context_length"),
            max_output_tokens=m.get("max_output_length"),
            supports_streaming=True,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_video_input=supports_video_input,
            supports_audio_input=supports_audio_input,
            supports_audio_output=supports_audio_output,
            supports_json_mode=supports_json,
            supports_structured_output=supports_structured,
            supports_reasoning=supports_reasoning,
            is_deprecated=bool(deprecation_date),
            deprecation_date=deprecation_date,
            created_timestamp=m.get("created"),
            open_weights=True,  # DeepInfra serves open-weight models.
            tags=list(input_mods) + [f"output:{o}" for o in output_mods],
            raw_api_data=m,
        )

        # Live pricing (per-token strings -> per-million USD).
        pricing = m.get("pricing") or {}
        input_price = _per_million(pricing.get("prompt"))
        output_price = _per_million(pricing.get("completion"))
        cached_price = _per_million(pricing.get("input_cache_read"))
        if input_price or output_price:
            normalized.raw_api_data["_pricing"] = {
                "input": input_price,
                "output": output_price,
            }
            if cached_price:
                normalized.raw_api_data["_pricing"]["cached_input"] = cached_price

        # DeepInfra-specific extension metadata.
        extension: dict[str, Any] = {}
        if m.get("hugging_face_id"):
            extension["hugging_face_id"] = m["hugging_face_id"]
        if m.get("quantization"):
            extension["quantization"] = m["quantization"]
        if input_mods:
            extension["input_modalities"] = input_mods
        if output_mods:
            extension["output_modalities"] = output_mods
        if features:
            extension["supported_features"] = list(features)
        datacenters = [
            dc.get("country_code")
            for dc in (m.get("datacenters") or [])
            if isinstance(dc, dict) and dc.get("country_code")
        ]
        if datacenters:
            extension["datacenters"] = datacenters
        if extension:
            normalized.raw_api_data["_extension"] = extension

        return normalized

    @staticmethod
    def _infer_model_type(input_mods: list[str], output_mods: list[str]) -> str:
        """Infer the card ``model_type`` from input/output modalities.

        Args:
            input_mods: Lower-cased input modality list.
            output_mods: Lower-cased output modality list.

        Returns:
            One of ``chat``, ``image-generation`` or ``tts``.
        """
        has_text_out = "text" in output_mods
        if "image" in output_mods and not has_text_out:
            return "image-generation"
        if "audio" in output_mods and not has_text_out:
            return "tts"
        return "chat"
