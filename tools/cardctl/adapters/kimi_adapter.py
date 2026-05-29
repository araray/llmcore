# tools/cardctl/adapters/kimi_adapter.py
"""Kimi (Moonshot AI) model-discovery adapter.

Canonical provider key: ``kimi`` (matches ``llmcore`` provider ``get_name()``,
the ``Provider.KIMI`` enum, and the ``default_cards/kimi/`` directory).

Unlike vanilla OpenAI-compatible ``/v1/models`` responses, the Moonshot
endpoint returns rich capability metadata which this adapter maps onto the
:class:`NormalizedModel`:

    - ``context_length``      → ``context_length``
    - ``supports_image_in``   → ``supports_vision``
    - ``supports_video_in``   → ``supports_video_input``
    - ``supports_reasoning``  → ``supports_reasoning``

The international platform (``https://api.moonshot.ai/v1``) is used by default;
override with ``--api-key`` / ``base_url`` for the China platform
(``https://api.moonshot.cn/v1``).  Keys are not interchangeable between the two.
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

#: Preview/snapshot model ids retired on 2026-05-25.
_DEPRECATED_PREFIXES = (
    "kimi-k2-0905",
    "kimi-k2-0711",
    "kimi-k2-turbo",
    "kimi-k2-thinking",
    "kimi-latest",
    "kimi-thinking-preview",
    "moonshot-v1-auto",  # convenience router, not a distinct model card
)


class KimiAdapter(OpenAICompatAdapter):
    """Adapter for the Kimi Open Platform (Moonshot AI)."""

    provider_name = "kimi"
    api_key_env_var = "MOONSHOT_API_KEY"
    base_url = "https://api.moonshot.ai/v1"

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        return mid.startswith("kimi") or mid.startswith("moonshot")

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        mid = normalized.model_id

        # Capability metadata exposed by the Moonshot /v1/models endpoint.
        if raw.get("context_length"):
            normalized.context_length = int(raw["context_length"])
        normalized.supports_vision = bool(raw.get("supports_image_in", False))
        normalized.supports_video_input = bool(raw.get("supports_video_in", False))
        normalized.supports_reasoning = bool(raw.get("supports_reasoning", False))

        # All current Kimi/Moonshot chat models support tools + JSON modes.
        normalized.supports_tools = True
        normalized.supports_json_mode = True
        normalized.supports_structured_output = True

        # Architecture defaults (enrichment TOML may override per-model).
        normalized.architecture_family = "kimi" if mid.startswith("kimi") else "moonshot"
        normalized.architecture_type = "moe" if mid.startswith("kimi-k2") else "transformer"

        # Lifecycle.
        if any(mid.startswith(p) for p in _DEPRECATED_PREFIXES):
            normalized.is_deprecated = True
            normalized.status = "deprecated"

        # Stash raw flags for the generic provider_extension overlay.
        normalized.raw_api_data.setdefault("_extension", {})
        normalized.raw_api_data["_extension"].update(
            {
                "supports_video_in": normalized.supports_video_input,
                "thinking_mode": _model_supports_thinking(mid),
            }
        )
        return normalized


def _model_supports_thinking(mid: str) -> bool:
    """Whether the ``thinking`` request field applies to *mid* (k2.5/k2.6)."""
    m = mid.lower()
    return m.startswith("kimi-k2.5") or m.startswith("kimi-k2.6")
