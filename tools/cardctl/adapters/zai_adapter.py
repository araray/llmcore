"""Z.ai (Zhipu AI / GLM) model discovery adapter (OpenAI-compatible).

Fetches models from the Z.ai Open Platform ``/models`` endpoint and enriches
them with GLM-specific metadata (MoE architecture, thinking mode, context
length, vision capability).

GLM models (2026-06):
- ``glm-5.2`` / ``glm-5.1`` / ``glm-5`` — 1M context, thinking + tools
- ``glm-5v-turbo`` — vision, 64K context
- ``glm-4.7`` / ``glm-4.6`` / ``glm-4.5`` — 128K–200K context
- ``glm-4.6v`` / ``glm-4.5v`` — vision-language models
- ``embedding-3`` / ``embedding-2`` — text embeddings

Service docs: https://docs.z.ai/

Canonical provider key: ``zai`` (default_cards/zai/).  Also reachable via the
``glm`` / ``zhipu`` / ``zhipuai`` / ``bigmodel`` aliases in the registry.
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

# GLM model metadata keyed by model_id prefix. The longest match wins, and it
# inherits fields from the family entries it extends (see ``_glm_metadata``).
_GLM_MODELS: dict[str, dict[str, Any]] = {
    # GLM-5.3 (2026-08-18) always thinks; the Flash/FlashX models take image,
    # video and file input (docs.z.ai, 2026-11-16).
    "glm-5.3-flash": {"architecture_type": "moe", "reasoning": True, "vision": True},
    "glm-5.3": {"context_length": 1_000_000, "architecture_type": "moe", "reasoning": True},
    "glm-5.2": {"context_length": 1_000_000, "architecture_type": "moe", "reasoning": True},
    "glm-5.1": {"context_length": 1_000_000, "architecture_type": "moe", "reasoning": True},
    "glm-5v-turbo": {"context_length": 65_536, "architecture_type": "moe", "vision": True},
    "glm-5-turbo": {"context_length": 1_000_000, "architecture_type": "moe", "reasoning": True},
    "glm-5": {"context_length": 1_000_000, "architecture_type": "moe", "reasoning": True},
    "glm-4.7": {"context_length": 131_072, "architecture_type": "moe", "reasoning": True},
    "glm-4.6v": {"context_length": 65_536, "architecture_type": "moe", "vision": True},
    "glm-4.6": {"context_length": 204_800, "architecture_type": "moe", "reasoning": True},
    "glm-4.5v": {"context_length": 65_536, "architecture_type": "moe", "vision": True},
    "glm-4.5": {"context_length": 131_072, "architecture_type": "moe", "reasoning": True},
}


def _glm_metadata(model_id: str) -> dict[str, Any]:
    """Metadata for *model_id*: the longest matching prefix plus its family.

    A narrow entry such as ``glm-5.3-flash`` only states what differs from its
    family and inherits the rest (``context_length`` among it) from the
    shorter entries it extends at a ``-`` boundary (``glm-5.3``). Siblings
    that merely share characters (``glm-5v-turbo`` and ``glm-5``) do not
    inherit from each other.
    """
    matches = [p for p in _GLM_MODELS if model_id.startswith(p)]
    if not matches:
        return {}
    longest = max(matches, key=len)
    meta: dict[str, Any] = {}
    for prefix in sorted(_GLM_MODELS, key=len):
        if longest == prefix or longest.startswith(prefix + "-"):
            meta.update(_GLM_MODELS[prefix])
    return meta


class ZaiAdapter(OpenAICompatAdapter):
    provider_name = "zai"
    api_key_env_var = "ZAI_API_KEY"
    base_url = "https://api.z.ai/api/paas/v4"

    #: GLM media models surfaced through dedicated media APIs, not chat cards.
    _MEDIA_SUFFIXES: tuple[str, ...] = ("image", "ocr", "tts", "asr", "voice", "video")

    def _include_model(self, model: dict[str, Any]) -> bool:
        mid = model.get("id", "")
        if mid.startswith("embedding"):
            return True
        if not mid.startswith("glm"):
            return False
        # Skip image/video/audio/OCR generation models (e.g. glm-image,
        # glm-ocr, glm-tts, glm-asr) — those are media APIs, not chat cards.
        tail = mid.split("-", 1)[-1] if "-" in mid else ""
        return not any(s in tail for s in self._MEDIA_SUFFIXES)

    def _enrich_model(self, normalized: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        mid = normalized.model_id
        normalized.architecture_family = "GLM"

        if mid.startswith("embedding"):
            normalized.model_type = "embedding"
            normalized.context_length = normalized.context_length or 8_192
            normalized.supports_streaming = False
            normalized.tags.append("embedding")
            return normalized

        # Chat / vision models support tools + JSON mode.
        normalized.supports_tools = True
        normalized.supports_json_mode = True

        meta = _glm_metadata(mid)
        if meta:
            if meta.get("context_length"):
                normalized.context_length = meta["context_length"]
            normalized.architecture_type = meta.get("architecture_type", "moe")
            if meta.get("reasoning"):
                normalized.supports_reasoning = True
                normalized.tags.append("reasoning")
            if meta.get("vision"):
                normalized.supports_vision = True
                normalized.tags.append("vision")
        else:
            normalized.architecture_type = "moe"

        if mid.startswith("glm-5"):
            normalized.tags.extend(["glm-5", "million-context"])

        return normalized
