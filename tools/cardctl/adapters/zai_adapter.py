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

# GLM model metadata keyed by model_id prefix (longest match wins).
_GLM_MODELS: dict[str, dict[str, Any]] = {
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

        # Longest-prefix match against the metadata table.
        meta = None
        for prefix in sorted(_GLM_MODELS, key=len, reverse=True):
            if mid.startswith(prefix):
                meta = _GLM_MODELS[prefix]
                break

        if meta:
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
