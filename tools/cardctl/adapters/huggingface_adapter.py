# tools/cardctl/adapters/huggingface_adapter.py
"""Hugging Face model discovery adapter for cardctl.

Uses the HF Hub API to discover models served by the Inference API,
filtering for chat, embedding, TTS, ASR, and image-generation models.

The HF Hub hosts hundreds of thousands of models.  This adapter focuses
on models that are actively served by the HF Inference API (``inference=warm``
filter), which is the subset usable through the llmcore HuggingFace provider.

Requires: ``pip install huggingface_hub httpx``
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

# Mapping from HF pipeline_tag to our model_type
_PIPELINE_TAG_MAP: dict[str, str] = {
    "text-generation": "chat",
    "text2text-generation": "chat",
    "conversational": "chat",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "automatic-speech-recognition": "stt",
    "text-to-speech": "tts",
    "text-to-audio": "tts",
    "text-to-image": "image-generation",
    "image-to-text": "vision",
    "image-text-to-text": "vision",
    "visual-question-answering": "vision",
    "document-question-answering": "vision",
    "image-classification": "image-classification",
    "object-detection": "object-detection",
    "image-segmentation": "image-segmentation",
    "translation": "translation",
    "summarization": "summarization",
    "text-classification": "text-classification",
    "token-classification": "ner",
    "fill-mask": "fill-mask",
    "zero-shot-classification": "zero-shot",
    "question-answering": "qa",
    "table-question-answering": "table-qa",
    "depth-estimation": "depth-estimation",
    "image-to-image": "image-to-image",
    "image-to-video": "image-to-video",
    "text-to-video": "text-to-video",
}

# Families known to support tool calling via chat template
_TOOL_CALLING_FAMILIES = {
    "llama", "qwen", "qwen2", "mistral", "mixtral",
    "deepseek", "phi", "gemma", "command-r",
}

# Popular model IDs to prioritize (these are typically served warm)
_PRIORITY_MODELS = {
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/QwQ-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-V3-0324",
    "microsoft/phi-4",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "facebook/mms-tts-eng",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
}


class HuggingFaceAdapter(BaseAdapter):
    """Cardctl adapter for discovering models via HuggingFace Hub API."""

    provider_name = "huggingface"
    api_key_env_var = "HF_TOKEN"
    base_url = "https://huggingface.co/api"
    requires_api_key = True

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        headers = {"Authorization": f"Bearer {api_key}"}
        result: list[NormalizedModel] = []

        # Fetch models served by the Inference API (warm)
        # We query multiple pipeline tags to get a good cross-section
        pipeline_tags = [
            "text-generation",
            "feature-extraction",
            "automatic-speech-recognition",
            "text-to-speech",
            "text-to-image",
            "image-text-to-text",
        ]

        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            seen_ids: set[str] = set()

            for tag in pipeline_tags:
                try:
                    url = f"{self.base_url}/models"
                    params = {
                        "pipeline_tag": tag,
                        "inference": "warm",
                        "sort": "downloads",
                        "direction": "-1",
                        "limit": "50",
                    }
                    resp = await client.get(
                        url, headers=headers, params=params
                    )
                    resp.raise_for_status()
                    models = resp.json()

                    for m in models:
                        model_id = m.get("modelId") or m.get("id", "")
                        if not model_id or model_id in seen_ids:
                            continue
                        seen_ids.add(model_id)

                        normalized = self._parse_model(m, tag)
                        if normalized:
                            result.append(normalized)

                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "Failed to fetch %s models: %s", tag, e
                    )
                except Exception as e:
                    logger.warning(
                        "Error fetching %s models: %s", tag, e
                    )

            # Also try to fetch info for priority models not yet seen
            for priority_id in _PRIORITY_MODELS:
                if priority_id in seen_ids:
                    continue
                try:
                    url = f"{self.base_url}/models/{priority_id}"
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        m = resp.json()
                        ptag = m.get("pipeline_tag", "text-generation")
                        normalized = self._parse_model(m, ptag)
                        if normalized:
                            result.append(normalized)
                            seen_ids.add(priority_id)
                except Exception:
                    pass

        logger.info("Fetched %d models from huggingface", len(result))
        return result

    def _parse_model(
        self, m: dict[str, Any], pipeline_tag: str
    ) -> NormalizedModel | None:
        """Parse a HuggingFace Hub API model response into NormalizedModel."""
        model_id = m.get("modelId") or m.get("id", "")
        if not model_id:
            return None

        # Determine model type from pipeline tag
        p_tag = m.get("pipeline_tag", pipeline_tag) or pipeline_tag
        model_type = _PIPELINE_TAG_MAP.get(p_tag, "chat")

        # Extract tags
        hf_tags = m.get("tags", []) or []

        # Detect capabilities from tags and model architecture
        is_chat = model_type == "chat"
        family = self._detect_family(model_id, hf_tags)
        supports_tools = is_chat and family in _TOOL_CALLING_FAMILIES
        supports_vision = (
            model_type == "vision"
            or "vision" in hf_tags
            or p_tag in ("image-text-to-text", "visual-question-answering")
        )

        # Context length heuristic from safetensors config
        config = m.get("config", {}) or {}
        ctx_len = None
        if isinstance(config, dict):
            # Try transformers config
            text_config = config.get("text_config", config)
            ctx_len = (
                text_config.get("max_position_embeddings")
                or text_config.get("max_sequence_length")
                or text_config.get("n_positions")
            )

        # Detect parameter count from tags/safetensors
        param_count = self._detect_param_count(m, hf_tags)

        # Lifecycle
        is_deprecated = "deprecated" in hf_tags
        created_at = m.get("createdAt") or m.get("created_at")

        normalized = NormalizedModel(
            model_id=model_id,
            provider="huggingface",
            display_name=model_id.split("/")[-1] if "/" in model_id else model_id,
            description=m.get("description", ""),
            model_type=model_type,
            context_length=ctx_len,
            supports_streaming=is_chat,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_audio_input=p_tag in (
                "automatic-speech-recognition", "audio-classification"
            ),
            supports_audio_output=p_tag in ("text-to-speech", "text-to-audio"),
            architecture_family=family,
            parameter_count=param_count,
            owned_by=model_id.split("/")[0] if "/" in model_id else None,
            is_deprecated=is_deprecated,
            open_weights=True,  # All HF Hub models are open-weight
            license=m.get("license"),
            tags=self._build_tags(model_type, supports_vision, supports_tools, hf_tags),
            raw_api_data=m,
        )

        return normalized

    def _detect_family(
        self, model_id: str, tags: list[str]
    ) -> str | None:
        """Detect model family from ID and tags."""
        model_lower = model_id.lower()
        for family in (
            "llama", "qwen", "mistral", "mixtral", "gemma",
            "phi", "deepseek", "falcon", "mpt", "bloom",
            "starcoder", "codellama", "vicuna", "whisper",
            "bert", "roberta", "flux", "stable-diffusion",
        ):
            if family in model_lower:
                return family
        return None

    def _detect_param_count(
        self, m: dict[str, Any], tags: list[str]
    ) -> str | None:
        """Try to detect parameter count from metadata."""
        # Check safetensors metadata
        safetensors = m.get("safetensors", {})
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if total and isinstance(total, (int, float)):
                if total >= 1e9:
                    return f"{total / 1e9:.0f}B"
                elif total >= 1e6:
                    return f"{total / 1e6:.0f}M"

        # Heuristic from model name
        model_id = m.get("modelId", "") or m.get("id", "")
        model_lower = model_id.lower()
        for suffix in (
            "405b", "236b", "180b", "141b", "123b", "110b",
            "72b", "70b", "65b", "34b", "32b", "30b", "27b",
            "22b", "14b", "13b", "12b", "11b", "8b", "7b",
            "4b", "3b", "2b", "1.5b", "1b", "500m", "350m",
            "125m", "82m",
        ):
            if suffix in model_lower:
                return suffix.upper()

        return None

    def _build_tags(
        self,
        model_type: str,
        vision: bool,
        tools: bool,
        hf_tags: list[str],
    ) -> list[str]:
        tags = ["open-weights"]
        if vision:
            tags.append("multimodal")
            tags.append("vision")
        if tools:
            tags.append("tool-calling")
        if model_type == "embedding":
            tags.append("embedding")
        if model_type in ("stt", "tts"):
            tags.append("audio")
            tags.append(model_type)
        if model_type == "image-generation":
            tags.append("image-generation")
        if any("reasoning" in t.lower() for t in hf_tags):
            tags.append("reasoning")
        return tags
