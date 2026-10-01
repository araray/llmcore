# tools/cardctl/adapters/replicate_adapter.py
"""Replicate adapter — curated set, enriched from each model's live schema.

Replicate hosts tens of thousands of community models. Enumerating them would
produce a card dump rather than a catalog, so the curated set is the
per-capability defaults the llmcore Replicate provider ships.

What makes these cards more than a hand-written list is the enrichment: each
entry is looked up on Replicate, which publishes the model's own OpenAPI schema,
so the card records the **real input field names** and the description the owner
wrote. That is the same schema the provider uses at runtime to map canonical
arguments onto whatever a model calls them, so the card documents what will
actually be sent.
"""

from __future__ import annotations

import logging

import httpx

from .base import NormalizedModel
from .curated import CuratedAdapter, CuratedModel

logger = logging.getLogger(__name__)


class ReplicateAdapter(CuratedAdapter):
    """Cards for Replicate's default per-capability models."""

    provider_name = "replicate"
    api_key_env_var = "REPLICATE_API_TOKEN"
    base_url = "https://api.replicate.com/v1"
    # A token is optional: without one the curated entries are still written,
    # just without live descriptions or input schemas.
    requires_api_key = False

    curated_models = (
        CuratedModel(
            model_id="black-forest-labs/flux-schnell",
            model_type="image-generation",
            capability="image_generate",
            display_name="FLUX.1 [schnell] (Replicate)",
            caps=("image_generation",),
            tags=("image", "flux"),
            owned_by="black-forest-labs",
        ),
        CuratedModel(
            model_id="black-forest-labs/flux-kontext-pro",
            model_type="image-generation",
            capability="image_edit",
            display_name="FLUX.1 Kontext [pro] (Replicate)",
            caps=("image_edit",),
            tags=("image", "editing", "flux"),
            owned_by="black-forest-labs",
        ),
        CuratedModel(
            model_id="nightmareai/real-esrgan",
            model_type="image-generation",
            capability="image_upscale",
            display_name="Real-ESRGAN (Replicate)",
            caps=("image_upscale",),
            tags=("image", "upscale"),
        ),
        CuratedModel(
            model_id="minimax/video-01",
            model_type="video-generation",
            capability="video_generate",
            display_name="MiniMax video-01 (Replicate)",
            caps=("video_generation",),
            tags=("video",),
            owned_by="minimax",
        ),
        CuratedModel(
            model_id="openai/whisper",
            model_type="stt",
            capability="asr",
            display_name="Whisper (Replicate)",
            caps=("transcription",),
            tags=("audio", "stt"),
            owned_by="openai",
            notes={
                "note": (
                    "A community model rather than an official one, so it runs by "
                    "version pin at /v1/predictions. Returns an OBJECT with a "
                    "'transcription' field, not a list of URLs."
                )
            },
        ),
        CuratedModel(
            model_id="jaaari/kokoro-82m",
            model_type="tts",
            capability="tts",
            display_name="Kokoro 82M (Replicate)",
            caps=("speech_synthesis",),
            tags=("audio", "tts"),
        ),
        CuratedModel(
            model_id="meta/musicgen",
            model_type="audio",
            capability="music",
            display_name="MusicGen (Replicate)",
            caps=("music_generation",),
            tags=("audio", "music"),
            owned_by="meta",
        ),
    )

    async def enrich(self, model: NormalizedModel, entry: CuratedModel) -> NormalizedModel:
        """Add the owner's description and the model's real input field names.

        A lookup failure is logged and the curated entry kept: the card is still
        correct about what llmcore will call, and losing the enrichment is not a
        reason to lose the card.
        """
        token = self.get_api_key()
        if not token:
            logger.info(
                "No %s set; writing curated Replicate cards without live schemas.",
                self.api_key_env_var,
            )
            return model

        url = f"{self.base_url.rstrip('/')}/models/{entry.model_id}"
        try:
            async with httpx.AsyncClient(timeout=40, follow_redirects=True) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            if resp.status_code >= 400:
                logger.warning(
                    "Replicate lookup for %s returned %s; keeping the curated entry.",
                    entry.model_id,
                    resp.status_code,
                )
                return model
            payload = resp.json()
        except Exception as e:  # enrichment is optional, the card is not
            logger.warning("Replicate lookup for %s failed: %s", entry.model_id, e)
            return model

        if payload.get("description"):
            model.description = str(payload["description"])
        version = payload.get("latest_version") or {}
        schemas = (version.get("openapi_schema") or {}).get("components", {}).get(
            "schemas", {}
        )
        inputs = (schemas.get("Input") or {}).get("properties") or {}
        if inputs:
            # The same schema the provider reads at runtime to map canonical
            # arguments onto this model's field names, so the card documents
            # what will actually be sent.
            model.raw_api_data["input_fields"] = sorted(inputs)
            model.raw_api_data["required_inputs"] = (
                (schemas.get("Input") or {}).get("required") or []
            )
        if version.get("id"):
            model.raw_api_data["latest_version"] = version["id"]
        if payload.get("owner"):
            model.owned_by = str(payload["owner"])
        return model
