# tools/cardctl/adapters/fal_adapter.py
"""fal.ai adapter — curated, because the model *is* the endpoint path.

fal hosts thousands of models and publishes no catalog route, so these are the
per-capability defaults the llmcore fal provider ships (see
``src/llmcore/providers/fal_provider.py``). Override any of them in
``[providers.fal.models]``; this adapter documents the defaults.
"""

from __future__ import annotations

from .curated import CuratedAdapter, CuratedModel


class FalAdapter(CuratedAdapter):
    """Cards for fal's default per-capability endpoints."""

    provider_name = "fal"
    api_key_env_var = "FAL_KEY"
    base_url = "https://queue.fal.run"

    curated_models = (
        CuratedModel(
            model_id="fal-ai/flux/schnell",
            model_type="image-generation",
            capability="image_generate",
            display_name="FLUX.1 [schnell] (fal)",
            description="Fast, low-cost text-to-image. llmcore's default fal image model.",
            caps=("image_generation",),
            tags=("image", "flux", "fast"),
            owned_by="black-forest-labs",
        ),
        CuratedModel(
            model_id="fal-ai/flux-pro/kontext",
            model_type="image-generation",
            capability="image_edit",
            display_name="FLUX.1 Kontext [pro] (fal)",
            description="Instruction-driven image editing.",
            caps=("image_edit",),
            tags=("image", "editing", "flux"),
            owned_by="black-forest-labs",
        ),
        CuratedModel(
            model_id="fal-ai/clarity-upscaler",
            model_type="image-generation",
            capability="image_upscale",
            display_name="Clarity Upscaler (fal)",
            description="Image upscaling and detail enhancement.",
            caps=("image_upscale",),
            tags=("image", "upscale"),
        ),
        CuratedModel(
            model_id="fal-ai/minimax-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="MiniMax Video (fal)",
            description="Text-to-video generation.",
            caps=("video_generation",),
            tags=("video",),
            owned_by="minimax",
        ),
        CuratedModel(
            model_id="fal-ai/film",
            model_type="video-generation",
            capability="video_interpolate",
            display_name="FILM (fal)",
            description=(
                "Frame interpolation between two stills. Takes an explicit "
                "start/end image pair rather than a frame list."
            ),
            caps=("video_interpolation",),
            tags=("video", "interpolation"),
        ),
        CuratedModel(
            model_id="fal-ai/mmaudio-v2",
            model_type="audio",
            capability="sfx",
            display_name="MMAudio v2 (fal)",
            description="Sound effects, optionally conditioned on video (foley).",
            caps=("sfx_generation",),
            tags=("audio", "sfx", "foley"),
        ),
        CuratedModel(
            model_id="fal-ai/stable-audio",
            model_type="audio",
            capability="music",
            display_name="Stable Audio (fal)",
            description="Text-to-music generation.",
            caps=("music_generation",),
            tags=("audio", "music"),
            owned_by="stability-ai",
        ),
        CuratedModel(
            model_id="fal-ai/kokoro",
            model_type="tts",
            capability="tts",
            display_name="Kokoro TTS (fal)",
            description="Text-to-speech.",
            caps=("speech_synthesis",),
            tags=("audio", "tts"),
        ),
        CuratedModel(
            model_id="fal-ai/whisper",
            model_type="stt",
            capability="asr",
            display_name="Whisper (fal)",
            description="Speech-to-text transcription.",
            caps=("transcription",),
            tags=("audio", "stt"),
            owned_by="openai",
        ),
    )
