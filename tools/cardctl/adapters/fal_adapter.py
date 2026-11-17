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
        # fal-ai/minimax-video is gone (404 on fal's Platform API, 2026-11-16).
        CuratedModel(
            model_id="fal-ai/minimax/hailuo-2.3/standard/text-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="MiniMax Hailuo 2.3 Standard, text-to-video (fal)",
            description=(
                "Text-to-video. Duration is \"6\" or \"10\" seconds (a string "
                "enum; llmcore rounds duration_seconds onto it)."
            ),
            caps=("video_generation",),
            tags=("video",),
            owned_by="minimax",
        ),
        CuratedModel(
            model_id="fal-ai/minimax/hailuo-2.3/standard/image-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="MiniMax Hailuo 2.3 Standard, image-to-video (fal)",
            description=(
                "Image-to-video from a first frame. llmcore routes a "
                "first_frame request here when the default video endpoint is in use."
            ),
            caps=("video_generation",),
            tags=("video", "image-to-video"),
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
            description="Foley: sound effects conditioned on a video (video_url is required).",
            caps=("sfx_generation",),
            tags=("audio", "sfx", "foley"),
        ),
        CuratedModel(
            model_id="fal-ai/mmaudio-v2/text-to-audio",
            model_type="audio",
            capability="sfx",
            display_name="MMAudio v2, text-to-audio (fal)",
            description="Sound effects from text alone; llmcore's route when no video is given.",
            caps=("sfx_generation",),
            tags=("audio", "sfx"),
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
            # The bare fal-ai/kokoro id has no schema; the language variants do.
            model_id="fal-ai/kokoro/american-english",
            model_type="tts",
            capability="tts",
            display_name="Kokoro TTS, American English (fal)",
            description="Text-to-speech (other languages: fal-ai/kokoro/<language>).",
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
