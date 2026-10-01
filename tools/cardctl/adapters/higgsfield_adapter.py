# tools/cardctl/adapters/higgsfield_adapter.py
"""Higgsfield adapter — curated from the published OpenAPI paths.

Higgsfield addresses models by endpoint path and publishes no catalog route; its
own documentation says to treat ``/docs/openapi.json`` as supplementary rather
than authoritative, and points at the console for the real list. These entries
are the paths the llmcore provider ships as defaults plus the other generation
routes the OpenAPI document declares.
"""

from __future__ import annotations

from .curated import CuratedAdapter, CuratedModel


class HiggsfieldAdapter(CuratedAdapter):
    """Cards for Higgsfield's documented generation endpoints."""

    provider_name = "higgsfield"
    api_key_env_var = "HIGGSFIELD_API_KEY"
    base_url = "https://api.higgsfield.ai"

    curated_models = (
        CuratedModel(
            model_id="higgsfield-ai/soul/standard",
            model_type="image-generation",
            capability="image_generate",
            display_name="Higgsfield Soul (standard)",
            description=(
                "Higgsfield's own text-to-image family. llmcore's default "
                "Higgsfield image model. Takes prompt, num_images, resolution "
                "and aspect_ratio."
            ),
            caps=("image_generation",),
            tags=("image", "soul"),
            owned_by="higgsfield",
        ),
        CuratedModel(
            model_id="minimax/hailuo-2.3/standard/text-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="Hailuo 2.3 standard (text-to-video)",
            description=(
                "Text-to-video. llmcore's default Higgsfield video model. "
                "Takes prompt, duration and prompt_optimizer."
            ),
            caps=("video_generation",),
            tags=("video", "hailuo"),
            owned_by="minimax",
        ),
        CuratedModel(
            model_id="minimax/hailuo-2.3/standard/image-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="Hailuo 2.3 standard (image-to-video)",
            description=(
                "Image-conditioned video. llmcore routes here automatically when "
                "a conditioning image is supplied to the text-to-video default."
            ),
            caps=("video_generation",),
            tags=("video", "hailuo", "image-to-video"),
            owned_by="minimax",
        ),
        CuratedModel(
            model_id="kling-video/v2.5-turbo/pro/text-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="Kling 2.5 Turbo Pro (text-to-video)",
            description="Text-to-video. Takes prompt, duration, cfg_scale, negative_prompt.",
            caps=("video_generation",),
            tags=("video", "kling", "pro"),
            owned_by="kuaishou",
        ),
        CuratedModel(
            model_id="kling-video/v2.5-turbo/pro/image-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="Kling 2.5 Turbo Pro (image-to-video)",
            description="Image-conditioned video at the pro tier.",
            caps=("video_generation",),
            tags=("video", "kling", "pro", "image-to-video"),
            owned_by="kuaishou",
        ),
        CuratedModel(
            model_id="kling-video/v2.5-turbo/standard/image-to-video",
            model_type="video-generation",
            capability="video_generate",
            display_name="Kling 2.5 Turbo Standard (image-to-video)",
            description="Image-conditioned video at the standard tier.",
            caps=("video_generation",),
            tags=("video", "kling", "image-to-video"),
            owned_by="kuaishou",
        ),
    )
