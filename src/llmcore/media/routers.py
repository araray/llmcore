# src/llmcore/media/routers.py
"""Per-modality routers for the LLMCore media subsystem.

A router is the caller-facing surface for one modality.  Its only jobs are to
resolve a capability to an adapter, call the protocol method, and hand any
returned job to the job manager so an expensive submission is never lost.  All
vendor logic lives in the adapters; all selection logic lives in
:class:`~llmcore.media.manager.MediaManager`.

Every router method accepts ``provider=`` and ``model=`` to pin the route
explicitly, and forwards unknown keyword arguments to the adapter so
vendor-specific parameters need no plumbing here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from .models import MediaCapability, MediaJob, MediaRef, MediaResult

if TYPE_CHECKING:  # pragma: no cover
    from .manager import MediaManager

logger = logging.getLogger(__name__)

__all__ = ["AudioRouter", "ImageRouter", "VideoRouter"]


class _BaseRouter:
    """Shared dispatch for the modality routers.

    Args:
        manager: The owning media manager, used for resolution and job tracking.
    """

    def __init__(self, manager: MediaManager) -> None:
        self._manager = manager

    async def _dispatch(
        self,
        capability: MediaCapability,
        method: str,
        *args: Any,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Resolve *capability* and invoke *method* on the chosen adapter.

        Args:
            capability: The operation being requested.
            method: Protocol method name to call on the adapter.
            *args: Positional arguments for the adapter method.
            provider: Pin the provider instance.
            model: Pin the model; also used to resolve the provider.
            **kwargs: Forwarded to the adapter, vendor parameters included.

        Returns:
            The adapter's result, with any job handle registered for tracking.
        """
        adapter = self._manager.resolve(capability, provider=provider, model=model)
        fn = getattr(adapter, method)

        # A callback URL is offered only to adapters that opt in. Most media
        # adapters forward unknown keyword arguments straight into the vendor
        # payload, so handing one to a provider that does not understand it
        # would post our callback URL to a model as a generation parameter.
        reserved = None
        if getattr(adapter, "accepts_webhook_url", False):
            reserved = self._manager.jobs.webhooks.reserve()
            if reserved is not None:
                kwargs["webhook_url"] = reserved[1]

        try:
            outcome = await fn(*args, model=model, **kwargs)
        except Exception:
            if reserved is not None:
                self._manager.jobs.webhooks.release(reserved[0])
            raise

        finalized = self._manager._finalize(outcome)
        if reserved is not None:
            job_id = getattr(finalized, "id", None)
            if job_id is not None and hasattr(finalized, "status"):
                self._manager.jobs.webhooks.bind(reserved[0], job_id)
            else:
                # The call answered synchronously; nothing will ever call back.
                self._manager.jobs.webhooks.release(reserved[0])
        return finalized

    def _stream(
        self,
        capability: MediaCapability,
        method: str,
        *args: Any,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Resolve *capability* and return the adapter's byte stream.

        Not a coroutine: streaming methods return an async iterator directly, so
        callers write ``async for chunk in router.stream_...(...)``.
        """
        adapter = self._manager.resolve(capability, provider=provider, model=model)
        return getattr(adapter, method)(*args, model=model, **kwargs)


class ImageRouter(_BaseRouter):
    """Image generation, editing, upscaling and OCR."""

    async def generate(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: Sequence[MediaRef] | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate images from a text prompt."""
        return await self._dispatch(
            MediaCapability.IMAGE_GENERATE,
            "generate_image_media",
            prompt,
            provider=provider,
            model=model,
            n=n,
            size=size,
            seed=seed,
            negative_prompt=negative_prompt,
            reference_images=reference_images,
            **kwargs,
        )

    async def edit(
        self,
        prompt: str,
        *,
        image: MediaRef,
        mask: MediaRef | None = None,
        provider: str | None = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Edit an existing image according to a prompt."""
        return await self._dispatch(
            MediaCapability.IMAGE_EDIT,
            "edit_image_media",
            prompt,
            provider=provider,
            model=model,
            image=image,
            mask=mask,
            n=n,
            size=size,
            **kwargs,
        )

    async def upscale(
        self,
        *,
        image: MediaRef,
        provider: str | None = None,
        model: str | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Increase an image's resolution."""
        return await self._dispatch(
            MediaCapability.IMAGE_UPSCALE,
            "upscale_image_media",
            provider=provider,
            model=model,
            image=image,
            scale=scale,
            **kwargs,
        )

    async def ocr(
        self,
        *,
        document: MediaRef,
        provider: str | None = None,
        model: str | None = None,
        pages: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Extract text from a document or image."""
        return await self._dispatch(
            MediaCapability.OCR,
            "ocr_media",
            provider=provider,
            model=model,
            document=document,
            pages=pages,
            **kwargs,
        )


class AudioRouter(_BaseRouter):
    """Speech synthesis, transcription, music and sound effects."""

    async def speak(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Synthesize speech from text in one shot."""
        return await self._dispatch(
            MediaCapability.TTS,
            "synthesize_speech_media",
            text,
            provider=provider,
            model=model,
            voice=voice,
            audio_format=audio_format,
            sample_rate_hz=sample_rate_hz,
            speed=speed,
            **kwargs,
        )

    def stream_tts(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech as it is produced."""
        return self._stream(
            MediaCapability.TTS_STREAM,
            "stream_speech_media",
            text,
            provider=provider,
            model=model,
            voice=voice,
            audio_format=audio_format,
            sample_rate_hz=sample_rate_hz,
            **kwargs,
        )

    async def transcribe(
        self,
        *,
        audio: MediaRef,
        provider: str | None = None,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Transcribe audio to text."""
        return await self._dispatch(
            MediaCapability.ASR,
            "transcribe_media",
            provider=provider,
            model=model,
            audio=audio,
            language=language,
            diarize=diarize,
            timestamps=timestamps,
            **kwargs,
        )

    async def open_transcription_session(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        language: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a realtime, bidirectional transcription session."""
        adapter = self._manager.resolve(
            MediaCapability.ASR_STREAM, provider=provider, model=model
        )
        return await adapter.open_transcription_session(
            model=model, language=language, sample_rate_hz=sample_rate_hz, **kwargs
        )

    async def music(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate music from a text prompt."""
        return await self._dispatch(
            MediaCapability.MUSIC,
            "generate_music_media",
            prompt,
            provider=provider,
            model=model,
            duration_seconds=duration_seconds,
            **kwargs,
        )

    async def sfx(
        self,
        prompt: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        video: MediaRef | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate sound effects from a prompt and/or a video (foley)."""
        return await self._dispatch(
            MediaCapability.SFX,
            "generate_sfx_media",
            prompt,
            provider=provider,
            model=model,
            video=video,
            duration_seconds=duration_seconds,
            **kwargs,
        )


class VideoRouter(_BaseRouter):
    """Video generation, editing and frame interpolation."""

    async def generate(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        first_frame: MediaRef | None = None,
        last_frame: MediaRef | None = None,
        reference_images: Sequence[MediaRef] | None = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        fps: float | None = None,
        with_audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Generate a video; returns a job handle.

        ``first_frame``/``last_frame`` request a *generative* transition, which
        is distinct from :meth:`interpolate`.
        """
        outcome = await self._dispatch(
            MediaCapability.VIDEO_GENERATE,
            "generate_video_media",
            prompt,
            provider=provider,
            model=model,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            duration_seconds=duration_seconds,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            fps=fps,
            with_audio=with_audio,
            seed=seed,
            **kwargs,
        )
        return outcome  # type: ignore[return-value]

    async def edit(
        self,
        prompt: str,
        *,
        video: MediaRef,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Edit an existing video according to a prompt."""
        outcome = await self._dispatch(
            MediaCapability.VIDEO_EDIT,
            "edit_video_media",
            prompt,
            provider=provider,
            model=model,
            video=video,
            **kwargs,
        )
        return outcome  # type: ignore[return-value]

    async def interpolate(
        self,
        *,
        video: MediaRef | None = None,
        frames: Sequence[MediaRef] | None = None,
        provider: str | None = None,
        model: str | None = None,
        target_fps: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Interpolate frames to raise FPS or blend between supplied frames."""
        outcome = await self._dispatch(
            MediaCapability.VIDEO_INTERPOLATE,
            "interpolate_video_media",
            provider=provider,
            model=model,
            video=video,
            frames=frames,
            target_fps=target_fps,
            **kwargs,
        )
        return outcome  # type: ignore[return-value]
