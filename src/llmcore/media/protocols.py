# src/llmcore/media/protocols.py
"""Capability protocols implemented by LLMCore media provider adapters.

Each protocol describes one coherent group of media operations.  An adapter
implements only what its vendor supports, and the routers in
:mod:`llmcore.media.routers` discover that with ``isinstance`` — so routing
logic never names a provider.

Three method-shape conventions, matching the three execution classes in
the media subsystem design spec:

* **Request/response** returns :class:`~llmcore.media.models.MediaResult`.
* **Byte stream** returns ``AsyncIterator[bytes]`` (or an async session object).
* **Long-running job** returns :class:`~llmcore.media.models.MediaJob`.

An operation whose execution class varies by *model* (image generation is
request/response on OpenAI but a queued job on fal) is annotated
``MediaResult | MediaJob``; callers either branch on the returned type or ask
:meth:`MediaCapableProvider.media_execution` up front.

Method names carry a ``_media`` suffix where they would otherwise collide with
the legacy :class:`~llmcore.providers.base.BaseProvider` media methods
(``generate_image``, ``transcribe_audio``, ``generate_speech``, ``ocr``), which
keep their current signatures and return types for backward compatibility.  The
suffix disappears at the next major version.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .models import (
    MediaCapability,
    MediaExecution,
    MediaJob,
    MediaRef,
    MediaResult,
)

__all__ = [
    "CAPABILITY_PROTOCOLS",
    "ASRProvider",
    "ImageEditProvider",
    "ImageGenerationProvider",
    "ImageUpscaleProvider",
    "MediaCapableProvider",
    "MediaJobPoller",
    "MusicProvider",
    "OCRMediaProvider",
    "SFXProvider",
    "StreamingASRProvider",
    "StreamingTTSProvider",
    "TTSProvider",
    "VideoEditProvider",
    "VideoGenerationProvider",
    "VideoInterpolationProvider",
    "VoiceDesignProvider",
]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@runtime_checkable
class MediaCapableProvider(Protocol):
    """Minimum surface every media adapter exposes.

    Implemented by every adapter regardless of which capability protocols it
    also satisfies, so the manager can enumerate and describe providers without
    knowing what they can do.
    """

    def get_name(self) -> str:
        """Return the provider instance name."""
        ...

    def media_capabilities(self) -> frozenset[MediaCapability]:
        """Return every capability this adapter can currently serve.

        Should reflect configuration (credentials present, optional dependency
        installed), not just what the vendor theoretically offers.
        """
        ...

    def media_execution(
        self, capability: MediaCapability, model: str | None = None
    ) -> MediaExecution:
        """Return how *capability* completes for *model* on this provider.

        Lets a caller know before submitting whether to expect a result, a
        stream or a job handle.
        """
        ...


@runtime_checkable
class MediaJobPoller(Protocol):
    """Implemented by adapters that return :class:`MediaJob` handles.

    :class:`~llmcore.media.jobs.MediaJobManager` drives these; adapters never
    implement their own polling loop, backoff or timeout policy.
    """

    async def poll_media_job(self, job: MediaJob) -> MediaJob:
        """Refresh *job* against the vendor and return the updated handle.

        Must be safe to call on a terminal job (returning it unchanged) so the
        manager does not need to pre-check.
        """
        ...

    async def cancel_media_job(self, job: MediaJob) -> MediaJob:
        """Ask the vendor to cancel *job* and return the updated handle."""
        ...


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


@runtime_checkable
class ImageGenerationProvider(Protocol):
    """Text-to-image generation."""

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: Sequence[MediaRef] | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate *n* images from *prompt*.

        Args:
            prompt: Text description of the desired image.
            model: Model id; the provider default when omitted.
            n: Number of images to produce.
            size: Vendor-specific size token (e.g. ``"1024x1024"``).
            seed: Sampling seed for reproducibility, where supported.
            negative_prompt: What to avoid, where supported.
            reference_images: Style or subject references, where supported.
            **kwargs: Vendor-specific parameters, passed through.

        Returns:
            A result, or a job handle for queue-based vendors.
        """
        ...


@runtime_checkable
class ImageEditProvider(Protocol):
    """Instruction-guided editing of an existing image."""

    async def edit_image_media(
        self,
        prompt: str,
        *,
        image: MediaRef,
        mask: MediaRef | None = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Edit *image* according to *prompt*, optionally restricted by *mask*."""
        ...


@runtime_checkable
class ImageUpscaleProvider(Protocol):
    """Resolution enhancement of an existing image."""

    async def upscale_image_media(
        self,
        *,
        image: MediaRef,
        model: str | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Upscale *image* by *scale*, or to the model's native target."""
        ...


@runtime_checkable
class OCRMediaProvider(Protocol):
    """Document/image text extraction returning normalized artifacts."""

    async def ocr_media(
        self,
        *,
        document: MediaRef,
        model: str | None = None,
        pages: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Extract text (and optionally layout) from *document*."""
        ...


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSProvider(Protocol):
    """One-shot text-to-speech."""

    async def synthesize_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Synthesize *text* into a single audio artifact."""
        ...


@runtime_checkable
class StreamingTTSProvider(Protocol):
    """Chunked text-to-speech for low-latency playback.

    Deliberately *not* a job: the value is receiving the first bytes before
    synthesis finishes, which a job handle cannot express.
    """

    def stream_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks as they are synthesized.

        Note this is a plain method returning an async iterator, not a coroutine
        — callers write ``async for chunk in provider.stream_speech_media(...)``.
        """
        ...


@runtime_checkable
class ASRProvider(Protocol):
    """Batch speech-to-text."""

    async def transcribe_media(
        self,
        *,
        audio: MediaRef,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Transcribe *audio*; text lands on the artifact's ``text`` field."""
        ...


@runtime_checkable
class StreamingASRProvider(Protocol):
    """Realtime speech-to-text over a bidirectional session.

    Returns an opaque session object rather than an iterator because realtime
    ASR is duplex: the caller pushes audio *and* consumes events.  Deepgram's
    existing sockets are the reference shape.
    """

    async def open_transcription_session(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a realtime transcription session."""
        ...


@runtime_checkable
class VoiceDesignProvider(Protocol):
    """Generate candidate *voices* from a description, not speech from text.

    Distinct from TTS, which renders text in a voice that already exists. A
    design call returns several **previews** — each a sample of a different
    candidate voice — so the caller can audition them and keep one. The
    artifacts therefore arrive as a set, and each carries the provider-side id
    needed to create the voice for real.
    """

    async def design_voice_media(
        self,
        description: str,
        *,
        model: str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate voice previews matching *description*.

        Args:
            description: What the voice should sound like.
            model: Provider voice-design model.
            text: Sample text to speak in each preview. Providers that can
                invent suitable text do so when this is omitted.
        """
        ...


@runtime_checkable
class MusicProvider(Protocol):
    """Text-to-music generation."""

    async def generate_music_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate music from *prompt*."""
        ...


@runtime_checkable
class SFXProvider(Protocol):
    """Sound-effect generation, optionally conditioned on a video."""

    async def generate_sfx_media(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        video: MediaRef | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult | MediaJob:
        """Generate sound effects from *prompt* and/or *video* (foley)."""
        ...


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


@runtime_checkable
class VideoGenerationProvider(Protocol):
    """Text/image-conditioned video generation.

    Almost always a long-running job, hence the :class:`MediaJob` return.
    """

    async def generate_video_media(
        self,
        prompt: str,
        *,
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
        """Generate a video.

        ``first_frame`` / ``last_frame`` express generative transitions, which
        are semantically distinct from frame interpolation — see
        :class:`VideoInterpolationProvider`.
        """
        ...


@runtime_checkable
class VideoEditProvider(Protocol):
    """Instruction-guided modification of an existing video."""

    async def edit_video_media(
        self,
        prompt: str,
        *,
        video: MediaRef,
        model: str | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Edit *video* according to *prompt*."""
        ...


@runtime_checkable
class VideoInterpolationProvider(Protocol):
    """Frame interpolation / FPS increase.

    Distinct from a generative first/last-frame transition: interpolation fills
    between *existing* frames rather than inventing new content.
    """

    async def interpolate_video_media(
        self,
        *,
        video: MediaRef | None = None,
        frames: Sequence[MediaRef] | None = None,
        model: str | None = None,
        target_fps: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Interpolate *video*, or between the supplied *frames*."""
        ...


# ---------------------------------------------------------------------------
# Capability → protocol mapping
# ---------------------------------------------------------------------------

#: Which protocol an adapter must implement to serve each capability.  The
#: routers use this for discovery, so adding a capability means adding one entry
#: here rather than editing routing code.
CAPABILITY_PROTOCOLS: dict[MediaCapability, type] = {
    MediaCapability.IMAGE_GENERATE: ImageGenerationProvider,
    MediaCapability.IMAGE_EDIT: ImageEditProvider,
    MediaCapability.IMAGE_UPSCALE: ImageUpscaleProvider,
    MediaCapability.IMAGE_VARIATE: ImageGenerationProvider,
    MediaCapability.OCR: OCRMediaProvider,
    MediaCapability.TTS: TTSProvider,
    MediaCapability.TTS_STREAM: StreamingTTSProvider,
    MediaCapability.ASR: ASRProvider,
    MediaCapability.ASR_STREAM: StreamingASRProvider,
    MediaCapability.VOICE_AGENT: StreamingASRProvider,
    MediaCapability.MUSIC: MusicProvider,
    MediaCapability.SFX: SFXProvider,
    MediaCapability.VIDEO_GENERATE: VideoGenerationProvider,
    MediaCapability.VIDEO_EDIT: VideoEditProvider,
    MediaCapability.VIDEO_INTERPOLATE: VideoInterpolationProvider,
    MediaCapability.VIDEO_EXTEND: VideoEditProvider,
    MediaCapability.VIDEO_REFRAME: VideoEditProvider,
    MediaCapability.VIDEO_UPSCALE: VideoEditProvider,
    MediaCapability.VOICE_DESIGN: VoiceDesignProvider,
}
