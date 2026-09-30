# src/llmcore/media/__init__.py
"""Generative media (image, audio, video) for LLMCore.

A sibling subsystem to chat providers and search providers, reached through
:attr:`llmcore.LLMCore.media`.  See ``docs/MEDIA_SUBSYSTEM_SPEC.md`` for the
design, and :mod:`llmcore.media.protocols` for what a provider adapter must
implement.

Quick shape::

    result = await llm.media.images.generate("an orange tabby")
    async for chunk in llm.media.audio.stream_tts("hello"):
        ...
    job = await llm.media.video.generate("a drone shot over dunes")
    result = await llm.media.wait(job)
"""

from .artifacts import ArtifactStore, MaterializePolicy
from .jobs import JobPolicy, MediaJobManager
from .manager import MediaManager
from .models import (
    AUDIO_CAPABILITIES,
    CAPABILITY_KINDS,
    IMAGE_CAPABILITIES,
    TERMINAL_JOB_STATUSES,
    VIDEO_CAPABILITIES,
    MediaArtifact,
    MediaCapability,
    MediaExecution,
    MediaJob,
    MediaJobStatus,
    MediaKind,
    MediaProvenance,
    MediaRef,
    MediaResult,
    MediaUsage,
)
from .protocols import (
    CAPABILITY_PROTOCOLS,
    ASRProvider,
    ImageEditProvider,
    ImageGenerationProvider,
    ImageUpscaleProvider,
    MediaCapableProvider,
    MediaJobPoller,
    MusicProvider,
    OCRMediaProvider,
    SFXProvider,
    StreamingASRProvider,
    StreamingTTSProvider,
    TTSProvider,
    VideoEditProvider,
    VideoGenerationProvider,
    VideoInterpolationProvider,
)
from .routers import AudioRouter, ImageRouter, VideoRouter

__all__ = [
    # manager + routers
    "MediaManager",
    "ImageRouter",
    "AudioRouter",
    "VideoRouter",
    # models
    "MediaKind",
    "MediaCapability",
    "MediaExecution",
    "MediaJobStatus",
    "MediaRef",
    "MediaArtifact",
    "MediaProvenance",
    "MediaUsage",
    "MediaResult",
    "MediaJob",
    "AUDIO_CAPABILITIES",
    "IMAGE_CAPABILITIES",
    "VIDEO_CAPABILITIES",
    "CAPABILITY_KINDS",
    "TERMINAL_JOB_STATUSES",
    # jobs + artifacts
    "MediaJobManager",
    "JobPolicy",
    "ArtifactStore",
    "MaterializePolicy",
    # protocols
    "MediaCapableProvider",
    "MediaJobPoller",
    "ImageGenerationProvider",
    "ImageEditProvider",
    "ImageUpscaleProvider",
    "OCRMediaProvider",
    "TTSProvider",
    "StreamingTTSProvider",
    "ASRProvider",
    "StreamingASRProvider",
    "MusicProvider",
    "SFXProvider",
    "VideoGenerationProvider",
    "VideoEditProvider",
    "VideoInterpolationProvider",
    "CAPABILITY_PROTOCOLS",
]
