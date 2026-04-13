# src/llmcore/models_multimodal.py
"""
Pydantic models for multimodal API results (TTS, STT, Image Generation).

These are provider-agnostic return types used by BaseProvider's optional
media generation / transcription methods.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Text-to-Speech (TTS) Result
# ---------------------------------------------------------------------------


class SpeechResult(BaseModel):
    """Result from a text-to-speech generation request.

    Attributes:
        audio_data: Raw audio bytes (the generated speech).
        format: Audio format of the data (mp3, wav, opus, flac, aac, pcm).
        model: Model used for generation.
        voice: Voice used for generation.
        duration_seconds: Duration of the generated audio (if available).
        metadata: Provider-specific metadata.
    """

    audio_data: bytes = Field(description="Raw audio bytes.")
    format: str = Field(
        default="mp3",
        description="Audio format (mp3, wav, opus, flac, aac, pcm).",
    )
    model: str = Field(description="Model used for generation.")
    voice: str = Field(description="Voice used.")
    duration_seconds: float | None = Field(
        default=None,
        description="Duration of generated audio in seconds.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Speech-to-Text (STT) / Transcription Result
# ---------------------------------------------------------------------------


class TranscriptionSegment(BaseModel):
    """A segment of a transcription with timing information.

    Attributes:
        text: The transcribed text for this segment.
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Speaker label (if diarization is available).
    """

    text: str = Field(description="Segment text.")
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    speaker: str | None = Field(
        default=None,
        description="Speaker label (for diarized transcription).",
    )


class TranscriptionResult(BaseModel):
    """Result from a speech-to-text / audio transcription request.

    Attributes:
        text: The full transcribed text.
        language: Detected or specified language (ISO-639-1).
        duration_seconds: Duration of the input audio.
        segments: Timed segments (if timestamp_granularities requested).
        model: Model used for transcription.
        metadata: Provider-specific metadata (logprobs, etc.).
    """

    text: str = Field(description="Full transcribed text.")
    language: str | None = Field(
        default=None,
        description="Detected language (ISO-639-1 code).",
    )
    duration_seconds: float | None = Field(
        default=None,
        description="Duration of input audio in seconds.",
    )
    segments: list[TranscriptionSegment] = Field(
        default_factory=list,
        description="Timed segments (if available).",
    )
    model: str = Field(description="Model used for transcription.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )


# ---------------------------------------------------------------------------
# Image Generation Result
# ---------------------------------------------------------------------------


class GeneratedImage(BaseModel):
    """A single generated image.

    Attributes:
        data: Base64-encoded image data (when response_format is b64_json).
        url: Temporary URL to the image (when response_format is url).
        revised_prompt: The prompt after model revision (if applicable).
        format: Image format (png, jpeg, webp).
    """

    data: str | None = Field(
        default=None,
        description="Base64-encoded image data.",
    )
    url: str | None = Field(
        default=None,
        description="Temporary URL to the generated image.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description="Model-revised prompt (if applicable).",
    )
    format: str = Field(
        default="png",
        description="Image format (png, jpeg, webp).",
    )


class ImageGenerationResult(BaseModel):
    """Result from an image generation request.

    Attributes:
        images: List of generated images.
        model: Model used for generation.
        metadata: Provider-specific metadata.
    """

    images: list[GeneratedImage] = Field(
        description="Generated images.",
    )
    model: str = Field(description="Model used for generation.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )
