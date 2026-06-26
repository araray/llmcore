# src/llmcore/models_multimodal.py
"""
Pydantic models for multimodal API results (TTS, STT, Image Generation).

These are provider-agnostic return types used by BaseProvider's optional
media generation / transcription methods.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

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


# ---------------------------------------------------------------------------
# OCR (Optical Character Recognition) Result
# ---------------------------------------------------------------------------


class OCRResult(BaseModel):
    """Result from an OCR / document intelligence request.

    Designed to capture the structured output of document OCR services
    such as Mistral OCR, which return page-level content with optional
    images, tables, and structured annotations.

    Attributes:
        pages: List of page result dicts, each typically containing
            ``index``, ``markdown`` (extracted text), and optionally
            ``images`` (list of image dicts with ``id``, ``image_base64``).
        model: OCR model used.
        document_annotation: Structured annotation (JSON schema extraction)
            if ``document_annotation_format`` was provided.
        pages_processed: Number of pages processed.
        doc_size_bytes: Original document size in bytes (if reported).
        metadata: Provider-specific metadata (usage info, etc.).
    """

    pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-page OCR results.",
    )
    model: str = Field(description="OCR model used.")
    document_annotation: Any | None = Field(
        default=None,
        description="Structured annotation result (if schema provided).",
    )
    pages_processed: int = Field(
        default=0,
        description="Number of pages processed.",
    )
    doc_size_bytes: int | None = Field(
        default=None,
        description="Original document size in bytes.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )


# ===========================================================================
# Streaming / Real-time event models (provider-neutral)
# ===========================================================================
#
# These models support real-time, bidirectional media providers (speech-to-text
# streaming, text-to-speech streaming, and voice agents).  They are intentionally
# provider-neutral: each event carries a ``provider`` discriminator and a ``raw``
# escape hatch holding the untouched provider payload, so they can be reused by
# any future realtime backend (e.g. OpenAI Realtime, Gemini Live) — not just
# Deepgram, which is the first consumer.
#
# Added in v0.49.x for the Deepgram provider (streaming STT/TTS + Voice Agent).


class StreamEventType(str, Enum):
    """Discriminator for :class:`TranscriptionStreamEvent`.

    Covers classic streaming STT events, Flux turn-detection events, and the
    generic socket lifecycle.  String-valued so events serialise cleanly to
    JSON and compare equal to plain strings.
    """

    # --- Streaming STT (Nova family) ---
    INTERIM = "interim"
    """A non-final (interim) hypothesis; text may change in later events."""
    FINAL = "final"
    """A finalised result for a segment (``is_final`` true on the wire)."""
    UTTERANCE_END = "utterance_end"
    """End-of-utterance marker emitted when ``utterance_end_ms`` elapses."""
    SPEECH_STARTED = "speech_started"
    """Voice-activity start marker (requires ``vad_events``)."""
    METADATA = "metadata"
    """Stream metadata frame (request id, model info, duration, …)."""

    # --- Flux (v2) conversational turn-detection events ---
    START_OF_TURN = "start_of_turn"
    """Flux detected the start of a user turn."""
    EAGER_END_OF_TURN = "eager_end_of_turn"
    """Flux's early end-of-turn signal (for latency-optimised pre-processing)."""
    TURN_RESUMED = "turn_resumed"
    """A previously eager-ended turn resumed (the user kept talking)."""
    END_OF_TURN = "end_of_turn"
    """Flux confirmed the end of a user turn."""
    UPDATE = "update"
    """Incremental transcript update within an in-progress Flux turn."""

    # --- Generic socket lifecycle ---
    OPEN = "open"
    CLOSE = "close"
    ERROR = "error"
    OTHER = "other"
    """Any event type not otherwise recognised; inspect ``raw``."""


class TranscriptionStreamEvent(BaseModel):
    """A single event from a streaming speech-to-text session.

    Attributes:
        type: The event discriminator (see :class:`StreamEventType`).
        text: Transcript text for this event (empty for non-transcript events).
        is_final: Whether this hypothesis is final (streaming STT).
        speech_final: Whether this marks the end of a spoken segment
            (endpoint detected).
        start: Segment start time in seconds (if provided).
        end: Segment end time in seconds (if provided).
        confidence: Confidence score for the top alternative (0.0-1.0).
        words: Raw per-word dicts (timings, confidence, speaker) when available.
        speaker: Speaker label for diarised streams (if available).
        channel_index: Channel index list for multichannel streams.
        provider: Provider that produced the event (default ``"deepgram"``).
        raw: The full, untouched provider payload for this event.
    """

    type: StreamEventType = Field(description="Event discriminator.")
    text: str = Field(default="", description="Transcript text for this event.")
    is_final: bool | None = Field(default=None, description="Final hypothesis?")
    speech_final: bool | None = Field(
        default=None, description="End-of-spoken-segment marker?"
    )
    start: float | None = Field(default=None, description="Start time (s).")
    end: float | None = Field(default=None, description="End time (s).")
    confidence: float | None = Field(
        default=None, description="Top-alternative confidence (0-1)."
    )
    words: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-word detail dicts."
    )
    speaker: str | None = Field(
        default=None, description="Speaker label (diarised streams)."
    )
    channel_index: list[int] | None = Field(
        default=None, description="Channel index (multichannel)."
    )
    provider: str = Field(default="deepgram", description="Producing provider.")
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Full provider payload."
    )


# ---------------------------------------------------------------------------
# Voice Agent events (provider-neutral)
# ---------------------------------------------------------------------------


class VoiceAgentEventType(str, Enum):
    """Discriminator for :class:`VoiceAgentEvent`."""

    WELCOME = "welcome"
    """Socket opened and the server acknowledged it."""
    SETTINGS_APPLIED = "settings_applied"
    """The agent ``Settings`` message was accepted."""
    CONVERSATION_TEXT = "conversation_text"
    """A turn of conversation text (user or assistant); see ``role``/``content``."""
    USER_STARTED_SPEAKING = "user_started_speaking"
    """The user began speaking (barge-in signal)."""
    AGENT_THINKING = "agent_thinking"
    """The agent is processing (LLM in flight)."""
    AGENT_STARTED_SPEAKING = "agent_started_speaking"
    """The agent began its spoken response."""
    AGENT_AUDIO_DONE = "agent_audio_done"
    """The agent finished streaming audio for an utterance."""
    AUDIO = "audio"
    """A chunk of agent audio bytes (see ``audio``)."""
    FUNCTION_CALL_REQUEST = "function_call_request"
    """The agent requested a function/tool call (see ``function_call``)."""
    PROMPT_UPDATED = "prompt_updated"
    THINK_UPDATED = "think_updated"
    SPEAK_UPDATED = "speak_updated"
    INJECTION_REFUSED = "injection_refused"
    """An ``inject_*`` request was refused (e.g. mid-utterance)."""
    ERROR = "error"
    WARNING = "warning"
    OPEN = "open"
    CLOSE = "close"
    OTHER = "other"
    """Any event type not otherwise recognised; inspect ``raw``."""


class VoiceAgentFunctionCall(BaseModel):
    """A function/tool call requested by a voice agent.

    Attributes:
        id: Provider-assigned identifier; echo it back in the response.
        name: Function name the agent wants to invoke.
        arguments: Parsed argument dict (best-effort; falls back to ``{}``).
        client_side: Whether the client is expected to execute the call
            (vs. server-side execution configured on the agent).
        raw: The full provider payload for the request.
    """

    id: str = Field(description="Function-call identifier.")
    name: str = Field(description="Function name.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed arguments."
    )
    client_side: bool = Field(
        default=True, description="Client is expected to execute the call."
    )
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Full provider payload."
    )


class VoiceAgentEvent(BaseModel):
    """A single event from a voice-agent session.

    Attributes:
        type: The event discriminator (see :class:`VoiceAgentEventType`).
        role: Speaker role for conversation text (``"user"``/``"assistant"``).
        content: Text content for conversation/text events.
        audio: Raw agent audio bytes for ``AUDIO`` events.
        function_call: The requested function call for
            ``FUNCTION_CALL_REQUEST`` events.
        provider: Provider that produced the event (default ``"deepgram"``).
        raw: The full, untouched provider payload for this event.
    """

    type: VoiceAgentEventType = Field(description="Event discriminator.")
    role: str | None = Field(default=None, description="Conversation role.")
    content: str | None = Field(default=None, description="Text content.")
    audio: bytes | None = Field(default=None, description="Agent audio bytes.")
    function_call: VoiceAgentFunctionCall | None = Field(
        default=None, description="Requested function call."
    )
    provider: str = Field(default="deepgram", description="Producing provider.")
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Full provider payload."
    )

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Text intelligence (analysis) result
# ---------------------------------------------------------------------------


class TextAnalysisResult(BaseModel):
    """Result from a text-intelligence / analysis request.

    Captures Deepgram's text-analysis surface (summarisation, topic detection,
    intent recognition, sentiment analysis) in a provider-neutral shape.

    Attributes:
        summary: Summary text (if ``summarize`` requested).
        topics: Detected topics (list of provider dicts).
        intents: Recognised intents (list of provider dicts).
        sentiments: Sentiment analysis payload (if requested).
        language: Detected/declared language code.
        model: Model used (if reported).
        request_id: Provider request identifier.
        metadata: Provider-specific metadata.
        raw: The full, untouched provider response.
    """

    summary: str | None = Field(default=None, description="Summary text.")
    topics: list[dict[str, Any]] = Field(
        default_factory=list, description="Detected topics."
    )
    intents: list[dict[str, Any]] = Field(
        default_factory=list, description="Recognised intents."
    )
    sentiments: dict[str, Any] | None = Field(
        default=None, description="Sentiment payload."
    )
    language: str | None = Field(default=None, description="Language code.")
    model: str | None = Field(default=None, description="Model used.")
    request_id: str | None = Field(
        default=None, description="Provider request id."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific metadata."
    )
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Full provider response."
    )
