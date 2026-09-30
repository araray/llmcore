# tests/media/test_deepgram_media_adapter.py
"""Deepgram as the reference media adapter (spec phase M2).

Deepgram is the first migration precisely because it already exercises batch
STT, realtime WebSocket STT and a bidirectional voice agent — the hard parts of
the abstraction.  These tests assert:

* it satisfies every capability protocol it declares;
* the new protocol methods *delegate* to the existing implementations rather
  than duplicating them, translating ``MediaRef`` in and ``MediaArtifact`` out;
* the twelve provider-specific methods are untouched (backward compatibility);
* ``MediaManager`` discovers and routes to it without naming it in code.

The Deepgram SDK is mocked throughout — no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.media import MediaCapability, MediaExecution, MediaManager, MediaRef
from llmcore.media.protocols import (
    CAPABILITY_PROTOCOLS,
    ASRProvider,
    MediaCapableProvider,
    StreamingASRProvider,
    StreamingTTSProvider,
    TTSProvider,
)
from llmcore.models_multimodal import SpeechResult, TranscriptionResult, TranscriptionSegment
from llmcore.providers.deepgram_provider import deepgram_available

# The Deepgram SDK is an optional extra (``pip install llmcore[deepgram]``).
# Constructing the provider touches real SDK symbols, so skip the module rather
# than fail when it is absent — mirroring tests/providers/test_deepgram_*.py.
# CI installs ``.[dev,all]``, so these do run there.
pytestmark = pytest.mark.skipif(
    not deepgram_available,
    reason="deepgram-sdk not installed (optional extra: pip install llmcore[deepgram])",
)


@pytest.fixture
def provider():
    """A DeepgramProvider with the SDK client mocked out."""
    with patch("llmcore.providers.deepgram_provider.AsyncDeepgramClient") as client_cls, patch(
        "llmcore.providers.deepgram_provider.deepgram_available", True
    ):
        client_cls.return_value = MagicMock()
        from llmcore.providers.deepgram_provider import DeepgramProvider

        return DeepgramProvider({"api_key": "test-key", "_instance_name": "deepgram"})


def _transcript(**kw) -> TranscriptionResult:
    return TranscriptionResult(
        text=kw.pop("text", "hello there"),
        language=kw.pop("language", "en"),
        duration_seconds=kw.pop("duration_seconds", 120.0),
        model=kw.pop("model", "nova-3"),
        segments=kw.pop("segments", [TranscriptionSegment(text="hello", start=0, end=1)]),
        metadata=kw.pop("metadata", {"request_id": "r1"}),
    )


def _speech(**kw) -> SpeechResult:
    return SpeechResult(
        audio_data=kw.pop("audio_data", b"ID3audio"),
        format=kw.pop("format", "mp3"),
        model=kw.pop("model", "aura-2-thalia-en"),
        voice=kw.pop("voice", "thalia"),
        duration_seconds=kw.pop("duration_seconds", 2.0),
        metadata=kw.pop("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_is_media_capable(self, provider):
        assert isinstance(provider, MediaCapableProvider)

    @pytest.mark.parametrize(
        "protocol", [ASRProvider, TTSProvider, StreamingTTSProvider, StreamingASRProvider]
    )
    def test_implements_audio_protocols(self, provider, protocol):
        assert isinstance(provider, protocol)

    def test_declares_only_speech_capabilities(self, provider):
        assert provider.media_capabilities() == frozenset(
            {
                MediaCapability.ASR,
                MediaCapability.ASR_STREAM,
                MediaCapability.TTS,
                MediaCapability.TTS_STREAM,
                MediaCapability.VOICE_AGENT,
            }
        )

    def test_every_declared_capability_is_backed(self, provider):
        """A declaration the manager would have to drop is a bug here."""
        unmet = [
            cap.value
            for cap in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[cap])
        ]
        assert unmet == []

    def test_declares_no_image_or_video(self, provider):
        caps = provider.media_capabilities()
        assert MediaCapability.IMAGE_GENERATE not in caps
        assert MediaCapability.VIDEO_GENERATE not in caps
        assert MediaCapability.OCR not in caps

    @pytest.mark.parametrize(
        ("capability", "expected"),
        [
            (MediaCapability.ASR, MediaExecution.REQUEST_RESPONSE),
            (MediaCapability.TTS, MediaExecution.REQUEST_RESPONSE),
            (MediaCapability.ASR_STREAM, MediaExecution.STREAM),
            (MediaCapability.TTS_STREAM, MediaExecution.STREAM),
            (MediaCapability.VOICE_AGENT, MediaExecution.STREAM),
        ],
    )
    def test_execution_classes(self, provider, capability, expected):
        assert provider.media_execution(capability) is expected

    def test_has_no_async_job_surface(self, provider):
        """Deepgram is request/response or live stream only."""
        assert all(
            provider.media_execution(c) is not MediaExecution.ASYNC_JOB
            for c in provider.media_capabilities()
        )


# ---------------------------------------------------------------------------
# transcribe_media
# ---------------------------------------------------------------------------


class TestTranscribeMedia:
    async def test_delegates_with_inline_bytes(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        result = await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"WAVDATA", mime_type="audio/wav")
        )
        payload, kwargs = provider.transcribe_audio.call_args[0], provider.transcribe_audio.call_args.kwargs
        assert payload[0] == b"WAVDATA"
        assert "url" not in kwargs
        assert result.capability is MediaCapability.ASR
        assert result.text == "hello there"

    async def test_remote_ref_uses_deepgram_url_path(self, provider):
        """A remote URL is handed to Deepgram, not downloaded locally."""
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        await provider.transcribe_media(audio=MediaRef.from_url("https://x/a.wav"))
        assert provider.transcribe_audio.call_args.kwargs["url"] == "https://x/a.wav"

    async def test_reads_a_local_path(self, provider, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"FROMFILE")
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        await provider.transcribe_media(audio=MediaRef.from_path(f))
        assert provider.transcribe_audio.call_args[0][0] == b"FROMFILE"

    async def test_maps_diarize_and_timestamps(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"x"), diarize=True, timestamps=True
        )
        kwargs = provider.transcribe_audio.call_args.kwargs
        assert kwargs["diarize"] is True
        # timestamps maps onto Deepgram's utterances, which is what produces timings
        assert kwargs["utterances"] is True

    async def test_forwards_language_and_vendor_kwargs(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"x"), language="pt", smart_format=True
        )
        kwargs = provider.transcribe_audio.call_args.kwargs
        assert kwargs["language"] == "pt" and kwargs["smart_format"] is True

    async def test_result_carries_artifact_and_usage(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript(duration_seconds=120.0))
        result = await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        artifact = result.artifact
        assert artifact.text == "hello there"
        assert artifact.provider_metadata["language"] == "en"
        assert result.usage.audio_minutes == 2.0  # 120s
        assert result.usage.basis == "per_audio_minute"
        assert result.provider == "deepgram"

    async def test_missing_duration_leaves_minutes_unset(self, provider):
        provider.transcribe_audio = AsyncMock(
            return_value=_transcript(duration_seconds=None)
        )
        result = await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        assert result.usage.audio_minutes is None


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TestSynthesizeSpeechMedia:
    async def test_delegates_and_normalizes(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech())
        result = await provider.synthesize_speech_media("hello world")
        assert provider.generate_speech.call_args[0][0] == "hello world"
        assert result.capability is MediaCapability.TTS
        assert result.artifact.data == b"ID3audio"
        assert result.artifact.mime_type == "audio/mpeg"

    async def test_maps_voice_format_and_speed(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech())
        await provider.synthesize_speech_media(
            "hi", voice="aura-2-luna-en", audio_format="linear16", speed=1.2
        )
        kwargs = provider.generate_speech.call_args.kwargs
        assert kwargs["voice"] == "aura-2-luna-en"
        assert kwargs["response_format"] == "linear16"
        assert kwargs["speed"] == 1.2

    async def test_sample_rate_is_forwarded_under_the_vendor_name(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech())
        await provider.synthesize_speech_media("hi", sample_rate_hz=24000)
        assert provider.generate_speech.call_args.kwargs["sample_rate"] == 24000

    async def test_omitted_options_are_not_forced(self, provider):
        """Unset options must not override the provider's configured defaults."""
        provider.generate_speech = AsyncMock(return_value=_speech())
        await provider.synthesize_speech_media("hi")
        kwargs = provider.generate_speech.call_args.kwargs
        for absent in ("voice", "response_format", "speed", "sample_rate"):
            assert absent not in kwargs

    async def test_usage_counts_characters(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech(duration_seconds=2.0))
        result = await provider.synthesize_speech_media("hello")
        assert result.usage.basis == "per_character"
        assert result.usage.characters == 5
        assert result.usage.seconds == 2.0


class TestStreamSpeechMedia:
    async def test_returns_an_iterator_not_a_coroutine(self, provider):
        async def _chunks(*_a, **_k):
            for piece in (b"a", b"b"):
                yield piece

        provider.stream_speech = _chunks
        stream = provider.stream_speech_media("hi")  # no await
        assert [c async for c in stream] == [b"a", b"b"]

    async def test_voice_folds_into_model(self, provider):
        """Deepgram encodes the voice in the model id."""
        captured: dict[str, Any] = {}

        async def _chunks(text, **kwargs):
            captured.update(kwargs)
            yield b""

        provider.stream_speech = _chunks
        [c async for c in provider.stream_speech_media("hi", voice="aura-2-luna-en")]
        assert captured["model"] == "aura-2-luna-en"

    async def test_explicit_model_beats_voice(self, provider):
        captured: dict[str, Any] = {}

        async def _chunks(text, **kwargs):
            captured.update(kwargs)
            yield b""

        provider.stream_speech = _chunks
        [
            c
            async for c in provider.stream_speech_media(
                "hi", model="aura-2-thalia-en", voice="aura-2-luna-en"
            )
        ]
        assert captured["model"] == "aura-2-thalia-en"


class TestOpenTranscriptionSession:
    async def test_delegates_to_the_socket(self, provider):
        sentinel = object()
        provider.open_transcription_socket = MagicMock(return_value=sentinel)
        session = await provider.open_transcription_session(
            model="nova-3", language="en", sample_rate_hz=16000
        )
        assert session is sentinel
        kwargs = provider.open_transcription_socket.call_args.kwargs
        assert kwargs["model"] == "nova-3"
        assert kwargs["language"] == "en"
        assert kwargs["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    LEGACY_METHODS = [
        "transcribe_audio",
        "generate_speech",
        "stream_speech",
        "transcribe_stream",
        "transcribe_stream_flux",
        "open_transcription_socket",
        "open_flux_socket",
        "open_speech_socket",
        "open_voice_agent",
        "run_voice_agent",
        "analyze_text",
        "grant_token",
        "get_projects",
    ]

    @pytest.mark.parametrize("name", LEGACY_METHODS)
    def test_legacy_method_still_present(self, provider, name):
        assert callable(getattr(provider, name))

    async def test_legacy_return_types_unchanged(self, provider):
        """The legacy surface still returns the legacy types, not MediaResult."""
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        assert isinstance(await provider.transcribe_audio(b"x"), TranscriptionResult)
        provider.generate_speech = AsyncMock(return_value=_speech())
        assert isinstance(await provider.generate_speech("x"), SpeechResult)

    def test_chat_completion_is_still_refused(self, provider):
        assert callable(provider.chat_completion)


# ---------------------------------------------------------------------------
# Routing through MediaManager
# ---------------------------------------------------------------------------


class TestRoutingThroughManager:
    def test_manager_discovers_deepgram(self, provider):
        pm = MagicMock()
        pm.get_available_providers.return_value = ["deepgram"]
        pm.get_provider.side_effect = lambda _n: provider
        media = MediaManager.from_provider_manager(pm, lambda k, d=None: d)
        assert media.adapter_names == ["deepgram"]

    def test_asr_routes_to_deepgram_by_default(self, provider):
        """The built-in preference puts Deepgram first for production ASR."""
        media = MediaManager({"deepgram": provider})
        assert media.who_can(MediaCapability.ASR) == ["deepgram"]
        assert media.resolve(MediaCapability.ASR) is provider

    def test_unsupported_capability_is_refused_with_hints(self, provider):
        from llmcore.exceptions import MediaCapabilityError

        media = MediaManager({"deepgram": provider})
        with pytest.raises(MediaCapabilityError) as exc:
            media.resolve(MediaCapability.VIDEO_GENERATE)
        assert "gemini" in str(exc.value) or "fal" in str(exc.value)

    async def test_router_transcribes_through_the_manager(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        media = MediaManager({"deepgram": provider})
        result = await media.audio.transcribe(audio=MediaRef.from_bytes(b"x"))
        assert result.text == "hello there"
        assert result.provider == "deepgram"

    async def test_router_streams_tts_through_the_manager(self, provider):
        async def _chunks(*_a, **_k):
            for piece in (b"x", b"y"):
                yield piece

        provider.stream_speech = _chunks
        media = MediaManager({"deepgram": provider})
        assert [c async for c in media.audio.stream_tts("hi")] == [b"x", b"y"]
