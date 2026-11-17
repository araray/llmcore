# tests/media/test_elevenlabs_realtime_stt.py
"""ElevenLabs realtime Scribe as ``asr_stream`` (live-verified 2026-11-16).

Live: TTS pcm_16000 of "The quick brown fox jumps over the lazy dog." streamed
in 100 ms chunks gave 3 interim events and the exact sentence as FINAL; with
include_timestamps, 17 word entries (0.12-3.04 s); a bad key arrives as an
``auth_error`` message, mapped to an ERROR event.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from llmcore.media.models import MediaCapability, MediaExecution
from llmcore.models_multimodal import StreamEventType
from llmcore.providers.elevenlabs_provider import (
    ElevenLabsProvider,
    ElevenLabsTranscriptionStream,
    _map_realtime_message,
)


class FakeSocket:
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming = [json.dumps(m) for m in incoming]
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        async def gen():
            for m in self.incoming:
                yield m
        return gen()


@pytest.fixture
def provider(monkeypatch) -> ElevenLabsProvider:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    return ElevenLabsProvider({"api_key": "xi-test"})


def test_capability_declared(provider):
    assert MediaCapability.ASR_STREAM in provider.media_capabilities()
    assert provider.media_execution(MediaCapability.ASR_STREAM) is MediaExecution.STREAM


async def test_session_url_headers_and_events(provider):
    socket = FakeSocket([
        {"message_type": "session_started", "session_id": "s1"},
        {"message_type": "partial_transcript", "text": "The quick"},
        {"message_type": "committed_transcript", "text": "The quick brown fox."},
        {"message_type": "committed_transcript_with_timestamps", "text": "The quick brown fox.",
         "words": [{"text": "The", "start": 0.12, "end": 0.3, "type": "word"},
                   {"text": " ", "start": 0.3, "end": 0.31, "type": "spacing"},
                   {"text": "fox.", "start": 1.0, "end": 1.4, "type": "word"}]},
    ])
    seen: dict[str, Any] = {}

    async def connect(url: str, additional_headers: dict[str, str]):
        seen.update(url=url, headers=additional_headers)
        return socket

    with patch("websockets.asyncio.client.connect", connect):
        async with provider.open_transcription_session(
            sample_rate_hz=16000, language="en", include_timestamps=True, keyterms=["llmcore", "Scribe"],
        ) as session:
            session.set_context("Earlier text")
            await session.send_audio(b"\x00\x01")
            await session.send_audio(b"\x02")
            await session.finalize()
            events = [e async for e in session]

    assert seen["url"].startswith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")
    assert "model_id=scribe_v2_realtime" in seen["url"] and "audio_format=pcm_16000" in seen["url"]
    assert "commit_strategy=vad" in seen["url"] and "include_timestamps=true" in seen["url"]
    assert "keyterms=llmcore&keyterms=Scribe" in seen["url"] and "language_code=en" in seen["url"]
    assert seen["headers"] == {"xi-api-key": "xi-test"}
    first, second, commit = socket.sent
    assert first["previous_text"] == "Earlier text" and "previous_text" not in second
    assert first["sample_rate"] == 16000 and commit["commit"] is True
    # with timestamps on, the plain commit is suppressed in favour of the worded twin
    assert [e.type for e in events] == [StreamEventType.OPEN, StreamEventType.INTERIM, StreamEventType.FINAL]
    final = events[-1]
    assert final.text == "The quick brown fox." and (final.start, final.end) == (0.12, 1.4)
    assert socket.closed


async def test_token_auth_skips_the_key(provider):
    seen: dict[str, Any] = {}

    async def connect(url: str, additional_headers: dict[str, str]):
        seen.update(url=url, headers=additional_headers)
        return FakeSocket([])

    with patch("websockets.asyncio.client.connect", connect):
        async with provider.open_transcription_session(token="single-use"):
            pass
    assert seen["headers"] == {} and "token=single-use" in seen["url"]


def test_errors_and_unknown_messages_map():
    err = _map_realtime_message({"message_type": "auth_error", "error": "You must be authenticated"})
    assert err.type is StreamEventType.ERROR and "authenticated" in err.text
    other = _map_realtime_message({"message_type": "redacted_transcript", "text": "{REDACTED}"})
    assert other.type is StreamEventType.OTHER and other.raw["text"] == "{REDACTED}"


async def test_unsupported_format_rejected(provider):
    with pytest.raises(ValueError, match="realtime audio format"):
        async with provider.open_transcription_session(sample_rate_hz=11025):
            pass


def test_stream_close_is_idempotent():
    import asyncio

    socket = FakeSocket([])
    stream = ElevenLabsTranscriptionStream(socket, sample_rate=16000)
    asyncio.run(stream.close())
    asyncio.run(stream.close())
    assert socket.closed
