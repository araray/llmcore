# tests/media/test_openai_realtime_stt.py
"""OpenAI realtime transcription as ``asr_stream`` (live-verified 2026-11-16).

Live: gpt-4o-transcribe with server VAD gave speech_started, utterance_end and
the exact sentence; gpt-live-transcribe rejects turn detection ("not supported
for this transcription model") and finalizes on an explicit commit.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from llmcore.media.models import MediaCapability, MediaExecution
from llmcore.models_multimodal import StreamEventType
from llmcore.providers.openai_provider import OpenAIProvider


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
def provider(monkeypatch) -> OpenAIProvider:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return OpenAIProvider({"api_key": "sk-test"})


def _connect(socket: FakeSocket, seen: dict[str, Any]):
    async def connect(url: str, additional_headers: dict[str, str]):
        seen.update(url=url, headers=additional_headers)
        return socket

    return connect


def test_declared(provider):
    assert MediaCapability.ASR_STREAM in provider.media_capabilities()
    assert provider.media_execution(MediaCapability.ASR_STREAM) is MediaExecution.STREAM


async def test_session_update_and_event_mapping(provider):
    socket = FakeSocket([
        {"type": "session.created"},
        {"type": "session.updated"},
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 120},
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "i1", "delta": "Real-time "},
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "i1", "delta": "works."},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": "Real-time works."},
    ])
    seen: dict[str, Any] = {}
    with patch("websockets.asyncio.client.connect", _connect(socket, seen)):
        async with provider.open_transcription_session(language="en", keywords=["llmcore"], noise_reduction="near_field") as s:
            await s.send_audio(b"\x00\x01")
            await s.finalize()
            events = [e async for e in s]
    assert seen["url"] == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert seen["headers"] == {"Authorization": "Bearer sk-test"}
    update, append, commit = socket.sent
    audio_in = update["session"]["audio"]["input"]
    assert update["type"] == "session.update" and update["session"]["type"] == "transcription"
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["transcription"] == {"model": "gpt-4o-transcribe", "language": "en", "keywords": ["llmcore"]}
    assert audio_in["turn_detection"] == {"type": "server_vad"}
    assert audio_in["noise_reduction"] == {"type": "near_field"}
    assert append["type"] == "input_audio_buffer.append" and commit["type"] == "input_audio_buffer.commit"
    assert [e.type for e in events] == [StreamEventType.OPEN, StreamEventType.SPEECH_STARTED,
                                        StreamEventType.INTERIM, StreamEventType.INTERIM, StreamEventType.FINAL]
    assert events[3].text == "Real-time works." and events[-1].text == "Real-time works."
    assert socket.closed


async def test_live_model_runs_without_turn_detection(provider):
    socket = FakeSocket([])
    with patch("websockets.asyncio.client.connect", _connect(socket, {})):
        async with provider.open_transcription_session(model="gpt-live-transcribe"):
            pass
    assert socket.sent[0]["session"]["audio"]["input"]["turn_detection"] is None


async def test_bad_format_and_rate(provider):
    with pytest.raises(ValueError, match="24 kHz"):
        async with provider.open_transcription_session(sample_rate_hz=16000):
            pass
    with pytest.raises(ValueError, match="pcm, pcmu or pcma"):
        async with provider.open_transcription_session(audio_format="opus"):
            pass
