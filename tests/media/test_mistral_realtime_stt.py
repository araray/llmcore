# tests/media/test_mistral_realtime_stt.py
"""Mistral realtime transcription (``asr_stream``) via the SDK's realtime client.

The live handshake on this account answers 429 (the plan has no quota for
voxtral-mini-transcribe-realtime-3, 2026-11-16); that path is asserted live in
the PR notes, and the event mapping is pinned here against the SDK's types.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mistralai.extra.realtime")

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.media.models import MediaCapability, MediaExecution  # noqa: E402
from llmcore.models_multimodal import StreamEventType  # noqa: E402
from llmcore.providers.mistral_provider import MistralProvider  # noqa: E402


class FakeConnection:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.sent: list[Any] = []
        self.send_audio = AsyncMock(side_effect=lambda b: self.sent.append(("audio", b)))
        self.flush_audio = AsyncMock(side_effect=lambda: self.sent.append(("flush", None)))
        self.end_audio = AsyncMock(side_effect=lambda: self.sent.append(("end", None)))
        self.close = AsyncMock()

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()


def _ev(kind: str, **fields: Any) -> NS:
    e = NS(type=kind, **fields)
    e.model_dump = lambda **_: {"type": kind, **fields}
    return e


@pytest.fixture
def provider(monkeypatch) -> MistralProvider:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    return MistralProvider({"api_key": "k"})


def test_declared(provider):
    assert MediaCapability.ASR_STREAM in provider.media_capabilities()
    assert provider.media_execution(MediaCapability.ASR_STREAM) is MediaExecution.STREAM


async def test_events_and_controls(provider):
    conn = FakeConnection([
        _ev("transcription.language", audio_language="en"),
        _ev("transcription.text.delta", text="Hello "),
        _ev("transcription.text.delta", text="world."),
        _ev("transcription.segment", text=" Hello world.", start=0.1, end=1.2),
        _ev("transcription.done", text="Hello world.", model="voxtral", usage={}, language="en"),
    ])
    client = MagicMock()
    client.audio.realtime.connect = AsyncMock(return_value=conn)
    with patch("mistralai.client.Mistral", return_value=client):
        async with provider.open_transcription_session(language="en", sample_rate_hz=16000, diarize=True) as s:
            await s.send_audio(b"\x00\x01")
            await s.finalize()
            await s.end()
            events = [e async for e in s]
    args, kwargs = client.audio.realtime.connect.await_args
    assert args == ("voxtral-mini-transcribe-realtime-3",)
    assert kwargs["audio_format"].sample_rate == 16000 and kwargs["language"] == "en" and kwargs["diarize"] is True
    assert [k for k, _ in conn.sent] == ["audio", "flush", "end"]
    assert [(e.type, e.text) for e in events] == [
        (StreamEventType.METADATA, "en"), (StreamEventType.INTERIM, "Hello "),
        (StreamEventType.INTERIM, "Hello world."), (StreamEventType.FINAL, "Hello world."),
        (StreamEventType.UTTERANCE_END, "Hello world.")]
    assert (events[3].start, events[3].end) == (0.1, 1.2)
    conn.close.assert_awaited()


async def test_quota_rejection_is_not_retryable(provider):
    client = MagicMock()
    client.audio.realtime.connect = AsyncMock(side_effect=RuntimeError("server rejected WebSocket connection: HTTP 429"))
    with patch("mistralai.client.Mistral", return_value=client):
        with pytest.raises(ProviderError, match="no quota") as info:
            async with provider.open_transcription_session():
                pass
    assert info.value.status_code == 429 and info.value.retryable is False
