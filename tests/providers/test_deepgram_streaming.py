# tests/providers/test_deepgram_streaming.py
"""Streaming unit tests for :class:`DeepgramProvider` (fake sockets, no network).

Covers the live surfaces added in Part 3:

* ``transcribe_stream`` / ``open_transcription_socket`` (``listen.v1``)
* ``transcribe_stream_flux`` / ``open_flux_socket`` (``listen.v2`` / Flux)
* ``stream_speech`` (REST string + WebSocket iterable) / ``open_speech_socket``

The fakes implement the same async-context-manager + ``send_*`` + ``__aiter__``
shape as the real SDK sockets, plus a ``wait_for_close`` switch so the
concurrent producer/consumer orchestration in the high-level methods terminates
deterministically (the receive loop ends only once the send side is closed,
mirroring real server behaviour).
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models_multimodal import StreamEventType
from llmcore.providers.deepgram_provider import DeepgramProvider

# Reuse the fake-client install helper from the core test module.
from .test_deepgram_provider import _install_fake_client


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any] | None = None
) -> DeepgramProvider:
    _install_fake_client(monkeypatch)
    cfg: dict[str, Any] = {"api_key": "dg-test-key", "_instance_name": "deepgram"}
    if config:
        cfg.update(config)
    return DeepgramProvider(cfg)


# ---------------------------------------------------------------------------
# Fake sockets
# ---------------------------------------------------------------------------


class _BaseFakeSocket:
    """Async-context-manager fake socket with a scripted event stream."""

    def __init__(self, messages: list[Any], *, wait_for_close: bool = True) -> None:
        self.messages = messages
        self.wait_for_close = wait_for_close
        self.closed = False
        self._closed_event = asyncio.Event()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self._closed_event.set()  # ensure any waiter unblocks on teardown
        return False

    async def __aiter__(self):
        for m in self.messages:
            await asyncio.sleep(0)  # cede control so the producer task can run
            yield m
        if self.wait_for_close:
            await self._closed_event.wait()


class _FakeSTTSocket(_BaseFakeSocket):
    def __init__(self, messages: list[Any], **kw: Any) -> None:
        super().__init__(messages, **kw)
        self.sent_audio: list[bytes] = []
        self.finalized = False
        self.keepalives = 0

    async def send_media(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def send_finalize(self) -> None:
        self.finalized = True

    async def send_keep_alive(self) -> None:
        self.keepalives += 1

    async def send_close_stream(self) -> None:
        self.closed = True
        self._closed_event.set()


class _FakeFluxSocket(_BaseFakeSocket):
    def __init__(self, messages: list[Any], **kw: Any) -> None:
        super().__init__(messages, **kw)
        self.sent_audio: list[bytes] = []
        self.configured: list[Any] = []

    async def send_media(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def send_configure(self, message: Any) -> None:
        self.configured.append(message)

    async def send_close_stream(self) -> None:
        self.closed = True
        self._closed_event.set()


class _FakeSpeakSocket(_BaseFakeSocket):
    def __init__(self, messages: list[Any], **kw: Any) -> None:
        super().__init__(messages, **kw)
        self.texts: list[str] = []
        self.flushed = False
        self.cleared = False

    async def send_text(self, message: Any) -> None:
        self.texts.append(message.text)

    async def send_flush(self) -> None:
        self.flushed = True

    async def send_clear(self) -> None:
        self.cleared = True

    async def send_close(self) -> None:
        self.closed = True
        self._closed_event.set()


class _Holder:
    """Minimal attribute holder for synthesizing missing client namespaces."""


def _bind_connect(target: Any, attr_path: list[str], socket: Any) -> dict[str, Any]:
    """Bind ``socket`` as the result of a ``connect(**kwargs)`` call.

    Intermediate namespace nodes are created on the fake client if absent (the
    core fake only pre-builds ``listen.v1`` / ``speak.v1.audio``).

    Returns a dict that captures the kwargs passed to ``connect``.
    """
    captured: dict[str, Any] = {}

    def _connect(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return socket

    node = target
    for a in attr_path[:-1]:
        child = getattr(node, a, None)
        if child is None:
            child = _Holder()
            setattr(node, a, child)
        node = child
    setattr(node, attr_path[-1], _connect)
    return captured


async def _aiter_bytes(chunks: list[bytes]):
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# Streaming STT (listen.v1)
# ---------------------------------------------------------------------------


def _stt_script() -> list[dict[str, Any]]:
    return [
        {
            "type": "Results",
            "is_final": False,
            "start": 0.0,
            "duration": 1.0,
            "speech_final": False,
            "channel_index": [0],
            "channel": {"alternatives": [{"transcript": "hello", "confidence": 0.5}]},
        },
        {
            "type": "Results",
            "is_final": True,
            "start": 0.0,
            "duration": 1.5,
            "speech_final": True,
            "channel_index": [0],
            "channel": {
                "alternatives": [
                    {"transcript": "hello world", "confidence": 0.95, "words": [{"word": "hello"}]}
                ]
            },
        },
        {"type": "UtteranceEnd", "channel": [0], "last_word_end": 1.5},
        {"type": "Metadata", "request_id": "r1", "duration": 1.5, "channels": 1},
    ]


@pytest.mark.asyncio
async def test_transcribe_stream_maps_events(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket(_stt_script())
    captured = _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _audio():
        yield b"a1"
        yield b"a2"

    events = [
        e
        async for e in p.transcribe_stream(
            _audio(), model="nova-3", interim_results=True, encoding="linear16", sample_rate=16000
        )
    ]

    types = [e.type for e in events]
    assert types == [
        StreamEventType.INTERIM,
        StreamEventType.FINAL,
        StreamEventType.UTTERANCE_END,
        StreamEventType.METADATA,
    ]
    assert events[0].text == "hello"
    assert events[0].is_final is False
    assert events[1].text == "hello world"
    assert events[1].is_final is True
    assert events[1].end == 1.5  # start + duration
    assert events[1].confidence == 0.95
    assert events[1].words == [{"word": "hello"}]
    assert events[2].end == 1.5

    # Producer side: audio pumped, finalized (default), then closed.
    assert socket.sent_audio == [b"a1", b"a2"]
    assert socket.finalized is True
    assert socket.closed is True
    # Connect params merged.
    assert captured["model"] == "nova-3"
    assert captured["interim_results"] is True
    assert captured["encoding"] == "linear16"
    assert captured["sample_rate"] == 16000


@pytest.mark.asyncio
async def test_transcribe_stream_no_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket(_stt_script())
    _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _audio():
        yield b"x"

    _ = [e async for e in p.transcribe_stream(_audio(), model="nova-3", finalize_on_close=False)]
    assert socket.finalized is False
    assert socket.closed is True


@pytest.mark.asyncio
async def test_transcribe_stream_uses_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(
        monkeypatch,
        {"stt": {"model": "nova-2", "smart_format": True, "streaming": {"vad_events": True}}},
    )
    socket = _FakeSTTSocket([])
    captured = _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _audio():
        yield b"x"

    _ = [e async for e in p.transcribe_stream(_audio())]
    assert captured["model"] == "nova-2"
    assert captured["smart_format"] is True
    assert captured["vad_events"] is True  # from stt.streaming defaults


@pytest.mark.asyncio
async def test_transcribe_stream_keepalive_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket([])
    _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _slow_audio():
        # Hold the stream open long enough for several keepalive ticks.
        for _ in range(3):
            await asyncio.sleep(0.02)
            yield b"chunk"

    _ = [
        e
        async for e in p.transcribe_stream(
            _slow_audio(), model="nova-3", keepalive_interval=0.005
        )
    ]
    assert socket.keepalives >= 1
    assert socket.closed is True


@pytest.mark.asyncio
async def test_transcribe_stream_producer_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket(_stt_script())
    _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _bad_audio():
        yield b"x"
        raise RuntimeError("mic exploded")

    with pytest.raises(ProviderError) as ei:
        _ = [e async for e in p.transcribe_stream(_bad_audio(), model="nova-3")]
    assert "producer" in str(ei.value).lower()
    # Even on producer failure the send side is closed (best-effort).
    assert socket.closed is True


@pytest.mark.asyncio
async def test_transcribe_stream_early_break_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket(_stt_script())
    _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async def _audio():
        for _ in range(1000):
            await asyncio.sleep(0)
            yield b"x"

    gen = p.transcribe_stream(_audio(), model="nova-3")
    first = await gen.__anext__()
    assert first.type == StreamEventType.INTERIM
    # Closing the generator must tear down pump/keepalive tasks without hanging.
    await asyncio.wait_for(gen.aclose(), timeout=1.0)


@pytest.mark.asyncio
async def test_open_transcription_socket_manual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSTTSocket(_stt_script(), wait_for_close=False)
    captured = _bind_connect(p.client, ["listen", "v1", "connect"], socket)

    async with p.open_transcription_socket(model="nova-3", language="en") as stream:
        await stream.send_audio(b"aa")
        await stream.keepalive()
        await stream.finalize()
        events = [e async for e in stream]
        await stream.close()

    assert socket.sent_audio == [b"aa"]
    assert socket.keepalives == 1
    assert socket.finalized is True
    assert socket.closed is True
    assert [e.type for e in events][:2] == [StreamEventType.INTERIM, StreamEventType.FINAL]
    assert captured["language"] == "en"


# ---------------------------------------------------------------------------
# Flux (listen.v2)
# ---------------------------------------------------------------------------


def _flux_script() -> list[dict[str, Any]]:
    return [
        {
            "type": "Connected",
            "request_id": "rid",
        },
        {
            "type": "TurnInfo",
            "event": "StartOfTurn",
            "transcript": "",
            "audio_window_start": 0.0,
            "audio_window_end": 0.0,
            "end_of_turn_confidence": 0.0,
            "words": [],
        },
        {
            "type": "TurnInfo",
            "event": "EndOfTurn",
            "transcript": "what time is it",
            "audio_window_start": 0.0,
            "audio_window_end": 2.0,
            "end_of_turn_confidence": 0.91,
            "words": [{"word": "what"}],
        },
    ]


@pytest.mark.asyncio
async def test_transcribe_stream_flux_maps_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeFluxSocket(_flux_script())
    captured = _bind_connect(p.client, ["listen", "v2", "connect"], socket)

    async def _audio():
        yield b"f1"

    events = [
        e
        async for e in p.transcribe_stream_flux(
            _audio(), model="flux-general-en", eot_threshold=0.7, encoding="linear16", sample_rate=16000
        )
    ]
    types = [e.type for e in events]
    assert types == [
        StreamEventType.OPEN,
        StreamEventType.START_OF_TURN,
        StreamEventType.END_OF_TURN,
    ]
    assert events[2].text == "what time is it"
    assert events[2].is_final is True
    assert events[2].confidence == 0.91
    assert events[2].end == 2.0
    assert socket.sent_audio == [b"f1"]
    assert socket.closed is True
    # Flux has no finalize; just model + params.
    assert captured["model"] == "flux-general-en"
    assert captured["eot_threshold"] == 0.7


@pytest.mark.asyncio
async def test_open_flux_socket_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeFluxSocket(_flux_script(), wait_for_close=False)
    _bind_connect(p.client, ["listen", "v2", "connect"], socket)

    async with p.open_flux_socket(model="flux-general-en") as stream:
        await stream.send_audio(b"zz")
        await stream.configure({"type": "Configure"})
        events = [e async for e in stream]
        await stream.close()

    assert socket.sent_audio == [b"zz"]
    assert socket.configured == [{"type": "Configure"}]
    assert socket.closed is True
    assert events[0].type == StreamEventType.OPEN


# ---------------------------------------------------------------------------
# Streaming TTS (speak.v1) — REST string + WS iterable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_speech_rest_string(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_generate(**kwargs: Any):
        captured.update(kwargs)
        return _aiter_bytes([b"AU", b"DIO"])

    p.client.speak.v1.audio.generate = _fake_generate

    chunks = [c async for c in p.stream_speech("hello", model="aura-2-thalia-en")]
    assert b"".join(chunks) == b"AUDIO"
    assert captured["text"] == "hello"
    assert captured["model"] == "aura-2-thalia-en"


@pytest.mark.asyncio
async def test_stream_speech_ws_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    # Audio bytes interleaved with a control frame (dict) that must be skipped.
    socket = _FakeSpeakSocket([b"AU", {"type": "Flushed"}, b"DIO"])
    captured = _bind_connect(p.client, ["speak", "v1", "connect"], socket)

    async def _text():
        yield "Hello "
        yield "world"

    chunks = [c async for c in p.stream_speech(_text(), model="aura-2-thalia-en", response_format="linear16")]
    assert b"".join(chunks) == b"AUDIO"
    assert socket.texts == ["Hello ", "world"]
    assert socket.flushed is True
    assert socket.closed is True
    assert captured["model"] == "aura-2-thalia-en"
    assert captured["encoding"] == "linear16"


@pytest.mark.asyncio
async def test_stream_speech_ws_producer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSpeakSocket([b"AU"])
    _bind_connect(p.client, ["speak", "v1", "connect"], socket)

    async def _bad_text():
        yield "Hello"
        raise RuntimeError("token source died")

    with pytest.raises(ProviderError) as ei:
        _ = [c async for c in p.stream_speech(_bad_text(), model="aura-2-thalia-en")]
    assert "producer" in str(ei.value).lower()
    assert socket.closed is True


@pytest.mark.asyncio
async def test_open_speech_socket_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    socket = _FakeSpeakSocket([b"aud1", b"aud2"])
    _bind_connect(p.client, ["speak", "v1", "connect"], socket)

    collected: list[bytes] = []

    async with p.open_speech_socket(model="aura-2-thalia-en") as stream:

        async def _consume():
            async for a in stream:
                collected.append(a)

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        await stream.send_text("hi there")
        await stream.flush()
        await stream.close()
        await asyncio.wait_for(task, timeout=1.0)

    assert collected == [b"aud1", b"aud2"]
    assert stream.socket.texts == ["hi there"]
    assert stream.socket.flushed is True
    assert stream.socket.closed is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
