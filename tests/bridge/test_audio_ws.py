"""End-to-end tests for the Tier-2 live-audio **WebSocket** transport.

These drive the real Starlette app (built by ``create_http_app``, which now
mounts the WS routes) through ``starlette.testclient.TestClient`` — a full ASGI
WebSocket path. They are intentionally *synchronous*: ``TestClient`` runs the
app in its own portal loop, and the bridge's duplex handling (a background pump
receiving while the main coroutine sends) exercises concurrent receive/send on
the socket.

Wire format mirrors the HTTP transport: one snake_case-JSON proto per text
frame. End-of-input is the empty ``{}`` sentinel (WebSocket has no half-close);
``TranscribeStream`` / ``SynthesizeStream`` also accept their in-band
``*_CONTROL_CLOSE`` frames. The fakes live in ``llmcore.bridge._testing`` and
are enabled by ``LLMCORE_BRIDGE_FAKE_AUDIO=1`` (read dynamically by
``BridgeCore.audio_enabled``).
"""

from __future__ import annotations

import base64
import warnings

import pytest

# Starlette 1.3.x emits a deprecation at import time nudging TestClient's HTTP
# backend from httpx to httpx2; it is incidental to WebSocket testing. Suppress
# it precisely at the import site (a pytest marker cannot catch an import-time
# warning raised during collection).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from llmcore.bridge._testing import FakeFacade, fake_count_tokens
from llmcore.bridge.core import BridgeCore
from llmcore.bridge.http_app import create_http_app

_TRANSCRIBE = "/llmcore.v1/AudioService/TranscribeStream"
_SYNTHESIZE = "/llmcore.v1/AudioService/SynthesizeStream"
_VOICE_AGENT = "/llmcore.v1/AudioService/VoiceAgent"


def _make_app():
    core = BridgeCore(
        FakeFacade(), count_tokens=fake_count_tokens, transports=("http",)
    )
    return create_http_app(core)


def _drain(ws) -> list[dict]:
    """Collect response frames until the server closes the socket."""
    out: list[dict] = []
    try:
        while True:
            out.append(ws.receive_json())
    except WebSocketDisconnect:
        pass
    return out


def test_ws_transcribe_stream(monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    client = TestClient(_make_app())
    with client.websocket_connect(_TRANSCRIBE) as ws:
        ws.send_json({"open": {"model": "fake-stt"}})
        ws.send_json({"audio": base64.b64encode(b"hello").decode()})
        ws.send_json({"audio": base64.b64encode(b"world").decode()})
        ws.send_json({"control": "STT_CONTROL_CLOSE"})
        events = _drain(ws)

    types = [e["type"] for e in events]
    assert types == [
        "STREAM_EVENT_TYPE_INTERIM",
        "STREAM_EVENT_TYPE_INTERIM",
        "STREAM_EVENT_TYPE_FINAL",
        "STREAM_EVENT_TYPE_UTTERANCE_END",
    ]
    interim_text = [e.get("text") for e in events if e["type"] == "STREAM_EVENT_TYPE_INTERIM"]
    assert interim_text == ["hello", "world"]
    final = next(e for e in events if e["type"] == "STREAM_EVENT_TYPE_FINAL")
    assert final["text"] == "hello world"
    assert final["is_final"] is True


def test_ws_transcribe_stream_eos_sentinel(monkeypatch):
    # The empty {} frame is an alternative end-of-input to STT_CONTROL_CLOSE.
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    client = TestClient(_make_app())
    with client.websocket_connect(_TRANSCRIBE) as ws:
        ws.send_json({"audio": base64.b64encode(b"foo").decode()})
        ws.send_json({})  # end-of-input sentinel
        events = _drain(ws)

    assert [e["type"] for e in events] == [
        "STREAM_EVENT_TYPE_INTERIM",
        "STREAM_EVENT_TYPE_FINAL",
        "STREAM_EVENT_TYPE_UTTERANCE_END",
    ]


def test_ws_synthesize_stream(monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    client = TestClient(_make_app())
    pieces = ["foo", "bar", "baz"]
    with client.websocket_connect(_SYNTHESIZE) as ws:
        ws.send_json({"open": {"model": "fake-tts"}})
        for p in pieces:
            ws.send_json({"text": p})
        ws.send_json({"control": "TTS_CONTROL_CLOSE"})
        outs = _drain(ws)

    audios = [base64.b64decode(o["audio"]) for o in outs]
    assert audios == [p.encode() for p in pieces]
    # int64 seq renders as a string in proto-JSON and is omitted when 0.
    assert [int(o.get("seq", 0)) for o in outs] == [0, 1, 2]
    assert b"".join(audios) == "".join(pieces).encode()


def test_ws_voice_agent_duplex(monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1")
    client = TestClient(_make_app())
    with client.websocket_connect(_VOICE_AGENT) as ws:
        # leading settings frame opens the session + selects the provider
        ws.send_json({"settings": {"provider_name": "fake"}})
        ws.send_json({"inject_user_message": "hi there"})
        ws.send_json({"audio": base64.b64encode(b"\x01\x02").decode()})
        ws.send_json({})  # end-of-input -> server emits CLOSE then closes 1000
        events = _drain(ws)

    types = [e["type"] for e in events]
    assert types[0] == "VOICE_AGENT_EVENT_TYPE_WELCOME"
    assert types[-1] == "VOICE_AGENT_EVENT_TYPE_CLOSE"
    convs = [
        (e.get("role"), e.get("content"))
        for e in events
        if e["type"] == "VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT"
    ]
    assert ("user", "hi there") in convs
    audios = [
        base64.b64decode(e["audio"])
        for e in events
        if e["type"] == "VOICE_AGENT_EVENT_TYPE_AUDIO"
    ]
    assert b"agent:\x01\x02" in audios


def test_ws_voice_agent_closed_when_disabled(monkeypatch):
    monkeypatch.delenv("LLMCORE_BRIDGE_FAKE_AUDIO", raising=False)
    client = TestClient(_make_app())
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(_VOICE_AGENT) as ws:
            ws.receive_json()
    assert excinfo.value.code == 1011
