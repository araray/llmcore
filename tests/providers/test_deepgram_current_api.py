# tests/providers/test_deepgram_current_api.py
"""Deepgram against deepgram-sdk 7.19 and the live API (2026-11-16).

- Flux TTS (``flux-<voice>-<lang>``) is GA on /v2/speak (speak.v2) with
  expressivity / normalize_volume / priority; Aura stays on /v1/speak;
- Flux TTS over the WebSocket speaks speak.v2 (Speak/Flush/Interrupt/Close,
  in-band Error frames); verified live with flux-haley-en;
- the SDK's transcribe_file/transcribe_url take no channels/sample_rate, so
  passing them (as the documented config allows) raised TypeError.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("deepgram")

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.providers.deepgram_provider import DeepgramProvider, _is_flux_tts  # noqa: E402


def test_flux_tts_model_detection():
    assert _is_flux_tts("flux-haley-en") and not _is_flux_tts("flux-general-en")
    assert not _is_flux_tts("aura-2-thalia-en") and not _is_flux_tts(None)


async def _gen(*chunks):
    for c in chunks:
        yield c


async def test_flux_routes_to_speak_v2_with_its_parameters():
    p = DeepgramProvider({"api_key": "k", "backend": "sdk"})
    p._client = MagicMock()
    p._client.speak.v2.audio.generate = MagicMock(return_value=_gen(b"ab"))
    out = await p.generate_speech("hi", model="flux-haley-en", expressivity=1, normalize_volume=True)
    kwargs = p._client.speak.v2.audio.generate.call_args.kwargs
    assert out.audio_data == b"ab" and kwargs["expressivity"] == 1 and kwargs["normalize_volume"] is True
    p._client.speak.v1.audio.generate.assert_not_called()


class _FakeSpeakSocket:
    """Stands in for the SDK's AsyncV2SocketClient (speak.v2)."""

    def __init__(self, frames):
        self.frames = frames
        self.sent = []

    async def send_speak(self, message):
        self.sent.append(("Speak", message.text))

    async def send_flush(self, message=None):
        self.sent.append(("Flush", None))

    async def send_interrupt(self, message=None):
        self.sent.append(("Interrupt", None))

    async def send_close(self, message=None):
        self.sent.append(("Close", None))

    def __aiter__(self):
        async def gen():
            import asyncio

            for f in self.frames:
                await asyncio.sleep(0.01)  # a server answers after the text arrives
                yield f
        return gen()


def _connect_returning(socket, seen):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def connect(**params):
        seen.update(params)
        yield socket

    return connect


async def test_flux_websocket_input_uses_speak_v2():
    p = DeepgramProvider({"api_key": "k", "backend": "sdk"})
    socket = _FakeSpeakSocket([{"type": "Connected"}, b"aud", b"io", {"type": "Flushed", "speech_id": "s1"}])
    seen: dict = {}
    p._client = MagicMock()
    p._client.speak.v2.connect = _connect_returning(socket, seen)

    async def text():
        yield "Hello "
        yield "there."

    audio = b"".join([c async for c in p.stream_speech(text(), model="flux-haley-en", expressivity=1, tag="t")])
    assert audio == b"audio"
    assert seen["model"] == "flux-haley-en" and seen["expressivity"] == 1 and seen["tag"] == "t"
    assert [k for k, _ in socket.sent][:3] == ["Speak", "Speak", "Flush"] and socket.sent[-1][0] == "Close"
    p._client.speak.v1.connect.assert_not_called()


async def test_flux_error_frame_raises_and_open_socket_interrupts():
    p = DeepgramProvider({"api_key": "k", "backend": "sdk"})
    socket = _FakeSpeakSocket([b"x", {"type": "Error", "code": "DATA-0002", "description": "pause marker over WebSocket"}])
    p._client = MagicMock()
    p._client.speak.v2.connect = _connect_returning(socket, {})
    async with p.open_speech_socket(model="flux-haley-en") as stream:
        await stream.send_text("hi")
        await stream.clear()
        with pytest.raises(ProviderError, match="DATA-0002") as info:
            async for _ in stream:
                pass
    assert ("Interrupt", None) in socket.sent and info.value.retryable is False


async def test_sdk_transcription_drops_direct_only_framing():
    p = DeepgramProvider({"api_key": "k", "backend": "sdk"})
    p._client = MagicMock()
    p._client.listen.v1.media.transcribe_file = AsyncMock(return_value={"results": {"channels": []}})
    try:
        await p.transcribe_audio(b"RIFF", channels=1, sample_rate=16000, language_hint=["en"], model="nova-3")
    except Exception:
        pass  # response shape is irrelevant here
    kwargs = p._client.listen.v1.media.transcribe_file.call_args.kwargs
    assert "channels" not in kwargs and "sample_rate" not in kwargs
    assert kwargs["language_hint"] == ["en"]
