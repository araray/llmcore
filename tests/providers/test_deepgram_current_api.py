# tests/providers/test_deepgram_current_api.py
"""Deepgram against deepgram-sdk 7.19 and the live API (2026-11-16).

- Flux TTS (``flux-<voice>-<lang>``) is GA on /v2/speak (speak.v2) with
  expressivity / normalize_volume / priority; Aura stays on /v1/speak;
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


async def test_flux_websocket_input_is_refused_clearly():
    p = DeepgramProvider({"api_key": "k", "backend": "sdk"})

    async def text():
        yield "hi"

    with pytest.raises(ProviderError, match="speak.v2"):
        async for _ in p.stream_speech(text(), model="flux-haley-en"):
            pass


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
