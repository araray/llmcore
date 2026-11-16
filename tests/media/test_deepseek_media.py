# tests/media/test_deepseek_media.py
"""DeepSeek tool APIs: TTS, ASR, OCR (api-docs.deepseek.com/guides, 2026-11-16).

Live: all three endpoints authenticate and answer 402 (insufficient balance)
on this account, mapped to a non-retryable ProviderError.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llmcore.exceptions import ProviderError
from llmcore.media import MediaRef
from llmcore.media.models import MediaCapability, MediaExecution
from llmcore.providers.deepseek_provider import DeepSeekProvider

TOOLS = "https://api.deepseek.com/v1/tools"


@pytest.fixture
def provider(monkeypatch) -> DeepSeekProvider:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return DeepSeekProvider({"api_key": "sk-test"})


def test_declared(provider):
    assert {c.value for c in provider.media_capabilities()} == {"tts", "tts_stream", "asr", "ocr"}
    assert provider.media_execution(MediaCapability.TTS_STREAM) is MediaExecution.STREAM


@respx.mock
async def test_tts(provider):
    route = respx.post(f"{TOOLS}/tts").mock(return_value=httpx.Response(200, content=b"RIFFwav"))
    result = await provider.synthesize_speech_media("Hello", voice="tide", audio_format="pcm")
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"text": "Hello", "voice_id": "tide", "format": "pcm"}
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    assert result.artifact.data == b"RIFFwav" and result.artifact.mime_type == "audio/L16;rate=24000"


@respx.mock
async def test_asr(provider):
    route = respx.post(f"{TOOLS}/asr").mock(return_value=httpx.Response(
        200, json={"request_id": "r1", "text": "What is the weather?", "duration_ms": 3200}))
    result = await provider.transcribe_media(audio=MediaRef(data=b"RIFF", mime_type="audio/wav"))
    assert b'name="file"' in route.calls[0].request.content
    assert result.text == "What is the weather?" and result.usage.seconds == 3.2


@respx.mock
async def test_ocr_tolerates_keepalive_newlines(provider):
    body = "\n\n\n" + json.dumps({"id": "ocr-1", "purpose": "text_recognition", "page_count": 1,
                                  "pages": [{"index": 0, "text": "Hello page", "finish_reason": "stop"}],
                                  "usage": {"input_tokens": 10, "output_tokens": 3}})
    route = respx.post(f"{TOOLS}/ocr").mock(return_value=httpx.Response(200, text=body))
    result = await provider.ocr_media(document=MediaRef(data=b"%PDF", mime_type="application/pdf"))
    assert b'name="purpose"' in route.calls[0].request.content and b"text_recognition" in route.calls[0].request.content
    assert "Hello page" in (result.text or "") and result.usage.input_tokens == 10


@respx.mock
async def test_balance_error(provider):
    respx.post(f"{TOOLS}/asr").mock(return_value=httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}))
    with pytest.raises(ProviderError, match="insufficient balance") as info:
        await provider.transcribe_media(audio=MediaRef(data=b"RIFF", mime_type="audio/wav"))
    assert info.value.status_code == 402 and info.value.retryable is False
