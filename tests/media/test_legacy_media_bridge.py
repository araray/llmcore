# tests/media/test_legacy_media_bridge.py
"""Mistral, Z.ai and Gemini media reachable through llm.media (2026-11-16).

Mistral and Z.ai had working speech / transcription / OCR (and Z.ai image and
video) only as legacy methods, so the media manager skipped them. Gemini had
no transcription surface although ``gemini-3.5-transcribe`` is served.

Live: Mistral transcribe_media -> voxtral-mini-latest, exact text; Gemini
transcribe_media -> gemini-3.5-transcribe, exact text. Mistral OCR/TTS and Z.ai
video reach the API and fail only on the accounts' quota/balance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.exceptions import ProviderError
from llmcore.media import MediaRef
from llmcore.media.models import MediaCapability, MediaExecution, MediaJobStatus, MediaKind
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaJobPoller
from llmcore.models_multimodal import OCRResult, SpeechResult, TranscriptionResult
from llmcore.providers.mistral_provider import MistralProvider
from llmcore.providers.zai_provider import ZaiProvider


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in ("MISTRAL_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _conforms(provider) -> list[str]:
    return [c.value for c in provider.media_capabilities() if not isinstance(provider, CAPABILITY_PROTOCOLS[c])]


def test_declared_capabilities_are_backed():
    mistral = MistralProvider({"api_key": "k"})
    zai = ZaiProvider({"api_key": "id.secret"})
    assert {c.value for c in mistral.media_capabilities()} == {"tts", "asr", "asr_stream", "ocr"}
    assert {c.value for c in zai.media_capabilities()} == {"tts", "asr", "ocr", "image_generate", "video_generate"}
    assert _conforms(mistral) == [] and _conforms(zai) == []
    assert isinstance(zai, MediaJobPoller)
    assert zai.media_execution(MediaCapability.VIDEO_GENERATE) is MediaExecution.ASYNC_JOB
    assert mistral.media_execution(MediaCapability.TTS) is MediaExecution.REQUEST_RESPONSE


async def test_bridge_translates_arguments_and_wraps_results():
    p = MistralProvider({"api_key": "k"})
    p.generate_speech = AsyncMock(return_value=SpeechResult(audio_data=b"mp3", format="mp3", model="voxtral-mini-tts-3", voice="v"))
    p.transcribe_audio = AsyncMock(return_value=TranscriptionResult(text="hello", model="voxtral-mini-latest", duration_seconds=6.0))
    p.ocr = AsyncMock(return_value=OCRResult(pages=[{"index": 0, "markdown": "# Title"}], model="mistral-ocr-latest", pages_processed=1))

    speech = await p.synthesize_speech_media("hi", voice="v", audio_format="wav", speed=1.1)
    assert p.generate_speech.await_args.kwargs == {"model": None, "voice": "v", "response_format": "wav", "speed": 1.1}
    assert speech.artifact.data == b"mp3" and speech.usage.characters == 2

    text = await p.transcribe_media(audio=MediaRef(data=b"RIFF", mime_type="audio/wav"), language="en", timestamps=True)
    assert p.transcribe_audio.await_args.args == (b"RIFF",)
    assert p.transcribe_audio.await_args.kwargs["timestamp_granularities"] == ["segment"]
    assert text.text == "hello" and text.usage.audio_minutes == 0.1

    await p.ocr_media(document=MediaRef.from_url("https://x/doc.pdf"), pages=[0])
    assert p.ocr.await_args.args == ("https://x/doc.pdf",) and p.ocr.await_args.kwargs["pages"] == [0]


def test_mistral_speech_has_a_default_model():
    # Without one the API answers 422 "No model provided for speech" (live).
    assert MistralProvider({"api_key": "k"}).tts_model == "voxtral-mini-tts-3"
    assert MistralProvider({"api_key": "k", "tts_model": "voxtral-mini-tts-latest"}).tts_model == "voxtral-mini-tts-latest"


async def test_zai_video_is_a_polled_job():
    p = ZaiProvider({"api_key": "id.secret"})
    p.generate_video = AsyncMock(return_value={"id": "task-1", "model": "cogvideox-3", "task_status": "PROCESSING"})
    job = await p.generate_video_media("a cat", first_frame=MediaRef(data=b"\x89PNG", mime_type="image/png"),
                                       duration_seconds=5, seed=3)
    kwargs = p.generate_video.await_args.kwargs
    assert kwargs["image_url"].startswith("data:image/png;base64,") and kwargs["duration"] == 5 and kwargs["wait"] is False
    assert job.provider_job_id == "task-1" and job.status is MediaJobStatus.RUNNING

    p.retrieve_video_result = AsyncMock(return_value={
        "task_status": "SUCCESS", "video_result": [{"url": "https://cdn/v.mp4", "cover_image_url": "https://cdn/c.jpg"}]})
    job = await p.poll_media_job(job)
    assert job.status is MediaJobStatus.SUCCEEDED
    assert job.artifacts[0].kind is MediaKind.VIDEO and job.artifacts[0].uri == "https://cdn/v.mp4"

    assert (await p.poll_media_job(job)) is job  # terminal jobs are returned unchanged
    with pytest.raises(ProviderError, match="cannot be cancelled"):
        await p.cancel_media_job(job)


async def test_zai_failed_task():
    p = ZaiProvider({"api_key": "id.secret"})
    p.generate_video = AsyncMock(return_value={"id": "t", "task_status": "FAIL", "error": "content policy"})
    job = await p.generate_video_media("x")
    assert job.status is MediaJobStatus.FAILED and job.error == "content policy"


async def test_gemini_transcription_reads_the_audio_transcription_part():
    pytest.importorskip("google.genai")
    from llmcore.providers.gemini_provider import GeminiProvider

    p = GeminiProvider({"api_key": "k"})
    assert MediaCapability.ASR in p.media_capabilities()
    part = SimpleNamespace(audio_transcription=SimpleNamespace(text="Gemini transcribes this."), text=None)
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        usage_metadata=SimpleNamespace(prompt_token_count=129, candidates_token_count=13),
        model_version="gemini-3.5-transcribe", response_id="r1",
    )
    p._client = MagicMock()
    p._client.aio.models.generate_content = AsyncMock(return_value=response)
    result = await p.transcribe_media(audio=MediaRef(data=b"ID3", mime_type="audio/mpeg"), language="en")
    call = p._client.aio.models.generate_content.await_args.kwargs
    assert call["model"] == "gemini-3.5-transcribe" and len(call["contents"]) == 2
    assert result.text == "Gemini transcribes this." and result.usage.input_tokens == 129
