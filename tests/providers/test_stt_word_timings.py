# tests/providers/test_stt_word_timings.py
"""Word timings, speakers and language survive ``transcribe_audio``.

For OpenAI, DeepInfra, Mistral and Deepgram the provider's response carries
word timings, speaker labels and the language; these tests feed hand-written
response fixtures (shaped like each API's documented / SDK-typed response)
through the real parsing code and assert they land in the
:class:`~llmcore.models_multimodal.TranscriptionResult`. Also covers the
DeepInfra TTS request body (no literal ``extra_body`` key).

No network: OpenAI's SDK client is an ``AsyncMock``, DeepInfra and Mistral go
through ``respx``, Deepgram through a fake SDK namespace.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from llmcore.models_multimodal import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    attach_words_to_segments,
)


@pytest.fixture(autouse=True)
def _offline_tiktoken(monkeypatch):
    """Patch tiktoken so provider construction never hits the network."""
    import tiktoken

    fake_enc = MagicMock()
    fake_enc.encode = lambda s: list(s.encode("utf-8"))
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda *_a, **_k: fake_enc)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda *_a, **_k: fake_enc)


def _multipart_fields(request: httpx.Request) -> list[tuple[str, str]]:
    """(name, value) pairs of the non-file fields of a multipart request."""
    import email.parser
    import email.policy

    ctype = request.headers["content-type"]
    raw = b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + request.content
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    out = []
    for part in msg.iter_parts():
        if part.get_filename():
            continue
        out.append((part.get_param("name", header="content-disposition"), part.get_content()))
    return out


# ---------------------------------------------------------------------------
# attach_words_to_segments
# ---------------------------------------------------------------------------
def test_attach_words_by_midpoint_and_gap():
    segs = [
        TranscriptionSegment(text="a b", start=0.0, end=1.0),
        TranscriptionSegment(text="c", start=2.0, end=3.0),
    ]
    words = [
        TranscriptionWord(word="a", start=0.0, end=0.4),
        TranscriptionWord(word="b", start=0.5, end=1.1),  # straddles the end
        TranscriptionWord(word="x", start=1.6, end=1.8),  # gap, nearer seg 2
        TranscriptionWord(word="c", start=2.1, end=2.9),
    ]
    out = attach_words_to_segments(segs, words)
    assert [w.word for w in out[0].words] == ["a", "b"]
    assert [w.word for w in out[1].words] == ["x", "c"]


def test_attach_words_without_segments_makes_one():
    words = [
        TranscriptionWord(word="hi", start=0.1, end=0.3),
        TranscriptionWord(word="there", start=0.4, end=0.9),
    ]
    out = attach_words_to_segments([], words, text="hi there")
    assert len(out) == 1
    assert (out[0].text, out[0].start, out[0].end) == ("hi there", 0.1, 0.9)
    assert len(out[0].words) == 2


def test_segment_words_round_trip_through_artifact():
    res = TranscriptionResult(
        text="hi",
        model="m",
        segments=[
            TranscriptionSegment(
                text="hi",
                start=0.0,
                end=0.5,
                words=[TranscriptionWord(word="hi", start=0.0, end=0.5, speaker="1")],
            )
        ],
    )
    back = TranscriptionResult.from_artifact(res.to_artifact())
    assert back.segments[0].words[0].speaker == "1"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
@pytest.fixture
def openai_provider():
    with patch("llmcore.providers.openai_provider.AsyncOpenAI") as client_cls:
        client_cls.return_value = MagicMock()
        from llmcore.providers.openai_provider import OpenAIProvider

        return OpenAIProvider({"api_key": "sk-test", "_instance_name": "openai"})


_OPENAI_VERBOSE = {
    "text": "Olá mundo. Tudo bem?",
    "language": "portuguese",
    "duration": 2.4,
    "segments": [
        {"id": 0, "seek": 0, "start": 0.0, "end": 1.1, "text": " Olá mundo.",
         "tokens": [1], "temperature": 0.0, "avg_logprob": -0.2,
         "compression_ratio": 1.0, "no_speech_prob": 0.01},
        {"id": 1, "seek": 0, "start": 1.2, "end": 2.4, "text": " Tudo bem?",
         "tokens": [2], "temperature": 0.0, "avg_logprob": -0.3,
         "compression_ratio": 1.0, "no_speech_prob": 0.01},
    ],
    "words": [
        {"word": "Olá", "start": 0.0, "end": 0.5},
        {"word": "mundo", "start": 0.5, "end": 1.1},
        {"word": "Tudo", "start": 1.2, "end": 1.7},
        {"word": "bem", "start": 1.7, "end": 2.4},
    ],
}


async def test_openai_whisper_requests_and_keeps_words(openai_provider):
    from openai.types.audio import TranscriptionVerbose

    create = AsyncMock(return_value=TranscriptionVerbose.model_validate(_OPENAI_VERBOSE))
    openai_provider._client.audio.transcriptions.create = create

    res = await openai_provider.transcribe_audio(b"RIFF", model="whisper-1")

    kw = create.call_args.kwargs
    assert kw["response_format"] == "verbose_json"
    assert kw["timestamp_granularities"] == ["word", "segment"]
    assert res.language == "portuguese"  # kept as the API gives it
    assert res.duration_seconds == 2.4
    assert [s.text for s in res.segments] == [" Olá mundo.", " Tudo bem?"]
    assert [w.word for w in res.segments[0].words] == ["Olá", "mundo"]
    assert [w.word for w in res.segments[1].words] == ["Tudo", "bem"]
    assert res.segments[1].words[1].end == 2.4
    assert res.segments[0].words[0].probability is None  # OpenAI gives none


async def test_openai_language_hint_wins(openai_provider):
    from openai.types.audio import TranscriptionVerbose

    openai_provider._client.audio.transcriptions.create = AsyncMock(
        return_value=TranscriptionVerbose.model_validate(_OPENAI_VERBOSE)
    )
    res = await openai_provider.transcribe_audio(b"RIFF", model="whisper-1", language="pt")
    assert res.language == "pt"


async def test_openai_gpt4o_transcribe_stays_json(openai_provider):
    from openai.types.audio import Transcription

    create = AsyncMock(return_value=Transcription.model_validate({"text": "hello"}))
    openai_provider._client.audio.transcriptions.create = create

    res = await openai_provider.transcribe_audio(b"RIFF", model="gpt-4o-transcribe")

    kw = create.call_args.kwargs
    assert "response_format" not in kw  # json (the only format it supports)
    assert "timestamp_granularities" not in kw
    assert res.text == "hello"
    assert res.segments == []


async def test_openai_diarized_json_keeps_speakers(openai_provider):
    from openai.types.audio import TranscriptionDiarized

    body = {
        "text": "Hi. Hello.",
        "duration": 2.0,
        "task": "transcribe",
        "segments": [
            {"id": "seg_0", "type": "transcript.text.segment", "start": 0.0,
             "end": 0.8, "text": "Hi.", "speaker": "A"},
            {"id": "seg_1", "type": "transcript.text.segment", "start": 1.0,
             "end": 2.0, "text": "Hello.", "speaker": "B"},
        ],
    }
    openai_provider._client.audio.transcriptions.create = AsyncMock(
        return_value=TranscriptionDiarized.model_validate(body)
    )
    res = await openai_provider.transcribe_audio(
        b"RIFF", model="gpt-4o-transcribe-diarize", response_format="diarized_json"
    )
    assert [s.speaker for s in res.segments] == ["A", "B"]


# ---------------------------------------------------------------------------
# DeepInfra (OpenAI-compatible Whisper over httpx)
# ---------------------------------------------------------------------------
@pytest.fixture
def deepinfra_provider():
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    return DeepInfraProvider({"api_key": "test-key", "_instance_name": "deepinfra", "timeout": 5})


_DI_URL = "https://api.deepinfra.com/v1/audio/transcriptions"


@respx.mock
async def test_deepinfra_whisper_requests_and_keeps_flat_words(deepinfra_provider):
    route = respx.post(_DI_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "hello there",
                "language": "en",
                "duration": 2.5,
                "segments": [
                    {"id": 0, "text": "hello", "start": 0.0, "end": 1.0},
                    {"id": 1, "text": "there", "start": 1.0, "end": 2.5},
                ],
                "words": [
                    {"word": "hello", "start": 0.1, "end": 0.9},
                    {"word": "there", "start": 1.1, "end": 2.4},
                ],
            },
        )
    )
    res = await deepinfra_provider.transcribe_audio(b"RIFF", model="openai/whisper-large-v3")

    fields = _multipart_fields(route.calls.last.request)
    assert ("response_format", "verbose_json") in fields
    assert ("timestamp_granularities[]", "word") in fields
    assert ("timestamp_granularities[]", "segment") in fields
    assert res.language == "en"
    assert [w.word for w in res.segments[0].words] == ["hello"]
    assert [w.word for w in res.segments[1].words] == ["there"]


@respx.mock
async def test_deepinfra_keeps_nested_words_with_probability(deepinfra_provider):
    respx.post(_DI_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "oi",
                "language": "pt",
                "segments": [
                    {"text": "oi", "start": 0.0, "end": 0.6,
                     "words": [{"word": "oi", "start": 0.05, "end": 0.55,
                                "probability": 0.93}]},
                ],
            },
        )
    )
    res = await deepinfra_provider.transcribe_audio(b"RIFF")
    w = res.segments[0].words[0]
    assert (w.word, w.start, w.end, w.probability) == ("oi", 0.05, 0.55, 0.93)
    assert res.language == "pt"


@respx.mock
async def test_deepinfra_tts_sends_extras_top_level(deepinfra_provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"RIFFfakeaudio")
    )
    await deepinfra_provider.generate_speech(
        "olá",
        voice="af_bella",
        model="hexgrad/Kokoro-82M",
        instructions="calm, slow",
        extra_body={"lang": "pt"},
        seed=7,
    )
    body = json.loads(route.calls.last.request.content)
    assert "extra_body" not in body
    assert body["instructions"] == "calm, slow"
    assert body["lang"] == "pt"
    assert body["seed"] == 7
    assert body["voice"] == "af_bella"


@respx.mock
async def test_deepinfra_tts_extras_do_not_override_named_args(deepinfra_provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"x")
    )
    await deepinfra_provider.generate_speech("hi", voice="a", extra_body={"voice": "b"})
    assert json.loads(route.calls.last.request.content)["voice"] == "a"


# ---------------------------------------------------------------------------
# Mistral (Voxtral)
# ---------------------------------------------------------------------------
@respx.mock
async def test_mistral_segments_speakers_language():
    from llmcore.providers.mistral_provider import MistralProvider

    provider = MistralProvider({"api_key": "test-key", "_instance_name": "mistral"})
    route = respx.post("https://api.mistral.ai/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "voxtral-mini-2507",
                "text": "Bom dia. Bom dia!",
                "language": "pt",
                "segments": [
                    {"type": "transcription_segment", "text": "Bom dia.",
                     "start": 0.0, "end": 0.9, "speaker_id": "speaker_0"},
                    {"type": "transcription_segment", "text": "Bom dia!",
                     "start": 1.2, "end": 2.0, "speaker_id": "speaker_1",
                     "score": 0.8},
                ],
                "usage": {"prompt_audio_seconds": 2},
            },
        )
    )
    res = await provider.transcribe_audio(
        b"RIFF",
        model="voxtral-mini-latest",
        timestamp_granularities=["segment"],
        diarize=True,
        context_bias=["Voxtral", "BellaVox"],
    )

    request = route.calls.last.request
    # A real multipart body, not one labelled with the client's JSON default.
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    fields = _multipart_fields(request)
    # Array fields go as repeated form fields, not as a JSON string.
    assert ("timestamp_granularities", "segment") in fields
    assert ("context_bias", "Voxtral") in fields
    assert ("context_bias", "BellaVox") in fields
    assert res.language == "pt"
    assert [(s.text, s.start, s.end, s.speaker) for s in res.segments] == [
        ("Bom dia.", 0.0, 0.9, "speaker_0"),
        ("Bom dia!", 1.2, 2.0, "speaker_1"),
    ]
    await provider.close()


# ---------------------------------------------------------------------------
# Deepgram (pre-recorded)
# ---------------------------------------------------------------------------
def _deepgram_provider(monkeypatch, response: dict[str, Any], captured: dict[str, Any]):
    from llmcore.providers import deepgram_provider as dgmod

    if not dgmod.deepgram_available:
        pytest.skip("deepgram-sdk not installed")

    class _NS:
        pass

    class _FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.listen = _NS()
            self.listen.v1 = _NS()
            self.listen.v1.media = _NS()

            async def _transcribe_file(**kwargs: Any) -> dict[str, Any]:
                captured.update(kwargs)
                return response

            self.listen.v1.media.transcribe_file = _transcribe_file

    monkeypatch.setattr(dgmod, "AsyncDeepgramClient", _FakeClient)
    return dgmod.DeepgramProvider(
        {"api_key": "dg-test-key", "_instance_name": "deepgram", "mip_opt_out": True}
    )


def _dg_word(word: str, start: float, end: float, conf: float, speaker: int | None = None):
    w: dict[str, Any] = {
        "word": word.lower().strip(".,!?"),
        "punctuated_word": word,
        "start": start,
        "end": end,
        "confidence": conf,
    }
    if speaker is not None:
        w["speaker"] = speaker
        w["speaker_confidence"] = 0.9
    return w


_DG_WORDS = [
    _dg_word("Olá,", 0.0, 0.4, 0.99, 0),
    _dg_word("mundo.", 0.4, 0.9, 0.97, 0),
    _dg_word("Oi!", 1.5, 1.8, 0.95, 1),
]


async def test_deepgram_utterances_words_speakers_language(monkeypatch):
    response = {
        "metadata": {"request_id": "r1", "duration": 2.0, "models": ["nova-3"]},
        "results": {
            "channels": [
                {
                    "detected_language": "pt",
                    "language_confidence": 0.98,
                    "alternatives": [
                        {"transcript": "Olá, mundo. Oi!", "confidence": 0.97,
                         "words": _DG_WORDS},
                    ],
                }
            ],
            "utterances": [
                {"start": 0.0, "end": 0.9, "confidence": 0.98, "channel": 0,
                 "transcript": "Olá, mundo.", "words": _DG_WORDS[:2], "speaker": 0,
                 "id": "u1"},
                {"start": 1.5, "end": 1.8, "confidence": 0.95, "channel": 0,
                 "transcript": "Oi!", "words": _DG_WORDS[2:], "speaker": 1,
                 "id": "u2"},
            ],
        },
    }
    captured: dict[str, Any] = {}
    p = _deepgram_provider(monkeypatch, response, captured)
    res = await p.transcribe_audio(
        b"RIFF", model="nova-3", diarize=True, utterances=True, detect_language=True
    )

    assert captured["mip_opt_out"] is True  # unchanged behaviour
    assert res.language == "pt"
    assert res.metadata["language_confidence"] == 0.98
    assert [s.speaker for s in res.segments] == ["0", "1"]
    w0 = res.segments[0].words[0]
    assert (w0.word, w0.start, w0.end, w0.probability, w0.speaker) == (
        "Olá,", 0.0, 0.4, 0.99, "0"
    )
    assert [w.word for w in res.segments[1].words] == ["Oi!"]
    assert res.segments[1].words[0].speaker == "1"


async def test_deepgram_paragraphs_get_channel_words(monkeypatch):
    response = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Olá, mundo. Oi!",
                            "words": _DG_WORDS,
                            "paragraphs": {
                                "transcript": "\nSpeaker 0: Olá, mundo.\n\nSpeaker 1: Oi!",
                                "paragraphs": [
                                    {"sentences": [{"text": "Olá, mundo.", "start": 0.0,
                                                    "end": 0.9}],
                                     "speaker": 0, "num_words": 2, "start": 0.0,
                                     "end": 0.9},
                                    {"sentences": [{"text": "Oi!", "start": 1.5,
                                                    "end": 1.8}],
                                     "speaker": 1, "num_words": 1, "start": 1.5,
                                     "end": 1.8},
                                ],
                            },
                        }
                    ]
                }
            ]
        },
    }
    p = _deepgram_provider(monkeypatch, response, {})
    res = await p.transcribe_audio(b"RIFF", language="pt-BR", diarize=True, smart_format=True)

    assert res.language == "pt-BR"  # no detection requested: the hint
    assert [(s.text, s.speaker) for s in res.segments] == [("Olá, mundo.", "0"), ("Oi!", "1")]
    assert [w.word for w in res.segments[0].words] == ["Olá,", "mundo."]
    assert [w.word for w in res.segments[1].words] == ["Oi!"]


async def test_deepgram_words_only_and_multi_language(monkeypatch):
    response = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {"transcript": "Olá, mundo. Oi!", "words": _DG_WORDS,
                         "languages": ["pt", "en"]}
                    ]
                }
            ]
        },
    }
    p = _deepgram_provider(monkeypatch, response, {})
    res = await p.transcribe_audio(b"RIFF", language="multi")

    assert res.language == "pt"
    assert len(res.segments) == 1
    seg = res.segments[0]
    assert (seg.text, seg.start, seg.end, seg.speaker) == ("Olá, mundo. Oi!", 0.0, 1.8, None)
    assert len(seg.words) == 3
