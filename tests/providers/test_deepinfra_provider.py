# tests/providers/test_deepinfra_provider.py
"""Tests for :mod:`llmcore.providers.deepinfra_provider`.

Design notes
------------
* The **real** ``openai`` SDK is used (it is a hard dependency of the provider).
  We deliberately do *not* replace ``sys.modules["openai"]`` with a ``MagicMock``
  because doing so breaks ``isinstance`` / class-identity for OpenAIProvider
  subclasses and produces spurious cross-test failures.
* ``tiktoken`` is patched offline (an autouse fixture) so provider construction
  never reaches out to download a BPE vocabulary. Mocking tiktoken is safe — it
  does not affect provider class identity.
* The OpenAI-compatible chat / image / embedding surfaces are exercised by
  patching the parent method or the AsyncOpenAI client; the native audio and
  model-discovery endpoints are exercised with ``respx`` (httpx interception).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _offline_tiktoken(monkeypatch):
    """Patch tiktoken so provider construction never hits the network."""
    import tiktoken

    fake_enc = MagicMock()
    fake_enc.encode = lambda s: list(s.encode("utf-8"))
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda *_a, **_k: fake_enc)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda *_a, **_k: fake_enc)
    yield


@pytest.fixture
def provider():
    """A DeepInfraProvider with an explicit test key and short timeout."""
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    return DeepInfraProvider({"api_key": "test-key", "_instance_name": "deepinfra", "timeout": 5})


# ---------------------------------------------------------------------------
# Construction / identity / key resolution
# ---------------------------------------------------------------------------
def test_init_defaults(provider):
    from llmcore.providers.deepinfra_provider import (
        DEFAULT_DEEPINFRA_BASE_URL,
        DEFAULT_DEEPINFRA_MODEL,
    )

    assert provider.get_name() == "deepinfra"
    assert provider.base_url == DEFAULT_DEEPINFRA_BASE_URL
    assert provider.default_model == DEFAULT_DEEPINFRA_MODEL
    # Native base is the OpenAI base with the trailing /openai stripped.
    assert provider._native_base == "https://api.deepinfra.com/v1"
    assert provider._default_context_length == 32768


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1"),
        ("https://api.deepinfra.com/v1/openai/", "https://api.deepinfra.com/v1"),
        ("https://proxy.local/inference", "https://proxy.local/inference"),
        (None, "https://api.deepinfra.com/v1"),
    ],
)
def test_native_base_derivation(base_url, expected):
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    assert DeepInfraProvider._derive_native_base(base_url) == expected


def test_api_key_from_env_token(monkeypatch):
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setenv("DEEPINFRA_TOKEN", "tok-123")
    p = DeepInfraProvider({"_instance_name": "deepinfra"})
    assert p.api_key == "tok-123"


def test_api_key_from_env_api_key_alias(monkeypatch):
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    monkeypatch.delenv("DEEPINFRA_TOKEN", raising=False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "alias-456")
    p = DeepInfraProvider({"_instance_name": "deepinfra"})
    assert p.api_key == "alias-456"


def test_custom_default_context_length():
    from llmcore.providers.deepinfra_provider import DeepInfraProvider

    p = DeepInfraProvider({"api_key": "k", "default_context_length": 8000})
    assert p._default_context_length == 8000
    assert p.get_max_context_length("unknown/model") == 8000


# ---------------------------------------------------------------------------
# Supported parameters
# ---------------------------------------------------------------------------
def test_supported_parameters_superset(provider):
    params = provider.get_supported_parameters("deepseek-ai/DeepSeek-R1")
    for key in (
        "reasoning",
        "reasoning_effort",
        "service_tier",
        "top_k",
        "min_p",
        "repetition_penalty",
        "extra_body",
        "max_tokens",
        "temperature",
    ):
        assert key in params, f"missing supported parameter: {key}"
    assert params["reasoning_effort"]["enum"] == ["none", "low", "medium", "high"]
    assert params["service_tier"]["enum"] == ["default", "priority"]


# ---------------------------------------------------------------------------
# chat_completion: DeepInfra extras must be routed via extra_body
# ---------------------------------------------------------------------------
async def test_chat_completion_routes_extra_body(provider, monkeypatch):
    from llmcore.providers.openai_provider import OpenAIProvider

    captured = {}

    async def fake_super(
        self, context, model=None, stream=False, tools=None, tool_choice=None, **kw
    ):
        captured.update(kw)
        captured["model"] = model
        return {"ok": True}

    monkeypatch.setattr(OpenAIProvider, "chat_completion", fake_super)

    out = await provider.chat_completion(
        [],
        model="deepseek-ai/DeepSeek-R1",
        reasoning_effort="high",
        service_tier="priority",
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.1,
        prompt_cache_key="sess-1",
        temperature=0.7,  # standard OpenAI param: must stay top-level
    )
    assert out == {"ok": True}
    eb = captured["extra_body"]
    assert eb == {
        "reasoning_effort": "high",
        "service_tier": "priority",
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "prompt_cache_key": "sess-1",
    }
    # Standard params remain top-level (not swallowed into extra_body).
    assert captured["temperature"] == 0.7
    assert "reasoning_effort" not in captured


async def test_chat_completion_merges_existing_extra_body(provider, monkeypatch):
    from llmcore.providers.openai_provider import OpenAIProvider

    captured = {}

    async def fake_super(
        self, context, model=None, stream=False, tools=None, tool_choice=None, **kw
    ):
        captured.update(kw)
        return {}

    monkeypatch.setattr(OpenAIProvider, "chat_completion", fake_super)
    await provider.chat_completion([], extra_body={"foo": 1}, min_p=0.1)
    assert captured["extra_body"] == {"foo": 1, "min_p": 0.1}


# ---------------------------------------------------------------------------
# Model discovery (/v1/models)
# ---------------------------------------------------------------------------
@respx.mock
async def test_get_models_details(provider):
    route = respx.get("https://api.deepinfra.com/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
                        "owned_by": "qwen",
                        "created": 1715,
                        "metadata": {
                            "context_length": 128000,
                            "max_tokens": 16384,
                            "tags": ["text-generation", "vision"],
                            "description": "VLM",
                            "pricing": {"input": 0.2},
                        },
                    },
                    {
                        "id": "BAAI/bge-m3",
                        "metadata": {
                            "context_length": 8192,
                            "tags": ["embeddings"],
                        },
                    },
                ]
            },
        )
    )
    models = await provider.get_models_details()
    assert route.called
    by_id = {m.id: m for m in models}

    vl = by_id["Qwen/Qwen2.5-VL-7B-Instruct"]
    assert vl.context_length == 128000
    assert vl.max_output_tokens == 16384
    assert vl.model_type == "chat"
    assert vl.supports_vision is True
    assert vl.supports_tools is True

    emb = by_id["BAAI/bge-m3"]
    assert emb.model_type == "embedding"
    assert emb.supports_tools is False
    # Discovery cache is populated for get_max_context_length.
    assert provider._discovery_context_cache["BAAI/bge-m3"] == 8192


@respx.mock
async def test_get_models_details_falls_back_on_http_error(provider, monkeypatch):
    respx.get("https://api.deepinfra.com/v1/models").mock(
        return_value=httpx.Response(500, text="boom")
    )
    from llmcore.providers.openai_provider import OpenAIProvider

    sentinel = [MagicMock(id="fallback-model")]
    monkeypatch.setattr(OpenAIProvider, "get_models_details", AsyncMock(return_value=sentinel))
    models = await provider.get_models_details()
    assert models is sentinel


# ---------------------------------------------------------------------------
# Context-length resolution chain
# ---------------------------------------------------------------------------
def test_get_max_context_length_from_registry(provider):
    # Seed card deepseek-ai/DeepSeek-V3 ships in default_cards/deepinfra.
    assert provider.get_max_context_length("deepseek-ai/DeepSeek-V3") == 131072


def test_get_max_context_length_from_cache(provider):
    provider._discovery_context_cache["acme/foo"] = 24000
    assert provider.get_max_context_length("acme/foo") == 24000


def test_get_max_context_length_default(provider):
    assert provider.get_max_context_length("totally/unknown") == 32768


# ---------------------------------------------------------------------------
# Text-to-Speech (/v1/audio/speech)
# ---------------------------------------------------------------------------
@respx.mock
async def test_generate_speech(provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"RIFFfakeaudio")
    )
    result = await provider.generate_speech(
        "hello world",
        voice="af_bella",
        model="hexgrad/Kokoro-82M",
        response_format="wav",
        speed=1.25,
    )
    assert route.called
    assert result.audio_data == b"RIFFfakeaudio"
    assert result.format == "wav"
    assert result.model == "hexgrad/Kokoro-82M"
    assert result.voice == "af_bella"

    import json

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == "hello world"
    assert body["model"] == "hexgrad/Kokoro-82M"
    assert body["response_format"] == "wav"
    assert body["speed"] == 1.25
    assert body["voice"] == "af_bella"
    # Bearer auth header present.
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"


@respx.mock
async def test_generate_speech_error(provider):
    respx.post("https://api.deepinfra.com/v1/audio/speech").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    from llmcore.exceptions import ProviderError

    with pytest.raises(ProviderError):
        await provider.generate_speech("hi")


# ---------------------------------------------------------------------------
# Speech-to-Text (/v1/audio/transcriptions, /v1/audio/translations)
# ---------------------------------------------------------------------------
@respx.mock
async def test_transcribe_audio(provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "hello there",
                "language": "en",
                "duration": 2.5,
                "segments": [
                    {"text": "hello", "start": 0.0, "end": 1.0},
                    {"text": "there", "start": 1.0, "end": 2.5},
                ],
            },
        )
    )
    result = await provider.transcribe_audio(
        b"\x00\x01audio", model="openai/whisper-large", language="en"
    )
    assert route.called
    assert result.text == "hello there"
    assert result.language == "en"
    assert result.duration_seconds == 2.5
    assert len(result.segments) == 2
    assert result.segments[0].text == "hello"
    assert result.segments[1].end == 2.5
    assert result.model == "openai/whisper-large"


@respx.mock
async def test_transcribe_timestamp_forces_verbose(provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    await provider.transcribe_audio(
        b"audio", response_format="json", timestamp_granularities=["word"]
    )
    sent = route.calls.last.request.content.decode("utf-8", "ignore")
    # multipart body contains the upgraded response_format
    assert "verbose_json" in sent
    assert "timestamp_granularities" in sent


@respx.mock
async def test_translate_audio(provider):
    route = respx.post("https://api.deepinfra.com/v1/audio/translations").mock(
        return_value=httpx.Response(200, json={"text": "translated"})
    )
    result = await provider.translate_audio(b"audio")
    assert route.called
    assert result.text == "translated"


# ---------------------------------------------------------------------------
# Image generation (defaults + forced b64_json)
# ---------------------------------------------------------------------------
async def test_generate_image_applies_defaults(provider, monkeypatch):
    from llmcore.providers.deepinfra_provider import DEFAULT_DEEPINFRA_IMAGE_MODEL
    from llmcore.providers.openai_provider import OpenAIProvider

    captured = {}

    async def fake_super(self, prompt, **kw):
        captured.update(kw)
        captured["prompt"] = prompt
        return MagicMock()

    monkeypatch.setattr(OpenAIProvider, "generate_image", fake_super)
    await provider.generate_image("an astronaut", size="1024x1024", response_format="url")
    assert captured["model"] == DEFAULT_DEEPINFRA_IMAGE_MODEL
    # response_format is forced to b64_json regardless of caller input.
    assert captured["response_format"] == "b64_json"
    assert captured["size"] == "1024x1024"


# ---------------------------------------------------------------------------
# Embeddings (/v1/openai/embeddings via SDK)
# ---------------------------------------------------------------------------
async def test_create_embeddings_defaults(provider):
    from llmcore.providers.deepinfra_provider import DEFAULT_DEEPINFRA_EMBEDDING_MODEL

    resp_obj = MagicMock()
    resp_obj.model_dump = MagicMock(
        return_value={"data": [{"embedding": [0.1, 0.2]}], "model": "m"}
    )
    provider._client = MagicMock()
    provider._client.embeddings.create = AsyncMock(return_value=resp_obj)

    out = await provider.create_embeddings(["hello"])
    provider._client.embeddings.create.assert_awaited_once()
    kwargs = provider._client.embeddings.create.await_args.kwargs
    assert kwargs["model"] == DEFAULT_DEEPINFRA_EMBEDDING_MODEL
    assert kwargs["encoding_format"] == "float"
    assert kwargs["input"] == ["hello"]
    assert out["data"][0]["embedding"] == [0.1, 0.2]
