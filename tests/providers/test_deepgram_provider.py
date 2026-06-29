# tests/providers/test_deepgram_provider.py
"""Unit tests for :class:`llmcore.providers.deepgram_provider.DeepgramProvider`.

These tests mock the ``deepgram-sdk`` entirely (no network). They cover:

* construction / credential resolution (api_key, env var, access_token,
  self-hosted environment, missing-credential errors);
* ``BaseProvider`` conformance (identity, supported params, context length,
  token heuristics, content extraction, ``chat_completion`` refusal);
* batch media (``transcribe_audio`` file + URL, ``generate_speech``) including
  parameter merging, response extraction, and error mapping.

Streaming / Voice-Agent behaviour is tested separately in
``test_deepgram_streaming.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.models import Message
from llmcore.models_multimodal import SpeechResult, TranscriptionResult
from llmcore.providers import deepgram_provider as dgmod
from llmcore.providers.deepgram_provider import DeepgramProvider, deepgram_available

# The Deepgram SDK is an optional extra (``pip install llmcore[deepgram]``).
# When it is not installed the SDK symbols the provider wraps are ``None``, so
# these unit tests (which exercise real SDK types via fakes) cannot run; skip the
# whole module rather than fail, mirroring the other optional-dependency suites.
pytestmark = pytest.mark.skipif(
    not deepgram_available,
    reason="deepgram-sdk not installed (optional extra: pip install llmcore[deepgram])",
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    """A stand-in for ``AsyncDeepgramClient`` with controllable namespaces.

    The provider only touches ``listen.v1.media.{transcribe_file,transcribe_url}``,
    ``speak.v1.audio.generate``, and (for ``close``) the private wrapper chain.
    Tests set the leaf callables they need.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

        # Build the nested namespace objects lazily via simple attribute holders.
        self.listen = _NS()
        self.listen.v1 = _NS()
        self.listen.v1.media = _NS()
        self.speak = _NS()
        self.speak.v1 = _NS()
        self.speak.v1.audio = _NS()
        self.agent = _NS()
        self.read = _NS()
        self.auth = _NS()
        self.manage = _NS()

        # A benign close chain: client._client_wrapper.httpx_client.httpx_client.aclose()
        self._client_wrapper = _NS()
        self._client_wrapper.httpx_client = _NS()
        self._client_wrapper.httpx_client.httpx_client = _AcloseSpy()


class _NS:
    """A trivial attribute namespace."""


class _AcloseSpy:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def _aiter_bytes(chunks: list[bytes]):
    """Async generator yielding the given byte chunks (fake TTS stream)."""
    for c in chunks:
        yield c


def _install_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the provider module's ``AsyncDeepgramClient`` with the fake."""
    monkeypatch.setattr(dgmod, "AsyncDeepgramClient", _FakeAsyncClient)
    monkeypatch.setattr(dgmod, "deepgram_available", True)


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any] | None = None
) -> DeepgramProvider:
    """Construct a provider with a fake client and a default api_key."""
    _install_fake_client(monkeypatch)
    cfg: dict[str, Any] = {"api_key": "dg-test-key", "_instance_name": "deepgram"}
    if config:
        cfg.update(config)
    return DeepgramProvider(cfg)


# ---------------------------------------------------------------------------
# Construction / credentials
# ---------------------------------------------------------------------------


def test_init_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    assert p.api_key == "dg-test-key"
    assert p.get_name() == "deepgram"
    assert isinstance(p.client, _FakeAsyncClient)
    assert p.client.init_kwargs["api_key"] == "dg-test-key"
    assert p.client.init_kwargs["timeout"] == 60
    assert p.client.init_kwargs["max_retries"] == 2


def test_init_api_key_from_custom_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_DG_KEY", "from-env")
    p = _make_provider(monkeypatch, {"api_key": None, "api_key_env_var": "MY_DG_KEY"})
    assert p.api_key == "from-env"


def test_init_api_key_from_default_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "default-env-key")
    _install_fake_client(monkeypatch)
    p = DeepgramProvider({"_instance_name": "deepgram"})
    assert p.api_key == "default-env-key"


def test_init_access_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch, {"access_token": "bearer-tok"})
    assert p.client.init_kwargs.get("access_token") == "bearer-tok"


def test_init_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_client(monkeypatch)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        DeepgramProvider({"_instance_name": "deepgram"})


def test_init_missing_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dgmod, "deepgram_available", False)
    with pytest.raises(ImportError):
        DeepgramProvider({"api_key": "k"})


def test_init_self_hosted_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _EnvSpyClient(_FakeAsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(dgmod, "AsyncDeepgramClient", _EnvSpyClient)
    monkeypatch.setattr(dgmod, "deepgram_available", True)
    p = DeepgramProvider(
        {
            "api_key": "k",
            "base_url": "https://dg.internal",
            "ws_url": "wss://dg.internal",
        }
    )
    env = captured.get("environment")
    assert env is not None
    # Overridden legs honoured; un-set legs fall back to production defaults.
    assert env.base == "https://dg.internal"
    assert env.production == "wss://dg.internal"
    assert env.agent == "wss://agent.deepgram.com"
    assert p.api_key == "k"


def test_capability_defaults_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {
        "stt": {"model": "nova-2", "smart_format": True, "streaming": {"interim_results": True}},
        "tts": {"model": "aura-luna-en", "streaming": {"flush_on_close": True}},
    }
    p = _make_provider(monkeypatch, cfg)
    assert p._stt_defaults["model"] == "nova-2"
    assert p._stt_stream_defaults["interim_results"] is True
    assert "streaming" not in p._stt_defaults  # popped out
    assert p._tts_defaults["model"] == "aura-luna-en"
    assert p._tts_stream_defaults["flush_on_close"] is True


# ---------------------------------------------------------------------------
# BaseProvider conformance
# ---------------------------------------------------------------------------


def test_get_supported_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    params = p.get_supported_parameters()
    assert "smart_format" in params["stt_batch"]
    assert "diarize" in params["stt_batch"]
    assert "encoding" in params["tts_batch"]
    assert "interim_results" in params["stt_streaming"]
    assert "eot_threshold" in params["flux"]
    assert "notes" in params


def test_get_max_context_length_default(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch, {"fallback_context_length": 1234})
    assert p.get_max_context_length() == 1234


def test_get_max_context_length_from_card(monkeypatch: pytest.MonkeyPatch) -> None:
    # nova-3 ships as a real card with the STT nominal context.
    p = _make_provider(monkeypatch)
    assert p.get_max_context_length("nova-3") == 1_000_000


@pytest.mark.asyncio
async def test_count_tokens_is_char_count(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    assert await p.count_tokens("hello") == 5
    assert await p.count_tokens("") == 0


@pytest.mark.asyncio
async def test_count_message_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    msgs = [Message(role="user", content="hi"), Message(role="assistant", content="yo")]
    # 2 + 4 + 2 + 4
    assert await p.count_message_tokens(msgs) == 12


def test_extract_response_content_transcription_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    assert p.extract_response_content({"text": "the transcript"}) == "the transcript"


def test_extract_response_content_raw_deepgram(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    raw = {"results": {"channels": [{"alternatives": [{"transcript": "raw text"}]}]}}
    assert p.extract_response_content(raw) == "raw text"


def test_extract_response_content_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    assert p.extract_response_content({}) == ""
    assert p.extract_response_content("not a dict") == ""  # type: ignore[arg-type]


def test_extract_delta_content(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    assert p.extract_delta_content({"text": "partial"}) == "partial"
    assert p.extract_delta_content({}) == ""


@pytest.mark.asyncio
async def test_chat_completion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    with pytest.raises(ProviderError) as ei:
        await p.chat_completion([Message(role="user", content="hi")])
    assert ei.value.status_code == 400
    assert "speech/audio" in str(ei.value).lower() or "transcribe" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_get_models_details_from_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    details = await p.get_models_details()
    ids = {d.id for d in details}
    assert "nova-3" in ids
    assert "aura-2-thalia-en" in ids
    by_id = {d.id: d for d in details}
    assert by_id["nova-3"].model_type == "stt"
    assert by_id["aura-2-thalia-en"].model_type == "tts"
    assert by_id["nova-3"].provider_name == "deepgram"


@pytest.mark.asyncio
async def test_close_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    spy = p.client._client_wrapper.httpx_client.httpx_client
    await p.close()
    assert spy.closed is True


@pytest.mark.asyncio
async def test_warm_up_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    # Should not raise and should not call any client methods.
    await p.warm_up()


# ---------------------------------------------------------------------------
# Batch STT
# ---------------------------------------------------------------------------


def _sample_stt_response() -> dict[str, Any]:
    return {
        "metadata": {
            "request_id": "req-123",
            "duration": 12.5,
            "models": ["nova-3"],
        },
        "results": {
            "channels": [{"alternatives": [{"transcript": "hello world", "confidence": 0.987}]}],
            "utterances": [{"transcript": "hello world", "start": 0.0, "end": 1.2, "speaker": 0}],
            "summary": {"short": "a greeting"},
        },
    }


@pytest.mark.asyncio
async def test_transcribe_audio_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_transcribe_file(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _sample_stt_response()

    p.client.listen.v1.media.transcribe_file = _fake_transcribe_file

    result = await p.transcribe_audio(
        b"RIFFfakeaudio", model="nova-3", language="en", diarize=True, utterances=True
    )

    assert isinstance(result, TranscriptionResult)
    assert result.text == "hello world"
    assert result.duration_seconds == 12.5
    assert result.model == "nova-3"
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.segments[0].speaker == "0"
    assert result.metadata["request_id"] == "req-123"
    assert result.metadata["summary"] == {"short": "a greeting"}
    # Param merge: explicit + kwargs forwarded; audio passed as request=.
    assert captured["request"] == b"RIFFfakeaudio"
    assert captured["model"] == "nova-3"
    assert captured["language"] == "en"
    assert captured["diarize"] is True
    assert captured["utterances"] is True
    # Provider-level mip_opt_out default folded in.
    assert captured["mip_opt_out"] is False


@pytest.mark.asyncio
async def test_transcribe_audio_uses_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(
        monkeypatch, {"stt": {"model": "nova-2", "smart_format": True, "punctuate": True}}
    )
    captured: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _sample_stt_response()

    p.client.listen.v1.media.transcribe_file = _fake
    await p.transcribe_audio(b"x")
    # No explicit model -> config default applies.
    assert captured["model"] == "nova-2"
    assert captured["smart_format"] is True
    assert captured["punctuate"] is True


@pytest.mark.asyncio
async def test_transcribe_audio_url(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_url(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _sample_stt_response()

    p.client.listen.v1.media.transcribe_url = _fake_url
    result = await p.transcribe_audio(b"", url="https://example.com/a.wav", model="nova-3")
    assert result.text == "hello world"
    assert captured["url"] == "https://example.com/a.wav"
    assert "request" not in captured


@pytest.mark.asyncio
async def test_transcribe_audio_prompt_maps_to_keyterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _sample_stt_response()

    p.client.listen.v1.media.transcribe_file = _fake
    await p.transcribe_audio(b"x", prompt="Kubernetes")
    assert captured["keyterm"] == ["Kubernetes"]


@pytest.mark.asyncio
async def test_transcribe_audio_redact_list_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _sample_stt_response()

    p.client.listen.v1.media.transcribe_file = _fake
    await p.transcribe_audio(b"x", redact=["pci", "ssn"])
    # redact must be sent as a comma-joined string.
    assert captured["redact"] == "pci,ssn"


@pytest.mark.asyncio
async def test_transcribe_audio_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    with pytest.raises(FileNotFoundError):
        await p.transcribe_audio("/no/such/file.wav")


@pytest.mark.asyncio
async def test_transcribe_audio_maps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)

    class _Err(dgmod._DeepgramApiError):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.status_code = 401
            self.headers = {"x-dg-request-id": "abc"}
            self.body = "Unauthorized"

    async def _boom(**kwargs: Any) -> dict[str, Any]:
        raise _Err()

    p.client.listen.v1.media.transcribe_file = _boom
    with pytest.raises(ProviderError) as ei:
        await p.transcribe_audio(b"x")
    assert ei.value.status_code == 401
    assert ei.value.retry_after_seconds is None
    assert ei.value.provider_name == "deepgram"


# ---------------------------------------------------------------------------
# Batch TTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_speech_collects_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_generate(**kwargs: Any):
        captured.update(kwargs)
        return _aiter_bytes([b"AB", b"CD", b"EF"])

    p.client.speak.v1.audio.generate = _fake_generate

    result = await p.generate_speech("Hello there", model="aura-2-thalia-en")
    assert isinstance(result, SpeechResult)
    assert result.audio_data == b"ABCDEF"
    assert result.model == "aura-2-thalia-en"
    assert result.voice == "aura-2-thalia-en"
    assert result.format == "mp3"  # default encoding
    assert captured["text"] == "Hello there"
    assert captured["model"] == "aura-2-thalia-en"
    assert captured["encoding"] == "mp3"


@pytest.mark.asyncio
async def test_generate_speech_voice_as_model(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)

    def _fake_generate(**kwargs: Any):
        return _aiter_bytes([b"x"])

    p.client.speak.v1.audio.generate = _fake_generate
    # No model= -> voice is used as the model id.
    result = await p.generate_speech("hi", voice="aura-luna-en")
    assert result.model == "aura-luna-en"


@pytest.mark.asyncio
async def test_generate_speech_encoding_and_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_generate(**kwargs: Any):
        captured.update(kwargs)
        return _aiter_bytes([b"x"])

    p.client.speak.v1.audio.generate = _fake_generate
    result = await p.generate_speech(
        "hi", model="aura-2-apollo-en", response_format="linear16", speed=1.25, sample_rate=24000
    )
    assert captured["encoding"] == "linear16"
    assert captured["speed"] == 1.25
    assert captured["sample_rate"] == 24000
    assert result.format == "linear16"


@pytest.mark.asyncio
async def test_generate_speech_maps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)

    class _Err(dgmod._DeepgramApiError):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.status_code = 400
            self.headers = {}
            self.body = "bad request"

    def _fake_generate(**kwargs: Any):
        async def _gen():
            raise _Err()
            yield b""  # pragma: no cover - unreachable

        return _gen()

    p.client.speak.v1.audio.generate = _fake_generate
    with pytest.raises(ProviderError) as ei:
        await p.generate_speech("hi")
    assert ei.value.status_code == 400


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(pytest.main([__file__, "-v"])))
