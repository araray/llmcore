# tests/media/test_huggingface_media_adapter.py
"""Hugging Face as a media adapter (spec phase M8).

The gate for this phase is the **custom weights / private repo** path. That is
not a model id on a shared router: it is a dedicated Inference Endpoint the
caller deployed, at their own URL, possibly serving weights nobody else can see.

Hugging Face is also structurally unlike every other media adapter, and the
tests below pin the two consequences that live validation surfaced:

* the same model id is routed to different third-party providers per *task*,
  each of which knows the model by **its own id**; and
* those providers receive **their own request shape**, not Hugging Face's — so
  this is the one adapter where the SDK is the right default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from llmcore.exceptions import ProviderError
from llmcore.media import MediaExecution, MediaKind, MediaRef
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider
from llmcore.providers.huggingface_provider import HuggingFaceProvider

HUB = "https://huggingface.co"
ROUTER = "https://router.huggingface.co"
FLUX = "black-forest-labs/FLUX.1-schnell"
WHISPER = "openai/whisper-large-v3-turbo"
KOKORO = "hexgrad/Kokoro-82M"

BASE: dict[str, Any] = {"api_key": "hf_test", "_instance_name": "huggingface"}


@pytest.fixture
def provider() -> HuggingFaceProvider:
    return HuggingFaceProvider(dict(BASE))


def _mapping(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"inferenceProviderMapping": list(entries)}


def _mock_mapping(model: str, *entries: dict[str, Any]):
    return respx.get(f"{HUB}/api/models/{model}").mock(
        return_value=httpx.Response(200, json=_mapping(*entries))
    )


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_is_media_capable(self, provider):
        assert isinstance(provider, MediaCapableProvider)

    def test_capability_spread(self, provider):
        assert {c.value for c in provider.media_capabilities()} == {
            "image_generate", "asr", "tts"
        }

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_nothing_is_a_job(self, provider):
        """Inference answers in one request; there is no queue to poll."""
        for cap in provider.media_capabilities():
            assert provider.media_execution(cap) is MediaExecution.REQUEST_RESPONSE

    def test_media_models_are_configurable(self):
        p = HuggingFaceProvider({**BASE, "media_models": {"tts": "my/voice"}})
        assert p._media_model_for("tts", None) == "my/voice"
        assert p._media_model_for("tts", "explicit/m") == "explicit/m"
        assert p._media_model_for("asr", None) == WHISPER

    def test_chat_still_works_alongside_media(self, provider):
        """Unlike fal/Replicate/ElevenLabs, this provider serves chat too."""
        assert hasattr(provider, "chat_completion")
        assert provider.default_model


# ---------------------------------------------------------------------------
# Provider routing — the HF-specific problem
# ---------------------------------------------------------------------------


class TestInferenceRouting:
    @respx.mock
    async def test_routing_records_provider_and_its_own_model_id(self, provider):
        """fal-ai knows FLUX as `fal-ai/flux/schnell`, not by its Hub id."""
        _mock_mapping(
            FLUX,
            {
                "provider": "fal-ai",
                "status": "live",
                "providerId": "fal-ai/flux/schnell",
                "task": "text-to-image",
            },
        )
        routing = await provider.get_inference_routing(FLUX)
        assert routing["text-to-image"] == ("fal-ai", "fal-ai/flux/schnell")
        await provider.close()

    @respx.mock
    async def test_providers_in_error_state_are_skipped(self, provider):
        """A provider listed as `error` is advertised but rejects the call."""
        _mock_mapping(
            FLUX,
            {"provider": "together", "status": "error", "providerId": FLUX,
             "task": "text-to-image"},
            {"provider": "nscale", "status": "live", "providerId": FLUX,
             "task": "text-to-image"},
        )
        routing = await provider.get_inference_routing(FLUX)
        assert routing["text-to-image"][0] == "nscale"
        await provider.close()

    @respx.mock
    async def test_routing_is_cached(self, provider):
        route = _mock_mapping(FLUX, {"provider": "nscale", "status": "live",
                                     "providerId": FLUX, "task": "text-to-image"})
        await provider.get_inference_routing(FLUX)
        await provider.get_inference_routing(FLUX)
        assert route.call_count == 1
        await provider.close()

    @respx.mock
    async def test_routing_failure_degrades(self, provider):
        """Routing is an optimization; losing it must not fail the call."""
        respx.get(f"{HUB}/api/models/{FLUX}").mock(return_value=httpx.Response(500))
        assert await provider.get_inference_routing(FLUX) == {}
        await provider.close()

    @respx.mock
    async def test_a_dict_shaped_mapping_is_accepted(self, provider):
        """The Hub has returned both a list and an object here."""
        respx.get(f"{HUB}/api/models/{FLUX}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "inferenceProviderMapping": {
                        "nscale": {"status": "live", "providerId": FLUX,
                                   "task": "text-to-image"}
                    }
                },
            )
        )
        routing = await provider.get_inference_routing(FLUX)
        assert routing["text-to-image"] == ("nscale", FLUX)
        await provider.close()


class TestUrlConstruction:
    def test_hf_inference_uses_the_models_path(self, provider):
        assert provider._router_url("hf-inference", WHISPER) == (
            f"{ROUTER}/hf-inference/models/{WHISPER}"
        )

    def test_third_party_providers_are_addressed_directly(self, provider):
        """Live-found: using /models/ here returns `Model not supported`."""
        assert provider._router_url("fal-ai", "fal-ai/flux/schnell") == (
            f"{ROUTER}/fal-ai/fal-ai/flux/schnell"
        )

    @respx.mock
    async def test_url_uses_the_providers_own_model_id(self, provider):
        _mock_mapping(FLUX, {"provider": "fal-ai", "status": "live",
                             "providerId": "fal-ai/flux/schnell", "task": "text-to-image"})
        assert await provider._media_url("image_generate", FLUX) == (
            f"{ROUTER}/fal-ai/fal-ai/flux/schnell"
        )
        await provider.close()

    @respx.mock
    async def test_unroutable_models_fall_back_to_hf_inference(self, provider):
        respx.get(f"{HUB}/api/models/{FLUX}").mock(return_value=httpx.Response(404))
        assert await provider._media_url("image_generate", FLUX) == (
            f"{ROUTER}/hf-inference/models/{FLUX}"
        )
        await provider.close()

    @respx.mock
    async def test_a_configured_provider_is_honoured(self):
        p = HuggingFaceProvider({**BASE, "provider": "nscale"})
        _mock_mapping(FLUX, {"provider": "nscale", "status": "live",
                             "providerId": FLUX, "task": "text-to-image"})
        assert await p._media_url("image_generate", FLUX) == f"{ROUTER}/nscale/{FLUX}"
        await p.close()


# ---------------------------------------------------------------------------
# The gate: dedicated Inference Endpoints
# ---------------------------------------------------------------------------


class TestDedicatedEndpoints:
    """Custom weights and private repos are a URL you deployed, not a model id."""

    @respx.mock
    async def test_a_configured_endpoint_wins_outright(self):
        p = HuggingFaceProvider(
            {**BASE, "endpoints": {"asr": "https://my-ep.endpoints.huggingface.cloud"}}
        )
        hub = respx.get(f"{HUB}/api/models/{WHISPER}")
        assert await p._media_url("asr", WHISPER) == (
            "https://my-ep.endpoints.huggingface.cloud"
        )
        assert hub.call_count == 0, "a deployment you own needs no routing lookup"
        await p.close()

    @respx.mock
    async def test_trailing_slashes_are_normalized(self):
        p = HuggingFaceProvider({**BASE, "endpoints": {"tts": "https://my-ep.example.com/"}})
        assert await p._media_url("tts", KOKORO) == "https://my-ep.example.com"
        await p.close()

    async def test_an_endpoint_switches_that_capability_to_direct_http(self):
        """Your own deployment speaks the standard schema, so no SDK translation."""
        p = HuggingFaceProvider({**BASE, "endpoints": {"tts": "https://my-ep.example.com"}})
        assert p._use_sdk_for("tts") is False
        assert p._use_sdk_for("image_generate") is True, "other capabilities unchanged"
        await p.close()

    @respx.mock
    async def test_generation_posts_to_the_endpoint(self):
        p = HuggingFaceProvider({**BASE, "endpoints": {"tts": "https://my-ep.example.com"}})
        route = respx.post("https://my-ep.example.com").mock(
            return_value=httpx.Response(
                200, content=b"audio", headers={"content-type": "audio/wav"}
            )
        )
        result = await p.synthesize_speech_media("hello")
        assert result.artifacts[0].data == b"audio"
        assert result.artifacts[0].mime_type == "audio/wav"
        import json

        assert json.loads(route.calls[0].request.content)["inputs"] == "hello"
        await p.close()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """The documented exception to llmcore's direct-REST-first rule."""

    def test_json_body_tasks_default_to_the_sdk(self, provider):
        """Third-party providers receive their own body shape, not HF's."""
        assert provider._use_sdk_for("image_generate") is True
        assert provider._use_sdk_for("tts") is True

    def test_binary_input_tasks_go_direct(self, provider):
        """The SDK sends raw audio with no Content-Type, which is rejected."""
        assert provider._use_sdk_for("asr") is False

    def test_explicit_httpx_overrides_everything(self):
        p = HuggingFaceProvider({**BASE, "media_backend": "httpx"})
        assert all(
            p._use_sdk_for(c) is False for c in ("image_generate", "tts", "asr")
        )

    def test_explicit_sdk_overrides_everything(self):
        p = HuggingFaceProvider(
            {**BASE, "media_backend": "sdk", "endpoints": {"tts": "https://e"}}
        )
        assert all(p._use_sdk_for(c) is True for c in ("image_generate", "tts", "asr"))


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestImageGeneration:
    async def test_sdk_path_returns_png_bytes(self, provider):
        image = MagicMock()

        def _save(buffer, format):  # noqa: A002 - PIL's own parameter name
            buffer.write(b"PNGDATA")

        image.save.side_effect = _save
        provider._client.text_to_image = AsyncMock(return_value=image)

        result = await provider.generate_image_media("a cat")
        artifact = result.artifacts[0]
        assert artifact.kind is MediaKind.IMAGE
        assert artifact.data == b"PNGDATA"
        assert artifact.mime_type == "image/png"
        assert artifact.provenance.generator == FLUX
        await provider.close()

    async def test_size_is_parsed_into_dimensions(self, provider):
        provider._client.text_to_image = AsyncMock(return_value=b"raw")
        await provider.generate_image_media("a cat", size="512x768")
        kwargs = provider._client.text_to_image.await_args.kwargs
        assert kwargs["width"] == 512 and kwargs["height"] == 768
        await provider.close()

    async def test_a_malformed_size_is_ignored_not_fatal(self, provider):
        provider._client.text_to_image = AsyncMock(return_value=b"raw")
        result = await provider.generate_image_media("a cat", size="huge")
        assert result.artifacts[0].data == b"raw"
        await provider.close()


class TestSpeech:
    async def test_tts_returns_audio(self, provider):
        provider._client.text_to_speech = AsyncMock(return_value=b"AUDIO")
        result = await provider.synthesize_speech_media("hello there")
        assert result.artifacts[0].kind is MediaKind.AUDIO
        assert result.artifacts[0].data == b"AUDIO"
        assert result.usage.basis == "per_character"
        assert result.usage.characters == 11
        await provider.close()

    async def test_no_consent_record_is_invented(self, provider):
        """HF serves open-weight voices and tracks no per-voice consent, so
        claiming anything here would be fabricating it."""
        provider._client.text_to_speech = AsyncMock(return_value=b"AUDIO")
        result = await provider.synthesize_speech_media("hi")
        assert result.artifacts[0].provenance.consent is None
        assert result.artifacts[0].provenance.generator == KOKORO
        await provider.close()


class TestTranscription:
    @respx.mock
    async def test_the_callers_mime_type_reaches_the_request(self, provider):
        """Live-found: hf-inference rejects a missing or octet-stream type."""
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        route = respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(200, json={"text": "hello there"})
        )
        result = await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"ID3", mime_type="audio/mpeg")
        )
        assert route.calls[0].request.headers["content-type"] == "audio/mpeg"
        assert result.artifacts[0].text == "hello there"
        assert result.artifacts[0].kind is MediaKind.TEXT
        await provider.close()

    @respx.mock
    async def test_remote_audio_is_fetched_then_posted(self, provider):
        """Unlike fal/Replicate, this API takes bytes, not a URL."""
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        respx.get("https://example.invalid/a.mp3").mock(
            return_value=httpx.Response(
                200, content=b"REMOTE", headers={"content-type": "audio/mpeg"}
            )
        )
        route = respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(200, json={"text": "fetched"})
        )
        result = await provider.transcribe_media(
            audio=MediaRef.from_url("https://example.invalid/a.mp3")
        )
        assert route.calls[0].request.content == b"REMOTE"
        assert result.artifacts[0].text == "fetched"
        await provider.close()

    @respx.mock
    async def test_a_list_response_is_unwrapped(self, provider):
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(200, json=[{"text": "first"}])
        )
        result = await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        assert result.artifacts[0].text == "first"
        await provider.close()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrorMapping:
    @respx.mock
    @pytest.mark.parametrize(
        ("status", "match"),
        [
            (401, "authentication failed"),
            (402, "no inference credits"),
            (404, "not found"),
            (429, "rate limit"),
            (503, "loading or unavailable"),
            (500, "inference error"),
        ],
    )
    async def test_status_mapping(self, provider, status, match):
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(status, text="boom")
        )
        with pytest.raises(ProviderError, match=match):
            await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        await provider.close()

    @respx.mock
    async def test_402_says_the_token_is_valid(self, provider):
        """Out of credit is not a credential problem."""
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(402, text="x")
        )
        with pytest.raises(ProviderError) as exc:
            await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        assert "the token is valid" in str(exc.value)
        assert exc.value.retryable is False
        await provider.close()

    @respx.mock
    async def test_a_loading_model_is_retryable(self, provider):
        _mock_mapping(WHISPER, {"provider": "hf-inference", "status": "live",
                                "providerId": WHISPER,
                                "task": "automatic-speech-recognition"})
        respx.post(f"{ROUTER}/hf-inference/models/{WHISPER}").mock(
            return_value=httpx.Response(503, text="loading")
        )
        with pytest.raises(ProviderError) as exc:
            await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        assert exc.value.retryable is True
        await provider.close()
