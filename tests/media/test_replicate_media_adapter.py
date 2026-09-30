# tests/media/test_replicate_media_adapter.py
"""Replicate as a media adapter (spec phase M7).

The spec is explicit that this must be *one generic prediction adapter plus
model-schema descriptors*, **not a class per model**. Replicate hosts tens of
thousands of community models; a library cannot enumerate them, and any
hardcoded field mapping is stale within a release.

So the load-bearing behaviour under test is the **schema-driven input
mapping**: llmcore's canonical protocol arguments are mapped onto whatever each
model actually calls them, read from that model's own published schema.
``flux-schnell`` wants ``prompt`` and ``num_outputs``; ``whisper`` wants
``audio``. Neither is hardcoded.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.media import (
    MediaCapability,
    MediaExecution,
    MediaKind,
    MediaManager,
    MediaRef,
)
from llmcore.media.models import MediaJob, MediaJobStatus
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider, MediaJobPoller
from llmcore.providers.replicate_provider import ReplicateProvider

API = "https://api.replicate.com"
FLUX = "black-forest-labs/flux-schnell"
WHISPER = "openai/whisper"
VERSION = "c846a69991daf4c0" + "0" * 48

BASE_CONFIG: dict[str, Any] = {
    "api_key": "r8-test",
    "_instance_name": "replicate",
    "backend": "httpx",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def provider() -> ReplicateProvider:
    return ReplicateProvider(dict(BASE_CONFIG))


def _model_payload(properties: dict[str, Any], version: str = VERSION) -> dict[str, Any]:
    return {
        "latest_version": {
            "id": version,
            "openapi_schema": {
                "components": {"schemas": {"Input": {"properties": properties}}}
            },
        }
    }


def _mock_schema(model: str, properties: dict[str, Any], version: str = VERSION):
    return respx.get(f"{API}/v1/models/{model}").mock(
        return_value=httpx.Response(200, json=_model_payload(properties, version))
    )


def _prediction(status: str = "starting", **extra: Any) -> dict[str, Any]:
    payload = {
        "id": "pred-1",
        "status": status,
        "urls": {
            "get": f"{API}/v1/predictions/pred-1",
            "cancel": f"{API}/v1/predictions/pred-1/cancel",
        },
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Construction & conformance
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("env_var", ["REPLICATE_API_TOKEN", "REPLICATE_API_KEY"])
    def test_token_from_either_env_spelling(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "from-env")
        assert ReplicateProvider({"backend": "httpx"})._api_key == "from-env"

    def test_missing_token_raises(self):
        with pytest.raises(ConfigError, match="Replicate API token not found"):
            ReplicateProvider({"backend": "httpx"})

    def test_backend_defaults_to_direct_rest(self, provider):
        assert provider._backend == "httpx"

    def test_auto_prefers_httpx(self):
        with patch.multiple(
            "llmcore.providers.replicate_provider",
            httpx_available=True,
            replicate_sdk_available=True,
        ):
            assert ReplicateProvider._resolve_backend("auto") == "httpx"

    def test_no_transport_raises(self):
        with patch.multiple(
            "llmcore.providers.replicate_provider",
            httpx_available=False,
            replicate_sdk_available=False,
        ):
            with pytest.raises(ConfigError, match="requires 'httpx'"):
                ReplicateProvider(dict(BASE_CONFIG))

    def test_models_are_configurable(self):
        p = ReplicateProvider({**BASE_CONFIG, "models": {"image_generate": "me/custom"}})
        assert p._model_for("image_generate", None) == "me/custom"
        assert p._model_for("image_generate", "explicit/m") == "explicit/m"
        assert p._model_for("asr", None) == WHISPER


class TestConformance:
    def test_is_media_capable_and_a_poller(self, provider):
        assert isinstance(provider, MediaCapableProvider)
        assert isinstance(provider, MediaJobPoller)

    def test_capability_spread(self, provider):
        assert {c.value for c in provider.media_capabilities()} == {
            "image_generate", "image_edit", "image_upscale",
            "video_generate", "asr", "tts", "music",
        }

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_everything_is_a_prediction(self, provider):
        for cap in provider.media_capabilities():
            assert provider.media_execution(cap) is MediaExecution.ASYNC_JOB

    def test_opts_into_webhooks(self, provider):
        assert provider.accepts_webhook_url is True

    def test_chat_is_refused_with_a_pointer(self, provider):
        import asyncio

        from llmcore.models import Message, Role

        with pytest.raises(ProviderError, match=r"llm\.media"):
            asyncio.run(provider.chat_completion([Message(role=Role.USER, content="hi")]))


# ---------------------------------------------------------------------------
# The design: model-schema descriptors
# ---------------------------------------------------------------------------


class TestSchemaDescriptors:
    @respx.mock
    async def test_schema_is_fetched_and_cached(self, provider):
        route = _mock_schema(FLUX, {"prompt": {}, "num_outputs": {}})
        assert set(await provider.get_input_schema(FLUX)) == {"prompt", "num_outputs"}
        await provider.get_input_schema(FLUX)
        assert route.call_count == 1, "one lookup per model, not per call"
        await provider.close()

    @respx.mock
    async def test_a_version_pin_shares_the_base_model_schema(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        assert await provider.get_input_schema(f"{FLUX}:{VERSION}") == {"prompt": {}}
        await provider.close()

    @respx.mock
    async def test_schema_failure_degrades_rather_than_raising(self, provider):
        """The schema is an optimization for field naming. Losing it should cost
        accuracy, not the submission."""
        respx.get(f"{API}/v1/models/{FLUX}").mock(return_value=httpx.Response(500, text="x"))
        assert await provider.get_input_schema(FLUX) == {}
        await provider.close()


class TestInputMapping:
    """The heart of the design: one adapter, many models, no hardcoding."""

    @respx.mock
    async def test_canonical_names_map_to_what_the_model_declares(self, provider):
        _mock_schema(FLUX, {"prompt": {}, "num_outputs": {}, "seed": {}})
        mapped = await provider._map_inputs(FLUX, {"prompt": "a cat", "n": 2, "seed": 7}, {})
        assert mapped == {"prompt": "a cat", "num_outputs": 2, "seed": 7}
        await provider.close()

    @respx.mock
    async def test_a_different_model_gets_different_names(self, provider):
        _mock_schema(WHISPER, {"audio": {}, "language": {}})
        mapped = await provider._map_inputs(WHISPER, {"audio": "u", "language": "en"}, {})
        assert mapped == {"audio": "u", "language": "en"}
        await provider.close()

    @respx.mock
    async def test_alias_resolution_picks_the_declared_spelling(self, provider):
        """A model calling it `input_image` still receives the caller's image."""
        _mock_schema("x/y", {"input_image": {}, "text": {}})
        mapped = await provider._map_inputs("x/y", {"image": "u", "prompt": "p"}, {})
        assert mapped == {"input_image": "u", "text": "p"}
        await provider.close()

    @respx.mock
    async def test_explicit_kwargs_always_win(self, provider):
        """The caller knows their model better than this mapping does."""
        _mock_schema(FLUX, {"prompt": {}, "num_outputs": {}})
        mapped = await provider._map_inputs(FLUX, {"prompt": "a", "n": 2}, {"num_outputs": 9})
        assert mapped["num_outputs"] == 9
        await provider.close()

    @respx.mock
    async def test_none_values_are_dropped(self, provider):
        _mock_schema(FLUX, {"prompt": {}, "seed": {}})
        mapped = await provider._map_inputs(FLUX, {"prompt": "a", "seed": None}, {})
        assert "seed" not in mapped
        await provider.close()

    @respx.mock
    async def test_without_a_schema_the_canonical_name_is_used(self, provider):
        """A 422 naming the field beats a silent drop."""
        respx.get(f"{API}/v1/models/x/y").mock(return_value=httpx.Response(404))
        mapped = await provider._map_inputs("x/y", {"prompt": "a", "image": "u"}, {})
        assert mapped == {"prompt": "a", "image": "u"}
        await provider.close()

    async def test_schema_use_can_be_disabled(self):
        p = ReplicateProvider({**BASE_CONFIG, "use_schema": False})
        assert await p._map_inputs(FLUX, {"prompt": "a", "n": 2}, {}) == {
            "prompt": "a",
            "num_outputs": 2,
        }
        await p.close()


# ---------------------------------------------------------------------------
# Which creation route — the live-found bug
# ---------------------------------------------------------------------------


class TestPredictionRouting:
    """Official models run unversioned; community models need a version pin.

    Live validation found `openai/whisper` 404ing on the official route. The
    first fix tried it and fell back on the 404 — but that costs *two*
    creation requests, and Replicate throttles accounts under $5 of credit to a
    burst of 1, turning a working call into a 429. The version comes from the
    schema lookup instead, which is a GET and does not count against it.
    """

    @respx.mock
    async def test_a_known_version_is_used_in_one_request(self, provider):
        _mock_schema(WHISPER, {"audio": {}})
        official = respx.post(f"{API}/v1/models/{WHISPER}/predictions")
        versioned = respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.transcribe_media(audio=MediaRef.from_url("https://e/a.mp3"))
        assert versioned.call_count == 1
        assert official.call_count == 0, "no speculative request that could 429"
        assert json.loads(versioned.calls[0].request.content)["version"] == VERSION
        await provider.close()

    @respx.mock
    async def test_an_explicit_version_pin_is_honoured(self, provider):
        versioned = respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.generate_image_media("x", model=f"{FLUX}:abc123")
        assert json.loads(versioned.calls[0].request.content)["version"] == "abc123"
        await provider.close()

    @respx.mock
    async def test_a_model_with_no_version_uses_the_official_route(self, provider):
        respx.get(f"{API}/v1/models/{FLUX}").mock(
            return_value=httpx.Response(200, json={"latest_version": {}})
        )
        official = respx.post(f"{API}/v1/models/{FLUX}/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.generate_image_media("x")
        assert official.call_count == 1
        await provider.close()

    @respx.mock
    async def test_schema_disabled_uses_the_official_route(self):
        p = ReplicateProvider({**BASE_CONFIG, "use_schema": False})
        official = respx.post(f"{API}/v1/models/{FLUX}/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await p.generate_image_media("x")
        assert official.call_count == 1
        await p.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @respx.mock
    async def test_submit_returns_a_tracked_job(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction("starting"))
        )
        job = await provider.generate_image_media("a cat")
        assert job.status is MediaJobStatus.QUEUED
        assert job.provider_job_id == "pred-1"
        assert job.poll_url == f"{API}/v1/predictions/pred-1"
        await provider.close()

    @respx.mock
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("starting", MediaJobStatus.QUEUED),
            ("processing", MediaJobStatus.RUNNING),
            ("succeeded", MediaJobStatus.SUCCEEDED),
            ("failed", MediaJobStatus.FAILED),
            ("canceled", MediaJobStatus.CANCELED),
        ],
    )
    async def test_status_mapping(self, provider, state, expected):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction("starting"))
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/v1/predictions/pred-1").mock(
            return_value=httpx.Response(
                200, json=_prediction(state, output=["https://cdn/x.png"])
            )
        )
        assert (await provider.poll_media_job(job)).status is expected
        await provider.close()

    @respx.mock
    async def test_poll_uses_the_url_replicate_returned(self, provider):
        """The fal lesson: don't rebuild routes the provider owns."""
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(
                201,
                json=_prediction(
                    "starting", urls={"get": f"{API}/v1/custom/route", "cancel": "c"}
                ),
            )
        )
        job = await provider.generate_image_media("x")
        route = respx.get(f"{API}/v1/custom/route").mock(
            return_value=httpx.Response(200, json=_prediction("succeeded", output=[]))
        )
        await provider.poll_media_job(job)
        assert route.call_count == 1
        await provider.close()

    @respx.mock
    async def test_error_fails_the_job(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction("starting"))
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/v1/predictions/pred-1").mock(
            return_value=httpx.Response(200, json=_prediction("failed", error="NSFW"))
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "NSFW" in job.error
        await provider.close()

    @respx.mock
    async def test_an_immediately_succeeded_prediction_is_applied(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(
                201, json=_prediction("succeeded", output=["https://cdn/a.png"])
            )
        )
        job = await provider.generate_image_media("x")
        assert job.succeeded
        assert job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()

    @respx.mock
    async def test_cancel_uses_the_returned_url(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction("processing"))
        )
        job = await provider.generate_image_media("x")
        route = respx.post(f"{API}/v1/predictions/pred-1/cancel").mock(
            return_value=httpx.Response(200, json=_prediction("canceled"))
        )
        job = await provider.cancel_media_job(job)
        assert route.call_count == 1
        assert job.status is MediaJobStatus.CANCELED
        await provider.close()

    @respx.mock
    async def test_poll_on_terminal_is_a_noop(self, provider):
        job = MediaJob(
            capability=MediaCapability.IMAGE_GENERATE,
            provider="replicate",
            model=FLUX,
            status=MediaJobStatus.SUCCEEDED,
            provider_job_id="pred-1",
        )
        assert (await provider.poll_media_job(job)) is job
        await provider.close()


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


class TestOutputExtraction:
    """Outputs are schema-defined, so they vary per model rather than per vendor."""

    def test_list_of_uris(self, provider):
        arts = provider._artifacts_from_output(
            ["https://cdn/a.png", "https://cdn/b.png"], MediaCapability.IMAGE_GENERATE
        )
        assert [a.uri for a in arts] == ["https://cdn/a.png", "https://cdn/b.png"]
        assert all(a.kind is MediaKind.IMAGE for a in arts)

    def test_single_uri(self, provider):
        arts = provider._artifacts_from_output(
            "https://cdn/a.mp4", MediaCapability.VIDEO_GENERATE
        )
        assert arts[0].kind is MediaKind.VIDEO

    def test_bare_string_is_text_not_a_uri(self, provider):
        arts = provider._artifacts_from_output("hello there", MediaCapability.ASR)
        assert arts[0].kind is MediaKind.TEXT
        assert arts[0].text == "hello there"

    def test_object_output_yields_text_from_the_known_field(self, provider):
        """Live-verified against whisper, whose output is an object."""
        arts = provider._artifacts_from_output(
            {"transcription": "he hoped there would be stew", "detected_language": "en"},
            MediaCapability.ASR,
        )
        assert arts[0].kind is MediaKind.TEXT
        assert arts[0].provider_metadata["field"] == "transcription"

    def test_object_output_also_yields_its_files(self, provider):
        arts = provider._artifacts_from_output(
            {"transcription": "hi", "srt_file": "https://cdn/x.srt"}, MediaCapability.ASR
        )
        kinds = {a.kind for a in arts}
        assert MediaKind.TEXT in kinds
        assert any(a.uri == "https://cdn/x.srt" for a in arts)

    def test_unknown_shape_is_not_fatal(self, provider):
        assert provider._artifacts_from_output({"mystery": 42}, MediaCapability.ASR) == []
        assert provider._artifacts_from_output(None, MediaCapability.ASR) == []

    @pytest.mark.parametrize(
        ("capability", "kind"),
        [
            (MediaCapability.IMAGE_GENERATE, MediaKind.IMAGE),
            (MediaCapability.IMAGE_UPSCALE, MediaKind.IMAGE),
            (MediaCapability.VIDEO_GENERATE, MediaKind.VIDEO),
            (MediaCapability.TTS, MediaKind.AUDIO),
            (MediaCapability.MUSIC, MediaKind.AUDIO),
        ],
    )
    def test_kind_follows_the_capability(self, provider, capability, kind):
        arts = provider._artifacts_from_output(["https://cdn/x"], capability)
        assert arts[0].kind is kind


class TestMimeInference:
    def test_extension_wins(self, provider):
        assert provider._mime_for("https://cdn/a.png") == "image/png"

    def test_query_strings_are_ignored(self, provider):
        assert provider._mime_for("https://cdn/a.mp4?token=x") == "video/mp4"

    def test_falls_back_to_the_requested_format(self, provider):
        """Replicate output URLs often carry no extension. The format we asked
        for is grounded in the request; a guess would not be."""
        assert provider._mime_for("https://cdn/abc123", "webp") == "image/webp"

    def test_unknown_stays_unknown(self, provider):
        assert provider._mime_for("https://cdn/abc123") is None

    @respx.mock
    async def test_requested_format_reaches_the_artifact(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(
                201,
                json=_prediction(
                    "succeeded",
                    output=["https://cdn/no-extension"],
                    input={"prompt": "x", "output_format": "webp"},
                ),
            )
        )
        job = await provider.generate_image_media("x")
        assert job.artifacts[0].mime_type == "image/webp"
        await provider.close()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestInputRefs:
    @respx.mock
    async def test_remote_refs_pass_through(self, provider):
        _mock_schema(WHISPER, {"audio": {}})
        route = respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.transcribe_media(audio=MediaRef.from_url("https://e/a.mp3"))
        assert json.loads(route.calls[0].request.content)["input"]["audio"] == (
            "https://e/a.mp3"
        )
        await provider.close()

    @respx.mock
    async def test_local_bytes_become_a_data_uri(self, provider):
        _mock_schema(WHISPER, {"audio": {}})
        route = respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"ID3", mime_type="audio/mpeg")
        )
        audio = json.loads(route.calls[0].request.content)["input"]["audio"]
        assert audio.startswith("data:audio/mpeg;base64,")
        await provider.close()


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


class TestWebhooks:
    @respx.mock
    async def test_a_callback_url_is_registered(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        route = respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(201, json=_prediction())
        )
        await provider.generate_image_media("x", webhook_url="https://h/cb/tok")
        body = json.loads(route.calls[0].request.content)
        assert body["webhook"] == "https://h/cb/tok"
        assert body["webhook_events_filter"] == ["completed"]
        assert "webhook_url" not in body["input"], "must not leak into the model input"
        await provider.close()

    async def test_delivery_is_applied(self, provider):
        job = MediaJob(
            capability=MediaCapability.IMAGE_GENERATE,
            provider="replicate",
            model=FLUX,
            status=MediaJobStatus.RUNNING,
            provider_job_id="pred-1",
        )
        job = await provider.apply_webhook_payload(
            job, _prediction("succeeded", output=["https://cdn/a.png"])
        )
        assert job.succeeded
        assert job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()

    async def test_mismatched_id_is_not_trusted(self, provider):
        polled: list[str] = []

        async def _poll(job):
            polled.append(job.id)
            return job

        provider.poll_media_job = _poll
        job = MediaJob(
            capability=MediaCapability.IMAGE_GENERATE,
            provider="replicate",
            model=FLUX,
            status=MediaJobStatus.RUNNING,
            provider_job_id="pred-1",
        )
        await provider.apply_webhook_payload(
            job, {"id": "someone-elses", "status": "succeeded", "output": ["x"]}
        )
        assert len(polled) == 1
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
            (402, "billing reasons"),
            (404, "not found"),
            (422, "schema may use different field names"),
            (429, "burst of 1"),
            (500, "API error"),
        ],
    )
    async def test_status_mapping(self, provider, status, match):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(status, text="boom")
        )
        with pytest.raises(ProviderError, match=match):
            await provider.generate_image_media("x")
        await provider.close()

    @respx.mock
    async def test_402_says_the_token_is_valid(self, provider):
        """A billing stop is not a credential problem."""
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(return_value=httpx.Response(402, text="x"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        assert "the token is valid" in str(exc.value)
        assert "REPLICATE_API_TOKEN" not in str(exc.value)
        await provider.close()

    @respx.mock
    async def test_rate_limits_are_retryable(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(return_value=httpx.Response(429, text="x"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        assert exc.value.retryable is True
        await provider.close()

    @respx.mock
    async def test_transport_errors_are_retryable(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        assert exc.value.retryable is True
        await provider.close()


class TestRouting:
    def test_registered_for_the_catalog_capabilities(self, provider):
        media = MediaManager({"replicate": provider})
        assert "replicate" in media.who_can(MediaCapability.IMAGE_GENERATE)
        assert "replicate" in media.who_can(MediaCapability.VIDEO_GENERATE)

    @respx.mock
    async def test_router_dispatch_end_to_end(self, provider):
        _mock_schema(FLUX, {"prompt": {}})
        respx.post(f"{API}/v1/predictions").mock(
            return_value=httpx.Response(
                201, json=_prediction("succeeded", output=["https://cdn/a.png"])
            )
        )
        media = MediaManager({"replicate": provider})
        job = await media.images.generate("x", provider="replicate")
        assert job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()
