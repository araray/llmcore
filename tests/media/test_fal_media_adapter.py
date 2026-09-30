# tests/media/test_fal_media_adapter.py
"""fal as a media adapter (spec phase M5) — the provider-neutrality test.

fal is a **marketplace**, not a first-party vendor, and its lifecycle differs
from everything migrated so far:

* every capability is a queue submission, including image generation;
* the "model" is an endpoint path, and the gallery changes constantly;
* inputs are URLs, so local bytes must be uploaded first;
* cancellation is a *request*, not a guarantee;
* output shapes vary per model rather than following one house schema.

The spec's rule for this phase is explicit: if the abstraction bends here, fix
the abstraction. These tests assert it did not have to.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.media import (
    JobPolicy,
    MediaCapability,
    MediaExecution,
    MediaJobStatus,
    MediaKind,
    MediaManager,
    MediaRef,
)
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider, MediaJobPoller
from llmcore.providers.fal_provider import FalProvider

QUEUE = "https://queue.fal.run"
FLUX = "fal-ai/flux/schnell"
FAST = JobPolicy(poll_initial_seconds=0.0, poll_max_seconds=0.0, job_timeout_seconds=5.0, jitter=0.0)

BASE_CONFIG: dict[str, Any] = {"api_key": "fal-test-key", "_instance_name": "fal", "timeout": 30}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FAL_KEY", "FAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def provider() -> FalProvider:
    return FalProvider({**BASE_CONFIG, "backend": "httpx"})


def _submission(request_id: str = "req-1", position: int = 3) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status_url": f"{QUEUE}/{FLUX}/requests/{request_id}/status",
        "response_url": f"{QUEUE}/{FLUX}/requests/{request_id}",
        "cancel_url": f"{QUEUE}/{FLUX}/requests/{request_id}/cancel",
        "queue_position": position,
    }


# ---------------------------------------------------------------------------
# Construction & conformance
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("env_var", ["FAL_KEY", "FAL_API_KEY"])
    def test_key_from_either_env_spelling(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "from-env")
        assert FalProvider({"backend": "httpx"})._api_key == "from-env"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "env")
        assert FalProvider({"api_key": "explicit", "backend": "httpx"})._api_key == "explicit"

    def test_missing_key_raises(self):
        with pytest.raises(ConfigError, match="fal API key not found"):
            FalProvider({"backend": "httpx"})

    def test_backend_defaults_to_direct_rest(self, provider):
        assert provider._backend == "httpx"

    def test_auto_prefers_httpx_over_the_sdk(self):
        with patch.multiple(
            "llmcore.providers.fal_provider", httpx_available=True, fal_sdk_available=True
        ):
            assert FalProvider._resolve_backend("auto") == "httpx"

    def test_unavailable_backend_falls_back(self):
        with patch.multiple(
            "llmcore.providers.fal_provider", httpx_available=True, fal_sdk_available=False
        ):
            assert FalProvider._resolve_backend("sdk") == "httpx"

    def test_no_transport_raises(self):
        with patch.multiple(
            "llmcore.providers.fal_provider", httpx_available=False, fal_sdk_available=False
        ):
            with pytest.raises(ConfigError, match="requires 'httpx'"):
                FalProvider(dict(BASE_CONFIG))


class TestConformance:
    def test_is_media_capable_and_a_poller(self, provider):
        assert isinstance(provider, MediaCapableProvider)
        assert isinstance(provider, MediaJobPoller)

    def test_declares_the_full_marketplace_spread(self, provider):
        caps = {c.value for c in provider.media_capabilities()}
        assert caps == {
            "image_generate", "image_edit", "image_upscale", "video_generate",
            "video_interpolate", "sfx", "music", "tts", "asr",
        }

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_everything_is_an_async_job(self, provider):
        """Even image generation: fal queues every request."""
        for cap in provider.media_capabilities():
            assert provider.media_execution(cap) is MediaExecution.ASYNC_JOB

    def test_chat_is_refused_with_a_pointer(self, provider):
        import asyncio

        from llmcore.models import Message, Role

        with pytest.raises(ProviderError, match=r"llm\.media"):
            asyncio.run(provider.chat_completion([Message(role=Role.USER, content="hi")]))

    async def test_model_details_describe_configured_endpoints(self, provider):
        details = await provider.get_models_details()
        ids = {d.id for d in details}
        assert FLUX in ids
        assert all(d.model_type == "media" for d in details)

    def test_endpoints_are_configurable(self):
        p = FalProvider(
            {**BASE_CONFIG, "backend": "httpx", "models": {"image_generate": "me/custom"}}
        )
        assert p._endpoint_for("image_generate", None) == "me/custom"
        assert p._endpoint_for("image_generate", "explicit/model") == "explicit/model"
        # untouched capabilities keep their defaults
        assert p._endpoint_for("video_interpolate", None) == "fal-ai/film"


# ---------------------------------------------------------------------------
# The queue lifecycle
# ---------------------------------------------------------------------------


class TestQueueLifecycle:
    @respx.mock
    async def test_submit_returns_a_queued_job(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("a tabby")
        assert job.status is MediaJobStatus.QUEUED
        assert job.provider_job_id == "req-1"
        assert job.queue_position == 3
        assert job.provider_metadata["endpoint"] == FLUX
        await provider.close()

    @respx.mock
    async def test_auth_header_uses_the_key_scheme(self, provider):
        route = respx.post(f"{QUEUE}/{FLUX}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_image_media("x")
        assert route.calls[0].request.headers["Authorization"] == "Key fal-test-key"
        await provider.close()

    @respx.mock
    async def test_none_valued_parameters_are_stripped(self, provider):
        """fal models reject unknown/None fields; only real values are sent."""
        route = respx.post(f"{QUEUE}/{FLUX}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_image_media("x", n=2)
        body = route.calls[0].request.content.decode()
        assert '"prompt"' in body and '"num_images"' in body
        assert "null" not in body
        await provider.close()

    @respx.mock
    async def test_poll_walks_queued_running_succeeded(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")

        status = respx.get(f"{QUEUE}/{FLUX}/requests/req-1/status")
        status.mock(
            return_value=httpx.Response(200, json={"status": "IN_PROGRESS", "queue_position": 0})
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.RUNNING

        status.mock(
            return_value=httpx.Response(
                200, json={"status": "COMPLETED", "metrics": {"inference_time": 1.25}}
            )
        )
        respx.get(f"{QUEUE}/{FLUX}/requests/req-1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "images": [
                        {
                            "url": "https://cdn.fal/i.png",
                            "content_type": "image/png",
                            "width": 1024,
                            "height": 768,
                        }
                    ]
                },
            )
        )
        job = await provider.poll_media_job(job)
        assert job.succeeded
        assert job.artifacts[0].uri == "https://cdn.fal/i.png"
        assert job.artifacts[0].width == 1024
        assert job.usage.compute_seconds == 1.25
        await provider.close()

    @respx.mock
    async def test_status_error_fails_the_job(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        respx.get(f"{QUEUE}/{FLUX}/requests/req-1/status").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED", "error": "NSFW blocked"})
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "NSFW" in (job.error or "")
        await provider.close()

    @respx.mock
    async def test_poll_on_terminal_job_is_a_noop(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        job.status = MediaJobStatus.SUCCEEDED
        route = respx.get(f"{QUEUE}/{FLUX}/requests/req-1/status")
        assert (await provider.poll_media_job(job)) is job
        assert route.call_count == 0
        await provider.close()

    @respx.mock
    async def test_lost_request_id_fails_loudly(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        job.provider_job_id = None
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "cannot be polled" in (job.error or "")
        await provider.close()


class TestCancellation:
    @respx.mock
    async def test_cancel_records_that_it_is_only_a_request(self, provider):
        """fal may still complete a request already being processed."""
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        respx.put(f"{QUEUE}/{FLUX}/requests/req-1/cancel").mock(
            return_value=httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
        )
        job = await provider.cancel_media_job(job)
        assert job.status is MediaJobStatus.CANCELED
        assert job.provider_metadata["cancellation"] == "requested"
        assert "may still complete" in job.provider_metadata["cancellation_note"]
        await provider.close()

    @respx.mock
    async def test_already_completed_is_not_an_error(self, provider):
        """A 400 ALREADY_COMPLETED means the work finished, not that we failed."""
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        respx.put(f"{QUEUE}/{FLUX}/requests/req-1/cancel").mock(
            return_value=httpx.Response(400, json={"status": "ALREADY_COMPLETED"})
        )
        respx.get(f"{QUEUE}/{FLUX}/requests/req-1/status").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"})
        )
        respx.get(f"{QUEUE}/{FLUX}/requests/req-1").mock(
            return_value=httpx.Response(200, json={"images": [{"url": "https://cdn.fal/a.png"}]})
        )
        job = await provider.cancel_media_job(job)
        assert job.succeeded
        assert job.provider_metadata["cancellation"] == "already_completed"
        await provider.close()

    @respx.mock
    async def test_cancel_on_terminal_job_is_a_noop(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        job = await provider.generate_image_media("x")
        job.status = MediaJobStatus.SUCCEEDED
        assert (await provider.cancel_media_job(job)).succeeded
        await provider.close()


# ---------------------------------------------------------------------------
# Marketplace-shaped outputs
# ---------------------------------------------------------------------------


class TestResultExtraction:
    @pytest.mark.parametrize(
        ("payload", "kind", "count"),
        [
            ({"images": [{"url": "u1"}, {"url": "u2"}]}, MediaKind.IMAGE, 2),
            ({"image": {"url": "u"}}, MediaKind.IMAGE, 1),
            ({"video": {"url": "v"}}, MediaKind.VIDEO, 1),
            ({"audio": {"url": "a"}}, MediaKind.AUDIO, 1),
            ({"audio_url": {"url": "a"}}, MediaKind.AUDIO, 1),
            ({"audio_file": {"url": "a"}}, MediaKind.AUDIO, 1),
        ],
    )
    def test_known_output_shapes(self, provider, payload, kind, count):
        """Output schemas vary per model; the extractor walks the known keys."""
        artifacts = provider._artifacts_from_result(payload)
        assert len(artifacts) == count
        assert all(a.kind is kind for a in artifacts)

    def test_bare_string_urls_are_accepted(self, provider):
        artifacts = provider._artifacts_from_result({"images": ["https://cdn.fal/x.png"]})
        assert artifacts[0].uri == "https://cdn.fal/x.png"

    def test_text_output_becomes_a_text_artifact(self, provider):
        artifacts = provider._artifacts_from_result({"text": "a transcript"})
        assert artifacts[0].kind is MediaKind.TEXT
        assert artifacts[0].text == "a transcript"

    def test_unknown_keys_are_ignored_not_fatal(self, provider):
        assert provider._artifacts_from_result({"something_new": {"url": "x"}}) == []

    def test_entries_without_a_url_are_skipped(self, provider):
        assert provider._artifacts_from_result({"images": [{"width": 10}]}) == []

    def test_extra_fields_are_preserved(self, provider):
        artifacts = provider._artifacts_from_result(
            {"images": [{"url": "u", "content_type": "image/png", "seed": 42}]}
        )
        assert artifacts[0].mime_type == "image/png"
        assert artifacts[0].provider_metadata["seed"] == 42


# ---------------------------------------------------------------------------
# URL inputs
# ---------------------------------------------------------------------------


class TestInputUploads:
    @respx.mock
    async def test_remote_refs_pass_straight_through(self, provider):
        """A URL must not round-trip through this process."""
        route = respx.post(f"{QUEUE}/fal-ai/flux-pro/kontext").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.edit_image_media(
            "brighter", image=MediaRef.from_url("https://example.invalid/in.png")
        )
        assert "https://example.invalid/in.png" in route.calls[0].request.content.decode()
        await provider.close()

    @respx.mock
    async def test_local_bytes_are_uploaded_to_the_cdn(self, provider):
        respx.post("https://rest.fal.ai/storage/auth/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "token": "tok",
                    "token_type": "Bearer",
                    "base_url": "https://v3.fal.media",
                },
            )
        )
        respx.post("https://v3.fal.media/files/upload").mock(
            return_value=httpx.Response(200, json={"access_url": "https://cdn.fal/uploaded.png"})
        )
        route = respx.post(f"{QUEUE}/fal-ai/flux-pro/kontext").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.edit_image_media(
            "x", image=MediaRef.from_bytes(b"PNG", mime_type="image/png")
        )
        assert "https://cdn.fal/uploaded.png" in route.calls[0].request.content.decode()
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
            (403, "authentication failed"),
            (404, "endpoint paths"),
            (429, "rate limit"),
            (500, "fal API error"),
        ],
    )
    async def test_status_mapping(self, provider, status, match):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(status, text="boom"))
        with pytest.raises(ProviderError, match=match):
            await provider.generate_image_media("x")
        await provider.close()

    @respx.mock
    async def test_transport_errors_are_retryable(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        assert exc.value.retryable is True
        await provider.close()


# ---------------------------------------------------------------------------
# The neutrality test
# ---------------------------------------------------------------------------


class TestProviderNeutrality:
    """fal's lifecycle differs from every prior adapter; the core must not care."""

    @respx.mock
    async def test_full_job_lifecycle_through_the_manager(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        media = MediaManager({"fal": provider}, job_policy=FAST)

        job = await media.images.generate("a tabby", provider="fal")
        assert media.jobs.get(job.id) is job

        calls = {"n": 0}

        def _status(_request):
            calls["n"] += 1
            state = "COMPLETED" if calls["n"] >= 2 else "IN_PROGRESS"
            return httpx.Response(200, json={"status": state})

        respx.get(f"{QUEUE}/{FLUX}/requests/req-1/status").mock(side_effect=_status)
        respx.get(f"{QUEUE}/{FLUX}/requests/req-1").mock(
            return_value=httpx.Response(200, json={"images": [{"url": "https://cdn.fal/i.png"}]})
        )

        result = await media.wait(job, timeout=5)
        assert calls["n"] == 2
        assert result.artifacts[0].uri == "https://cdn.fal/i.png"
        await provider.close()

    @respx.mock
    async def test_image_generation_is_a_job_here_and_a_result_elsewhere(self, provider):
        """The same router call returns different execution classes per provider.

        OpenAI answers images synchronously; fal queues them. Callers that use
        ``media.wait()`` are unaffected either way — which is the whole point of
        separating execution class from capability.
        """
        from llmcore.media.testing import FakeMediaProvider

        respx.post(f"{QUEUE}/{FLUX}").mock(return_value=httpx.Response(200, json=_submission()))
        media = MediaManager(
            {"fal": provider, "fake": FakeMediaProvider("fake")}, job_policy=FAST
        )
        from llmcore.media.models import MediaJob, MediaResult

        assert isinstance(await media.images.generate("x", provider="fal"), MediaJob)
        assert isinstance(await media.images.generate("x", provider="fake"), MediaResult)
        await provider.close()

    def test_routing_prefers_specialists_but_keeps_fal_for_breadth(self, provider):
        """fal is the default route for interpolation, a fallback elsewhere."""
        media = MediaManager({"fal": provider})
        assert media.who_can(MediaCapability.VIDEO_INTERPOLATE) == ["fal"]
        assert media.who_can(MediaCapability.SFX) == ["fal"]

    def test_no_change_was_needed_to_the_core_types(self):
        """Documentation-as-test: fal needed no new MediaJob fields.

        Everything fal reports maps onto the existing handle — the queue
        position onto ``queue_position``, the endpoint and raw payloads onto
        ``provider_metadata``, compute time onto ``MediaUsage.compute_seconds``.
        """
        from llmcore.media.models import MediaJob

        fields = set(MediaJob.__dataclass_fields__)
        for needed in ("queue_position", "provider_job_id", "poll_url", "provider_metadata"):
            assert needed in fields


# ---------------------------------------------------------------------------
# SDK backend
# ---------------------------------------------------------------------------


class TestSdkBackend:
    @pytest.fixture
    def sdk_provider(self) -> FalProvider:
        with patch("llmcore.providers.fal_provider.fal_client") as fal_mod, patch(
            "llmcore.providers.fal_provider.fal_sdk_available", True
        ):
            fal_mod.AsyncClient.return_value = MagicMock()
            return FalProvider({**BASE_CONFIG, "backend": "sdk"})

    def test_uses_the_sdk_client(self, sdk_provider):
        assert sdk_provider._backend == "sdk"
        assert sdk_provider._sdk is not None

    async def test_submit_goes_through_the_sdk(self, sdk_provider):
        sdk_provider._sdk.submit = AsyncMock(return_value=MagicMock(request_id="sdk-1"))
        job = await sdk_provider.generate_image_media("x")
        assert job.provider_job_id == "sdk-1"
        assert sdk_provider._sdk.submit.call_args[0][0] == FLUX

    async def test_status_and_result_go_through_the_sdk(self, sdk_provider):
        sdk_provider._sdk.submit = AsyncMock(return_value=MagicMock(request_id="sdk-1"))
        job = await sdk_provider.generate_image_media("x")

        class Completed:
            pass

        sdk_provider._sdk.status = AsyncMock(return_value=Completed())
        sdk_provider._sdk.result = AsyncMock(
            return_value={"images": [{"url": "https://cdn.fal/s.png"}]}
        )
        job = await sdk_provider.poll_media_job(job)
        assert job.succeeded
        assert job.artifacts[0].uri == "https://cdn.fal/s.png"

    async def test_upload_goes_through_the_sdk(self, sdk_provider):
        sdk_provider._sdk.upload = AsyncMock(return_value="https://cdn.fal/sdk-upload.png")
        sdk_provider._sdk.submit = AsyncMock(return_value=MagicMock(request_id="sdk-2"))
        await sdk_provider.edit_image_media("x", image=MediaRef.from_bytes(b"PNG"))
        assert sdk_provider._sdk.upload.await_count == 1


# ---------------------------------------------------------------------------
# Regressions found by live validation against the real fal API
# ---------------------------------------------------------------------------


class TestQueueAddressing:
    """fal namespaces the queue by *application*, not by model path.

    A request submitted to ``fal-ai/flux/schnell`` is tracked under
    ``fal-ai/flux``; polling the full path returns ``405 Method Not Allowed``.
    Live validation caught this on the very first real call.
    """

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("fal-ai/flux/schnell", "fal-ai/flux"),
            ("fal-ai/flux-pro/kontext", "fal-ai/flux-pro"),
            ("fal-ai/flux/dev/image-to-image", "fal-ai/flux"),
            ("fal-ai/film", "fal-ai/film"),
            ("fal-ai/whisper", "fal-ai/whisper"),
        ],
    )
    def test_queue_path_is_app_scoped(self, provider, endpoint, expected):
        assert provider._queue_path(endpoint) == expected

    @respx.mock
    async def test_urls_from_the_submission_win(self, provider):
        """fal owns the shape of its own routes; we use what it hands back."""
        respx.post(f"{QUEUE}/{FLUX}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "status_url": "https://queue.fal.run/custom/route/status",
                    "response_url": "https://queue.fal.run/custom/route",
                    "cancel_url": "https://queue.fal.run/custom/route/cancel",
                },
            )
        )
        job = await provider.generate_image_media("x")
        status = respx.get("https://queue.fal.run/custom/route/status").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"})
        )
        result = respx.get("https://queue.fal.run/custom/route").mock(
            return_value=httpx.Response(200, json={"images": [{"url": "u"}]})
        )
        job = await provider.poll_media_job(job)
        assert job.succeeded
        assert status.call_count == 1 and result.call_count == 1
        await provider.close()

    @respx.mock
    async def test_reconstructed_urls_drop_the_model_subpath(self, provider):
        """Fallback path, when a submission carries no URLs (e.g. the SDK)."""
        respx.post(f"{QUEUE}/{FLUX}").mock(
            return_value=httpx.Response(200, json={"request_id": "req-1"})
        )
        job = await provider.generate_image_media("x")
        assert job.poll_url is None

        # fal-ai/flux, NOT fal-ai/flux/schnell
        status = respx.get(f"{QUEUE}/fal-ai/flux/requests/req-1/status").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"})
        )
        result = respx.get(f"{QUEUE}/fal-ai/flux/requests/req-1").mock(
            return_value=httpx.Response(200, json={"images": [{"url": "u"}]})
        )
        job = await provider.poll_media_job(job)
        assert job.succeeded
        assert status.call_count == 1 and result.call_count == 1
        await provider.close()

    @respx.mock
    async def test_cancel_uses_the_app_scoped_path_too(self, provider):
        respx.post(f"{QUEUE}/{FLUX}").mock(
            return_value=httpx.Response(200, json={"request_id": "req-1"})
        )
        job = await provider.generate_image_media("x")
        cancel = respx.put(f"{QUEUE}/fal-ai/flux/requests/req-1/cancel").mock(
            return_value=httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
        )
        job = await provider.cancel_media_job(job)
        assert cancel.call_count == 1
        assert job.status is MediaJobStatus.CANCELED
        await provider.close()


class TestUploadBackends:
    """Storage lives on its own host, and accounts differ in which it offers.

    Live validation hit both halves: posting to ``fal.run`` made the router read
    ``storage/upload`` as an owner/app pair (404), and once on the right host,
    ``storage_type=gcs`` answered ``400 Invalid storage type``.
    """

    @respx.mock
    async def test_cdn_v3_is_tried_first(self, provider):
        token = respx.post("https://rest.fal.ai/storage/auth/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "token": "cdn-tok",
                    "token_type": "Bearer",
                    "base_url": "https://v3.fal.media",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
            )
        )
        upload = respx.post("https://v3.fal.media/files/upload").mock(
            return_value=httpx.Response(200, json={"access_url": "https://cdn.fal/v3.png"})
        )
        initiate = respx.post("https://rest.fal.ai/storage/upload/initiate")

        url = await provider._upload(MediaRef.from_bytes(b"PNG", mime_type="image/png"))
        assert url == "https://cdn.fal/v3.png"
        assert token.call_count == 1 and upload.call_count == 1
        assert initiate.call_count == 0  # the fallback stayed unused
        assert upload.calls[0].request.headers["Authorization"] == "Bearer cdn-tok"
        await provider.close()

    @respx.mock
    async def test_signed_url_flow_covers_accounts_without_cdn_v3(self, provider):
        respx.post("https://rest.fal.ai/storage/auth/token").mock(
            return_value=httpx.Response(400, json={"detail": "Invalid storage type"})
        )
        initiate = respx.post("https://rest.fal.ai/storage/upload/initiate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "upload_url": "https://signed.fal/put",
                    "file_url": "https://cdn.fal/legacy.png",
                },
            )
        )
        put = respx.put("https://signed.fal/put").mock(return_value=httpx.Response(200))

        url = await provider._upload(MediaRef.from_bytes(b"PNG", mime_type="image/png"))
        assert url == "https://cdn.fal/legacy.png"
        assert initiate.call_count == 1 and put.call_count == 1
        assert initiate.calls[0].request.url.params["storage_type"] == "gcs"
        await provider.close()

    @respx.mock
    async def test_both_failing_reports_both_causes(self, provider):
        respx.post("https://rest.fal.ai/storage/auth/token").mock(
            return_value=httpx.Response(400, text="no cdn")
        )
        respx.post("https://rest.fal.ai/storage/upload/initiate").mock(
            return_value=httpx.Response(400, text="no gcs")
        )
        with pytest.raises(ProviderError) as exc:
            await provider._upload(MediaRef.from_bytes(b"PNG"))
        message = str(exc.value)
        assert "_upload_via_cdn_v3" in message and "_upload_via_signed_url" in message
        assert exc.value.retryable is True
        await provider.close()

    @respx.mock
    async def test_uploads_never_touch_the_queue_host(self, provider):
        """The 404 that started this: storage is not a model endpoint."""
        respx.post("https://rest.fal.ai/storage/auth/token").mock(
            return_value=httpx.Response(
                200, json={"token": "t", "token_type": "Bearer", "base_url": "https://v3.fal.media"}
            )
        )
        respx.post("https://v3.fal.media/files/upload").mock(
            return_value=httpx.Response(200, json={"access_url": "https://cdn.fal/x.png"})
        )
        await provider._upload(MediaRef.from_bytes(b"PNG"))
        assert not any("fal.run/storage" in str(c.request.url) for c in respx.calls)
        await provider.close()


class TestInterpolationPayload:
    """FILM takes an explicit pair, not a frame list (live ``422``)."""

    @respx.mock
    async def test_frames_become_start_and_end_urls(self, provider):
        route = respx.post(f"{QUEUE}/fal-ai/film").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.interpolate_video_media(
            frames=[MediaRef.from_url("https://x/a.jpg"), MediaRef.from_url("https://x/b.jpg")]
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["start_image_url"] == "https://x/a.jpg"
        assert body["end_image_url"] == "https://x/b.jpg"
        assert "frames" not in body
        await provider.close()

    @respx.mock
    async def test_more_than_two_frames_uses_the_endpoints(self, provider, caplog):
        route = respx.post(f"{QUEUE}/fal-ai/film").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.interpolate_video_media(
            frames=[MediaRef.from_url(f"https://x/{c}.jpg") for c in "abc"]
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["start_image_url"].endswith("a.jpg")
        assert body["end_image_url"].endswith("c.jpg")
        await provider.close()

    @respx.mock
    async def test_explicit_urls_are_not_overridden(self, provider):
        route = respx.post(f"{QUEUE}/fal-ai/film").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.interpolate_video_media(
            frames=[MediaRef.from_url("https://x/a.jpg")],
            start_image_url="https://override/start.jpg",
        )
        import json

        assert json.loads(route.calls[0].request.content)["start_image_url"] == (
            "https://override/start.jpg"
        )
        await provider.close()
