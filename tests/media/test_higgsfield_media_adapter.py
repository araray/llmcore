# tests/media/test_higgsfield_media_adapter.py
"""Higgsfield as a media adapter.

Higgsfield is structurally close to fal — the model *is* the endpoint path,
every call is a queued request, and the API returns the URLs to track it — so
the adapter follows the fal pattern on purpose rather than inventing a second
one. The tests concentrate on the three places it genuinely differs:

* credentials are a **pair** (``Key id:secret``), not one opaque token;
* ``403`` means **out of credits**, not bad auth — verified live;
* ``nsfw`` is its own terminal state, which must not be retried like a failure;
* text-to-video and image-to-video are **separate endpoints**, so supplying an
  image has to change the path rather than the body.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.media import MediaCapability, MediaExecution, MediaKind, MediaManager, MediaRef
from llmcore.media.models import MediaJob, MediaJobStatus
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider, MediaJobPoller
from llmcore.providers.higgsfield_provider import HiggsfieldProvider

API = "https://api.higgsfield.ai"
SOUL = "higgsfield-ai/soul/standard"
T2V = "minimax/hailuo-2.3/standard/text-to-video"
I2V = "minimax/hailuo-2.3/standard/image-to-video"

CONFIG: dict[str, Any] = {
    "api_key": "keyid:keysecret",
    "_instance_name": "higgsfield",
    "backend": "httpx",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HIGGSFIELD_API_KEY", "HIGGSFIELD_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def provider() -> HiggsfieldProvider:
    return HiggsfieldProvider(dict(CONFIG))


def _submission(request_id: str = "req-1", status: str = "queued") -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "status_url": f"{API}/requests/{request_id}/status",
        "cancel_url": f"{API}/requests/{request_id}/cancel",
    }


# ---------------------------------------------------------------------------
# Credentials — a pair, not a token
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_key_from_the_env(self, monkeypatch):
        monkeypatch.setenv("HIGGSFIELD_API_KEY", "a:b")
        assert HiggsfieldProvider({"backend": "httpx"})._api_key == "a:b"

    def test_halves_can_be_configured_separately(self):
        p = HiggsfieldProvider(
            {**CONFIG, "api_key": None, "api_key_id": "the-id",
             "api_key_secret": "the-secret"}
        )
        assert p._api_key == "the-id:the-secret"

    def test_a_missing_key_explains_the_pair_format(self):
        with pytest.raises(ConfigError, match="key-id>:<key-secret"):
            HiggsfieldProvider({"backend": "httpx"})

    def test_a_single_token_warns_early(self, caplog):
        """The likeliest misconfiguration. A 401 much later is a poor way to
        learn the credential needed two halves."""
        import logging

        with caplog.at_level(logging.WARNING):
            HiggsfieldProvider({**CONFIG, "api_key": "just-one-token"})
        assert any("does not contain ':'" in r.message for r in caplog.records)

    @respx.mock
    async def test_the_auth_header_uses_the_key_scheme(self, provider):
        route = respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_image_media("a cat")
        assert route.calls[0].request.headers["authorization"] == "Key keyid:keysecret"
        await provider.close()


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_is_media_capable_and_a_poller(self, provider):
        assert isinstance(provider, MediaCapableProvider)
        assert isinstance(provider, MediaJobPoller)

    def test_capabilities(self, provider):
        assert {c.value for c in provider.media_capabilities()} == {
            "image_generate", "video_generate"
        }

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_everything_is_an_async_job(self, provider):
        for cap in provider.media_capabilities():
            assert provider.media_execution(cap) is MediaExecution.ASYNC_JOB

    def test_opts_into_webhooks(self, provider):
        assert provider.accepts_webhook_url is True

    def test_chat_is_refused_with_a_pointer(self, provider):
        import asyncio

        from llmcore.models import Message, Role

        with pytest.raises(ProviderError, match=r"llm\.media"):
            asyncio.run(provider.chat_completion([Message(role=Role.USER, content="hi")]))

    def test_backend_defaults_to_direct_rest(self, provider):
        assert provider._backend == "httpx"

    def test_no_transport_raises(self):
        with patch.multiple(
            "llmcore.providers.higgsfield_provider",
            httpx_available=False,
            higgsfield_sdk_available=False,
        ):
            with pytest.raises(ConfigError, match="requires 'httpx'"):
                HiggsfieldProvider(dict(CONFIG))

    async def test_model_details_describe_configured_paths(self, provider):
        details = await provider.get_models_details()
        assert {d.id for d in details} == {SOUL, T2V}
        assert all(d.model_type == "media" for d in details)
        await provider.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @respx.mock
    async def test_submit_returns_a_queued_job(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("a cat", n=2, aspect_ratio="1:1")
        assert job.status is MediaJobStatus.QUEUED
        assert job.provider_job_id == "req-1"
        assert job.poll_url == f"{API}/requests/req-1/status"
        await provider.close()

    @respx.mock
    async def test_none_valued_parameters_are_stripped(self, provider):
        route = respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_image_media("a cat")
        body = json.loads(route.calls[0].request.content)
        assert body == {"prompt": "a cat"}
        await provider.close()

    @respx.mock
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("queued", MediaJobStatus.QUEUED),
            ("in_progress", MediaJobStatus.RUNNING),
            ("completed", MediaJobStatus.SUCCEEDED),
            ("failed", MediaJobStatus.FAILED),
            ("canceled", MediaJobStatus.CANCELED),
        ],
    )
    async def test_status_mapping(self, provider, state, expected):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/requests/req-1/status").mock(
            return_value=httpx.Response(
                200, json={**_submission(status=state), "images": [{"url": "u"}]}
            )
        )
        assert (await provider.poll_media_job(job)).status is expected
        await provider.close()

    @respx.mock
    async def test_poll_uses_the_returned_status_url(self, provider):
        """The fal lesson: use the URLs the provider hands back."""
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "status": "queued",
                    "status_url": f"{API}/custom/track",
                    "cancel_url": f"{API}/custom/stop",
                },
            )
        )
        job = await provider.generate_image_media("x")
        route = respx.get(f"{API}/custom/track").mock(
            return_value=httpx.Response(
                200, json={"request_id": "req-1", "status": "completed",
                           "images": [{"url": "https://cdn/a.png"}]}
            )
        )
        job = await provider.poll_media_job(job)
        assert route.call_count == 1
        assert job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()

    @respx.mock
    async def test_cancel_uses_the_returned_cancel_url(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        route = respx.post(f"{API}/requests/req-1/cancel").mock(
            return_value=httpx.Response(200, json={"status": "canceled"})
        )
        job = await provider.cancel_media_job(job)
        assert route.call_count == 1
        assert job.status is MediaJobStatus.CANCELED
        await provider.close()

    @respx.mock
    async def test_an_immediately_completed_submission_is_applied(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(
                200, json={**_submission(status="completed"),
                           "images": [{"url": "https://cdn/a.png", "width": 1024}]}
            )
        )
        job = await provider.generate_image_media("x")
        assert job.succeeded
        assert job.artifacts[0].width == 1024
        await provider.close()

    @respx.mock
    async def test_a_lost_request_id_fails_loudly(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        job.provider_job_id = None
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "cannot be polled" in job.error
        await provider.close()


# ---------------------------------------------------------------------------
# nsfw — a refusal, not a malfunction
# ---------------------------------------------------------------------------


class TestContentRefusal:
    @respx.mock
    async def test_nsfw_is_terminal_and_explains_itself(self, provider):
        """Retrying a content refusal as though it were transient wastes money
        and never succeeds, so the error says so explicitly."""
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/requests/req-1/status").mock(
            return_value=httpx.Response(200, json=_submission(status="nsfw"))
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "content policy" in job.error
        assert "not a transient error" in job.error
        await provider.close()

    @respx.mock
    async def test_the_raw_state_is_preserved_for_callers(self, provider):
        """FAILED alone cannot distinguish a refusal from a crash."""
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/requests/req-1/status").mock(
            return_value=httpx.Response(200, json=_submission(status="nsfw"))
        )
        job = await provider.poll_media_job(job)
        assert job.provider_metadata["raw_status"] == "nsfw"
        await provider.close()

    @respx.mock
    async def test_a_reported_error_fails_the_job(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        job = await provider.generate_image_media("x")
        respx.get(f"{API}/requests/req-1/status").mock(
            return_value=httpx.Response(
                200, json={**_submission(status="failed"), "error": "upstream exploded"}
            )
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "upstream exploded" in job.error
        await provider.close()


# ---------------------------------------------------------------------------
# Video: separate endpoints for text and image conditioning
# ---------------------------------------------------------------------------


class TestVideoRouting:
    @respx.mock
    async def test_text_to_video_uses_the_default_path(self, provider):
        route = respx.post(f"{API}/{T2V}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_video_media("a drone shot", duration_seconds=6)
        body = json.loads(route.calls[0].request.content)
        assert body["prompt"] == "a drone shot" and body["duration"] == 6
        assert "image_url" not in body
        await provider.close()

    @respx.mock
    async def test_supplying_an_image_switches_to_the_image_to_video_path(self, provider):
        """Higgsfield splits these into separate endpoints; posting an image to
        the text-only path is a 422."""
        t2v = respx.post(f"{API}/{T2V}")
        i2v = respx.post(f"{API}/{I2V}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_video_media(
            "animate it", image=MediaRef.from_url("https://e/a.png")
        )
        assert i2v.call_count == 1 and t2v.call_count == 0
        body = json.loads(i2v.calls[0].request.content)
        assert body["image_url"] == "https://e/a.png"
        await provider.close()

    @respx.mock
    async def test_an_unmapped_model_warns_rather_than_silently_failing(
        self, provider, caplog
    ):
        import logging

        route = respx.post(f"{API}/custom/model").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        with caplog.at_level(logging.WARNING):
            await provider.generate_video_media(
                "x", model="custom/model", image=MediaRef.from_url("https://e/a.png")
            )
        assert route.call_count == 1
        assert any("image-to-video counterpart" in r.message for r in caplog.records)
        await provider.close()

    @respx.mock
    async def test_local_bytes_become_a_data_uri(self, provider):
        route = respx.post(f"{API}/{I2V}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_video_media(
            "x", image=MediaRef.from_bytes(b"PNG", mime_type="image/png")
        )
        body = json.loads(route.calls[0].request.content)
        assert body["image_url"].startswith("data:image/png;base64,")
        await provider.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class TestResultExtraction:
    @pytest.mark.parametrize(
        ("payload", "kind", "count"),
        [
            ({"images": [{"url": "a"}, {"url": "b"}]}, MediaKind.IMAGE, 2),
            ({"image": {"url": "a"}}, MediaKind.IMAGE, 1),
            ({"video": {"url": "v"}}, MediaKind.VIDEO, 1),
            ({"audio": {"url": "s"}}, MediaKind.AUDIO, 1),
            ({"audios": [{"url": "s"}]}, MediaKind.AUDIO, 1),
        ],
    )
    def test_documented_result_shapes(self, provider, payload, kind, count):
        artifacts = provider._artifacts_from_result(payload)
        assert len(artifacts) == count
        assert all(a.kind is kind for a in artifacts)

    def test_a_nested_results_object_is_handled(self, provider):
        artifacts = provider._artifacts_from_result({"results": {"images": [{"url": "a"}]}})
        assert len(artifacts) == 1

    def test_an_unknown_shape_is_not_fatal(self, provider):
        assert provider._artifacts_from_result({"mystery": {"url": "x"}}) == []

    def test_entries_without_a_url_are_skipped(self, provider):
        assert provider._artifacts_from_result({"images": [{"width": 10}]}) == []

    def test_dimensions_and_extras_are_preserved(self, provider):
        artifacts = provider._artifacts_from_result(
            {"video": {"url": "https://cdn/v.mp4", "width": 1920, "height": 1080,
                       "duration": 6.0, "seed": 42}}
        )
        art = artifacts[0]
        assert (art.width, art.height, art.duration_seconds) == (1920, 1080, 6.0)
        assert art.mime_type == "video/mp4"
        assert art.provider_metadata["seed"] == 42


# ---------------------------------------------------------------------------
# Errors — the live-verified taxonomy
# ---------------------------------------------------------------------------


class TestErrorMapping:
    @respx.mock
    async def test_403_not_enough_credits_is_not_an_auth_failure(self, provider):
        """Verified live: a valid credential on an empty account returns
        403 not_enough_credits."""
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(403, text='{"detail":"not_enough_credits"}')
        )
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        message = str(exc.value)
        assert "out of credits" in message
        assert "the credential is valid" in message
        assert "HIGGSFIELD_API_KEY" not in message
        assert exc.value.retryable is False
        await provider.close()

    @respx.mock
    async def test_401_is_an_auth_failure_and_names_the_pair_format(self, provider):
        """Verified live: bad credentials return 401 Invalid credentials."""
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(401, text='{"detail":"Invalid credentials"}')
        )
        with pytest.raises(ProviderError, match="key-id>:<key-secret"):
            await provider.generate_image_media("x")
        await provider.close()

    @respx.mock
    async def test_404_model_not_found_points_at_the_console(self, provider):
        """Verified live: an unknown path returns 404 model_not_found."""
        respx.post(f"{API}/nope/x").mock(
            return_value=httpx.Response(404, text='{"detail":"model_not_found"}')
        )
        with pytest.raises(ProviderError, match="console.higgsfield.ai"):
            await provider.generate_image_media("x", model="nope/x")
        await provider.close()

    @respx.mock
    @pytest.mark.parametrize(
        ("status", "match", "retryable"),
        [(422, "rejected the parameters", False), (429, "rate limit", True),
         (500, "API error", True)],
    )
    async def test_other_statuses(self, provider, status, match, retryable):
        respx.post(f"{API}/{SOUL}").mock(return_value=httpx.Response(status, text="boom"))
        with pytest.raises(ProviderError, match=match) as exc:
            await provider.generate_image_media("x")
        assert exc.value.retryable is retryable
        await provider.close()

    @respx.mock
    async def test_transport_errors_are_retryable(self, provider):
        respx.post(f"{API}/{SOUL}").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_image_media("x")
        assert exc.value.retryable is True
        await provider.close()


# ---------------------------------------------------------------------------
# Webhooks and routing
# ---------------------------------------------------------------------------


class TestWebhooks:
    @respx.mock
    async def test_a_callback_url_is_registered(self, provider):
        route = respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(200, json=_submission())
        )
        await provider.generate_image_media("x", webhook_url="https://h/cb/tok")
        body = json.loads(route.calls[0].request.content)
        assert body["webhook_url"] == "https://h/cb/tok"
        await provider.close()

    async def test_delivery_is_applied(self, provider):
        job = MediaJob(
            capability=MediaCapability.IMAGE_GENERATE, provider="higgsfield",
            model=SOUL, status=MediaJobStatus.RUNNING, provider_job_id="req-1",
        )
        job = await provider.apply_webhook_payload(
            job, {"request_id": "req-1", "status": "completed",
                  "images": [{"url": "https://cdn/a.png"}]}
        )
        assert job.succeeded and job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()

    async def test_a_mismatched_request_id_is_not_trusted(self, provider):
        polled: list[str] = []

        async def _poll(job):
            polled.append(job.id)
            return job

        provider.poll_media_job = _poll
        job = MediaJob(
            capability=MediaCapability.IMAGE_GENERATE, provider="higgsfield",
            model=SOUL, status=MediaJobStatus.RUNNING, provider_job_id="req-1",
        )
        await provider.apply_webhook_payload(
            job, {"request_id": "someone-else", "status": "completed"}
        )
        assert len(polled) == 1
        await provider.close()


class TestRouting:
    def test_registered_for_image_and_video(self, provider):
        media = MediaManager({"higgsfield": provider})
        assert "higgsfield" in media.who_can(MediaCapability.IMAGE_GENERATE)
        assert "higgsfield" in media.who_can(MediaCapability.VIDEO_GENERATE)

    @respx.mock
    async def test_router_dispatch_end_to_end(self, provider):
        respx.post(f"{API}/{SOUL}").mock(
            return_value=httpx.Response(
                200, json={**_submission(status="completed"),
                           "images": [{"url": "https://cdn/a.png"}]}
            )
        )
        media = MediaManager({"higgsfield": provider})
        job = await media.images.generate("x", provider="higgsfield")
        assert job.artifacts[0].uri == "https://cdn/a.png"
        await provider.close()

    def test_models_are_configurable(self):
        p = HiggsfieldProvider({**CONFIG, "models": {"image_generate": "me/custom"}})
        assert p._model_for("image_generate", None) == "me/custom"
        assert p._model_for("image_generate", "explicit/m") == "explicit/m"
        assert p._model_for("video_generate", None) == T2V
