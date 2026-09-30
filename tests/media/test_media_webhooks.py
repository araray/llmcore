# tests/media/test_media_webhooks.py
"""The generic webhook receiver (spec §2.8, phase M9).

Three properties are load-bearing and each has its own section below:

* **polling is always the fallback** — a deployment with no public ingress must
  keep working, so a callback can only ever save latency;
* **the URL is a credential** — it is handed to a third party and travels the
  public internet, so tokens are signed, single-use and un-transplantable;
* **a delivery is untrusted input** — it may report on a job llmcore already
  submitted, and nothing more.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llmcore.media import (
    JobPolicy,
    MediaCapability,
    MediaManager,
    WebhookRegistry,
    create_webhook_app,
)
from llmcore.media.jobs import MediaJobManager
from llmcore.media.models import MediaJob, MediaJobStatus
from llmcore.media.testing import FakeMediaProvider

FAST = JobPolicy(poll_initial_seconds=0.01, poll_max_seconds=0.01, job_timeout_seconds=5, jitter=0)


def _job(**kw: Any) -> MediaJob:
    return MediaJob(
        capability=kw.pop("capability", MediaCapability.VIDEO_GENERATE),
        provider=kw.pop("provider", "fake"),
        model=kw.pop("model", "m"),
        status=kw.pop("status", MediaJobStatus.QUEUED),
        **kw,
    )


# ---------------------------------------------------------------------------
# Poll-only is the default
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_no_base_url_means_disabled(self):
        """The common case — local development with no public ingress."""
        assert WebhookRegistry().enabled is False

    def test_issue_returns_none_rather_than_raising(self):
        """Absence of ingress is a deployment fact, not an error."""
        assert WebhookRegistry().issue(_job()) is None
        assert WebhookRegistry().reserve() is None

    def test_manager_still_waits_by_polling(self):
        media = MediaManager({"fake": FakeMediaProvider("fake")}, job_policy=FAST)
        assert media.jobs.webhooks.enabled is False

    async def test_wait_completes_with_webhooks_disabled(self):
        media = MediaManager({"fake": FakeMediaProvider("fake")}, job_policy=FAST)
        job = await media.video.generate("x", provider="fake")
        result = await media.wait(job, timeout=5)
        assert result.artifacts


# ---------------------------------------------------------------------------
# The URL is a credential
# ---------------------------------------------------------------------------


class TestTokenSecurity:
    @pytest.fixture
    def registry(self) -> WebhookRegistry:
        return WebhookRegistry("https://hooks.example.com", secret="s3cret")

    def test_url_is_built_from_the_base(self, registry):
        url = registry.issue(_job())
        assert url.startswith("https://hooks.example.com/media/jobs/")

    def test_token_is_single_use(self, registry):
        token, _ = registry.reserve()
        registry.bind(token, "job-1")
        assert registry.consume(token) == "job-1"
        assert registry.consume(token) is None, "a replayed delivery must be refused"

    def test_unknown_token_is_refused(self, registry):
        assert registry.verify("not.a.real.token") is None

    def test_tampered_signature_is_refused(self, registry):
        token, _ = registry.reserve()
        registry.bind(token, "job-1")
        nonce = token.split(".")[0]
        assert registry.verify(f"{nonce}.{'0' * 32}") is None

    def test_tampered_nonce_is_refused(self, registry):
        token, _ = registry.reserve()
        registry.bind(token, "job-1")
        tag = token.split(".")[1]
        assert registry.verify(f"tampered.{tag}") is None

    def test_token_from_another_registry_is_refused(self, registry):
        other = WebhookRegistry("https://hooks.example.com", secret="different")
        token, _ = other.reserve()
        other.bind(token, "job-1")
        assert registry.verify(token) is None

    def test_unbound_token_is_refused(self, registry):
        """The window between submit and job creation carries no authority."""
        token, _ = registry.reserve()
        assert registry.verify(token) is None

    def test_binding_cannot_be_moved(self, registry):
        token, _ = registry.reserve()
        registry.bind(token, "job-1")
        registry.bind(token, "job-2")  # ignored: already bound
        assert registry.verify(token) == "job-1"

    def test_released_token_is_gone(self, registry):
        token, _ = registry.reserve()
        registry.release(token)
        registry.bind(token, "job-1")
        assert registry.verify(token) is None

    def test_revoke_clears_tokens_for_a_job(self, registry):
        token, _ = registry.reserve()
        registry.bind(token, "job-1")
        registry.revoke("job-1")
        assert registry.verify(token) is None
        assert registry.pending() == 0

    def test_each_reservation_is_distinct(self, registry):
        tokens = {registry.reserve()[0] for _ in range(50)}
        assert len(tokens) == 50


# ---------------------------------------------------------------------------
# A delivery is untrusted input
# ---------------------------------------------------------------------------


class _WebhookProvider(FakeMediaProvider):
    """A fake adapter that can parse its own callbacks."""

    accepts_webhook_url = True

    def __init__(self, name: str = "fake") -> None:
        super().__init__(name)
        self.webhook_urls: list[str | None] = []
        self.applied: list[dict[str, Any]] = []

    async def apply_webhook_payload(self, job: MediaJob, payload: dict[str, Any]) -> MediaJob:
        self.applied.append(payload)
        job.status = MediaJobStatus.SUCCEEDED
        job.provider_metadata["via"] = "webhook"
        return job

    async def generate_video_media(self, prompt: str, **kwargs: Any) -> MediaJob:
        self.webhook_urls.append(kwargs.pop("webhook_url", None))
        return await super().generate_video_media(prompt, **kwargs)


class TestDeliveryHandling:
    @pytest.fixture
    def manager(self) -> MediaJobManager:
        provider = _WebhookProvider()
        return MediaJobManager(
            {"fake": provider}.get,
            policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )

    async def test_valid_delivery_is_applied(self, manager):
        job = manager.track(_job())
        token, _ = manager.webhooks.reserve()
        manager.webhooks.bind(token, job.id)

        delivery = await manager.handle_webhook(token, {"status": "OK"})
        assert delivery.accepted is True
        assert manager.get(job.id).status is MediaJobStatus.SUCCEEDED
        assert manager.get(job.id).provider_metadata["via"] == "webhook"

    async def test_unknown_token_cannot_create_a_job(self, manager):
        """The security property: a callback reports, it never creates."""
        before = len(manager.list())
        delivery = await manager.handle_webhook("bogus.token", {"status": "OK"})
        assert delivery.accepted is False
        assert delivery.status_code == 404
        assert len(manager.list()) == before

    async def test_replay_is_refused(self, manager):
        job = manager.track(_job())
        token, _ = manager.webhooks.reserve()
        manager.webhooks.bind(token, job.id)
        assert (await manager.handle_webhook(token, {"status": "OK"})).accepted
        second = await manager.handle_webhook(token, {"status": "OK"})
        assert second.accepted is False

    async def test_late_delivery_on_a_finished_job_is_accepted_quietly(self, manager):
        """The poll loop beat the callback. Report success so the vendor stops
        retrying a delivery we no longer need."""
        job = manager.track(_job(status=MediaJobStatus.SUCCEEDED))
        token, _ = manager.webhooks.reserve()
        manager.webhooks.bind(token, job.id)
        delivery = await manager.handle_webhook(token, {"status": "OK"})
        assert delivery.accepted is True

    async def test_a_failing_parser_does_not_take_down_the_receiver(self, manager):
        job = manager.track(_job())
        token, _ = manager.webhooks.reserve()
        manager.webhooks.bind(token, job.id)

        async def _boom(*_a, **_k):
            raise RuntimeError("bad payload")

        manager._resolver("fake").apply_webhook_payload = _boom
        delivery = await manager.handle_webhook(token, {"status": "OK"})
        assert delivery.accepted is False
        assert delivery.status_code == 500

    async def test_provider_without_a_parser_falls_back_to_polling(self):
        """A callback that cannot be read still says *when* to look."""
        provider = FakeMediaProvider("fake")  # no apply_webhook_payload
        manager = MediaJobManager(
            {"fake": provider}.get,
            policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        job = manager.track(_job())
        token, _ = manager.webhooks.reserve()
        manager.webhooks.bind(token, job.id)
        delivery = await manager.handle_webhook(token, {"anything": True})
        assert delivery.accepted is True


# ---------------------------------------------------------------------------
# Webhook / poll equivalence
# ---------------------------------------------------------------------------


class TestEquivalence:
    async def test_a_callback_ends_the_wait_early(self):
        """The latency win, and the reason both paths share one code path."""
        provider = _WebhookProvider()
        media = MediaManager(
            {"fake": provider},
            job_policy=JobPolicy(
                poll_initial_seconds=30, poll_max_seconds=30, job_timeout_seconds=10, jitter=0
            ),
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        job = await media.video.generate("x", provider="fake")
        # The fake finishes on the first poll, so force it to stay pending: the
        # only thing that can end a 30s sleep early is the callback.
        job.status = MediaJobStatus.RUNNING
        token = next(iter(media.jobs.webhooks._tickets))

        async def _deliver():
            await asyncio.sleep(0.05)
            await media.jobs.handle_webhook(token, {"status": "OK"})

        started = asyncio.get_running_loop().time()
        _, result = await asyncio.gather(_deliver(), media.wait(job, timeout=10))
        elapsed = asyncio.get_running_loop().time() - started

        assert result.artifacts or result.capability is MediaCapability.VIDEO_GENERATE
        assert media.jobs.get(job.id).status is MediaJobStatus.SUCCEEDED
        assert elapsed < 5, "the callback should have cut the 30s backoff short"

    async def test_the_router_offers_a_url_to_opted_in_adapters(self):
        provider = _WebhookProvider()
        media = MediaManager(
            {"fake": provider},
            job_policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        await media.video.generate("x", provider="fake")
        assert provider.webhook_urls[0].startswith("https://h.example.com/media/jobs/")

    async def test_no_url_is_offered_when_disabled(self):
        provider = _WebhookProvider()
        media = MediaManager({"fake": provider}, job_policy=FAST)
        await media.video.generate("x", provider="fake")
        assert provider.webhook_urls == [None]

    async def test_adapters_that_did_not_opt_in_are_never_offered_one(self):
        """A callback URL forwarded as a vendor parameter would be a leak."""
        provider = FakeMediaProvider("fake")
        media = MediaManager(
            {"fake": provider},
            job_policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        await media.video.generate("x", provider="fake")
        recorded = [c for c in provider.calls if c[0] == "generate_video_media"]
        assert "webhook_url" not in recorded[0][1]

    async def test_a_synchronous_result_releases_its_token(self):
        """Nothing will ever call back about an image that already arrived."""
        provider = _WebhookProvider()
        media = MediaManager(
            {"fake": provider},
            job_policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        await media.images.generate("x", provider="fake")
        assert media.jobs.webhooks.pending() == 0

    async def test_a_failed_submission_releases_its_token(self):
        provider = _WebhookProvider()

        async def _boom(*_a, **_k):
            raise RuntimeError("submit failed")

        provider.generate_video_media = _boom
        media = MediaManager(
            {"fake": provider},
            job_policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        with pytest.raises(RuntimeError):
            await media.video.generate("x", provider="fake")
        assert media.jobs.webhooks.pending() == 0

    async def test_terminal_jobs_drop_their_tokens(self):
        provider = _WebhookProvider()
        media = MediaManager(
            {"fake": provider},
            job_policy=FAST,
            webhooks=WebhookRegistry("https://h.example.com", secret="s"),
        )
        job = await media.video.generate("x", provider="fake")
        await media.wait(job, timeout=5)
        assert media.jobs.webhooks.pending() == 0


# ---------------------------------------------------------------------------
# The ASGI app
# ---------------------------------------------------------------------------


async def _call(app: Any, path: str, body: bytes = b"{}", method: str = "POST"):
    sent: list[dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "method": method, "path": path}, receive, send)
    return sent[0]["status"], sent[1]["body"]


class TestAsgiApp:
    @pytest.fixture
    def wired(self):
        provider = _WebhookProvider()
        registry = WebhookRegistry("https://h.example.com", secret="s")
        manager = MediaJobManager({"fake": provider}.get, policy=FAST, webhooks=registry)
        return manager, registry, create_webhook_app(manager, registry)

    async def test_valid_post_applies_the_delivery(self, wired):
        manager, registry, app = wired
        job = manager.track(_job())
        token, _url = registry.reserve()
        registry.bind(token, job.id)

        status, body = await _call(app, f"/media/jobs/{token}", b'{"status":"OK"}')
        assert status == 200
        assert json.loads(body) == {"ok": True}
        assert manager.get(job.id).status is MediaJobStatus.SUCCEEDED

    async def test_unknown_token_is_404_not_401(self, wired):
        """Confirming a token exists but is spent would leak its validity."""
        _, _, app = wired
        status, _ = await _call(app, "/media/jobs/bogus.token")
        assert status == 404

    async def test_get_is_rejected(self, wired):
        _, _, app = wired
        status, _ = await _call(app, "/media/jobs/x", method="GET")
        assert status == 405

    async def test_malformed_json_is_rejected(self, wired):
        manager, registry, app = wired
        job = manager.track(_job())
        token, _ = registry.reserve()
        registry.bind(token, job.id)
        status, _ = await _call(app, f"/media/jobs/{token}", b"not json")
        assert status == 400

    async def test_chunked_bodies_are_reassembled(self, wired):
        manager, registry, app = wired
        job = manager.track(_job())
        token, _ = registry.reserve()
        registry.bind(token, job.id)

        chunks = [b'{"sta', b'tus":', b'"OK"}']
        sent: list[dict[str, Any]] = []
        it = iter(chunks)

        async def receive():
            piece = next(it)
            return {"type": "http.request", "body": piece, "more_body": piece != chunks[-1]}

        async def send(message):
            sent.append(message)

        await app({"type": "http", "method": "POST", "path": f"/media/jobs/{token}"}, receive, send)
        assert sent[0]["status"] == 200


# ---------------------------------------------------------------------------
# fal's own callback shape
# ---------------------------------------------------------------------------


class TestFalWebhookPayload:
    @pytest.fixture
    def fal(self):
        from llmcore.providers.fal_provider import FalProvider

        return FalProvider({"api_key": "k", "_instance_name": "fal", "backend": "httpx"})

    def _fal_job(self, request_id: str = "req-1") -> MediaJob:
        return MediaJob(
            capability=MediaCapability.IMAGE_GENERATE,
            provider="fal",
            model="fal-ai/flux/schnell",
            status=MediaJobStatus.QUEUED,
            provider_job_id=request_id,
            provider_metadata={"endpoint": "fal-ai/flux/schnell"},
        )

    def test_fal_opts_in(self, fal):
        assert fal.accepts_webhook_url is True

    async def test_success_payload_produces_artifacts(self, fal):
        job = await fal.apply_webhook_payload(
            self._fal_job(),
            {
                "request_id": "req-1",
                "status": "OK",
                "payload": {"images": [{"url": "https://cdn.fal/i.png"}]},
            },
        )
        assert job.succeeded
        assert job.artifacts[0].uri == "https://cdn.fal/i.png"
        await fal.close()

    async def test_error_payload_fails_the_job(self, fal):
        job = await fal.apply_webhook_payload(
            self._fal_job(), {"request_id": "req-1", "status": "ERROR", "error": "NSFW"}
        )
        assert job.status is MediaJobStatus.FAILED
        assert "NSFW" in job.error
        await fal.close()

    async def test_mismatched_request_id_is_not_trusted(self, fal):
        """Anyone who learns a callback URL can POST to it; a wrong artifact is
        worse than a slow one, so a mismatch falls back to the queue."""
        polled: list[str] = []

        async def _poll(job):
            polled.append(job.id)
            return job

        fal.poll_media_job = _poll
        await fal.apply_webhook_payload(
            self._fal_job(),
            {"request_id": "someone-elses", "status": "OK", "payload": {"images": []}},
        )
        assert len(polled) == 1
        await fal.close()

    async def test_unrecognised_status_falls_back_to_polling(self, fal):
        polled: list[str] = []

        async def _poll(job):
            polled.append(job.id)
            return job

        fal.poll_media_job = _poll
        await fal.apply_webhook_payload(self._fal_job(), {"request_id": "req-1", "status": "???"})
        assert len(polled) == 1
        await fal.close()
