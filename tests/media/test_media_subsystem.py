# tests/media/test_media_subsystem.py
"""Tests for the media manager, routers, job manager and artifact store.

Everything runs against the in-package FakeMediaProvider, so the whole
subsystem is exercised with no network and no vendor account.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from llmcore.exceptions import (
    MediaCapabilityError,
    MediaError,
    MediaJobError,
    MediaJobTimeoutError,
)
from llmcore.media import (
    ArtifactStore,
    JobPolicy,
    MaterializePolicy,
    MediaArtifact,
    MediaCapability,
    MediaExecution,
    MediaJobStatus,
    MediaKind,
    MediaManager,
    MediaRef,
)
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider
from llmcore.media.testing import FakeMediaProvider

# A fast policy so timeout/backoff tests do not actually sleep for seconds.
FAST = JobPolicy(poll_initial_seconds=0.0, poll_max_seconds=0.0, job_timeout_seconds=5.0, jitter=0.0)


@pytest.fixture
def fake() -> FakeMediaProvider:
    return FakeMediaProvider("fake")


@pytest.fixture
def manager(fake: FakeMediaProvider) -> MediaManager:
    return MediaManager({"fake": fake}, job_policy=FAST)


# ---------------------------------------------------------------------------
# Protocol coverage
# ---------------------------------------------------------------------------


class TestProtocolCoverage:
    def test_every_capability_maps_to_a_protocol(self):
        missing = [c for c in MediaCapability if c not in CAPABILITY_PROTOCOLS]
        assert missing == []

    def test_fake_satisfies_every_mapped_protocol(self, fake):
        unmet = [
            cap.value
            for cap, proto in CAPABILITY_PROTOCOLS.items()
            if not isinstance(fake, proto)
        ]
        assert unmet == []

    def test_fake_is_media_capable(self, fake):
        assert isinstance(fake, MediaCapableProvider)

    def test_execution_classes_are_declared(self, fake):
        assert fake.media_execution(MediaCapability.VIDEO_GENERATE) is MediaExecution.ASYNC_JOB
        assert fake.media_execution(MediaCapability.TTS_STREAM) is MediaExecution.STREAM
        assert (
            fake.media_execution(MediaCapability.IMAGE_GENERATE)
            is MediaExecution.REQUEST_RESPONSE
        )


# ---------------------------------------------------------------------------
# Discovery and resolution
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_capabilities_lists_providers(self, manager):
        caps = manager.capabilities()
        assert caps[MediaCapability.IMAGE_GENERATE] == ["fake"]
        assert len(caps) == len(list(MediaCapability))

    def test_who_can(self, manager):
        assert manager.who_can(MediaCapability.VIDEO_GENERATE) == ["fake"]
        assert manager.who_can("image_generate") == ["fake"]

    def test_no_adapters_is_not_an_error(self):
        m = MediaManager()
        assert m.has_adapters() is False
        assert m.capabilities() == {}
        assert m.who_can(MediaCapability.TTS) == []

    def test_register_and_unregister(self, manager):
        manager.register_adapter("second", FakeMediaProvider("second"))
        assert manager.adapter_names == ["fake", "second"]
        manager.unregister_adapter("second")
        assert manager.adapter_names == ["fake"]

    def test_declared_but_unimplemented_capability_is_dropped(self):
        # Declares TTS without implementing the TTS protocol.
        class Bare:
            def get_name(self) -> str:
                return "bare"

            def media_capabilities(self):
                return frozenset({MediaCapability.TTS})

            def media_execution(self, capability, model=None):
                return MediaExecution.REQUEST_RESPONSE

        m = MediaManager({"bare": Bare()})
        assert m.who_can(MediaCapability.TTS) == []

    def test_broken_adapter_does_not_break_discovery(self):
        class Broken:
            def get_name(self) -> str:
                return "broken"

            def media_capabilities(self):
                raise RuntimeError("boom")

            def media_execution(self, capability, model=None):
                return MediaExecution.REQUEST_RESPONSE

        m = MediaManager({"broken": Broken(), "fake": FakeMediaProvider("fake")})
        assert m.who_can(MediaCapability.TTS) == ["fake"]


class TestResolution:
    def test_explicit_provider(self, manager, fake):
        assert manager.resolve(MediaCapability.TTS, provider="fake") is fake

    def test_explicit_unknown_provider_raises(self, manager):
        with pytest.raises(MediaCapabilityError, match="not configured"):
            manager.resolve(MediaCapability.TTS, provider="nope")

    def test_explicit_provider_lacking_capability_raises(self):
        limited = FakeMediaProvider("limited", capabilities=[MediaCapability.TTS])
        m = MediaManager({"limited": limited})
        with pytest.raises(MediaCapabilityError, match="does not support"):
            m.resolve(MediaCapability.VIDEO_GENERATE, provider="limited")

    def test_no_candidate_raises_with_hints(self):
        m = MediaManager()
        with pytest.raises(MediaCapabilityError) as exc:
            m.resolve(MediaCapability.VIDEO_GENERATE)
        # the built-in preference list becomes the actionable hint
        assert "fal" in str(exc.value) or "gemini" in str(exc.value)

    def test_routing_preference_decides_between_providers(self):
        a = FakeMediaProvider("alpha")
        b = FakeMediaProvider("beta")
        m = MediaManager(
            {"alpha": a, "beta": b},
            routing={MediaCapability.IMAGE_GENERATE: ("beta", "alpha")},
        )
        assert m.who_can(MediaCapability.IMAGE_GENERATE) == ["beta", "alpha"]
        assert m.resolve(MediaCapability.IMAGE_GENERATE) is b

    def test_unconfigured_names_in_routing_are_skipped(self):
        m = MediaManager(
            {"fake": FakeMediaProvider("fake")},
            routing={MediaCapability.TTS: ("elevenlabs", "fake")},
        )
        assert m.who_can(MediaCapability.TTS) == ["fake"]


class TestRoutingConfig:
    def test_routing_read_from_config(self):
        store = {"media.routing": {"image_generate": ["beta", "alpha"]}}
        get = lambda k, d=None: store.get(k, d)
        routing = MediaManager._routing_from_config(get)
        assert routing[MediaCapability.IMAGE_GENERATE] == ("beta", "alpha")

    def test_unknown_capability_in_config_is_ignored(self):
        get = lambda k, d=None: {"media.routing": {"teleport": ["x"]}}.get(k, d)
        assert MediaManager._routing_from_config(get) == {}

    def test_scalar_value_is_accepted(self):
        get = lambda k, d=None: {"media.routing": {"tts": "fake"}}.get(k, d)
        assert MediaManager._routing_from_config(get)[MediaCapability.TTS] == ("fake",)

    def test_non_table_routing_is_ignored(self):
        get = lambda k, d=None: {"media.routing": ["nope"]}.get(k, d)
        assert MediaManager._routing_from_config(get) == {}


class TestFromProviderManager:
    def _pm(self, providers: dict):
        pm = MagicMock()
        pm.get_available_providers.return_value = list(providers)
        pm.get_provider.side_effect = lambda n: providers[n]
        return pm

    def test_discovers_only_media_capable_providers(self):
        chat_only = MagicMock(spec=["get_name"])
        m = MediaManager.from_provider_manager(
            self._pm({"chat": chat_only, "fake": FakeMediaProvider("fake")})
        )
        assert m.adapter_names == ["fake"]

    def test_broken_provider_is_skipped(self):
        pm = MagicMock()
        pm.get_available_providers.return_value = ["bad", "fake"]
        good = FakeMediaProvider("fake")
        pm.get_provider.side_effect = lambda n: (_ for _ in ()).throw(RuntimeError()) if n == "bad" else good
        m = MediaManager.from_provider_manager(pm)
        assert m.adapter_names == ["fake"]

    def test_config_is_optional(self):
        m = MediaManager.from_provider_manager(self._pm({}))
        assert m.has_adapters() is False
        assert m.artifacts.policy is MaterializePolicy.ON_EXPIRY

    def test_config_drives_artifact_policy(self, tmp_path):
        store = {"media.artifact_materialize": "always", "media.artifact_path": str(tmp_path)}
        m = MediaManager.from_provider_manager(self._pm({}), lambda k, d=None: store.get(k, d))
        assert m.artifacts.policy is MaterializePolicy.ALWAYS
        assert m.artifacts.base_path == tmp_path

    def test_unknown_policy_falls_back(self):
        store = {"media.artifact_materialize": "sometimes"}
        m = MediaManager.from_provider_manager(self._pm({}), lambda k, d=None: store.get(k, d))
        assert m.artifacts.policy is MaterializePolicy.ON_EXPIRY


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


class TestImageRouter:
    async def test_generate_forwards_arguments(self, manager, fake):
        r = await manager.images.generate("a tabby", n=3, size="512x512", seed=7)
        assert r.capability is MediaCapability.IMAGE_GENERATE
        assert len(r.artifacts) == 3
        method, kwargs = fake.calls[-1]
        assert method == "generate_image_media"
        assert kwargs["prompt"] == "a tabby" and kwargs["n"] == 3
        assert kwargs["size"] == "512x512" and kwargs["seed"] == 7

    async def test_vendor_kwargs_pass_through(self, manager, fake):
        await manager.images.generate("x", guidance_scale=8.5)
        assert fake.calls[-1][1]["guidance_scale"] == 8.5

    async def test_edit_and_upscale(self, manager):
        ref = MediaRef.from_bytes(b"img", mime_type="image/png")
        assert (await manager.images.edit("night", image=ref)).capability is (
            MediaCapability.IMAGE_EDIT
        )
        assert (await manager.images.upscale(image=ref, scale=2)).capability is (
            MediaCapability.IMAGE_UPSCALE
        )

    async def test_ocr_returns_text(self, manager):
        r = await manager.images.ocr(document=MediaRef.from_bytes(b"pdf"))
        assert r.text == "fake ocr text"


class TestAudioRouter:
    async def test_speak(self, manager):
        r = await manager.audio.speak("hello", voice="rachel")
        assert r.artifacts[0].kind is MediaKind.AUDIO

    async def test_stream_tts_yields_chunks(self, manager):
        chunks = [c async for c in manager.audio.stream_tts("one two three")]
        assert chunks == [b"one", b"two", b"three"]

    async def test_transcribe(self, manager):
        r = await manager.audio.transcribe(audio=MediaRef.from_bytes(b"wav"))
        assert r.text == "fake transcript"

    async def test_open_session(self, manager):
        s = await manager.audio.open_transcription_session(model="nova")
        assert s == {"session": "fake", "model": "nova"}

    async def test_music_and_sfx(self, manager):
        assert (await manager.audio.music("lofi")).capability is MediaCapability.MUSIC
        assert (await manager.audio.sfx("thunder")).capability is MediaCapability.SFX


class TestVideoRouter:
    async def test_generate_returns_tracked_job(self, manager):
        job = await manager.video.generate("dunes", duration_seconds=8, with_audio=True)
        assert job.capability is MediaCapability.VIDEO_GENERATE
        assert manager.jobs.get(job.id) is job

    async def test_edit_and_interpolate_return_jobs(self, manager):
        ref = MediaRef.from_url("https://x/in.mp4")
        assert (await manager.video.edit("brighter", video=ref)).capability is (
            MediaCapability.VIDEO_EDIT
        )
        assert (await manager.video.interpolate(video=ref, target_fps=60)).capability is (
            MediaCapability.VIDEO_INTERPOLATE
        )


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


class TestJobPolicy:
    def test_backoff_grows_and_is_capped(self):
        p = JobPolicy(poll_initial_seconds=1, poll_max_seconds=8, jitter=0.0)
        assert [p.delay_for(i) for i in range(1, 6)] == [1, 2, 4, 8, 8]

    def test_jitter_stays_in_band(self):
        p = JobPolicy(poll_initial_seconds=10, poll_max_seconds=10, jitter=0.5)
        assert all(5.0 <= p.delay_for(1) <= 15.0 for _ in range(50))

    def test_max_is_never_below_initial(self):
        p = JobPolicy(poll_initial_seconds=30, poll_max_seconds=1)
        assert p.poll_max_seconds == 30

    def test_backoff_never_overflows(self):
        """A long-running job can be polled thousands of times.

        Without an exponent cap, ``2 ** attempt`` stops converting to float and
        the whole wait loop dies with OverflowError — which a multi-hour video
        job would actually reach.
        """
        p = JobPolicy(poll_initial_seconds=0.0, poll_max_seconds=0.0, jitter=0.0)
        assert p.delay_for(10_000) == 0.0
        p2 = JobPolicy(poll_initial_seconds=2, poll_max_seconds=30, jitter=0.0)
        assert p2.delay_for(5_000) == 30

    def test_zero_initial_delay_is_allowed(self):
        p = JobPolicy(poll_initial_seconds=0, poll_max_seconds=0, jitter=0.0)
        assert p.delay_for(1) == 0.0

    def test_from_config(self):
        store = {
            "media.jobs.poll_initial_seconds": 5,
            "media.jobs.poll_max_seconds": 50,
            "media.jobs.job_timeout_seconds": 60,
        }
        p = JobPolicy.from_config(lambda k, d=None: store.get(k, d))
        assert (p.poll_initial_seconds, p.poll_max_seconds, p.job_timeout_seconds) == (5, 50, 60)


class TestJobLifecycle:
    async def test_wait_succeeds(self, manager):
        job = await manager.video.generate("dunes")
        finished = await manager.jobs.wait(job)
        assert finished.succeeded and finished.artifacts

    async def test_manager_wait_returns_result(self, manager):
        job = await manager.video.generate("dunes")
        result = await manager.wait(job)
        assert result.capability is MediaCapability.VIDEO_GENERATE
        assert result.usage.seconds == 4.0

    async def test_progress_is_reported_across_polls(self):
        fake = FakeMediaProvider("fake", poll_count=4)
        m = MediaManager({"fake": fake}, job_policy=FAST)
        job = await m.video.generate("x")
        job = await m.jobs.poll(job)
        assert job.status is MediaJobStatus.RUNNING
        assert 0 < job.progress < 1

    async def test_poll_on_terminal_job_is_a_noop(self, manager, fake):
        job = await manager.video.generate("x")
        await manager.jobs.wait(job)
        before = len(fake.calls)
        assert (await manager.jobs.poll(job)) is job
        assert len(fake.calls) == before

    async def test_failure_raises(self):
        m = MediaManager({"fake": FakeMediaProvider("fake", fail_jobs=True)}, job_policy=FAST)
        job = await m.video.generate("x")
        with pytest.raises(MediaJobError, match="fake failure"):
            await m.jobs.wait(job)

    async def test_failure_can_be_returned_instead_of_raised(self):
        m = MediaManager({"fake": FakeMediaProvider("fake", fail_jobs=True)}, job_policy=FAST)
        job = await m.video.generate("x")
        finished = await m.jobs.wait(job, raise_on_failure=False)
        assert finished.status is MediaJobStatus.FAILED

    async def test_timeout_raises_but_keeps_the_job_alive(self):
        # never completes within the budget
        m = MediaManager({"fake": FakeMediaProvider("fake", poll_count=10_000)}, job_policy=FAST)
        job = await m.video.generate("x")
        with pytest.raises(MediaJobTimeoutError, match="still live"):
            await m.jobs.wait(job, timeout=0.05)
        # the handle survives: an expensive generation is not discarded
        assert m.jobs.get(job.id) is not None
        assert not job.is_terminal

    async def test_immediate_success_without_polling(self):
        m = MediaManager({"fake": FakeMediaProvider("fake", poll_count=0)}, job_policy=FAST)
        job = await m.video.generate("x")
        assert job.succeeded
        assert (await m.jobs.wait(job)).succeeded

    async def test_cancel(self, manager):
        job = await manager.video.generate("x")
        assert (await manager.jobs.cancel(job)).status is MediaJobStatus.CANCELED

    async def test_cancel_terminal_is_noop(self, manager):
        job = await manager.video.generate("x")
        await manager.jobs.wait(job)
        assert (await manager.jobs.cancel(job)).succeeded

    async def test_cancel_all_active(self, manager):
        await manager.video.generate("a")
        await manager.video.generate("b")
        cancelled = await manager.jobs.cancel_all()
        assert len(cancelled) == 2

    async def test_cancel_all_swallows_failures(self, manager, fake):
        job = await manager.video.generate("a")

        async def boom(_job):
            raise RuntimeError("vendor down")

        fake.cancel_media_job = boom
        assert await manager.jobs.cancel_all() == []
        assert not job.is_terminal

    async def test_registry_listing_and_forget(self, manager):
        a = await manager.video.generate("a")
        b = await manager.video.generate("b")
        await manager.jobs.wait(a)
        assert {j.id for j in manager.jobs.list()} == {a.id, b.id}
        assert [j.id for j in manager.jobs.list(active_only=True)] == [b.id]
        manager.jobs.forget(b.id)
        assert manager.jobs.get(b.id) is None

    async def test_unknown_provider_cannot_be_polled(self, manager):
        job = await manager.video.generate("x")
        manager.unregister_adapter("fake")
        with pytest.raises(MediaJobError, match="no longer configured"):
            await manager.jobs.poll(job)

    async def test_non_polling_provider_is_rejected(self):
        class NoPoll(FakeMediaProvider):
            poll_media_job = None  # type: ignore[assignment]

        adapter = FakeMediaProvider("fake")
        m = MediaManager({"fake": adapter}, job_policy=FAST)
        job = await m.video.generate("x")
        m.register_adapter("fake", MagicMock(spec=["get_name", "media_capabilities", "media_execution"]))
        with pytest.raises(MediaJobError, match="does not implement"):
            await m.jobs.poll(job)

    async def test_close_leaves_active_jobs_running(self, manager):
        job = await manager.video.generate("x")
        await manager.close()
        assert not job.is_terminal


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class TestArtifactStore:
    def test_put_is_content_addressed_and_idempotent(self, tmp_path):
        store = ArtifactStore(tmp_path)
        cs1, p1 = store.put(b"hello", suffix=".txt")
        cs2, p2 = store.put(b"hello", suffix=".txt")
        assert cs1 == cs2 and p1 == p2
        assert store.has(cs1, suffix=".txt")
        # sharded two levels deep
        assert p1.relative_to(tmp_path).parts[:2] == (cs1[:2], cs1[2:4])

    def test_put_leaves_no_partial_file(self, tmp_path):
        store = ArtifactStore(tmp_path)
        store.put(b"data")
        assert list(tmp_path.rglob("*.part")) == []

    @pytest.mark.parametrize(
        ("policy", "with_expiry", "expected"),
        [
            (MaterializePolicy.NEVER, True, False),
            (MaterializePolicy.ALWAYS, False, True),
            (MaterializePolicy.ON_EXPIRY, True, True),
            (MaterializePolicy.ON_EXPIRY, False, False),
        ],
    )
    def test_should_materialize(self, tmp_path, policy, with_expiry, expected):
        store = ArtifactStore(tmp_path, policy=policy)
        a = MediaArtifact(
            kind=MediaKind.IMAGE,
            uri="https://x/y.png",
            expires_at=datetime.now(UTC) + timedelta(hours=1) if with_expiry else None,
        )
        assert store.should_materialize(a) is expected

    def test_inline_bytes_are_never_refetched(self, tmp_path):
        store = ArtifactStore(tmp_path, policy=MaterializePolicy.ALWAYS)
        assert store.should_materialize(
            MediaArtifact(kind=MediaKind.IMAGE, uri="u", data=b"x")
        ) is False

    async def test_materialize_rewrites_uri_and_keeps_source(self, tmp_path):
        async def fetch(url: str) -> bytes:
            return b"PNG" + url.encode()

        store = ArtifactStore(tmp_path, policy=MaterializePolicy.ALWAYS, fetcher=fetch)
        a = MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y.png", mime_type="image/png")
        m = await store.materialize(a)
        assert m.uri.startswith("file://") and m.data == b"PNGhttps://x/y.png"
        assert m.checksum_sha256 and m.expires_at is None
        assert m.provider_metadata["source_uri"] == "https://x/y.png"

    async def test_materialize_is_a_noop_under_never(self, tmp_path):
        store = ArtifactStore(tmp_path, policy=MaterializePolicy.NEVER, fetcher=None)
        a = MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y.png")
        assert (await store.materialize(a)) is a

    async def test_materialize_without_fetcher_raises(self, tmp_path):
        store = ArtifactStore(tmp_path, policy=MaterializePolicy.ALWAYS)
        with pytest.raises(MediaError, match="requires a fetcher"):
            await store.materialize(MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y"))

    async def test_materialize_wraps_fetch_failures(self, tmp_path):
        async def boom(_url):
            raise RuntimeError("404")

        store = ArtifactStore(tmp_path, policy=MaterializePolicy.ALWAYS, fetcher=boom)
        with pytest.raises(MediaError, match="Failed to materialize"):
            await store.materialize(MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y"))

    async def test_download_uses_inline_bytes(self, tmp_path):
        store = ArtifactStore(tmp_path)
        dest = tmp_path / "out" / "a.png"
        out = await store.download(MediaArtifact(kind=MediaKind.IMAGE, data=b"raw"), dest)
        assert out.read_bytes() == b"raw"

    async def test_download_forces_fetch_even_under_never(self, tmp_path):
        async def fetch(_url):
            return b"fetched"

        store = ArtifactStore(tmp_path, policy=MaterializePolicy.NEVER, fetcher=fetch)
        dest = tmp_path / "b.png"
        await store.download(MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y"), dest)
        assert dest.read_bytes() == b"fetched"

    def test_gc_without_keep_set_is_a_noop(self, tmp_path):
        store = ArtifactStore(tmp_path)
        cs, _ = store.put(b"keep")
        assert store.gc() == 0
        assert store.has(cs)

    def test_gc_removes_unlisted(self, tmp_path):
        store = ArtifactStore(tmp_path)
        keep, _ = store.put(b"keep")
        drop, _ = store.put(b"drop")
        assert store.gc(keep_checksums={keep}) == 1
        assert store.has(keep) and not store.has(drop)

    def test_construction_does_not_touch_the_filesystem(self, tmp_path):
        target = tmp_path / "not-created-yet"
        ArtifactStore(target)
        assert not target.exists()
