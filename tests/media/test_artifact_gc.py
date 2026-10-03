"""Evicting cached artifacts, and why that is safe.

`ArtifactStore.gc` existed and had no callers, because it took a *keep-set*
and nothing computed one — a collector with no way to say what was still in
use. Worse, eviction was not recoverable: materialization re-points an
artifact's `uri` at the local copy, so deleting that copy left the artifact
pointing at a dead path even though the provider URL had been preserved
under `provider_metadata["source_uri"]`.

So recovery comes first and eviction second: the store is only a *cache* if
a missing file can be re-fetched, and only then is throwing one away a
tuning decision rather than data loss.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llmcore.media.artifacts import ArtifactStore, GcReport
from llmcore.media.models import MediaArtifact, MediaKind


def artifact(uri="https://example.com/a.png") -> MediaArtifact:
    return MediaArtifact(kind=MediaKind.IMAGE, uri=uri, mime_type="image/png")


@pytest.fixture
def store(tmp_path):
    calls: list[str] = []

    async def fetcher(uri: str) -> bytes:
        calls.append(uri)
        return b"the original bytes"

    s = ArtifactStore(base_path=tmp_path, fetcher=fetcher)
    s._test_calls = calls          # type: ignore[attr-defined]
    return s


def drop_data(art: MediaArtifact) -> MediaArtifact:
    return MediaArtifact(**{**{f: getattr(art, f) for f in art.__slots__},
                            "data": None})


class TestRecovery:
    @pytest.mark.asyncio
    async def test_an_evicted_artifact_is_refetched_from_its_source(self, store):
        stored = await store.materialize(artifact(), force=True)
        assert store.gc(max_age_days=0.0).removed == 1

        again = await store.materialize(drop_data(stored), force=True)
        assert again.data == b"the original bytes"
        assert store._test_calls[-1] == "https://example.com/a.png"

    @pytest.mark.asyncio
    async def test_an_intact_local_file_is_not_refetched(self, store):
        stored = await store.materialize(artifact(), force=True)
        before = len(store._test_calls)
        # data present -> returned as-is, no fetch at all
        assert (await store.materialize(stored, force=True)).data is not None
        assert len(store._test_calls) == before

    @pytest.mark.asyncio
    async def test_without_a_recorded_source_it_does_not_pretend(self, store):
        # A file:// artifact that never came from a provider has nothing to
        # fall back to; the uri is left alone rather than invented.
        orphan = MediaArtifact(kind=MediaKind.IMAGE,
                               uri=(Path("/nonexistent/x.png")).as_uri())
        recovered = store._recover_source(orphan)
        assert recovered.uri == orphan.uri

    def test_a_non_local_uri_is_untouched(self, store):
        art = artifact()
        assert store._recover_source(art) is art


class TestSafetyOfTheDefault:
    def test_calling_gc_with_no_policy_removes_nothing(self, store, tmp_path):
        store.put(b"x")
        report = store.gc()
        assert report.removed == 0
        assert len(store.entries()) == 1

    def test_an_empty_store_is_fine(self, store):
        assert store.gc(max_age_days=0.0) == GcReport(
            removed=0, freed_bytes=0, kept=0)


class TestReferenceCollection:
    def test_unreferenced_files_are_evicted(self, store):
        keep, _ = store.put(b"keep me")
        store.put(b"evict me")
        report = store.gc(keep_checksums={keep})
        assert report.removed == 1
        assert [c for c, *_ in store.entries()] == [keep]

    def test_live_checksums_builds_the_keep_set(self, store):
        digest, _ = store.put(b"keep me")
        art = MediaArtifact(kind=MediaKind.IMAGE, uri="x",
                            checksum_sha256=digest)
        assert store.live_checksums([art]) == {digest}

    def test_an_empty_keep_set_evicts_everything(self, store):
        # Distinct from None: passing an empty set is a positive statement
        # that nothing is live.
        store.put(b"a")
        store.put(b"b")
        assert store.gc(keep_checksums=set()).removed == 2


class TestAgeAndBudget:
    def _age(self, path: Path, days: float) -> None:
        old = time.time() - days * 86400
        import os

        os.utime(path, (old, old))

    def test_stale_files_are_evicted(self, store):
        _, fresh = store.put(b"fresh")
        _, stale = store.put(b"stale")
        self._age(stale, 30)
        report = store.gc(max_age_days=7)
        assert report.removed == 1
        assert fresh.exists() and not stale.exists()

    def test_budget_evicts_oldest_first_and_stops_when_it_fits(self, store):
        paths = []
        for i, blob in enumerate((b"a" * 100, b"b" * 100, b"c" * 100)):
            _, path = store.put(blob)
            self._age(path, 10 - i)        # first is oldest
            paths.append(path)
        report = store.gc(max_total_bytes=250)
        assert report.removed == 1          # 300 -> 200 fits
        assert not paths[0].exists()
        assert paths[1].exists() and paths[2].exists()

    def test_an_explicit_keep_outranks_age(self, store):
        digest, path = store.put(b"old but live")
        self._age(path, 100)
        assert store.gc(max_age_days=1, keep_checksums={digest}).removed == 0
        assert path.exists()

    def test_an_explicit_keep_outranks_the_budget(self, store):
        digest, path = store.put(b"x" * 500)
        self._age(path, 100)
        assert store.gc(max_total_bytes=1, keep_checksums={digest}).removed == 0

    def test_policies_compose_without_double_counting(self, store):
        _, a = store.put(b"a" * 100)
        _, b = store.put(b"b" * 100)
        self._age(a, 100)
        self._age(b, 100)
        report = store.gc(max_age_days=1, max_total_bytes=0)
        assert report.removed == 2          # not 4


class TestReporting:
    def test_dry_run_reports_without_removing(self, store):
        store.put(b"doomed")
        report = store.gc(max_age_days=0.0, dry_run=True)
        assert report.dry_run and report.removed == 1
        assert len(store.entries()) == 1

    def test_the_report_says_why_each_file_went(self, store):
        _, path = store.put(b"stale")
        import os

        old = time.time() - 99 * 86400
        os.utime(path, (old, old))
        report = store.gc(max_age_days=1)
        assert any(d.endswith(":stale") for d in report.details)

    def test_freed_bytes_and_kept_are_accounted(self, store):
        store.put(b"x" * 10)
        store.put(b"y" * 20)
        report = store.gc(keep_checksums=set())
        assert report.freed_bytes == 30
        assert report.kept == 0

    def test_total_bytes_matches_the_entries(self, store):
        store.put(b"x" * 7)
        assert store.total_bytes() == 7 == sum(s for _c, _p, s, _m in store.entries())
