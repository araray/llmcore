# tests/media/test_media_models.py
"""Tests for the provider-agnostic media types.

Covers MediaRef construction/validation, artifact expiry and materialization
signalling, usage/result projection, job lifecycle, and the capability tables.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llmcore.media.models import (
    AUDIO_CAPABILITIES,
    CAPABILITY_KINDS,
    IMAGE_CAPABILITIES,
    TERMINAL_JOB_STATUSES,
    VIDEO_CAPABILITIES,
    MediaArtifact,
    MediaCapability,
    MediaJob,
    MediaJobStatus,
    MediaKind,
    MediaProvenance,
    MediaRef,
    MediaResult,
    MediaUsage,
)


class TestMediaRef:
    def test_from_bytes(self):
        r = MediaRef.from_bytes(b"abc", mime_type="image/png")
        assert r.read_bytes() == b"abc"
        assert r.is_remote is False

    def test_from_path_infers_mime_and_filename(self, tmp_path):
        p = tmp_path / "pic.png"
        p.write_bytes(b"data")
        r = MediaRef.from_path(p)
        assert r.mime_type == "image/png"
        assert r.filename == "pic.png"
        assert r.read_bytes() == b"data"

    def test_from_url_is_remote(self):
        r = MediaRef.from_url("https://example.invalid/a.mp4")
        assert r.is_remote is True
        assert r.mime_type == "video/mp4"

    def test_remote_read_bytes_refuses(self):
        r = MediaRef.from_url("https://example.invalid/a.png")
        with pytest.raises(ValueError, match="provider adapter must fetch"):
            r.read_bytes()

    def test_exactly_one_source_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            MediaRef()
        with pytest.raises(ValueError, match="exactly one"):
            MediaRef(url="https://x", data=b"y")

    def test_as_data_uri(self):
        r = MediaRef.from_bytes(b"abc", mime_type="image/png")
        assert r.as_data_uri() == "data:image/png;base64,YWJj"

    def test_as_data_uri_without_mime_falls_back(self):
        assert MediaRef.from_bytes(b"abc").as_data_uri().startswith(
            "data:application/octet-stream;base64,"
        )

    def test_from_artifact_prefers_bytes(self):
        a = MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y.png", data=b"raw")
        assert MediaRef.from_artifact(a).data == b"raw"

    def test_from_artifact_falls_back_to_uri(self):
        a = MediaArtifact(kind=MediaKind.IMAGE, uri="https://x/y.png")
        assert MediaRef.from_artifact(a).url == "https://x/y.png"

    def test_from_artifact_requires_content(self):
        with pytest.raises(ValueError, match="neither data nor uri"):
            MediaRef.from_artifact(MediaArtifact(kind=MediaKind.IMAGE))


class TestMediaArtifact:
    def test_expiry_flags(self):
        past = MediaArtifact(
            kind=MediaKind.IMAGE, uri="u", expires_at=datetime.now(UTC) - timedelta(minutes=1)
        )
        future = MediaArtifact(
            kind=MediaKind.IMAGE, uri="u", expires_at=datetime.now(UTC) + timedelta(hours=1)
        )
        assert past.is_expired is True
        assert future.is_expired is False

    def test_needs_materialization_requires_uri_and_expiry(self):
        assert MediaArtifact(
            kind=MediaKind.IMAGE, uri="u", expires_at=datetime.now(UTC)
        ).needs_materialization is True
        # no expiry stated -> we cannot know it dies
        assert MediaArtifact(kind=MediaKind.IMAGE, uri="u").needs_materialization is False
        # already has bytes
        assert MediaArtifact(
            kind=MediaKind.IMAGE, uri="u", data=b"x", expires_at=datetime.now(UTC)
        ).needs_materialization is False

    def test_with_data_sets_checksum_and_preserves_fields(self):
        a = MediaArtifact(kind=MediaKind.AUDIO, uri="u", mime_type="audio/mpeg", sample_rate_hz=44100)
        b = a.with_data(b"hello")
        assert b.data == b"hello"
        assert b.checksum_sha256 == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        assert b.sample_rate_hz == 44100 and b.mime_type == "audio/mpeg"

    def test_to_dict_summarizes_bytes_not_embeds_them(self):
        d = MediaArtifact(kind=MediaKind.IMAGE, data=b"0" * 4096).to_dict()
        assert d["data"] == "<4096 bytes>"
        assert d["kind"] == "image"

    def test_to_dict_serializes_datetimes(self):
        when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert MediaArtifact(kind=MediaKind.IMAGE, expires_at=when).to_dict()["expires_at"] == (
            when.isoformat()
        )

    def test_provenance_is_carried(self):
        a = MediaArtifact(
            kind=MediaKind.IMAGE,
            provenance=MediaProvenance(watermarked=True, generator="fake-1"),
        )
        assert a.provenance.watermarked is True
        assert a.to_dict()["provenance"]["generator"] == "fake-1"


class TestMediaResult:
    def _result(self, **kw):
        return MediaResult(
            capability=MediaCapability.IMAGE_GENERATE, provider="p", model="m", **kw
        )

    def test_artifact_shortcut(self):
        a = MediaArtifact(kind=MediaKind.IMAGE)
        assert self._result(artifacts=(a,)).artifact is a

    def test_artifact_shortcut_raises_when_empty(self):
        with pytest.raises(IndexError, match="produced no artifacts"):
            _ = self._result().artifact

    def test_text_joins_artifact_text(self):
        r = self._result(
            artifacts=(
                MediaArtifact(kind=MediaKind.TEXT, text="one"),
                MediaArtifact(kind=MediaKind.TEXT, text="two"),
            )
        )
        assert r.text == "one\ntwo"

    def test_text_is_none_without_text(self):
        assert self._result(artifacts=(MediaArtifact(kind=MediaKind.IMAGE),)).text is None

    def test_usage_keeps_native_units(self):
        u = MediaUsage(provider="p", model="m", basis="per_second", seconds=8.0, images=None)
        d = u.to_dict()
        assert d["seconds"] == 8.0 and d["basis"] == "per_second"
        # a unit the vendor did not report stays absent rather than synthesized
        assert d["input_tokens"] is None


class TestMediaJob:
    def _job(self, **kw):
        return MediaJob(
            capability=MediaCapability.VIDEO_GENERATE, provider="p", model="m", **kw
        )

    def test_default_status_and_id(self):
        j = self._job()
        assert j.status is MediaJobStatus.QUEUED
        assert j.id.startswith("mj_") and not j.is_terminal

    @pytest.mark.parametrize("status", sorted(TERMINAL_JOB_STATUSES))
    def test_terminal_statuses(self, status):
        assert self._job(status=status).is_terminal is True

    @pytest.mark.parametrize(
        "status", [MediaJobStatus.QUEUED, MediaJobStatus.RUNNING]
    )
    def test_non_terminal_statuses(self, status):
        assert self._job(status=status).is_terminal is False

    def test_to_result_requires_success(self):
        with pytest.raises(ValueError, match="not succeeded"):
            self._job(status=MediaJobStatus.FAILED).to_result()

    def test_to_result_projects_artifacts_and_usage(self):
        usage = MediaUsage(provider="p", model="m", seconds=4.0)
        j = self._job(
            status=MediaJobStatus.SUCCEEDED,
            artifacts=[MediaArtifact(kind=MediaKind.VIDEO, uri="u")],
            usage=usage,
        )
        r = j.to_result()
        assert r.capability is MediaCapability.VIDEO_GENERATE
        assert r.artifacts[0].uri == "u"
        assert r.usage is usage

    def test_touch_advances_updated_at(self):
        j = self._job()
        before = j.updated_at
        j.touch()
        assert j.updated_at >= before


class TestCapabilityTables:
    def test_every_capability_has_a_kind(self):
        missing = [c for c in MediaCapability if c not in CAPABILITY_KINDS]
        assert missing == []

    def test_router_groups_are_disjoint_and_complete(self):
        groups = [AUDIO_CAPABILITIES, IMAGE_CAPABILITIES, VIDEO_CAPABILITIES]
        union = set().union(*groups)
        assert union == set(MediaCapability)
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                assert not (a & b)

    def test_text_producing_capabilities_report_text(self):
        for cap in (MediaCapability.ASR, MediaCapability.ASR_STREAM, MediaCapability.OCR):
            assert CAPABILITY_KINDS[cap] is MediaKind.TEXT
