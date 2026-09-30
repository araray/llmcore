# src/llmcore/media/testing.py
"""In-repo fake media adapter, for exercising the subsystem without a network.

:class:`FakeMediaProvider` implements every capability protocol, so router
dispatch, capability discovery, selection policy, job lifecycle and artifact
materialization can all be tested without a vendor account.  It is also the
reference for what a real adapter must implement.

Shipped inside the package (not under ``tests/``) so that downstream projects
building their own media adapters can use it in their suites too.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import (
    MediaArtifact,
    MediaCapability,
    MediaExecution,
    MediaJob,
    MediaJobStatus,
    MediaKind,
    MediaRef,
    MediaResult,
    MediaUsage,
)

__all__ = ["FakeMediaProvider"]

#: Capabilities the fake serves as long-running jobs rather than immediately.
_JOB_CAPABILITIES: frozenset[MediaCapability] = frozenset(
    {
        MediaCapability.VIDEO_GENERATE,
        MediaCapability.VIDEO_EDIT,
        MediaCapability.VIDEO_INTERPOLATE,
    }
)


class FakeMediaProvider:
    """A deterministic, offline media adapter.

    Args:
        name: Provider instance name reported by :meth:`get_name`.
        capabilities: Capabilities to declare; every capability by default.
        poll_count: How many polls a job needs before succeeding. ``0`` makes
            jobs succeed on submission.
        fail_jobs: Make every job terminate ``FAILED``, for error-path tests.
        declare_only: Capabilities to declare **without** implementing, to test
            that the manager drops unbacked declarations.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        capabilities: Sequence[MediaCapability] | None = None,
        poll_count: int = 1,
        fail_jobs: bool = False,
        declare_only: Sequence[MediaCapability] | None = None,
    ) -> None:
        self._name = name
        self._capabilities = frozenset(capabilities) if capabilities is not None else frozenset(
            MediaCapability
        )
        self._declare_only = frozenset(declare_only or ())
        self._poll_count = max(0, int(poll_count))
        self._fail_jobs = fail_jobs
        self._polls: dict[str, int] = {}
        #: Every call recorded as ``(method, kwargs)``, for assertions.
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # --- MediaCapableProvider ---

    def get_name(self) -> str:
        """Return the instance name."""
        return self._name

    def media_capabilities(self) -> frozenset[MediaCapability]:
        """Return declared capabilities, including any declare-only ones."""
        return self._capabilities | self._declare_only

    def media_execution(
        self, capability: MediaCapability, model: str | None = None
    ) -> MediaExecution:
        """Return the execution class the fake uses for *capability*."""
        if capability in _JOB_CAPABILITIES:
            return MediaExecution.ASYNC_JOB
        if capability in (MediaCapability.TTS_STREAM, MediaCapability.ASR_STREAM):
            return MediaExecution.STREAM
        return MediaExecution.REQUEST_RESPONSE

    # --- helpers ---

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, {k: v for k, v in kwargs.items() if v is not None}))

    def _artifact(self, kind: MediaKind, payload: str, *, expires: bool = False) -> MediaArtifact:
        data = payload.encode()
        return MediaArtifact(
            kind=kind,
            uri=f"https://fake.invalid/{hashlib.sha256(data).hexdigest()[:12]}",
            mime_type={"image": "image/png", "audio": "audio/mpeg", "video": "video/mp4"}.get(
                kind.value, "text/plain"
            ),
            checksum_sha256=hashlib.sha256(data).hexdigest(),
            expires_at=(datetime.now(UTC) + timedelta(hours=1)) if expires else None,
            provider_metadata={"fake": True},
        )

    def _result(
        self, capability: MediaCapability, payload: str, *, model: str | None, n: int = 1,
        kind: MediaKind | None = None, text: str | None = None,
    ) -> MediaResult:
        resolved_kind = kind or {
            True: MediaKind.IMAGE,
        }.get(False, MediaKind.IMAGE)
        artifacts = [self._artifact(resolved_kind, f"{payload}-{i}") for i in range(n)]
        if text is not None:
            artifacts = [
                MediaArtifact(**{**{f: getattr(a, f) for f in a.__slots__}, "text": text})
                for a in artifacts
            ]
        return MediaResult(
            capability=capability,
            provider=self._name,
            model=model or "fake-model",
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self._name, model=model or "fake-model", basis="per_call", images=n
            ),
            raw={"fake": True},
        )

    def _job(self, capability: MediaCapability, *, model: str | None) -> MediaJob:
        job = MediaJob(
            capability=capability,
            provider=self._name,
            model=model or "fake-model",
            provider_job_id=f"fake-{len(self._polls) + 1}",
            idempotency_key=f"idem-{len(self._polls) + 1}",
        )
        self._polls[job.id] = 0
        if self._poll_count == 0:
            self._complete(job)
        return job

    def _complete(self, job: MediaJob) -> None:
        if self._fail_jobs:
            job.status = MediaJobStatus.FAILED
            job.error = "fake failure"
            return
        job.status = MediaJobStatus.SUCCEEDED
        job.progress = 1.0
        job.artifacts = [self._artifact(MediaKind.VIDEO, job.id, expires=True)]
        job.usage = MediaUsage(provider=self._name, model=job.model, basis="per_second", seconds=4.0)

    # --- MediaJobPoller ---

    async def poll_media_job(self, job: MediaJob) -> MediaJob:
        """Advance *job* one step toward completion."""
        self._record("poll_media_job", {"job": job.id})
        if job.is_terminal:
            return job
        self._polls[job.id] = self._polls.get(job.id, 0) + 1
        if self._polls[job.id] >= self._poll_count:
            self._complete(job)
        else:
            job.status = MediaJobStatus.RUNNING
            job.progress = self._polls[job.id] / self._poll_count
        return job

    async def cancel_media_job(self, job: MediaJob) -> MediaJob:
        """Mark *job* cancelled."""
        self._record("cancel_media_job", {"job": job.id})
        job.status = MediaJobStatus.CANCELED
        return job

    # --- image ---

    async def generate_image_media(
        self, prompt: str, *, model: str | None = None, n: int = 1, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce *n* fake images."""
        self._record("generate_image_media", {"prompt": prompt, "model": model, "n": n, **kwargs})
        return self._result(MediaCapability.IMAGE_GENERATE, prompt, model=model, n=n)

    async def edit_image_media(
        self, prompt: str, *, image: MediaRef, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake edited image."""
        self._record("edit_image_media", {"prompt": prompt, "model": model, **kwargs})
        return self._result(MediaCapability.IMAGE_EDIT, prompt, model=model)

    async def upscale_image_media(
        self, *, image: MediaRef, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake upscaled image."""
        self._record("upscale_image_media", {"model": model, **kwargs})
        return self._result(MediaCapability.IMAGE_UPSCALE, "upscaled", model=model)

    async def ocr_media(
        self, *, document: MediaRef, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce fake extracted text."""
        self._record("ocr_media", {"model": model, **kwargs})
        return self._result(
            MediaCapability.OCR, "ocr", model=model, kind=MediaKind.TEXT, text="fake ocr text"
        )

    # --- audio ---

    async def synthesize_speech_media(
        self, text: str, *, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake audio artifact."""
        self._record("synthesize_speech_media", {"text": text, "model": model, **kwargs})
        return self._result(MediaCapability.TTS, text, model=model, kind=MediaKind.AUDIO)

    def stream_speech_media(
        self, text: str, *, model: str | None = None, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Yield the text back as fake audio chunks, one word at a time."""
        self._record("stream_speech_media", {"text": text, "model": model, **kwargs})

        async def _gen() -> AsyncIterator[bytes]:
            for word in text.split():
                yield word.encode()

        return _gen()

    async def transcribe_media(
        self, *, audio: MediaRef, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake transcript."""
        self._record("transcribe_media", {"model": model, **kwargs})
        return self._result(
            MediaCapability.ASR, "asr", model=model, kind=MediaKind.TEXT, text="fake transcript"
        )

    async def open_transcription_session(self, *, model: str | None = None, **kwargs: Any) -> Any:
        """Return a trivial stand-in session object."""
        self._record("open_transcription_session", {"model": model, **kwargs})
        return {"session": "fake", "model": model}

    async def generate_music_media(
        self, prompt: str, *, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake music artifact."""
        self._record("generate_music_media", {"prompt": prompt, "model": model, **kwargs})
        return self._result(MediaCapability.MUSIC, prompt, model=model, kind=MediaKind.AUDIO)

    async def generate_sfx_media(
        self, prompt: str | None = None, *, model: str | None = None, **kwargs: Any
    ) -> MediaResult | MediaJob:
        """Produce a fake SFX artifact."""
        self._record("generate_sfx_media", {"prompt": prompt, "model": model, **kwargs})
        return self._result(MediaCapability.SFX, prompt or "sfx", model=model, kind=MediaKind.AUDIO)

    # --- video ---

    async def generate_video_media(
        self, prompt: str, *, model: str | None = None, **kwargs: Any
    ) -> MediaJob:
        """Submit a fake video job."""
        self._record("generate_video_media", {"prompt": prompt, "model": model, **kwargs})
        return self._job(MediaCapability.VIDEO_GENERATE, model=model)

    async def edit_video_media(
        self, prompt: str, *, video: MediaRef, model: str | None = None, **kwargs: Any
    ) -> MediaJob:
        """Submit a fake video-edit job."""
        self._record("edit_video_media", {"prompt": prompt, "model": model, **kwargs})
        return self._job(MediaCapability.VIDEO_EDIT, model=model)

    async def interpolate_video_media(
        self, *, video: MediaRef | None = None, model: str | None = None, **kwargs: Any
    ) -> MediaJob:
        """Submit a fake interpolation job."""
        self._record("interpolate_video_media", {"model": model, **kwargs})
        return self._job(MediaCapability.VIDEO_INTERPOLATE, model=model)
