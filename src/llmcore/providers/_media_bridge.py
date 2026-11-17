# src/llmcore/providers/_media_bridge.py
"""Expose a provider's legacy media methods through the media protocols.

Mistral and Z.ai implement speech, transcription, OCR (and Z.ai images and
video) as the older ``BaseProvider`` methods: ``generate_speech``,
``transcribe_audio``, ``ocr``, ``generate_image``, ``generate_video``. The
media manager only routes to providers that implement
:class:`~llmcore.media.protocols.MediaCapableProvider`, so ``llm.media`` could
not reach any of it. This mixin adds the protocol methods on top of the
existing ones; a provider opts in by listing capabilities in
``_MEDIA_CAPABILITIES``.

The legacy result models already know how to become artifacts
(``SpeechResult.to_artifact`` and friends, in ``models_multimodal``), so the
bridge only translates arguments and wraps results.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ..media.models import MediaCapability, MediaExecution, MediaJob, MediaRef, MediaResult

logger = logging.getLogger(__name__)


async def _ref_bytes(ref: MediaRef) -> bytes:
    """The bytes behind *ref*, fetching a remote one."""
    if ref.is_remote:
        from ..media.artifacts import default_fetcher

        return await default_fetcher()(ref.url or "")
    return ref.read_bytes()


def _ref_url(ref: MediaRef | None) -> str | None:
    """A URL for *ref*: remote refs as-is, local bytes as a data URI."""
    if ref is None:
        return None
    if ref.is_remote:
        return ref.url
    mime = ref.mime_type or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(ref.read_bytes()).decode('ascii')}"


class LegacyMediaBridge:
    """Mixin: media-protocol methods implemented by the legacy media methods.

    Subclasses set ``_MEDIA_CAPABILITIES`` to the capability names they really
    serve; methods for undeclared capabilities exist but are never routed to.
    """

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset()

    def media_capabilities(self) -> frozenset[MediaCapability]:
        """Capabilities this provider serves through ``llm.media``."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._MEDIA_CAPABILITIES)

    def media_execution(
        self, capability: MediaCapability, model: str | None = None
    ) -> MediaExecution:
        """Video is a vendor task (a job); everything else answers in one call."""
        from ..media.models import MediaCapability, MediaExecution

        if capability is MediaCapability.VIDEO_GENERATE:
            return MediaExecution.ASYNC_JOB
        return MediaExecution.REQUEST_RESPONSE

    # --- speech -------------------------------------------------------------

    async def synthesize_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Text to speech via the provider's ``generate_speech``."""
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if sample_rate_hz is not None:
            logger.debug("%s TTS has no sample-rate parameter; ignoring it.", self.get_name())
        options: dict[str, Any] = {}
        if voice is not None:
            options["voice"] = voice
        if audio_format is not None:
            options["response_format"] = audio_format
        if speed is not None:
            options["speed"] = speed
        speech = await self.generate_speech(text, model=model, **options, **kwargs)  # type: ignore[attr-defined]
        return MediaResult(
            capability=MediaCapability.TTS,
            provider=self.get_name(),
            model=speech.model,
            artifacts=(speech.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(), model=speech.model, basis="per_character",
                characters=len(text),
            ),
            raw=dict(speech.metadata),
        )

    # --- transcription ------------------------------------------------------

    async def transcribe_media(
        self,
        *,
        audio: MediaRef,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Speech to text via the provider's ``transcribe_audio``.

        ``timestamps`` asks for segment timings; ``diarize`` is passed through
        for the models that support it.
        """
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        extra: dict[str, Any] = dict(kwargs)
        if timestamps:
            extra.setdefault("timestamp_granularities", ["segment"])
        if diarize is not None:
            extra.setdefault("diarize", diarize)
        transcript = await self.transcribe_audio(  # type: ignore[attr-defined]
            await _ref_bytes(audio), model=model, language=language, **extra
        )
        seconds = transcript.duration_seconds
        return MediaResult(
            capability=MediaCapability.ASR,
            provider=self.get_name(),
            model=transcript.model,
            artifacts=(transcript.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(), model=transcript.model, basis="per_audio_minute",
                audio_minutes=seconds / 60.0 if seconds else None, seconds=seconds,
            ),
            raw=dict(transcript.metadata),
        )

    # --- OCR ----------------------------------------------------------------

    async def ocr_media(
        self,
        *,
        document: MediaRef,
        model: str | None = None,
        pages: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Document OCR via the provider's ``ocr``.

        A remote document is passed as its URL (both vendors fetch URLs);
        local bytes are uploaded.
        """
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        source: Any = document.url if document.is_remote else document.read_bytes()
        result = await self.ocr(  # type: ignore[attr-defined]
            source, model=model, pages=list(pages) if pages is not None else None, **kwargs
        )
        return MediaResult(
            capability=MediaCapability.OCR,
            provider=self.get_name(),
            model=result.model,
            artifacts=(result.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(), model=result.model, basis="per_page",
                raw={"pages_processed": result.pages_processed},
            ),
            raw=dict(result.metadata),
        )

    # --- images -------------------------------------------------------------

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: Sequence[MediaRef] | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Text to image via the provider's ``generate_image``."""
        from ..exceptions import ProviderError
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if reference_images:
            raise ProviderError(
                self.get_name(),
                "This provider's image endpoint takes no reference images.",
                status_code=400,
                retryable=False,
            )
        for name, value in (("seed", seed), ("negative_prompt", negative_prompt)):
            if value is not None:
                logger.debug("%s image generation ignores '%s'.", self.get_name(), name)
        result = await self.generate_image(prompt, model=model, n=n, size=size, **kwargs)  # type: ignore[attr-defined]
        return MediaResult(
            capability=MediaCapability.IMAGE_GENERATE,
            provider=self.get_name(),
            model=result.model,
            artifacts=tuple(result.to_artifacts()),
            usage=MediaUsage(
                provider=self.get_name(), model=result.model, basis="per_image",
                images=len(result.images),
            ),
            raw=dict(result.metadata),
        )


class ZaiVideoJobs:
    """Mixin: Z.ai (CogVideoX) video as a polled :class:`MediaJob`.

    ``POST /videos/generations`` returns a task id; ``GET /async-result/{id}``
    reports ``task_status`` (``PROCESSING`` / ``SUCCESS`` / ``FAIL``) and, on
    success, ``video_result: [{"url", "cover_image_url"}]``.
    """

    async def generate_video_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        first_frame: MediaRef | None = None,
        last_frame: MediaRef | None = None,
        reference_images: Any = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        fps: float | None = None,
        with_audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Submit a video task; poll it with :meth:`poll_media_job`."""
        from ..media.models import MediaCapability, MediaJob, MediaJobStatus

        dropped = [
            name for name, value in (
                ("last_frame", last_frame), ("reference_images", reference_images),
                ("aspect_ratio", aspect_ratio), ("seed", seed),
            ) if value is not None
        ]
        if dropped:
            logger.warning("Z.ai video takes no %s; dropped.", ", ".join(dropped))
        task = await self.generate_video(  # type: ignore[attr-defined]
            prompt,
            model=model,
            image_url=_ref_url(first_frame),
            size=resolution,
            duration=int(duration_seconds) if duration_seconds is not None else None,
            fps=int(fps) if fps is not None else None,
            with_audio=with_audio,
            wait=False,
            **kwargs,
        )
        job = MediaJob(
            capability=MediaCapability.VIDEO_GENERATE,
            provider=self.get_name(),
            model=str(task.get("model") or model or getattr(self, "default_video_model", "")),
            status=MediaJobStatus.QUEUED,
            provider_job_id=task.get("id"),
            provider_metadata={"task": task},
        )
        return self._apply_video_task(job, task)

    async def poll_media_job(self, job: MediaJob) -> MediaJob:
        """Refresh a video task via ``GET /async-result/{id}``."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        if not job.provider_job_id:
            job.status = MediaJobStatus.FAILED
            job.error = "Lost the Z.ai task id; the job cannot be polled."
            return job
        result = await self.retrieve_video_result(job.provider_job_id)  # type: ignore[attr-defined]
        return self._apply_video_task(job, result)

    async def cancel_media_job(self, job: MediaJob) -> MediaJob:
        """Z.ai has no cancel endpoint for video tasks; say so."""
        from ..exceptions import ProviderError

        raise ProviderError(
            self.get_name(),
            "Z.ai video tasks cannot be cancelled once submitted; the task runs to "
            "completion and is billed.",
            model_name=job.model,
            retryable=False,
        )

    def _apply_video_task(self, job: MediaJob, task: dict[str, Any]) -> MediaJob:
        """Fold a task/result payload into *job*."""
        from ..media.models import MediaArtifact, MediaJobStatus, MediaKind

        job.provider_metadata["last_status"] = task
        job.touch()
        status = str(task.get("task_status") or "").upper()
        if status in ("FAIL", "FAILED"):
            job.status = MediaJobStatus.FAILED
            job.error = str(task.get("error") or task.get("message") or "Z.ai video task failed.")
        elif status == "SUCCESS":
            job.artifacts = [
                MediaArtifact(
                    kind=MediaKind.VIDEO,
                    uri=item.get("url"),
                    mime_type="video/mp4",
                    provider_metadata={"cover_image_url": item.get("cover_image_url")},
                )
                for item in task.get("video_result") or []
                if isinstance(item, dict) and item.get("url")
            ]
            job.status = MediaJobStatus.SUCCEEDED
            job.progress = 1.0
        else:
            job.status = MediaJobStatus.RUNNING if status == "PROCESSING" else MediaJobStatus.QUEUED
        return job
