# src/llmcore/media/models.py
"""Provider-agnostic types for the LLMCore media subsystem.

These are the contract between callers and media provider adapters, for
generative image, audio and video work.  They are deliberately decoupled from
any vendor SDK so that switching between OpenAI, Google, fal, ElevenLabs,
Replicate or a Hugging Face endpoint does not change how results are read —
mirroring how :mod:`llmcore.search.models` normalizes search results and
:class:`~llmcore.providers.base.BaseProvider` normalizes chat responses.

Design notes
------------
* **Three execution classes, not one.**  Image generation is usually
  request/response, speech is often a byte stream, and video is almost always a
  long-running job.  :class:`MediaResult` covers the first,
  ``AsyncIterator[bytes]`` the second, and :class:`MediaJob` the third.  Forcing
  all three into one shape is the main modelling mistake to avoid.
* **Inputs are refs, not bytes.**  :class:`MediaRef` lets a caller pass a URL, a
  path, raw bytes, or a previously produced :class:`MediaArtifact`; the adapter
  decides whether its API wants an upload, a URL or inline base64.  Callers never
  hand-roll base64.
* **Artifacts know when they die.**  Every aggregator returns short-lived URLs,
  so :attr:`MediaArtifact.expires_at` and :attr:`MediaArtifact.checksum_sha256`
  are first-class: they let :class:`~llmcore.media.artifacts.ArtifactStore`
  decide what to materialize, and let callers dedupe.
* **Usage keeps native units.**  Media vendors bill per image, per megapixel,
  per second of video, per audio-minute, per character or per compute-second.
  :class:`MediaUsage` records whichever units the vendor reported plus an
  explicitly-stamped cost estimate, rather than inventing a token count.
* ``raw`` / ``provider_metadata`` always preserve the vendor payload so power
  users can reach fields the normalizer does not surface.

See ``docs/MEDIA_SUBSYSTEM_SPEC.md`` for the full design.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "AUDIO_CAPABILITIES",
    "CAPABILITY_KINDS",
    "IMAGE_CAPABILITIES",
    "TERMINAL_JOB_STATUSES",
    "VIDEO_CAPABILITIES",
    "MediaArtifact",
    "MediaCapability",
    "MediaExecution",
    "MediaJob",
    "MediaJobStatus",
    "MediaKind",
    "MediaProvenance",
    "MediaRef",
    "MediaResult",
    "MediaUsage",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MediaKind(StrEnum):
    """The modality of a media input or output."""

    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class MediaCapability(StrEnum):
    """A specific operation a media provider can perform.

    Capability is the unit of routing and of model-card declaration: callers ask
    for a capability, and the router resolves it to a provider that implements
    the matching protocol.  Provider names never appear in routing logic.
    """

    # --- audio ---
    TTS = "tts"
    TTS_STREAM = "tts_stream"
    ASR = "asr"
    ASR_STREAM = "asr_stream"
    VOICE_AGENT = "voice_agent"
    MUSIC = "music"
    SFX = "sfx"
    VOICE_DESIGN = "voice_design"

    # --- image ---
    IMAGE_GENERATE = "image_generate"
    IMAGE_EDIT = "image_edit"
    IMAGE_UPSCALE = "image_upscale"
    IMAGE_VARIATE = "image_variate"
    OCR = "ocr"

    # --- video ---
    VIDEO_GENERATE = "video_generate"
    VIDEO_EDIT = "video_edit"
    VIDEO_INTERPOLATE = "video_interpolate"
    VIDEO_REFRAME = "video_reframe"
    VIDEO_UPSCALE = "video_upscale"
    VIDEO_EXTEND = "video_extend"


class MediaExecution(StrEnum):
    """How an operation completes.

    Declared per model in its card so callers can tell, before submitting,
    whether to expect a result, a stream or a job handle.
    """

    REQUEST_RESPONSE = "request_response"
    STREAM = "stream"
    ASYNC_JOB = "async_job"


class MediaJobStatus(StrEnum):
    """Lifecycle state of a long-running media job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


#: Capabilities grouped by the router that owns them.
AUDIO_CAPABILITIES: frozenset[MediaCapability] = frozenset(
    {
        MediaCapability.TTS,
        MediaCapability.TTS_STREAM,
        MediaCapability.ASR,
        MediaCapability.ASR_STREAM,
        MediaCapability.VOICE_AGENT,
        MediaCapability.MUSIC,
        MediaCapability.SFX,
        MediaCapability.VOICE_DESIGN,
    }
)

IMAGE_CAPABILITIES: frozenset[MediaCapability] = frozenset(
    {
        MediaCapability.IMAGE_GENERATE,
        MediaCapability.IMAGE_EDIT,
        MediaCapability.IMAGE_UPSCALE,
        MediaCapability.IMAGE_VARIATE,
        MediaCapability.OCR,
    }
)

VIDEO_CAPABILITIES: frozenset[MediaCapability] = frozenset(
    {
        MediaCapability.VIDEO_GENERATE,
        MediaCapability.VIDEO_EDIT,
        MediaCapability.VIDEO_INTERPOLATE,
        MediaCapability.VIDEO_REFRAME,
        MediaCapability.VIDEO_UPSCALE,
        MediaCapability.VIDEO_EXTEND,
    }
)

#: The modality each capability produces.  OCR is the one audio/image capability
#: whose *output* modality differs from its router's modality.
CAPABILITY_KINDS: dict[MediaCapability, MediaKind] = {
    **dict.fromkeys(AUDIO_CAPABILITIES, MediaKind.AUDIO),
    **dict.fromkeys(IMAGE_CAPABILITIES, MediaKind.IMAGE),
    **dict.fromkeys(VIDEO_CAPABILITIES, MediaKind.VIDEO),
    MediaCapability.ASR: MediaKind.TEXT,
    MediaCapability.ASR_STREAM: MediaKind.TEXT,
    MediaCapability.OCR: MediaKind.TEXT,
}

#: Job statuses from which no further transition occurs.
TERMINAL_JOB_STATUSES: frozenset[MediaJobStatus] = frozenset(
    {
        MediaJobStatus.SUCCEEDED,
        MediaJobStatus.FAILED,
        MediaJobStatus.CANCELED,
        MediaJobStatus.EXPIRED,
    }
)


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def _serialize(value: Any) -> Any:
    """Recursively make a value JSON-compatible.

    ``datetime`` becomes ISO-8601, ``bytes`` becomes a byte count marker rather
    than inline base64 (media payloads are far too large to serialize by
    accident), and enums become their values.

    Args:
        value: Any value that may contain nested datetimes, bytes or enums.

    Returns:
        A structurally identical value that ``json.dumps`` accepts.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MediaRef:
    """A reference to media supplied *to* a provider.

    Exactly one of :attr:`url`, :attr:`path` or :attr:`data` is set.  Adapters
    call :meth:`read_bytes` or :meth:`as_data_uri` when their API needs inline
    content, and use :attr:`url` directly when it accepts a remote reference —
    so callers never encode base64 themselves.

    Attributes:
        url: A remote URL the provider can fetch.
        path: A local filesystem path.
        data: Raw bytes.
        mime_type: Media type; inferred from ``path``/``url`` when omitted.
        filename: Preferred upload filename.
    """

    url: str | None = None
    path: Path | None = None
    data: bytes | None = None
    mime_type: str | None = None
    filename: str | None = None

    def __post_init__(self) -> None:
        provided = [x is not None for x in (self.url, self.path, self.data)]
        if sum(provided) != 1:
            raise ValueError("MediaRef requires exactly one of url, path or data.")

    # --- constructors ---

    @classmethod
    def from_url(cls, url: str, *, mime_type: str | None = None) -> MediaRef:
        """Build a ref from a remote URL."""
        return cls(url=url, mime_type=mime_type or _guess_mime(url))

    @classmethod
    def from_path(cls, path: str | Path, *, mime_type: str | None = None) -> MediaRef:
        """Build a ref from a local file path."""
        p = Path(path)
        return cls(
            path=p,
            mime_type=mime_type or _guess_mime(p.name),
            filename=p.name,
        )

    @classmethod
    def from_bytes(
        cls, data: bytes, *, mime_type: str | None = None, filename: str | None = None
    ) -> MediaRef:
        """Build a ref from raw bytes."""
        return cls(data=data, mime_type=mime_type, filename=filename)

    @classmethod
    def from_artifact(cls, artifact: MediaArtifact) -> MediaRef:
        """Chain a previously produced artifact back in as an input.

        Prefers inline bytes when the artifact carries them, otherwise its URI.

        Raises:
            ValueError: If the artifact has neither ``data`` nor ``uri``.
        """
        if artifact.data is not None:
            return cls(data=artifact.data, mime_type=artifact.mime_type)
        if artifact.uri:
            return cls(url=artifact.uri, mime_type=artifact.mime_type)
        raise ValueError("MediaArtifact carries neither data nor uri; cannot build a ref.")

    # --- accessors ---

    def read_bytes(self) -> bytes:
        """Return the referenced content as bytes.

        Returns:
            The raw content.

        Raises:
            ValueError: If this ref is a URL (the adapter must fetch it, since
                only the adapter knows which client and auth to use).
        """
        if self.data is not None:
            return self.data
        if self.path is not None:
            return self.path.read_bytes()
        raise ValueError(
            "MediaRef holds a URL; the provider adapter must fetch it with its own client."
        )

    def as_data_uri(self) -> str:
        """Return the content as a ``data:`` URI for inline APIs."""
        payload = base64.b64encode(self.read_bytes()).decode("ascii")
        return f"data:{self.mime_type or 'application/octet-stream'};base64,{payload}"

    @property
    def is_remote(self) -> bool:
        """Whether this ref points at a URL the provider must fetch."""
        return self.url is not None


def _guess_mime(name: str) -> str | None:
    """Best-effort media type from a filename or URL."""
    return mimetypes.guess_type(name)[0]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MediaProvenance:
    """Content-credential / provenance metadata attached to generated media.

    Several vendors now emit C2PA-style credentials or SynthID-style watermarks.
    Modelled from day one so it is not retrofitted later.

    Attributes:
        watermarked: Whether the provider states the output is watermarked.
        c2pa_manifest: Raw C2PA manifest, when supplied.
        generator: Model or system credited with producing the asset.
        provider_declared: Whether these facts come from the provider (``True``)
            or were inferred locally (``False``).
    """

    watermarked: bool | None = None
    c2pa_manifest: Mapping[str, Any] | None = None
    generator: str | None = None
    provider_declared: bool = True


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    """One produced media asset.

    Attributes:
        kind: Modality of this asset.
        uri: Provider URL or artifact-store URI.
        data: Inline bytes, for small results.
        mime_type: Media type of the asset.
        width: Pixel width (image/video).
        height: Pixel height (image/video).
        duration_seconds: Duration (audio/video).
        sample_rate_hz: Sample rate (audio).
        fps: Frames per second (video).
        frame_count: Total frames (video).
        text: Transcribed or extracted text (ASR/OCR outputs).
        checksum_sha256: Hex digest of the bytes, when known.
        expires_at: When ``uri`` stops resolving, when the provider says so.
        provenance: Content-credential metadata.
        provider_metadata: Untouched vendor fields.
    """

    kind: MediaKind
    uri: str | None = None
    data: bytes | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    text: str | None = None
    checksum_sha256: str | None = None
    expires_at: datetime | None = None
    provenance: MediaProvenance | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Whether :attr:`expires_at` is in the past."""
        return self.expires_at is not None and self.expires_at <= _utcnow()

    @property
    def needs_materialization(self) -> bool:
        """Whether the bytes should be fetched before the URI dies.

        ``True`` when the asset exists only as a URI that carries an expiry.
        """
        return self.data is None and self.uri is not None and self.expires_at is not None

    def with_data(self, data: bytes) -> MediaArtifact:
        """Return a copy carrying *data* and its checksum.

        Args:
            data: The materialized bytes.

        Returns:
            A new artifact with ``data`` and ``checksum_sha256`` populated.
        """
        return MediaArtifact(
            **{
                **{f: getattr(self, f) for f in self.__slots__},
                "data": data,
                "checksum_sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible view; inline bytes are summarized, never embedded."""
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class MediaUsage:
    """Normalized-but-honest billing units for one media operation.

    Media vendors bill in incompatible units, so every native unit the provider
    reported is kept alongside an explicitly stamped estimate.  Never synthesize
    a unit the vendor did not report.

    Attributes:
        provider: Provider instance name.
        model: Model or endpoint that ran.
        basis: Vendor's billing basis (e.g. ``"per_image"``, ``"per_second"``).
        images: Number of images produced.
        megapixels: Megapixels produced.
        seconds: Seconds of output media.
        audio_minutes: Minutes of audio processed or produced.
        characters: Characters consumed (common for TTS).
        input_tokens: Input tokens, when the vendor bills tokens.
        output_tokens: Output tokens, when the vendor bills tokens.
        compute_seconds: Billed compute time (aggregators).
        estimated_cost_usd: Best-effort cost estimate.
        pricing_as_of: Date stamp of the pricing used for the estimate.
        raw: Untouched vendor usage payload.
    """

    provider: str
    model: str
    basis: str | None = None
    images: int | None = None
    megapixels: float | None = None
    seconds: float | None = None
    audio_minutes: float | None = None
    characters: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    compute_seconds: float | None = None
    estimated_cost_usd: float | None = None
    pricing_as_of: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible view."""
        return _serialize(asdict(self))


@dataclass(frozen=True, slots=True)
class MediaResult:
    """The outcome of a request/response media operation.

    Attributes:
        capability: Which operation produced this.
        provider: Provider instance name.
        model: Model or endpoint that ran.
        artifacts: Produced assets, in provider order.
        usage: Billing units, when reported.
        raw: Untouched vendor response.
    """

    capability: MediaCapability
    provider: str
    model: str
    artifacts: Sequence[MediaArtifact] = field(default_factory=tuple)
    usage: MediaUsage | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def artifact(self) -> MediaArtifact:
        """The first artifact.

        Raises:
            IndexError: If the operation produced nothing.
        """
        if not self.artifacts:
            raise IndexError(
                f"{self.capability} on {self.provider}/{self.model} produced no artifacts."
            )
        return self.artifacts[0]

    @property
    def text(self) -> str | None:
        """Concatenated text across artifacts, for ASR/OCR results."""
        parts = [a.text for a in self.artifacts if a.text]
        return "\n".join(parts) if parts else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible view."""
        return _serialize(asdict(self))


@dataclass(slots=True)
class MediaJob:
    """Handle for a long-running media operation.

    Normalizes fal queue requests, Replicate predictions, Veo operations and
    Luma generations without pretending their native APIs are identical: the
    vendor's own identifiers and poll targets live in :attr:`provider_job_id`
    and :attr:`poll_url`, and the whole payload stays in
    :attr:`provider_metadata`.

    Attributes:
        id: llmcore-local job id, stable across resumes.
        capability: Which operation this job performs.
        provider: Provider instance name.
        model: Model or endpoint that runs it.
        status: Current lifecycle state.
        provider_job_id: The vendor's identifier.
        poll_url: Vendor status URL, when it supplies one.
        artifacts: Produced assets once succeeded.
        usage: Billing units, when reported.
        error: Failure message when ``status`` is ``FAILED``.
        progress: Fractional progress in ``[0, 1]`` when the vendor reports it.
        idempotency_key: Client-generated key so a retried submit cannot
            double-bill, and a resumed process re-attaches instead of
            resubmitting.
        created_at: When llmcore submitted the job.
        updated_at: Last status refresh.
        queue_position: Position in the vendor queue, when reported.
        provider_metadata: Untouched vendor payload.
    """

    capability: MediaCapability
    provider: str
    model: str
    status: MediaJobStatus = MediaJobStatus.QUEUED
    id: str = field(default_factory=lambda: f"mj_{uuid.uuid4().hex[:16]}")
    provider_job_id: str | None = None
    poll_url: str | None = None
    artifacts: list[MediaArtifact] = field(default_factory=list)
    usage: MediaUsage | None = None
    error: str | None = None
    progress: float | None = None
    idempotency_key: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    queue_position: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether no further status transition will occur."""
        return self.status in TERMINAL_JOB_STATUSES

    @property
    def succeeded(self) -> bool:
        """Whether the job completed successfully."""
        return self.status is MediaJobStatus.SUCCEEDED

    def to_result(self) -> MediaResult:
        """Project a succeeded job into a :class:`MediaResult`.

        Returns:
            The equivalent request/response result, so callers can treat both
            execution classes uniformly once a job finishes.

        Raises:
            ValueError: If the job has not succeeded.
        """
        if not self.succeeded:
            raise ValueError(
                f"Job {self.id} is {self.status}, not succeeded; cannot project to a result."
            )
        return MediaResult(
            capability=self.capability,
            provider=self.provider,
            model=self.model,
            artifacts=tuple(self.artifacts),
            usage=self.usage,
            raw=dict(self.provider_metadata),
        )

    def touch(self) -> None:
        """Mark the job as just refreshed."""
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible view."""
        return _serialize(asdict(self))
