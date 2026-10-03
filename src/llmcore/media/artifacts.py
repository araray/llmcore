# src/llmcore/media/artifacts.py
"""Persistence for produced media assets.

Every media aggregator returns **short-lived URLs**.  A caller that stores the
URI instead of the bytes gets a dead link hours later, which is why
:class:`~llmcore.media.models.MediaArtifact` carries ``expires_at`` and
``checksum_sha256`` and why this store exists.

:class:`ArtifactStore` is content-addressed: bytes land at a path derived from
their SHA-256, so re-materializing the same asset is free and two providers
returning identical output cost one file.  The fetch itself is injected rather
than hard-wired, so the store has no opinion about HTTP clients and no hard
dependency on ``httpx``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..exceptions import MediaError
from .models import MediaArtifact

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_ARTIFACT_PATH", "ArtifactStore", "MaterializePolicy"]

DEFAULT_ARTIFACT_PATH = "~/.llmcore/media"

#: Signature of a byte fetcher: URL in, bytes out.
Fetcher = Callable[[str], Awaitable[bytes]]


class MaterializePolicy(StrEnum):
    """When to fetch and persist remote artifact bytes.

    Attributes:
        ALWAYS: Fetch every remote artifact.
        ON_EXPIRY: Fetch only artifacts whose URI carries an expiry — the
            default, because it protects against dead links without
            downloading assets the caller may never read.
        NEVER: Never fetch; callers handle URIs themselves.
    """

    ALWAYS = "always"
    ON_EXPIRY = "on_expiry"
    NEVER = "never"


@dataclass(frozen=True)
class GcReport:
    """What a :meth:`ArtifactStore.gc` pass did, or would have done."""

    removed: int
    freed_bytes: int
    kept: int
    dry_run: bool = False
    #: ``"<checksum>:<reason>"`` per removed file, for auditing a surprise.
    details: list[str] = field(default_factory=list)


class ArtifactStore:
    """Content-addressed local store for media artifacts.

    Args:
        base_path: Root directory; ``~`` is expanded. Created lazily on first
            write, so constructing a store never touches the filesystem.
        policy: When :meth:`materialize` should fetch remote bytes.
        fetcher: Async callable that downloads a URL. Required only for
            materialization of remote artifacts.
    """

    def __init__(
        self,
        base_path: str | Path = DEFAULT_ARTIFACT_PATH,
        *,
        policy: MaterializePolicy | str = MaterializePolicy.ON_EXPIRY,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.base_path = Path(base_path).expanduser()
        self.policy = MaterializePolicy(policy)
        self._fetcher = fetcher

    # --- addressing ---

    def path_for(self, checksum: str, *, suffix: str = "") -> Path:
        """Return the on-disk path for a content hash.

        Sharded two levels by hash prefix so a large store stays navigable.

        Args:
            checksum: Hex SHA-256 digest.
            suffix: Optional file extension, including the dot.
        """
        return self.base_path / checksum[:2] / checksum[2:4] / f"{checksum}{suffix}"

    def has(self, checksum: str, *, suffix: str = "") -> bool:
        """Whether the content hash is already stored."""
        return self.path_for(checksum, suffix=suffix).exists()

    # --- writing ---

    def put(self, data: bytes, *, suffix: str = "") -> tuple[str, Path]:
        """Store *data* and return its ``(checksum, path)``.

        Idempotent: storing identical bytes twice writes once.
        """
        checksum = hashlib.sha256(data).hexdigest()
        path = self.path_for(checksum, suffix=suffix)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic publish, so a crash never leaves a partial file
            logger.debug("Stored media artifact %s (%d bytes)", checksum[:12], len(data))
        return checksum, path

    def should_materialize(self, artifact: MediaArtifact) -> bool:
        """Whether *artifact* should be fetched under the active policy."""
        if artifact.data is not None or not artifact.uri:
            return False
        if self.policy is MaterializePolicy.NEVER:
            return False
        if self.policy is MaterializePolicy.ALWAYS:
            return True
        return artifact.expires_at is not None

    async def materialize(
        self,
        artifact: MediaArtifact,
        *,
        fetcher: Fetcher | None = None,
        force: bool = False,
    ) -> MediaArtifact:
        """Fetch and persist *artifact*'s bytes when the policy calls for it.

        Args:
            artifact: The artifact to materialize.
            fetcher: Overrides the store's fetcher for this call.
            force: Materialize regardless of policy (but never re-fetch bytes
                the artifact already carries).

        Returns:
            An artifact carrying ``data``, ``checksum_sha256`` and a local
            ``file://`` URI — or the original, unchanged, when the policy says
            not to fetch.

        Raises:
            MediaError: If materialization is required but no fetcher is
                available, or the download fails.
        """
        if artifact.data is not None:
            return artifact
        if not (force or self.should_materialize(artifact)):
            return artifact

        # An artifact already materialized has had its uri re-pointed at the
        # local copy. If that copy is gone -- evicted by `gc`, or deleted by
        # hand -- fetching the uri would fail against a dead path. Fall back
        # to the provider URL that materialization preserved, which is what
        # makes this store a cache rather than a store of record: a file can
        # always be re-fetched, so losing one is recoverable.
        artifact = self._recover_source(artifact)

        fetch = fetcher or self._fetcher
        if fetch is None:
            raise MediaError(
                "Artifact materialization requires a fetcher; none was configured. "
                "Install httpx or pass fetcher= explicitly."
            )
        if not artifact.uri:
            raise MediaError("Artifact has no uri to materialize.")

        try:
            data = await fetch(artifact.uri)
        except MediaError:
            raise
        except Exception as e:
            raise MediaError(f"Failed to materialize artifact from {artifact.uri}: {e}") from e

        suffix = _suffix_for(artifact)
        checksum, path = self.put(data, suffix=suffix)
        stored = artifact.with_data(data)
        # Re-point the URI at the local copy; the provider URL is preserved in
        # provider_metadata so provenance is not lost.
        return MediaArtifact(
            **{
                **{f: getattr(stored, f) for f in stored.__slots__},
                "uri": path.as_uri(),
                "checksum_sha256": checksum,
                "expires_at": None,
                "provider_metadata": {
                    **dict(stored.provider_metadata),
                    "source_uri": artifact.uri,
                },
            }
        )

    # --- reading ---

    async def download(
        self,
        artifact: MediaArtifact,
        destination: str | Path,
        *,
        fetcher: Fetcher | None = None,
    ) -> Path:
        """Write *artifact* to *destination* and return the path.

        Uses inline bytes when present, the local store when already
        materialized, and the fetcher otherwise.
        """
        dest = Path(destination).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if artifact.data is not None:
            dest.write_bytes(artifact.data)
            return dest

        materialized = await self.materialize(artifact, fetcher=fetcher, force=True)
        if materialized.data is None:  # pragma: no cover - defensive
            raise MediaError("Materialization produced no bytes.")
        dest.write_bytes(materialized.data)
        return dest

    @staticmethod
    def _recover_source(artifact: MediaArtifact) -> MediaArtifact:
        """Point a stale local artifact back at the URL it came from.

        Materialization re-points ``uri`` at the local copy and records the
        original under ``provider_metadata["source_uri"]``. When the local
        copy no longer exists, that recorded URL is the only way back, so it
        is restored here. An artifact whose local file is intact, or which
        never had a source URL, is returned unchanged.
        """
        uri = artifact.uri or ""
        if not uri.startswith("file://"):
            return artifact
        local = Path(unquote(urlparse(uri).path))
        if local.exists():
            return artifact
        source = dict(artifact.provider_metadata).get("source_uri")
        if not source:
            return artifact
        logger.debug(
            "Local artifact %s is gone; re-fetching from %s", local, source
        )
        return MediaArtifact(
            **{
                **{f: getattr(artifact, f) for f in artifact.__slots__},
                "uri": str(source),
            }
        )

    def live_checksums(self, artifacts: Iterable[MediaArtifact]) -> set[str]:
        """Digests referenced by *artifacts*, for use as a keep-set.

        Exists because :meth:`gc` takes a keep-set and nothing computed one,
        which is why it had no callers: there was a collector but no way to
        say what was still in use.
        """
        live: set[str] = set()
        for artifact in artifacts:
            digest = getattr(artifact, "checksum_sha256", None)
            if digest:
                live.add(str(digest))
        return live

    def entries(self) -> list[tuple[str, Path, int, float]]:
        """Every stored file as ``(checksum, path, size_bytes, mtime)``."""
        if not self.base_path.exists():
            return []
        found = []
        for path in sorted(self.base_path.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - raced deletion
                continue
            found.append((path.stem, path, stat.st_size, stat.st_mtime))
        return found

    def total_bytes(self) -> int:
        """Size of the store on disk."""
        return sum(size for _c, _p, size, _m in self.entries())

    def gc(
        self,
        *,
        keep_checksums: set[str] | None = None,
        max_age_days: float | None = None,
        max_total_bytes: int | None = None,
        dry_run: bool = False,
    ) -> GcReport:
        """Evict stored files under one or more policies.

        Three policies, which compose. A file is removed if **any** of them
        selects it, and a file named in ``keep_checksums`` is never removed
        by the others -- an explicit keep always wins, so a caller who knows
        what is live cannot be overruled by an age rule.

        Args:
            keep_checksums: Digests to retain. Supplying this *also* enables
                reference-based collection: anything not named is evicted.
                Passing ``None`` leaves reference-based collection off, so a
                mistaken call with no arguments cannot wipe the store.
            max_age_days: Evict files last modified longer ago than this.
            max_total_bytes: Evict oldest-first until the store fits.
            dry_run: Report what would be removed and remove nothing.

        Returns:
            A :class:`GcReport`. Eviction is safe because the store is a
            cache: :meth:`materialize` re-fetches from the preserved source
            URL when a local copy is missing.
        """
        entries = self.entries()
        keep = set(keep_checksums or ())
        doomed: dict[Path, str] = {}

        if keep_checksums is not None:
            for checksum, path, _size, _mtime in entries:
                if checksum not in keep:
                    doomed[path] = "unreferenced"

        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            for checksum, path, _size, mtime in entries:
                if checksum not in keep and mtime < cutoff:
                    doomed.setdefault(path, "stale")

        if max_total_bytes is not None:
            # Oldest first, and only as far as needed to fit.
            survivors = sorted(
                ((c, p, s, m) for c, p, s, m in entries if p not in doomed),
                key=lambda e: e[3],
            )
            total = sum(s for _c, _p, s, _m in survivors)
            for checksum, path, size, _mtime in survivors:
                if total <= max_total_bytes:
                    break
                if checksum in keep:
                    continue
                doomed[path] = "over_budget"
                total -= size

        freed = 0
        removed: list[str] = []
        for path, reason in doomed.items():
            try:
                size = path.stat().st_size
            except OSError:  # pragma: no cover - raced deletion
                continue
            if not dry_run:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover
                    continue
            freed += size
            removed.append(f"{path.stem}:{reason}")

        return GcReport(
            removed=len(removed),
            freed_bytes=freed,
            kept=len(entries) - len(removed),
            dry_run=dry_run,
            details=sorted(removed),
        )


def _suffix_for(artifact: MediaArtifact) -> str:
    """Best-effort file extension for an artifact."""
    import mimetypes

    if artifact.mime_type:
        ext = mimetypes.guess_extension(artifact.mime_type)
        if ext:
            return ext
    if artifact.uri:
        tail = Path(artifact.uri.split("?", 1)[0]).suffix
        if tail:
            return tail
    return ""


def default_fetcher(timeout: float = 120.0) -> Fetcher:
    """Return an ``httpx``-backed fetcher, if httpx is installed.

    Args:
        timeout: Per-request timeout in seconds.

    Returns:
        An async fetcher suitable for :class:`ArtifactStore`.

    Raises:
        MediaError: If ``httpx`` is not available.
    """
    try:
        import httpx
    except ImportError as e:  # pragma: no cover - httpx is present in practice
        raise MediaError(
            "The 'httpx' package is required to download media artifacts."
        ) from e

    async def _fetch(url: str) -> bytes:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    return _fetch
