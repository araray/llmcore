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
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path

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

    def gc(self, *, keep_checksums: set[str] | None = None) -> int:
        """Delete stored files not in *keep_checksums*.

        Args:
            keep_checksums: Digests to retain; everything else is removed. When
                ``None``, nothing is deleted (a no-op, so a mistaken call cannot
                wipe the store).

        Returns:
            Number of files removed.
        """
        if keep_checksums is None:
            return 0
        removed = 0
        if not self.base_path.exists():
            return 0
        for path in self.base_path.rglob("*"):
            if path.is_file() and path.stem not in keep_checksums:
                path.unlink()
                removed += 1
        return removed


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
