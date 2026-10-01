# src/llmcore/runtimes/state.py
"""On-disk runtime state (spec phase R1).

Safety rule 2 of the spec: *a runtime that llmcore started is recorded in
inspectable state so it can always be found and killed — a leaked VM is a
leaked credit card.*

That makes this store a safety mechanism, not a cache, and it drives three
choices:

* **Inspectable**, one indented-JSON file per runtime under a directory a human
  can list and read. Someone who suspects they are being billed must be able to
  find out with ``ls`` and ``cat``, without llmcore running and without a tool
  that understands a binary format.
* **Written before the work, not after.** A record is persisted when
  provisioning *begins*, because the dangerous window is a crash between
  assignment and bookkeeping — the case where money is burning and nothing
  knows about it.
* **Readable even when damaged.** A malformed file is reported and skipped
  rather than raising, because one bad record must not hide the other runtimes
  still running.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .models import RuntimeHandle

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_STATE_DIR", "RuntimeStateStore"]

#: Where runtime records live unless configured otherwise.
DEFAULT_STATE_DIR = "~/.llmcore/runtimes"


class RuntimeStateStore:
    """Persists :class:`RuntimeHandle` records as one JSON file each.

    Args:
        state_dir: Directory for records. ``~`` is expanded. Created on demand.
    """

    def __init__(self, state_dir: str | Path = DEFAULT_STATE_DIR) -> None:
        self._dir = Path(str(state_dir)).expanduser()

    @property
    def directory(self) -> Path:
        """The directory records are written to."""
        return self._dir

    def path_for(self, name: str) -> Path:
        """Return the record path for runtime *name*."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return self._dir / f"{safe}.json"

    def save(self, handle: RuntimeHandle) -> Path:
        """Write *handle* to disk atomically and return its path.

        Atomic because a torn write is worse than no write: a half-written
        record can make a live, billing runtime unreadable, which is exactly the
        situation this store exists to prevent.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(handle.name)
        handle.state_path = target

        fd, tmp = tempfile.mkstemp(dir=str(self._dir), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(handle.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.debug("Persisted runtime state for '%s' at %s", handle.name, target)
        return target

    def load(self, name: str) -> RuntimeHandle | None:
        """Return the record for *name*, or ``None`` if there is none."""
        path = self.path_for(name)
        if not path.is_file():
            return None
        return self._read(path)

    def load_all(self) -> list[RuntimeHandle]:
        """Return every readable record, newest first.

        Unreadable records are logged and skipped — one corrupt file must not
        hide the runtimes that are still running and still billing.
        """
        if not self._dir.is_dir():
            return []
        handles: list[RuntimeHandle] = []
        for path in sorted(self._dir.glob("*.json")):
            handle = self._read(path)
            if handle is not None:
                handles.append(handle)
        handles.sort(key=lambda h: h.started_at, reverse=True)
        return handles

    def delete(self, name: str) -> bool:
        """Remove the record for *name*. Returns whether one existed."""
        path = self.path_for(name)
        if not path.is_file():
            return False
        path.unlink(missing_ok=True)
        logger.debug("Removed runtime state for '%s'", name)
        return True

    def _read(self, path: Path) -> RuntimeHandle | None:
        """Parse one record, tolerating damage."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(
                "Unreadable runtime state file %s (%s). It is being skipped, but if a "
                "runtime is still running it will not appear in status output — "
                "check the backend directly.",
                path,
                e,
            )
            return None
        if not isinstance(payload, dict):
            logger.warning("Runtime state file %s is not an object; skipping.", path)
            return None
        return RuntimeHandle.from_dict(payload, state_path=path)
