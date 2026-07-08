# src/llmcore/agents/prompts/usage_store.py
"""Persistent prompt-usage telemetry (optional, [grimoire.metrics]).

The ``GrimoirePromptRegistryAdapter`` keeps in-memory ``PromptMetrics`` for
cognitive-phase compatibility; this module adds the OPTIONAL durable sink so
prompt A/B work and spell optimization can span processes.

Storage lives llmcore-side (not in grimoire) by design: grimoire layers may
be read-only packaged data, and usage is host telemetry, not prompt content.
The adapter's ``version_id`` (``grimoire:<template>:<spell>:<version>:<hash>``)
already carries everything needed to join usage back to spell versions later.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Default JSONL location (XDG state dir, overridable via config).
DEFAULT_USAGE_PATH = "~/.local/state/llmcore/prompt_usage.jsonl"


@runtime_checkable
class PromptUsageStore(Protocol):
    """Anything with a ``record(**fields)`` method can be a usage store."""

    def record(self, **fields: Any) -> None:  # pragma: no cover - protocol
        ...


class JsonlPromptUsageStore:
    """Append-only JSONL usage store (one record per ``record_use``).

    Writes are line-atomic (single ``write`` of one line under a lock) and
    best-effort: any I/O failure is logged at DEBUG and swallowed — telemetry
    must never break a phase.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        raw = str(path) if path else os.environ.get(
            "LLMCORE_PROMPT_USAGE_PATH", DEFAULT_USAGE_PATH
        )
        self._path = Path(raw).expanduser()
        self._lock = threading.Lock()
        self._dir_ready = False

    @property
    def path(self) -> Path:
        """The JSONL file path."""
        return self._path

    def record(self, **fields: Any) -> None:
        """Append one usage record with a UTC timestamp."""
        record = {"ts": datetime.now(UTC).isoformat(), **fields}
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                if not self._dir_ready:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._dir_ready = True
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # never break the caller on telemetry
            logger.debug("Prompt usage write failed (%s): %s", self._path, exc)


__all__ = ["DEFAULT_USAGE_PATH", "JsonlPromptUsageStore", "PromptUsageStore"]
