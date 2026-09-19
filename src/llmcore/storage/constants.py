# src/llmcore/storage/constants.py
"""
Shared constants and helpers for llmcore storage backends.

Currently hosts the SQLite-style ':memory:' path token and its detection
helper, so file-based backends never mistake the token for a real
filesystem path (which used to create a literal ':memory:/' directory
in the process CWD).
"""

from __future__ import annotations

#: SQLite-style token meaning "do not persist to disk".
#: File-based storage backends must never treat it as a filesystem path.
IN_MEMORY_PATH_TOKEN = ":memory:"


def is_in_memory_path(path: object) -> bool:
    """Return ``True`` if ``path`` is the special ``':memory:'`` token.

    The match is exact (mirroring SQLite's own semantics); any other value —
    including ``None`` or non-string objects — returns ``False``.
    """
    return isinstance(path, str) and path == IN_MEMORY_PATH_TOKEN
