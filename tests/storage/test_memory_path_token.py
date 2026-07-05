# tests/storage/test_memory_path_token.py
"""
Regression tests for the ':memory:' path token (SQLite-style "do not persist").

Storage backends configured with path=':memory:' must never treat the token as
a filesystem path. Historically JsonSessionStorage and ChromaVectorStorage
created a literal ':memory:/' directory in the process CWD (containing session
JSON files, chroma.sqlite3, context_presets/ and episodes/).
"""

from __future__ import annotations

import pathlib

import pytest

from llmcore.models import ChatSession, Message, Role
from llmcore.storage.constants import IN_MEMORY_PATH_TOKEN, is_in_memory_path
from llmcore.storage.json_session import JsonSessionStorage

MEMORY_DIR = pathlib.Path(IN_MEMORY_PATH_TOKEN)


def _assert_no_memory_dir(base: pathlib.Path) -> None:
    """Assert that no literal ':memory:' entry was created under ``base``."""
    assert not (base / IN_MEMORY_PATH_TOKEN).exists(), (
        f"A literal '{IN_MEMORY_PATH_TOKEN}' filesystem entry was created in {base}"
    )


# ==============================================================================
# Token helper
# ==============================================================================


def test_is_in_memory_path_matches_exact_token_only():
    assert is_in_memory_path(":memory:")
    assert not is_in_memory_path(" :memory: ")
    assert not is_in_memory_path("memory")
    assert not is_in_memory_path("/data/:memory:")
    assert not is_in_memory_path(None)
    assert not is_in_memory_path(pathlib.Path(":memory:"))


# ==============================================================================
# JsonSessionStorage
# ==============================================================================


async def test_json_session_storage_memory_token_creates_no_memory_dir(tmp_path, monkeypatch):
    """':memory:' must map to an ephemeral temp dir, not a ':memory:/' directory."""
    monkeypatch.chdir(tmp_path)

    storage = JsonSessionStorage()
    await storage.initialize({"path": IN_MEMORY_PATH_TOKEN})
    try:
        _assert_no_memory_dir(tmp_path)

        backing_dir = storage._storage_dir
        assert backing_dir.is_dir()
        assert IN_MEMORY_PATH_TOKEN not in str(backing_dir)
        # The backing dir must live outside the CWD (an ephemeral temp location).
        assert tmp_path not in backing_dir.parents and backing_dir != tmp_path

        # Storage must be fully functional in ':memory:' mode.
        session = ChatSession(
            id="mem-session-1",
            messages=[Message(role=Role.USER, content="hello")],
        )
        await storage.save_session(session)
        loaded = await storage.get_session("mem-session-1")
        assert loaded is not None
        assert loaded.messages[0].content == "hello"

        _assert_no_memory_dir(tmp_path)
    finally:
        await storage.close()

    # close() removes the ephemeral backing directory.
    assert not backing_dir.exists()
    _assert_no_memory_dir(tmp_path)


async def test_json_session_storage_real_path_behavior_unchanged(tmp_path):
    """Real paths still create the directory tree exactly as before."""
    target = tmp_path / "sessions"
    storage = JsonSessionStorage()
    await storage.initialize({"path": str(target)})
    try:
        assert target.is_dir()
        assert (target / "context_presets").is_dir()
        assert (target / "episodes").is_dir()

        session = ChatSession(id="disk-session-1", messages=[])
        await storage.save_session(session)
        assert (target / "disk-session-1.json").is_file()
    finally:
        await storage.close()

    # close() must not delete user-provided directories.
    assert target.is_dir()
    assert (target / "disk-session-1.json").is_file()


# ==============================================================================
# SqliteSessionStorage
# ==============================================================================


async def test_sqlite_session_storage_memory_token_creates_no_memory_dir(tmp_path, monkeypatch):
    """SQLite natively supports ':memory:'; ensure no stray filesystem entry appears."""
    aiosqlite = pytest.importorskip("aiosqlite")
    assert aiosqlite is not None

    from llmcore.storage.sqlite_session import SqliteSessionStorage

    monkeypatch.chdir(tmp_path)

    storage = SqliteSessionStorage()
    await storage.initialize({"path": IN_MEMORY_PATH_TOKEN})
    try:
        _assert_no_memory_dir(tmp_path)

        session = ChatSession(
            id="mem-sqlite-1",
            messages=[Message(role=Role.USER, content="hi")],
        )
        await storage.save_session(session)
        loaded = await storage.get_session("mem-sqlite-1")
        assert loaded is not None
        _assert_no_memory_dir(tmp_path)
    finally:
        await storage.close()

    _assert_no_memory_dir(tmp_path)


# ==============================================================================
# ChromaVectorStorage
# ==============================================================================


async def test_chroma_vector_storage_memory_token_creates_no_memory_dir(tmp_path, monkeypatch):
    """':memory:' must select the in-memory Chroma client, not a persistent one."""
    from llmcore.storage.chromadb_vector import ChromaVectorStorage, chromadb_available

    if not chromadb_available:
        pytest.skip("chromadb not installed")

    monkeypatch.chdir(tmp_path)

    storage = ChromaVectorStorage()
    await storage.initialize({"path": IN_MEMORY_PATH_TOKEN})
    try:
        _assert_no_memory_dir(tmp_path)
        # No persistent artifacts (chroma.sqlite3) may appear in the CWD either.
        assert not (tmp_path / "chroma.sqlite3").exists()
        assert storage._client is not None
    finally:
        await storage.close()

    _assert_no_memory_dir(tmp_path)
