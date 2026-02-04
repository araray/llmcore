# src/llmcore/storage/sqlite_session.py
"""
SQLite database storage for ChatSession objects and related data.

This module implements the BaseSessionStorage interface using the aiosqlite
library. It acts as an orchestrator, managing the database connection and
delegating specific data domain operations (Context Presets, Episodes) to
specialized helper modules.

REFACTORED: This class has been streamlined to focus on ChatSession management.
The logic for Context Presets and Episodes has been moved to
`sqlite_preset_helpers.py` and `sqlite_episode_helpers.py` respectively.
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone, UTC
from typing import Any, Dict, List, Optional

try:
    import aiosqlite

    aiosqlite_available = True
except ImportError:
    aiosqlite_available = False
    aiosqlite = None

from ..exceptions import ConfigError, SessionStorageError, StorageError
from ..models import (
    ChatSession,
    ContextItem,
    ContextItemType,
    ContextPreset,
    Episode,
    Message,
    Role,
)
from . import sqlite_episode_helpers, sqlite_preset_helpers
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)

# Default table names
DEFAULT_SESSIONS_TABLE = "sessions"
DEFAULT_MESSAGES_TABLE = "messages"
DEFAULT_SESSION_CONTEXT_ITEMS_TABLE = "session_context_items"
DEFAULT_CONTEXT_PRESETS_TABLE = "context_presets"
DEFAULT_CONTEXT_PRESET_ITEMS_TABLE = "context_preset_items"
DEFAULT_EPISODES_TABLE = "episodes"


class SqliteSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession, ContextPreset, and Episode objects in a
    SQLite database, delegating preset and episode logic to helper modules.
    """

    _db_path: pathlib.Path
    _conn: Optional["aiosqlite.Connection"] = None
    _sessions_table_name: str
    _messages_table_name: str
    _session_context_items_table_name: str
    _context_presets_table_name: str
    _context_preset_items_table_name: str
    _episodes_table_name: str

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the SQLite database and create all necessary tables.

        Args:
            config: Configuration dictionary with database path and table names.

        Raises:
            ConfigError: If 'path' is not provided or aiosqlite is not installed.
            SessionStorageError: If the database cannot be initialized.
        """
        if not aiosqlite_available:
            raise ConfigError(
                "aiosqlite library is not installed. Please install `llmcore[sqlite]`."
            )

        db_path_str = config.get("path")
        if not db_path_str:
            raise ConfigError("SQLite session storage 'path' not specified in configuration.")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))
        self._sessions_table_name = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table_name = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        self._session_context_items_table_name = config.get(
            "session_context_items_table_name", DEFAULT_SESSION_CONTEXT_ITEMS_TABLE
        )
        self._context_presets_table_name = config.get(
            "context_presets_table_name", DEFAULT_CONTEXT_PRESETS_TABLE
        )
        self._context_preset_items_table_name = config.get(
            "context_preset_items_table_name", DEFAULT_CONTEXT_PRESET_ITEMS_TABLE
        )
        self._episodes_table_name = config.get("episodes_table_name", DEFAULT_EPISODES_TABLE)

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA foreign_keys = ON;")
            await self._create_tables_if_not_exist()
            await self._conn.commit()
            logger.info(f"SQLite storage initialized at: {self._db_path.resolve()}")
        except aiosqlite.Error as e:
            raise SessionStorageError(f"Could not initialize SQLite database: {e}")

    async def _create_tables_if_not_exist(self) -> None:
        """Creates all required tables in the SQLite database."""
        # Sessions table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._sessions_table_name} (
                id TEXT PRIMARY KEY, name TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, metadata TEXT
            )
        """)
        # Messages table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._messages_table_name} (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, timestamp TEXT NOT NULL, tokens INTEGER, metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
            )
        """)
        await self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON {self._messages_table_name} (session_id, timestamp);"
        )

        # Session ContextItems table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._session_context_items_table_name} (
                id TEXT NOT NULL, session_id TEXT NOT NULL, item_type TEXT NOT NULL,
                source_id TEXT, content TEXT NOT NULL, tokens INTEGER, metadata TEXT,
                timestamp TEXT NOT NULL, is_truncated INTEGER DEFAULT 0, original_tokens INTEGER,
                PRIMARY KEY (session_id, id),
                FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
            )
        """)
        # ContextPresets table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._context_presets_table_name} (
                name TEXT PRIMARY KEY, description TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, metadata TEXT
            )
        """)
        # ContextPresetItems table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._context_preset_items_table_name} (
                item_id TEXT NOT NULL, preset_name TEXT NOT NULL, type TEXT NOT NULL,
                content TEXT, source_identifier TEXT, metadata TEXT,
                PRIMARY KEY (preset_name, item_id),
                FOREIGN KEY (preset_name) REFERENCES {self._context_presets_table_name}(name) ON DELETE CASCADE
            )
        """)
        # Episodes table
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._episodes_table_name} (
                episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL, event_type TEXT NOT NULL, data TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
            )
        """)
        await self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_episodes_session_timestamp ON {self._episodes_table_name} (session_id, timestamp);"
        )

    # --- Session Management ---

    async def save_session(self, session: ChatSession) -> None:
        """Saves or updates a session, its messages, and context items."""
        if not self._conn:
            raise SessionStorageError("Database connection not initialized.")
        try:
            await self._conn.execute("BEGIN;")
            await self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._sessions_table_name} (id, name, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    session.id,
                    session.name,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    json.dumps(session.metadata or {}),
                ),
            )

            await self._conn.execute(
                f"DELETE FROM {self._messages_table_name} WHERE session_id = ?", (session.id,)
            )
            if session.messages:
                messages_data = [
                    (
                        msg.id,
                        session.id,
                        str(msg.role),
                        msg.content,
                        msg.timestamp.isoformat(),
                        msg.tokens,
                        json.dumps(msg.metadata or {}),
                    )
                    for msg in session.messages
                ]
                await self._conn.executemany(
                    f"INSERT INTO {self._messages_table_name} (id, session_id, role, content, timestamp, tokens, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    messages_data,
                )

            await self._conn.execute(
                f"DELETE FROM {self._session_context_items_table_name} WHERE session_id = ?",
                (session.id,),
            )
            if session.context_items:
                context_items_data = [
                    (
                        item.id,
                        session.id,
                        str(item.type),
                        item.source_id,
                        item.content,
                        item.tokens,
                        json.dumps(item.metadata or {}),
                        item.timestamp.isoformat(),
                        1 if item.is_truncated else 0,
                        item.original_tokens,
                    )
                    for item in session.context_items
                ]
                await self._conn.executemany(
                    f"INSERT INTO {self._session_context_items_table_name} (id, session_id, item_type, source_id, content, tokens, metadata, timestamp, is_truncated, original_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    context_items_data,
                )
            await self._conn.commit()
        except aiosqlite.Error as e:
            await self._conn.rollback()
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")

    async def get_session(self, session_id: str) -> ChatSession | None:
        """Retrieves a complete session from SQLite."""
        if not self._conn:
            raise SessionStorageError("Database connection not initialized.")
        try:
            async with self._conn.execute(
                f"SELECT * FROM {self._sessions_table_name} WHERE id = ?", (session_id,)
            ) as cursor:
                session_row = await cursor.fetchone()
            if not session_row:
                return None

            session_data = dict(session_row)
            session_data["metadata"] = json.loads(session_data.get("metadata") or "{}")
            session_data["created_at"] = datetime.fromisoformat(
                session_data["created_at"].replace("Z", "+00:00")
            )
            session_data["updated_at"] = datetime.fromisoformat(
                session_data["updated_at"].replace("Z", "+00:00")
            )

            messages: list[Message] = []
            async with self._conn.execute(
                f"SELECT * FROM {self._messages_table_name} WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ) as cursor:
                async for row in cursor:
                    msg_dict = dict(row)
                    msg_dict["metadata"] = json.loads(msg_dict.get("metadata") or "{}")
                    msg_dict["role"] = Role(msg_dict["role"])
                    msg_dict["timestamp"] = datetime.fromisoformat(
                        msg_dict["timestamp"].replace("Z", "+00:00")
                    )
                    messages.append(Message.model_validate(msg_dict))
            session_data["messages"] = messages

            context_items: list[ContextItem] = []
            async with self._conn.execute(
                f"SELECT * FROM {self._session_context_items_table_name} WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ) as cursor:
                async for row in cursor:
                    item_dict = dict(row)
                    item_dict["metadata"] = json.loads(item_dict.get("metadata") or "{}")
                    item_dict["type"] = ContextItemType(item_dict.pop("item_type"))
                    item_dict["timestamp"] = datetime.fromisoformat(
                        item_dict["timestamp"].replace("Z", "+00:00")
                    )
                    item_dict["is_truncated"] = bool(item_dict.get("is_truncated", 0))
                    context_items.append(ContextItem.model_validate(item_dict))
            session_data["context_items"] = context_items

            return ChatSession.model_validate(session_data)
        except aiosqlite.Error as e:
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Lists session metadata from SQLite."""
        if not self._conn:
            raise SessionStorageError("Database connection not initialized.")
        try:
            async with self._conn.execute(f"""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata,
                       (SELECT COUNT(*) FROM {self._messages_table_name} m WHERE m.session_id = s.id) as message_count,
                       (SELECT COUNT(*) FROM {self._session_context_items_table_name} ci WHERE ci.session_id = s.id) as context_item_count
                FROM {self._sessions_table_name} s ORDER BY s.updated_at DESC
            """) as cursor:
                rows = await cursor.fetchall()

            return [dict(row) for row in rows]
        except aiosqlite.Error as e:
            raise SessionStorageError(f"Database error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session and its associated data from SQLite."""
        if not self._conn:
            raise SessionStorageError("Database connection not initialized.")
        try:
            cursor = await self._conn.execute(
                f"DELETE FROM {self._sessions_table_name} WHERE id = ?", (session_id,)
            )
            await self._conn.commit()
            return cursor.rowcount > 0
        except aiosqlite.Error as e:
            await self._conn.rollback()
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")

    async def update_session_name(self, session_id: str, new_name: str) -> bool:
        """Updates the name for a specific session in SQLite."""
        if not self._conn:
            raise SessionStorageError("Database connection not initialized.")
        try:
            cursor = await self._conn.execute(
                f"UPDATE {self._sessions_table_name} SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, datetime.now(UTC).isoformat(), session_id),
            )
            await self._conn.commit()
            return cursor.rowcount > 0
        except aiosqlite.Error as e:
            await self._conn.rollback()
            raise SessionStorageError(
                f"Database error updating session name for '{session_id}': {e}"
            )

    # --- Context Preset Management (Delegated) ---

    async def save_context_preset(self, preset: ContextPreset) -> None:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        await sqlite_preset_helpers.save_context_preset(
            self._conn,
            preset,
            self._context_presets_table_name,
            self._context_preset_items_table_name,
        )

    async def get_context_preset(self, preset_name: str) -> ContextPreset | None:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        return await sqlite_preset_helpers.get_context_preset(
            self._conn,
            preset_name,
            self._context_presets_table_name,
            self._context_preset_items_table_name,
        )

    async def list_context_presets(self) -> list[dict[str, Any]]:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        return await sqlite_preset_helpers.list_context_presets(
            self._conn, self._context_presets_table_name, self._context_preset_items_table_name
        )

    async def delete_context_preset(self, preset_name: str) -> bool:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        return await sqlite_preset_helpers.delete_context_preset(
            self._conn, preset_name, self._context_presets_table_name
        )

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        return await sqlite_preset_helpers.rename_context_preset(
            self._conn,
            old_name,
            new_name,
            self._context_presets_table_name,
            self._context_preset_items_table_name,
        )

    # --- Episodic Memory Management (Delegated) ---

    async def add_episode(self, episode: Episode) -> None:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        await sqlite_episode_helpers.add_episode(self._conn, episode, self._episodes_table_name)

    async def get_episodes(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]:
        if not self._conn:
            raise StorageError("Database connection not initialized.")
        return await sqlite_episode_helpers.get_episodes(
            self._conn, session_id, self._episodes_table_name, limit, offset
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("aiosqlite storage connection closed.")
