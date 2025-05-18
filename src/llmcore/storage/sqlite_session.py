# src/llmcore/storage/sqlite_session.py
"""
SQLite database storage for ChatSession objects using aiosqlite.

This module implements the BaseSessionStorage interface using the
aiosqlite library for native asynchronous database operations.
It now includes support for storing and retrieving `ContextItem` objects
associated with each session in a separate table.
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

try:
    import aiosqlite
    aiosqlite_available = True
except ImportError:
    aiosqlite_available = False
    aiosqlite = None # type: ignore

from ..models import ChatSession, Message, Role, ContextItem, ContextItemType # Added ContextItem, ContextItemType
from ..exceptions import SessionStorageError, ConfigError
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_ITEMS_TABLE = "context_items"

class SqliteSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a SQLite database using aiosqlite.

    Stores session, message, and context_item data in tables within a single DB file.
    """
    _db_path: pathlib.Path
    _conn: Optional[aiosqlite.Connection] = None
    _sessions_table_name: str = "sessions" # Default, can be overridden by config in future if needed
    _messages_table_name: str = "messages" # Default
    _context_items_table_name: str # Will be set from config or default

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SQLite database asynchronously using aiosqlite.
        Creates tables for sessions, messages, and context_items if they don't exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'path': The directory path for storing the database file.
                    'context_items_table_name' (optional): Name for the context_items table.

        Raises:
            ConfigError: If 'path' is not provided or aiosqlite is not installed.
            SessionStorageError: If the database cannot be initialized or tables created.
        """
        if not aiosqlite_available:
            raise ConfigError("aiosqlite library is not installed. Please install `aiosqlite` or `llmcore[sqlite]`.")

        db_path_str = config.get("path")
        if not db_path_str:
            raise ConfigError("SQLite session storage 'path' not specified in configuration.")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))
        self._context_items_table_name = config.get("context_items_table_name", DEFAULT_CONTEXT_ITEMS_TABLE)


        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA foreign_keys = ON;")

            # Sessions table
            await self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._sessions_table_name} (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT -- Store as JSON text
                )
            """)
            # Messages table
            await self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._messages_table_name} (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens INTEGER,
                    metadata TEXT, -- Store as JSON text
                    FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
                )
            """)
            await self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON {self._messages_table_name} (session_id, timestamp);
            """)

            # ContextItems table
            await self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._context_items_table_name} (
                    id TEXT NOT NULL, -- ID of the context item itself
                    session_id TEXT NOT NULL,
                    item_type TEXT NOT NULL, -- Stores ContextItemType enum value (e.g., "user_text")
                    source_id TEXT,
                    content TEXT NOT NULL,
                    tokens INTEGER,
                    metadata TEXT, -- Store as JSON text
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (session_id, id), -- Item ID should be unique within a session
                    FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
                )
            """)
            await self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_context_items_session_id ON {self._context_items_table_name} (session_id);
            """)

            await self._conn.commit()
            logger.info(f"SQLite session storage initialized at: {self._db_path.resolve()} with tables: "
                        f"{self._sessions_table_name}, {self._messages_table_name}, {self._context_items_table_name}")

        except aiosqlite.Error as e:
            logger.error(f"Failed to initialize aiosqlite database at {self._db_path}: {e}")
            if self._conn: await self._conn.close()
            self._conn = None
            raise SessionStorageError(f"Could not initialize SQLite database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during SQLite initialization: {e}", exc_info=True)
            if self._conn: await self._conn.close()
            self._conn = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session, its messages, and its context_items in the database.
        Uses UPSERT for session metadata and replaces messages/context_items for simplicity.

        Args:
            session: The ChatSession object to save.
        Raises:
            SessionStorageError: If DB connection isn't initialized or saving fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized. Call initialize() first.")

        session_metadata_json = json.dumps(session.metadata or {})

        try:
            await self._conn.execute("BEGIN;") # Start transaction

            # UPSERT session metadata
            await self._conn.execute(f"""
                INSERT OR REPLACE INTO {self._sessions_table_name} (id, name, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.id, session.name, session.created_at.isoformat(),
                session.updated_at.isoformat(), session_metadata_json
            ))

            # Replace messages
            await self._conn.execute(f"DELETE FROM {self._messages_table_name} WHERE session_id = ?", (session.id,))
            if session.messages:
                messages_data = [(msg.id, session.id, str(msg.role), msg.content,
                                  msg.timestamp.isoformat(), msg.tokens, json.dumps(msg.metadata or {}))
                                 for msg in session.messages]
                await self._conn.executemany(f"""
                    INSERT INTO {self._messages_table_name} (id, session_id, role, content, timestamp, tokens, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, messages_data)

            # Replace context_items
            await self._conn.execute(f"DELETE FROM {self._context_items_table_name} WHERE session_id = ?", (session.id,))
            if session.context_items:
                context_items_data = [(item.id, session.id, str(item.type.value), item.source_id, item.content,
                                       item.tokens, json.dumps(item.metadata or {}), item.timestamp.isoformat())
                                      for item in session.context_items]
                await self._conn.executemany(f"""
                    INSERT INTO {self._context_items_table_name} (id, session_id, item_type, source_id, content, tokens, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, context_items_data)

            await self._conn.commit()
            logger.debug(f"Session '{session.id}' with {len(session.messages)} messages and {len(session.context_items)} context items saved to SQLite.")
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error saving session '{session.id}': {e}")
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a specific chat session, its messages, and its context_items by ID.

        Args:
            session_id: The ID of the session to retrieve.
        Returns:
            The ChatSession object if found, otherwise None.
        Raises:
            SessionStorageError: If DB connection isn't initialized or retrieval fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        try:
            async with self._conn.execute(f"SELECT * FROM {self._sessions_table_name} WHERE id = ?", (session_id,)) as cursor:
                session_row = await cursor.fetchone()
            if not session_row:
                logger.debug(f"Session '{session_id}' not found in SQLite.")
                return None

            session_data = dict(session_row)
            session_data["metadata"] = json.loads(session_data["metadata"] or '{}')

            # Fetch messages
            messages: List[Message] = []
            async with self._conn.execute(f"""
                SELECT id, session_id, role, content, timestamp, tokens, metadata
                FROM {self._messages_table_name} WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,)) as cursor:
                async for msg_row_data in cursor:
                    msg_dict = dict(msg_row_data)
                    try:
                        msg_dict["metadata"] = json.loads(msg_dict["metadata"] or '{}')
                        msg_dict["role"] = Role(msg_dict["role"])
                        ts_str = msg_dict["timestamp"]
                        msg_dict["timestamp"] = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                        messages.append(Message.model_validate(msg_dict))
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid message data for session {session_id}, msg_id {msg_dict.get('id')}: {e}")
            session_data["messages"] = messages

            # Fetch context_items
            context_items: List[ContextItem] = []
            async with self._conn.execute(f"""
                SELECT id, session_id, item_type, source_id, content, tokens, metadata, timestamp
                FROM {self._context_items_table_name} WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,)) as cursor:
                async for item_row_data in cursor:
                    item_dict = dict(item_row_data)
                    try:
                        item_dict["metadata"] = json.loads(item_dict["metadata"] or '{}')
                        item_dict["type"] = ContextItemType(item_dict.pop("item_type")) # Rename column to match model
                        ts_str = item_dict["timestamp"]
                        item_dict["timestamp"] = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                        context_items.append(ContextItem.model_validate(item_dict))
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid context_item data for session {session_id}, item_id {item_dict.get('id')}: {e}")
            session_data["context_items"] = context_items

            # Validate and create ChatSession object
            chat_session = ChatSession.model_validate(session_data)
            logger.debug(f"Session '{session_id}' loaded from SQLite with {len(messages)} messages and {len(context_items)} context items.")
            return chat_session

        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error retrieving session '{session_id}': {e}")
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions, returning metadata only.
        Includes count of messages and context_items.

        Returns:
            A list of dictionaries, each representing session metadata.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        session_metadata_list: List[Dict[str, Any]] = []
        try:
            async with self._conn.execute(f"""
                SELECT
                    s.id, s.name, s.created_at, s.updated_at, s.metadata,
                    (SELECT COUNT(*) FROM {self._messages_table_name} m WHERE m.session_id = s.id) as message_count,
                    (SELECT COUNT(*) FROM {self._context_items_table_name} ci WHERE ci.session_id = s.id) as context_item_count
                FROM {self._sessions_table_name} s
                ORDER BY s.updated_at DESC
            """) as cursor:
                async for row in cursor:
                    data = dict(row)
                    try:
                        data["metadata"] = json.loads(data["metadata"] or '{}')
                    except json.JSONDecodeError:
                        data["metadata"] = {}
                    session_metadata_list.append(data)
            logger.debug(f"Found {len(session_metadata_list)} sessions in SQLite.")
            return session_metadata_list
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error listing sessions: {e}")
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """Delete a specific chat session. Cascades to messages and context_items."""
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")
        try:
            cursor = await self._conn.execute(f"DELETE FROM {self._sessions_table_name} WHERE id = ?", (session_id,))
            await self._conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"Session '{session_id}' and its associated messages/context_items deleted from SQLite.")
                return True
            logger.warning(f"Attempted to delete non-existent session '{session_id}' from SQLite.")
            return False
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error deleting session '{session_id}': {e}")
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                await self._conn.close()
                self._conn = None
                logger.info("aiosqlite session storage connection closed.")
            except aiosqlite.Error as e:
                logger.error(f"Error closing aiosqlite connection: {e}")
