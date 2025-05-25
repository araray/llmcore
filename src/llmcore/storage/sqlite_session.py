# src/llmcore/storage/sqlite_session.py
"""
SQLite database storage for ChatSession objects and ContextPreset objects
using aiosqlite.

This module implements the BaseSessionStorage interface using the
aiosqlite library for native asynchronous database operations.
It now includes support for storing and retrieving ContextPreset objects
in dedicated tables.
"""

import json
import logging
import os
import pathlib
import re # For validating preset names for rename, though Pydantic should handle creation
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
    aiosqlite_available = True
except ImportError:
    aiosqlite_available = False
    aiosqlite = None # type: ignore

from ..exceptions import ConfigError, SessionStorageError, StorageError
from ..models import (ChatSession, ContextItem, ContextItemType, Message, Role,
                      ContextPreset, ContextPresetItem) # Added ContextPreset, ContextPresetItem
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)

# Default table names
DEFAULT_SESSIONS_TABLE = "sessions"
DEFAULT_MESSAGES_TABLE = "messages"
DEFAULT_SESSION_CONTEXT_ITEMS_TABLE = "session_context_items" # Renamed for clarity
DEFAULT_CONTEXT_PRESETS_TABLE = "context_presets"
DEFAULT_CONTEXT_PRESET_ITEMS_TABLE = "context_preset_items"


class SqliteSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession and ContextPreset objects in a SQLite database
    using aiosqlite.

    Stores session, message, session_context_item, context_preset, and
    context_preset_item data in respective tables within a single DB file.
    """
    _db_path: pathlib.Path
    _conn: Optional[aiosqlite.Connection] = None
    _sessions_table_name: str
    _messages_table_name: str
    _session_context_items_table_name: str # Table for ContextItems linked to ChatSessions
    _context_presets_table_name: str # New table for ContextPresets
    _context_preset_items_table_name: str # New table for ContextPresetItems

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SQLite database asynchronously using aiosqlite.
        Creates tables for sessions, messages, session context items,
        context presets, and context preset items if they don't exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'path': The directory path for storing the database file.
                    'sessions_table_name' (optional)
                    'messages_table_name' (optional)
                    'session_context_items_table_name' (optional)
                    'context_presets_table_name' (optional)
                    'context_preset_items_table_name' (optional)

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
        self._sessions_table_name = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table_name = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        self._session_context_items_table_name = config.get("session_context_items_table_name", DEFAULT_SESSION_CONTEXT_ITEMS_TABLE)
        self._context_presets_table_name = config.get("context_presets_table_name", DEFAULT_CONTEXT_PRESETS_TABLE)
        self._context_preset_items_table_name = config.get("context_preset_items_table_name", DEFAULT_CONTEXT_PRESET_ITEMS_TABLE)

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row # type: ignore
            await self._conn.execute("PRAGMA foreign_keys = ON;")

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
            await self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON {self._messages_table_name} (session_id, timestamp);")

            # Session ContextItems table (for items in ChatSession.context_items)
            await self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._session_context_items_table_name} (
                    id TEXT NOT NULL, session_id TEXT NOT NULL, item_type TEXT NOT NULL,
                    source_id TEXT, content TEXT NOT NULL, tokens INTEGER, metadata TEXT,
                    timestamp TEXT NOT NULL, is_truncated INTEGER DEFAULT 0, original_tokens INTEGER,
                    PRIMARY KEY (session_id, id),
                    FOREIGN KEY (session_id) REFERENCES {self._sessions_table_name}(id) ON DELETE CASCADE
                )
            """)
            await self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_session_context_items_session_id ON {self._session_context_items_table_name} (session_id);")

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
            await self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_context_preset_items_preset_name ON {self._context_preset_items_table_name} (preset_name);")

            await self._conn.commit()
            logger.info(f"SQLite storage initialized at: {self._db_path.resolve()} with tables: "
                        f"{self._sessions_table_name}, {self._messages_table_name}, {self._session_context_items_table_name}, "
                        f"{self._context_presets_table_name}, {self._context_preset_items_table_name}")

        except aiosqlite.Error as e: # type: ignore
            logger.error(f"Failed to initialize aiosqlite database at {self._db_path}: {e}")
            if self._conn: await self._conn.close(); self._conn = None
            raise SessionStorageError(f"Could not initialize SQLite database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during SQLite initialization: {e}", exc_info=True)
            if self._conn: await self._conn.close(); self._conn = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        """Saves/updates session, messages, and session_context_items. (Docstring updated)"""
        if not self._conn: raise SessionStorageError("Database connection not initialized.")
        session_metadata_json = json.dumps(session.metadata or {})
        try:
            await self._conn.execute("BEGIN;")
            await self._conn.execute(f"""
                INSERT OR REPLACE INTO {self._sessions_table_name} (id, name, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session.id, session.name, session.created_at.isoformat(), session.updated_at.isoformat(), session_metadata_json))

            await self._conn.execute(f"DELETE FROM {self._messages_table_name} WHERE session_id = ?", (session.id,))
            if session.messages:
                messages_data = [(msg.id, session.id, str(msg.role), msg.content, msg.timestamp.isoformat(),
                                  msg.tokens, json.dumps(msg.metadata or {})) for msg in session.messages]
                await self._conn.executemany(f"INSERT INTO {self._messages_table_name} (id, session_id, role, content, timestamp, tokens, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)", messages_data)

            await self._conn.execute(f"DELETE FROM {self._session_context_items_table_name} WHERE session_id = ?", (session.id,))
            if session.context_items:
                context_items_data = [(item.id, session.id, str(item.type), item.source_id, item.content, item.tokens,
                                       json.dumps(item.metadata or {}), item.timestamp.isoformat(),
                                       1 if item.is_truncated else 0, item.original_tokens)
                                      for item in session.context_items]
                await self._conn.executemany(f"INSERT INTO {self._session_context_items_table_name} (id, session_id, item_type, source_id, content, tokens, metadata, timestamp, is_truncated, original_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", context_items_data)
            await self._conn.commit()
            logger.debug(f"Session '{session.id}' with {len(session.messages)} messages and {len(session.context_items)} context items saved to SQLite.")
        except aiosqlite.Error as e: # type: ignore
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
        """Retrieves session, messages, and session_context_items. (Docstring updated)"""
        if not self._conn: raise SessionStorageError("Database connection not initialized.")
        try:
            async with self._conn.execute(f"SELECT * FROM {self._sessions_table_name} WHERE id = ?", (session_id,)) as cursor:
                session_row = await cursor.fetchone()
            if not session_row: logger.debug(f"Session '{session_id}' not found in SQLite."); return None

            session_data = dict(session_row)
            session_data["metadata"] = json.loads(session_data.get("metadata") or '{}')
            session_data["created_at"] = datetime.fromisoformat(session_data["created_at"].replace('Z', '+00:00')) if session_data.get("created_at") else datetime.now(timezone.utc)
            session_data["updated_at"] = datetime.fromisoformat(session_data["updated_at"].replace('Z', '+00:00')) if session_data.get("updated_at") else datetime.now(timezone.utc)

            messages: List[Message] = []
            async with self._conn.execute(f"SELECT * FROM {self._messages_table_name} WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)) as cursor:
                async for msg_row_data in cursor:
                    msg_dict = dict(msg_row_data)
                    try:
                        msg_dict["metadata"] = json.loads(msg_dict.get("metadata") or '{}'); msg_dict["role"] = Role(msg_dict["role"])
                        ts_str = msg_dict["timestamp"]; msg_dict["timestamp"] = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                        messages.append(Message.model_validate(msg_dict))
                    except (json.JSONDecodeError, ValueError, TypeError) as e: logger.warning(f"Skipping invalid message data for session {session_id}, msg_id {msg_dict.get('id')}: {e}")
            session_data["messages"] = messages

            context_items: List[ContextItem] = []
            async with self._conn.execute(f"SELECT * FROM {self._session_context_items_table_name} WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)) as cursor:
                async for item_row_data in cursor:
                    item_dict = dict(item_row_data)
                    try:
                        item_dict["metadata"] = json.loads(item_dict.get("metadata") or '{}')
                        item_dict["type"] = ContextItemType(item_dict.pop("item_type")) # Use "item_type" from DB
                        ts_str = item_dict["timestamp"]; item_dict["timestamp"] = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                        item_dict["is_truncated"] = bool(item_dict.get("is_truncated", 0)) # Convert from INTEGER
                        context_items.append(ContextItem.model_validate(item_dict))
                    except (json.JSONDecodeError, ValueError, TypeError) as e: logger.warning(f"Skipping invalid session_context_item data for session {session_id}, item_id {item_dict.get('id')}: {e}")
            session_data["context_items"] = context_items

            chat_session = ChatSession.model_validate(session_data)
            logger.debug(f"Session '{session_id}' loaded from SQLite with {len(messages)} messages and {len(context_items)} context items.")
            return chat_session
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error retrieving session '{session_id}': {e}")
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists session metadata, including message and session_context_item counts. (Docstring updated)"""
        if not self._conn: raise SessionStorageError("Database connection is not initialized.")
        session_metadata_list: List[Dict[str, Any]] = []
        try:
            async with self._conn.execute(f"""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata,
                       (SELECT COUNT(*) FROM {self._messages_table_name} m WHERE m.session_id = s.id) as message_count,
                       (SELECT COUNT(*) FROM {self._session_context_items_table_name} ci WHERE ci.session_id = s.id) as context_item_count
                FROM {self._sessions_table_name} s ORDER BY s.updated_at DESC
            """) as cursor:
                async for row in cursor:
                    data = dict(row)
                    try: data["metadata"] = json.loads(data.get("metadata") or '{}')
                    except json.JSONDecodeError: data["metadata"] = {}
                    session_metadata_list.append(data)
            logger.debug(f"Found {len(session_metadata_list)} sessions in SQLite.")
            return session_metadata_list
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error listing sessions: {e}")
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """Deletes session and associated messages/session_context_items. (Docstring updated)"""
        if not self._conn: raise SessionStorageError("Database connection is not initialized.")
        try:
            cursor = await self._conn.execute(f"DELETE FROM {self._sessions_table_name} WHERE id = ?", (session_id,))
            await self._conn.commit() # CASCADE DELETE handles related tables
            deleted_count = cursor.rowcount
            if deleted_count > 0: logger.info(f"Session '{session_id}' and associated data deleted from SQLite."); return True
            logger.warning(f"Attempted to delete non-existent session '{session_id}' from SQLite."); return False
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error deleting session '{session_id}': {e}")
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    # --- New methods for Context Preset Management ---

    async def save_context_preset(self, preset: ContextPreset) -> None:
        """Saves or updates a context preset and its items in SQLite."""
        if not self._conn: raise StorageError("Database connection not initialized for presets.")
        preset_metadata_json = json.dumps(preset.metadata or {})
        try:
            await self._conn.execute("BEGIN;")
            await self._conn.execute(f"""
                INSERT OR REPLACE INTO {self._context_presets_table_name} (name, description, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (preset.name, preset.description, preset.created_at.isoformat(), preset.updated_at.isoformat(), preset_metadata_json))

            await self._conn.execute(f"DELETE FROM {self._context_preset_items_table_name} WHERE preset_name = ?", (preset.name,))
            if preset.items:
                items_data = [(item.item_id, preset.name, str(item.type), item.content,
                               item.source_identifier, json.dumps(item.metadata or {}))
                              for item in preset.items]
                await self._conn.executemany(f"""
                    INSERT INTO {self._context_preset_items_table_name} (item_id, preset_name, type, content, source_identifier, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, items_data)
            await self._conn.commit()
            logger.info(f"Context preset '{preset.name}' with {len(preset.items)} items saved to SQLite.")
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error saving context preset '{preset.name}': {e}")
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset save: {rb_e}")
            raise StorageError(f"Database error saving context preset '{preset.name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving context preset '{preset.name}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset save: {rb_e}")
            raise StorageError(f"Unexpected error saving context preset '{preset.name}': {e}")

    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """Retrieves a specific context preset and its items by name from SQLite."""
        if not self._conn: raise StorageError("Database connection not initialized for presets.")
        try:
            async with self._conn.execute(f"SELECT * FROM {self._context_presets_table_name} WHERE name = ?", (preset_name,)) as cursor:
                preset_row = await cursor.fetchone()
            if not preset_row: logger.debug(f"Context preset '{preset_name}' not found in SQLite."); return None

            preset_data = dict(preset_row)
            preset_data["metadata"] = json.loads(preset_data.get("metadata") or '{}')
            preset_data["created_at"] = datetime.fromisoformat(preset_data["created_at"].replace('Z', '+00:00'))
            preset_data["updated_at"] = datetime.fromisoformat(preset_data["updated_at"].replace('Z', '+00:00'))

            items: List[ContextPresetItem] = []
            async with self._conn.execute(f"SELECT * FROM {self._context_preset_items_table_name} WHERE preset_name = ?", (preset_name,)) as cursor:
                async for item_row_data in cursor:
                    item_dict = dict(item_row_data)
                    try:
                        item_dict["metadata"] = json.loads(item_dict.get("metadata") or '{}')
                        # Type is already string from DB, ContextPresetItem model will validate
                        items.append(ContextPresetItem.model_validate(item_dict))
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid context_preset_item data for preset {preset_name}, item_id {item_dict.get('item_id')}: {e}")
            preset_data["items"] = items

            context_preset = ContextPreset.model_validate(preset_data)
            logger.debug(f"Context preset '{preset_name}' loaded from SQLite with {len(items)} items.")
            return context_preset
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error retrieving context preset '{preset_name}': {e}")
            raise StorageError(f"Database error retrieving context preset '{preset_name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error retrieving context preset '{preset_name}': {e}")

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """Lists context preset metadata from SQLite, including item counts."""
        if not self._conn: raise StorageError("Database connection not initialized for presets.")
        preset_metadata_list: List[Dict[str, Any]] = []
        try:
            async with self._conn.execute(f"""
                SELECT p.name, p.description, p.created_at, p.updated_at, p.metadata,
                       (SELECT COUNT(*) FROM {self._context_preset_items_table_name} pi WHERE pi.preset_name = p.name) as item_count
                FROM {self._context_presets_table_name} p ORDER BY p.updated_at DESC
            """) as cursor:
                async for row in cursor:
                    data = dict(row)
                    try: data["metadata"] = json.loads(data.get("metadata") or '{}')
                    except json.JSONDecodeError: data["metadata"] = {}
                    preset_metadata_list.append(data)
            logger.debug(f"Found {len(preset_metadata_list)} context presets in SQLite.")
            return preset_metadata_list
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error listing context presets: {e}")
            raise StorageError(f"Database error listing context presets: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing context presets: {e}", exc_info=True)
            raise StorageError(f"Unexpected error listing context presets: {e}")

    async def delete_context_preset(self, preset_name: str) -> bool:
        """Deletes a context preset and its items (due to CASCADE DELETE) from SQLite."""
        if not self._conn: raise StorageError("Database connection not initialized for presets.")
        try:
            cursor = await self._conn.execute(f"DELETE FROM {self._context_presets_table_name} WHERE name = ?", (preset_name,))
            await self._conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0: logger.info(f"Context preset '{preset_name}' and its items deleted from SQLite."); return True
            logger.warning(f"Attempted to delete non-existent context preset '{preset_name}' from SQLite."); return False
        except aiosqlite.Error as e: # type: ignore
            logger.error(f"aiosqlite error deleting context preset '{preset_name}': {e}")
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset delete: {rb_e}")
            raise StorageError(f"Database error deleting context preset '{preset_name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting context preset '{preset_name}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset delete: {rb_e}")
            raise StorageError(f"Unexpected error deleting context preset '{preset_name}': {e}")

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """Renames a context preset in SQLite. This is complex due to PK/FK relationships."""
        if not self._conn: raise StorageError("Database connection not initialized for presets.")
        if old_name == new_name: logger.info(f"Preset rename: old and new names are identical ('{old_name}'). No action taken."); return True

        # Validate new_name (Pydantic model should do this on ContextPreset instantiation)
        try: ContextPreset(name=new_name, items=[])
        except ValueError as ve: logger.error(f"Invalid new preset name '{new_name}': {ve}"); raise

        try:
            async with self._conn.execute(f"SELECT 1 FROM {self._context_presets_table_name} WHERE name = ?", (new_name,)) as cursor:
                if await cursor.fetchone():
                    logger.warning(f"Cannot rename preset: new name '{new_name}' already exists.")
                    return False

            # SQLite doesn't directly support ON UPDATE CASCADE for PKs easily.
            # Strategy: Read old, insert new, delete old (within a transaction).
            old_preset = await self.get_context_preset(old_name)
            if not old_preset:
                logger.warning(f"Cannot rename preset: old name '{old_name}' not found.")
                return False

            # Create a new preset object with the new name and updated timestamp
            renamed_preset = ContextPreset(
                name=new_name,
                description=old_preset.description,
                items=old_preset.items, # Copy items
                created_at=old_preset.created_at, # Preserve original creation time
                updated_at=datetime.now(timezone.utc), # Update modification time
                metadata=old_preset.metadata
            )

            await self._conn.execute("BEGIN;")
            # Save the new preset (this will insert new preset and its items)
            await self.save_context_preset(renamed_preset) # Uses the logic already defined

            # Delete the old preset (this will cascade delete its items from context_preset_items)
            cursor = await self._conn.execute(f"DELETE FROM {self._context_presets_table_name} WHERE name = ?", (old_name,))

            if cursor.rowcount == 0: # Should not happen if get_context_preset found it
                await self._conn.rollback()
                logger.error(f"Rename failed: old preset '{old_name}' disappeared during transaction.")
                return False

            await self._conn.commit()
            logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}' in SQLite.")
            return True

        except (aiosqlite.Error, StorageError) as e: # type: ignore
            logger.error(f"Database error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset rename: {rb_e}")
            # If it was a StorageError from save_context_preset, re-raise it. Otherwise, wrap.
            if isinstance(e, StorageError): raise
            raise StorageError(f"Database error renaming context preset: {e}")
        except Exception as e:
            logger.error(f"Unexpected error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            try: await self._conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed for preset rename: {rb_e}")
            raise StorageError(f"Unexpected error renaming context preset: {e}")


    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                await self._conn.close()
                self._conn = None
                logger.info("aiosqlite storage connection (sessions & presets) closed.")
            except aiosqlite.Error as e: # type: ignore
                logger.error(f"Error closing aiosqlite connection: {e}")
