# src/llmcore/storage/sqlite_session.py
"""
SQLite database storage for ChatSession objects.

This module implements the BaseSessionStorage interface using the
standard sqlite3 library. Database operations are run in a separate
thread using asyncio.to_thread to maintain the async interface.
"""

import asyncio
import json
import logging
import os
import pathlib
import sqlite3
from typing import List, Optional, Dict, Any

from ..models import ChatSession, Message, Role # Import Role for deserialization
from ..exceptions import SessionStorageError, ConfigError
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)


class SqliteSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a SQLite database.

    Stores session and message data in tables within a single DB file.
    Uses asyncio.to_thread for non-blocking operations.
    """
    _db_path: pathlib.Path
    _conn: Optional[sqlite3.Connection] = None

    # --- Synchronous Helper Methods (to be run in thread) ---

    def _sync_initialize(self, config: Dict[str, Any]) -> None:
        """Synchronous initialization logic."""
        db_path_str = config.get("path") # Reuse 'path' key like JsonStorage
        if not db_path_str:
            raise ConfigError("SQLite session storage 'path' not specified in configuration.")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False) # Allow access from other threads
            self._conn.row_factory = sqlite3.Row # Access columns by name

            with self._conn: # Use context manager for transaction
                cursor = self._conn.cursor()
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                # Create messages table with foreign key and cascade delete
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        tokens INTEGER,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    )
                """)
                # Ensure foreign keys are enabled (important!)
                cursor.execute("PRAGMA foreign_keys = ON;")

            logger.info(f"SQLite session storage initialized at: {self._db_path.resolve()}")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database at {self._db_path}: {e}")
            self._conn = None # Ensure connection is None on failure
            raise SessionStorageError(f"Could not initialize SQLite database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during SQLite initialization: {e}")
            self._conn = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    def _sync_save_session(self, session: ChatSession) -> None:
        """Synchronous save/update logic."""
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        session_metadata_json = json.dumps(session.metadata or {})

        try:
            with self._conn: # Transaction management
                cursor = self._conn.cursor()
                # Use INSERT OR REPLACE (UPSERT) for session metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (id, name, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session.id,
                    session.name,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session_metadata_json
                ))

                # Efficiently update messages: Delete existing and insert all current messages
                # This is simpler than tracking individual message changes.
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))

                if session.messages: # Only insert if there are messages
                    messages_data = [
                        (
                            msg.id,
                            session.id, # Use the session's ID
                            msg.role.value,
                            msg.content,
                            msg.timestamp.isoformat(),
                            msg.tokens,
                            json.dumps(msg.metadata or {})
                        )
                        for msg in session.messages
                    ]
                    cursor.executemany("""
                        INSERT INTO messages (id, session_id, role, content, timestamp, tokens, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, messages_data)

            logger.debug(f"Session '{session.id}' saved to SQLite.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving session '{session.id}': {e}")
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}")
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")

    def _sync_get_session(self, session_id: str) -> Optional[ChatSession]:
        """Synchronous retrieval logic."""
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session_row = cursor.fetchone()

            if not session_row:
                logger.debug(f"Session '{session_id}' not found in SQLite.")
                return None

            session_data = dict(session_row)
            session_data["metadata"] = json.loads(session_data["metadata"] or '{}')

            cursor.execute("""
                SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,))

            messages = []
            for msg_row in cursor.fetchall():
                msg_data = dict(msg_row)
                msg_data["metadata"] = json.loads(msg_data["metadata"] or '{}')
                # Convert role string back to Enum
                msg_data["role"] = Role(msg_data["role"])
                # Convert timestamp string back to datetime
                msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"])
                # Create Message object (adjust if Message constructor changes)
                messages.append(Message(**msg_data))

            # Create ChatSession object (adjust if constructor changes)
            chat_session = ChatSession(
                session_id=session_data["id"],
                name=session_data["name"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                updated_at=datetime.fromisoformat(session_data["updated_at"]),
                metadata=session_data["metadata"],
                messages=messages
            )
            logger.debug(f"Session '{session_id}' loaded from SQLite.")
            return chat_session

        except sqlite3.Error as e:
            logger.error(f"SQLite error retrieving session '{session_id}': {e}")
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}")
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    def _sync_list_sessions(self) -> List[Dict[str, Any]]:
        """Synchronous listing logic."""
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        session_metadata_list: List[Dict[str, Any]] = []
        try:
            cursor = self._conn.cursor()
            # Query includes message count for efficiency
            cursor.execute("""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """)

            for row in cursor.fetchall():
                data = dict(row)
                data["metadata"] = json.loads(data["metadata"] or '{}')
                session_metadata_list.append(data)

            logger.debug(f"Found {len(session_metadata_list)} sessions in SQLite.")
            return session_metadata_list
        except sqlite3.Error as e:
            logger.error(f"SQLite error listing sessions: {e}")
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}")
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    def _sync_delete_session(self, session_id: str) -> bool:
        """Synchronous deletion logic."""
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        try:
            with self._conn: # Transaction
                cursor = self._conn.cursor()
                # Delete the session. Messages are deleted by CASCADE constraint.
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                deleted_count = cursor.rowcount # Number of rows affected

            if deleted_count > 0:
                logger.info(f"Session '{session_id}' deleted from SQLite.")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent session '{session_id}' from SQLite.")
                return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error deleting session '{session_id}': {e}")
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}")
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    def _sync_close(self) -> None:
        """Synchronous closing logic."""
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                logger.info("SQLite session storage connection closed.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite connection: {e}")
                # Don't raise here, just log the error during cleanup

    # --- Async Interface Methods ---

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the SQLite database asynchronously."""
        await asyncio.to_thread(self._sync_initialize, config)

    async def save_session(self, session: ChatSession) -> None:
        """Save or update a chat session asynchronously."""
        await asyncio.to_thread(self._sync_save_session, session)

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a specific chat session asynchronously."""
        return await asyncio.to_thread(self._sync_get_session, session_id)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List available persistent chat sessions asynchronously."""
        return await asyncio.to_thread(self._sync_list_sessions)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a specific chat session asynchronously."""
        return await asyncio.to_thread(self._sync_delete_session, session_id)

    async def close(self) -> None:
        """Close the database connection asynchronously."""
        # Closing can be potentially blocking, though usually fast.
        await asyncio.to_thread(self._sync_close)
