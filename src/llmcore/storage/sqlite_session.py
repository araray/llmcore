# src/llmcore/storage/sqlite_session.py
"""
SQLite database storage for ChatSession objects using aiosqlite.

This module implements the BaseSessionStorage interface using the
aiosqlite library for native asynchronous database operations.
"""

import json
import logging
import os
import pathlib
from typing import List, Optional, Dict, Any

# Use aiosqlite for native async operations
try:
    import aiosqlite
    aiosqlite_available = True
except ImportError:
    aiosqlite_available = False
    aiosqlite = None # type: ignore

from ..models import ChatSession, Message, Role
from ..exceptions import SessionStorageError, ConfigError
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)


class SqliteSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a SQLite database using aiosqlite.

    Stores session and message data in tables within a single DB file.
    """
    _db_path: pathlib.Path
    _conn: Optional[aiosqlite.Connection] = None # Use aiosqlite connection type

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SQLite database asynchronously using aiosqlite.

        Args:
            config: Configuration dictionary. Expected keys:
                    'path': The directory path for storing the database file.

        Raises:
            ConfigError: If the 'path' is not provided in the config.
            SessionStorageError: If the database cannot be initialized.
            ImportError: If aiosqlite is not installed.
        """
        if not aiosqlite_available:
            raise ImportError("aiosqlite library is not installed. Please install `aiosqlite` or `llmcore[sqlite]`.")

        db_path_str = config.get("path") # Reuse 'path' key like JsonStorage
        if not db_path_str:
            raise ConfigError("SQLite session storage 'path' not specified in configuration.")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))

        try:
            # Ensure parent directory exists (synchronous part is ok here)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect asynchronously
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row # Use aiosqlite's Row factory

            # Enable foreign keys (important!)
            await self._conn.execute("PRAGMA foreign_keys = ON;")

            # Create tables if they don't exist (use await for execute)
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT -- Store as JSON text
                )
            """)
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens INTEGER,
                    selected INTEGER DEFAULT 1, -- Store boolean as INTEGER
                    metadata TEXT, -- Store as JSON text
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            await self._conn.commit() # Commit table creation
            logger.info(f"SQLite session storage initialized asynchronously at: {self._db_path.resolve()}")

        except aiosqlite.Error as e:
            logger.error(f"Failed to initialize aiosqlite database at {self._db_path}: {e}")
            if self._conn: await self._conn.close() # Attempt to close connection on error
            self._conn = None # Ensure connection is None on failure
            raise SessionStorageError(f"Could not initialize SQLite database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during SQLite initialization: {e}", exc_info=True)
            if self._conn: await self._conn.close()
            self._conn = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session in the database asynchronously.

        Args:
            session: The ChatSession object to save.

        Raises:
            SessionStorageError: If the database connection is not initialized or saving fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized. Call initialize() first.")

        session_metadata_json = json.dumps(session.metadata or {})

        try:
            # Use execute for single operations, executemany for batch inserts
            # Use INSERT OR REPLACE (UPSERT) for session metadata
            await self._conn.execute("""
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
            await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))

            if session.messages: # Only insert if there are messages
                messages_data = [
                    (
                        msg.id,
                        session.id, # Use the session's ID
                        msg.role.value,
                        msg.content,
                        msg.timestamp.isoformat(),
                        msg.tokens,
                        1 if msg.selected else 0, # Convert bool to int
                        json.dumps(msg.metadata or {})
                    )
                    for msg in session.messages
                ]
                await self._conn.executemany("""
                    INSERT INTO messages (id, session_id, role, content, timestamp, tokens, selected, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, messages_data)

            await self._conn.commit() # Commit the transaction
            logger.debug(f"Session '{session.id}' saved to SQLite asynchronously.")
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error saving session '{session.id}': {e}")
            # Consider attempting rollback, though commit might have failed partially
            # await self._conn.rollback() # Rollback might not be needed or possible depending on error
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a specific chat session by its ID asynchronously.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.

        Raises:
            SessionStorageError: If the database connection is not initialized or retrieval fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        try:
            # Use execute for single query, fetchone for single result
            async with self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
                session_row = await cursor.fetchone()

            if not session_row:
                logger.debug(f"Session '{session_id}' not found in SQLite.")
                return None

            # Convert row to dictionary (aiosqlite.Row behaves like a dict)
            session_data = dict(session_row)
            session_data["metadata"] = json.loads(session_data["metadata"] or '{}')

            # Fetch messages
            messages = []
            async with self._conn.execute("""
                SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,)) as cursor:
                async for msg_row in cursor:
                    msg_data = dict(msg_row)
                    msg_data["metadata"] = json.loads(msg_data["metadata"] or '{}')
                    # Convert role string back to Enum
                    try:
                        msg_data["role"] = Role(msg_data["role"])
                    except ValueError:
                         logger.warning(f"Invalid role '{msg_data['role']}' found in database for message {msg_data['id']}. Skipping message.")
                         continue # Skip message with invalid role

                    # Convert timestamp string back to datetime
                    try:
                        msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"])
                    except ValueError:
                         logger.warning(f"Invalid timestamp '{msg_data['timestamp']}' found for message {msg_data['id']}. Using current time.")
                         msg_data["timestamp"] = datetime.now() # Fallback

                    # Convert selected int back to bool
                    msg_data["selected"] = bool(msg_data["selected"])

                    # Create Message object
                    messages.append(Message(**msg_data))

            # Create ChatSession object (handle potential datetime parsing errors)
            try:
                 created_at = datetime.fromisoformat(session_data["created_at"])
                 updated_at = datetime.fromisoformat(session_data["updated_at"])
            except ValueError:
                 logger.error(f"Invalid datetime format found for session {session_id}. Cannot load session.")
                 raise SessionStorageError(f"Invalid datetime format in database for session {session_id}.")

            chat_session = ChatSession(
                session_id=session_data["id"], # Use session_id parameter for constructor
                name=session_data["name"],
                created_at=created_at,
                updated_at=updated_at,
                metadata=session_data["metadata"],
                messages=messages
            )
            logger.debug(f"Session '{session_id}' loaded from SQLite asynchronously.")
            return chat_session

        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error retrieving session '{session_id}': {e}")
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions asynchronously, returning metadata only.

        Returns:
            A list of dictionaries, each representing session metadata.

        Raises:
            SessionStorageError: If the database connection is not initialized or listing fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        session_metadata_list: List[Dict[str, Any]] = []
        try:
            # Query includes message count for efficiency
            async with self._conn.execute("""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """) as cursor:
                async for row in cursor:
                    data = dict(row)
                    data["metadata"] = json.loads(data["metadata"] or '{}')
                    session_metadata_list.append(data)

            logger.debug(f"Found {len(session_metadata_list)} sessions in SQLite asynchronously.")
            return session_metadata_list
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error listing sessions: {e}")
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific chat session asynchronously.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was found and deleted successfully, False otherwise.

        Raises:
            SessionStorageError: If the database connection is not initialized or deletion fails.
        """
        if not self._conn:
            raise SessionStorageError("Database connection is not initialized.")

        try:
            # Use execute for single operation, check rowcount for success
            cursor = await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await self._conn.commit() # Commit the deletion
            deleted_count = cursor.rowcount # Number of rows affected

            if deleted_count > 0:
                logger.info(f"Session '{session_id}' deleted from SQLite asynchronously.")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent session '{session_id}' from SQLite.")
                return False
        except aiosqlite.Error as e:
            logger.error(f"aiosqlite error deleting session '{session_id}': {e}")
            # Consider rollback if needed, though commit might have failed
            # await self._conn.rollback()
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """Close the database connection asynchronously."""
        if self._conn:
            try:
                await self._conn.close()
                self._conn = None
                logger.info("aiosqlite session storage connection closed.")
            except aiosqlite.Error as e:
                logger.error(f"Error closing aiosqlite connection: {e}")
                # Log error during cleanup, but don't raise
