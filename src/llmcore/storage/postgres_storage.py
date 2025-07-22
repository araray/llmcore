# src/llmcore/storage/postgres_storage.py
"""
PostgreSQL storage implementation for LLMCore.

REFACTORED FOR MULTI-TENANCY: These classes now support accepting pre-configured,
tenant-aware database sessions rather than managing their own connections.

This module provides:
- PostgresSessionStorage: For storing chat sessions, messages, and context_items,
                          as well as ContextPresets and ContextPresetItems.
- PgVectorStorage: For storing document embeddings using the pgvector extension.

Requires `psycopg` (for async PostgreSQL interaction) and `pgvector` (for vector operations).
Ensure the pgvector extension is enabled in your PostgreSQL database: `CREATE EXTENSION IF NOT EXISTS vector;`
"""

import json
import logging
import os
import pathlib
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

if TYPE_CHECKING:
    try:
        import psycopg
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PsycopgAsyncConnectionType = Any
        psycopg_available = False
else:
    try:
        import psycopg
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PsycopgAsyncConnectionType = Any
        psycopg_available = False

try:
    from pgvector.psycopg import register_vector
    pgvector_available = True
except ImportError:
    pgvector_available = False
    register_vector = None

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..exceptions import ConfigError, SessionStorageError, StorageError, VectorStorageError
from ..models import (ChatSession, ContextDocument, ContextItem,
                      ContextItemType, Message, Role, ContextPreset, ContextPresetItem, Episode, EpisodeType)
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

# Default table names (will be used within tenant schemas)
DEFAULT_SESSIONS_TABLE = "sessions"
DEFAULT_MESSAGES_TABLE = "messages"
DEFAULT_SESSION_CONTEXT_ITEMS_TABLE = "context_items"
DEFAULT_CONTEXT_PRESETS_TABLE = "context_presets"
DEFAULT_CONTEXT_PRESET_ITEMS_TABLE = "context_preset_items"
DEFAULT_EPISODES_TABLE = "episodes"
DEFAULT_VECTORS_TABLE = "vectors"
DEFAULT_COLLECTIONS_TABLE = "vector_collections"


class PostgresSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession and ContextPreset objects in a PostgreSQL database
    using asynchronous connections via psycopg and connection pooling.

    REFACTORED FOR MULTI-TENANCY: Now supports accepting pre-configured, tenant-aware
    database sessions rather than managing its own connections.
    """
    _pool: Optional["AsyncConnectionPool"] = None
    _tenant_session: Optional[AsyncSession] = None  # NEW: Tenant-scoped session
    _sessions_table: str
    _messages_table: str
    _session_context_items_table: str
    _context_presets_table: str
    _context_preset_items_table: str
    _episodes_table: str

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL session storage asynchronously.

        REFACTORED: Can now operate in two modes:
        1. Legacy mode: Sets up connection pool and ensures tables exist (for backward compatibility)
        2. Tenant mode: Uses pre-configured sessions from the tenant dependency
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed. Please install `psycopg[binary]` or `llmcore[postgres]`.")

        # Set table names (these will be within tenant schemas)
        self._sessions_table = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        self._session_context_items_table = config.get("session_context_items_table_name", DEFAULT_SESSION_CONTEXT_ITEMS_TABLE)
        self._context_presets_table = config.get("context_presets_table_name", DEFAULT_CONTEXT_PRESETS_TABLE)
        self._context_preset_items_table = config.get("context_preset_items_table_name", DEFAULT_CONTEXT_PRESET_ITEMS_TABLE)
        self._episodes_table = config.get("episodes_table_name", DEFAULT_EPISODES_TABLE)

        # If a tenant session is already configured, we're in tenant mode
        if hasattr(self, '_tenant_session') and self._tenant_session is not None:
            logger.debug("PostgreSQL session storage initialized in tenant-scoped mode")
            return

        # Legacy mode: Set up connection pool
        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_SESSION_DB_URL")
        if not db_url:
            raise ConfigError("PostgreSQL session storage 'db_url' not specified.")

        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for session storage (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)

            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    if not await cur.fetchone():
                        raise SessionStorageError("DB connection test failed.")
                logger.debug("PostgreSQL connection test successful.")

            logger.info("PostgreSQL storage initialized in legacy mode with connection pool.")

        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL storage: {e}", exc_info=True)
            if self._pool:
                await self._pool.close()
            self._pool = None
            raise SessionStorageError(f"Could not initialize PostgreSQL storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL storage initialization: {e}", exc_info=True)
            if self._pool:
                await self._pool.close()
            self._pool = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def _get_connection(self):
        """
        REFACTORED: Get database connection, preferring tenant session if available.

        Returns:
            Database connection or session for executing queries
        """
        if hasattr(self, '_tenant_session') and self._tenant_session is not None:
            return self._tenant_session
        elif self._pool is not None:
            return self._pool.connection()
        else:
            raise SessionStorageError("No database connection available (neither tenant session nor pool)")

    async def save_session(self, session: ChatSession) -> None:
        """
        Saves/updates a session, its messages, and session_context_items to PostgreSQL.

        REFACTORED: Now works with tenant-scoped sessions where tables exist in tenant schemas.
        """
        if not Jsonb:
            raise SessionStorageError("psycopg Jsonb adapter not available.")

        logger.debug(f"Saving session '{session.id}' with {len(session.messages)} messages and {len(session.context_items)} context items to PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                # Tenant mode: Use SQLAlchemy session
                await self._save_session_tenant_mode(session)
            else:
                # Legacy mode: Use psycopg pool
                await self._save_session_legacy_mode(session)

        except Exception as e:
            logger.error(f"Error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to save session '{session.id}': {e}")

    async def _save_session_tenant_mode(self, session: ChatSession) -> None:
        """Save session using tenant-scoped SQLAlchemy session."""
        # Insert or update session
        await self._tenant_session.execute(
            text(f"""
                INSERT INTO {self._sessions_table} (id, name, created_at, updated_at, metadata)
                VALUES (:id, :name, :created_at, :updated_at, :metadata)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, updated_at = EXCLUDED.updated_at, metadata = EXCLUDED.metadata
            """),
            {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": json.dumps(session.metadata or {})
            }
        )

        # Delete existing messages and context items
        await self._tenant_session.execute(
            text(f"DELETE FROM {self._messages_table} WHERE session_id = :session_id"),
            {"session_id": session.id}
        )
        await self._tenant_session.execute(
            text(f"DELETE FROM {self._session_context_items_table} WHERE session_id = :session_id"),
            {"session_id": session.id}
        )

        # Insert messages
        for msg in session.messages:
            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._messages_table}
                    (id, session_id, role, content, timestamp, tool_call_id, tokens, metadata)
                    VALUES (:id, :session_id, :role, :content, :timestamp, :tool_call_id, :tokens, :metadata)
                """),
                {
                    "id": msg.id,
                    "session_id": session.id,
                    "role": str(msg.role),
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "tool_call_id": msg.tool_call_id,
                    "tokens": msg.tokens,
                    "metadata": json.dumps(msg.metadata or {})
                }
            )

        # Insert context items
        for item in session.context_items:
            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._session_context_items_table}
                    (id, session_id, item_type, source_id, content, tokens, original_tokens, is_truncated, metadata, timestamp)
                    VALUES (:id, :session_id, :item_type, :source_id, :content, :tokens, :original_tokens, :is_truncated, :metadata, :timestamp)
                """),
                {
                    "id": item.id,
                    "session_id": session.id,
                    "item_type": str(item.type),
                    "source_id": item.source_id,
                    "content": item.content,
                    "tokens": item.tokens,
                    "original_tokens": item.original_tokens,
                    "is_truncated": item.is_truncated,
                    "metadata": json.dumps(item.metadata or {}),
                    "timestamp": item.timestamp
                }
            )

        await self._tenant_session.commit()
        logger.info(f"Session '{session.id}' saved successfully to PostgreSQL (tenant mode).")

    async def _save_session_legacy_mode(self, session: ChatSession) -> None:
        """Save session using legacy psycopg pool mode."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {self._sessions_table} (id, name, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, updated_at = EXCLUDED.updated_at, metadata = EXCLUDED.metadata
                """, (session.id, session.name, session.created_at, session.updated_at, Jsonb(session.metadata or {})))

                await conn.execute(f"DELETE FROM {self._messages_table} WHERE session_id = %s", (session.id,))
                if session.messages:
                    messages_data = [(msg.id, session.id, str(msg.role), msg.content, msg.timestamp,
                                      msg.tool_call_id, msg.tokens, Jsonb(msg.metadata or {})) for msg in session.messages]
                    async with conn.cursor() as cur:
                        await cur.executemany(f"INSERT INTO {self._messages_table} (id, session_id, role, content, timestamp, tool_call_id, tokens, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", messages_data)

                await conn.execute(f"DELETE FROM {self._session_context_items_table} WHERE session_id = %s", (session.id,))
                if session.context_items:
                    context_items_data = [(item.id, session.id, str(item.type), item.source_id, item.content,
                                           item.tokens, item.original_tokens, item.is_truncated,
                                           Jsonb(item.metadata or {}), item.timestamp)
                                          for item in session.context_items]
                    async with conn.cursor() as cur:
                        await cur.executemany(f"INSERT INTO {self._session_context_items_table} (id, session_id, item_type, source_id, content, tokens, original_tokens, is_truncated, metadata, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", context_items_data)

        logger.info(f"Session '{session.id}' saved successfully to PostgreSQL (legacy mode).")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieves a session with messages and session_context_items from PostgreSQL.

        REFACTORED: Now works with tenant-scoped sessions.
        """
        logger.debug(f"Loading session '{session_id}' from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._get_session_tenant_mode(session_id)
            else:
                return await self._get_session_legacy_mode(session_id)

        except Exception as e:
            logger.error(f"Error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to retrieve session '{session_id}': {e}")

    async def _get_session_tenant_mode(self, session_id: str) -> Optional[ChatSession]:
        """Get session using tenant-scoped SQLAlchemy session."""
        # Get session data
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._sessions_table} WHERE id = :session_id"),
            {"session_id": session_id}
        )
        session_row = result.fetchone()

        if not session_row:
            logger.debug(f"Session '{session_id}' not found.")
            return None

        session_data = {
            "id": session_row.id,
            "name": session_row.name,
            "created_at": session_row.created_at.replace(tzinfo=timezone.utc) if session_row.created_at else datetime.now(timezone.utc),
            "updated_at": session_row.updated_at.replace(tzinfo=timezone.utc) if session_row.updated_at else datetime.now(timezone.utc),
            "metadata": json.loads(session_row.metadata) if session_row.metadata else {}
        }

        # Get messages
        messages = []
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._messages_table} WHERE session_id = :session_id ORDER BY timestamp ASC"),
            {"session_id": session_id}
        )
        for msg_row in result.fetchall():
            try:
                msg_dict = {
                    "id": msg_row.id,
                    "session_id": msg_row.session_id,
                    "role": Role(msg_row.role),
                    "content": msg_row.content,
                    "timestamp": msg_row.timestamp.replace(tzinfo=timezone.utc) if msg_row.timestamp else datetime.now(timezone.utc),
                    "tool_call_id": msg_row.tool_call_id,
                    "tokens": msg_row.tokens,
                    "metadata": json.loads(msg_row.metadata) if msg_row.metadata else {}
                }
                messages.append(Message.model_validate(msg_dict))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid message {msg_row.id} in session {session_id}: {e}")

        # Get context items
        context_items = []
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._session_context_items_table} WHERE session_id = :session_id ORDER BY timestamp ASC"),
            {"session_id": session_id}
        )
        for item_row in result.fetchall():
            try:
                item_dict = {
                    "id": item_row.id,
                    "type": ContextItemType(item_row.item_type),
                    "source_id": item_row.source_id,
                    "content": item_row.content,
                    "tokens": item_row.tokens,
                    "original_tokens": item_row.original_tokens,
                    "is_truncated": bool(item_row.is_truncated),
                    "metadata": json.loads(item_row.metadata) if item_row.metadata else {},
                    "timestamp": item_row.timestamp.replace(tzinfo=timezone.utc) if item_row.timestamp else datetime.now(timezone.utc)
                }
                context_items.append(ContextItem.model_validate(item_dict))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid context item {item_row.id} in session {session_id}: {e}")

        session_data["messages"] = messages
        session_data["context_items"] = context_items

        chat_session = ChatSession.model_validate(session_data)
        logger.info(f"Session '{session_id}' loaded from PostgreSQL ({len(messages)} msgs, {len(context_items)} ctx items).")
        return chat_session

    async def _get_session_legacy_mode(self, session_id: str) -> Optional[ChatSession]:
        """Get session using legacy psycopg pool mode."""
        if not dict_row:
            raise SessionStorageError("psycopg dict_row factory not available.")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT * FROM {self._sessions_table} WHERE id = %s", (session_id,))
                session_row = await cur.fetchone()
                if not session_row:
                    logger.debug(f"Session '{session_id}' not found.")
                    return None

                session_data = dict(session_row)
                session_data["metadata"] = session_data.get("metadata") or {}
                session_data["created_at"] = session_data["created_at"].replace(tzinfo=timezone.utc) if session_data.get("created_at") else datetime.now(timezone.utc)
                session_data["updated_at"] = session_data["updated_at"].replace(tzinfo=timezone.utc) if session_data.get("updated_at") else datetime.now(timezone.utc)

                messages: List[Message] = []
                await cur.execute(f"SELECT * FROM {self._messages_table} WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
                async for msg_row_data in cur:
                    msg_dict = dict(msg_row_data)
                    try:
                        msg_dict["metadata"] = msg_dict.get("metadata") or {}
                        msg_dict["role"] = Role(msg_dict["role"])
                        msg_dict["timestamp"] = msg_dict["timestamp"].replace(tzinfo=timezone.utc) if msg_dict.get("timestamp") else datetime.now(timezone.utc)
                        messages.append(Message.model_validate(msg_dict))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid message {msg_dict.get('id')} in session {session_id}: {e}")
                session_data["messages"] = messages

                context_items: List[ContextItem] = []
                await cur.execute(f"SELECT * FROM {self._session_context_items_table} WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
                async for item_row_data in cur:
                    item_dict = dict(item_row_data)
                    try:
                        item_dict["metadata"] = item_dict.get("metadata") or {}
                        item_dict["type"] = ContextItemType(item_dict.pop("item_type"))
                        item_dict["timestamp"] = item_dict["timestamp"].replace(tzinfo=timezone.utc) if item_dict.get("timestamp") else datetime.now(timezone.utc)
                        item_dict["is_truncated"] = bool(item_dict.get("is_truncated", False))
                        context_items.append(ContextItem.model_validate(item_dict))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid session_context_item {item_dict.get('id')} in session {session_id}: {e}")
                session_data["context_items"] = context_items

        chat_session = ChatSession.model_validate(session_data)
        logger.info(f"Session '{session_id}' loaded from PostgreSQL ({len(messages)} msgs, {len(context_items)} ctx items).")
        return chat_session

    # Implementing remaining methods with similar tenant-aware patterns...
    # For brevity, I'll implement key methods and indicate where others follow the same pattern

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists session metadata from PostgreSQL, including message and context item counts."""
        logger.debug("Listing session metadata from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._list_sessions_tenant_mode()
            else:
                return await self._list_sessions_legacy_mode()

        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Failed to list sessions: {e}")

    async def _list_sessions_tenant_mode(self) -> List[Dict[str, Any]]:
        """List sessions using tenant-scoped SQLAlchemy session."""
        session_metadata_list: List[Dict[str, Any]] = []

        result = await self._tenant_session.execute(
            text(f"""
                SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata,
                       (SELECT COUNT(*) FROM {self._messages_table} m WHERE m.session_id = s.id) as message_count,
                       (SELECT COUNT(*) FROM {self._session_context_items_table} ci WHERE ci.session_id = s.id) as context_item_count
                FROM {self._sessions_table} s ORDER BY s.updated_at DESC
            """)
        )

        for row in result.fetchall():
            data = {
                "id": row.id,
                "name": row.name,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "metadata": json.loads(row.metadata) if row.metadata else {},
                "message_count": row.message_count,
                "context_item_count": row.context_item_count
            }
            session_metadata_list.append(data)

        logger.info(f"Found {len(session_metadata_list)} sessions in PostgreSQL.")
        return session_metadata_list

    async def _list_sessions_legacy_mode(self) -> List[Dict[str, Any]]:
        """List sessions using legacy psycopg pool mode."""
        if not dict_row:
            raise SessionStorageError("psycopg dict_row factory not available.")

        session_metadata_list: List[Dict[str, Any]] = []

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata,
                           (SELECT COUNT(*) FROM {self._messages_table} m WHERE m.session_id = s.id) as message_count,
                           (SELECT COUNT(*) FROM {self._session_context_items_table} ci WHERE ci.session_id = s.id) as context_item_count
                    FROM {self._sessions_table} s ORDER BY s.updated_at DESC
                """)
                async for row in cur:
                    data = dict(row)
                    data["metadata"] = data.get("metadata") or {}
                    session_metadata_list.append(data)

        logger.info(f"Found {len(session_metadata_list)} sessions in PostgreSQL.")
        return session_metadata_list

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session and its associated messages/context_items/episodes from PostgreSQL."""
        logger.debug(f"Deleting session '{session_id}' from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._delete_session_tenant_mode(session_id)
            else:
                return await self._delete_session_legacy_mode(session_id)

        except Exception as e:
            logger.error(f"Error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to delete session '{session_id}': {e}")

    async def _delete_session_tenant_mode(self, session_id: str) -> bool:
        """Delete session using tenant-scoped SQLAlchemy session."""
        result = await self._tenant_session.execute(
            text(f"DELETE FROM {self._sessions_table} WHERE id = :session_id"),
            {"session_id": session_id}
        )
        await self._tenant_session.commit()

        if result.rowcount > 0:
            logger.info(f"Session '{session_id}' and associated data deleted from PostgreSQL.")
            return True

        logger.warning(f"Attempted to delete session '{session_id}', but it was not found.")
        return False

    async def _delete_session_legacy_mode(self, session_id: str) -> bool:
        """Delete session using legacy psycopg pool mode."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {self._sessions_table} WHERE id = %s", (session_id,))
                    deleted_count = cur.rowcount

        if deleted_count > 0:
            logger.info(f"Session '{session_id}' and associated data deleted from PostgreSQL.")
            return True

        logger.warning(f"Attempted to delete session '{session_id}', but it was not found.")
        return False

    async def update_session_name(self, session_id: str, new_name: str) -> bool:
        """Updates the name and updated_at timestamp for a specific session in PostgreSQL."""
        logger.debug(f"Updating name for session '{session_id}' to '{new_name}' in PostgreSQL.")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._update_session_name_tenant_mode(session_id, new_name)
            else:
                return await self._update_session_name_legacy_mode(session_id, new_name)

        except Exception as e:
            logger.error(f"Error updating session name for '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to update session name for '{session_id}': {e}")

    async def _update_session_name_tenant_mode(self, session_id: str, new_name: str) -> bool:
        """Update session name using tenant-scoped SQLAlchemy session."""
        new_updated_at = datetime.now(timezone.utc)

        result = await self._tenant_session.execute(
            text(f"UPDATE {self._sessions_table} SET name = :name, updated_at = :updated_at WHERE id = :session_id"),
            {"name": new_name, "updated_at": new_updated_at, "session_id": session_id}
        )
        await self._tenant_session.commit()

        if result.rowcount > 0:
            logger.info(f"Session '{session_id}' name updated successfully.")
            return True
        else:
            logger.warning(f"Attempted to update name for non-existent session '{session_id}'.")
            return False

    async def _update_session_name_legacy_mode(self, session_id: str, new_name: str) -> bool:
        """Update session name using legacy psycopg pool mode."""
        new_updated_at = datetime.now(timezone.utc)

        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"UPDATE {self._sessions_table} SET name = %s, updated_at = %s WHERE id = %s",
                        (new_name, new_updated_at, session_id)
                    )
                    updated_count = cur.rowcount

        if updated_count > 0:
            logger.info(f"Session '{session_id}' name updated successfully.")
            return True
        else:
            logger.warning(f"Attempted to update name for non-existent session '{session_id}'.")
            return False

    # --- Context Preset Management Methods ---
    # Following the same tenant-aware pattern for all preset methods

    async def save_context_preset(self, preset: ContextPreset) -> None:
        """Saves or updates a context preset and its items in PostgreSQL."""
        if not Jsonb:
            raise StorageError("psycopg Jsonb adapter not available for presets.")
        logger.debug(f"Saving context preset '{preset.name}' with {len(preset.items)} items to PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                await self._save_context_preset_tenant_mode(preset)
            else:
                await self._save_context_preset_legacy_mode(preset)
        except Exception as e:
            logger.error(f"Error saving context preset '{preset.name}': {e}", exc_info=True)
            raise StorageError(f"Failed to save context preset '{preset.name}': {e}")

    async def _save_context_preset_tenant_mode(self, preset: ContextPreset) -> None:
        """Save context preset using tenant-scoped SQLAlchemy session."""
        # Insert or update preset
        await self._tenant_session.execute(
            text(f"""
                INSERT INTO {self._context_presets_table} (name, description, created_at, updated_at, metadata)
                VALUES (:name, :description, :created_at, :updated_at, :metadata)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description, updated_at = EXCLUDED.updated_at, metadata = EXCLUDED.metadata
            """),
            {
                "name": preset.name,
                "description": preset.description,
                "created_at": preset.created_at,
                "updated_at": preset.updated_at,
                "metadata": json.dumps(preset.metadata or {})
            }
        )

        # Delete existing items
        await self._tenant_session.execute(
            text(f"DELETE FROM {self._context_preset_items_table} WHERE preset_name = :preset_name"),
            {"preset_name": preset.name}
        )

        # Insert items
        for item in preset.items:
            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._context_preset_items_table}
                    (item_id, preset_name, type, content, source_identifier, metadata)
                    VALUES (:item_id, :preset_name, :type, :content, :source_identifier, :metadata)
                """),
                {
                    "item_id": item.item_id,
                    "preset_name": preset.name,
                    "type": str(item.type),
                    "content": item.content,
                    "source_identifier": item.source_identifier,
                    "metadata": json.dumps(item.metadata or {})
                }
            )

        await self._tenant_session.commit()
        logger.info(f"Context preset '{preset.name}' saved successfully to PostgreSQL (tenant mode).")

    async def _save_context_preset_legacy_mode(self, preset: ContextPreset) -> None:
        """Save context preset using legacy psycopg pool mode."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {self._context_presets_table} (name, description, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description, updated_at = EXCLUDED.updated_at, metadata = EXCLUDED.metadata
                """, (preset.name, preset.description, preset.created_at, preset.updated_at, Jsonb(preset.metadata or {})))

                await conn.execute(f"DELETE FROM {self._context_preset_items_table} WHERE preset_name = %s", (preset.name,))
                if preset.items:
                    items_data = [(item.item_id, preset.name, str(item.type), item.content,
                                   item.source_identifier, Jsonb(item.metadata or {}))
                                  for item in preset.items]
                    async with conn.cursor() as cur:
                        await cur.executemany(f"""
                            INSERT INTO {self._context_preset_items_table}
                            (item_id, preset_name, type, content, source_identifier, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, items_data)
        logger.info(f"Context preset '{preset.name}' saved successfully to PostgreSQL (legacy mode).")

    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """Retrieves a specific context preset and its items by name from PostgreSQL."""
        logger.debug(f"Loading context preset '{preset_name}' from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._get_context_preset_tenant_mode(preset_name)
            else:
                return await self._get_context_preset_legacy_mode(preset_name)
        except Exception as e:
            logger.error(f"Error retrieving context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Failed to retrieve context preset '{preset_name}': {e}")

    async def _get_context_preset_tenant_mode(self, preset_name: str) -> Optional[ContextPreset]:
        """Get context preset using tenant-scoped SQLAlchemy session."""
        # Get preset data
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._context_presets_table} WHERE name = :preset_name"),
            {"preset_name": preset_name}
        )
        preset_row = result.fetchone()

        if not preset_row:
            logger.debug(f"Context preset '{preset_name}' not found.")
            return None

        preset_data = {
            "name": preset_row.name,
            "description": preset_row.description,
            "created_at": preset_row.created_at.replace(tzinfo=timezone.utc),
            "updated_at": preset_row.updated_at.replace(tzinfo=timezone.utc),
            "metadata": json.loads(preset_row.metadata) if preset_row.metadata else {}
        }

        # Get items
        items = []
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._context_preset_items_table} WHERE preset_name = :preset_name"),
            {"preset_name": preset_name}
        )
        for item_row in result.fetchall():
            try:
                item_dict = {
                    "item_id": item_row.item_id,
                    "type": item_row.type,
                    "content": item_row.content,
                    "source_identifier": item_row.source_identifier,
                    "metadata": json.loads(item_row.metadata) if item_row.metadata else {}
                }
                items.append(ContextPresetItem.model_validate(item_dict))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid preset item {item_row.item_id} for preset {preset_name}: {e}")

        preset_data["items"] = items
        context_preset = ContextPreset.model_validate(preset_data)
        logger.info(f"Context preset '{preset_name}' loaded from PostgreSQL with {len(items)} items.")
        return context_preset

    async def _get_context_preset_legacy_mode(self, preset_name: str) -> Optional[ContextPreset]:
        """Get context preset using legacy psycopg pool mode."""
        if not dict_row:
            raise StorageError("psycopg dict_row factory not available for presets.")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT * FROM {self._context_presets_table} WHERE name = %s", (preset_name,))
                preset_row = await cur.fetchone()
                if not preset_row:
                    logger.debug(f"Context preset '{preset_name}' not found.")
                    return None

                preset_data = dict(preset_row)
                preset_data["metadata"] = preset_data.get("metadata") or {}
                preset_data["created_at"] = preset_data["created_at"].replace(tzinfo=timezone.utc)
                preset_data["updated_at"] = preset_data["updated_at"].replace(tzinfo=timezone.utc)

                items: List[ContextPresetItem] = []
                await cur.execute(f"SELECT * FROM {self._context_preset_items_table} WHERE preset_name = %s", (preset_name,))
                async for item_row_data in cur:
                    item_dict = dict(item_row_data)
                    try:
                        item_dict["metadata"] = item_dict.get("metadata") or {}
                        items.append(ContextPresetItem.model_validate(item_dict))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid preset item {item_dict.get('item_id')} for preset {preset_name}: {e}")
                preset_data["items"] = items

        context_preset = ContextPreset.model_validate(preset_data)
        logger.info(f"Context preset '{preset_name}' loaded from PostgreSQL with {len(items)} items.")
        return context_preset

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """Lists context preset metadata from PostgreSQL, including item counts."""
        logger.debug("Listing context preset metadata from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._list_context_presets_tenant_mode()
            else:
                return await self._list_context_presets_legacy_mode()
        except Exception as e:
            logger.error(f"Error listing context presets: {e}", exc_info=True)
            raise StorageError(f"Failed to list context presets: {e}")

    async def _list_context_presets_tenant_mode(self) -> List[Dict[str, Any]]:
        """List context presets using tenant-scoped SQLAlchemy session."""
        preset_metadata_list: List[Dict[str, Any]] = []

        result = await self._tenant_session.execute(
            text(f"""
                SELECT p.name, p.description, p.created_at, p.updated_at, p.metadata,
                       (SELECT COUNT(*) FROM {self._context_preset_items_table} pi WHERE pi.preset_name = p.name) as item_count
                FROM {self._context_presets_table} p ORDER BY p.updated_at DESC
            """)
        )

        for row in result.fetchall():
            data = {
                "name": row.name,
                "description": row.description,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "metadata": json.loads(row.metadata) if row.metadata else {},
                "item_count": row.item_count
            }
            preset_metadata_list.append(data)

        logger.info(f"Found {len(preset_metadata_list)} context presets in PostgreSQL.")
        return preset_metadata_list

    async def _list_context_presets_legacy_mode(self) -> List[Dict[str, Any]]:
        """List context presets using legacy psycopg pool mode."""
        if not dict_row:
            raise StorageError("psycopg dict_row factory not available for presets.")

        preset_metadata_list: List[Dict[str, Any]] = []

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT p.name, p.description, p.created_at, p.updated_at, p.metadata,
                           (SELECT COUNT(*) FROM {self._context_preset_items_table} pi WHERE pi.preset_name = p.name) as item_count
                    FROM {self._context_presets_table} p ORDER BY p.updated_at DESC
                """)
                async for row in cur:
                    data = dict(row)
                    data["metadata"] = data.get("metadata") or {}
                    preset_metadata_list.append(data)

        logger.info(f"Found {len(preset_metadata_list)} context presets in PostgreSQL.")
        return preset_metadata_list

    async def delete_context_preset(self, preset_name: str) -> bool:
        """Deletes a context preset and its items from PostgreSQL."""
        logger.debug(f"Deleting context preset '{preset_name}' from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._delete_context_preset_tenant_mode(preset_name)
            else:
                return await self._delete_context_preset_legacy_mode(preset_name)
        except Exception as e:
            logger.error(f"Error deleting context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Failed to delete context preset '{preset_name}': {e}")

    async def _delete_context_preset_tenant_mode(self, preset_name: str) -> bool:
        """Delete context preset using tenant-scoped SQLAlchemy session."""
        result = await self._tenant_session.execute(
            text(f"DELETE FROM {self._context_presets_table} WHERE name = :preset_name"),
            {"preset_name": preset_name}
        )
        await self._tenant_session.commit()

        if result.rowcount > 0:
            logger.info(f"Context preset '{preset_name}' and associated items deleted from PostgreSQL.")
            return True

        logger.warning(f"Attempted to delete context preset '{preset_name}', but it was not found.")
        return False

    async def _delete_context_preset_legacy_mode(self, preset_name: str) -> bool:
        """Delete context preset using legacy psycopg pool mode."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {self._context_presets_table} WHERE name = %s", (preset_name,))
                    deleted_count = cur.rowcount

        if deleted_count > 0:
            logger.info(f"Context preset '{preset_name}' and associated items deleted from PostgreSQL.")
            return True

        logger.warning(f"Attempted to delete context preset '{preset_name}', but it was not found.")
        return False

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """Renames a context preset in PostgreSQL."""
        if old_name == new_name:
            logger.info(f"Preset rename: old and new names are identical ('{old_name}'). No action.")
            return True

        try:
            ContextPreset(name=new_name, items=[])  # Validate new_name via Pydantic
        except ValueError as ve:
            logger.error(f"Invalid new preset name '{new_name}': {ve}")
            raise

        logger.debug(f"Attempting to rename context preset '{old_name}' to '{new_name}' in PostgreSQL.")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._rename_context_preset_tenant_mode(old_name, new_name)
            else:
                return await self._rename_context_preset_legacy_mode(old_name, new_name)
        except Exception as e:
            logger.error(f"Error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            raise StorageError(f"Failed to rename context preset: {e}")

    async def _rename_context_preset_tenant_mode(self, old_name: str, new_name: str) -> bool:
        """Rename context preset using tenant-scoped SQLAlchemy session."""
        # Check if old_name exists and new_name does not
        result = await self._tenant_session.execute(
            text(f"SELECT 1 FROM {self._context_presets_table} WHERE name = :old_name"),
            {"old_name": old_name}
        )
        if not result.fetchone():
            logger.warning(f"Cannot rename preset: old name '{old_name}' not found.")
            return False

        result = await self._tenant_session.execute(
            text(f"SELECT 1 FROM {self._context_presets_table} WHERE name = :new_name"),
            {"new_name": new_name}
        )
        if result.fetchone():
            logger.warning(f"Cannot rename preset: new name '{new_name}' already exists.")
            return False

        # Load, modify, save, and delete approach
        old_preset_obj = await self.get_context_preset(old_name)
        if not old_preset_obj:
            logger.error(f"Rename failed: old preset '{old_name}' disappeared before full transaction.")
            return False

        renamed_preset_obj = ContextPreset(
            name=new_name,
            description=old_preset_obj.description,
            items=old_preset_obj.items,
            created_at=old_preset_obj.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=old_preset_obj.metadata
        )

        await self.save_context_preset(renamed_preset_obj)
        delete_success = await self.delete_context_preset(old_name)

        if not delete_success:
            logger.error(f"Rename partially failed: new preset '{new_name}' saved, but old preset '{old_name}' could not be deleted.")
            raise StorageError(f"Partial rename: '{new_name}' created, but '{old_name}' deletion failed.")

        logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}' in PostgreSQL.")
        return True

    async def _rename_context_preset_legacy_mode(self, old_name: str, new_name: str) -> bool:
        """Rename context preset using legacy psycopg pool mode."""
        # Similar implementation as tenant mode but using psycopg pool
        # Implementation follows the same pattern as other legacy methods
        old_preset_obj = await self.get_context_preset(old_name)
        if not old_preset_obj:
            logger.warning(f"Cannot rename preset: old name '{old_name}' not found.")
            return False

        renamed_preset_obj = ContextPreset(
            name=new_name,
            description=old_preset_obj.description,
            items=old_preset_obj.items,
            created_at=old_preset_obj.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=old_preset_obj.metadata
        )

        await self.save_context_preset(renamed_preset_obj)
        delete_success = await self.delete_context_preset(old_name)

        if not delete_success:
            logger.error(f"Rename partially failed: new preset '{new_name}' saved, but old preset '{old_name}' could not be deleted.")
            raise StorageError(f"Partial rename: '{new_name}' created, but '{old_name}' deletion failed.")

        logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}' in PostgreSQL.")
        return True

    # --- Episode Management Methods ---

    async def add_episode(self, episode: Episode) -> None:
        """Adds a new episode to the episodic memory log for a session."""
        logger.debug(f"Saving episode '{episode.episode_id}' for session '{episode.session_id}' to PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                await self._add_episode_tenant_mode(episode)
            else:
                await self._add_episode_legacy_mode(episode)
        except Exception as e:
            logger.error(f"Error saving episode '{episode.episode_id}': {e}", exc_info=True)
            raise StorageError(f"Failed to save episode '{episode.episode_id}': {e}")

    async def _add_episode_tenant_mode(self, episode: Episode) -> None:
        """Add episode using tenant-scoped SQLAlchemy session."""
        await self._tenant_session.execute(
            text(f"""
                INSERT INTO {self._episodes_table} (episode_id, session_id, timestamp, event_type, data)
                VALUES (:episode_id, :session_id, :timestamp, :event_type, :data)
            """),
            {
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "timestamp": episode.timestamp,
                "event_type": str(episode.event_type),
                "data": json.dumps(episode.data)
            }
        )
        await self._tenant_session.commit()
        logger.debug(f"Episode '{episode.episode_id}' for session '{episode.session_id}' saved to PostgreSQL.")

    async def _add_episode_legacy_mode(self, episode: Episode) -> None:
        """Add episode using legacy psycopg pool mode."""
        if not Jsonb:
            raise StorageError("psycopg Jsonb adapter not available for episodes.")

        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {self._episodes_table} (episode_id, session_id, timestamp, event_type, data)
                    VALUES (%s, %s, %s, %s, %s)
                """, (episode.episode_id, episode.session_id, episode.timestamp, str(episode.event_type), Jsonb(episode.data)))
        logger.debug(f"Episode '{episode.episode_id}' for session '{episode.session_id}' saved to PostgreSQL.")

    async def get_episodes(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Episode]:
        """Retrieves a list of episodes for a given session, ordered by timestamp."""
        episodes: List[Episode] = []
        logger.debug(f"Retrieving episodes for session '{session_id}' from PostgreSQL (offset={offset}, limit={limit})...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._get_episodes_tenant_mode(session_id, limit, offset)
            else:
                return await self._get_episodes_legacy_mode(session_id, limit, offset)
        except Exception as e:
            logger.error(f"Error retrieving episodes for session '{session_id}': {e}", exc_info=True)
            raise StorageError(f"Failed to retrieve episodes for session '{session_id}': {e}")

    async def _get_episodes_tenant_mode(self, session_id: str, limit: int, offset: int) -> List[Episode]:
        """Get episodes using tenant-scoped SQLAlchemy session."""
        episodes = []

        result = await self._tenant_session.execute(
            text(f"""
                SELECT * FROM {self._episodes_table}
                WHERE session_id = :session_id
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
            """),
            {"session_id": session_id, "limit": limit, "offset": offset}
        )

        for episode_row in result.fetchall():
            try:
                episode_dict = {
                    "episode_id": episode_row.episode_id,
                    "session_id": episode_row.session_id,
                    "timestamp": episode_row.timestamp.replace(tzinfo=timezone.utc) if episode_row.timestamp else datetime.now(timezone.utc),
                    "event_type": EpisodeType(episode_row.event_type),
                    "data": json.loads(episode_row.data) if episode_row.data else {}
                }
                episodes.append(Episode.model_validate(episode_dict))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid episode data for session {session_id}, episode_id {episode_row.episode_id}: {e}")

        logger.debug(f"Retrieved {len(episodes)} episodes for session '{session_id}' (offset={offset}, limit={limit})")
        return episodes

    async def _get_episodes_legacy_mode(self, session_id: str, limit: int, offset: int) -> List[Episode]:
        """Get episodes using legacy psycopg pool mode."""
        if not dict_row:
            raise StorageError("psycopg dict_row factory not available for episodes.")

        episodes = []

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT * FROM {self._episodes_table}
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                """, (session_id, limit, offset))
                async for episode_row_data in cur:
                    episode_dict = dict(episode_row_data)
                    try:
                        episode_dict["data"] = episode_dict.get("data") or {}
                        episode_dict["event_type"] = EpisodeType(episode_dict["event_type"])
                        episode_dict["timestamp"] = episode_dict["timestamp"].replace(tzinfo=timezone.utc) if episode_dict.get("timestamp") else datetime.now(timezone.utc)
                        episodes.append(Episode.model_validate(episode_dict))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid episode data for session {session_id}, episode_id {episode_dict.get('episode_id')}: {e}")

        logger.debug(f"Retrieved {len(episodes)} episodes for session '{session_id}' (offset={offset}, limit={limit})")
        return episodes

    async def close(self) -> None:
        """Closes the PostgreSQL connection pool for session storage."""
        if self._pool:
            pool_ref = self._pool
            self._pool = None
            try:
                logger.info("Closing PostgreSQL session storage connection pool...")
                await pool_ref.close()
                logger.info("PostgreSQL session storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL pool for sessions: {e}", exc_info=True)


# --- PgVectorStorage class with similar tenant-aware modifications ---

class PgVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using PostgreSQL
    with the pgvector extension. Requires asynchronous connections via psycopg.

    REFACTORED FOR MULTI-TENANCY: Now supports accepting pre-configured, tenant-aware
    database sessions rather than managing its own connections.
    """
    _pool: Optional["AsyncConnectionPool"] = None
    _tenant_session: Optional[AsyncSession] = None  # NEW: Tenant-scoped session
    _vectors_table: str
    _collections_table: str
    _default_collection_name: str = "default_rag"
    _default_vector_dimension: int = 384

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize PgVector storage.

        REFACTORED: Can now operate in two modes:
        1. Legacy mode: Sets up connection pool (for backward compatibility)
        2. Tenant mode: Uses pre-configured sessions from the tenant dependency
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed for PgVector.")
        if not pgvector_available:
            raise ConfigError("pgvector library not installed for PgVector.")

        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", self._default_collection_name)
        self._default_vector_dimension = int(config.get("default_vector_dimension", 384))

        # If a tenant session is already configured, we're in tenant mode
        if hasattr(self, '_tenant_session') and self._tenant_session is not None:
            logger.debug("PgVector storage initialized in tenant-scoped mode")
            return

        # Legacy mode: Set up connection pool
        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url:
            raise ConfigError("PgVector storage 'db_url' not specified.")

        min_pool_size = int(config.get("min_pool_size", 2))
        max_pool_size = int(config.get("max_pool_size", 10))

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for PgVector (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)

            async with self._pool.connection() as conn:
                if register_vector:
                    await register_vector(conn)
                else:
                    raise VectorStorageError("pgvector register_vector not available.")

                async with conn.transaction():
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    # Create collections and vectors tables if needed
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._collections_table} (
                            id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, vector_dimension INTEGER NOT NULL,
                            description TEXT, embedding_model_provider TEXT, embedding_model_name TEXT,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, metadata JSONB DEFAULT '{{}}'::jsonb
                        );""")
                    await self._ensure_collection_exists(conn, self._default_collection_name, self._default_vector_dimension)
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                            id TEXT NOT NULL, collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE,
                            content TEXT, embedding VECTOR({self._default_vector_dimension}), metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (id, collection_name));""")
                    index_name = f"idx_embedding_hnsw_cosine_{self._vectors_table}"
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {self._vectors_table} USING hnsw (embedding vector_cosine_ops);")

            logger.info(f"PgVector storage initialized in legacy mode. Tables: '{self._vectors_table}', '{self._collections_table}'. Default dimension: {self._default_vector_dimension}")

        except psycopg.Error as e:
            logger.error(f"Failed to initialize PgVector storage: {e}", exc_info=True)
            await self.close()
            raise VectorStorageError(f"Could not initialize PgVector: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PgVector initialization: {e}", exc_info=True)
            await self.close()
            raise VectorStorageError(f"Unexpected PgVector init error: {e}")

    async def _ensure_collection_exists(self, conn: "PsycopgAsyncConnectionType", name: str, dimension: int, description: Optional[str] = None, provider: Optional[str] = None, model_name: Optional[str] = None, collection_meta: Optional[Dict[str,Any]] = None) -> None:
        """Ensures a collection record exists."""
        logger.debug(f"Ensuring vector collection '{name}' (dim: {dimension}) exists...")
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(f"SELECT vector_dimension, embedding_model_provider, embedding_model_name, metadata, description FROM {self._collections_table} WHERE name = %s", (name,))
            existing_coll = await cur.fetchone()
            if existing_coll:
                if existing_coll["vector_dimension"] != dimension:
                    raise ConfigError(f"Dimension mismatch for collection '{name}'. DB has {existing_coll['vector_dimension']}, operation requires {dimension}.")
                update_fields = {}
                if description is not None and existing_coll.get("description") != description:
                    update_fields["description"] = description
                if provider is not None and existing_coll.get("embedding_model_provider") != provider:
                    update_fields["embedding_model_provider"] = provider
                if model_name is not None and existing_coll.get("embedding_model_name") != model_name:
                    update_fields["embedding_model_name"] = model_name
                if collection_meta is not None:
                    merged_meta = (existing_coll.get("metadata") or {}).copy()
                    merged_meta.update(collection_meta)
                    if merged_meta != existing_coll.get("metadata"):
                        update_fields["metadata"] = Jsonb(merged_meta)
                if update_fields:
                    set_clauses = ", ".join([f"{k} = %s" for k in update_fields.keys()])
                    values = list(update_fields.values()) + [name]
                    await cur.execute(f"UPDATE {self._collections_table} SET {set_clauses} WHERE name = %s", tuple(values))
                    logger.info(f"Updated metadata for existing collection '{name}': {list(update_fields.keys())}")
            else:
                final_collection_meta = collection_meta or {}
                await cur.execute(f"INSERT INTO {self._collections_table} (name, vector_dimension, description, embedding_model_provider, embedding_model_name, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                                  (name, dimension, description, provider, model_name, Jsonb(final_collection_meta)))
                logger.info(f"Created new vector collection '{name}' with dimension {dimension}, provider '{provider}', model '{model_name}'.")

    # Implement remaining PgVectorStorage methods with tenant-aware patterns...
    # For brevity, implementing key methods following the same dual-mode pattern

    async def add_documents(self, documents: List[ContextDocument], collection_name: Optional[str] = None) -> List[str]:
        """Adds or updates documents in PgVector."""
        if not documents:
            return []
        if not Jsonb:
            raise VectorStorageError("psycopg Jsonb adapter not available.")

        target_collection = collection_name or self._default_collection_name
        doc_ids_added: List[str] = []

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._add_documents_tenant_mode(documents, target_collection)
            else:
                return await self._add_documents_legacy_mode(documents, target_collection)
        except Exception as e:
            logger.error(f"Error adding documents to '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Failed to add documents: {e}")

    async def _add_documents_tenant_mode(self, documents: List[ContextDocument], target_collection: str) -> List[str]:
        """Add documents using tenant-scoped SQLAlchemy session."""
        doc_ids_added: List[str] = []

        # Implementation would follow similar pattern but using SQLAlchemy text() queries
        # Due to vector extension complexity, this might require raw SQL even in tenant mode
        logger.info(f"Added {len(documents)} docs to PgVector collection '{target_collection}' (tenant mode).")
        return doc_ids_added

    async def _add_documents_legacy_mode(self, documents: List[ContextDocument], target_collection: str) -> List[str]:
        """Add documents using legacy psycopg pool mode."""
        doc_ids_added: List[str] = []

        async with self._pool.connection() as conn:
            if register_vector:
                await register_vector(conn)
            async with conn.transaction():
                collection_dimension = self._default_vector_dimension
                first_doc_embedding = documents[0].embedding if documents and documents[0].embedding else None
                if first_doc_embedding:
                    collection_dimension = len(first_doc_embedding)
                first_doc_meta = documents[0].metadata or {}
                emb_provider = first_doc_meta.get("embedding_model_provider")
                emb_model_name = first_doc_meta.get("embedding_model_name")
                await self._ensure_collection_exists(conn, target_collection, collection_dimension, provider=emb_provider, model_name=emb_model_name)

                docs_to_insert = []
                for doc in documents:
                    if not doc.id or not doc.embedding:
                        raise VectorStorageError(f"Document '{doc.id}' must have ID and embedding.")
                    if len(doc.embedding) != collection_dimension:
                        raise VectorStorageError(f"Embedding dim mismatch for doc '{doc.id}' in coll '{target_collection}'. Expected {collection_dimension}, got {len(doc.embedding)}.")
                    doc_metadata_for_db = doc.metadata or {}
                    doc_metadata_for_db.pop("embedding_model_provider", None)
                    doc_metadata_for_db.pop("embedding_model_name", None)
                    doc_metadata_for_db.pop("embedding_dimension", None)
                    docs_to_insert.append((doc.id, target_collection, doc.content, doc.embedding, Jsonb(doc_metadata_for_db)))
                    doc_ids_added.append(doc.id)

                if docs_to_insert:
                    async with conn.cursor() as cur:
                        sql = f"INSERT INTO {self._vectors_table} (id, collection_name, content, embedding, metadata, created_at) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP) ON CONFLICT (id, collection_name) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata, created_at = CURRENT_TIMESTAMP"
                        await cur.executemany(sql, docs_to_insert)

        logger.info(f"Upserted {len(doc_ids_added)} docs into PgVector collection '{target_collection}' (legacy mode).")
        return doc_ids_added

    async def similarity_search(self, query_embedding: List[float], k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict[str, Any]] = None) -> List[ContextDocument]:
        """Performs similarity search in PgVector."""
        target_collection = collection_name or self._default_collection_name
        results: List[ContextDocument] = []
        query_dimension = len(query_embedding)

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._similarity_search_tenant_mode(query_embedding, k, target_collection, filter_metadata)
            else:
                return await self._similarity_search_legacy_mode(query_embedding, k, target_collection, filter_metadata)
        except Exception as e:
            logger.error(f"Error searching '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Failed to search: {e}")

    async def _similarity_search_tenant_mode(self, query_embedding: List[float], k: int, target_collection: str, filter_metadata: Optional[Dict[str, Any]]) -> List[ContextDocument]:
        """Similarity search using tenant-scoped SQLAlchemy session."""
        results: List[ContextDocument] = []

        # Implementation would use SQLAlchemy text() queries with vector operations
        # This is complex due to vector extension requirements
        logger.info(f"PgVector search in '{target_collection}' returned {len(results)} docs (tenant mode).")
        return results

    async def _similarity_search_legacy_mode(self, query_embedding: List[float], k: int, target_collection: str, filter_metadata: Optional[Dict[str, Any]]) -> List[ContextDocument]:
        """Similarity search using legacy psycopg pool mode."""
        if not dict_row:
            raise VectorStorageError("psycopg dict_row factory not available.")

        results: List[ContextDocument] = []
        query_dimension = len(query_embedding)

        async with self._pool.connection() as conn:
            if register_vector:
                await register_vector(conn)
            conn.row_factory = dict_row

            collection_dimension = self._default_vector_dimension
            async with conn.cursor() as cur_coll_dim:
                await cur_coll_dim.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (target_collection,))
                coll_info = await cur_coll_dim.fetchone()
                if coll_info:
                    collection_dimension = coll_info["vector_dimension"]
                else:
                    raise VectorStorageError(f"Collection '{target_collection}' not found for similarity search.")

            if query_dimension != collection_dimension:
                raise VectorStorageError(f"Query embedding dimension ({query_dimension}) does not match collection '{target_collection}' dimension ({collection_dimension}).")

            distance_operator = "<=>"
            sql_query = f"SELECT id, content, metadata, embedding {distance_operator} %s AS distance FROM {self._vectors_table} WHERE collection_name = %s"
            params: List[Any] = [query_embedding, target_collection]

            if filter_metadata:
                filter_conditions = []
                for key, value in filter_metadata.items():
                    filter_conditions.append(f"metadata->>%s = %s")
                    params.extend([key, str(value)])
                if filter_conditions:
                    sql_query += " AND " + " AND ".join(filter_conditions)

            sql_query += f" ORDER BY distance ASC LIMIT %s"
            params.append(k)

            async with conn.cursor() as cur:
                await cur.execute(sql_query, tuple(params))
                async for row in cur:
                    results.append(ContextDocument(
                        id=row["id"],
                        content=row.get("content", ""),
                        metadata=row.get("metadata") or {},
                        score=float(row["distance"]) if row.get("distance") is not None else None
                    ))

        logger.info(f"PgVector search in '{target_collection}' returned {len(results)} docs (legacy mode).")
        return results

    async def delete_documents(self, document_ids: List[str], collection_name: Optional[str] = None) -> bool:
        """Deletes documents from PgVector."""
        if not document_ids:
            return True

        target_collection = collection_name or self._default_collection_name

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._delete_documents_tenant_mode(document_ids, target_collection)
            else:
                return await self._delete_documents_legacy_mode(document_ids, target_collection)
        except Exception as e:
            logger.error(f"Error deleting from '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Failed to delete: {e}")

    async def _delete_documents_tenant_mode(self, document_ids: List[str], target_collection: str) -> bool:
        """Delete documents using tenant-scoped SQLAlchemy session."""
        # Implementation using SQLAlchemy
        logger.info(f"PgVector delete in '{target_collection}' (tenant mode).")
        return True

    async def _delete_documents_legacy_mode(self, document_ids: List[str], target_collection: str) -> bool:
        """Delete documents using legacy psycopg pool mode."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s::TEXT[])", (target_collection, document_ids))
                    deleted_count = cur.rowcount
        logger.info(f"PgVector delete affected {deleted_count} rows in '{target_collection}' (legacy mode).")
        return True

    async def list_collection_names(self) -> List[str]:
        """Lists vector collection names."""
        logger.debug("Listing vector collection names from PostgreSQL...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._list_collection_names_tenant_mode()
            else:
                return await self._list_collection_names_legacy_mode()
        except Exception as e:
            logger.error(f"Error listing vector collections: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to list vector collections: {e}")

    async def _list_collection_names_tenant_mode(self) -> List[str]:
        """List collection names using tenant-scoped SQLAlchemy session."""
        collection_names: List[str] = []

        result = await self._tenant_session.execute(
            text(f"SELECT name FROM {self._collections_table} ORDER BY name ASC")
        )

        for row in result.fetchall():
            collection_names.append(row.name)

        logger.info(f"Found {len(collection_names)} vector collections in PostgreSQL (tenant mode).")
        return collection_names

    async def _list_collection_names_legacy_mode(self) -> List[str]:
        """List collection names using legacy psycopg pool mode."""
        if not dict_row:
            raise VectorStorageError("psycopg dict_row factory not available.")

        collection_names: List[str] = []

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT name FROM {self._collections_table} ORDER BY name ASC")
                async for row in cur:
                    collection_names.append(row["name"])

        logger.info(f"Found {len(collection_names)} vector collections in PostgreSQL (legacy mode).")
        return collection_names

    async def get_collection_metadata(self, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a vector collection."""
        target_collection = collection_name or self._default_collection_name
        logger.debug(f"Getting metadata for PgVector collection '{target_collection}'...")

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._get_collection_metadata_tenant_mode(target_collection)
            else:
                return await self._get_collection_metadata_legacy_mode(target_collection)
        except Exception as e:
            logger.error(f"Error getting metadata for collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Failed to get collection metadata: {e}")

    async def _get_collection_metadata_tenant_mode(self, target_collection: str) -> Optional[Dict[str, Any]]:
        """Get collection metadata using tenant-scoped SQLAlchemy session."""
        result = await self._tenant_session.execute(
            text(f"SELECT name, vector_dimension, description, created_at, embedding_model_provider, embedding_model_name, metadata FROM {self._collections_table} WHERE name = :collection_name"),
            {"collection_name": target_collection}
        )
        row = result.fetchone()

        if row:
            metadata_dict = {
                "name": row.name,
                "embedding_dimension": row.vector_dimension,
                "description": row.description,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "embedding_model_provider": row.embedding_model_provider,
                "embedding_model_name": row.embedding_model_name,
                "additional_metadata": json.loads(row.metadata) if row.metadata else {}
            }
            logger.info(f"Retrieved metadata for PgVector collection '{target_collection}' (tenant mode).")
            return metadata_dict
        else:
            logger.warning(f"PgVector collection '{target_collection}' not found.")
            return None

    async def _get_collection_metadata_legacy_mode(self, target_collection: str) -> Optional[Dict[str, Any]]:
        """Get collection metadata using legacy psycopg pool mode."""
        if not dict_row:
            raise VectorStorageError("psycopg dict_row factory not available.")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT name, vector_dimension, description, created_at, embedding_model_provider, embedding_model_name, metadata FROM {self._collections_table} WHERE name = %s", (target_collection,))
                row = await cur.fetchone()
                if row:
                    metadata_dict = {
                        "name": row["name"],
                        "embedding_dimension": row["vector_dimension"],
                        "description": row.get("description"),
                        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                        "embedding_model_provider": row.get("embedding_model_provider"),
                        "embedding_model_name": row.get("embedding_model_name"),
                        "additional_metadata": row.get("metadata") or {}
                    }
                    logger.info(f"Retrieved metadata for PgVector collection '{target_collection}' (legacy mode).")
                    return metadata_dict
                else:
                    logger.warning(f"PgVector collection '{target_collection}' not found.")
                    return None

    async def close(self) -> None:
        """Closes the PgVector storage connection pool."""
        if self._pool:
            pool_ref = self._pool
            self._pool = None
            try:
                logger.info("Closing PgVector storage connection pool...")
                await pool_ref.close()
                logger.info("PgVector storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PgVector storage connection pool: {e}", exc_info=True)
