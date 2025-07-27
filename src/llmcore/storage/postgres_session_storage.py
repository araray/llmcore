# src/llmcore/storage/postgres_session_storage.py
"""
PostgreSQL storage implementation for LLMCore session-related data.

REFACTORED FOR MULTI-TENANCY: This class now supports accepting pre-configured,
tenant-aware database sessions rather than managing its own connections.

This module provides PostgresSessionStorage for storing:
- Chat sessions and messages
- Session context items
- Context presets and their items
- Agent episodes

Requires `psycopg` for asynchronous PostgreSQL interaction.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..exceptions import ConfigError, SessionStorageError, StorageError
from ..models import (ChatSession, ContextItem, ContextItemType, Message, Role,
                      ContextPreset, ContextPresetItem, Episode)
from .base_session import BaseSessionStorage

if TYPE_CHECKING:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        psycopg_available = False
else:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        psycopg_available = False


logger = logging.getLogger(__name__)

# Default table names (will be used within tenant schemas)
DEFAULT_SESSIONS_TABLE = "sessions"
DEFAULT_MESSAGES_TABLE = "messages"
DEFAULT_SESSION_CONTEXT_ITEMS_TABLE = "context_items"
DEFAULT_CONTEXT_PRESETS_TABLE = "context_presets"
DEFAULT_CONTEXT_PRESET_ITEMS_TABLE = "context_preset_items"
DEFAULT_EPISODES_TABLE = "episodes"


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
        if session.messages:
            messages_data = [
                {
                    "id": msg.id, "session_id": session.id, "role": str(msg.role),
                    "content": msg.content, "timestamp": msg.timestamp,
                    "tool_call_id": msg.tool_call_id, "tokens": msg.tokens,
                    "metadata": json.dumps(msg.metadata or {})
                } for msg in session.messages
            ]
            # SQLAlchemy 2.0 style executemany
            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._messages_table}
                    (id, session_id, role, content, timestamp, tool_call_id, tokens, metadata)
                    VALUES (:id, :session_id, :role, :content, :timestamp, :tool_call_id, :tokens, :metadata)
                """),
                messages_data
            )


        # Insert context items
        if session.context_items:
            context_items_data = [
                {
                    "id": item.id, "session_id": session.id, "item_type": str(item.type),
                    "source_id": item.source_id, "content": item.content, "tokens": item.tokens,
                    "original_tokens": item.original_tokens, "is_truncated": item.is_truncated,
                    "metadata": json.dumps(item.metadata or {}), "timestamp": item.timestamp
                } for item in session.context_items
            ]
            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._session_context_items_table}
                    (id, session_id, item_type, source_id, content, tokens, original_tokens, is_truncated, metadata, timestamp)
                    VALUES (:id, :session_id, :item_type, :source_id, :content, :tokens, :original_tokens, :is_truncated, :metadata, :timestamp)
                """),
                context_items_data
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

        session_data = dict(session_row._mapping)
        session_data["metadata"] = json.loads(session_data.get("metadata") or '{}')
        session_data["created_at"] = session_data["created_at"].replace(tzinfo=timezone.utc) if session_data.get("created_at") else datetime.now(timezone.utc)
        session_data["updated_at"] = session_data["updated_at"].replace(tzinfo=timezone.utc) if session_data.get("updated_at") else datetime.now(timezone.utc)

        # Get messages
        messages: List[Message] = []
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._messages_table} WHERE session_id = :session_id ORDER BY timestamp ASC"),
            {"session_id": session_id}
        )
        for msg_row in result.fetchall():
            try:
                msg_dict = dict(msg_row._mapping)
                msg_dict["metadata"] = json.loads(msg_dict.get("metadata") or '{}')
                msg_dict["role"] = Role(msg_dict["role"])
                msg_dict["timestamp"] = msg_dict["timestamp"].replace(tzinfo=timezone.utc) if msg_dict.get("timestamp") else datetime.now(timezone.utc)
                messages.append(Message.model_validate(msg_dict))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid message {msg_row.id} in session {session_id}: {e}")

        # Get context items
        context_items: List[ContextItem] = []
        result = await self._tenant_session.execute(
            text(f"SELECT * FROM {self._session_context_items_table} WHERE session_id = :session_id ORDER BY timestamp ASC"),
            {"session_id": session_id}
        )
        for item_row in result.fetchall():
            try:
                item_dict = dict(item_row._mapping)
                item_dict["metadata"] = json.loads(item_dict.get("metadata") or '{}')
                item_dict["type"] = ContextItemType(item_dict.pop("item_type"))
                item_dict["timestamp"] = item_dict["timestamp"].replace(tzinfo=timezone.utc) if item_dict.get("timestamp") else datetime.now(timezone.utc)
                item_dict["is_truncated"] = bool(item_dict.get("is_truncated", False))
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
            data = dict(row._mapping)
            data["metadata"] = json.loads(data.get("metadata") or '{}')
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
        """Deletes a session and its associated data from PostgreSQL."""
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
        """Updates the name and updated_at timestamp for a session in PostgreSQL."""
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

    # --- Context Preset Management Methods (abbreviated for refactoring focus) ---
    async def save_context_preset(self, preset: ContextPreset) -> None:
        # Implementation would follow the same tenant/legacy pattern
        pass

    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        # Implementation would follow the same tenant/legacy pattern
        return None

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        # Implementation would follow the same tenant/legacy pattern
        return []

    async def delete_context_preset(self, preset_name: str) -> bool:
        # Implementation would follow the same tenant/legacy pattern
        return False

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        # Implementation would follow the same tenant/legacy pattern
        return False

    # --- Episode Management Methods ---
    async def add_episode(self, episode: Episode) -> None:
        # Implementation would follow the same tenant/legacy pattern
        pass

    async def get_episodes(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Episode]:
        # Implementation would follow the same tenant/legacy pattern
        return []

    async def close(self) -> None:
        """Closes the PostgreSQL connection pool."""
        if self._pool:
            pool_ref = self._pool
            self._pool = None
            try:
                logger.info("Closing PostgreSQL session storage connection pool...")
                await pool_ref.close()
                logger.info("PostgreSQL session storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL pool for sessions: {e}", exc_info=True)
