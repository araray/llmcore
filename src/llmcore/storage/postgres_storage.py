# src/llmcore/storage/postgres_storage.py
"""
PostgreSQL storage implementation for LLMCore.

This module provides:
- PostgresSessionStorage: For storing chat sessions, messages, and context_items.
- PgVectorStorage: For storing document embeddings using the pgvector extension.

Requires `psycopg` (for async PostgreSQL interaction) and `pgvector` (for vector operations).
Ensure the pgvector extension is enabled in your PostgreSQL database: `CREATE EXTENSION IF NOT EXISTS vector;`
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        psycopg_available = True
    except ImportError:
        psycopg = None # type: ignore
        dict_row = None # type: ignore
        Jsonb = None # type: ignore
        AsyncConnectionPool = None # type: ignore
        PsycopgAsyncConnectionType = Any
        psycopg_available = False
else:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PsycopgAsyncConnectionType = Any
        psycopg_available = False

try:
    from pgvector.psycopg import register_vector # type: ignore
    pgvector_available = True
except ImportError:
    pgvector_available = False
    register_vector = None # type: ignore

from ..models import ChatSession, Message, Role, ContextDocument, ContextItem, ContextItemType # Added ContextItem, ContextItemType
from ..exceptions import SessionStorageError, VectorStorageError, ConfigError
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_TABLE = "llmcore_sessions"
DEFAULT_MESSAGES_TABLE = "llmcore_messages"
DEFAULT_CONTEXT_ITEMS_TABLE = "llmcore_context_items" # New default table name
DEFAULT_VECTORS_TABLE = "llmcore_vectors"
DEFAULT_COLLECTIONS_TABLE = "llmcore_vector_collections"

class PostgresSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a PostgreSQL database
    using asynchronous connections via psycopg and connection pooling.
    Includes storage for messages and context_items.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _sessions_table: str
    _messages_table: str
    _context_items_table: str # New table name attribute

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL session storage asynchronously.
        Sets up connection pool and ensures tables for sessions, messages, and context_items exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'db_url', 'sessions_table_name' (opt), 'messages_table_name' (opt),
                    'context_items_table_name' (opt), 'min_pool_size' (opt), 'max_pool_size' (opt).
        Raises:
            ConfigError, SessionStorageError.
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed. Please install `psycopg[binary]` or `llmcore[postgres]`.")

        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_SESSION_DB_URL")
        if not db_url:
            raise ConfigError("PostgreSQL session storage 'db_url' not specified.")

        self._sessions_table = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        self._context_items_table = config.get("context_items_table_name", DEFAULT_CONTEXT_ITEMS_TABLE) # Get table name from config
        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for session storage (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn: # Test connection
                async with conn.cursor() as cur: await cur.execute("SELECT 1;")
                if not await cur.fetchone(): raise SessionStorageError("DB connection test failed.")
                logger.debug("PostgreSQL connection test successful.")

                logger.debug(f"Ensuring session tables '{self._sessions_table}', '{self._messages_table}', and '{self._context_items_table}' exist...")
                async with conn.transaction():
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._sessions_table} (
                            id TEXT PRIMARY KEY, name TEXT, created_at TIMESTAMPTZ NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL, metadata JSONB
                        )""")
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._messages_table} (
                            id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES {self._sessions_table}(id) ON DELETE CASCADE,
                            role TEXT NOT NULL, content TEXT NOT NULL, timestamp TIMESTAMPTZ NOT NULL,
                            tokens INTEGER, metadata JSONB
                        )""")
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table}_session_timestamp ON {self._messages_table} (session_id, timestamp);")

                    # ContextItems table
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._context_items_table} (
                            id TEXT NOT NULL, -- ID of the context item
                            session_id TEXT NOT NULL REFERENCES {self._sessions_table}(id) ON DELETE CASCADE,
                            item_type TEXT NOT NULL, -- ContextItemType enum value
                            source_id TEXT,
                            content TEXT NOT NULL,
                            tokens INTEGER,
                            metadata JSONB,
                            timestamp TIMESTAMPTZ NOT NULL,
                            PRIMARY KEY (session_id, id) -- Item ID unique within a session
                        )""")
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._context_items_table}_session_id ON {self._context_items_table} (session_id);")

            logger.info(f"PostgreSQL session storage initialized. Tables: '{self._sessions_table}', '{self._messages_table}', '{self._context_items_table}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL session storage: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise SessionStorageError(f"Could not initialize PostgreSQL session storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL session storage initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        """Saves/updates a session, its messages, and context_items to PostgreSQL."""
        if not self._pool: raise SessionStorageError("PostgreSQL connection pool not initialized.")
        if not Jsonb: raise SessionStorageError("psycopg Jsonb adapter not available.")

        logger.debug(f"Saving session '{session.id}' with {len(session.messages)} messages and {len(session.context_items)} context items to PostgreSQL...")
        try:
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
                                          msg.tokens, Jsonb(msg.metadata or {})) for msg in session.messages]
                        async with conn.cursor() as cur:
                            await cur.executemany(f"INSERT INTO {self._messages_table} (id, session_id, role, content, timestamp, tokens, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)", messages_data)

                    await conn.execute(f"DELETE FROM {self._context_items_table} WHERE session_id = %s", (session.id,))
                    if session.context_items:
                        context_items_data = [(item.id, session.id, str(item.type.value), item.source_id, item.content,
                                               item.tokens, Jsonb(item.metadata or {}), item.timestamp)
                                              for item in session.context_items]
                        async with conn.cursor() as cur:
                            await cur.executemany(f"INSERT INTO {self._context_items_table} (id, session_id, item_type, source_id, content, tokens, metadata, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", context_items_data)
            logger.info(f"Session '{session.id}' saved successfully to PostgreSQL.")
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a session with messages and context_items from PostgreSQL."""
        if not self._pool: raise SessionStorageError("PostgreSQL connection pool not initialized.")
        if not dict_row: raise SessionStorageError("psycopg dict_row factory not available.")

        logger.debug(f"Loading session '{session_id}' from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    await cur.execute(f"SELECT * FROM {self._sessions_table} WHERE id = %s", (session_id,))
                    session_row = await cur.fetchone()
                    if not session_row: logger.debug(f"Session '{session_id}' not found."); return None

                    session_data = dict(session_row)
                    session_data["metadata"] = session_data.get("metadata") or {}

                    # Fetch messages
                    messages: List[Message] = []
                    await cur.execute(f"SELECT * FROM {self._messages_table} WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
                    async for msg_row_data in cur:
                        msg_dict = dict(msg_row_data)
                        try:
                            msg_dict["metadata"] = msg_dict.get("metadata") or {}
                            msg_dict["role"] = Role(msg_dict["role"])
                            # TIMESTAMPTZ from postgres is already timezone-aware
                            messages.append(Message.model_validate(msg_dict))
                        except (ValueError, TypeError) as e: logger.warning(f"Skipping invalid message {msg_dict.get('id')} in session {session_id}: {e}")
                    session_data["messages"] = messages

                    # Fetch context_items
                    context_items: List[ContextItem] = []
                    await cur.execute(f"SELECT * FROM {self._context_items_table} WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
                    async for item_row_data in cur:
                        item_dict = dict(item_row_data)
                        try:
                            item_dict["metadata"] = item_dict.get("metadata") or {}
                            item_dict["type"] = ContextItemType(item_dict.pop("item_type"))
                            context_items.append(ContextItem.model_validate(item_dict))
                        except (ValueError, TypeError) as e: logger.warning(f"Skipping invalid context_item {item_dict.get('id')} in session {session_id}: {e}")
                    session_data["context_items"] = context_items

            chat_session = ChatSession.model_validate(session_data)
            logger.info(f"Session '{session_id}' loaded from PostgreSQL ({len(messages)} msgs, {len(context_items)} ctx items).")
            return chat_session
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists session metadata from PostgreSQL, including message and context_item counts."""
        if not self._pool: raise SessionStorageError("PostgreSQL connection pool not initialized.")
        if not dict_row: raise SessionStorageError("psycopg dict_row factory not available.")

        session_metadata_list: List[Dict[str, Any]] = []
        logger.debug("Listing session metadata from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    await cur.execute(f"""
                        SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata,
                               (SELECT COUNT(*) FROM {self._messages_table} m WHERE m.session_id = s.id) as message_count,
                               (SELECT COUNT(*) FROM {self._context_items_table} ci WHERE ci.session_id = s.id) as context_item_count
                        FROM {self._sessions_table} s
                        ORDER BY s.updated_at DESC
                    """)
                    async for row in cur:
                        data = dict(row)
                        data["metadata"] = data.get("metadata") or {}
                        session_metadata_list.append(data)
            logger.info(f"Found {len(session_metadata_list)} sessions in PostgreSQL.")
            return session_metadata_list
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session and its associated messages/context_items from PostgreSQL."""
        if not self._pool: raise SessionStorageError("PostgreSQL connection pool not initialized.")
        logger.debug(f"Deleting session '{session_id}' from PostgreSQL...")
        try:
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
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """Closes the PostgreSQL connection pool."""
        if self._pool:
            pool_ref = self._pool; self._pool = None
            try:
                logger.info("Closing PostgreSQL session storage connection pool...")
                await pool_ref.close()
                logger.info("PostgreSQL session storage connection pool closed.")
            except Exception as e: logger.error(f"Error closing PostgreSQL pool: {e}", exc_info=True)


class PgVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using PostgreSQL
    with the pgvector extension. Requires asynchronous connections via psycopg.
    (Implementation remains largely the same as previous version, ensure it's compatible
     with the shared pool if db_url is the same as session storage, or uses its own pool).
    For simplicity, this example assumes it manages its own pool based on its config.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _vectors_table: str
    _collections_table: str
    _default_collection_name: str = "llmcore_default_rag"
    _default_vector_dimension: int = 384

    async def initialize(self, config: Dict[str, Any]) -> None:
        if not psycopg_available: raise ConfigError("psycopg library not installed for PgVector.")
        if not pgvector_available: raise ConfigError("pgvector library not installed for PgVector.")

        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url: raise ConfigError("PgVector storage 'db_url' not specified.")

        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", self._default_collection_name)
        self._default_vector_dimension = config.get("default_vector_dimension", 384)
        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for PgVector (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn)
                else: raise VectorStorageError("pgvector register_vector not available.")

                async with conn.transaction():
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._collections_table} (
                            id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, vector_dimension INTEGER NOT NULL,
                            description TEXT, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);""")
                    await self._ensure_collection_exists(conn, self._default_collection_name, self._default_vector_dimension)
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                            id TEXT NOT NULL, collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE,
                            content TEXT, embedding VECTOR, metadata JSONB, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (id, collection_name));""")
                    index_name = f"idx_embedding_cosine_{self._vectors_table}"
                    # Note: Index creation on vector columns depends on the pgvector version and desired index type.
                    # The following is a generic example. Specific tuning might be needed.
                    # Example for HNSW index (pgvector 0.5.0+):
                    # await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {self._vectors_table} USING hnsw (embedding vector_cosine_ops);")
                    # Example for IVFFlat index (older versions or specific needs):
                    # await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {self._vectors_table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
                    # For simplicity, we'll use a basic GIN index if HNSW/IVFFlat syntax is complex or version-dependent here.
                    # Or, recommend manual index creation. For now, let's use a common cosine GIN index example.
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._vectors_table} USING gin (embedding vector_cosine_ops);
                    """)


            logger.info(f"PgVector storage initialized. Tables: '{self._vectors_table}', '{self._collections_table}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PgVector storage: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise VectorStorageError(f"Could not initialize PgVector: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PgVector initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise VectorStorageError(f"Unexpected PgVector init error: {e}")

    async def _ensure_collection_exists(self, conn: PsycopgAsyncConnectionType, name: str, dimension: int, description: Optional[str] = None) -> None:
        logger.debug(f"Ensuring vector collection '{name}' (dim: {dimension}) exists...")
        async with conn.cursor() as cur:
            await cur.execute(f"INSERT INTO {self._collections_table} (name, vector_dimension, description) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING", (name, dimension, description))
            if cur.rowcount > 0: logger.info(f"Created new vector collection '{name}' dim {dimension}.")
            else:
                 await cur.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (name,))
                 existing_info = await cur.fetchone()
                 if existing_info and existing_info[0] != dimension:
                      raise ConfigError(f"Dimension mismatch for collection '{name}'. DB has {existing_info[0]}, requested {dimension}.")

    async def add_documents(self, documents: List[ContextDocument], collection_name: Optional[str] = None) -> List[str]:
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not documents: return []
        if not Jsonb: raise VectorStorageError("psycopg Jsonb adapter not available.")
        target_collection = collection_name or self._default_collection_name
        doc_ids_added: List[str] = []
        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn)
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (target_collection,))
                        coll_info = await cur.fetchone()
                        if not coll_info:
                            first_doc_dim = len(documents[0].embedding) if documents and documents[0].embedding else self._default_vector_dimension
                            await self._ensure_collection_exists(conn, target_collection, first_doc_dim)
                            collection_dimension = first_doc_dim
                        else: collection_dimension = coll_info[0]

                    docs_to_insert = []
                    for doc in documents:
                        if not doc.id or not doc.embedding or len(doc.embedding) != collection_dimension:
                            raise VectorStorageError(f"Invalid doc or embedding dimension for doc '{doc.id}' in collection '{target_collection}'. Expected {collection_dimension}.")
                        docs_to_insert.append((doc.id, target_collection, doc.content, doc.embedding, Jsonb(doc.metadata or {})))
                        doc_ids_added.append(doc.id)

                    if docs_to_insert:
                        async with conn.cursor() as cur:
                            sql = f"INSERT INTO {self._vectors_table} (id, collection_name, content, embedding, metadata) VALUES (%s, %s, %s, %s::vector, %s) ON CONFLICT (id, collection_name) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata, created_at = CURRENT_TIMESTAMP"
                            await cur.executemany(sql, docs_to_insert)
            logger.info(f"Upserted {len(doc_ids_added)} docs into PgVector collection '{target_collection}'.")
            return doc_ids_added
        except psycopg.Error as e: logger.error(f"PgVector DB error adding docs to '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error adding docs: {e}")
        except VectorStorageError: raise
        except Exception as e: logger.error(f"Unexpected error adding docs to '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error adding docs: {e}")

    async def similarity_search(self, query_embedding: List[float], k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict[str, Any]] = None) -> List[ContextDocument]:
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not dict_row: raise VectorStorageError("psycopg dict_row factory not available.")
        target_collection = collection_name or self._default_collection_name
        results: List[ContextDocument] = []
        query_dimension = len(query_embedding)
        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn)
                conn.row_factory = dict_row
                distance_operator = "<=>" # Cosine distance
                sql_query = f"SELECT id, content, metadata, embedding {distance_operator} %s::vector({query_dimension}) AS distance FROM {self._vectors_table} WHERE collection_name = %s"
                params: List[Any] = [query_embedding, target_collection]
                if filter_metadata:
                    filter_clauses = [f"metadata->>%s = %s" for key in filter_metadata.keys()]
                    sql_query += " AND " + " AND ".join(filter_clauses)
                    for key, value in filter_metadata.items(): params.extend([key, str(value)])
                sql_query += f" ORDER BY distance ASC LIMIT %s"
                params.append(k)
                async with conn.cursor() as cur:
                    await cur.execute(sql_query, tuple(params))
                    async for row in cur:
                        results.append(ContextDocument(id=row["id"], content=row.get("content", ""), metadata=row.get("metadata") or {}, score=float(row["distance"]) if row.get("distance") is not None else None))
            logger.info(f"PgVector search in '{target_collection}' returned {len(results)} docs.")
            return results
        except psycopg.Error as e: logger.error(f"PgVector DB error searching '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error searching: {e}")
        except Exception as e: logger.error(f"Unexpected error searching '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error searching: {e}")

    async def delete_documents(self, document_ids: List[str], collection_name: Optional[str] = None) -> bool:
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not document_ids: return True
        target_collection = collection_name or self._default_collection_name
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s)", (target_collection, document_ids))
                        deleted_count = cur.rowcount
            logger.info(f"PgVector delete affected {deleted_count} rows in '{target_collection}'.")
            return True
        except psycopg.Error as e: logger.error(f"PgVector DB error deleting from '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error deleting: {e}")
        except Exception as e: logger.error(f"Unexpected error deleting from '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error deleting: {e}")

    async def close(self) -> None:
        if self._pool:
            pool_ref = self._pool; self._pool = None
            try: logger.info("Closing PgVector storage pool..."); await pool_ref.close(); logger.info("PgVector pool closed.")
            except Exception as e: logger.error(f"Error closing PgVector pool: {e}", exc_info=True)
