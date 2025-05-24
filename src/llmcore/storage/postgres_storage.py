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
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

if TYPE_CHECKING:
    try:
        import psycopg # type: ignore
        from psycopg.abc import \
            AsyncConnection as PsycopgAsyncConnectionType  # type: ignore
        from psycopg.rows import dict_row  # type: ignore
        from psycopg.types.json import Jsonb  # type: ignore
        from psycopg_pool import AsyncConnectionPool  # type: ignore
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
    from pgvector.psycopg import register_vector # type: ignore
    pgvector_available = True
except ImportError:
    pgvector_available = False
    register_vector = None

from ..exceptions import ConfigError, SessionStorageError, VectorStorageError
from ..models import (ChatSession, ContextDocument, ContextItem,
                      ContextItemType, Message, Role)
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_TABLE = "llmcore_sessions"
DEFAULT_MESSAGES_TABLE = "llmcore_messages"
DEFAULT_CONTEXT_ITEMS_TABLE = "llmcore_context_items"
DEFAULT_VECTORS_TABLE = "llmcore_vectors"
DEFAULT_COLLECTIONS_TABLE = "llmcore_vector_collections"

class PostgresSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a PostgreSQL database
    using asynchronous connections via psycopg and connection pooling.
    Includes storage for messages and context_items.
    """
    _pool: Optional["AsyncConnectionPool"] = None
    _sessions_table: str
    _messages_table: str
    _context_items_table: str

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
        self._context_items_table = config.get("context_items_table_name", DEFAULT_CONTEXT_ITEMS_TABLE)
        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for session storage (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur: await cur.execute("SELECT 1;") # type: ignore
                if not await cur.fetchone(): raise SessionStorageError("DB connection test failed.") # type: ignore
                logger.debug("PostgreSQL connection test successful.")

                logger.debug(f"Ensuring session tables '{self._sessions_table}', '{self._messages_table}', and '{self._context_items_table}' exist...")
                async with conn.transaction(): # type: ignore
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

                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._context_items_table} (
                            id TEXT NOT NULL,
                            session_id TEXT NOT NULL REFERENCES {self._sessions_table}(id) ON DELETE CASCADE,
                            item_type TEXT NOT NULL,
                            source_id TEXT,
                            content TEXT NOT NULL,
                            tokens INTEGER,
                            metadata JSONB,
                            timestamp TIMESTAMPTZ NOT NULL,
                            PRIMARY KEY (session_id, id)
                        )""")
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._context_items_table}_session_id ON {self._context_items_table} (session_id);")

            logger.info(f"PostgreSQL session storage initialized. Tables: '{self._sessions_table}', '{self._messages_table}', '{self._context_items_table}'.")
        except psycopg.Error as e: # type: ignore
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
                async with conn.transaction(): # type: ignore
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
                        async with conn.cursor() as cur: # type: ignore
                            await cur.executemany(f"INSERT INTO {self._messages_table} (id, session_id, role, content, timestamp, tokens, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)", messages_data)

                    await conn.execute(f"DELETE FROM {self._context_items_table} WHERE session_id = %s", (session.id,))
                    if session.context_items:
                        context_items_data = [(item.id, session.id, str(item.type),
                                               item.source_id, item.content,
                                               item.tokens, Jsonb(item.metadata or {}), item.timestamp)
                                              for item in session.context_items]
                        async with conn.cursor() as cur: # type: ignore
                            await cur.executemany(f"INSERT INTO {self._context_items_table} (id, session_id, item_type, source_id, content, tokens, metadata, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", context_items_data)
            logger.info(f"Session '{session.id}' saved successfully to PostgreSQL.")
        except psycopg.Error as e: # type: ignore
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
                conn.row_factory = dict_row # type: ignore
                async with conn.cursor() as cur: # type: ignore
                    await cur.execute(f"SELECT * FROM {self._sessions_table} WHERE id = %s", (session_id,))
                    session_row = await cur.fetchone()
                    if not session_row: logger.debug(f"Session '{session_id}' not found."); return None

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
                        except (ValueError, TypeError) as e: logger.warning(f"Skipping invalid message {msg_dict.get('id')} in session {session_id}: {e}")
                    session_data["messages"] = messages

                    context_items: List[ContextItem] = []
                    await cur.execute(f"SELECT * FROM {self._context_items_table} WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
                    async for item_row_data in cur:
                        item_dict = dict(item_row_data)
                        try:
                            item_dict["metadata"] = item_dict.get("metadata") or {}
                            item_dict["type"] = ContextItemType(item_dict.pop("item_type"))
                            item_dict["timestamp"] = item_dict["timestamp"].replace(tzinfo=timezone.utc) if item_dict.get("timestamp") else datetime.now(timezone.utc)
                            context_items.append(ContextItem.model_validate(item_dict))
                        except (ValueError, TypeError) as e: logger.warning(f"Skipping invalid context_item {item_dict.get('id')} in session {session_id}: {e}")
                    session_data["context_items"] = context_items

            chat_session = ChatSession.model_validate(session_data)
            logger.info(f"Session '{session_id}' loaded from PostgreSQL ({len(messages)} msgs, {len(context_items)} ctx items).")
            return chat_session
        except psycopg.Error as e: # type: ignore
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
                conn.row_factory = dict_row # type: ignore
                async with conn.cursor() as cur: # type: ignore
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
        except psycopg.Error as e: # type: ignore
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
                async with conn.transaction(): # type: ignore # Ensures atomicity
                    async with conn.cursor() as cur: # type: ignore
                        # CASCADE DELETE on foreign keys handles messages and context_items
                        await cur.execute(f"DELETE FROM {self._sessions_table} WHERE id = %s", (session_id,))
                        deleted_count = cur.rowcount
            if deleted_count > 0:
                logger.info(f"Session '{session_id}' and associated data deleted from PostgreSQL.")
                return True
            logger.warning(f"Attempted to delete session '{session_id}', but it was not found.")
            return False
        except psycopg.Error as e: # type: ignore
            logger.error(f"PostgreSQL error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """Closes the PostgreSQL connection pool."""
        if self._pool:
            pool_ref = self._pool; self._pool = None # type: ignore
            try:
                logger.info("Closing PostgreSQL session storage connection pool...")
                await pool_ref.close() # type: ignore
                logger.info("PostgreSQL session storage connection pool closed.")
            except Exception as e: logger.error(f"Error closing PostgreSQL pool: {e}", exc_info=True)


class PgVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using PostgreSQL
    with the pgvector extension. Requires asynchronous connections via psycopg.
    """
    _pool: Optional["AsyncConnectionPool"] = None
    _vectors_table: str
    _collections_table: str
    _default_collection_name: str = "llmcore_default_rag"
    _default_vector_dimension: int = 384 # Default, e.g., for all-MiniLM-L6-v2

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize PgVector storage: set up pool, ensure pgvector extension and tables.
        Args:
            config: Keys: 'db_url', 'vectors_table_name' (opt), 'collections_table_name' (opt),
                    'default_collection' (opt), 'default_vector_dimension' (opt), pool sizes.
        """
        if not psycopg_available: raise ConfigError("psycopg library not installed for PgVector.")
        if not pgvector_available: raise ConfigError("pgvector library not installed for PgVector.")

        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url: raise ConfigError("PgVector storage 'db_url' not specified.")

        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", self._default_collection_name)
        self._default_vector_dimension = int(config.get("default_vector_dimension", 384))
        min_pool_size = int(config.get("min_pool_size", 2))
        max_pool_size = int(config.get("max_pool_size", 10))

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for PgVector (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn) # type: ignore
                else: raise VectorStorageError("pgvector register_vector not available.")

                async with conn.transaction(): # type: ignore
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    # Updated collections table to store more metadata, though not fully used by get_collection_metadata yet
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._collections_table} (
                            id SERIAL PRIMARY KEY,
                            name TEXT UNIQUE NOT NULL,
                            vector_dimension INTEGER NOT NULL,
                            description TEXT,
                            embedding_model_provider TEXT,
                            embedding_model_name TEXT,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB DEFAULT '{{}}'::jsonb
                        );""")
                    # Ensure default collection exists with the default dimension
                    await self._ensure_collection_exists(conn, self._default_collection_name, self._default_vector_dimension)

                    # Vectors table schema needs to refer to the collection's actual dimension
                    # This is tricky if dimension is dynamic per collection. For now, use default.
                    # A better approach might be to create tables per collection or use JSONB for embeddings
                    # if dimensions vary wildly and a single table is used.
                    # For simplicity, assuming a common dimension or that collections are pre-configured.
                    # The CREATE TABLE for vectors_table should ideally use the specific collection's dimension.
                    # This example uses a fixed dimension from default_vector_dimension for the table schema.
                    # If different dimensions are needed per collection, this schema needs adjustment
                    # or multiple tables, or a schema-less vector column type if supported.
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                            id TEXT NOT NULL,
                            collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE,
                            content TEXT,
                            embedding VECTOR({self._default_vector_dimension}), -- Dimension set at table creation
                            metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (id, collection_name));""") # Composite primary key

                    # Example HNSW index for cosine distance. Adjust as needed.
                    # The dimension in the index should match the table's embedding column.
                    index_name = f"idx_embedding_hnsw_cosine_{self._vectors_table}"
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._vectors_table} USING hnsw (embedding vector_cosine_ops);
                    """)
            logger.info(f"PgVector storage initialized. Tables: '{self._vectors_table}', '{self._collections_table}'. Default dimension for new tables: {self._default_vector_dimension}")
        except psycopg.Error as e: # type: ignore
            logger.error(f"Failed to initialize PgVector storage: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise VectorStorageError(f"Could not initialize PgVector: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PgVector initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None; raise VectorStorageError(f"Unexpected PgVector init error: {e}")

    async def _ensure_collection_exists(self, conn: "PsycopgAsyncConnectionType", name: str, dimension: int, description: Optional[str] = None, provider: Optional[str] = None, model_name: Optional[str] = None, collection_meta: Optional[Dict[str,Any]] = None) -> None:
        """Ensures a collection record exists, creating or verifying its dimension."""
        logger.debug(f"Ensuring vector collection '{name}' (dim: {dimension}) exists...")
        async with conn.cursor(row_factory=dict_row) as cur: # type: ignore
            await cur.execute(f"SELECT vector_dimension, metadata FROM {self._collections_table} WHERE name = %s", (name,))
            existing_coll = await cur.fetchone()
            if existing_coll:
                if existing_coll["vector_dimension"] != dimension:
                    raise ConfigError(f"Dimension mismatch for collection '{name}'. DB has {existing_coll['vector_dimension']}, operation requires {dimension}.")
                # Optionally update description or other metadata if provided
                # For now, we don't update existing collection's description/provider/model_name via _ensure_collection_exists
                # This method primarily ensures it exists with the correct dimension.
                # A separate method would be needed to update collection metadata like provider/model.
            else:
                # Collection doesn't exist, create it
                final_collection_meta = collection_meta or {}
                # Add specific embedding info if provided, otherwise they remain NULL
                await cur.execute(f"""
                    INSERT INTO {self._collections_table}
                    (name, vector_dimension, description, embedding_model_provider, embedding_model_name, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (name, dimension, description, provider, model_name, Jsonb(final_collection_meta)))
                logger.info(f"Created new vector collection '{name}' with dimension {dimension}.")

    async def add_documents(self, documents: List[ContextDocument], collection_name: Optional[str] = None) -> List[str]:
        """Adds or updates documents in the specified PgVector collection."""
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not documents: return []
        if not Jsonb: raise VectorStorageError("psycopg Jsonb adapter not available.")

        target_collection = collection_name or self._default_collection_name
        doc_ids_added: List[str] = []

        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn) # type: ignore
                async with conn.transaction(): # type: ignore
                    # Determine collection dimension, create collection if it doesn't exist
                    collection_dimension = self._default_vector_dimension
                    first_doc_embedding = documents[0].embedding if documents and documents[0].embedding else None
                    if first_doc_embedding:
                        collection_dimension = len(first_doc_embedding)

                    # Extract embedding model info from the first document's metadata if available (e.g., from apykatu)
                    # This is a convention for this method.
                    first_doc_meta = documents[0].metadata or {}
                    emb_provider = first_doc_meta.get("embedding_model_provider")
                    emb_model_name = first_doc_meta.get("embedding_model_name")

                    await self._ensure_collection_exists(conn, target_collection, collection_dimension, provider=emb_provider, model_name=emb_model_name)

                    docs_to_insert = []
                    for doc in documents:
                        if not doc.id or not doc.embedding:
                            raise VectorStorageError(f"Document '{doc.id}' must have an ID and an embedding.")
                        if len(doc.embedding) != collection_dimension:
                            raise VectorStorageError(f"Embedding dimension mismatch for doc '{doc.id}' in collection '{target_collection}'. Expected {collection_dimension}, got {len(doc.embedding)}.")
                        docs_to_insert.append((doc.id, target_collection, doc.content, doc.embedding, Jsonb(doc.metadata or {})))
                        doc_ids_added.append(doc.id)

                    if docs_to_insert:
                        async with conn.cursor() as cur: # type: ignore
                            # Use ON CONFLICT to perform an UPSERT
                            sql = f"""
                                INSERT INTO {self._vectors_table} (id, collection_name, content, embedding, metadata, created_at)
                                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                                ON CONFLICT (id, collection_name) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    embedding = EXCLUDED.embedding,
                                    metadata = EXCLUDED.metadata,
                                    created_at = CURRENT_TIMESTAMP
                            """
                            await cur.executemany(sql, docs_to_insert)
            logger.info(f"Upserted {len(doc_ids_added)} docs into PgVector collection '{target_collection}'.")
            return doc_ids_added
        except psycopg.Error as e: logger.error(f"PgVector DB error adding docs to '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error adding docs: {e}") # type: ignore
        except VectorStorageError: raise # Re-raise specific errors
        except Exception as e: logger.error(f"Unexpected error adding docs to '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error adding docs: {e}")

    async def similarity_search(self, query_embedding: List[float], k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict[str, Any]] = None) -> List[ContextDocument]:
        """Performs similarity search in PgVector."""
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not dict_row: raise VectorStorageError("psycopg dict_row factory not available.")
        target_collection = collection_name or self._default_collection_name
        results: List[ContextDocument] = []
        query_dimension = len(query_embedding)
        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn) # type: ignore
                conn.row_factory = dict_row # type: ignore

                # Verify collection dimension against query embedding dimension
                collection_dimension = self._default_vector_dimension # Fallback
                async with conn.cursor() as cur_coll_dim: # type: ignore
                    await cur_coll_dim.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (target_collection,))
                    coll_info = await cur_coll_dim.fetchone()
                    if coll_info: collection_dimension = coll_info["vector_dimension"]
                    else: raise VectorStorageError(f"Collection '{target_collection}' not found for similarity search.")

                if query_dimension != collection_dimension:
                    raise VectorStorageError(f"Query embedding dimension ({query_dimension}) does not match collection '{target_collection}' dimension ({collection_dimension}).")

                # Using vector_cosine_ops for cosine distance (smaller is better)
                # Other options: vector_l2_ops (Euclidean), vector_ip_ops (inner product, larger is better)
                # For inner product, order by "embedding <#> %s DESC"
                distance_operator = "<=>" # Cosine distance

                sql_query = f"SELECT id, content, metadata, embedding {distance_operator} %s AS distance FROM {self._vectors_table} WHERE collection_name = %s"
                params: List[Any] = [query_embedding, target_collection]

                if filter_metadata:
                    filter_conditions = []
                    # Note: JSONB filter syntax for specific keys. Example: metadata->>'key' = 'value'
                    for key, value in filter_metadata.items():
                        # Ensure value is string for text comparison in JSONB, adjust if other types needed
                        filter_conditions.append(f"metadata->>%s = %s")
                        params.extend([key, str(value)])
                    if filter_conditions:
                        sql_query += " AND " + " AND ".join(filter_conditions)

                sql_query += f" ORDER BY distance ASC LIMIT %s" # ASC for cosine distance
                params.append(k)

                async with conn.cursor() as cur: # type: ignore
                    await cur.execute(sql_query, tuple(params))
                    async for row in cur:
                        results.append(ContextDocument(
                            id=row["id"],
                            content=row.get("content", ""),
                            metadata=row.get("metadata") or {},
                            score=float(row["distance"]) if row.get("distance") is not None else None # Distance as score
                        ))
            logger.info(f"PgVector search in '{target_collection}' returned {len(results)} docs.")
            return results
        except psycopg.Error as e: logger.error(f"PgVector DB error searching '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error searching: {e}") # type: ignore
        except Exception as e: logger.error(f"Unexpected error searching '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error searching: {e}")

    async def delete_documents(self, document_ids: List[str], collection_name: Optional[str] = None) -> bool:
        """Deletes documents from PgVector by their IDs."""
        if not self._pool: raise VectorStorageError("PgVector pool not initialized.")
        if not document_ids: return True # Nothing to delete
        target_collection = collection_name or self._default_collection_name
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # type: ignore
                    async with conn.cursor() as cur: # type: ignore
                        # Delete from vectors table where ID is in the list and collection matches
                        await cur.execute(f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s::TEXT[])", (target_collection, document_ids))
                        deleted_count = cur.rowcount
            logger.info(f"PgVector delete affected {deleted_count} rows in '{target_collection}'.")
            return True # Operation attempted, success determined by rowcount or lack of error
        except psycopg.Error as e: logger.error(f"PgVector DB error deleting from '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"DB error deleting: {e}") # type: ignore
        except Exception as e: logger.error(f"Unexpected error deleting from '{target_collection}': {e}", exc_info=True); raise VectorStorageError(f"Unexpected error deleting: {e}")

    async def list_collection_names(self) -> List[str]:
        """Lists the names of all available vector collections from PostgreSQL."""
        if not self._pool:
            raise VectorStorageError("PgVector connection pool not initialized.")
        if not dict_row: # Ensure dict_row is available
            raise VectorStorageError("psycopg dict_row factory not available.")

        collection_names: List[str] = []
        logger.debug("Listing vector collection names from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row # type: ignore
                async with conn.cursor() as cur: # type: ignore
                    # Query the dedicated collections table
                    await cur.execute(f"SELECT name FROM {self._collections_table} ORDER BY name ASC")
                    async for row in cur:
                        collection_names.append(row["name"])
            logger.info(f"Found {len(collection_names)} vector collections in PostgreSQL.")
            return collection_names
        except psycopg.Error as e: # type: ignore
            logger.error(f"PostgreSQL error listing vector collections: {e}", exc_info=True)
            raise VectorStorageError(f"Database error listing vector collections: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing vector collections: {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error listing vector collections: {e}")

    async def get_collection_metadata(self, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for a specific vector collection from PostgreSQL.
        Currently returns name, vector_dimension (as embedding_dimension), description, and created_at.
        Does NOT return embedding_model_provider or embedding_model_name as they are not
        part of the current _collections_table schema in a queryable way for this purpose.
        """
        if not self._pool:
            raise VectorStorageError("PgVector connection pool not initialized.")
        if not dict_row:
            raise VectorStorageError("psycopg dict_row factory not available.")

        target_collection = collection_name or self._default_collection_name
        logger.debug(f"Getting metadata for PgVector collection '{target_collection}'...")

        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row # type: ignore
                async with conn.cursor() as cur: # type: ignore
                    # Query the collections table for the specified name
                    await cur.execute(f"""
                        SELECT name, vector_dimension, description, created_at,
                               embedding_model_provider, embedding_model_name, metadata
                        FROM {self._collections_table} WHERE name = %s
                    """, (target_collection,))
                    row = await cur.fetchone()

                    if row:
                        # Construct metadata dictionary from available fields
                        # Map vector_dimension to embedding_dimension for consistency
                        # with what ContextManager might expect.
                        metadata_dict = {
                            "name": row["name"],
                            "embedding_dimension": row["vector_dimension"],
                            "description": row.get("description"),
                            "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                            "embedding_model_provider": row.get("embedding_model_provider"),
                            "embedding_model_name": row.get("embedding_model_name"),
                            "additional_metadata": row.get("metadata") or {} # Stored JSONB metadata
                        }
                        # Filter out None values for cleaner output if desired, but often better to include them
                        # metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}
                        logger.info(f"Retrieved metadata for PgVector collection '{target_collection}': {metadata_dict}")
                        return metadata_dict
                    else:
                        logger.warning(f"PgVector collection '{target_collection}' not found in {self._collections_table}.")
                        return None
        except psycopg.Error as e: # type: ignore
            logger.error(f"PostgreSQL error getting metadata for collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error getting collection metadata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting metadata for collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error getting collection metadata: {e}")


    async def close(self) -> None:
        """Closes the PgVector storage connection pool."""
        if self._pool:
            pool_ref = self._pool; self._pool = None # type: ignore
            try:
                logger.info("Closing PgVector storage connection pool...")
                await pool_ref.close() # type: ignore
                logger.info("PgVector storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PgVector storage connection pool: {e}", exc_info=True)
