# src/llmcore/storage/postgres_storage.py
"""
PostgreSQL storage implementation for LLMCore.

This module provides:
- PostgresSessionStorage: For storing chat sessions and messages.
- PgVectorStorage: For storing document embeddings using the pgvector extension.

Requires `psycopg` (for async PostgreSQL interaction) and `pgvector` (for vector operations).
Ensure the pgvector extension is enabled in your PostgreSQL database: `CREATE EXTENSION IF NOT EXISTS vector;`
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncIterator

# psycopg for PostgreSQL interaction (async version)
try:
    import psycopg
    from psycopg.rows import dict_row # For fetching rows as dictionaries
    from psycopg.types.json import Jsonb # For efficient JSON storage
    from psycopg_pool import AsyncConnectionPool # For managing connections
    # pgvector for vector type (optional, only if using PgVectorStorage)
    try:
        from pgvector.psycopg import register_vector # type: ignore
        pgvector_available = True
    except ImportError:
        pgvector_available = False
        register_vector = None # type: ignore
        logger = logging.getLogger(__name__)
        logger.warning("pgvector library not found. PgVectorStorage will not be fully functional for vector operations.")

    psycopg_available = True
except ImportError:
    psycopg_available = False
    psycopg = None # type: ignore
    dict_row = None # type: ignore
    Jsonb = None # type: ignore
    AsyncConnectionPool = None # type: ignore
    pgvector_available = False
    register_vector = None # type: ignore


from ..models import ChatSession, Message, Role, ContextDocument
from ..exceptions import SessionStorageError, VectorStorageError, ConfigError
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

# Default table names
DEFAULT_SESSIONS_TABLE = "llmcore_sessions"
DEFAULT_MESSAGES_TABLE = "llmcore_messages"
DEFAULT_VECTORS_TABLE = "llmcore_vectors"
DEFAULT_COLLECTIONS_TABLE = "llmcore_vector_collections" # For managing different RAG collections

class PostgresSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in a PostgreSQL database.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _sessions_table: str
    _messages_table: str

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL session storage.

        Args:
            config: Configuration dictionary. Expected keys:
                    'db_url': PostgreSQL connection string (e.g., "postgresql://user:pass@host:port/dbname").
                    'sessions_table_name' (optional): Name for the sessions table.
                    'messages_table_name' (optional): Name for the messages table.
                    'min_pool_size' (optional): Minimum connections in the pool.
                    'max_pool_size' (optional): Maximum connections in the pool.

        Raises:
            ConfigError: If 'db_url' is not provided or psycopg is not installed.
            SessionStorageError: If the database connection or table creation fails.
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed. Please install `psycopg[binary]` or `llmcore[postgres]`.")

        db_url = config.get("db_url")
        if not db_url:
            db_url = os.environ.get("LLMCORE_STORAGE_SESSION_DB_URL") # Check env var as fallback
        if not db_url:
            raise ConfigError("PostgreSQL session storage 'db_url' not specified in configuration or LLMCORE_STORAGE_SESSION_DB_URL env var.")

        self._sessions_table = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            # Create an asynchronous connection pool
            # The pool handles reconnections and connection management.
            self._pool = AsyncConnectionPool(
                conninfo=db_url,
                min_size=min_pool_size,
                max_size=max_pool_size,
                # open=False # Pool starts empty and connects on demand
            )
            # Test connection by acquiring and releasing one
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    if not await cur.fetchone():
                         raise SessionStorageError("Database connection test failed.")

            # Create tables if they don't exist
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use a transaction for schema changes
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._sessions_table} (
                            id TEXT PRIMARY KEY,
                            name TEXT,
                            created_at TIMESTAMPTZ NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL,
                            metadata JSONB
                        )
                    """)
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._messages_table} (
                            id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL REFERENCES {self._sessions_table}(id) ON DELETE CASCADE,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            tokens INTEGER,
                            metadata JSONB
                        )
                    """)
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self._messages_table}_session_timestamp
                        ON {self._messages_table} (session_id, timestamp);
                    """)
            logger.info(f"PostgreSQL session storage initialized. Tables: '{self._sessions_table}', '{self._messages_table}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL session storage: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None
            raise SessionStorageError(f"Could not initialize PostgreSQL session storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL session storage initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")

        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use a transaction
                    # Upsert session metadata
                    await conn.execute(f"""
                        INSERT INTO {self._sessions_table} (id, name, created_at, updated_at, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """, (
                        session.id,
                        session.name,
                        session.created_at,
                        session.updated_at,
                        Jsonb(session.metadata or {})
                    ))

                    # Delete existing messages for this session before inserting current ones
                    await conn.execute(f"DELETE FROM {self._messages_table} WHERE session_id = %s", (session.id,))

                    if session.messages:
                        messages_data = [
                            (
                                msg.id,
                                session.id,
                                str(msg.role),
                                msg.content,
                                msg.timestamp,
                                msg.tokens,
                                Jsonb(msg.metadata or {})
                            ) for msg in session.messages
                        ]
                        # Use copy_to_table for efficient batch insert if psycopg version supports it well,
                        # otherwise use executemany. For simplicity, using executemany here.
                        async with conn.cursor() as cur:
                            await cur.executemany(f"""
                                INSERT INTO {self._messages_table}
                                (id, session_id, role, content, timestamp, tokens, metadata)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, messages_data)
            logger.debug(f"Session '{session.id}' saved to PostgreSQL.")
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")


    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")

        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row # Fetch rows as dictionaries
                async with conn.cursor() as cur:
                    await cur.execute(f"SELECT * FROM {self._sessions_table} WHERE id = %s", (session_id,))
                    session_row = await cur.fetchone()

                    if not session_row:
                        return None

                    # Fetch messages for this session
                    await cur.execute(f"""
                        SELECT id, session_id, role, content, timestamp, tokens, metadata
                        FROM {self._messages_table} WHERE session_id = %s ORDER BY timestamp ASC
                    """, (session_id,))
                    message_rows = await cur.fetchall()

            messages = []
            for msg_row in message_rows:
                try:
                    msg_data = dict(msg_row) # Ensure it's a plain dict
                    msg_data["role"] = Role(msg_data["role"]) # Convert role string to Enum
                    # Timestamps should be timezone-aware from DB (TIMESTAMPTZ)
                    messages.append(Message.model_validate(msg_data))
                except ValueError as ve: # For Role enum or other Pydantic validation
                    logger.warning(f"Invalid data for message {msg_row.get('id')}: {ve}. Skipping.")
                    continue


            # Pydantic handles datetime parsing from ISO strings if they are stored as text
            # but TIMESTAMPTZ should return datetime objects directly.
            chat_session = ChatSession.model_validate({
                "id": session_row["id"],
                "name": session_row["name"],
                "created_at": session_row["created_at"],
                "updated_at": session_row["updated_at"],
                "metadata": session_row["metadata"] or {}, # Ensure metadata is a dict
                "messages": messages
            })
            logger.debug(f"Session '{session_id}' loaded from PostgreSQL.")
            return chat_session
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e: # Catches Pydantic validation errors too
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")

        session_metadata_list: List[Dict[str, Any]] = []
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    await cur.execute(f"""
                        SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata, COUNT(m.id) as message_count
                        FROM {self._sessions_table} s
                        LEFT JOIN {self._messages_table} m ON s.id = m.session_id
                        GROUP BY s.id, s.name, s.created_at, s.updated_at, s.metadata
                        ORDER BY s.updated_at DESC
                    """)
                    async for row in cur:
                        # Ensure metadata is a dict, not None
                        data = dict(row)
                        data["metadata"] = data.get("metadata") or {}
                        session_metadata_list.append(data)
            logger.debug(f"Found {len(session_metadata_list)} sessions in PostgreSQL.")
            return session_metadata_list
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Database error listing sessions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error listing sessions: {e}")

    async def delete_session(self, session_id: str) -> bool:
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")

        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use transaction
                    async with conn.cursor() as cur:
                        # Deletion cascades to messages table due to FOREIGN KEY ON DELETE CASCADE
                        await cur.execute(f"DELETE FROM {self._sessions_table} WHERE id = %s", (session_id,))
                        deleted_count = cur.rowcount

            if deleted_count > 0:
                logger.info(f"Session '{session_id}' deleted from PostgreSQL.")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent session '{session_id}' from PostgreSQL.")
                return False
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.close()
                self._pool = None
                logger.info("PostgreSQL session storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL session storage connection pool: {e}", exc_info=True)


class PgVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using PostgreSQL with pgvector.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _vectors_table: str
    _collections_table: str # Table to manage RAG collections
    _default_collection_name: str = "llmcore_default_rag"
    _default_vector_dimension: int = 384 # Default, should be configurable or detected

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PgVector storage.

        Args:
            config: Configuration dictionary. Expected keys:
                    'db_url': PostgreSQL connection string.
                    'vectors_table_name' (optional): Name for the vectors table.
                    'collections_table_name' (optional): Name for vector collections metadata table.
                    'default_collection' (optional): Default RAG collection name.
                    'default_vector_dimension' (optional): Default dimension for vector columns.
                    'min_pool_size' (optional): Minimum connections in the pool.
                    'max_pool_size' (optional): Maximum connections in the pool.
        Raises:
            ConfigError: If 'db_url' is not provided or psycopg/pgvector is not installed.
            VectorStorageError: If the database connection or table creation fails.
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed. Please install `psycopg[binary]` or `llmcore[postgres]`.")
        if not pgvector_available:
            raise ConfigError("pgvector library not installed. Please install `pgvector` or `llmcore[postgres]`.")

        db_url = config.get("db_url")
        if not db_url:
            db_url = os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url:
            raise ConfigError("PgVector storage 'db_url' not specified in configuration or LLMCORE_STORAGE_VECTOR_DB_URL env var.")

        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", self._default_collection_name)
        self._default_vector_dimension = config.get("default_vector_dimension", 384) # e.g., for all-MiniLM-L6-v2

        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn:
                # Register pgvector types for this connection
                await register_vector(conn) # type: ignore
                async with conn.transaction():
                    # Ensure pgvector extension is enabled
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # Table for RAG collections metadata
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._collections_table} (
                            id SERIAL PRIMARY KEY,
                            name TEXT UNIQUE NOT NULL,
                            vector_dimension INTEGER NOT NULL,
                            description TEXT,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    # Ensure default collection exists
                    await self._ensure_collection_exists(conn, self._default_collection_name, self._default_vector_dimension)

                    # Table for vectors, partitioned by collection_id or using collection_name directly
                    # For simplicity here, using collection_name in the vectors table.
                    # A separate collections table is better for managing dimensions per collection.
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                            id TEXT NOT NULL, -- Document ID provided by user/system
                            collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE,
                            content TEXT,
                            embedding VECTOR({self._default_vector_dimension}), -- Default dimension, can vary per collection
                            metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (id, collection_name) -- Composite primary key
                        );
                    """)
                    # Index for similarity search (e.g., HNSW or IVFFlat)
                    # Example HNSW index on embedding column for cosine distance:
                    # CREATE INDEX IF NOT EXISTS idx_embedding_hnsw_{self._vectors_table}
                    # ON {self._vectors_table} USING hnsw (embedding vector_cosine_ops);
                    # For L2 distance: vector_l2_ops
                    # For inner product: vector_ip_ops
                    # Index creation depends on the specific needs and pgvector version.
                    # For now, let's assume a basic index or let users create optimized ones.
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_collection_embedding
                        ON {self._vectors_table} USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100);
                    """)
                    # Alternative: GIN index on metadata if filtering is common
                    # CREATE INDEX IF NOT EXISTS idx_metadata_gin_{self._vectors_table}
                    # ON {self._vectors_table} USING gin (metadata);

            logger.info(f"PgVector storage initialized. Tables: '{self._vectors_table}', '{self._collections_table}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PgVector storage: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None
            raise VectorStorageError(f"Could not initialize PgVector storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PgVector storage initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close()
            self._pool = None
            raise VectorStorageError(f"Unexpected PgVector initialization error: {e}")

    async def _ensure_collection_exists(self, conn: psycopg.AsyncConnection, name: str, dimension: int, description: Optional[str] = None) -> None:
        """Ensures a collection exists in the collections table."""
        async with conn.cursor() as cur:
            await cur.execute(
                f"INSERT INTO {self._collections_table} (name, vector_dimension, description) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (name, dimension, description)
            )
            if cur.rowcount > 0:
                logger.info(f"Created new vector collection '{name}' with dimension {dimension}.")


    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")
        if not documents:
            return []

        target_collection = collection_name or self._default_collection_name
        doc_ids_added: List[str] = []

        try:
            async with self._pool.connection() as conn:
                await register_vector(conn) # type: ignore
                async with conn.transaction(): # Use a transaction
                    # Get collection dimension
                    coll_dim_cur = await conn.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (target_collection,))
                    coll_info = await coll_dim_cur.fetchone()
                    if not coll_info:
                        # Auto-create collection if it doesn't exist, using dimension of first doc or default
                        first_doc_dim = len(documents[0].embedding) if documents[0].embedding else self._default_vector_dimension
                        logger.warning(f"Collection '{target_collection}' not found. Auto-creating with dimension {first_doc_dim}.")
                        await self._ensure_collection_exists(conn, target_collection, first_doc_dim)
                        collection_dimension = first_doc_dim
                    else:
                        collection_dimension = coll_info[0]


                    docs_to_insert = []
                    for doc in documents:
                        if not doc.id:
                            raise VectorStorageError("Document must have an ID for PgVector storage.")
                        if not doc.embedding:
                            raise VectorStorageError(f"Document '{doc.id}' must have an embedding.")
                        if len(doc.embedding) != collection_dimension:
                            raise VectorStorageError(f"Document '{doc.id}' embedding dimension {len(doc.embedding)} "
                                                     f"does not match collection '{target_collection}' dimension {collection_dimension}.")

                        docs_to_insert.append((
                            doc.id,
                            target_collection,
                            doc.content,
                            doc.embedding, # pgvector handles list of floats
                            Jsonb(doc.metadata or {})
                        ))
                        doc_ids_added.append(doc.id)

                    if docs_to_insert:
                        # Use ON CONFLICT DO UPDATE for upsert behavior
                        # Note: Ensure embedding column has the correct dimension for the collection.
                        # The table was created with _default_vector_dimension. If collections have different
                        # dimensions, the table schema or insertion logic needs to be more dynamic,
                        # or use separate tables per dimension/collection.
                        # For now, assuming all docs in a batch go to a collection with consistent dimension.
                        async with conn.cursor() as cur:
                            await cur.executemany(f"""
                                INSERT INTO {self._vectors_table} (id, collection_name, content, embedding, metadata)
                                VALUES (%s, %s, %s, %s::vector({collection_dimension}), %s)
                                ON CONFLICT (id, collection_name) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    embedding = EXCLUDED.embedding,
                                    metadata = EXCLUDED.metadata,
                                    created_at = CURRENT_TIMESTAMP
                            """, docs_to_insert)
            logger.info(f"Upserted {len(doc_ids_added)} documents into PgVector collection '{target_collection}'.")
            return doc_ids_added
        except psycopg.Error as e:
            logger.error(f"PgVector error adding documents to collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error adding documents to collection '{target_collection}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error adding documents to collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error adding documents to collection '{target_collection}': {e}")

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")

        target_collection = collection_name or self._default_collection_name
        results: List[ContextDocument] = []

        try:
            async with self._pool.connection() as conn:
                await register_vector(conn) # type: ignore
                conn.row_factory = dict_row

                # Build query with optional metadata filter
                # Note: pgvector uses <-> for L2 distance, <#> for inner product (negative for similarity),
                # <=> for cosine distance. Cosine distance is often preferred.
                # A smaller cosine distance means more similar.
                sql_query = f"""
                    SELECT id, content, metadata, embedding <=> %s::vector({len(query_embedding)}) AS distance
                    FROM {self._vectors_table}
                    WHERE collection_name = %s
                """
                params: List[Any] = [query_embedding, target_collection]

                if filter_metadata:
                    # Basic metadata filtering example (exact match on top-level keys)
                    # For more complex filtering (e.g., nested, ranges), you'd need more complex SQL
                    # or use JSONB operators like @>
                    # Example: WHERE metadata @> %s
                    # params.append(Jsonb(filter_metadata))
                    # sql_query += " AND metadata @> %s"
                    for key, value in filter_metadata.items():
                        sql_query += f" AND metadata->>%s = %s" # Assumes string values in metadata for simplicity
                        params.append(key)
                        params.append(str(value))


                sql_query += " ORDER BY distance ASC LIMIT %s"
                params.append(k)

                async with conn.cursor() as cur:
                    await cur.execute(sql_query, tuple(params))
                    async for row in cur:
                        results.append(ContextDocument(
                            id=row["id"],
                            content=row["content"],
                            metadata=row["metadata"] or {},
                            score=float(row["distance"]) if row["distance"] is not None else None,
                            embedding=None # Not typically returned from search to save bandwidth
                        ))
            logger.debug(f"PgVector search in '{target_collection}' returned {len(results)} documents.")
            return results
        except psycopg.Error as e:
            logger.error(f"PgVector error during similarity search in '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error during similarity search in '{target_collection}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error during similarity search in '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error during similarity search in '{target_collection}': {e}")

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")
        if not document_ids:
            return True

        target_collection = collection_name or self._default_collection_name
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use transaction
                    async with conn.cursor() as cur:
                        # Delete based on composite key (id, collection_name)
                        # Using `IN %s` with a tuple of IDs
                        await cur.execute(
                            f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s)",
                            (target_collection, document_ids)
                        )
                        deleted_count = cur.rowcount
            logger.info(f"Attempted deletion of {deleted_count}/{len(document_ids)} documents from PgVector collection '{target_collection}'.")
            return True # Return True if operation was attempted
        except psycopg.Error as e:
            logger.error(f"PgVector error deleting documents from '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error deleting documents from '{target_collection}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting documents from '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error deleting documents from '{target_collection}': {e}")

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.close()
                self._pool = None
                logger.info("PgVector storage connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing PgVector storage connection pool: {e}", exc_info=True)
