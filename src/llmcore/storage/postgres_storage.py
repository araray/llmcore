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
import pathlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncIterator, TYPE_CHECKING

# psycopg for PostgreSQL interaction (async version)
# Conditional imports for type checking and runtime to handle potential import issues
if TYPE_CHECKING:
    try:
        import psycopg
        from psycopg.rows import dict_row # For fetching rows as dictionaries
        from psycopg.types.json import Jsonb # For efficient JSON storage
        from psycopg_pool import AsyncConnectionPool # For managing connections
        # Import the specific type under an alias for hinting
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        psycopg_available = True
    except ImportError:
        # Define fallbacks ONLY for type checking if import fails
        psycopg = None # type: ignore
        dict_row = None # type: ignore
        Jsonb = None # type: ignore
        AsyncConnectionPool = None # type: ignore
        PsycopgAsyncConnectionType = Any # Fallback type hint
        psycopg_available = False
else:
    # Runtime imports (handle potential ImportError gracefully)
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool
        # Import the actual class for runtime use under the same alias
        from psycopg.abc import AsyncConnection as PsycopgAsyncConnectionType
        psycopg_available = True
    except ImportError:
        # Define fallbacks for runtime if import fails
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PsycopgAsyncConnectionType = Any # Define as Any at runtime
        psycopg_available = False


# pgvector for vector type (optional, only if using PgVectorStorage)
try:
    from pgvector.psycopg import register_vector # type: ignore
    pgvector_available = True
except ImportError:
    pgvector_available = False
    register_vector = None # type: ignore
    logger = logging.getLogger(__name__)
    # Log warning only once during import
    # logger.warning("pgvector library not found. PgVectorStorage will not be fully functional.")


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
    Manages persistence of ChatSession objects in a PostgreSQL database
    using asynchronous connections via psycopg and connection pooling.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _sessions_table: str
    _messages_table: str

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL session storage asynchronously.

        Sets up the connection pool and ensures the necessary tables exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'db_url': PostgreSQL connection string (e.g., "postgresql://user:pass@host:port/dbname").
                              Can also be set via LLMCORE_STORAGE_SESSION_DB_URL env var.
                    'sessions_table_name' (optional): Name for the sessions table (default: llmcore_sessions).
                    'messages_table_name' (optional): Name for the messages table (default: llmcore_messages).
                    'min_pool_size' (optional): Minimum connections in the pool (default: 2).
                    'max_pool_size' (optional): Maximum connections in the pool (default: 10).

        Raises:
            ConfigError: If 'db_url' is not provided or psycopg is not installed.
            SessionStorageError: If the database connection or table creation fails.
        """
        if not psycopg_available:
            raise ConfigError("psycopg library not installed. Please install `psycopg[binary]` or `llmcore[postgres]`.")

        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_SESSION_DB_URL")
        if not db_url:
            raise ConfigError("PostgreSQL session storage 'db_url' not specified in configuration or LLMCORE_STORAGE_SESSION_DB_URL env var.")

        self._sessions_table = config.get("sessions_table_name", DEFAULT_SESSIONS_TABLE)
        self._messages_table = config.get("messages_table_name", DEFAULT_MESSAGES_TABLE)
        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for session storage (min: {min_pool_size}, max: {max_pool_size})...")
            # Create an asynchronous connection pool
            self._pool = AsyncConnectionPool(
                conninfo=db_url,
                min_size=min_pool_size,
                max_size=max_pool_size,
                # Pool starts empty and connects on demand by default
            )
            # Test connection by acquiring and releasing one
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;") # Simple query to test connection
                    if not await cur.fetchone():
                         raise SessionStorageError("Database connection test failed.")
                logger.debug("PostgreSQL connection test successful.")

            # Create tables if they don't exist within a transaction
            async with self._pool.connection() as conn:
                logger.debug(f"Ensuring session tables '{self._sessions_table}' and '{self._messages_table}' exist...")
                async with conn.transaction():
                    # Sessions table stores metadata for each conversation
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._sessions_table} (
                            id TEXT PRIMARY KEY,          -- Unique session identifier
                            name TEXT,                    -- Optional human-readable name
                            created_at TIMESTAMPTZ NOT NULL, -- Session creation timestamp (with timezone)
                            updated_at TIMESTAMPTZ NOT NULL, -- Last modification timestamp (with timezone)
                            metadata JSONB                -- Store additional metadata as JSONB
                        )
                    """)
                    # Messages table stores individual messages linked to a session
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._messages_table} (
                            id TEXT PRIMARY KEY,          -- Unique message identifier
                            session_id TEXT NOT NULL REFERENCES {self._sessions_table}(id) ON DELETE CASCADE, -- Link to sessions table, delete messages if session is deleted
                            role TEXT NOT NULL,           -- 'system', 'user', or 'assistant'
                            content TEXT NOT NULL,        -- The text content of the message
                            timestamp TIMESTAMPTZ NOT NULL, -- Message creation timestamp (with timezone)
                            tokens INTEGER,               -- Optional token count
                            metadata JSONB                -- Store additional metadata as JSONB
                        )
                    """)
                    # Index for faster retrieval of messages within a session, ordered by time
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self._messages_table}_session_timestamp
                        ON {self._messages_table} (session_id, timestamp);
                    """)
            logger.info(f"PostgreSQL session storage initialized successfully. Tables: '{self._sessions_table}', '{self._messages_table}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL session storage: {e}", exc_info=True)
            if self._pool: await self._pool.close() # Attempt cleanup on failure
            self._pool = None
            raise SessionStorageError(f"Could not initialize PostgreSQL session storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL session storage initialization: {e}", exc_info=True)
            if self._pool: await self._pool.close() # Attempt cleanup on failure
            self._pool = None
            raise SessionStorageError(f"Unexpected initialization error: {e}")

    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session and its messages in the database asynchronously.

        Uses an UPSERT operation for the session metadata and replaces all messages
        for the given session ID with the current messages in the session object.

        Args:
            session: The ChatSession object to save.

        Raises:
            SessionStorageError: If the connection pool is not initialized or saving fails.
        """
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")
        if not Jsonb: # Ensure Jsonb adapter is available
             raise SessionStorageError("psycopg Jsonb adapter not available. Ensure psycopg[binary] or appropriate extras are installed.")

        logger.debug(f"Saving session '{session.id}' with {len(session.messages)} messages to PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Ensure atomicity
                    # UPSERT session metadata: Insert or update if exists based on primary key 'id'
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
                        session.created_at, # Assumes timezone-aware datetime
                        session.updated_at, # Assumes timezone-aware datetime
                        Jsonb(session.metadata or {}) # Use Jsonb adapter for metadata
                    ))

                    # Replace messages: Delete existing messages first, then insert current ones
                    await conn.execute(f"DELETE FROM {self._messages_table} WHERE session_id = %s", (session.id,))

                    if session.messages:
                        messages_data = [
                            (
                                msg.id,
                                session.id,
                                str(msg.role), # Convert Role enum to string
                                msg.content,
                                msg.timestamp, # Assumes timezone-aware datetime
                                msg.tokens,
                                Jsonb(msg.metadata or {}) # Use Jsonb adapter for metadata
                            ) for msg in session.messages
                        ]
                        # Use executemany for efficient batch insertion
                        async with conn.cursor() as cur:
                            await cur.executemany(f"""
                                INSERT INTO {self._messages_table}
                                (id, session_id, role, content, timestamp, tokens, metadata)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, messages_data)
                        logger.debug(f"Inserted {len(messages_data)} messages for session '{session.id}'.")
            logger.info(f"Session '{session.id}' saved successfully to PostgreSQL.")
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error saving session '{session.id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")


    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a specific chat session and its messages by ID asynchronously.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.

        Raises:
            SessionStorageError: If the connection pool is not initialized or retrieval fails.
        """
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")
        if not dict_row: # Ensure dict_row factory is available
            raise SessionStorageError("psycopg dict_row factory not available.")

        logger.debug(f"Loading session '{session_id}' from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row # Fetch results as dictionaries
                async with conn.cursor() as cur:
                    # Fetch session metadata
                    await cur.execute(f"SELECT * FROM {self._sessions_table} WHERE id = %s", (session_id,))
                    session_row = await cur.fetchone()

                    if not session_row:
                        logger.debug(f"Session '{session_id}' not found.")
                        return None

                    # Fetch associated messages, ordered by timestamp
                    await cur.execute(f"""
                        SELECT id, session_id, role, content, timestamp, tokens, metadata
                        FROM {self._messages_table} WHERE session_id = %s ORDER BY timestamp ASC
                    """, (session_id,))
                    message_rows = await cur.fetchall()

            # Reconstruct messages, converting role string back to Enum
            messages = []
            for msg_row in message_rows:
                try:
                    msg_data = dict(msg_row) # Convert row object to dict
                    msg_data["role"] = Role(msg_data["role"]) # Convert role string back to Role enum
                    # TIMESTAMPTZ should return timezone-aware datetime objects
                    messages.append(Message.model_validate(msg_data)) # Validate data using Pydantic model
                except ValueError as ve:
                    # Log and skip messages with invalid roles or other validation errors
                    logger.warning(f"Invalid data for message {msg_row.get('id')} in session {session_id}: {ve}. Skipping.")
                    continue

            # Reconstruct the ChatSession object using Pydantic validation
            # Ensure metadata is a dict, defaulting to {} if None/null in DB
            session_data = dict(session_row)
            session_data["metadata"] = session_data.get("metadata") or {}
            session_data["messages"] = messages
            chat_session = ChatSession.model_validate(session_data)

            logger.info(f"Session '{session_id}' loaded successfully from PostgreSQL ({len(messages)} messages).")
            return chat_session
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error retrieving session '{session_id}': {e}")
        except Exception as e: # Catches Pydantic validation errors too
            logger.error(f"Unexpected error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error retrieving session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions asynchronously, returning metadata only.

        Includes session ID, name, timestamps, metadata, and message count.

        Returns:
            A list of dictionaries, each representing session metadata, ordered by update time descending.

        Raises:
            SessionStorageError: If the connection pool is not initialized or listing fails.
        """
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")
        if not dict_row:
            raise SessionStorageError("psycopg dict_row factory not available.")

        session_metadata_list: List[Dict[str, Any]] = []
        logger.debug("Listing session metadata from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    # Query joins sessions with messages to count messages efficiently
                    await cur.execute(f"""
                        SELECT s.id, s.name, s.created_at, s.updated_at, s.metadata, COUNT(m.id) as message_count
                        FROM {self._sessions_table} s
                        LEFT JOIN {self._messages_table} m ON s.id = m.session_id
                        GROUP BY s.id, s.name, s.created_at, s.updated_at, s.metadata
                        ORDER BY s.updated_at DESC
                    """)
                    async for row in cur:
                        data = dict(row)
                        # Ensure metadata is a dict, not None
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
        """
        Delete a specific chat session and its associated messages asynchronously.

        Deletion cascades to the messages table due to the foreign key constraint.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was found and deleted successfully, False otherwise.

        Raises:
            SessionStorageError: If the connection pool is not initialized or deletion fails.
        """
        if not self._pool:
            raise SessionStorageError("PostgreSQL connection pool is not initialized.")

        logger.debug(f"Deleting session '{session_id}' from PostgreSQL...")
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use transaction for atomicity
                    async with conn.cursor() as cur:
                        # Delete the session; messages are deleted via CASCADE constraint
                        await cur.execute(f"DELETE FROM {self._sessions_table} WHERE id = %s", (session_id,))
                        deleted_count = cur.rowcount # Check how many rows were affected

            if deleted_count > 0:
                logger.info(f"Session '{session_id}' deleted successfully from PostgreSQL.")
                return True
            else:
                logger.warning(f"Attempted to delete session '{session_id}', but it was not found.")
                return False
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Database error deleting session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """Closes the PostgreSQL connection pool asynchronously."""
        if self._pool:
            pool_ref = self._pool # Store ref in case of concurrent calls
            self._pool = None # Mark as closed immediately
            try:
                logger.info("Closing PostgreSQL session storage connection pool...")
                await pool_ref.close()
                logger.info("PostgreSQL session storage connection pool closed successfully.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL session storage connection pool: {e}", exc_info=True)
                # Log error but don't raise during cleanup


class PgVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using PostgreSQL
    with the pgvector extension. Requires asynchronous connections via psycopg.
    """
    _pool: Optional[AsyncConnectionPool] = None
    _vectors_table: str
    _collections_table: str
    _default_collection_name: str = "llmcore_default_rag"
    _default_vector_dimension: int = 384 # Example default, adjust as needed

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PgVector storage asynchronously.

        Sets up the connection pool and ensures the necessary tables (including
        a collections table) and the pgvector extension exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'db_url': PostgreSQL connection string. Can also be set via
                              LLMCORE_STORAGE_VECTOR_DB_URL env var.
                    'vectors_table_name' (optional): Name for the main vectors table.
                    'collections_table_name' (optional): Name for vector collections metadata table.
                    'default_collection' (optional): Default RAG collection name.
                    'default_vector_dimension' (optional): Default dimension for vector columns if not specified per collection.
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

        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url:
            raise ConfigError("PgVector storage 'db_url' not specified in configuration or LLMCORE_STORAGE_VECTOR_DB_URL env var.")

        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", self._default_collection_name)
        # Store default dimension, but prioritize dimension from collections table
        self._default_vector_dimension = config.get("default_vector_dimension", 384)

        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)

        try:
            logger.debug(f"Initializing PostgreSQL connection pool for vector storage (min: {min_pool_size}, max: {max_pool_size})...")
            self._pool = AsyncConnectionPool(conninfo=db_url, min_size=min_pool_size, max_size=max_pool_size)
            async with self._pool.connection() as conn:
                logger.debug("Registering pgvector type adapter...")
                if register_vector:
                    await register_vector(conn) # Register pgvector types for this connection
                else:
                    # This case should be caught by the check at the top, but log defensively
                    logger.error("pgvector register_vector function not available during initialization.")
                    raise VectorStorageError("pgvector library functions not available.")

                async with conn.transaction():
                    logger.debug("Ensuring 'vector' extension exists...")
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # Table for managing RAG collections and their properties (like dimension)
                    logger.debug(f"Ensuring collections table '{self._collections_table}' exists...")
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._collections_table} (
                            id SERIAL PRIMARY KEY,          -- Auto-incrementing ID
                            name TEXT UNIQUE NOT NULL,      -- Unique name for the collection
                            vector_dimension INTEGER NOT NULL, -- Dimension of vectors in this collection
                            description TEXT,               -- Optional description
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- Creation timestamp
                        );
                    """)
                    # Ensure the default collection exists in the metadata table
                    await self._ensure_collection_exists(conn, self._default_collection_name, self._default_vector_dimension)

                    # Main table for storing vectors
                    logger.debug(f"Ensuring vectors table '{self._vectors_table}' exists...")
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                            id TEXT NOT NULL,               -- Document ID provided by user/system
                            collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE, -- Link to collections table
                            content TEXT,                   -- Text content of the document (optional)
                            embedding VECTOR,               -- Stored vector (dimension enforced on insert/query)
                            metadata JSONB,                 -- Additional metadata
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, -- Timestamp
                            PRIMARY KEY (id, collection_name) -- Ensure unique documents within a collection
                        );
                    """)
                    # Example index for similarity search (choose appropriate type and metric)
                    # HNSW with cosine distance is often a good starting point.
                    index_name = f"idx_embedding_cosine_{self._vectors_table}"
                    logger.debug(f"Ensuring vector index '{index_name}' exists...")
                    # Note: Creating HNSW indexes can take time on large tables.
                    # Consider creating indexes manually or providing configuration options.
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._vectors_table} USING hnsw (embedding vector_cosine_ops);
                    """)

            logger.info(f"PgVector storage initialized successfully. Tables: '{self._vectors_table}', '{self._collections_table}'.")
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

    async def _ensure_collection_exists(self, conn: PsycopgAsyncConnectionType, name: str, dimension: int, description: Optional[str] = None) -> None:
        """
        Ensures a collection record exists in the collections table.

        Args:
            conn: The active database connection.
            name: The name of the collection.
            dimension: The vector dimension for the collection.
            description: Optional description for the collection.
        """
        logger.debug(f"Ensuring vector collection '{name}' (dim: {dimension}) exists...")
        async with conn.cursor() as cur:
            # Insert the collection if it doesn't exist based on the unique name constraint
            await cur.execute(
                f"INSERT INTO {self._collections_table} (name, vector_dimension, description) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (name, dimension, description)
            )
            if cur.rowcount > 0:
                logger.info(f"Created new vector collection '{name}' with dimension {dimension}.")
            else:
                 # Verify dimension matches if collection already exists
                 await cur.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (name,))
                 existing_info = await cur.fetchone()
                 if existing_info and existing_info[0] != dimension:
                      logger.error(f"Collection '{name}' already exists with dimension {existing_info[0]}, but requested dimension is {dimension}.")
                      raise ConfigError(f"Dimension mismatch for existing collection '{name}'. Expected {existing_info[0]}, got {dimension}.")


    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Adds or updates multiple documents in the specified vector collection asynchronously.

        Validates document IDs, embeddings, and embedding dimensions against the collection's
        configured dimension before performing an UPSERT operation.

        Args:
            documents: A list of ContextDocument objects to add/update.
            collection_name: The target collection name. Uses default if None.

        Returns:
            A list of IDs of the added/updated documents.

        Raises:
            VectorStorageError: If the pool is not initialized, documents are invalid,
                                dimension mismatch occurs, or a database error happens.
        """
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")
        if not documents:
            logger.debug("add_documents called with empty list, nothing to add.")
            return []
        if not Jsonb: # Check required adapter
            raise VectorStorageError("psycopg Jsonb adapter not available.")

        target_collection = collection_name or self._default_collection_name
        doc_ids_added: List[str] = []
        logger.debug(f"Adding/updating {len(documents)} documents in PgVector collection '{target_collection}'...")

        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn) # Ensure type is registered for this connection
                async with conn.transaction():
                    # 1. Get the required vector dimension for the target collection
                    async with conn.cursor() as cur:
                        await cur.execute(f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s", (target_collection,))
                        coll_info = await cur.fetchone()
                        if not coll_info:
                            # Collection doesn't exist, try to auto-create based on first document
                            first_doc_dim = len(documents[0].embedding) if documents[0].embedding else self._default_vector_dimension
                            logger.warning(f"Collection '{target_collection}' not found. Auto-creating with dimension {first_doc_dim}.")
                            await self._ensure_collection_exists(conn, target_collection, first_doc_dim)
                            collection_dimension = first_doc_dim
                        else:
                            collection_dimension = coll_info[0]
                        logger.debug(f"Target collection '{target_collection}' requires dimension {collection_dimension}.")

                    # 2. Prepare data for insertion, validating dimensions
                    docs_to_insert = []
                    for doc in documents:
                        if not doc.id: raise VectorStorageError("Document is missing an ID.")
                        if not doc.embedding: raise VectorStorageError(f"Document '{doc.id}' is missing an embedding.")
                        if len(doc.embedding) != collection_dimension:
                            raise VectorStorageError(f"Document '{doc.id}' embedding dimension ({len(doc.embedding)}) "
                                                     f"does not match collection '{target_collection}' dimension ({collection_dimension}).")

                        docs_to_insert.append((
                            doc.id,
                            target_collection,
                            doc.content,
                            doc.embedding, # Pass as list, cast in SQL
                            Jsonb(doc.metadata or {}) # Use Jsonb adapter
                        ))
                        doc_ids_added.append(doc.id)

                    # 3. Perform batch UPSERT using executemany
                    if docs_to_insert:
                        async with conn.cursor() as cur:
                            # Use ::vector(%s) for explicit casting with dimension parameter
                            sql = f"""
                                INSERT INTO {self._vectors_table} (id, collection_name, content, embedding, metadata)
                                VALUES (%s, %s, %s, %s::vector(%s), %s)
                                ON CONFLICT (id, collection_name) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    embedding = EXCLUDED.embedding,
                                    metadata = EXCLUDED.metadata,
                                    created_at = CURRENT_TIMESTAMP
                            """
                            # Add dimension parameter to each tuple for casting
                            data_with_dim = [(d[0], d[1], d[2], d[3], collection_dimension, d[4]) for d in docs_to_insert]
                            await cur.executemany(sql, data_with_dim)
                            logger.debug(f"Executed UPSERT for {len(docs_to_insert)} documents.")

            logger.info(f"Successfully upserted {len(doc_ids_added)} documents into PgVector collection '{target_collection}'.")
            return doc_ids_added
        except psycopg.Error as e:
            logger.error(f"PgVector database error adding documents to collection '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error adding documents to collection '{target_collection}': {e}")
        except VectorStorageError: # Re-raise validation errors
            raise
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
        """
        Perform a similarity search asynchronously using pgvector distance operators.

        Args:
            query_embedding: The vector embedding of the query.
            k: The number of nearest neighbors to retrieve.
            collection_name: The target collection name. Uses default if None.
            filter_metadata: Optional dictionary for filtering based on metadata.

        Returns:
            A list of ContextDocument objects, ordered by similarity, containing
            ID, content, metadata, and similarity score.

        Raises:
            VectorStorageError: If the pool is not initialized or a database error occurs.
        """
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")
        if not dict_row: # Check row factory
            raise VectorStorageError("psycopg dict_row factory not available.")

        target_collection = collection_name or self._default_collection_name
        results: List[ContextDocument] = []
        query_dimension = len(query_embedding)
        logger.debug(f"Performing similarity search (k={k}) in PgVector collection '{target_collection}'...")

        try:
            async with self._pool.connection() as conn:
                if register_vector: await register_vector(conn) # Ensure type is registered
                conn.row_factory = dict_row

                # Choose distance operator (e.g., <=> for cosine, <-> for L2)
                # Cosine distance is often preferred for semantic similarity.
                # Smaller distance means more similar for cosine and L2.
                distance_operator = "<=>"
                order_direction = "ASC" # Order by distance ascending (most similar first)

                # Base SQL query
                sql_query = f"""
                    SELECT id, content, metadata, embedding {distance_operator} %s::vector({query_dimension}) AS distance
                    FROM {self._vectors_table}
                    WHERE collection_name = %s
                """
                params: List[Any] = [query_embedding, target_collection]

                # Add metadata filtering if provided
                if filter_metadata:
                    filter_clauses = []
                    for key, value in filter_metadata.items():
                        # Example: Simple key-value equality using JSONB ->> operator (text comparison)
                        # For more complex filtering (e.g., numeric ranges, contains), use appropriate JSONB operators.
                        filter_clauses.append(f"metadata->>%s = %s")
                        params.extend([key, str(value)]) # Add key and value to params
                    if filter_clauses:
                        sql_query += " AND " + " AND ".join(filter_clauses)
                        logger.debug(f"Applying metadata filter: {filter_metadata}")

                # Add ordering and limit
                sql_query += f" ORDER BY distance {order_direction} LIMIT %s"
                params.append(k)

                # Execute query
                async with conn.cursor() as cur:
                    await cur.execute(sql_query, tuple(params))
                    async for row in cur:
                        # Create ContextDocument from row data
                        results.append(ContextDocument(
                            id=row["id"],
                            content=row.get("content", ""), # Handle potentially null content
                            metadata=row.get("metadata") or {}, # Ensure metadata is a dict
                            score=float(row["distance"]) if row.get("distance") is not None else None,
                            embedding=None # Embeddings usually not needed in search results
                        ))
            logger.info(f"PgVector similarity search in '{target_collection}' returned {len(results)} documents.")
            return results
        except psycopg.Error as e:
            logger.error(f"PgVector database error during similarity search in '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error during similarity search in '{target_collection}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error during similarity search in '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error during similarity search in '{target_collection}': {e}")

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete documents asynchronously by their IDs from the specified collection.

        Args:
            document_ids: A list of document IDs to delete.
            collection_name: The target collection name. Uses default if None.

        Returns:
            True if the deletion operation was attempted successfully, False otherwise.

        Raises:
            VectorStorageError: If the pool is not initialized or a database error occurs.
        """
        if not self._pool:
            raise VectorStorageError("PgVector connection pool is not initialized.")
        if not document_ids:
            logger.debug("delete_documents called with empty ID list.")
            return True

        target_collection = collection_name or self._default_collection_name
        logger.debug(f"Deleting {len(document_ids)} documents from PgVector collection '{target_collection}'...")
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction(): # Use transaction
                    async with conn.cursor() as cur:
                        # Delete using collection_name and list of IDs (`= ANY(%s)`)
                        await cur.execute(
                            f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s)",
                            (target_collection, document_ids)
                        )
                        deleted_count = cur.rowcount # Get number of rows affected
            logger.info(f"Deletion attempt affected {deleted_count} rows in PgVector collection '{target_collection}' for {len(document_ids)} requested IDs.")
            # Return True indicating the operation completed, even if some IDs weren't found
            return True
        except psycopg.Error as e:
            logger.error(f"PgVector database error deleting documents from '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Database error deleting documents from '{target_collection}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting documents from '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Unexpected error deleting documents from '{target_collection}': {e}")

    async def close(self) -> None:
        """Closes the PgVector connection pool asynchronously."""
        if self._pool:
            pool_ref = self._pool
            self._pool = None # Mark as closed
            try:
                logger.info("Closing PgVector storage connection pool...")
                await pool_ref.close()
                logger.info("PgVector storage connection pool closed successfully.")
            except Exception as e:
                logger.error(f"Error closing PgVector storage connection pool: {e}", exc_info=True)
