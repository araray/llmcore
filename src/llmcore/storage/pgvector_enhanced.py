# src/llmcore/storage/pgvector_enhanced.py
"""
Enhanced PostgreSQL + pgvector Storage Implementation.

Phase 2 (NEXUS): Extended vector storage with advanced features:
- HNSW index optimization with configurable parameters
- Batch upsert operations with chunking and retry
- Filtered similarity search with metadata conditions
- Hybrid search combining full-text and vector similarity
- Multi-user isolation via user_id enforcement
- Connection pool optimization with lifecycle management

This module extends the base PgVectorStorage with production-grade
features for high-performance vector search.

Usage:
    from llmcore.storage.pgvector_enhanced import EnhancedPgVectorStorage
    from llmcore.storage.abstraction import StorageContext, HNSWConfig

    storage = EnhancedPgVectorStorage()
    await storage.initialize({
        "db_url": "postgresql://...",
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
    })

    # User-scoped context
    ctx = StorageContext(user_id="user_123")

    # Batch upsert
    result = await storage.batch_upsert_documents(
        documents, collection_name="my_collection", context=ctx
    )

    # Hybrid search
    results = await storage.hybrid_search(
        query_text="semantic search query",
        query_embedding=[...],
        k=10,
        context=ctx
    )
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..exceptions import ConfigError, VectorStorageError
from ..models import ContextDocument
from .base_vector import BaseVectorStorage
from .abstraction import (
    StorageContext,
    BackendCapabilities,
    POSTGRES_PGVECTOR_CAPABILITIES,
    HNSWConfig,
    VectorSearchConfig,
    BatchConfig,
    BatchResult,
    PoolConfig,
    chunk_list,
    execute_with_retry,
)

if TYPE_CHECKING:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool, PoolTimeout
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PoolTimeout = None
        psycopg_available = False
else:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool, PoolTimeout
        psycopg_available = True
    except ImportError:
        psycopg = None
        dict_row = None
        Jsonb = None
        AsyncConnectionPool = None
        PoolTimeout = None
        psycopg_available = False

try:
    from pgvector.psycopg import register_vector_async
    pgvector_available = True
except ImportError:
    try:
        # Fallback for older pgvector versions
        from pgvector.psycopg import register_vector as register_vector_async
        pgvector_available = True
    except ImportError:
        pgvector_available = False
        register_vector_async = None


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_VECTORS_TABLE = "vectors"
DEFAULT_COLLECTIONS_TABLE = "vector_collections"
DEFAULT_COLLECTION_NAME = "default_rag"
DEFAULT_VECTOR_DIMENSION = 1536  # OpenAI ada-002 dimension


# =============================================================================
# HELPER DATACLASSES
# =============================================================================

@dataclass
class CollectionInfo:
    """Metadata about a vector collection."""
    name: str
    vector_dimension: int
    description: Optional[str] = None
    embedding_model_provider: Optional[str] = None
    embedding_model_name: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "vector_dimension": self.vector_dimension,
            "description": self.description,
            "embedding_model_provider": self.embedding_model_provider,
            "embedding_model_name": self.embedding_model_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
            "document_count": self.document_count,
        }


@dataclass
class HybridSearchResult:
    """Result from hybrid search operation."""
    document: ContextDocument
    vector_score: float
    text_score: float
    combined_score: float
    rank: int


# =============================================================================
# ENHANCED PGVECTOR STORAGE
# =============================================================================

class EnhancedPgVectorStorage(BaseVectorStorage):
    """
    Enhanced PostgreSQL + pgvector storage with Phase 2 features.

    Provides:
    - HNSW index configuration and optimization
    - Batch document operations with retry logic
    - Filtered similarity search
    - Hybrid search (vector + full-text)
    - Multi-user isolation
    - Connection pool lifecycle management

    Attributes:
        _pool: Connection pool for database operations
        _vectors_table: Name of the vectors table
        _collections_table: Name of the collections metadata table
        _hnsw_config: HNSW index configuration
        _pool_config: Connection pool configuration
        _capabilities: Backend capability declaration
    """

    def __init__(self):
        """Initialize storage (configuration applied in initialize())."""
        self._pool: Optional["AsyncConnectionPool"] = None
        self._tenant_session: Optional[AsyncSession] = None
        self._vectors_table: str = DEFAULT_VECTORS_TABLE
        self._collections_table: str = DEFAULT_COLLECTIONS_TABLE
        self._default_collection_name: str = DEFAULT_COLLECTION_NAME
        self._default_vector_dimension: int = DEFAULT_VECTOR_DIMENSION
        self._hnsw_config: HNSWConfig = HNSWConfig()
        self._pool_config: PoolConfig = PoolConfig()
        self._capabilities: BackendCapabilities = POSTGRES_PGVECTOR_CAPABILITIES
        self._initialized: bool = False

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the enhanced pgvector storage.

        Config options:
            db_url: PostgreSQL connection URL
            vectors_table_name: Name for vectors table (default: "vectors")
            collections_table_name: Name for collections table
            default_collection: Default collection name
            default_vector_dimension: Default embedding dimension
            hnsw_m: HNSW m parameter (default: 16)
            hnsw_ef_construction: HNSW build parameter (default: 64)
            hnsw_ef_search: HNSW search parameter (default: 40)
            min_pool_size: Minimum pool connections
            max_pool_size: Maximum pool connections
            pool_max_idle_seconds: Connection idle timeout
            pool_max_lifetime_seconds: Connection lifetime

        Args:
            config: Configuration dictionary

        Raises:
            ConfigError: If required dependencies are missing
            VectorStorageError: If initialization fails
        """
        if not psycopg_available:
            raise ConfigError(
                "psycopg library not installed. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )
        if not pgvector_available:
            raise ConfigError(
                "pgvector library not installed. "
                "Install with: pip install pgvector"
            )

        # Table configuration
        self._vectors_table = config.get("vectors_table_name", DEFAULT_VECTORS_TABLE)
        self._collections_table = config.get("collections_table_name", DEFAULT_COLLECTIONS_TABLE)
        self._default_collection_name = config.get("default_collection", DEFAULT_COLLECTION_NAME)
        self._default_vector_dimension = int(config.get("default_vector_dimension", DEFAULT_VECTOR_DIMENSION))

        # HNSW configuration
        self._hnsw_config = HNSWConfig(
            m=int(config.get("hnsw_m", 16)),
            ef_construction=int(config.get("hnsw_ef_construction", 64)),
            ef_search=int(config.get("hnsw_ef_search", 40)),
        )

        # Pool configuration
        self._pool_config = PoolConfig(
            min_size=int(config.get("min_pool_size", 2)),
            max_size=int(config.get("max_pool_size", 10)),
            max_idle_seconds=float(config.get("pool_max_idle_seconds", 300.0)),
            max_lifetime_seconds=float(config.get("pool_max_lifetime_seconds", 3600.0)),
            statement_cache_size=int(config.get("statement_cache_size", 100)),
        )

        # If tenant session is already configured, skip pool initialization
        if hasattr(self, '_tenant_session') and self._tenant_session is not None:
            logger.debug("Enhanced PgVector storage initialized in tenant-scoped mode")
            self._initialized = True
            return

        # Get database URL
        db_url = config.get("db_url") or os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL")
        if not db_url:
            raise ConfigError("PgVector storage 'db_url' not specified in config or environment")

        try:
            logger.debug(
                f"Initializing PostgreSQL connection pool "
                f"(min: {self._pool_config.min_size}, max: {self._pool_config.max_size})..."
            )

            # Create connection pool (don't open immediately - do it explicitly)
            self._pool = AsyncConnectionPool(
                conninfo=db_url,
                min_size=self._pool_config.min_size,
                max_size=self._pool_config.max_size,
                open=False,  # Don't auto-open in constructor (deprecated behavior)
            )

            # Explicitly open the pool
            await self._pool.open()

            # Initialize database schema
            await self._ensure_schema()

            self._initialized = True
            logger.info(
                f"Enhanced PgVector storage initialized. "
                f"Table: {self._vectors_table}, "
                f"HNSW(m={self._hnsw_config.m}, ef={self._hnsw_config.ef_construction})"
            )

        except psycopg.Error as e:
            logger.error(f"Failed to initialize PgVector storage: {e}", exc_info=True)
            await self.close()
            raise VectorStorageError(f"Database initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected initialization error: {e}", exc_info=True)
            await self.close()
            raise VectorStorageError(f"Initialization failed: {e}")

    async def _ensure_schema(self) -> None:
        """
        Ensure database schema exists with optimized indexes.

        Creates:
        - vector_collections table for collection metadata
        - vectors table with HNSW index
        - User isolation index
        - Full-text search index for hybrid search
        """
        if not self._pool:
            return

        async with self._pool.connection() as conn:
            if register_vector_async:
                await register_vector_async(conn)

            async with conn.transaction():
                # Create pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Create pg_trgm for text search (may already exist from session storage)
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

                # Collections metadata table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._collections_table} (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        vector_dimension INTEGER NOT NULL,
                        description TEXT,
                        embedding_model_provider TEXT,
                        embedding_model_name TEXT,
                        hnsw_m INTEGER DEFAULT 16,
                        hnsw_ef_construction INTEGER DEFAULT 64,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)

                # Ensure default collection exists
                await self._ensure_collection_exists(
                    conn,
                    self._default_collection_name,
                    self._default_vector_dimension
                )

                # Vectors table with user_id for isolation
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._vectors_table} (
                        id TEXT NOT NULL,
                        collection_name TEXT NOT NULL REFERENCES {self._collections_table}(name) ON DELETE CASCADE,
                        user_id TEXT,
                        content TEXT,
                        embedding VECTOR({self._default_vector_dimension}),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id, collection_name)
                    );
                """)

                # HNSW index for fast similarity search
                hnsw_options = self._hnsw_config.to_index_options()
                index_name = f"idx_{self._vectors_table}_embedding_hnsw"
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self._vectors_table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH {hnsw_options};
                """)

                # User isolation index
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_user
                    ON {self._vectors_table} (user_id)
                    WHERE user_id IS NOT NULL;
                """)

                # Collection filtering index
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_collection
                    ON {self._vectors_table} (collection_name);
                """)

                # Composite index for user + collection filtering
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_user_collection
                    ON {self._vectors_table} (user_id, collection_name)
                    WHERE user_id IS NOT NULL;
                """)

                # GIN index for full-text search on content (for hybrid search)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_content_trgm
                    ON {self._vectors_table}
                    USING gin (content gin_trgm_ops)
                    WHERE content IS NOT NULL;
                """)

                # GIN index for metadata JSONB queries
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._vectors_table}_metadata
                    ON {self._vectors_table}
                    USING gin (metadata jsonb_path_ops);
                """)

        logger.debug("Database schema verified/created successfully")

    async def _ensure_collection_exists(
        self,
        conn: Any,
        name: str,
        dimension: int,
        description: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ensure a collection record exists, creating or updating as needed.

        Args:
            conn: Database connection
            name: Collection name
            dimension: Vector dimension
            description: Optional description
            provider: Embedding model provider
            model_name: Embedding model name
            metadata: Additional metadata
        """
        if not dict_row:
            raise VectorStorageError("psycopg dict_row not available")

        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s",
                (name,)
            )
            existing = await cur.fetchone()

            if existing:
                if existing["vector_dimension"] != dimension:
                    raise ConfigError(
                        f"Collection '{name}' exists with dimension {existing['vector_dimension']}, "
                        f"but {dimension} was requested"
                    )
                # Update metadata if provided
                if description or provider or model_name or metadata:
                    updates = []
                    params = []
                    if description:
                        updates.append("description = %s")
                        params.append(description)
                    if provider:
                        updates.append("embedding_model_provider = %s")
                        params.append(provider)
                    if model_name:
                        updates.append("embedding_model_name = %s")
                        params.append(model_name)
                    if metadata:
                        updates.append("metadata = metadata || %s")
                        params.append(Jsonb(metadata))

                    if updates:
                        params.append(name)
                        await cur.execute(
                            f"UPDATE {self._collections_table} SET {', '.join(updates)} WHERE name = %s",
                            tuple(params)
                        )
            else:
                # Create new collection
                await cur.execute(
                    f"""
                    INSERT INTO {self._collections_table}
                    (name, vector_dimension, description, embedding_model_provider,
                     embedding_model_name, hnsw_m, hnsw_ef_construction, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        name, dimension, description, provider, model_name,
                        self._hnsw_config.m, self._hnsw_config.ef_construction,
                        Jsonb(metadata or {})
                    )
                )
                logger.info(f"Created vector collection '{name}' with dimension {dimension}")

    # =========================================================================
    # CAPABILITIES
    # =========================================================================

    def get_capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        return self._capabilities

    async def health_check(self) -> bool:
        """
        Perform a quick health check.

        Returns:
            True if backend is healthy
        """
        if not self._pool:
            return False

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    return bool(await cur.fetchone())
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================

    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None
    ) -> List[str]:
        """
        Add or update documents in the vector store.

        Args:
            documents: Documents to add (must have id and embedding)
            collection_name: Target collection (default: default collection)
            context: Storage context for user isolation

        Returns:
            List of document IDs that were added/updated

        Raises:
            VectorStorageError: If operation fails
        """
        if not documents:
            return []

        target_collection = collection_name or self._default_collection_name
        user_id = context.user_id if context else None

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._add_documents_tenant(documents, target_collection, user_id)
            else:
                return await self._add_documents_pool(documents, target_collection, user_id)
        except Exception as e:
            logger.error(f"Error adding documents to '{target_collection}': {e}", exc_info=True)
            raise VectorStorageError(f"Failed to add documents: {e}")

    async def _add_documents_pool(
        self,
        documents: List[ContextDocument],
        collection_name: str,
        user_id: Optional[str]
    ) -> List[str]:
        """Add documents using connection pool."""
        if not Jsonb:
            raise VectorStorageError("psycopg Jsonb adapter not available")

        added_ids: List[str] = []

        async with self._pool.connection() as conn:
            if register_vector_async:
                await register_vector_async(conn)

            async with conn.transaction():
                # Determine collection dimension from first document
                first_embedding = documents[0].embedding if documents[0].embedding else None
                if not first_embedding:
                    raise VectorStorageError("Documents must have embeddings")

                dimension = len(first_embedding)

                # Get embedding model info from first document metadata
                first_meta = documents[0].metadata or {}
                provider = first_meta.get("embedding_model_provider")
                model_name = first_meta.get("embedding_model_name")

                # Ensure collection exists
                await self._ensure_collection_exists(
                    conn, collection_name, dimension,
                    provider=provider, model_name=model_name
                )

                # Prepare documents for insertion
                for doc in documents:
                    if not doc.id or not doc.embedding:
                        raise VectorStorageError(f"Document must have id and embedding")

                    if len(doc.embedding) != dimension:
                        raise VectorStorageError(
                            f"Embedding dimension mismatch: expected {dimension}, got {len(doc.embedding)}"
                        )

                    # Clean metadata (remove embedding model info to avoid duplication)
                    doc_metadata = (doc.metadata or {}).copy()
                    doc_metadata.pop("embedding_model_provider", None)
                    doc_metadata.pop("embedding_model_name", None)
                    doc_metadata.pop("embedding_dimension", None)

                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"""
                            INSERT INTO {self._vectors_table}
                            (id, collection_name, user_id, content, embedding, metadata, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (id, collection_name) DO UPDATE SET
                                user_id = EXCLUDED.user_id,
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (
                                doc.id, collection_name, user_id,
                                doc.content, doc.embedding, Jsonb(doc_metadata)
                            )
                        )
                    added_ids.append(doc.id)

        logger.info(f"Added {len(added_ids)} documents to '{collection_name}'")
        return added_ids

    async def _add_documents_tenant(
        self,
        documents: List[ContextDocument],
        collection_name: str,
        user_id: Optional[str]
    ) -> List[str]:
        """Add documents using tenant session."""
        # Implementation similar to pool version but using SQLAlchemy text()
        added_ids: List[str] = []

        for doc in documents:
            if not doc.id or not doc.embedding:
                raise VectorStorageError("Document must have id and embedding")

            doc_metadata = (doc.metadata or {}).copy()
            doc_metadata.pop("embedding_model_provider", None)
            doc_metadata.pop("embedding_model_name", None)

            await self._tenant_session.execute(
                text(f"""
                    INSERT INTO {self._vectors_table}
                    (id, collection_name, user_id, content, embedding, metadata, updated_at)
                    VALUES (:id, :collection, :user_id, :content, :embedding, :metadata, CURRENT_TIMESTAMP)
                    ON CONFLICT (id, collection_name) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "id": doc.id,
                    "collection": collection_name,
                    "user_id": user_id,
                    "content": doc.content,
                    "embedding": str(doc.embedding),
                    "metadata": json.dumps(doc_metadata),
                }
            )
            added_ids.append(doc.id)

        await self._tenant_session.commit()
        logger.info(f"Added {len(added_ids)} documents to '{collection_name}' (tenant mode)")
        return added_ids

    async def batch_upsert_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None,
        batch_config: Optional[BatchConfig] = None
    ) -> BatchResult[str]:
        """
        Batch upsert documents with chunking and retry logic.

        Args:
            documents: Documents to upsert
            collection_name: Target collection
            context: Storage context for user isolation
            batch_config: Batch operation configuration

        Returns:
            BatchResult with successful/failed document IDs
        """
        if not documents:
            return BatchResult(successful=[], failed=[], total=0, duration_ms=0)

        config = batch_config or BatchConfig()
        target_collection = collection_name or self._default_collection_name
        start_time = time.time()

        successful: List[str] = []
        failed: List[Tuple[ContextDocument, str]] = []

        # Chunk documents
        chunks = chunk_list(documents, config.chunk_size)

        for chunk_idx, chunk in enumerate(chunks):
            try:
                async def upsert_chunk():
                    return await self.add_documents(chunk, target_collection, context)

                if config.retry_failed:
                    chunk_ids = await execute_with_retry(
                        upsert_chunk,
                        max_retries=config.max_retries,
                        retryable_exceptions=(VectorStorageError, psycopg.Error if psycopg else Exception)
                    )
                else:
                    chunk_ids = await upsert_chunk()

                successful.extend(chunk_ids)
                logger.debug(f"Batch chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_ids)} documents")

            except Exception as e:
                error_msg = str(e)
                if config.on_error == "raise":
                    raise VectorStorageError(f"Batch upsert failed at chunk {chunk_idx}: {e}")
                elif config.on_error == "collect":
                    for doc in chunk:
                        failed.append((doc, error_msg))
                # "skip" just continues

        duration_ms = (time.time() - start_time) * 1000

        result = BatchResult(
            successful=successful,
            failed=[(doc.id, msg) for doc, msg in failed],
            total=len(documents),
            duration_ms=duration_ms
        )

        logger.info(
            f"Batch upsert complete: {result.success_count}/{result.total} succeeded "
            f"({result.success_rate:.1f}%) in {result.duration_ms:.0f}ms"
        )

        return result

    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        context: Optional[StorageContext] = None,
        search_config: Optional[VectorSearchConfig] = None
    ) -> List[ContextDocument]:
        """
        Perform similarity search with user isolation and metadata filtering.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            collection_name: Collection to search
            filter_metadata: Metadata filters
            context: Storage context for user isolation
            search_config: Advanced search configuration

        Returns:
            List of matching documents ordered by similarity
        """
        target_collection = collection_name or self._default_collection_name
        user_id = context.user_id if context else None
        config = search_config or VectorSearchConfig(k=k, filter_metadata=filter_metadata)

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                return await self._similarity_search_tenant(
                    query_embedding, k, target_collection, user_id, config
                )
            else:
                return await self._similarity_search_pool(
                    query_embedding, k, target_collection, user_id, config
                )
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise VectorStorageError(f"Search failed: {e}")

    async def _similarity_search_pool(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: str,
        user_id: Optional[str],
        config: VectorSearchConfig
    ) -> List[ContextDocument]:
        """Similarity search using connection pool."""
        if not dict_row:
            raise VectorStorageError("psycopg dict_row not available")

        results: List[ContextDocument] = []
        query_dim = len(query_embedding)

        async with self._pool.connection() as conn:
            if register_vector_async:
                await register_vector_async(conn)

            # Set HNSW search parameter
            if config.hnsw_config:
                await conn.execute(config.hnsw_config.to_search_options())
            else:
                await conn.execute(f"SET hnsw.ef_search = {self._hnsw_config.ef_search}")

            # Verify collection exists and get dimension
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT vector_dimension FROM {self._collections_table} WHERE name = %s",
                    (collection_name,)
                )
                coll = await cur.fetchone()
                if not coll:
                    raise VectorStorageError(f"Collection '{collection_name}' not found")

                if query_dim != coll["vector_dimension"]:
                    raise VectorStorageError(
                        f"Query dimension {query_dim} doesn't match collection dimension {coll['vector_dimension']}"
                    )

            # Build query - cast embedding to vector type for pgvector operators
            distance_op = config.get_distance_operator()
            sql = f"""
                SELECT id, content, metadata, embedding {distance_op} %s::vector AS distance
                FROM {self._vectors_table}
                WHERE collection_name = %s
            """
            params: List[Any] = [query_embedding, collection_name]

            # User isolation
            if user_id:
                sql += " AND (user_id = %s OR user_id IS NULL)"
                params.append(user_id)

            # Metadata filters
            if config.filter_metadata:
                for key, value in config.filter_metadata.items():
                    sql += f" AND metadata->>%s = %s"
                    params.extend([key, str(value)])

            # Minimum score threshold
            if config.min_score is not None:
                # For cosine distance, lower is better (0 = identical)
                # Convert min_score (similarity) to max_distance
                max_distance = 1.0 - config.min_score
                sql += f" AND embedding {distance_op} %s::vector < %s"
                params.extend([query_embedding, max_distance])

            sql += f" ORDER BY distance ASC LIMIT %s"
            params.append(k)

            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(sql, tuple(params))
                async for row in cur:
                    # Convert distance to similarity score
                    distance = float(row["distance"]) if row.get("distance") is not None else 0.0
                    score = 1.0 - distance if config.distance_metric == "cosine" else -distance

                    doc = ContextDocument(
                        id=row["id"],
                        content=row.get("content", ""),
                        metadata=row.get("metadata") or {},
                        score=score
                    )

                    if config.include_embeddings:
                        # Would need to fetch embedding separately or include in query
                        pass

                    results.append(doc)

        logger.debug(f"Similarity search returned {len(results)} results from '{collection_name}'")
        return results

    async def _similarity_search_tenant(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: str,
        user_id: Optional[str],
        config: VectorSearchConfig
    ) -> List[ContextDocument]:
        """Similarity search using tenant session."""
        results: List[ContextDocument] = []
        distance_op = config.get_distance_operator()

        # Build parameterized query - cast to vector type for pgvector operators
        sql = f"""
            SELECT id, content, metadata, embedding {distance_op} :embedding::vector AS distance
            FROM {self._vectors_table}
            WHERE collection_name = :collection
        """
        params: Dict[str, Any] = {
            "embedding": str(query_embedding),
            "collection": collection_name,
        }

        if user_id:
            sql += " AND (user_id = :user_id OR user_id IS NULL)"
            params["user_id"] = user_id

        if config.filter_metadata:
            for i, (key, value) in enumerate(config.filter_metadata.items()):
                sql += f" AND metadata->>:filter_key_{i} = :filter_val_{i}"
                params[f"filter_key_{i}"] = key
                params[f"filter_val_{i}"] = str(value)

        sql += " ORDER BY distance ASC LIMIT :k"
        params["k"] = k

        result = await self._tenant_session.execute(text(sql), params)
        for row in result.fetchall():
            row_map = row._mapping
            distance = float(row_map["distance"]) if row_map.get("distance") is not None else 0.0
            score = 1.0 - distance if config.distance_metric == "cosine" else -distance

            results.append(ContextDocument(
                id=row_map["id"],
                content=row_map.get("content", ""),
                metadata=json.loads(row_map.get("metadata") or '{}'),
                score=score
            ))

        return results

    # =========================================================================
    # HYBRID SEARCH
    # =========================================================================

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 10,
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None,
        vector_weight: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from:
        1. Vector similarity search
        2. Trigram-based full-text search

        Args:
            query_text: Text query for full-text search
            query_embedding: Query vector for similarity search
            k: Number of results to return
            collection_name: Collection to search
            context: Storage context for user isolation
            vector_weight: Weight for vector similarity (0-1)
            filter_metadata: Metadata filters

        Returns:
            List of HybridSearchResult with combined scores
        """
        target_collection = collection_name or self._default_collection_name
        user_id = context.user_id if context else None
        text_weight = 1.0 - vector_weight

        if not self._pool:
            raise VectorStorageError("Connection pool not initialized")

        try:
            async with self._pool.connection() as conn:
                if register_vector_async:
                    await register_vector_async(conn)

                # Set HNSW search parameter
                await conn.execute(f"SET hnsw.ef_search = {self._hnsw_config.ef_search}")

                # Build hybrid query using RRF (Reciprocal Rank Fusion)
                # RRF score = 1/(k + rank_vector) * vector_weight + 1/(k + rank_text) * text_weight
                rrf_k = 60  # Standard RRF constant

                # CTE for vector search with ranks - cast to vector type for pgvector operators
                vector_cte = f"""
                    vector_search AS (
                        SELECT id, content, metadata,
                               embedding <=> %s::vector AS vector_distance,
                               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vector_rank
                        FROM {self._vectors_table}
                        WHERE collection_name = %s
                """

                params: List[Any] = [query_embedding, query_embedding, target_collection]

                if user_id:
                    vector_cte += " AND (user_id = %s OR user_id IS NULL)"
                    params.append(user_id)

                if filter_metadata:
                    for key, value in filter_metadata.items():
                        vector_cte += f" AND metadata->>%s = %s"
                        params.extend([key, str(value)])

                vector_cte += f" LIMIT {k * 3})"  # Fetch more for RRF

                # CTE for text search with ranks
                text_cte = f"""
                    text_search AS (
                        SELECT id, content, metadata,
                               1 - similarity(content, %s) AS text_distance,
                               ROW_NUMBER() OVER (ORDER BY similarity(content, %s) DESC) AS text_rank
                        FROM {self._vectors_table}
                        WHERE collection_name = %s AND content IS NOT NULL
                """

                params.extend([query_text, query_text, target_collection])

                if user_id:
                    text_cte += " AND (user_id = %s OR user_id IS NULL)"
                    params.append(user_id)

                if filter_metadata:
                    for key, value in filter_metadata.items():
                        text_cte += f" AND metadata->>%s = %s"
                        params.extend([key, str(value)])

                text_cte += f" LIMIT {k * 3})"

                # Combine using RRF
                sql = f"""
                    WITH {vector_cte}, {text_cte}
                    SELECT
                        COALESCE(v.id, t.id) AS id,
                        COALESCE(v.content, t.content) AS content,
                        COALESCE(v.metadata, t.metadata) AS metadata,
                        COALESCE(v.vector_distance, 1.0) AS vector_distance,
                        COALESCE(t.text_distance, 1.0) AS text_distance,
                        COALESCE(v.vector_rank, {k * 3 + 1}) AS vector_rank,
                        COALESCE(t.text_rank, {k * 3 + 1}) AS text_rank,
                        (
                            {vector_weight} / ({rrf_k} + COALESCE(v.vector_rank, {k * 3 + 1})) +
                            {text_weight} / ({rrf_k} + COALESCE(t.text_rank, {k * 3 + 1}))
                        ) AS rrf_score
                    FROM vector_search v
                    FULL OUTER JOIN text_search t ON v.id = t.id
                    ORDER BY rrf_score DESC
                    LIMIT %s
                """
                params.append(k)

                results: List[HybridSearchResult] = []
                conn.row_factory = dict_row

                async with conn.cursor() as cur:
                    await cur.execute(sql, tuple(params))
                    rank = 1
                    async for row in cur:
                        vector_score = 1.0 - float(row["vector_distance"])
                        text_score = 1.0 - float(row["text_distance"])

                        doc = ContextDocument(
                            id=row["id"],
                            content=row.get("content", ""),
                            metadata=row.get("metadata") or {},
                            score=float(row["rrf_score"])
                        )

                        results.append(HybridSearchResult(
                            document=doc,
                            vector_score=vector_score,
                            text_score=text_score,
                            combined_score=float(row["rrf_score"]),
                            rank=rank
                        ))
                        rank += 1

                logger.debug(f"Hybrid search returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise VectorStorageError(f"Hybrid search failed: {e}")

    # =========================================================================
    # COLLECTION MANAGEMENT
    # =========================================================================

    async def list_collection_names(self) -> List[str]:
        """List all collection names."""
        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                result = await self._tenant_session.execute(
                    text(f"SELECT name FROM {self._collections_table} ORDER BY name")
                )
                return [row[0] for row in result.fetchall()]
            else:
                async with self._pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(f"SELECT name FROM {self._collections_table} ORDER BY name")
                        return [row[0] async for row in cur]
        except Exception as e:
            logger.error(f"Error listing collections: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to list collections: {e}")

    async def get_collection_metadata(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a collection."""
        target = collection_name or self._default_collection_name

        try:
            if hasattr(self, '_tenant_session') and self._tenant_session is not None:
                result = await self._tenant_session.execute(
                    text(f"""
                        SELECT name, vector_dimension, description, embedding_model_provider,
                               embedding_model_name, hnsw_m, hnsw_ef_construction, created_at, metadata
                        FROM {self._collections_table} WHERE name = :name
                    """),
                    {"name": target}
                )
                row = result.fetchone()
            else:
                async with self._pool.connection() as conn:
                    conn.row_factory = dict_row
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"""
                            SELECT name, vector_dimension, description, embedding_model_provider,
                                   embedding_model_name, hnsw_m, hnsw_ef_construction, created_at, metadata
                            FROM {self._collections_table} WHERE name = %s
                            """,
                            (target,)
                        )
                        row = await cur.fetchone()

            if not row:
                return None

            if hasattr(row, '_mapping'):
                row = row._mapping

            return {
                "name": row["name"],
                "embedding_dimension": row["vector_dimension"],
                "description": row.get("description"),
                "embedding_model_provider": row.get("embedding_model_provider"),
                "embedding_model_name": row.get("embedding_model_name"),
                "hnsw_m": row.get("hnsw_m"),
                "hnsw_ef_construction": row.get("hnsw_ef_construction"),
                "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                "metadata": row.get("metadata") or {},
            }

        except Exception as e:
            logger.error(f"Error getting collection metadata: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to get collection metadata: {e}")

    async def get_collection_info(
        self,
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None
    ) -> Optional[CollectionInfo]:
        """
        Get detailed collection information including document count.

        Args:
            collection_name: Collection name
            context: Storage context (filters count by user if provided)

        Returns:
            CollectionInfo or None if not found
        """
        target = collection_name or self._default_collection_name
        user_id = context.user_id if context else None

        try:
            metadata = await self.get_collection_metadata(target)
            if not metadata:
                return None

            # Get document count
            count_sql = f"SELECT COUNT(*) FROM {self._vectors_table} WHERE collection_name = %s"
            params: List[Any] = [target]

            if user_id:
                count_sql += " AND (user_id = %s OR user_id IS NULL)"
                params.append(user_id)

            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(count_sql, tuple(params))
                    count_row = await cur.fetchone()
                    doc_count = count_row[0] if count_row else 0

            return CollectionInfo(
                name=metadata["name"],
                vector_dimension=metadata["embedding_dimension"],
                description=metadata.get("description"),
                embedding_model_provider=metadata.get("embedding_model_provider"),
                embedding_model_name=metadata.get("embedding_model_name"),
                created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None,
                metadata=metadata.get("metadata", {}),
                document_count=doc_count,
            )

        except Exception as e:
            logger.error(f"Error getting collection info: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to get collection info: {e}")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        description: Optional[str] = None,
        embedding_model_provider: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        hnsw_config: Optional[HNSWConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CollectionInfo:
        """
        Create a new vector collection.

        Args:
            name: Collection name (must be unique)
            dimension: Vector dimension
            description: Optional description
            embedding_model_provider: Provider name
            embedding_model_name: Model name
            hnsw_config: HNSW index configuration
            metadata: Additional metadata

        Returns:
            CollectionInfo for the created collection

        Raises:
            VectorStorageError: If collection already exists
        """
        config = hnsw_config or self._hnsw_config

        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    # Check if collection exists
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"SELECT 1 FROM {self._collections_table} WHERE name = %s",
                            (name,)
                        )
                        if await cur.fetchone():
                            raise VectorStorageError(f"Collection '{name}' already exists")

                    # Create collection
                    await self._ensure_collection_exists(
                        conn, name, dimension,
                        description=description,
                        provider=embedding_model_provider,
                        model_name=embedding_model_name,
                        metadata=metadata
                    )

            return CollectionInfo(
                name=name,
                vector_dimension=dimension,
                description=description,
                embedding_model_provider=embedding_model_provider,
                embedding_model_name=embedding_model_name,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
                document_count=0,
            )

        except VectorStorageError:
            raise
        except Exception as e:
            logger.error(f"Error creating collection: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to create collection: {e}")

    async def delete_collection(
        self,
        collection_name: str,
        force: bool = False
    ) -> bool:
        """
        Delete a collection and all its documents.

        Args:
            collection_name: Collection to delete
            force: If True, delete even if collection has documents

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    # Check document count
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"SELECT COUNT(*) FROM {self._vectors_table} WHERE collection_name = %s",
                            (collection_name,)
                        )
                        count = (await cur.fetchone())[0]

                        if count > 0 and not force:
                            raise VectorStorageError(
                                f"Collection '{collection_name}' has {count} documents. "
                                "Use force=True to delete anyway."
                            )

                        # Delete collection (cascades to vectors)
                        await cur.execute(
                            f"DELETE FROM {self._collections_table} WHERE name = %s",
                            (collection_name,)
                        )
                        deleted = cur.rowcount > 0

            if deleted:
                logger.info(f"Deleted collection '{collection_name}' ({count} documents)")
            return deleted

        except VectorStorageError:
            raise
        except Exception as e:
            logger.error(f"Error deleting collection: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to delete collection: {e}")

    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None
    ) -> bool:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete
            collection_name: Collection containing documents
            context: Storage context for user isolation

        Returns:
            True if any documents were deleted
        """
        if not document_ids:
            return True

        target = collection_name or self._default_collection_name
        user_id = context.user_id if context else None

        try:
            sql = f"DELETE FROM {self._vectors_table} WHERE collection_name = %s AND id = ANY(%s::TEXT[])"
            params: List[Any] = [target, document_ids]

            if user_id:
                sql += " AND user_id = %s"
                params.append(user_id)

            async with self._pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(sql, tuple(params))
                        deleted = cur.rowcount

            logger.info(f"Deleted {deleted} documents from '{target}'")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting documents: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to delete documents: {e}")

    async def get_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
        context: Optional[StorageContext] = None,
        include_embedding: bool = False
    ) -> Optional[ContextDocument]:
        """
        Get a single document by ID.

        Args:
            document_id: Document ID
            collection_name: Collection containing document
            context: Storage context for user isolation
            include_embedding: Whether to include embedding in result

        Returns:
            ContextDocument or None if not found
        """
        target = collection_name or self._default_collection_name
        user_id = context.user_id if context else None

        try:
            select_cols = "id, content, metadata"
            if include_embedding:
                select_cols += ", embedding"

            sql = f"SELECT {select_cols} FROM {self._vectors_table} WHERE collection_name = %s AND id = %s"
            params: List[Any] = [target, document_id]

            if user_id:
                sql += " AND (user_id = %s OR user_id IS NULL)"
                params.append(user_id)

            async with self._pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    await cur.execute(sql, tuple(params))
                    row = await cur.fetchone()

            if not row:
                return None

            doc = ContextDocument(
                id=row["id"],
                content=row.get("content", ""),
                metadata=row.get("metadata") or {},
            )

            if include_embedding and row.get("embedding"):
                doc.embedding = list(row["embedding"])

            return doc

        except Exception as e:
            logger.error(f"Error getting document: {e}", exc_info=True)
            raise VectorStorageError(f"Failed to get document: {e}")

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self._pool:
            pool = self._pool
            self._pool = None
            try:
                logger.info("Closing PgVector storage connection pool...")
                await pool.close()
                logger.info("PgVector storage connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}", exc_info=True)

        self._initialized = False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnhancedPgVectorStorage",
    "CollectionInfo",
    "HybridSearchResult",
]
