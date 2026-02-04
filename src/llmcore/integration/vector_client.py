# src/llmcore/integration/vector_client.py
"""
LLMCore Vector Client Adapter.

Phase 3 (SYMBIOSIS): Provides a ChromaDB-compatible interface that delegates
all vector operations to LLMCore's unified storage layer.

This adapter enables SemantiScan to seamlessly switch between direct ChromaDB
usage and LLMCore delegation via configuration.

Key Features:
- Same interface as ChromaDBClient (add_chunks, query)
- Delegates to LLMCore's add_documents_to_vector_store and search_vector_store
- Supports batch operations with retry logic
- Multi-user isolation via StorageContext
- Hybrid search support (when backend supports it)

Usage:
    from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig
    from llmcore.api import LLMCore

    # Create LLMCore instance
    llmcore = await LLMCore.create(config)

    # Create vector client config
    client_config = LLMCoreVectorClientConfig(
        collection_name="codebase_default",
        user_id="user_123",  # Optional for isolation
    )

    # Create client
    client = await LLMCoreVectorClient.create(llmcore, client_config)

    # Use like ChromaDBClient
    client.add_chunks(chunks, embeddings)
    results = client.query(query_embedding, top_k=10)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from llmcore.api import LLMCore

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOL DEFINITION
# =============================================================================


@runtime_checkable
class VectorClientProtocol(Protocol):
    """
    Protocol defining the minimal vector client interface.

    This protocol ensures compatibility between ChromaDBClient and
    LLMCoreVectorClient, enabling seamless backend switching.
    """

    def add_chunks(
        self,
        chunks: list[Any],  # List[Chunk] from semantiscan
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the vector store."""
        ...

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store for similar chunks."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LLMCoreVectorClientConfig:
    """
    Configuration for LLMCoreVectorClient.

    Attributes:
        collection_name: Name of the vector collection to use.
        user_id: Optional user ID for multi-tenant isolation.
        namespace: Optional namespace prefix for collection names.
        batch_size: Batch size for bulk operations (default: 100).
        retry_count: Number of retries for failed operations (default: 3).
        retry_delay: Delay between retries in seconds (default: 1.0).
        enable_hybrid_search: Enable hybrid search if backend supports it.
        vector_weight: Weight for vector similarity in hybrid search (default: 0.7).
    """

    collection_name: str = "codebase_default"
    user_id: str | None = None
    namespace: str | None = None
    batch_size: int = 100
    retry_count: int = 3
    retry_delay: float = 1.0
    enable_hybrid_search: bool = False
    vector_weight: float = 0.7

    def get_full_collection_name(self) -> str:
        """Get the fully qualified collection name with namespace."""
        if self.namespace:
            return f"{self.namespace}_{self.collection_name}"
        return self.collection_name


# =============================================================================
# CHUNK ADAPTER
# =============================================================================


@dataclass
class ChunkAdapter:
    """
    Adapter for converting SemantiScan Chunk objects to LLMCore format.

    SemantiScan uses its own Chunk dataclass with specific fields.
    This adapter handles the conversion without requiring direct import.
    """

    @staticmethod
    def chunk_to_document(chunk: Any, embedding: list[float]) -> dict[str, Any]:
        """
        Convert a SemantiScan Chunk to LLMCore document format.

        Args:
            chunk: SemantiScan Chunk object (duck-typed).
            embedding: Pre-computed embedding vector.

        Returns:
            Dictionary suitable for LLMCore add_documents_to_vector_store.
        """
        # Extract fields using duck typing (Chunk has id, content, metadata)
        chunk_id = getattr(chunk, "id", str(uuid.uuid4()))
        content = getattr(chunk, "content", "")
        metadata = getattr(chunk, "metadata", {})

        # Build core metadata (same filtering as ChromaDBClient)
        core_metadata = {}

        # Required core keys for filtering
        required_core_keys = ["file_path", "repo_name", "commit_hash"]
        for key in required_core_keys:
            value = metadata.get(key)
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    core_metadata[key] = value
                else:
                    # Convert to string for compatibility
                    core_metadata[key] = str(value)

        # Additional useful metadata
        simple_keys = ["start_line", "end_line", "language", "chunk_type", "chunk_method"]
        for key in simple_keys:
            value = metadata.get(key)
            if value is not None and isinstance(value, (str, int, float, bool)):
                core_metadata[key] = value

        return {
            "id": chunk_id,
            "content": content,
            "metadata": core_metadata,
            "embedding": embedding,  # Pre-computed embedding
        }

    @staticmethod
    def document_to_result(
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        distance: float,
    ) -> dict[str, Any]:
        """
        Convert LLMCore search result to ChromaDB-compatible format.

        Args:
            doc_id: Document ID.
            content: Document content.
            metadata: Document metadata.
            distance: Similarity distance (lower is better).

        Returns:
            Dictionary matching ChromaDB query result format.
        """
        return {
            "id": doc_id,
            "distance": distance,
            "metadata": metadata,
            "document": content,
        }


# =============================================================================
# MAIN CLIENT CLASS
# =============================================================================


class LLMCoreVectorClient:
    """
    LLMCore Vector Client - ChromaDB-compatible interface with LLMCore delegation.

    This class provides the same interface as SemantiScan's ChromaDBClient,
    but delegates all vector operations to LLMCore's storage layer.

    This enables SemantiScan to use LLMCore's unified storage system
    (PostgreSQL/pgvector or ChromaDB) via configuration, without code changes.

    Thread Safety:
        The client uses an internal event loop for async-to-sync bridging.
        Create one client per thread for concurrent usage.

    Example:
        ```python
        from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig

        # Configuration
        config = LLMCoreVectorClientConfig(
            collection_name="my_codebase",
            user_id="project_alpha",
        )

        # Create client
        client = await LLMCoreVectorClient.create(llmcore_instance, config)

        # Add chunks (same as ChromaDBClient)
        client.add_chunks(chunks, embeddings)

        # Query (same as ChromaDBClient)
        results = client.query(query_embedding, top_k=10, where_filter={"repo_name": "myrepo"})
        ```
    """

    def __init__(
        self,
        llmcore: LLMCore,
        config: LLMCoreVectorClientConfig,
    ):
        """
        Initialize the vector client.

        Note: Use `LLMCoreVectorClient.create()` factory method for async initialization.

        Args:
            llmcore: Initialized LLMCore instance.
            config: Client configuration.
        """
        self._llmcore = llmcore
        self._config = config
        self._collection_name = config.get_full_collection_name()
        self._initialized = False

        # Event loop for async-to-sync bridging
        self._loop: asyncio.AbstractEventLoop | None = None

        logger.info(
            f"LLMCoreVectorClient created: collection='{self._collection_name}', "
            f"user_id={config.user_id}"
        )

    @classmethod
    async def create(
        cls,
        llmcore: LLMCore,
        config: LLMCoreVectorClientConfig | None = None,
    ) -> LLMCoreVectorClient:
        """
        Factory method to create and initialize the vector client.

        Args:
            llmcore: Initialized LLMCore instance.
            config: Client configuration (uses defaults if not provided).

        Returns:
            Initialized LLMCoreVectorClient instance.

        Raises:
            RuntimeError: If initialization fails.
        """
        if config is None:
            config = LLMCoreVectorClientConfig()

        client = cls(llmcore, config)
        await client._initialize()
        return client

    async def _initialize(self) -> None:
        """
        Perform async initialization.

        This ensures the collection exists in the vector store.
        """
        try:
            # Verify LLMCore is ready
            if not hasattr(self._llmcore, "_storage_manager"):
                raise RuntimeError("LLMCore instance not properly initialized")

            # For pgvector backend, we might need to create the collection
            # For ChromaDB, get_or_create_collection is implicit
            vector_storage = self._llmcore._storage_manager.vector_storage

            # Check if we have EnhancedPgVectorStorage
            if hasattr(vector_storage, "create_collection"):
                try:
                    # Try to get or create the collection
                    from llmcore.storage.abstraction import StorageContext

                    ctx = StorageContext(user_id=self._config.user_id)
                    await vector_storage.create_collection(
                        self._collection_name,
                        vector_dimension=None,  # Will be set on first insert
                        if_not_exists=True,
                        context=ctx,
                    )
                except Exception as e:
                    # Collection might already exist, log and continue
                    logger.debug(f"Collection creation note: {e}")

            self._initialized = True
            logger.info(f"LLMCoreVectorClient initialized: collection='{self._collection_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize LLMCoreVectorClient: {e}", exc_info=True)
            raise RuntimeError(f"LLMCoreVectorClient initialization failed: {e}") from e

    def _run_async(self, coro):
        """
        Run an async coroutine in a sync context.

        This handles the common case where SemantiScan calls sync methods
        but LLMCore uses async internally.

        Strategy:
        - If no event loop is running: use asyncio.run() directly
        - If an event loop IS running: run in a new thread to avoid blocking
        """
        import concurrent.futures

        try:
            asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run() directly
            # Run the coroutine in a separate thread with its own event loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=60.0)  # 60s timeout
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(coro)

    # =========================================================================
    # PUBLIC INTERFACE (ChromaDB-compatible)
    # =========================================================================

    def add_chunks(
        self,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add chunks and their embeddings to the vector store.

        This method matches ChromaDBClient.add_chunks() interface.

        Args:
            chunks: List of SemantiScan Chunk objects.
            embeddings: List of embedding vectors (parallel to chunks).

        Raises:
            ValueError: If chunks and embeddings length mismatch.
            RuntimeError: If storage operation fails.
        """
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            logger.warning(
                f"Mismatch between chunks ({len(chunks) if chunks else 0}) "
                f"and embeddings ({len(embeddings) if embeddings else 0}). Skipping."
            )
            return

        logger.info(f"Adding {len(chunks)} chunks to collection '{self._collection_name}'...")

        try:
            # Convert chunks to LLMCore document format
            documents = [
                ChunkAdapter.chunk_to_document(chunk, embedding)
                for chunk, embedding in zip(chunks, embeddings)
            ]

            # Run the async operation
            self._run_async(self._add_documents_async(documents))

            logger.info(f"Successfully added {len(chunks)} chunks.")

        except Exception as e:
            logger.error(
                f"Failed to add chunks to collection '{self._collection_name}': {e}", exc_info=True
            )
            # Don't re-raise to match ChromaDBClient behavior

    async def _add_documents_async(self, documents: list[dict[str, Any]]) -> list[str]:
        """
        Async implementation of document addition.

        Args:
            documents: List of document dicts with id, content, metadata, embedding.

        Returns:
            List of added document IDs.
        """
        # Get vector storage
        vector_storage = self._llmcore._storage_manager.vector_storage

        # Check if we have batch support (EnhancedPgVectorStorage)
        if hasattr(vector_storage, "batch_upsert_documents"):
            from llmcore.models import ContextDocument
            from llmcore.storage.abstraction import BatchConfig, StorageContext

            ctx = StorageContext(user_id=self._config.user_id)
            batch_config = BatchConfig(
                chunk_size=self._config.batch_size,
                max_retries=self._config.retry_count,
            )

            # Convert to ContextDocument objects
            context_docs = [
                ContextDocument(
                    id=doc["id"],
                    content=doc["content"],
                    embedding=doc.get("embedding"),
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]

            result = await vector_storage.batch_upsert_documents(
                documents=context_docs,
                collection_name=self._collection_name,
                context=ctx,
                batch_config=batch_config,
            )

            return [doc.id for doc in context_docs]

        else:
            # Fall back to standard add_documents
            # Note: We need to handle pre-computed embeddings
            return await self._llmcore.add_documents_to_vector_store(
                documents=documents,
                collection_name=self._collection_name,
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for similar chunks.

        This method matches ChromaDBClient.query() interface.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to retrieve.
            where_filter: Optional metadata filter dictionary.

        Returns:
            List of result dictionaries with id, distance, metadata, document.
        """
        log_msg = f"Querying collection '{self._collection_name}' with top_k={top_k}"
        if where_filter:
            log_msg += f", filter={where_filter}"
        logger.info(log_msg)

        try:
            results = self._run_async(self._query_async(query_embedding, top_k, where_filter))

            logger.info(f"Retrieved {len(results)} results.")
            return results

        except Exception as e:
            logger.error(
                f"Failed to query collection '{self._collection_name}': {e}", exc_info=True
            )
            return []

    async def _query_async(
        self,
        query_embedding: list[float],
        top_k: int,
        where_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """
        Async implementation of query.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results.
            where_filter: Metadata filter.

        Returns:
            List of result dictionaries.
        """
        vector_storage = self._llmcore._storage_manager.vector_storage

        # Check if we have enhanced similarity search
        if hasattr(vector_storage, "similarity_search"):
            from llmcore.storage.abstraction import StorageContext, VectorSearchConfig

            ctx = StorageContext(user_id=self._config.user_id)
            search_config = VectorSearchConfig(
                k=top_k,
                filter_metadata=where_filter,
            )

            results = await vector_storage.similarity_search(
                query_embedding=query_embedding,
                k=top_k,
                collection_name=self._collection_name,
                context=ctx,
                search_config=search_config,
            )

            # Convert to ChromaDB-compatible format
            formatted_results = []
            for doc in results:
                # doc is ContextDocument or similar
                formatted_results.append(
                    {
                        "id": doc.id,
                        "distance": getattr(doc, "score", 0.0) or getattr(doc, "distance", 0.0),
                        "metadata": doc.metadata or {},
                        "document": doc.content,
                    }
                )

            return formatted_results

        else:
            # Fall back to basic search via embedding manager
            results = await self._llmcore.search_vector_store(
                query="",  # Not used when we have embedding
                k=top_k,
                collection_name=self._collection_name,
                metadata_filter=where_filter,
            )

            # Convert to ChromaDB-compatible format
            formatted_results = []
            for doc in results:
                formatted_results.append(
                    {
                        "id": doc.id,
                        "distance": getattr(doc, "score", 0.0),
                        "metadata": doc.metadata or {},
                        "document": doc.content,
                    }
                )

            return formatted_results

    # =========================================================================
    # EXTENDED INTERFACE (Beyond ChromaDB)
    # =========================================================================

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None,
        vector_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        This is an extended feature available when using pgvector backend.

        Args:
            query_text: Text query for full-text search component.
            query_embedding: Embedding vector for similarity search component.
            top_k: Number of results.
            where_filter: Metadata filter.
            vector_weight: Weight for vector vs text (0.0-1.0, default from config).

        Returns:
            List of result dictionaries with combined scores.
        """
        if vector_weight is None:
            vector_weight = self._config.vector_weight

        vector_storage = self._llmcore._storage_manager.vector_storage

        if hasattr(vector_storage, "hybrid_search"):
            from llmcore.storage.abstraction import StorageContext

            ctx = StorageContext(user_id=self._config.user_id)

            results = await vector_storage.hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                k=top_k,
                collection_name=self._collection_name,
                context=ctx,
                vector_weight=vector_weight,
                filter_metadata=where_filter,
            )

            # Convert to ChromaDB-compatible format with hybrid scores
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "id": result.document.id,
                        "distance": 1.0 - result.combined_score,  # Convert score to distance
                        "metadata": result.document.metadata or {},
                        "document": result.document.content,
                        "vector_score": result.vector_score,
                        "text_score": result.text_score,
                        "combined_score": result.combined_score,
                    }
                )

            return formatted_results

        else:
            # Fall back to regular vector search
            logger.warning(
                "Hybrid search not available with current backend, "
                "falling back to vector-only search."
            )
            return await self._query_async(query_embedding, top_k, where_filter)

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Document count, or 0 if retrieval fails.
        """
        try:
            return self._run_async(self._count_async())
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0

    async def _count_async(self) -> int:
        """Async implementation of count."""
        vector_storage = self._llmcore._storage_manager.vector_storage

        if hasattr(vector_storage, "get_collection_info"):
            from llmcore.storage.abstraction import StorageContext

            ctx = StorageContext(user_id=self._config.user_id)
            info = await vector_storage.get_collection_info(self._collection_name, context=ctx)
            return info.document_count if info else 0

        return 0

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @property
    def user_id(self) -> str | None:
        """Get the user ID for isolation."""
        return self._config.user_id


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_vector_client(
    llmcore: LLMCore,
    collection_name: str = "codebase_default",
    user_id: str | None = None,
    **kwargs,
) -> LLMCoreVectorClient:
    """
    Factory function to create an LLMCore vector client.

    This is a convenience function for creating the client with minimal setup.

    Args:
        llmcore: Initialized LLMCore instance.
        collection_name: Name of the vector collection.
        user_id: Optional user ID for isolation.
        **kwargs: Additional config options.

    Returns:
        Initialized LLMCoreVectorClient instance.
    """
    config = LLMCoreVectorClientConfig(
        collection_name=collection_name,
        user_id=user_id,
        **kwargs,
    )
    return await LLMCoreVectorClient.create(llmcore, config)
