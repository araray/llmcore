# src/llmcore/storage/chromadb_vector.py
"""
ChromaDB vector storage implementation for the LLMCore library.

Uses the chromadb library to interact with a ChromaDB instance
(either persistent or in-memory).
"""

import asyncio
import logging
import os
import pathlib

from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING # Added TYPE_CHECKING


# Import chromadb client library conditionally
try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollection # Specific type hint
    from chromadb.errors import IDAlreadyExistsError, CollectionNotFoundError
    chromadb_available = True
    # Define type alias for cleaner hinting if available
    ChromaClientType = chromadb.Client
except ImportError:
    chromadb_available = False
    chromadb = None # type: ignore
    ChromaCollection = None # type: ignore
    IDAlreadyExistsError = Exception # type: ignore
    CollectionNotFoundError = Exception # type: ignore
    # Use a forward reference string if the library is not available
    ChromaClientType = "chromadb.Client" # type: ignore


from ..models import ContextDocument
from ..exceptions import VectorStorageError, ConfigError
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to allow type hint without runtime error if chromadb is missing
if TYPE_CHECKING:
    # This block is only evaluated by type checkers, not at runtime
    if chromadb_available:
        from chromadb.Client import Client as ActualChromaClientType
    else:
        ActualChromaClientType = Any # Fallback type hint for checker if import failed


class ChromaVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using ChromaDB.

    Connects to a persistent ChromaDB instance based on the configured path.
    Operations are run in threads using asyncio.to_thread as the chromadb
    client is primarily synchronous.
    """
    # Use the type alias or string literal for the hint
    _client: Optional[ChromaClientType] = None
    _storage_path: Optional[str] = None
    _default_collection_name: str = "llmcore_default_rag"
    _collection_cache: Dict[str, ChromaCollection] = {} # Cache for collection objects

    # --- Synchronous Helper Methods (to be run in thread) ---

    def _sync_initialize(self, config: Dict[str, Any]) -> None:
        """Synchronous initialization logic for ChromaDB client."""
        if not chromadb_available:
            # This error should be caught earlier by EmbeddingManager/StorageManager
            # based on config, but double-check here.
            raise ImportError("ChromaDB client library not installed. Please install `chromadb` or `llmcore[chromadb]`.")

        self._storage_path = config.get("path") # Path for persistent storage
        self._default_collection_name = config.get("default_collection", self._default_collection_name)

        try:
            if self._storage_path:
                expanded_path = os.path.expanduser(self._storage_path)
                # Ensure parent directory exists (sync is ok here)
                pathlib.Path(expanded_path).parent.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=expanded_path)
                logger.info(f"ChromaDB persistent client initialized at: {expanded_path}")
            else:
                self._client = chromadb.Client() # In-memory client
                logger.info("ChromaDB in-memory client initialized.")

            self._client.heartbeat() # Check connection
            logger.debug("ChromaDB client connection confirmed.")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client (path: {self._storage_path}): {e}", exc_info=True)
            self._client = None
            raise VectorStorageError(f"Could not initialize ChromaDB client: {e}")

    def _sync_get_collection(self, collection_name: Optional[str]) -> ChromaCollection:
        """Synchronously gets or creates a ChromaDB collection."""
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")

        target_collection_name = collection_name or self._default_collection_name
        if not target_collection_name:
             raise ConfigError("Vector storage collection name is not specified or configured.")

        if target_collection_name in self._collection_cache:
            return self._collection_cache[target_collection_name]

        try:
            # get_or_create_collection handles thread safety internally for client interactions
            collection = self._client.get_or_create_collection(name=target_collection_name)
            logger.debug(f"Accessed ChromaDB collection: '{target_collection_name}'")
            self._collection_cache[target_collection_name] = collection
            return collection
        except Exception as e:
            logger.error(f"Failed to get or create ChromaDB collection '{target_collection_name}': {e}", exc_info=True)
            raise ConfigError(f"Could not access ChromaDB collection '{target_collection_name}': {e}")

    def _sync_add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str]
    ) -> List[str]:
        """Synchronous logic to add documents."""
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection_name or self._default_collection_name

        doc_ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        contents: List[str] = []

        for doc in documents:
            if not doc.id: raise VectorStorageError("Document must have an ID.")
            if not doc.embedding: raise VectorStorageError(f"Document '{doc.id}' must have an embedding.")
            doc_ids.append(doc.id)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata or {})
            contents.append(doc.content)

        if not doc_ids: return []

        try:
            collection.upsert(ids=doc_ids, embeddings=embeddings, metadatas=metadatas, documents=contents)
            logger.info(f"Upserted {len(doc_ids)} documents into ChromaDB collection '{target_collection_name}'.")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB collection '{target_collection_name}': {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB add_documents failed: {e}")

    def _sync_similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[ContextDocument]:
        """Synchronous logic for similarity search."""
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection_name or self._default_collection_name

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata,
                include=['metadatas', 'documents', 'distances']
            )
            logger.debug(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results from collection '{target_collection_name}'.")

            context_docs: List[ContextDocument] = []
            if not results or not results.get('ids') or not results['ids'][0]: return []

            ids = results['ids'][0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            contents = results.get('documents', [[]])[0]

            for i, doc_id in enumerate(ids):
                context_docs.append(
                    ContextDocument(
                        id=doc_id,
                        content=contents[i] if contents and i < len(contents) else "",
                        metadata=metadatas[i] if metadatas and i < len(metadatas) else {},
                        score=distances[i] if distances and i < len(distances) else None,
                        embedding=None
                    )
                )
            return context_docs
        except CollectionNotFoundError:
             logger.error(f"ChromaDB collection '{target_collection_name}' not found during search.")
             raise ConfigError(f"Collection '{target_collection_name}' not found.")
        except Exception as e:
            logger.error(f"Failed similarity search in ChromaDB collection '{target_collection_name}': {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB similarity_search failed: {e}")

    def _sync_delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str]
    ) -> bool:
        """Synchronous logic to delete documents."""
        if not document_ids: return True
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection_name or self._default_collection_name

        try:
            collection.delete(ids=document_ids)
            logger.info(f"Attempted deletion of {len(document_ids)} documents from ChromaDB collection '{target_collection_name}'.")
            return True
        except CollectionNotFoundError:
             logger.error(f"ChromaDB collection '{target_collection_name}' not found during delete.")
             raise ConfigError(f"Collection '{target_collection_name}' not found for deletion.")
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB collection '{target_collection_name}': {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB delete_documents failed: {e}")

    def _sync_close(self) -> None:
        """Synchronous closing logic."""
        if self._client:
            try:
                if hasattr(self._client, 'reset'):
                    self._client.reset()
                logger.info("ChromaDB client reset/closed.")
            except Exception as e:
                logger.error(f"Error during ChromaDB client reset/close: {e}", exc_info=True)
            finally:
                 self._client = None
                 self._collection_cache.clear()

    # --- Async Interface Methods (using asyncio.to_thread) ---

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the ChromaDB client asynchronously."""
        await asyncio.to_thread(self._sync_initialize, config)

    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Add or update multiple documents asynchronously."""
        return await asyncio.to_thread(self._sync_add_documents, documents, collection_name)

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        """Perform a similarity search asynchronously."""
        return await asyncio.to_thread(
            self._sync_similarity_search,
            query_embedding, k, collection_name, filter_metadata
        )

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents asynchronously."""
        return await asyncio.to_thread(self._sync_delete_documents, document_ids, collection_name)

    async def close(self) -> None:
        """Clean up resources asynchronously."""
        await asyncio.to_thread(self._sync_close)
