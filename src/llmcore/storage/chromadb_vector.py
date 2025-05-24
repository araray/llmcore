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
from typing import (TYPE_CHECKING, Any, Dict, List,
                    Optional, Tuple)

# Import chromadb client library conditionally
try:
    import chromadb
    from chromadb.api.models.Collection import \
        Collection as ChromaCollection
    from chromadb.errors import (
        AuthorizationError, IDAlreadyExistsError)
    chromadb_available = True
    ChromaClientType = chromadb.Client
except ImportError:
    chromadb_available = False
    chromadb = None # type: ignore
    ChromaCollection = None # type: ignore
    IDAlreadyExistsError = Exception # type: ignore
    AuthorizationError = Exception # type: ignore
    ChromaClientType = "chromadb.Client" # type: ignore


from ..exceptions import ConfigError, VectorStorageError
from ..models import ContextDocument
from .base_vector import BaseVectorStorage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    if chromadb_available:
        from chromadb.Client import Client as ActualChromaClientType
    else:
        ActualChromaClientType = Any


class ChromaVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using ChromaDB.

    Connects to a persistent ChromaDB instance based on the configured path.
    Operations are run in threads using asyncio.to_thread as the chromadb
    client is primarily synchronous.
    """
    _client: Optional[ChromaClientType] = None
    _storage_path: Optional[str] = None
    _default_collection_name: str = "llmcore_default_rag"
    _collection_cache: Dict[str, ChromaCollection] = {}

    def _sync_initialize(self, config: Dict[str, Any]) -> None:
        """Synchronous initialization logic for ChromaDB client."""
        if not chromadb_available:
            raise ImportError("ChromaDB client library not installed. Please install `chromadb` or `llmcore[chromadb]`.")

        self._storage_path = config.get("path")
        self._default_collection_name = config.get("default_collection", self._default_collection_name)

        try:
            if self._storage_path:
                expanded_path = os.path.expanduser(self._storage_path)
                pathlib.Path(expanded_path).parent.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=expanded_path)
                logger.info(f"ChromaDB persistent client initialized at: {expanded_path}")
            else:
                self._client = chromadb.Client()
                logger.info("ChromaDB in-memory client initialized.")

            self._client.heartbeat()
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
                where=filter_metadata, # type: ignore
                include=['metadatas', 'documents', 'distances']
            )
            logger.debug(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results from collection '{target_collection_name}'.")

            context_docs: List[ContextDocument] = []
            ids_list = results.get('ids')
            if not ids_list or not ids_list[0]:
                return []

            ids = ids_list[0]
            distances_list = results.get('distances')
            distances = distances_list[0] if distances_list and distances_list[0] is not None else [None] * len(ids)

            metadatas_list = results.get('metadatas')
            metadatas = metadatas_list[0] if metadatas_list and metadatas_list[0] is not None else [{}] * len(ids)

            contents_list = results.get('documents')
            contents = contents_list[0] if contents_list and contents_list[0] is not None else [""] * len(ids)

            for i, doc_id in enumerate(ids):
                context_docs.append(
                    ContextDocument(
                        id=doc_id,
                        content=contents[i] if i < len(contents) else "",
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        score=distances[i] if i < len(distances) else None,
                        embedding=None
                    )
                )
            return context_docs
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
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB collection '{target_collection_name}': {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB delete_documents failed: {e}")

    def _sync_list_collection_names(self) -> List[str]:
        """Synchronous logic to list collection names."""
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB list_collections failed: {e}")

    def _sync_close(self) -> None:
        """Synchronous closing logic for ChromaDB."""
        if self._client:
            try:
                logger.info("ChromaDB client cleanup: No explicit reset/close called. Resources are typically managed by the client's lifecycle or garbage collection.")
            except AuthorizationError as auth_e:
                logger.warning(f"ChromaDB client reset is disabled by config and was attempted: {auth_e}")
            except Exception as e:
                logger.error(f"Error during ChromaDB client resource management (if any explicit close was attempted): {e}", exc_info=True)
            finally:
                 self._client = None
                 self._collection_cache.clear()
        logger.info("ChromaDB vector storage resources cleared/dereferenced.")

    async def initialize(self, config: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._sync_initialize, config)

    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        return await asyncio.to_thread(self._sync_add_documents, documents, collection_name)

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        return await asyncio.to_thread(
            self._sync_similarity_search,
            query_embedding, k, collection_name, filter_metadata
        )

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        return await asyncio.to_thread(self._sync_delete_documents, document_ids, collection_name)

    async def list_collection_names(self) -> List[str]:
        """Lists the names of all available collections in the vector store asynchronously."""
        return await asyncio.to_thread(self._sync_list_collection_names)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)
