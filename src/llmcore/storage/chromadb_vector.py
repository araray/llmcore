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
    from chromadb.errors import ( # type: ignore[attr-defined] # errors module exists
        AuthorizationError, IDAlreadyExistsError, CollectionNotFoundError) # Added CollectionNotFoundError
    chromadb_available = True
    ChromaClientType = chromadb.Client
except ImportError:
    chromadb_available = False
    chromadb = None # type: ignore
    ChromaCollection = None # type: ignore
    IDAlreadyExistsError = Exception # type: ignore
    AuthorizationError = Exception # type: ignore
    CollectionNotFoundError = Exception # type: ignore
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
                # Ensure the parent directory exists for ChromaDB's persistent storage path
                pathlib.Path(expanded_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=expanded_path)
                logger.info(f"ChromaDB persistent client initialized at: {expanded_path}")
            else:
                self._client = chromadb.Client() # In-memory client
                logger.info("ChromaDB in-memory client initialized.")

            # Test connection (heartbeat might not be available for all client types or versions)
            # A simple way to test is to try listing collections.
            if self._client:
                self._client.list_collections() # This will raise if client is not properly connected/configured
                logger.debug("ChromaDB client connection confirmed via list_collections().")
            else:
                # This case should ideally be caught by the client instantiation itself.
                raise VectorStorageError("ChromaDB client object is None after initialization attempt.")


        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client (path: {self._storage_path}): {e}", exc_info=True)
            self._client = None
            raise VectorStorageError(f"Could not initialize ChromaDB client: {e}")

    def _sync_get_collection(self, collection_name: Optional[str]) -> ChromaCollection:
        """
        Synchronously gets or creates a ChromaDB collection.
        Caches retrieved collection objects.

        Raises:
            VectorStorageError: If client not initialized.
            ConfigError: If collection name is invalid or access fails.
        """
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")

        target_collection_name = collection_name or self._default_collection_name
        if not target_collection_name:
             raise ConfigError("Vector storage collection name is not specified or configured.")

        if target_collection_name in self._collection_cache:
            return self._collection_cache[target_collection_name]

        try:
            # get_or_create_collection is idempotent and suitable here.
            # Default distance metric (cosine) is often set at collection creation.
            # If apykatu sets it, this will just retrieve it.
            # If LLMCore is the first to access, it might create with ChromaDB's default.
            collection = self._client.get_or_create_collection(
                name=target_collection_name,
                # metadata={"hnsw:space": "cosine"} # Example if creating new
            )
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
        target_collection_name = collection.name # Use actual name from collection object

        doc_ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        contents: List[str] = []

        for doc in documents:
            if not doc.id: raise VectorStorageError("Document must have an ID.")
            if not doc.embedding: raise VectorStorageError(f"Document '{doc.id}' must have an embedding.")
            doc_ids.append(doc.id)
            embeddings.append(doc.embedding)
            # Ensure metadata is a flat dictionary of supported types by ChromaDB
            # (str, int, float, bool). Pydantic models or complex nested structures
            # in doc.metadata need to be serialized or flattened appropriately.
            # For now, assume doc.metadata is already compliant or ChromaDB handles simple dicts.
            serializable_meta = {}
            if doc.metadata:
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        serializable_meta[k] = v
                    else:
                        # Attempt to serialize other types, log warning if complex
                        try:
                            serializable_meta[k] = str(v) # Simple string conversion
                            if not isinstance(v, (list, dict)): # Avoid logging for simple lists/dicts
                                logger.debug(f"Metadata value for key '{k}' in doc '{doc.id}' was converted to string: '{str(v)[:50]}...'")
                        except:
                            logger.warning(f"Metadata value for key '{k}' in doc '{doc.id}' is complex and could not be easily serialized to string. Skipping this metadata field.")
            metadatas.append(serializable_meta)
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
        target_collection_name = collection.name

        try:
            # ChromaDB's query method expects query_embeddings as List[List[float]]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata, # type: ignore [arg-type] # ChromaDB expects Dict[str, WhereValue]
                include=['metadatas', 'documents', 'distances']
            )
            logger.debug(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results from collection '{target_collection_name}'.")

            context_docs: List[ContextDocument] = []
            ids_list = results.get('ids')
            if not ids_list or not ids_list[0]: # Results are list of lists
                return []

            # Extract results for the single query embedding
            ids = ids_list[0]
            distances_list = results.get('distances')
            distances = distances_list[0] if distances_list and distances_list[0] is not None else [None] * len(ids)

            metadatas_list = results.get('metadatas')
            metadatas = metadatas_list[0] if metadatas_list and metadatas_list[0] is not None else [{}] * len(ids)

            contents_list = results.get('documents')
            contents = contents_list[0] if contents_list and contents_list[0] is not None else [""] * len(ids)

            for i, doc_id_val in enumerate(ids):
                # Ensure all components exist for the current index before creating ContextDocument
                doc_content = contents[i] if i < len(contents) and contents[i] is not None else ""
                doc_metadata = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
                doc_distance = distances[i] if i < len(distances) and distances[i] is not None else None

                context_docs.append(
                    ContextDocument(
                        id=str(doc_id_val), # Ensure ID is string
                        content=doc_content,
                        metadata=doc_metadata,
                        score=float(doc_distance) if doc_distance is not None else None, # Chroma uses distance as score
                        embedding=None # Embeddings are not typically returned by query
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
        target_collection_name = collection.name

        try:
            collection.delete(ids=document_ids)
            logger.info(f"Attempted deletion of {len(document_ids)} documents from ChromaDB collection '{target_collection_name}'.")
            return True # ChromaDB delete doesn't return specific success/fail count per ID easily
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

    def _sync_get_collection_metadata(self, collection_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Synchronous logic to get collection metadata."""
        try:
            collection = self._sync_get_collection(collection_name) # This handles default name
            # The .metadata attribute of a ChromaCollection object holds its metadata
            if collection.metadata is not None:
                # Ensure it's a plain dict for consistent return type
                return dict(collection.metadata)
            return None # Return None if metadata is explicitly None
        except CollectionNotFoundError: # Specific ChromaDB error if collection doesn't exist
            logger.warning(f"Collection '{collection_name or self._default_collection_name}' not found when trying to get metadata.")
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for ChromaDB collection '{collection_name or self._default_collection_name}': {e}", exc_info=True)
            # Re-raise as VectorStorageError for consistent API error type
            raise VectorStorageError(f"ChromaDB get_collection_metadata failed: {e}")


    def _sync_close(self) -> None:
        """Synchronous closing logic for ChromaDB."""
        if self._client:
            try:
                # ChromaDB's PersistentClient doesn't have an explicit close().
                # It relies on its __del__ for cleanup or the underlying SQLite connection
                # being closed if it's using duckdb. For server-based clients, reset() might be used.
                # For PersistentClient, simply dereferencing might be enough.
                # However, if reset is available and safe, it can be called.
                if hasattr(self._client, 'reset'): # Check if reset method exists
                    logger.info("Calling ChromaDB client.reset() to clear in-memory state.")
                    self._client.reset() # Resets the client's state, including heartbeat
                else:
                    logger.info("ChromaDB client (PersistentClient) does not have a direct close/reset method. Resources are typically managed by its lifecycle.")
            except AuthorizationError as auth_e: # Specific ChromaDB error
                logger.warning(f"ChromaDB client reset/cleanup might be restricted by config or permissions: {auth_e}")
            except Exception as e:
                logger.error(f"Error during ChromaDB client resource management: {e}", exc_info=True)
            finally:
                 self._client = None # Dereference to allow GC
                 self._collection_cache.clear()
        logger.info("ChromaDB vector storage resources cleared/dereferenced.")

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the ChromaDB client asynchronously by running sync init in a thread."""
        await asyncio.to_thread(self._sync_initialize, config)

    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Adds documents to ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_add_documents, documents, collection_name)

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        """Performs similarity search in ChromaDB asynchronously."""
        return await asyncio.to_thread(
            self._sync_similarity_search,
            query_embedding, k, collection_name, filter_metadata
        )

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Deletes documents from ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_delete_documents, document_ids, collection_name)

    async def list_collection_names(self) -> List[str]:
        """Lists collection names from ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_list_collection_names)

    async def get_collection_metadata(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a specific ChromaDB collection asynchronously.
        """
        return await asyncio.to_thread(self._sync_get_collection_metadata, collection_name)

    async def close(self) -> None:
        """Closes/resets the ChromaDB client asynchronously."""
        await asyncio.to_thread(self._sync_close)
