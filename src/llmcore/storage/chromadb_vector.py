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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# --- Conditional Import for ChromaDB ---
_chromadb_module = None
_ChromaCollection_class = None
_ChromaClient_class = None
chromadb_available = False

# Base error types, default to generic Exception. These will be updated if specific imports succeed.
ChromaError: type[Exception] = Exception
AuthorizationError: type[Exception] = Exception
IDAlreadyExistsError: type[Exception] = Exception
CollectionNotFoundError: type[Exception] = Exception  # Primary target for robust fallback

try:
    import chromadb as _chromadb_module_imported

    _chromadb_module = _chromadb_module_imported
    logger.debug("Successfully imported 'chromadb' module.")

    from chromadb.api.models.Collection import Collection as _ChromaCollection_imported

    _ChromaCollection_class = _ChromaCollection_imported
    logger.debug("Successfully imported 'chromadb.api.models.Collection'.")

    _ChromaClient_class = (
        _chromadb_module.Client
    )  # Access Client after chromadb is confirmed imported
    chromadb_available = True
    logger.info(
        "Core ChromaDB components ('chromadb' module and 'Collection' model) imported successfully. ChromaDB is considered available."
    )

    # Attempt to import specific error types from chromadb.errors
    # These are best-effort imports; if they fail, the broader Exception type will be used.
    try:
        from chromadb.errors import ChromaError as ActualChromaError

        ChromaError = ActualChromaError  # type: ignore[misc, no-redef]
        logger.debug("Successfully imported 'chromadb.errors.ChromaError'.")
    except ImportError:
        logger.warning(
            "'chromadb.errors.ChromaError' not found. Using base 'Exception' for ChromaError."
        )

    try:
        from chromadb.errors import AuthorizationError as ActualAuthorizationError

        AuthorizationError = ActualAuthorizationError  # type: ignore[misc, no-redef]
        logger.debug("Successfully imported 'chromadb.errors.AuthorizationError'.")
    except ImportError:
        logger.warning(
            "'chromadb.errors.AuthorizationError' not found. Using base 'Exception' for AuthorizationError."
        )

    try:
        from chromadb.errors import IDAlreadyExistsError as ActualIDAlreadyExistsError

        IDAlreadyExistsError = ActualIDAlreadyExistsError  # type: ignore[misc, no-redef]
        logger.debug("Successfully imported 'chromadb.errors.IDAlreadyExistsError'.")
    except ImportError:
        logger.warning(
            "'chromadb.errors.IDAlreadyExistsError' not found. Using base 'Exception' for IDAlreadyExistsError."
        )

    try:
        from chromadb.errors import NotFoundError as ActualNotFoundError

        CollectionNotFoundError = ActualNotFoundError  # type: ignore[misc, no-redef]
        logger.info("Aliased 'CollectionNotFoundError' to 'chromadb.errors.NotFoundError'.")
    except ImportError:
        logger.warning(
            "'chromadb.errors.NotFoundError' also not found. "
            "Aliasing 'CollectionNotFoundError' to 'ChromaError' (if available and not base Exception) or base 'Exception'."
        )
        if ChromaError is not Exception:  # Check if ChromaError was successfully imported
            CollectionNotFoundError = ChromaError  # type: ignore[misc, no-redef]


except ImportError:
    # This block is hit if 'import chromadb' or 'from chromadb.api.models.Collection import Collection' fails.
    logger.error(
        "Failed to import core 'chromadb' library or 'Collection' model. "
        "ChromaDB functionality will be unavailable."
    )
    chromadb_available = False
    _chromadb_module = None
    _ChromaCollection_class = None
    _ChromaClient_class = None
    # All error types remain their default 'Exception' as defined above.

# --- End Conditional Import ---


from ..exceptions import ConfigError, VectorStorageError
from ..models import ContextDocument
from .base_vector import BaseVectorStorage

if TYPE_CHECKING:
    if _chromadb_module and _ChromaClient_class:
        ActualChromaClientType = _ChromaClient_class
    else:
        ActualChromaClientType = Any  # Fallback for type checker if imports failed


class ChromaVectorStorage(BaseVectorStorage):
    """
    Manages persistence and retrieval of vector embeddings using ChromaDB.

    Connects to a persistent ChromaDB instance based on the configured path.
    Operations are run in threads using asyncio.to_thread as the chromadb
    client is primarily synchronous.
    """

    _client: Any | None = None  # Use Any for client type due to conditional import
    _storage_path: str | None = None
    _default_collection_name: str = "llmcore_default_rag"
    _collection_cache: dict[str, Any] = {}  # Use Any for collection type

    def _sync_initialize(self, config: dict[str, Any]) -> None:
        """Synchronous initialization logic for ChromaDB client."""
        if not chromadb_available or not _chromadb_module or not _ChromaClient_class:
            raise ImportError(
                "Core ChromaDB client library components not installed or accessible. Please install/reinstall `chromadb` or `llmcore[chromadb]`."
            )

        self._storage_path = config.get("path")
        self._default_collection_name = config.get(
            "default_collection", self._default_collection_name
        )

        try:
            if self._storage_path:
                expanded_path = os.path.expanduser(self._storage_path)
                pathlib.Path(expanded_path).mkdir(parents=True, exist_ok=True)
                self._client = _chromadb_module.PersistentClient(path=expanded_path)
                logger.info(f"ChromaDB persistent client initialized at: {expanded_path}")
            else:
                self._client = _ChromaClient_class()  # In-memory client
                logger.info("ChromaDB in-memory client initialized.")

            if self._client:
                self._client.list_collections()
                logger.debug("ChromaDB client connection confirmed via list_collections().")
            else:
                raise VectorStorageError(
                    "ChromaDB client object is None after initialization attempt."
                )

        except Exception as e:
            logger.error(
                f"Failed to initialize ChromaDB client (path: {self._storage_path}): {e}",
                exc_info=True,
            )
            self._client = None
            raise VectorStorageError(f"Could not initialize ChromaDB client: {e}")

    def _sync_get_collection(
        self, collection_name: str | None
    ) -> Any:  # Return Any for collection
        """
        Synchronously gets or creates a ChromaDB collection.
        Caches retrieved collection objects.
        """
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")

        target_collection_name = collection_name or self._default_collection_name
        if not target_collection_name:
            raise ConfigError("Vector storage collection name is not specified or configured.")

        if target_collection_name in self._collection_cache:
            return self._collection_cache[target_collection_name]

        try:
            collection = self._client.get_or_create_collection(
                name=target_collection_name,
            )
            logger.debug(f"Accessed ChromaDB collection: '{target_collection_name}'")
            self._collection_cache[target_collection_name] = collection
            return collection
        except Exception as e:
            logger.error(
                f"Failed to get or create ChromaDB collection '{target_collection_name}': {e}",
                exc_info=True,
            )
            raise ConfigError(
                f"Could not access ChromaDB collection '{target_collection_name}': {e}"
            )

    def _sync_add_documents(
        self, documents: list[ContextDocument], collection_name: str | None
    ) -> list[str]:
        """Synchronous logic to add documents."""
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection.name

        doc_ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        contents: list[str] = []

        for doc in documents:
            if not doc.id:
                raise VectorStorageError("Document must have an ID.")
            if not doc.embedding:
                raise VectorStorageError(f"Document '{doc.id}' must have an embedding.")
            doc_ids.append(doc.id)
            embeddings.append(doc.embedding)
            serializable_meta = {}
            if doc.metadata:
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        serializable_meta[k] = v
                    else:
                        try:
                            serializable_meta[k] = str(v)
                            if not isinstance(v, (list, dict)):
                                logger.debug(
                                    f"Metadata value for key '{k}' in doc '{doc.id}' was converted to string: '{str(v)[:50]}...'"
                                )
                        except:
                            logger.warning(
                                f"Metadata value for key '{k}' in doc '{doc.id}' is complex and could not be easily serialized to string. Skipping this metadata field."
                            )
            metadatas.append(serializable_meta)
            contents.append(doc.content)

        if not doc_ids:
            return []

        try:
            collection.upsert(
                ids=doc_ids, embeddings=embeddings, metadatas=metadatas, documents=contents
            )
            logger.info(
                f"Upserted {len(doc_ids)} documents into ChromaDB collection '{target_collection_name}'."
            )
            return doc_ids
        except Exception as e:
            logger.error(
                f"Failed to add documents to ChromaDB collection '{target_collection_name}': {e}",
                exc_info=True,
            )
            raise VectorStorageError(f"ChromaDB add_documents failed: {e}")

    def _sync_similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        collection_name: str | None,
        filter_metadata: dict[str, Any] | None,
    ) -> list[ContextDocument]:
        """Synchronous logic for similarity search."""
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection.name

        try:
            # ChromaDB's query method expects query_embeddings as List[List[float]]
            # The 'where' filter in ChromaDB expects a specific format, e.g., {"source": "wiki"}
            # or more complex ones like {"$and": [...]}. Ensure filter_metadata matches this.
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata,  # Pass filter_metadata directly
                include=["metadatas", "documents", "distances"],
            )
            logger.debug(
                f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results from collection '{target_collection_name}'."
            )

            context_docs: list[ContextDocument] = []
            ids_list = results.get("ids")
            if not ids_list or not ids_list[0]:  # Results are list of lists
                return []

            # Extract results for the single query embedding
            ids = ids_list[0]
            distances_list = results.get("distances")
            distances = (
                distances_list[0]
                if distances_list and distances_list[0] is not None
                else [None] * len(ids)
            )

            metadatas_list = results.get("metadatas")
            metadatas = (
                metadatas_list[0]
                if metadatas_list and metadatas_list[0] is not None
                else [{}] * len(ids)
            )

            contents_list = results.get("documents")
            contents = (
                contents_list[0]
                if contents_list and contents_list[0] is not None
                else [""] * len(ids)
            )

            for i, doc_id_val in enumerate(ids):
                doc_content = contents[i] if i < len(contents) and contents[i] is not None else ""
                doc_metadata = (
                    metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
                )
                doc_distance = (
                    distances[i] if i < len(distances) and distances[i] is not None else None
                )

                context_docs.append(
                    ContextDocument(
                        id=str(doc_id_val),  # Ensure ID is string
                        content=doc_content,
                        metadata=doc_metadata,
                        score=float(doc_distance)
                        if doc_distance is not None
                        else None,  # Chroma uses distance as score
                        embedding=None,  # Embeddings are not typically returned by query
                    )
                )
            return context_docs
        except Exception as e:
            logger.error(
                f"Failed similarity search in ChromaDB collection '{target_collection_name}': {e}",
                exc_info=True,
            )
            raise VectorStorageError(f"ChromaDB similarity_search failed: {e}")

    def _sync_delete_documents(
        self, document_ids: list[str], collection_name: str | None
    ) -> bool:
        """Synchronous logic to delete documents."""
        if not document_ids:
            return True
        collection = self._sync_get_collection(collection_name)
        target_collection_name = collection.name

        try:
            collection.delete(ids=document_ids)
            logger.info(
                f"Attempted deletion of {len(document_ids)} documents from ChromaDB collection '{target_collection_name}'."
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete documents from ChromaDB collection '{target_collection_name}': {e}",
                exc_info=True,
            )
            raise VectorStorageError(f"ChromaDB delete_documents failed: {e}")

    def _sync_list_collection_names(self) -> list[str]:
        """Synchronous logic to list collection names."""
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB list_collections failed: {e}")

    def _sync_get_collection_metadata(
        self, collection_name: str | None
    ) -> dict[str, Any] | None:
        """Synchronous logic to get collection metadata."""
        try:
            collection = self._sync_get_collection(collection_name)  # This handles default name
            if collection.metadata is not None:
                return dict(collection.metadata)
            return None
        except CollectionNotFoundError:
            logger.warning(
                f"Collection '{collection_name or self._default_collection_name}' not found when trying to get metadata."
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to get metadata for ChromaDB collection '{collection_name or self._default_collection_name}': {e}",
                exc_info=True,
            )
            raise VectorStorageError(f"ChromaDB get_collection_metadata failed: {e}")

    def _sync_get_collection_info(self, collection_name: str | None) -> dict[str, Any] | None:
        """
        Synchronous logic to get detailed collection information.

        Returns dict with: name, count, embedding_dimension, metadata, or None if not found.
        """
        try:
            target_name = collection_name or self._default_collection_name
            collection = self._sync_get_collection(target_name)
            # Get document count
            count = collection.count()
            # Get collection metadata
            metadata = dict(collection.metadata) if collection.metadata else {}
            # Try to get embedding dimension from a sample if possible
            embedding_dimension = None
            if count > 0:
                try:
                    # Get a single item to check embedding dimension
                    sample = collection.peek(limit=1)
                    if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                        embedding_dimension = len(sample["embeddings"][0])
                except Exception:
                    pass  # Dimension detection is best-effort
            return {
                "name": target_name,
                "count": count,
                "embedding_dimension": embedding_dimension,
                "metadata": metadata,
            }
        except CollectionNotFoundError:
            logger.debug(
                f"Collection '{collection_name or self._default_collection_name}' not found."
            )
            return None
        except Exception as e:
            logger.error(f"Failed to get info for ChromaDB collection: {e}", exc_info=True)
            raise VectorStorageError(f"ChromaDB get_collection_info failed: {e}")

    def _sync_delete_collection(self, collection_name: str, force: bool = False) -> bool:
        """
        Synchronous logic to delete a collection.

        Args:
            collection_name: Name of collection to delete
            force: If False, raises error if collection has documents

        Returns:
            True if deleted successfully

        Raises:
            VectorStorageError: If collection doesn't exist or has documents (when force=False)
        """
        if not self._client:
            raise VectorStorageError("ChromaDB client is not initialized.")
        try:
            # Check if collection exists
            try:
                collection = self._client.get_collection(name=collection_name)
            except Exception:
                raise VectorStorageError(f"Collection '{collection_name}' not found.")

            # Check document count if force is False
            if not force:
                count = collection.count()
                if count > 0:
                    raise VectorStorageError(
                        f"Collection '{collection_name}' has {count} documents. "
                        "Use force=True to delete anyway."
                    )

            # Delete the collection
            self._client.delete_collection(name=collection_name)

            # Clear from cache if present
            if collection_name in self._collection_cache:
                del self._collection_cache[collection_name]

            logger.info(f"Deleted ChromaDB collection: {collection_name}")
            return True
        except VectorStorageError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to delete ChromaDB collection '{collection_name}': {e}", exc_info=True
            )
            raise VectorStorageError(f"ChromaDB delete_collection failed: {e}")

    def _sync_close(self) -> None:
        """Synchronous closing logic for ChromaDB."""
        if self._client:
            try:
                if hasattr(self._client, "reset"):
                    logger.info("Calling ChromaDB client.reset() to clear in-memory state.")
                    self._client.reset()
                else:
                    logger.info(
                        "ChromaDB client does not have a direct close/reset method. Resources are typically managed by its lifecycle."
                    )
            except AuthorizationError as auth_e:
                logger.warning(
                    f"ChromaDB client reset/cleanup might be restricted by config or permissions: {auth_e}"
                )
            except Exception as e:
                logger.error(
                    f"Error during ChromaDB client resource management: {e}", exc_info=True
                )
            finally:
                self._client = None
                self._collection_cache.clear()
        logger.info("ChromaDB vector storage resources cleared/dereferenced.")

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initializes the ChromaDB client asynchronously by running sync init in a thread."""
        await asyncio.to_thread(self._sync_initialize, config)

    async def add_documents(
        self, documents: list[ContextDocument], collection_name: str | None = None
    ) -> list[str]:
        """Adds documents to ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_add_documents, documents, collection_name)

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        collection_name: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[ContextDocument]:
        """Performs similarity search in ChromaDB asynchronously."""
        return await asyncio.to_thread(
            self._sync_similarity_search, query_embedding, k, collection_name, filter_metadata
        )

    async def delete_documents(
        self, document_ids: list[str], collection_name: str | None = None
    ) -> bool:
        """Deletes documents from ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_delete_documents, document_ids, collection_name)

    async def list_collection_names(self) -> list[str]:
        """Lists collection names from ChromaDB asynchronously."""
        return await asyncio.to_thread(self._sync_list_collection_names)

    async def get_collection_metadata(
        self, collection_name: str | None = None
    ) -> dict[str, Any] | None:
        """
        Retrieves the metadata associated with a specific ChromaDB collection asynchronously.
        """
        return await asyncio.to_thread(self._sync_get_collection_metadata, collection_name)

    # Alias for API compatibility (api.py uses list_collections)
    async def list_collections(self) -> list[str]:
        """Alias for list_collection_names for API compatibility."""
        return await self.list_collection_names()

    async def get_collection_info(
        self, collection_name: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get detailed information about a collection.

        Returns dict with: name, count, embedding_dimension, metadata
        Returns None if collection doesn't exist.
        """
        return await asyncio.to_thread(self._sync_get_collection_info, collection_name)

    async def delete_collection(self, collection_name: str, force: bool = False) -> bool:
        """
        Delete a collection from ChromaDB.

        Args:
            collection_name: Name of collection to delete
            force: If False, raises error if collection has documents

        Returns:
            True if deleted successfully

        Raises:
            VectorStorageError: If deletion fails
        """
        return await asyncio.to_thread(self._sync_delete_collection, collection_name, force)

    async def close(self) -> None:
        """Closes/resets the ChromaDB client asynchronously."""
        await asyncio.to_thread(self._sync_close)
