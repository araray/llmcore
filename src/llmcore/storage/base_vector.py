# src/llmcore/storage/base_vector.py
"""
Abstract Base Class for Vector Storage backends.

This module defines the interface that all vector storage implementations
(e.g., ChromaDB, PostgreSQL+pgvector) must adhere to within the LLMCore library,
enabling Retrieval Augmented Generation (RAG).
"""

import abc
from typing import Any, Dict, List, Optional

# Import ContextDocument for type hinting
# Use forward reference if needed, but direct should be fine
from ..models import ContextDocument


class BaseVectorStorage(abc.ABC):
    """
    Abstract Base Class for vector embedding storage.

    Defines the standard methods required for managing the persistence
    and retrieval of documents and their vector embeddings, primarily for RAG.
    Implementations handle specifics for different vector databases.
    """

    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the vector storage backend with given configuration.

        Should be called asynchronously to set up connections, clients,
        or required resources based on the provided config.

        Args:
            config: Backend-specific configuration dictionary derived from
                    the main LLMCore configuration (e.g., path, db_url,
                    default_collection).
        """
        pass

    @abc.abstractmethod
    async def add_documents(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Add or update multiple documents in the specified collection.

        Each document should have content and an embedding (unless the backend
        generates embeddings internally). If a document ID already exists,
        it should typically be updated (upsert behavior).

        Args:
            documents: A list of ContextDocument objects to add/update.
                       Each document should ideally have an 'id', 'content',
                       'embedding', and optional 'metadata'.
            collection_name: The name of the collection to add documents to.
                             If None, the implementation might use a default
                             collection defined in its configuration.

        Returns:
            A list of IDs of the added/updated documents.

        Raises:
            VectorStorageError: If adding documents fails.
            ConfigError: If the specified collection is invalid or cannot be accessed.
        """
        pass

    @abc.abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        """
        Perform a similarity search for documents based on a query embedding.

        Retrieves the top 'k' most similar documents to the given embedding
        within the specified collection, optionally applying metadata filters.

        Args:
            query_embedding: The vector embedding of the query text.
            k: The number of top similar documents to retrieve.
            collection_name: The name of the collection to search within.
                             If None, uses the default collection.
            filter_metadata: Optional dictionary to filter results based on
                             document metadata (e.g., {"source": "wiki"}).
                             Support depends on the backend implementation.

        Returns:
            A list of ContextDocument objects representing the search results,
            ordered by similarity (most similar first). Results should include
            content, metadata, and ideally the similarity score. Embeddings
            in the returned documents are optional.

        Raises:
            VectorStorageError: If the search operation fails.
            ConfigError: If the specified collection is invalid or cannot be accessed.
        """
        pass

    @abc.abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete documents from the specified collection by their IDs.

        Args:
            document_ids: A list of unique document IDs to delete.
            collection_name: The name of the collection to delete from.
                             If None, uses the default collection.

        Returns:
            True if deletion was attempted (regardless of whether all IDs were found),
            False if a fundamental error occurred during the operation. Exact
            success reporting might depend on the backend.

        Raises:
            VectorStorageError: If the deletion operation fails fundamentally.
            ConfigError: If the specified collection is invalid or cannot be accessed.
        """
        pass

    @abc.abstractmethod
    async def list_collection_names(self) -> List[str]:
        """
        List the names of all available collections in the vector store.

        Returns:
            A list of collection name strings.

        Raises:
            VectorStorageError: If listing collections fails.
        """
        pass

    @abc.abstractmethod
    async def get_collection_metadata(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the metadata associated with a specific collection.

        This metadata is expected to include details about the embedding model
        used for the collection, such as 'embedding_model_provider',
        'embedding_model_name', and 'embedding_dimension', if stored by the
        ingestion process (e.g., by Apykatu).

        Args:
            collection_name: The name of the collection whose metadata is to be retrieved.
                             If None, the implementation might use a default collection
                             defined in its configuration.

        Returns:
            A dictionary containing the collection's metadata if the collection
            exists and metadata is available, otherwise None.

        Raises:
            VectorStorageError: If retrieving collection metadata fails.
            ConfigError: If the specified collection is invalid or cannot be accessed.
        """
        pass

    # --- Optional Collection Management Methods ---
    # These might be useful but are not strictly required by the core RAG flow

    # @abc.abstractmethod
    # async def create_collection(self, collection_name: str, metadata: Optional[Dict] = None) -> None:
    #     """Explicitly create a new collection."""
    #     pass

    # @abc.abstractmethod
    # async def delete_collection(self, collection_name: str) -> None:
    #     """Delete an entire collection."""
    #     pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Clean up resources used by the vector storage backend.

        Should be called asynchronously to close connections, clients, etc.
        """
        pass
