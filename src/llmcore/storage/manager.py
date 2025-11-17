# src/llmcore/storage/manager.py
"""
Storage Manager for LLMCore.

Handles the initialization and management of session and vector storage backends
based on the application's configuration. Now includes episodic memory management
through the session storage backends.

REFACTORED FOR LIBRARY MODE (Step 1.3): This manager now directly instantiates
and holds storage backend instances rather than acting as a factory. All multi-tenant
db_session parameters have been removed for single-tenant library usage.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Type, List

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore

from ..exceptions import ConfigError, StorageError
from ..models import Episode
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage
# Import concrete vector storage implementations
from .chromadb_vector import ChromaVectorStorage
# Import concrete session storage implementations
from .json_session import JsonSessionStorage
from .pgvector_storage import PgVectorStorage
from .postgres_session_storage import PostgresSessionStorage
from .sqlite_session import SqliteSessionStorage

logger = logging.getLogger(__name__)

# --- Mappings from config type string to class ---
SESSION_STORAGE_MAP: Dict[str, Type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
    "postgres": PostgresSessionStorage,
}

VECTOR_STORAGE_MAP: Dict[str, Type[BaseVectorStorage]] = {
    "chromadb": ChromaVectorStorage,
    "pgvector": PgVectorStorage,
}
# --- End Mappings ---


class StorageManager:
    """
    Manages the initialization and access to storage backends.

    REFACTORED (Step 1.3): Now operates in library mode. Storage backend instances
    are created once during initialization and held internally, rather than being
    created on-demand per-request. This provides a simple, single-tenant API suitable
    for library usage.
    """
    _config: ConfyConfig
    _session_storage_type: Optional[str] = None
    _vector_storage_type: Optional[str] = None
    _session_storage_config: Dict[str, Any] = {}
    _vector_storage_config: Dict[str, Any] = {}

    # REFACTORED: Storage instances are now held directly
    _session_storage_instance: Optional[BaseSessionStorage] = None
    _vector_storage_instance: Optional[BaseVectorStorage] = None

    def __init__(self, config: ConfyConfig):
        """
        Initializes the StorageManager.

        REFACTORED (Step 1.3): Constructor now only stores configuration.
        Actual storage backend instantiation is deferred to initialize_storages().

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
        """
        self._config = config
        logger.info("StorageManager initialized for library mode (single-tenant).")

    async def initialize_storages(self) -> None:
        """
        Parses storage configuration and creates storage backend instances.

        REFACTORED (Step 1.3): Now actually instantiates the storage backends
        based on configuration, rather than just parsing config. The backends
        are held internally and accessed via properties.

        Raises:
            ConfigError: If storage configuration is invalid.
            StorageError: If storage backend instantiation fails.
        """
        # Parse configuration
        await self._parse_session_storage_config()
        await self._parse_vector_storage_config()

        # Instantiate storage backends
        await self._instantiate_session_storage()
        await self._instantiate_vector_storage()

        logger.info("Storage backends initialized successfully.")

    async def _parse_session_storage_config(self) -> None:
        """
        Parses session storage configuration from the config object.

        Sets _session_storage_type and _session_storage_config based on
        the [storage.session] section of the configuration.
        """
        session_storage_config = self._config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")

        if not session_storage_type:
            logger.warning("No session storage type configured ('storage.session.type'). Session persistence disabled.")
            self._session_storage_type = None
            return

        if session_storage_type.lower() not in SESSION_STORAGE_MAP:
            raise ConfigError(f"Unsupported session storage type configured: '{session_storage_type}'. "
                              f"Available types: {list(SESSION_STORAGE_MAP.keys())}")

        self._session_storage_type = session_storage_type.lower()
        self._session_storage_config = session_storage_config
        logger.info(f"Session storage type '{session_storage_type}' configured.")

    async def _parse_vector_storage_config(self) -> None:
        """
        Parses vector storage configuration from the config object.

        Sets _vector_storage_type and _vector_storage_config based on
        the [storage.vector] section of the configuration.
        """
        vector_storage_config = self._config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")

        if not vector_storage_type:
            logger.warning("No vector storage type configured ('storage.vector.type'). RAG functionality will be unavailable.")
            self._vector_storage_type = None
            return

        if vector_storage_type.lower() not in VECTOR_STORAGE_MAP:
            raise ConfigError(f"Unsupported vector storage type configured: '{vector_storage_type}'. "
                              f"Available types: {list(VECTOR_STORAGE_MAP.keys())}.")

        self._vector_storage_type = vector_storage_type.lower()
        self._vector_storage_config = vector_storage_config
        logger.info(f"Vector storage type '{vector_storage_type}' configured.")

    async def _instantiate_session_storage(self) -> None:
        """
        Creates and initializes the session storage backend instance.

        REFACTORED (Step 1.3): New method that instantiates the storage backend
        once during initialization, rather than on-demand per-request.

        Raises:
            StorageError: If instantiation or initialization fails.
        """
        if self._session_storage_type is None:
            logger.info("Session storage not configured; skipping instantiation.")
            self._session_storage_instance = None
            return

        session_storage_cls = SESSION_STORAGE_MAP[self._session_storage_type]

        try:
            self._session_storage_instance = session_storage_cls()
            await self._session_storage_instance.initialize(self._session_storage_config)
            logger.info(f"Session storage backend '{self._session_storage_type}' instantiated and initialized.")
        except Exception as e:
            logger.error(f"Failed to instantiate session storage backend '{self._session_storage_type}': {e}", exc_info=True)
            raise StorageError(f"Session storage instantiation failed: {e}")

    async def _instantiate_vector_storage(self) -> None:
        """
        Creates and initializes the vector storage backend instance.

        REFACTORED (Step 1.3): New method that instantiates the storage backend
        once during initialization, rather than on-demand per-request.

        Raises:
            StorageError: If instantiation or initialization fails.
        """
        if self._vector_storage_type is None:
            logger.info("Vector storage not configured; skipping instantiation.")
            self._vector_storage_instance = None
            return

        vector_storage_cls = VECTOR_STORAGE_MAP[self._vector_storage_type]

        try:
            self._vector_storage_instance = vector_storage_cls()
            await self._vector_storage_instance.initialize(self._vector_storage_config)
            logger.info(f"Vector storage backend '{self._vector_storage_type}' instantiated and initialized.")
        except Exception as e:
            logger.error(f"Failed to instantiate vector storage backend '{self._vector_storage_type}': {e}", exc_info=True)
            raise StorageError(f"Vector storage instantiation failed: {e}")

    @property
    def session_storage(self) -> BaseSessionStorage:
        """
        Returns the initialized session storage backend instance.

        REFACTORED (Step 1.3): Changed from get_session_storage(db_session) method
        to a simple property that returns the internally-held instance.

        Returns:
            BaseSessionStorage: The initialized storage backend instance.

        Raises:
            StorageError: If session storage is not configured or not initialized.
        """
        if self._session_storage_instance is None:
            if self._session_storage_type is None:
                raise StorageError("Session storage is not configured ('storage.session.type' missing).")
            else:
                raise StorageError("Session storage not initialized. Call initialize_storages() first.")

        return self._session_storage_instance

    @property
    def vector_storage(self) -> BaseVectorStorage:
        """
        Returns the initialized vector storage backend instance.

        REFACTORED (Step 1.3): Changed from get_vector_storage(db_session) method
        to a simple property that returns the internally-held instance.

        Returns:
            BaseVectorStorage: The initialized storage backend instance.

        Raises:
            StorageError: If vector storage is not configured or not initialized.
        """
        if self._vector_storage_instance is None:
            if self._vector_storage_type is None:
                raise StorageError("Vector storage is not configured ('storage.vector.type' missing). RAG is unavailable.")
            else:
                raise StorageError("Vector storage not initialized. Call initialize_storages() first.")

        return self._vector_storage_instance

    # --- Convenience methods for episodic memory (delegate to session storage) ---

    async def add_episode(self, episode: Episode) -> None:
        """
        Adds a new episode to the episodic memory log through the session storage backend.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            episode: The Episode object to add.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        await self.session_storage.add_episode(episode)
        logger.debug(f"Episode '{episode.episode_id}' added to session '{episode.session_id}' via StorageManager.")

    async def get_episodes(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Episode]:
        """
        Retrieves episodes for a given session through the session storage backend.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            session_id: The ID of the session to retrieve episodes for.
            limit: The maximum number of episodes to return.
            offset: The number of episodes to skip (for pagination).

        Returns:
            A list of Episode objects.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        episodes = await self.session_storage.get_episodes(session_id, limit, offset)
        logger.debug(f"Retrieved {len(episodes)} episodes for session '{session_id}' via StorageManager.")
        return episodes

    async def get_episode_count(self, session_id: str) -> int:
        """
        Gets the total count of episodes for a session.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            session_id: The ID of the session to count episodes for.

        Returns:
            The total number of episodes for the session.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        # Get episodes in batches to count total without loading all into memory
        total_count = 0
        batch_size = 1000
        offset = 0

        while True:
            batch = await self.session_storage.get_episodes(session_id, limit=batch_size, offset=offset)
            batch_count = len(batch)
            total_count += batch_count

            if batch_count < batch_size:
                # We've reached the end
                break
            offset += batch_size

        logger.debug(f"Total episode count for session '{session_id}': {total_count}")
        return total_count

    async def list_vector_collection_names(self) -> List[str]:
        """
        Lists the names of all available collections in the configured vector store.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the vector_storage property.

        Returns:
            A list of collection name strings.

        Raises:
            StorageError: If vector storage is not configured, failed to initialize,
                          or if the backend fails to list collections.
        """
        try:
            return await self.vector_storage.list_collection_names()
        except NotImplementedError:
            logger.error(f"Vector storage backend {type(self.vector_storage).__name__} does not implement list_collection_names.")
            raise StorageError(f"Listing collections not supported by {type(self.vector_storage).__name__}.")
        except Exception as e:
            logger.error(f"Error listing vector collections via {type(self.vector_storage).__name__}: {e}", exc_info=True)
            raise StorageError(f"Failed to list vector collections: {e}")

    async def close_storages(self) -> None:
        """
        Closes connections for all initialized storage backends.

        REFACTORED (Step 1.3): Now closes the internally-held storage instances.
        """
        if self._session_storage_instance:
            try:
                await self._session_storage_instance.close()
                logger.info("Session storage backend closed.")
            except Exception as e:
                logger.warning(f"Error closing session storage: {e}")

        if self._vector_storage_instance:
            try:
                await self._vector_storage_instance.close()
                logger.info("Vector storage backend closed.")
            except Exception as e:
                logger.warning(f"Error closing vector storage: {e}")

        logger.info("Storage manager cleanup complete.")

    async def close(self) -> None:
        """
        Alias for close_storages() to maintain API compatibility.

        LLMCore.close() calls StorageManager.close(), so this method
        delegates to close_storages() which does the actual cleanup.
        """
        await self.close_storages()
