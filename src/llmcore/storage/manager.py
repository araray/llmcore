# src/llmcore/storage/manager.py
"""
Storage Manager for LLMCore.

Handles the dynamic loading and management of session and vector storage backends
based on the application's configuration. Now includes episodic memory management
through the session storage backends.

REFACTORED FOR MULTI-TENANCY: This manager now acts as a factory that provides
tenant-scoped storage accessors rather than holding global storage instances.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Type, List
from sqlalchemy.ext.asyncio import AsyncSession

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

    REFACTORED: Now acts as a factory for tenant-scoped storage backends
    rather than holding global instances. Reads configuration and provides
    methods to create tenant-aware storage accessors.
    """
    _config: ConfyConfig
    _session_storage_type: Optional[str] = None
    _vector_storage_type: Optional[str] = None
    _session_storage_config: Dict[str, Any] = {}
    _vector_storage_config: Dict[str, Any] = {}

    def __init__(self, config: ConfyConfig):
        """
        Initializes the StorageManager.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
        """
        self._config = config
        logger.info("StorageManager initialized as factory.")
        # Configuration parsing is deferred to initialize_storages()

    async def initialize_storages(self) -> None:
        """
        Parses storage configuration but does not create global instances.

        In the new multi-tenant architecture, storage instances are created
        on-demand with tenant-specific configurations.
        """
        await self._parse_session_storage_config()
        await self._parse_vector_storage_config()
        logger.info("Storage configuration parsing complete.")

    async def _parse_session_storage_config(self) -> None:
        """Parses session storage configuration."""
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
        """Parses vector storage configuration."""
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

    def get_session_storage(self, db_session: Optional[AsyncSession] = None) -> BaseSessionStorage:
        """
        Returns a session storage instance, optionally configured for tenant-specific use.

        REFACTORED: Now accepts an optional tenant-scoped database session.
        For PostgreSQL backends, this session will be pre-configured with the
        correct schema search path.

        Args:
            db_session: Optional tenant-scoped database session for PostgreSQL backends

        Returns:
            BaseSessionStorage: Storage instance ready for use

        Raises:
            StorageError: If session storage is not configured or initialization fails.
        """
        if self._session_storage_type is None:
            if not self._config.get("storage.session.type"):
                raise StorageError("Session storage is not configured ('storage.session.type' missing).")
            else:
                raise StorageError("Session storage failed to parse configuration (check logs for details).")

        session_storage_cls = SESSION_STORAGE_MAP[self._session_storage_type]

        try:
            storage_instance = session_storage_cls()

            # For PostgreSQL, inject the tenant-scoped session if provided
            if self._session_storage_type == "postgres" and db_session is not None:
                # PostgresSessionStorage will be modified to accept pre-configured sessions
                storage_instance._tenant_session = db_session
                logger.debug("Injected tenant-scoped session into PostgreSQL storage backend")

            # Initialize with configuration
            asyncio.create_task(storage_instance.initialize(self._session_storage_config))

            return storage_instance

        except Exception as e:
            logger.error(f"Failed to create session storage instance '{self._session_storage_type}': {e}", exc_info=True)
            raise StorageError(f"Session storage creation failed: {e}")

    def get_vector_storage(self, db_session: Optional[AsyncSession] = None) -> BaseVectorStorage:
        """
        Returns a vector storage instance, optionally configured for tenant-specific use.

        REFACTORED: Now accepts an optional tenant-scoped database session.
        For PgVector backends, this session will be pre-configured with the
        correct schema search path.

        Args:
            db_session: Optional tenant-scoped database session for PgVector backends

        Returns:
            BaseVectorStorage: Storage instance ready for use

        Raises:
            StorageError: If vector storage is not configured or initialization fails.
        """
        if self._vector_storage_type is None:
            if not self._config.get("storage.vector.type"):
                raise StorageError("Vector storage is not configured ('storage.vector.type' missing). RAG is unavailable.")
            else:
                raise StorageError("Vector storage failed to parse configuration (check logs for details). RAG is unavailable.")

        vector_storage_cls = VECTOR_STORAGE_MAP[self._vector_storage_type]

        try:
            storage_instance = vector_storage_cls()

            # For PgVector, inject the tenant-scoped session if provided
            if self._vector_storage_type == "pgvector" and db_session is not None:
                # PgVectorStorage will be modified to accept pre-configured sessions
                storage_instance._tenant_session = db_session
                logger.debug("Injected tenant-scoped session into PgVector storage backend")

            # Initialize with configuration
            asyncio.create_task(storage_instance.initialize(self._vector_storage_config))

            return storage_instance

        except Exception as e:
            logger.error(f"Failed to create vector storage instance '{self._vector_storage_type}': {e}", exc_info=True)
            raise StorageError(f"Vector storage creation failed: {e}")

    # --- Legacy methods for backward compatibility during transition ---

    async def add_episode(self, episode: Episode, db_session: Optional[AsyncSession] = None) -> None:
        """
        Adds a new episode to the episodic memory log through the session storage backend.

        REFACTORED: Now accepts optional tenant-scoped database session.

        Args:
            episode: The Episode object to add.
            db_session: Optional tenant-scoped database session

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        session_storage = self.get_session_storage(db_session)
        await session_storage.add_episode(episode)
        logger.debug(f"Episode '{episode.episode_id}' added to session '{episode.session_id}' via StorageManager.")

    async def get_episodes(self, session_id: str, limit: int = 100, offset: int = 0, db_session: Optional[AsyncSession] = None) -> List[Episode]:
        """
        Retrieves episodes for a given session through the session storage backend.

        REFACTORED: Now accepts optional tenant-scoped database session.

        Args:
            session_id: The ID of the session to retrieve episodes for.
            limit: The maximum number of episodes to return.
            offset: The number of episodes to skip (for pagination).
            db_session: Optional tenant-scoped database session

        Returns:
            A list of Episode objects.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        session_storage = self.get_session_storage(db_session)
        episodes = await session_storage.get_episodes(session_id, limit, offset)
        logger.debug(f"Retrieved {len(episodes)} episodes for session '{session_id}' via StorageManager.")
        return episodes

    async def get_episode_count(self, session_id: str, db_session: Optional[AsyncSession] = None) -> int:
        """
        Gets the total count of episodes for a session.

        REFACTORED: Now accepts optional tenant-scoped database session.

        Args:
            session_id: The ID of the session to count episodes for.
            db_session: Optional tenant-scoped database session

        Returns:
            The total number of episodes for the session.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        session_storage = self.get_session_storage(db_session)

        # Get episodes in batches to count total without loading all into memory
        total_count = 0
        batch_size = 1000
        offset = 0

        while True:
            batch = await session_storage.get_episodes(session_id, limit=batch_size, offset=offset)
            batch_count = len(batch)
            total_count += batch_count

            if batch_count < batch_size:
                # We've reached the end
                break
            offset += batch_size

        logger.debug(f"Total episode count for session '{session_id}': {total_count}")
        return total_count

    async def list_vector_collection_names(self, db_session: Optional[AsyncSession] = None) -> List[str]:
        """
        Lists the names of all available collections in the configured vector store.

        REFACTORED: Now accepts optional tenant-scoped database session.

        Args:
            db_session: Optional tenant-scoped database session

        Returns:
            A list of collection name strings.

        Raises:
            StorageError: If vector storage is not configured, failed to initialize,
                          or if the backend fails to list collections.
        """
        vector_storage = self.get_vector_storage(db_session)
        try:
            return await vector_storage.list_collection_names()
        except NotImplementedError:
            logger.error(f"Vector storage backend {type(vector_storage).__name__} does not implement list_collection_names.")
            raise StorageError(f"Listing collections not supported by {type(vector_storage).__name__}.")
        except Exception as e:
            logger.error(f"Error listing vector collections via {type(vector_storage).__name__}: {e}", exc_info=True)
            raise StorageError(f"Failed to list vector collections: {e}")

    async def close_storages(self) -> None:
        """
        Closes connections for all initialized storage backends.

        REFACTORED: In the new architecture, storage instances are created
        on-demand, so this method primarily serves as cleanup for any
        cached connections or resources.
        """
        logger.info("Storage manager cleanup complete (factory pattern - no persistent connections).")

    async def close(self) -> None:
        """
        Alias for close_storages() to maintain API compatibility.

        LLMCore.close() calls StorageManager.close(), so this method
        delegates to close_storages() which does the actual cleanup.
        """
        await self.close_storages()
