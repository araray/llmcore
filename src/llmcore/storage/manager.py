# src/llmcore/storage/manager.py
"""
Storage Manager for LLMCore.

Handles the dynamic loading and management of session and vector storage backends
based on the application's configuration.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Type

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore


from ..exceptions import ConfigError, StorageError
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage
# Import concrete vector storage implementations
from .chromadb_vector import ChromaVectorStorage
# Import concrete session storage implementations
from .json_session import JsonSessionStorage
from .postgres_storage import PgVectorStorage  # Added for Phase 3
from .postgres_storage import PostgresSessionStorage  # Added for Phase 3
from .sqlite_session import SqliteSessionStorage

logger = logging.getLogger(__name__)

# --- Mappings from config type string to class ---
SESSION_STORAGE_MAP: Dict[str, Type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
    "postgres": PostgresSessionStorage, # Added for Phase 3
}

VECTOR_STORAGE_MAP: Dict[str, Type[BaseVectorStorage]] = {
    "chromadb": ChromaVectorStorage,
    "pgvector": PgVectorStorage, # Added for Phase 3
}
# --- End Mappings ---


class StorageManager:
    """
    Manages the initialization and access to storage backends.

    Reads configuration and instantiates the appropriate session and
    vector storage classes.
    """
    _session_storage: Optional[BaseSessionStorage] = None
    _vector_storage: Optional[BaseVectorStorage] = None
    _config: ConfyConfig

    def __init__(self, config: ConfyConfig):
        """
        Initializes the StorageManager.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
        """
        self._config = config
        logger.info("StorageManager initialized.")
        # Initialization of actual storage backends is deferred to initialize_storages()

    async def initialize_storages(self) -> None:
        """
        Initializes both session and vector storage backends based on configuration.
        Should be called asynchronously after StorageManager instantiation.
        """
        await self._initialize_session_storage()
        await self._initialize_vector_storage()
        logger.info("Storage backends initialization attempt complete.")

    async def _initialize_session_storage(self) -> None:
        """Initializes the session storage backend."""
        session_storage_config = self._config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")

        if not session_storage_type:
            logger.warning("No session storage type configured ('storage.session.type'). Session persistence disabled.")
            self._session_storage = None
            return

        session_storage_cls = SESSION_STORAGE_MAP.get(session_storage_type.lower())
        if not session_storage_cls:
            raise ConfigError(f"Unsupported session storage type configured: '{session_storage_type}'. "
                              f"Available types: {list(SESSION_STORAGE_MAP.keys())}")

        try:
            self._session_storage = session_storage_cls()
            await self._session_storage.initialize(session_storage_config)
            logger.info(f"Session storage backend '{session_storage_type}' initialized successfully.")
        except ImportError as e:
             logger.error(f"Failed to initialize session storage '{session_storage_type}': Missing required library. "
                          f"Install dependencies for '{session_storage_type}' (e.g., 'pip install llmcore[{session_storage_type}]'). Error: {e}")
             self._session_storage = None
             raise StorageError(f"Session storage initialization failed due to missing dependency: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize session storage backend '{session_storage_type}': {e}", exc_info=True)
            self._session_storage = None
            raise StorageError(f"Session storage initialization failed: {e}")

    async def _initialize_vector_storage(self) -> None:
        """Initializes the vector storage backend."""
        vector_storage_config = self._config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")

        if not vector_storage_type:
            logger.warning("No vector storage type configured ('storage.vector.type'). RAG functionality will be unavailable.")
            self._vector_storage = None
            return

        vector_storage_cls = VECTOR_STORAGE_MAP.get(vector_storage_type.lower())
        if not vector_storage_cls:
            raise ConfigError(f"Unsupported vector storage type configured: '{vector_storage_type}'. "
                              f"Available types: {list(VECTOR_STORAGE_MAP.keys())}.")

        try:
            self._vector_storage = vector_storage_cls()
            await self._vector_storage.initialize(vector_storage_config)
            logger.info(f"Vector storage backend '{vector_storage_type}' initialized successfully.")
        except ImportError as e:
             logger.error(f"Failed to initialize vector storage '{vector_storage_type}': Missing required library. "
                          f"Install dependencies for '{vector_storage_type}' (e.g., 'pip install llmcore[{vector_storage_type}]'). Error: {e}")
             self._vector_storage = None
             raise StorageError(f"Vector storage initialization failed due to missing dependency: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize vector storage backend '{vector_storage_type}': {e}", exc_info=True)
            self._vector_storage = None
            raise StorageError(f"Vector storage initialization failed: {e}")

    def get_session_storage(self) -> BaseSessionStorage:
        """
        Returns the initialized session storage instance.

        Raises:
            StorageError: If session storage is not configured or failed to initialize.
        """
        if self._session_storage is None:
            if not self._config.get("storage.session.type"):
                 raise StorageError("Session storage is not configured ('storage.session.type' missing).")
            else:
                 raise StorageError("Session storage failed to initialize (check logs for details).")
        return self._session_storage

    def get_vector_storage(self) -> BaseVectorStorage:
        """
        Returns the initialized vector storage instance.

        Raises:
            StorageError: If vector storage is not configured or failed to initialize.
        """
        if self._vector_storage is None:
            if not self._config.get("storage.vector.type"):
                 raise StorageError("Vector storage is not configured ('storage.vector.type' missing). RAG is unavailable.")
            else:
                 raise StorageError("Vector storage failed to initialize (check logs for details). RAG is unavailable.")
        return self._vector_storage

    async def close_storages(self) -> None:
        """Closes connections for all initialized storage backends."""
        logger.info("Closing storage connections...")
        close_tasks = []
        if self._session_storage:
            if hasattr(self._session_storage, 'close') and asyncio.iscoroutinefunction(self._session_storage.close):
                close_tasks.append(self._close_single_storage("Session", self._session_storage))
            else:
                logger.debug(f"Session storage backend {type(self._session_storage).__name__} does not have an async 'close' method or it's not callable as such.")

        if self._vector_storage:
            if hasattr(self._vector_storage, 'close') and asyncio.iscoroutinefunction(self._vector_storage.close):
                close_tasks.append(self._close_single_storage("Vector", self._vector_storage))
            else:
                 logger.debug(f"Vector storage backend {type(self._vector_storage).__name__} does not have an async 'close' method or it's not callable as such.")

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error during storage closure: {result}", exc_info=result)

        logger.info("Storage connections closure attempt complete.")

    async def _close_single_storage(self, storage_type: str, storage_instance: Any):
        """Helper coroutine to close a single storage instance and log errors."""
        try:
            await storage_instance.close()
            logger.info(f"{storage_type} storage of type {type(storage_instance).__name__} closed.")
        except Exception as e:
            logger.error(f"Error closing {storage_type} storage (type {type(storage_instance).__name__}): {e}", exc_info=True)
            raise
