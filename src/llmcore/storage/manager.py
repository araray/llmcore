# src/llmcore/storage/manager.py
"""
Storage Manager for LLMCore.

Handles the dynamic loading and management of session and vector storage backends
based on the application's configuration.
"""

import logging
from typing import Dict, Any, Type, Optional

from confy.loader import Config as ConfyConfig

from ..exceptions import ConfigError, StorageError
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage # Import BaseVectorStorage

# Import concrete implementations (add more as they are created)
from .json_session import JsonSessionStorage
from .sqlite_session import SqliteSessionStorage
# from .postgres_storage import PostgresSessionStorage, PgVectorStorage # Example for Phase 3
# from .chromadb_vector import ChromaVectorStorage # Example for Phase 2

logger = logging.getLogger(__name__)

# --- Mappings from config type string to class ---
SESSION_STORAGE_MAP: Dict[str, Type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
    # "postgres": PostgresSessionStorage, # Add when implemented
}

VECTOR_STORAGE_MAP: Dict[str, Type[BaseVectorStorage]] = {
    # "chromadb": ChromaVectorStorage, # Add when implemented
    # "pgvector": PgVectorStorage, # Add when implemented
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

    async def initialize_storages(self) -> None:
        """
        Initializes both session and vector storage backends based on configuration.
        Should be called asynchronously after StorageManager instantiation.
        """
        await self._initialize_session_storage()
        await self._initialize_vector_storage()
        logger.info("Storage backends initialization complete.")

    async def _initialize_session_storage(self) -> None:
        """Initializes the session storage backend."""
        session_storage_config = self._config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")

        if not session_storage_type:
            logger.warning("No session storage type configured. Session persistence disabled.")
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
        except Exception as e:
            logger.error(f"Failed to initialize session storage backend '{session_storage_type}': {e}", exc_info=True)
            self._session_storage = None # Ensure it's None on failure
            raise StorageError(f"Session storage initialization failed: {e}")

    async def _initialize_vector_storage(self) -> None:
        """Initializes the vector storage backend."""
        vector_storage_config = self._config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")

        if not vector_storage_type:
            logger.warning("No vector storage type configured. RAG functionality may be limited.")
            self._vector_storage = None
            return

        vector_storage_cls = VECTOR_STORAGE_MAP.get(vector_storage_type.lower())
        if not vector_storage_cls:
            # If RAG is intended, this should probably be a ConfigError.
            # For now, warn, as RAG might not be used.
            logger.warning(f"Unsupported vector storage type configured: '{vector_storage_type}'. "
                           f"Available types: {list(VECTOR_STORAGE_MAP.keys())}. RAG might not work.")
            self._vector_storage = None
            # Consider raising ConfigError here if vector storage is deemed essential
            # raise ConfigError(f"Unsupported vector storage type: '{vector_storage_type}'.")
            return

        try:
            self._vector_storage = vector_storage_cls()
            # Assuming vector storage also has an async initialize method
            # Adjust if the specific vector store library requires different initialization
            if hasattr(self._vector_storage, 'initialize') and callable(self._vector_storage.initialize):
                 await self._vector_storage.initialize(vector_storage_config) # type: ignore
            else:
                 logger.warning(f"Vector storage backend '{vector_storage_type}' does not have an async 'initialize' method.")
                 # Handle synchronous initialization if needed, potentially in a thread

            logger.info(f"Vector storage backend '{vector_storage_type}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize vector storage backend '{vector_storage_type}': {e}", exc_info=True)
            self._vector_storage = None # Ensure it's None on failure
            raise StorageError(f"Vector storage initialization failed: {e}")

    def get_session_storage(self) -> BaseSessionStorage:
        """
        Returns the initialized session storage instance.

        Raises:
            StorageError: If session storage is not configured or failed to initialize.
        """
        if self._session_storage is None:
            raise StorageError("Session storage is not configured or failed to initialize.")
        return self._session_storage

    def get_vector_storage(self) -> BaseVectorStorage:
        """
        Returns the initialized vector storage instance.

        Raises:
            StorageError: If vector storage is not configured or failed to initialize.
        """
        if self._vector_storage is None:
            raise StorageError("Vector storage is not configured or failed to initialize.")
        return self._vector_storage

    async def close_storages(self) -> None:
        """Closes connections for all initialized storage backends."""
        logger.info("Closing storage connections...")
        if self._session_storage:
            try:
                await self._session_storage.close()
                logger.info("Session storage closed.")
            except Exception as e:
                logger.error(f"Error closing session storage: {e}", exc_info=True)
        if self._vector_storage:
            try:
                # Assuming vector storage also has an async close method
                if hasattr(self._vector_storage, 'close') and callable(self._vector_storage.close):
                    await self._vector_storage.close() # type: ignore
                    logger.info("Vector storage closed.")
                else:
                     logger.warning("Vector storage backend does not have an async 'close' method.")
                     # Handle synchronous closing if needed
            except Exception as e:
                logger.error(f"Error closing vector storage: {e}", exc_info=True)
        logger.info("Storage connections closure attempt complete.")
