# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.

This module defines the `LLMCore` class, which is the primary entry point
for applications to interact with various LLM functionalities, including
chat completions, session management, and Retrieval Augmented Generation (RAG).
"""

import asyncio
import logging
import importlib.resources # For loading package data
import pathlib # For path operations with importlib.resources
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Type

# Import models and exceptions for type hinting and user access
from .models import ChatSession, ContextDocument, Message, Role
from .exceptions import (
    LLMCoreError, ProviderError, SessionNotFoundError, ConfigError,
    StorageError, SessionStorageError, VectorStorageError, # Added StorageError base
    EmbeddingError, ContextLengthError, MCPError
)
# Import storage base and implementations
from .storage.base_session import BaseSessionStorage
from .storage.json_session import JsonSessionStorage
from .storage.sqlite_session import SqliteSessionStorage
# Placeholder for vector storage base (will be added later)
# from .storage.base_vector import BaseVectorStorage

# Import confy.Config for type hinting the config object
try:
    from confy.loader import Config as ConfyConfig
    import tomli # For parsing the default TOML config
except ImportError as e:
    ConfyConfig = Dict[str, Any] # type: ignore
    logging.getLogger(__name__).warning(
        f"Could not import confy or tomli at module level: {e}. "
        "LLMCore initialization will fail if they are not installed."
    )

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Mapping for storage types to classes
SESSION_STORAGE_MAP: Dict[str, Type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
    # "postgres": PostgresSessionStorage, # To be added later
}

# Placeholder for vector storage map
# VECTOR_STORAGE_MAP: Dict[str, Type[BaseVectorStorage]] = {
#     "chromadb": ChromaVectorStorage,
#     "pgvector": PgVectorStorage,
# }


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, and
    Retrieval Augmented Generation (RAG) using configurable providers,
    storage backends, and embedding models.

    Note: The __init__ method is async due to storage initialization.
          Instantiate using 'instance = await LLMCore(...)'.
    """

    config: ConfyConfig # Publicly accessible resolved config
    _session_storage: BaseSessionStorage # Internal session storage instance
    # _vector_storage: BaseVectorStorage # Internal vector storage instance (placeholder)

    # Make __init__ async because storage initialization is async
    def __init__(self):
        """Private constructor. Use `create` classmethod for async initialization."""
        # This constructor should ideally be minimal or private.
        # The actual async initialization happens in `create`.
        pass

    @classmethod
    async def create(
        cls,
        config_overrides: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[str] = None,
        env_prefix: Optional[str] = "LLMCORE"
    ) -> "LLMCore":
        """
        Asynchronously creates and initializes an LLMCore instance.

        Loads configuration, initializes storage backends, and prepares managers.

        Args:
            config_overrides: Dictionary of configuration overrides.
            config_file_path: Path to a custom configuration file.
            env_prefix: Prefix for environment variable overrides.

        Returns:
            An initialized LLMCore instance.

        Raises:
            ConfigError: If configuration loading or validation fails.
            StorageError: If storage backend initialization fails.
            ImportError: If required dependencies are missing.
        """
        instance = cls() # Create instance using the minimal __init__
        logger.info("Initializing LLMCore asynchronously...")

        # --- Step 1: Initialize self.config using confy.Config(...) ---
        try:
            from confy.loader import Config as ActualConfyConfig
            import tomli as actual_tomli

            default_config_dict = {}
            try:
                if hasattr(importlib.resources, 'files'):
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # type: ignore
                        default_config_dict = actual_tomli.load(f)
                else:
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8')
                    default_config_dict = actual_tomli.loads(default_config_content)
                logger.debug("Successfully loaded and parsed default_config.toml.")
            except FileNotFoundError:
                logger.error("Packaged default_config.toml not found.")
                raise ConfigError("Critical error: Packaged default configuration is missing.")
            except actual_tomli.TOMLDecodeError as e:
                logger.error(f"Error parsing packaged default_config.toml: {e}")
                raise ConfigError(f"Critical error: Packaged default configuration is malformed: {e}")
            except Exception as e:
                logger.error(f"Could not load default_config.toml: {e}")
                raise ConfigError(f"Failed to load default configuration: {e}")

            instance.config = ActualConfyConfig(
                defaults=default_config_dict,
                file_path=config_file_path,
                prefix=env_prefix,
                overrides_dict=config_overrides,
                mandatory=[]
            )
            logger.info("confy configuration loaded successfully.")
            logger.debug(f"Effective default provider: {instance.config.get('llmcore.default_provider')}")
            logger.debug(f"Effective session storage type: {instance.config.get('storage.session.type')}")
            logger.debug(f"Effective vector storage type: {instance.config.get('storage.vector.type')}")

        except ImportError:
            logger.critical("Essential dependency 'confy' or 'tomli' is not installed.")
            raise ConfigError("Setup error: 'confy' or 'tomli' library not found.")
        except ConfigError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLMCore configuration: {e}", exc_info=True)
            raise ConfigError(f"Configuration initialization failed: {e}")

        # --- Step 2: Initialize Storage Backends ---
        await instance._initialize_storage()

        # --- Step 3: Initialize Other Managers (Placeholders) ---
        # self.provider_manager = ProviderManager(self.config)
        # self.embedding_manager = EmbeddingManager(self.config)
        # self.session_manager = SessionManager(self._session_storage) # Pass initialized storage
        # self.context_manager = ContextManager(...)
        logger.info("LLMCore managers (placeholders) initialization step.")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    async def _initialize_storage(self) -> None:
        """Initializes session and vector storage backends based on config."""
        # --- Initialize Session Storage ---
        session_storage_config = self.config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")
        logger.info(f"Initializing session storage of type: {session_storage_type}")

        if not session_storage_type:
            raise ConfigError("Session storage type ('storage.session.type') not configured.")

        session_storage_cls = SESSION_STORAGE_MAP.get(session_storage_type.lower())
        if not session_storage_cls:
            raise ConfigError(f"Unsupported session storage type: '{session_storage_type}'. "
                              f"Available types: {list(SESSION_STORAGE_MAP.keys())}")

        try:
            self._session_storage = session_storage_cls() # Instantiate
            # Pass the specific config section for the type (e.g., storage.session)
            await self._session_storage.initialize(session_storage_config)
            logger.info(f"Session storage backend '{session_storage_type}' initialized successfully.")
        except ConfigError as e: # Catch config errors during init
             logger.error(f"Configuration error during session storage initialization: {e}")
             raise ConfigError(f"Session storage config error: {e}")
        except SessionStorageError as e: # Catch storage-specific init errors
             logger.error(f"Failed to initialize session storage backend '{session_storage_type}': {e}")
             raise StorageError(f"Session storage init failed: {e}") # Wrap in generic StorageError
        except Exception as e:
             logger.error(f"Unexpected error initializing session storage '{session_storage_type}': {e}", exc_info=True)
             raise StorageError(f"Unexpected session storage init error: {e}")

        # --- Initialize Vector Storage (Placeholder) ---
        vector_storage_config = self.config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")
        logger.info(f"Initializing vector storage of type: {vector_storage_type} (placeholder)")

        if not vector_storage_type:
            logger.warning("Vector storage type ('storage.vector.type') not configured. RAG features will be unavailable.")
            self._vector_storage = None # type: ignore # Explicitly None if not configured
        else:
            # vector_storage_cls = VECTOR_STORAGE_MAP.get(vector_storage_type.lower())
            # if not vector_storage_cls:
            #     raise ConfigError(f"Unsupported vector storage type: '{vector_storage_type}'. "
            #                       f"Available types: {list(VECTOR_STORAGE_MAP.keys())}")
            # try:
            #     self._vector_storage = vector_storage_cls()
            #     await self._vector_storage.initialize(vector_storage_config)
            #     logger.info(f"Vector storage backend '{vector_storage_type}' initialized successfully.")
            # except ConfigError as e: ...
            # except VectorStorageError as e: ...
            # except Exception as e: ...
            logger.warning(f"Vector storage type '{vector_storage_type}' configured, but implementation is pending.")
            self._vector_storage = None # type: ignore # Placeholder

    # --- Core Chat Method (Skeleton - unchanged) ---
    async def chat(
        self,
        message: str,
        *, # Force subsequent arguments to be keyword-only
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        # RAG parameters
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        # Provider specific arguments (e.g., temperature, max_tokens for response)
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Sends a message to the configured LLM, managing context and session."""
        logger.debug(
            f"LLMCore.chat called with message: '{message[:50]}...', session_id: {session_id}, "
            f"provider: {provider_name}, model: {model_name}, stream: {stream}, RAG: {enable_rag}"
        )
        # --- Placeholder Implementation ---
        if stream:
            async def dummy_stream():
                yield "Placeholder streamed response chunk 1. "
                await asyncio.sleep(0.1)
                yield "Placeholder streamed response chunk 2."
            return dummy_stream()
        else:
            await asyncio.sleep(0.1)
            return f"Placeholder response to: '{message}'"

    # --- Session Management Methods (Skeletons - updated to use self._session_storage) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a specific chat session object (including messages)."""
        logger.debug(f"LLMCore.get_session called for session_id: {session_id}")
        if not hasattr(self, '_session_storage') or not self._session_storage:
             raise LLMCoreError("Session storage is not initialized.")
        try:
            return await self._session_storage.get_session(session_id)
        except SessionStorageError as e:
            logger.error(f"Storage error getting session '{session_id}': {e}")
            raise # Re-raise specific storage error
        except Exception as e:
            logger.error(f"Unexpected error getting session '{session_id}': {e}", exc_info=True)
            raise LLMCoreError(f"Failed to get session '{session_id}': {e}")


    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists available persistent chat sessions (metadata only)."""
        logger.debug("LLMCore.list_sessions called.")
        if not hasattr(self, '_session_storage') or not self._session_storage:
             raise LLMCoreError("Session storage is not initialized.")
        try:
            return await self._session_storage.list_sessions()
        except SessionStorageError as e:
            logger.error(f"Storage error listing sessions: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
            raise LLMCoreError(f"Failed to list sessions: {e}")


    async def delete_session(self, session_id: str) -> bool:
        """Deletes a persistent chat session from storage."""
        logger.debug(f"LLMCore.delete_session called for session_id: {session_id}")
        if not hasattr(self, '_session_storage') or not self._session_storage:
             raise LLMCoreError("Session storage is not initialized.")
        try:
            return await self._session_storage.delete_session(session_id)
        except SessionStorageError as e:
            logger.error(f"Storage error deleting session '{session_id}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting session '{session_id}': {e}", exc_info=True)
            raise LLMCoreError(f"Failed to delete session '{session_id}': {e}")


    # --- RAG / Vector Store Management Methods (Skeletons - unchanged) ---
    async def add_document_to_vector_store(
        self, content: str, *, metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None, collection_name: Optional[str] = None
    ) -> str:
        """Adds a single document (text content) to the configured vector store."""
        logger.debug(f"LLMCore.add_document_to_vector_store called for collection: {collection_name}, doc_id: {doc_id}")
        if not hasattr(self, '_vector_storage') or not self._vector_storage:
             raise LLMCoreError("Vector storage is not initialized or configured.")
        raise NotImplementedError("RAG/Vector store management not yet implemented.")

    async def add_documents_to_vector_store(
        self, documents: List[Dict[str, Any]], *, collection_name: Optional[str] = None
    ) -> List[str]:
        """Adds multiple documents to the configured vector store in a batch."""
        logger.debug(f"LLMCore.add_documents_to_vector_store called for collection: {collection_name}")
        if not hasattr(self, '_vector_storage') or not self._vector_storage:
             raise LLMCoreError("Vector storage is not initialized or configured.")
        raise NotImplementedError("RAG/Vector store management not yet implemented.")

    async def search_vector_store(
        self, query: str, *, k: int, collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[ContextDocument]:
        """Performs a similarity search for relevant documents in the vector store."""
        logger.debug(f"LLMCore.search_vector_store called for query: '{query[:50]}...', k: {k}, collection: {collection_name}")
        if not hasattr(self, '_vector_storage') or not self._vector_storage:
             raise LLMCoreError("Vector storage is not initialized or configured.")
        raise NotImplementedError("RAG/Vector store management not yet implemented.")

    async def delete_documents_from_vector_store(
        self, document_ids: List[str], *, collection_name: Optional[str] = None
    ) -> bool:
        """Deletes documents from the vector store by their IDs."""
        logger.debug(f"LLMCore.delete_documents_from_vector_store called for IDs: {document_ids}, collection: {collection_name}")
        if not hasattr(self, '_vector_storage') or not self._vector_storage:
             raise LLMCoreError("Vector storage is not initialized or configured.")
        raise NotImplementedError("RAG/Vector store management not yet implemented.")

    # --- Provider Info Methods (Skeletons - unchanged) ---
    def get_available_providers(self) -> List[str]:
        """Lists the names of all configured LLM providers."""
        logger.debug("LLMCore.get_available_providers called.")
        # To be implemented: Uses self.provider_manager
        raise NotImplementedError("Provider info methods not yet implemented.")

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Lists available models for a specific configured provider."""
        logger.debug(f"LLMCore.get_models_for_provider called for provider: {provider_name}")
        # To be implemented: Uses self.provider_manager
        raise NotImplementedError("Provider info methods not yet implemented.")

    # --- Utility / Cleanup ---
    async def close(self):
        """
        Closes connections for storage backends and potentially other resources.
        Should be called when the application using LLMCore is shutting down.
        """
        logger.info("LLMCore.close() called. Cleaning up resources...")
        # Close session storage
        if hasattr(self, '_session_storage') and self._session_storage:
            try:
                await self._session_storage.close()
                logger.info("Session storage closed.")
            except Exception as e:
                logger.error(f"Error closing session storage: {e}", exc_info=True)

        # Close vector storage (placeholder)
        if hasattr(self, '_vector_storage') and self._vector_storage:
             try:
                 # await self._vector_storage.close() # Uncomment when implemented
                 logger.info("Vector storage closed (placeholder).")
             except Exception as e:
                 logger.error(f"Error closing vector storage: {e}", exc_info=True)

        # Close provider manager (placeholder)
        # if hasattr(self, 'provider_manager') and self.provider_manager:
        #     await self.provider_manager.close()
        #     logger.info("Provider manager closed (placeholder).")

        logger.info("LLMCore resources cleanup complete.")

    # __await__ is no longer needed as initialization is done via async classmethod `create`

    async def __aenter__(self):
        """Allows using LLMCore with `async with`."""
        logger.debug("LLMCore.__aenter__ called.")
        # Initialization is now handled by `create`, so just return self
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensures resources are cleaned up when exiting `async with` block."""
        logger.debug(f"LLMCore.__aexit__ called. exc_type: {exc_type}")
        await self.close()
