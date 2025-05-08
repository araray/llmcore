# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
"""

import asyncio
import logging
import importlib.resources
import pathlib
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Type

# Models and Exceptions
from .models import ChatSession, ContextDocument, Message, Role
from .exceptions import (
    LLMCoreError, ProviderError, SessionNotFoundError, ConfigError,
    StorageError, SessionStorageError, VectorStorageError,
    EmbeddingError, ContextLengthError, MCPError
)
# Storage
from .storage.base_session import BaseSessionStorage
from .storage.json_session import JsonSessionStorage
from .storage.sqlite_session import SqliteSessionStorage
# Sessions
from .sessions.manager import SessionManager
# Context
from .context.manager import ContextManager
# Providers
from .providers.base import BaseProvider
from .providers.ollama_provider import OllamaProvider # Import Ollama directly for now

# confy and tomli
try:
    from confy.loader import Config as ConfyConfig
    import tomli
except ImportError as e:
    ConfyConfig = Dict[str, Any] # type: ignore
    logging.getLogger(__name__).warning(f"confy or tomli not imported: {e}")

logger = logging.getLogger(__name__)

# Storage mapping
SESSION_STORAGE_MAP: Dict[str, Type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
}

# --- Temporary Provider Mapping (until ProviderManager) ---
# This allows us to instantiate the default provider directly for Phase 1
TEMP_PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
    "ollama": OllamaProvider,
    # Add other providers here if needed for temporary testing before ProviderManager
    # "openai": OpenAIProvider,
    # "anthropic": AnthropicProvider,
}
# --- End Temporary Provider Mapping ---


class LLMCore:
    """
    Main class for interacting with Large Language Models.
    Instantiate using 'instance = await LLMCore.create(...)'.
    """

    config: ConfyConfig
    _session_storage: BaseSessionStorage
    _session_manager: SessionManager
    _context_manager: ContextManager
    _provider: BaseProvider # Temporarily store the single default provider instance

    # _vector_storage: BaseVectorStorage
    # _provider_manager: ProviderManager # Will replace _provider later
    # _embedding_manager: EmbeddingManager

    def __init__(self):
        """Private constructor. Use `create` classmethod for async initialization."""
        pass

    @classmethod
    async def create(
        cls,
        config_overrides: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[str] = None,
        env_prefix: Optional[str] = "LLMCORE"
    ) -> "LLMCore":
        """Asynchronously creates and initializes an LLMCore instance."""
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        # --- Step 1: Initialize Config ---
        try:
            from confy.loader import Config as ActualConfyConfig
            import tomli as actual_tomli
            default_config_dict = {}
            try:
                # Load default config (handling different Python versions)
                if hasattr(importlib.resources, 'files'):
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # type: ignore
                        default_config_dict = actual_tomli.load(f)
                else:
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8')
                    default_config_dict = actual_tomli.loads(default_config_content)
            except Exception as e:
                 raise ConfigError(f"Failed to load default configuration: {e}")

            instance.config = ActualConfyConfig(
                defaults=default_config_dict, file_path=config_file_path,
                prefix=env_prefix, overrides_dict=config_overrides, mandatory=[]
            )
            logger.info("confy configuration loaded successfully.")
        except ImportError:
            raise ConfigError("Setup error: 'confy' or 'tomli' library not found.")
        except ConfigError: raise
        except Exception as e:
            raise ConfigError(f"Configuration initialization failed: {e}")

        # --- Step 2: Initialize Storage Backends ---
        await instance._initialize_storage()

        # --- Step 3: Initialize Session Manager ---
        try:
            instance._session_manager = SessionManager(instance._session_storage)
            logger.info("SessionManager initialized successfully.")
        except LLMCoreError as e:
             raise LLMCoreError(f"Failed to initialize SessionManager: {e}")

        # --- Step 4: Initialize Provider (Temporary - using default) ---
        # This replaces ProviderManager for Phase 1 basic functionality
        try:
            default_provider_name = instance.config.get('llmcore.default_provider', 'ollama')
            provider_cls = TEMP_PROVIDER_MAP.get(default_provider_name.lower())
            if not provider_cls:
                 raise ConfigError(f"Default provider '{default_provider_name}' not found or not supported in temporary map.")

            provider_config = instance.config.get(f'providers.{default_provider_name}', {})
            instance._provider = provider_cls(provider_config) # Instantiate the default provider
            logger.info(f"Default provider '{default_provider_name}' initialized temporarily.")
        except ConfigError as e:
            raise # Re-raise config errors
        except Exception as e:
            logger.error(f"Failed to initialize default provider: {e}", exc_info=True)
            raise ProviderError("default", f"Initialization failed for default provider: {e}")

        # --- Step 5: Initialize Embedding Manager (Placeholder) ---
        logger.warning("EmbeddingManager initialization is pending.")

        # --- Step 6: Initialize Context Manager ---
        try:
            instance._context_manager = ContextManager(instance.config)
            logger.info("ContextManager initialized successfully.")
        except Exception as e:
            raise LLMCoreError(f"ContextManager initialization failed: {e}")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    async def _initialize_storage(self) -> None:
        """Initializes session and vector storage backends based on config."""
        # Session Storage
        session_storage_config = self.config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")
        if not session_storage_type: raise ConfigError("Session storage type not configured.")
        session_storage_cls = SESSION_STORAGE_MAP.get(session_storage_type.lower())
        if not session_storage_cls: raise ConfigError(f"Unsupported session storage type: '{session_storage_type}'.")
        try:
            self._session_storage = session_storage_cls()
            await self._session_storage.initialize(session_storage_config)
            logger.info(f"Session storage backend '{session_storage_type}' initialized.")
        except Exception as e: raise StorageError(f"Session storage init failed: {e}")

        # Vector Storage (Placeholder)
        vector_storage_config = self.config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")
        logger.info(f"Initializing vector storage of type: {vector_storage_type} (placeholder)")
        self._vector_storage = None # type: ignore

    # --- Core Chat Method ---
    async def chat(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None, # Ignored for now, uses default provider
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        enable_rag: bool = False, # Placeholder
        rag_retrieval_k: Optional[int] = None, # Placeholder
        rag_collection_name: Optional[str] = None, # Placeholder
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Sends a message to the configured LLM, managing context and session."""
        # --- Provider Selection (Temporary) ---
        # For Phase 1, we ignore provider_name and use the single initialized default provider
        if provider_name and provider_name.lower() != self._provider.get_name().lower():
             logger.warning(f"Provider '{provider_name}' requested, but only default provider "
                            f"'{self._provider.get_name()}' is available in Phase 1. Using default.")
        active_provider = self._provider
        target_model = model_name or active_provider.default_model # Use specified or provider's default
        # --- End Temporary Provider Selection ---

        logger.debug(
            f"LLMCore.chat: session='{session_id}', provider='{active_provider.get_name()}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}"
        )

        try:
            # 1. Load or create session
            # Note: load_or_create raises SessionNotFoundError if ID provided but not found
            chat_session = await self._session_manager.load_or_create_session(
                session_id, system_message
            )
            # Add user message to the session object in memory
            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}'")

            # --- RAG Placeholder ---
            rag_results_content: Optional[List[str]] = None
            if enable_rag:
                logger.warning("RAG search and result handling not yet implemented.")
            # --- End RAG Placeholder ---

            # 2. Prepare context using ContextManager
            context_payload: List[Message] = await self._context_manager.prepare_context(
                session=chat_session,
                provider=active_provider,
                model_name=target_model,
                rag_results=rag_results_content,
            )
            logger.info(f"Prepared context with {len(context_payload)} messages for model '{target_model}'.")

            # 3. Call provider's chat_completion
            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload,
                model=target_model,
                stream=stream,
                **provider_kwargs
            )

            # 4. Process response and save session
            if stream:
                # --- Streaming Path (Placeholder for Task 2.1) ---
                logger.warning("Streaming response handling not yet fully implemented in LLMCore.chat.")
                async def dummy_stream_placeholder():
                    # Simulate basic streaming from OllamaProvider structure
                    full_response_content = ""
                    async for chunk in response_data_or_generator: # type: ignore
                        text_delta = ""
                        if isinstance(chunk, dict):
                             # Basic parsing for Ollama stream chunk
                             text_delta = chunk.get('message', {}).get('content', '')
                        if text_delta:
                             full_response_content += text_delta
                             yield text_delta
                    # Save after stream (if persistent and save enabled)
                    if save_session and session_id:
                        assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                        logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}' after stream.")
                        await self._session_manager.save_session(chat_session)

                return dummy_stream_placeholder()
                # --- End Streaming Placeholder ---
            else:
                # --- Non-Streaming Path (Implemented for Task 1.8) ---
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response from provider for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(active_provider.get_name(), "Invalid response format received from provider.")

                response_data = response_data_or_generator

                # Extract content based on expected provider format (Ollama example)
                # TODO: Refactor this parsing logic into provider-specific methods or a helper later
                message_part = response_data.get('message', {})
                if isinstance(message_part, dict):
                    full_response_content = message_part.get('content', '')
                else:
                     # Handle cases where response might be different (e.g., older Ollama /generate)
                     full_response_content = response_data.get('response', '') # Fallback for /generate

                if not full_response_content and response_data:
                     logger.warning(f"Could not extract 'content' from non-streaming response: {response_data}")
                     full_response_content = str(response_data) # Fallback to string representation

                logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                # Add assistant message and save session (if persistent and save enabled)
                if save_session and session_id:
                    assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                    logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")
                    await self._session_manager.save_session(chat_session)

                return full_response_content
                # --- End Non-Streaming Path ---

        except (SessionNotFoundError, SessionStorageError, ProviderError, ContextLengthError, ConfigError) as e:
             logger.error(f"Chat failed: {e}")
             raise # Propagate specific, known errors
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             raise LLMCoreError(f"Chat execution failed: {e}")


    # --- Session Management Methods (Unchanged) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a specific chat session object (including messages)."""
        logger.debug(f"LLMCore.get_session called for session_id: {session_id}")
        if not hasattr(self, '_session_manager') or not self._session_manager:
             raise LLMCoreError("Session manager is not initialized.")
        return await self._session_manager.load_or_create_session(session_id=session_id)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists available persistent chat sessions (metadata only)."""
        logger.debug("LLMCore.list_sessions called.")
        if not hasattr(self, '_session_storage') or not self._session_storage:
             raise LLMCoreError("Session storage is not initialized.")
        return await self._session_storage.list_sessions()

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a persistent chat session from storage."""
        logger.debug(f"LLMCore.delete_session called for session_id: {session_id}")
        if not hasattr(self, '_session_storage') or not self._session_storage:
             raise LLMCoreError("Session storage is not initialized.")
        return await self._session_storage.delete_session(session_id)

    # --- RAG / Vector Store Management Methods (Skeletons - unchanged) ---
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict] = None, doc_id: Optional[str] = None, collection_name: Optional[str] = None) -> str: raise NotImplementedError
    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str] = None) -> List[str]: raise NotImplementedError
    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict] = None) -> List[ContextDocument]: raise NotImplementedError
    async def delete_documents_from_vector_store(self, document_ids: List[str], *, collection_name: Optional[str] = None) -> bool: raise NotImplementedError


    # --- Provider Info Methods (Skeletons - unchanged) ---
    def get_available_providers(self) -> List[str]: raise NotImplementedError
    def get_models_for_provider(self, provider_name: str) -> List[str]: raise NotImplementedError

    # --- Utility / Cleanup ---
    async def close(self):
        """Closes connections for storage backends and potentially other resources."""
        logger.info("LLMCore.close() called. Cleaning up resources...")
        # Close session storage
        if hasattr(self, '_session_storage') and self._session_storage:
            try: await self._session_storage.close(); logger.info("Session storage closed.")
            except Exception as e: logger.error(f"Error closing session storage: {e}", exc_info=True)
        # Close provider (temporary instance)
        if hasattr(self, '_provider') and self._provider and hasattr(self._provider, 'close'):
             try: await self._provider.close(); logger.info("Default provider closed.") # type: ignore
             except Exception as e: logger.error(f"Error closing default provider: {e}", exc_info=True)
        # (Close vector storage, provider manager etc. when implemented)
        logger.info("LLMCore resources cleanup complete.")

    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc_val, exc_tb): await self.close()
