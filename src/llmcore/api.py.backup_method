# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.

This module provides the main LLMCore class for interacting with Large Language Models,
managing conversation sessions, and performing Retrieval Augmented Generation (RAG).
"""

import asyncio
import importlib.resources
import json
import logging
import pathlib
import uuid
from datetime import datetime, timezone
from typing import (Any, AsyncGenerator, Dict, List, Optional, Tuple, Type,
                    Union)

import aiofiles

from .memory.manager import MemoryManager
from .embedding.manager import EmbeddingManager
from .exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
from .models import (ChatSession, ContextDocument, ContextItem,
                     ContextItemType, Message, Role, ContextPreparationDetails,
                     ContextPreset, ContextPresetItem, ModelDetails, Tool, ToolCall, ToolResult)
from .providers.base import BaseProvider
from .providers.manager import ProviderManager
from .sessions.manager import SessionManager
from .storage.manager import StorageManager

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any]  # type: ignore [no-redef]
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore [assignment]

logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, Retrieval Augmented
    Generation (RAG), and dynamic configuration. It is initialized asynchronously
    using the `LLMCore.create()` classmethod.

    This is a pure library implementation - no service components or async task queues.
    """
    config: ConfyConfig
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _memory_manager: MemoryManager
    _embedding_manager: EmbeddingManager
    _transient_sessions_cache: Dict[str, ChatSession]
    _transient_last_interaction_info_cache: Dict[str, ContextPreparationDetails]
    _log_raw_payloads_enabled: bool
    _llmcore_log_level_str: str

    def __init__(self):
        """
        Private constructor. Use `LLMCore.create()` for initialization.
        """
        self._transient_sessions_cache = {}
        self._transient_last_interaction_info_cache = {}

    @classmethod
    async def create(
        cls,
        config_overrides: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[str] = None,
        env_prefix: Optional[str] = "LLMCORE"
    ) -> "LLMCore":
        """
        Asynchronously creates and initializes an LLMCore instance.

        This factory method is the recommended way to create LLMCore instances.
        It loads configuration from multiple sources (defaults, file, env vars, overrides)
        and initializes all necessary managers.

        Args:
            config_overrides: Optional dictionary of configuration overrides
            config_file_path: Optional path to a TOML configuration file
            env_prefix: Environment variable prefix (default: "LLMCORE")

        Returns:
            Initialized LLMCore instance

        Raises:
            ConfigError: If configuration loading fails
            LLMCoreError: If initialization fails
        """
        instance = cls()
        await instance._initialize_from_config(config_overrides, config_file_path, env_prefix)
        return instance

    async def _initialize_from_config(
        self,
        config_overrides: Optional[Dict[str, Any]],
        config_file_path: Optional[str],
        env_prefix: Optional[str]
    ):
        """
        Initializes or re-initializes all components from a configuration.

        This method is used by both `create` and `reload_config`.
        It sets up all managers (providers, storage, sessions, memory, embeddings)
        based on the loaded configuration.

        Args:
            config_overrides: Optional configuration overrides
            config_file_path: Optional path to config file
            env_prefix: Environment variable prefix
        """
        logger.info("Initializing LLMCore components from configuration...")
        try:
            from confy.loader import Config as ActualConfyConfig
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required.")

            # Load default configuration from package
            default_config_dict = {}
            if hasattr(importlib.resources, 'files'):
                default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                with default_config_path_obj.open('rb') as f:
                    default_config_dict = tomllib.load(f)
            else:
                default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8')  # type: ignore
                default_config_dict = tomllib.loads(default_config_content)  # type: ignore

            # Create configuration object with layered precedence
            self.config = ActualConfyConfig(
                defaults=default_config_dict,
                file_path=config_file_path,
                prefix=env_prefix,
                overrides_dict=config_overrides
            )
        except Exception as e:
            raise ConfigError(f"LLMCore configuration loading failed: {e}")

        # Configure logging
        self._log_raw_payloads_enabled = self.config.get('llmcore.log_raw_payloads', False)
        self._llmcore_log_level_str = self.config.get('llmcore.log_level', 'INFO').upper()
        logging.getLogger("llmcore").setLevel(logging.getLevelName(self._llmcore_log_level_str))
        logger.info(f"LLMCore logger level set to: {self._llmcore_log_level_str}")

        # Initialize core managers
        self._provider_manager = ProviderManager(self.config)
        self._storage_manager = StorageManager(self.config)
        await self._storage_manager.initialize_storages()
        self._session_manager = SessionManager(self._storage_manager.get_session_storage())
        self._embedding_manager = EmbeddingManager(self.config)

        # Pre-load default embedding model if configured
        default_embedding_model = self.config.get('llmcore.default_embedding_model')
        if default_embedding_model:
            await self._embedding_manager.get_model(default_embedding_model)

        # Initialize memory manager (coordinates context building and RAG)
        self._memory_manager = MemoryManager(
            config=self.config,
            provider_manager=self._provider_manager,
            storage_manager=self._storage_manager,
            embedding_manager=self._embedding_manager
        )

        logger.info("LLMCore components initialization complete.")

    async def reload_config(self) -> None:
        """
        Performs a live, state-aware reload of the configuration.

        This method re-reads all configuration sources, re-initializes all
        managers (providers, storage, etc.), and restores transient state
        like in-memory chat sessions. This allows for dynamic updates to a
        long-running LLMCore instance without a full restart.

        **Enhanced Implementation Notes:**
        - Preserves transient sessions cache to prevent loss of non-persistent chat sessions
        - Preserves context preparation details cache for consistency
        - Implements proper error handling with state restoration on failure
        - Follows strict sequence: Preserve -> Shutdown -> Reload -> Restore
        - Logs detailed information for operational visibility

        **State Preservation:**
        The most critical aspect is preserving `_transient_sessions_cache` which contains
        active, non-persistent chat sessions. Losing this data would terminate ongoing
        conversations for users who haven't explicitly saved their sessions.
        """
        logger.info("Beginning configuration reload with state preservation...")

        # Step 1: Preserve transient state
        saved_sessions = self._transient_sessions_cache.copy()
        saved_context_info = self._transient_last_interaction_info_cache.copy()
        logger.debug(f"Preserved {len(saved_sessions)} transient sessions and {len(saved_context_info)} context info entries")

        # Step 2: Attempt to reload configuration
        old_config = self.config
        try:
            # Re-initialize from configuration (uses the same sources as create())
            await self._initialize_from_config(
                config_overrides=None,
                config_file_path=None,
                env_prefix="LLMCORE"
            )
            logger.info("Configuration reloaded successfully")

        except Exception as e:
            # Critical: Restore previous configuration on failure
            logger.error(f"Configuration reload failed: {e}. Restoring previous configuration.", exc_info=True)
            self.config = old_config
            raise ConfigError(f"Failed to reload configuration: {e}")

        # Step 3: Restore transient state
        self._transient_sessions_cache = saved_sessions
        self._transient_last_interaction_info_cache = saved_context_info
        logger.info(f"Restored {len(saved_sessions)} transient sessions and {len(saved_context_info)} context info entries")

        logger.info("Configuration reload complete with full state restoration")

    def get_available_providers(self) -> List[str]:
        """
        Returns a list of provider names that are currently configured and available.

        Returns:
            List of provider names (e.g., ['openai', 'anthropic', 'ollama'])
        """
        return self._provider_manager.list_providers()

    def get_provider_details(self, provider_name: Optional[str] = None) -> ModelDetails:
        """
        Gets detailed information about a specific provider or the default provider.

        Args:
            provider_name: Optional provider name. If None, returns default provider details.

        Returns:
            ModelDetails object containing provider information

        Raises:
            ProviderError: If the specified provider is not found
        """
        provider = self._provider_manager.get_provider(provider_name)
        return ModelDetails(
            provider_name=provider.get_name(),
            model_name=provider.default_model,
            supports_streaming=provider.supports_streaming,
            supports_tools=provider.supports_tools,
            max_context_length=provider.get_max_context_length(provider.default_model)
        )

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to an LLM and returns the response.

        This is the primary method for interacting with LLMs. It handles session management,
        context preparation, RAG integration, and both streaming and non-streaming responses.

        Args:
            message: The user's input message
            session_id: Optional session ID for conversation continuity
            system_message: Optional system message to define LLM behavior
            provider_name: Optional provider override (e.g., 'openai', 'anthropic')
            model_name: Optional model override (e.g., 'gpt-4', 'claude-3-opus')
            stream: If True, returns an async generator for streaming responses
            save_session: If True, saves the conversation turn to storage
            enable_rag: If True, retrieves relevant context from vector store
            rag_retrieval_k: Number of documents to retrieve for RAG
            rag_collection_name: Vector store collection name for RAG
            rag_metadata_filter: Optional metadata filter for RAG queries
            tools: Optional list of tools available to the LLM
            tool_choice: Optional tool choice strategy ('auto', 'required', or specific tool name)
            **provider_kwargs: Additional provider-specific parameters

        Returns:
            For non-streaming: The complete response string
            For streaming: An async generator yielding response chunks

        Raises:
            ProviderError: If provider interaction fails
            ContextLengthError: If context exceeds model's maximum length
            SessionStorageError: If session save/load fails
            VectorStorageError: If RAG retrieval fails
        """
        # Get the active provider
        active_provider = self._provider_manager.get_provider(provider_name)
        actual_model = model_name or active_provider.default_model

        # Validate provider kwargs against supported parameters
        supported_params = active_provider.get_supported_parameters(actual_model)
        for key in provider_kwargs:
            if key not in supported_params:
                raise ValueError(
                    f"Unsupported parameter '{key}' for provider '{active_provider.get_name()}'. "
                    f"Supported parameters are: {list(supported_params.keys())}"
                )

        # Load or create session
        chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
        if not session_id:  # If it was a temporary session, cache it
            self._transient_sessions_cache[chat_session.id] = chat_session

        # Add user message to session
        chat_session.add_message(message, Role.USER)

        # Prepare context (includes history, RAG, context management)
        context_details = await self._memory_manager.prepare_context(
            session=chat_session,
            provider_name=active_provider.get_name(),
            model_name=actual_model,
            enable_rag=enable_rag,
            rag_k=rag_retrieval_k,
            rag_collection=rag_collection_name,
            rag_filter=rag_metadata_filter
        )
        context_payload = context_details.prepared_messages

        # Cache context details for clients like llmchat
        self._transient_last_interaction_info_cache[chat_session.id] = context_details

        # Call provider
        response_data = await active_provider.chat_completion(
            context=context_payload,
            model=actual_model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **provider_kwargs
        )

        # Handle response
        if stream:
            return self._stream_response_wrapper(response_data, active_provider, chat_session, save_session)  # type: ignore
        else:
            full_content = self._extract_full_content(response_data, active_provider)
            chat_session.add_message(full_content, Role.ASSISTANT)
            if save_session:
                await self._session_manager.save_session(chat_session)
            return full_content

    async def _stream_response_wrapper(
        self, provider_stream: AsyncGenerator, provider: BaseProvider,
        session: ChatSession, do_save: bool
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider's stream, yields text, and handles session saving.

        Args:
            provider_stream: The async generator from the provider
            provider: The provider instance for response extraction
            session: The chat session to update
            do_save: Whether to save the session after streaming completes

        Yields:
            Text chunks from the LLM response
        """
        full_response = ""
        try:
            async for chunk in provider_stream:
                text_delta = self._extract_delta_content(chunk, provider)
                if text_delta:
                    full_response += text_delta
                    yield text_delta
        finally:
            if full_response:
                session.add_message(full_response, Role.ASSISTANT)
                if do_save:
                    await self._session_manager.save_session(session)

    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> str:
        """
        Extracts full response content from a non-streaming response.

        Args:
            response_data: The response dictionary from the provider
            provider: The provider instance

        Returns:
            The extracted text content
        """
        return provider.extract_response_content(response_data)

    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """
        Extracts delta content from a streaming chunk.

        Args:
            chunk: A single chunk from the streaming response
            provider: The provider instance

        Returns:
            The extracted text delta
        """
        return provider.extract_delta_content(chunk)

    def get_last_interaction_context_info(self, session_id: str) -> Optional[ContextPreparationDetails]:
        """
        Retrieves the context preparation details from the most recent interaction.

        This method is essential for clients like llmchat that need to display
        information about context usage, token counts, RAG documents used, etc.

        Args:
            session_id: The session ID to query

        Returns:
            ContextPreparationDetails if available, None otherwise
        """
        return self._transient_last_interaction_info_cache.get(session_id)

    async def list_sessions(self, limit: Optional[int] = None) -> List[ChatSession]:
        """
        Lists all available chat sessions.

        Args:
            limit: Optional maximum number of sessions to return

        Returns:
            List of ChatSession objects
        """
        return await self._session_manager.list_sessions(limit=limit)

    async def get_session(self, session_id: str) -> ChatSession:
        """
        Retrieves a specific chat session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            ChatSession object

        Raises:
            SessionNotFoundError: If the session doesn't exist
        """
        return await self._session_manager.get_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        """
        Deletes a chat session.

        Args:
            session_id: The session ID to delete

        Raises:
            SessionNotFoundError: If the session doesn't exist
        """
        await self._session_manager.delete_session(session_id)
        # Also remove from transient caches if present
        self._transient_sessions_cache.pop(session_id, None)
        self._transient_last_interaction_info_cache.pop(session_id, None)

    async def add_documents_to_vector_store(
        self,
        documents: List[ContextDocument],
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adds documents to the vector store for RAG.

        Args:
            documents: List of ContextDocument objects to add
            collection_name: Optional collection name (uses default if not specified)
            metadata_filter: Optional metadata to attach to documents

        Raises:
            VectorStorageError: If document addition fails
        """
        vector_storage = self._storage_manager.get_vector_storage(collection_name)
        await vector_storage.add_documents(documents, metadata_filter)

    async def search_vector_store(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[ContextDocument]:
        """
        Searches the vector store for relevant documents.

        Args:
            query: The search query
            k: Number of results to return
            collection_name: Optional collection name
            metadata_filter: Optional metadata filter

        Returns:
            List of relevant ContextDocument objects

        Raises:
            VectorStorageError: If search fails
        """
        vector_storage = self._storage_manager.get_vector_storage(collection_name)
        return await vector_storage.search(query, k=k, metadata_filter=metadata_filter)

    async def close(self) -> None:
        """
        Closes all connections and cleans up resources.

        This method should be called when shutting down to ensure proper cleanup
        of database connections, HTTP clients, and other resources.
        """
        logger.info("Closing LLMCore instance...")
        try:
            # Close storage connections
            if hasattr(self, '_storage_manager'):
                await self._storage_manager.close()

            # Close provider connections
            if hasattr(self, '_provider_manager'):
                await self._provider_manager.close_all()

            logger.info("LLMCore instance closed successfully")
        except Exception as e:
            logger.error(f"Error during LLMCore shutdown: {e}", exc_info=True)
