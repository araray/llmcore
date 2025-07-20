# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
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

from ..memory.manager import MemoryManager
from ..embedding.manager import EmbeddingManager
from ..exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError, ValueError)
from ..models import (ChatSession, ContextDocument, ContextItem,
                     ContextItemType, Message, Role, ContextPreparationDetails,
                     ContextPreset, ContextPresetItem, ModelDetails, Tool, ToolCall, ToolResult)
from ..providers.base import BaseProvider
from ..providers.manager import ProviderManager
from ..sessions.manager import SessionManager
from ..storage.manager import StorageManager
from ..agents.manager import AgentManager  # Add this import

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None # type: ignore [assignment]

logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, Retrieval Augmented
    Generation (RAG), and dynamic configuration. It is initialized asynchronously
    using the `LLMCore.create()` classmethod.
    """
    config: ConfyConfig
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _memory_manager: MemoryManager
    _embedding_manager: EmbeddingManager
    _agent_manager: AgentManager  # Add this attribute
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
        """
        logger.info("Initializing LLMCore components from configuration...")
        try:
            from confy.loader import Config as ActualConfyConfig
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required.")

            default_config_dict = {}
            if hasattr(importlib.resources, 'files'):
                default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                with default_config_path_obj.open('rb') as f:
                    default_config_dict = tomllib.load(f)
            else:
                default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8') # type: ignore
                default_config_dict = tomllib.loads(default_config_content) # type: ignore

            self.config = ActualConfyConfig(
                defaults=default_config_dict,
                file_path=config_file_path,
                prefix=env_prefix,
                overrides_dict=config_overrides
            )
        except Exception as e:
            raise ConfigError(f"LLMCore configuration loading failed: {e}")

        self._log_raw_payloads_enabled = self.config.get('llmcore.log_raw_payloads', False)
        self._llmcore_log_level_str = self.config.get('llmcore.log_level', 'INFO').upper()
        logging.getLogger("llmcore").setLevel(logging.getLevelName(self._llmcore_log_level_str))
        logger.info(f"LLMCore logger level set to: {self._llmcore_log_level_str}")

        self._provider_manager = ProviderManager(self.config)
        self._storage_manager = StorageManager(self.config)
        await self._storage_manager.initialize_storages()
        self._session_manager = SessionManager(self._storage_manager.get_session_storage())
        self._embedding_manager = EmbeddingManager(self.config)
        default_embedding_model = self.config.get('llmcore.default_embedding_model')
        if default_embedding_model:
            await self._embedding_manager.get_model(default_embedding_model)
        self._memory_manager = MemoryManager(
            config=self.config,
            provider_manager=self._provider_manager,
            storage_manager=self._storage_manager,
            embedding_manager=self._embedding_manager
        )
        # Initialize the AgentManager
        self._agent_manager = AgentManager(
            self._provider_manager,
            self._memory_manager,
            self._storage_manager
        )
        logger.info("LLMCore components initialization complete.")

    def get_agent_manager(self) -> AgentManager:
        """
        Returns the initialized AgentManager instance.

        Returns:
            AgentManager: The agent manager for orchestrating autonomous agent tasks
        """
        return self._agent_manager

    async def reload_config(self) -> None:
        """
        Performs a live, state-aware reload of the configuration.

        This method re-reads all configuration sources, re-initializes all
        managers (providers, storage, etc.), and restores transient state
        like in-memory chat sessions. This allows for dynamic updates to a
        long-running LLMCore instance without a full restart.
        """
        logger.info("Starting live configuration reload...")
        # 1. Preserve State
        preserved_sessions = self._transient_sessions_cache.copy()
        preserved_context_info = self._transient_last_interaction_info_cache.copy()
        logger.debug(f"Preserving {len(preserved_sessions)} transient sessions and {len(preserved_context_info)} context info entries.")

        # 2. Graceful Shutdown
        await self.close()

        # 3. Reload and Re-initialize
        # The original config file path and overrides are not stored, so we assume
        # the reload is based on the files on disk and environment variables.
        # This is the standard behavior for `confy.reload()`.
        try:
            self.config.reload()
            logger.info("Confy configuration reloaded from sources.")
            # Re-run initialization logic with the new config object
            await self._initialize_from_config(None, self.config._config_file_path_loaded_from, self.config._prefix)
        except Exception as e:
            logger.error(f"Failed to reload configuration and re-initialize managers: {e}", exc_info=True)
            # Attempt to restore to a usable state might be complex. For now, log and raise.
            raise ConfigError(f"Configuration reload failed: {e}")

        # 4. Restore State
        self._transient_sessions_cache = preserved_sessions
        self._transient_last_interaction_info_cache = preserved_context_info
        logger.info(f"Configuration reload complete. Restored {len(self._transient_sessions_cache)} transient sessions.")

    async def chat(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        use_mcp_tools: Optional[List[str]] = None, # Placeholder for future MCP integration
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, RAG, and tool-calling.

        Args:
            message: The user's input message.
            session_id: The ID of the conversation session.
            system_message: A message defining the LLM's behavior.
            provider_name: Overrides the default LLM provider for this call.
            model_name: Overrides the default model for the chosen provider.
            stream: If True, returns an async generator of text chunks.
            save_session: If True, saves the conversation turn to storage.
            tools: A list of `Tool` objects available for the LLM to call.
            tool_choice: A string to control how the model uses tools (e.g., "auto", "any").
            use_mcp_tools: (For future use) A list of remote MCP tools to use.
            **provider_kwargs: Additional arguments passed directly to the provider's API,
                               which are validated before the call.

        Returns:
            The full response string or a stream of response chunks.

        Raises:
            ProviderError, ContextLengthError, ConfigError, ValueError (for invalid params).
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        actual_model = model_name or active_provider.default_model
        if not actual_model:
            raise ConfigError(f"Target model for provider '{active_provider.get_name()}' is not defined.")

        # Pre-flight parameter validation
        supported_params = active_provider.get_supported_parameters(actual_model)
        for key in provider_kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for provider '{active_provider.get_name()}'. "
                                 f"Supported parameters are: {list(supported_params.keys())}")

        # Session handling logic... (remains largely the same)
        chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
        if not session_id: # If it was a temporary session, cache it
            self._transient_sessions_cache[chat_session.id] = chat_session
        chat_session.add_message(message, Role.USER)

        # Context preparation logic... (remains largely the same)
        context_details = await self._memory_manager.prepare_context(
            session=chat_session,
            provider_name=active_provider.get_name(),
            model_name=actual_model
        )
        context_payload = context_details.prepared_messages
        self._transient_last_interaction_info_cache[chat_session.id] = context_details

        # Provider call
        response_data = await active_provider.chat_completion(
            context=context_payload,
            model=actual_model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **provider_kwargs
        )

        # Response handling...
        if stream:
            return self._stream_response_wrapper(response_data, active_provider, chat_session, save_session) # type: ignore
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
        """Wraps provider's stream, yields text, and handles session saving."""
        full_response = ""
        try:
            async for chunk in provider_stream:
                # This part will need enhancement to handle tool_call deltas
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
        """Extracts full response content from a non-streaming response."""
        # This part will need enhancement to handle tool_calls in the response
        try:
            return response_data['choices'][0]['message']['content'] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Could not extract content from {provider.get_name()} response: {response_data}")
            return ""

    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts text delta from a streaming chunk."""
        # This part will need enhancement to handle tool_call deltas
        try:
            return chunk['choices'][0]['delta'].get('content', '') or ""
        except (KeyError, IndexError, TypeError):
            return ""

    # --- Other methods ---

    async def get_models_details_for_provider(self, provider_name: str) -> List[ModelDetails]:
        """
        Gets detailed information for all available models from a specific provider.

        Args:
            provider_name: The name of the provider to query.

        Returns:
            A list of `ModelDetails` objects.
        """
        provider = self._provider_manager.get_provider(provider_name)
        return await provider.get_models_details()

    def get_available_providers(self) -> List[str]:
        """Lists the names of all successfully loaded provider instances."""
        return self._provider_manager.get_available_providers()

    async def close(self):
        """Closes all provider and storage connections gracefully."""
        logger.info("Closing LLMCore resources...")
        await asyncio.gather(
            self._provider_manager.close_providers(),
            self._storage_manager.close_storages(),
            self._embedding_manager.close(),
            return_exceptions=True
        )
        self._transient_sessions_cache.clear()
        self._transient_last_interaction_info_cache.clear()
        logger.info("LLMCore resources cleanup complete.")

    # --- Methods below this line are simplified for brevity and would be fully implemented ---

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        return await self._session_manager.get_session_if_exists(session_id)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return await self._storage_manager.get_session_storage().list_sessions()

    async def delete_session(self, session_id: str) -> bool:
        # Also clear from transient cache
        self._transient_sessions_cache.pop(session_id, None)
        self._transient_last_interaction_info_cache.pop(session_id, None)
        return await self._storage_manager.get_session_storage().delete_session(session_id)

    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], collection_name: Optional[str] = None) -> List[str]:
        # Simplified implementation
        contents = [doc["content"] for doc in documents]
        embeddings = await self._embedding_manager.generate_embeddings(contents)
        docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata", {})) for d, emb in zip(documents, embeddings)]
        return await self._storage_manager.get_vector_storage().add_documents(docs_to_add, collection_name)

    async def search_vector_store(self, query: str, k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict] = None) -> List[ContextDocument]:
        query_embedding = await self._embedding_manager.generate_embedding(query)
        return await self._storage_manager.get_vector_storage().similarity_search(query_embedding, k, collection_name, filter_metadata)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
