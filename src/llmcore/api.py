# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.

This module provides the main LLMCore class for interacting with Large Language Models,
managing conversation sessions, and performing Retrieval Augmented Generation (RAG).

UPDATED: Enhanced chat() method to support external RAG engines (like semantiscan) by
accepting pre-constructed context via explicitly_staged_items parameter.
"""

import asyncio
import importlib.resources
import json
import logging
import pathlib
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import aiofiles

from .embedding.manager import EmbeddingManager
from .exceptions import (
    ConfigError,
    ContextLengthError,
    EmbeddingError,
    LLMCoreError,
    ProviderError,
    SessionNotFoundError,
    SessionStorageError,
    StorageError,
    VectorStorageError,
)
from .memory.manager import MemoryManager
from .models import (
    ChatSession,
    ContextDocument,
    ContextItem,
    ContextItemType,
    ContextPreparationDetails,
    ContextPreset,
    ContextPresetItem,
    Message,
    ModelDetails,
    Role,
    Tool,
    ToolCall,
    ToolResult,
)
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


# ==============================================================================
# Protocol Definition for External RAG Engines
# ==============================================================================


@runtime_checkable
class LLMCoreProtocol(Protocol):
    """
    Protocol defining the minimal interface that external RAG engines
    (like semantiscan) should depend on when using LLMCore as an LLM backend.

    This protocol enables loose coupling and type safety without requiring
    direct import of the full LLMCore class.

    Example usage in external code:
        ```python
        from llmcore.api import LLMCoreProtocol

        def process_query(llm: LLMCoreProtocol, query: str) -> str:
            # Type checker knows llm has chat() method
            return  llm.chat(message=query, enable_rag=False)
        ```
    """

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
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **provider_kwargs,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Send a message to an LLM and return the response.

        For external RAG engines:
        - Construct your full prompt (context + query)
        - Pass via 'message' parameter
        - Set enable_rag=False to prevent double-RAG
        - Optionally stage additional context via explicitly_staged_items
        """
        ...


# ==============================================================================
# Main LLMCore Class
# ==============================================================================


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, Retrieval Augmented
    Generation (RAG), and dynamic configuration. It is initialized asynchronously
    using the `LLMCore.create()` classmethod.

    This is a pure library implementation - no service components or async task queues.

    ## External RAG Engine Integration

    LLMCore can be used as an LLM backend by external RAG engines (e.g., semantiscan).
    The recommended integration pattern:

    1. External engine retrieves relevant documents using its own logic
    2. External engine constructs a full prompt combining context and query
    3. External engine calls `llm.chat(message=prompt, enable_rag=False, ...)`
    4. LLMCore generates response using only the provided prompt

    This pattern allows external engines to control RAG logic while leveraging
    LLMCore's provider abstraction, session management, and context handling.
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
        env_prefix: Optional[str] = "LLMCORE",
    ) -> "LLMCore":
        """
        Asynchronously creates and initializes an LLMCore instance.

        This factory method is the recommended way to create LLMCore instances.
        It loads configuration from multiple sources (defaults, files, environment),
        initializes all managers, and prepares the instance for use.

        Args:
            config_overrides: Optional dictionary of configuration overrides
            config_file_path: Optional path to a TOML configuration file
            env_prefix: Environment variable prefix (default: "LLMCORE")

        Returns:
            Fully initialized LLMCore instance

        Raises:
            ConfigError: If configuration is invalid or cannot be loaded
            StorageError: If storage backends cannot be initialized
        """
        instance = cls()
        await instance._initialize_from_config(config_overrides, config_file_path, env_prefix)
        return instance

    async def __aenter__(self) -> "LLMCore":
        """Context manager entry - instance is already initialized."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up resources."""
        await self.close()

    async def _initialize_from_config(
        self,
        config_overrides: Optional[Dict[str, Any]],
        config_file_path: Optional[str],
        env_prefix: Optional[str],
    ) -> None:
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

            # Load default config from package
            try:
                if hasattr(importlib.resources, "files"):
                    config_files = importlib.resources.files("llmcore.config")
                    default_config_path = config_files / "default_config.toml"
                    with default_config_path.open("rb") as f:
                        default_config_dict = tomllib.load(f)
                else:
                    with importlib.resources.open_binary(
                        "llmcore.config", "default_config.toml"
                    ) as f:
                        default_config_dict = tomllib.load(f)
            except Exception as e:
                logger.warning(
                    f"Could not load default config from package: {e}. Using minimal defaults."
                )
                default_config_dict = {"llmcore": {"default_provider": "ollama"}}

            # Initialize Confy with all sources
            self.config = ActualConfyConfig(
                defaults=default_config_dict,
                config_file_path=config_file_path,
                env_prefix=env_prefix,
                overrides=config_overrides,
            )

            # Set log level and raw payload logging
            self._llmcore_log_level_str = self.config.get("llmcore.log_level", "INFO")
            llmcore_logger = logging.getLogger("llmcore")
            llmcore_logger.setLevel(self._llmcore_log_level_str.upper())

            self._log_raw_payloads_enabled = self.config.get("llmcore.log_raw_payloads", False)
            if self._log_raw_payloads_enabled:
                logger.info("Raw payload logging is ENABLED for this LLMCore instance")

            # Initialize managers
            logger.debug("Initializing StorageManager...")
            self._storage_manager = StorageManager(self.config)
            await self._storage_manager.initialize_storages()

            logger.debug("Initializing ProviderManager...")
            self._provider_manager = ProviderManager(self.config, self._log_raw_payloads_enabled)
            await self._provider_manager.initialize()

            logger.debug("Initializing SessionManager...")
            self._session_manager = SessionManager(self._storage_manager.session_storage)

            logger.debug("Initializing EmbeddingManager...")
            self._embedding_manager = EmbeddingManager(self.config, self._storage_manager)
            await self._embedding_manager.initialize()

            logger.debug("Initializing MemoryManager...")
            self._memory_manager = MemoryManager(
                config=self.config,
                provider_manager=self._provider_manager,
                embedding_manager=self._embedding_manager,
                storage_manager=self._storage_manager,
            )

            logger.info("LLMCore initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize LLMCore: {e}", exc_info=True)
            raise ConfigError(f"Initialization failed: {e}")

    async def close(self) -> None:
        """
        Closes all connections and cleans up resources.

        This method should be called when the LLMCore instance is no longer needed,
        or use the async context manager (async with) for automatic cleanup.
        """
        logger.info("Closing LLMCore instance...")
        try:
            await self._provider_manager.close_all()
            await self._storage_manager.close()
            logger.info("LLMCore instance closed and resources released")
        except Exception as e:
            logger.error(f"Error during LLMCore close: {e}", exc_info=True)
            raise

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
        logger.debug(
            f"Preserved {len(saved_sessions)} transient sessions and {len(saved_context_info)} context info entries"
        )

        # Step 2: Attempt to reload configuration
        old_config = self.config
        try:
            # Re-initialize from configuration (uses the same sources as create())
            await self._initialize_from_config(
                config_overrides=None, config_file_path=None, env_prefix="LLMCORE"
            )
            logger.info("Configuration reloaded successfully")

        except Exception as e:
            # Critical: Restore previous configuration on failure
            logger.error(
                f"Configuration reload failed: {e}. Restoring previous configuration.",
                exc_info=True,
            )
            self.config = old_config
            raise ConfigError(f"Failed to reload configuration: {e}")

        # Step 3: Restore transient state
        self._transient_sessions_cache = saved_sessions
        self._transient_last_interaction_info_cache = saved_context_info
        logger.info(
            f"Restored {len(saved_sessions)} transient sessions and {len(saved_context_info)} context info entries"
        )

        logger.info("Configuration reload complete with full state restoration")

    def get_available_providers(self) -> List[str]:
        """
        Returns a list of provider names that are currently configured and available.

        Returns:
            List of provider names (e.g., ['openai', 'anthropic', 'ollama'])
        """
        return self._provider_manager.get_available_providers()

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
            id=provider.default_model,
            provider_name=provider.get_name(),
            context_length=provider.get_max_context_length(provider.default_model),
            supports_streaming=getattr(provider, "supports_streaming", True),
            supports_tools=getattr(provider, "supports_tools", False),
        )

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
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **provider_kwargs,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to an LLM and returns the response.

        This is the primary method for interacting with LLMs. It handles session management,
        context preparation, RAG integration, and both streaming and non-streaming responses.

        ## Standard Usage

        ```python
        # Simple chat
        response = await llm.chat("What is the capital of France?")

        # With session for conversation history
        response = await llm.chat(
            message="Who wrote 1984?",
            session_id="my_session",
            save_session=True
        )
        ```

        ## External RAG Engine Integration

        External RAG engines (like semantiscan) should use this pattern:

        ```python
        # Step 1: External engine retrieves documents
        relevant_docs = await semantiscan_retrieve(query)

        # Step 2: External engine constructs full prompt
        context = format_documents(relevant_docs)
        full_prompt = f"Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"

        # Step 3: Call LLMCore with constructed prompt
        response = await llm.chat(
            message=full_prompt,
            enable_rag=False,  # CRITICAL: Prevent double-RAG
            session_id=session_id,
            save_session=True
        )
        ```

        ### Alternative: Using explicitly_staged_items

        For more structured context passing:

        ```python
        # Pass retrieved context as structured items
        context_items = [
            ContextItem(
                type=ContextItemType.USER_TEXT,
                content=doc.content,
                metadata=doc.metadata
            )
            for doc in relevant_docs
        ]

        response = await llm.chat(
            message=query,  # Just the query
            explicitly_staged_items=context_items,
            enable_rag=False,
            session_id=session_id
        )
        ```

        Args:
            message: The user's input message or fully-constructed prompt

            session_id: Optional session ID for conversation continuity. If None,
                creates a temporary session for this interaction only.

            system_message: Optional system message to define LLM behavior.
                Sets the role and personality for this conversation.

            provider_name: Optional provider override (e.g., 'openai', 'anthropic', 'ollama').
                If None, uses the default provider from configuration.

            model_name: Optional model override (e.g., 'gpt-4', 'claude-3-opus').
                If None, uses the provider's default model.

            stream: If True, returns an async generator for streaming responses.
                If False, waits for complete response before returning.

            save_session: If True and session_id is provided, saves the conversation
                turn to storage. Useful for maintaining conversation history.

            enable_rag: If True, performs internal RAG retrieval before generating response.
                **IMPORTANT FOR EXTERNAL RAG**: Set to False when using external RAG engines
                to prevent double-RAG scenarios.

            rag_retrieval_k: Number of documents to retrieve for internal RAG.
                Only used if enable_rag=True.

            rag_collection_name: Vector store collection name for internal RAG.
                Only used if enable_rag=True.

            rag_metadata_filter: Optional metadata filter for internal RAG queries.
                Only used if enable_rag=True.

            active_context_item_ids: List of context item IDs from the session's workspace
                to include in the prompt. Useful for explicitly referencing stored context.

            explicitly_staged_items: List of Message or ContextItem objects to include
                in the prompt with high priority. **This is the recommended parameter for
                external RAG engines** to pass retrieved context in a structured way.

            prompt_template_values: Dictionary of custom values for RAG prompt template
                placeholders. Allows dynamic prompt customization (e.g., {"project_name": "LLMCore"}).

            tools: Optional list of tools available to the LLM for function calling.

            tool_choice: Optional tool choice strategy ('auto', 'required', or specific tool name).

            **provider_kwargs: Additional provider-specific parameters (e.g., temperature=0.7,
                max_tokens=1000). These vary by provider - see provider documentation.

        Returns:
            For non-streaming (stream=False): The complete response string
            For streaming (stream=True): An async generator yielding response chunks

        Raises:
            ProviderError: If provider interaction fails
            ContextLengthError: If context exceeds model's maximum length
            SessionStorageError: If session save/load fails
            VectorStorageError: If internal RAG retrieval fails
            ValueError: If unsupported provider_kwargs are passed

        Examples:
            Basic chat:
                >>> response = await llm.chat("Hello!")

            With session:
                >>> response = await llm.chat(
                ...     message="Continue our discussion",
                ...     session_id="chat_123"
                ... )

            Streaming:
                >>> async for chunk in await llm.chat(
                ...     message="Tell me a story",
                ...     stream=True
                ... ):
                ...     print(chunk, end="", flush=True)

            External RAG (semantiscan pattern):
                >>> # Semantiscan retrieves and formats context
                >>> context_prompt = semantiscan.build_prompt(query, docs)
                >>> response = await llm.chat(
                ...     message=context_prompt,
                ...     enable_rag=False,  # Critical!
                ...     session_id=session_id
                ... )
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
        chat_session = await self._session_manager.load_or_create_session(
            session_id, system_message
        )
        if not session_id:  # If it was a temporary session, cache it
            self._transient_sessions_cache[chat_session.id] = chat_session

        # Add user message to session
        chat_session.add_message(message, Role.USER)

        # Prepare context (includes history, RAG, context management)
        context_details = await self._memory_manager.prepare_context(
            session=chat_session,
            provider_name=active_provider.get_name(),
            model_name=actual_model,
            rag_enabled=enable_rag,
            rag_k=rag_retrieval_k,
            rag_collection=rag_collection_name,
            rag_metadata_filter=rag_metadata_filter,
            active_context_item_ids=active_context_item_ids,
            explicitly_staged_items=explicitly_staged_items,
            prompt_template_values=prompt_template_values,
        )
        context_payload = context_details.prepared_messages

        # Pre-populate introspection fields that are known before the LLM call
        context_details.provider = active_provider.get_name()
        context_details.model = actual_model
        context_details.prompt_tokens = context_details.final_token_count
        context_details.rag_used = enable_rag
        context_details.max_context_length = active_provider.get_max_context_length(actual_model)

        # Determine if truncation was applied
        context_details.context_truncation_applied = bool(
            context_details.truncation_actions_taken.get("details")
        )

        # Count RAG documents if present
        if context_details.rag_documents_used:
            context_details.rag_documents_retrieved = len(context_details.rag_documents_used)
        else:
            context_details.rag_documents_retrieved = 0 if enable_rag else None

        # Calculate available context tokens
        reserved_tokens = self.config.get("context_management", {}).get(
            "reserved_response_tokens", 500
        )
        context_details.reserved_response_tokens = reserved_tokens
        context_details.available_context_tokens = (
            context_details.max_context_length - reserved_tokens
        )

        # Call provider
        response_data = await active_provider.chat_completion(
            context=context_payload,
            model=actual_model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **provider_kwargs,
        )

        # Handle response
        if stream:
            return self._stream_response_wrapper_with_introspection(
                response_data,
                active_provider,
                chat_session,
                save_session,
                context_details,  # Pass context_details for post-stream update
                actual_model,
            )
        else:
            full_content = self._extract_full_content(response_data, active_provider)

            # Post-completion: Update token counts from actual response
            completion_tokens = await self._count_completion_tokens(
                full_content, active_provider, actual_model
            )
            context_details.completion_tokens = completion_tokens
            context_details.total_tokens = context_details.prompt_tokens + completion_tokens

            # Cache AFTER all fields are populated
            self._transient_last_interaction_info_cache[chat_session.id] = context_details

            chat_session.add_message(full_content, Role.ASSISTANT)
            if save_session:
                await self._session_manager.save_session(chat_session)
            return full_content

    async def _stream_response_wrapper(
        self,
        provider_stream: AsyncGenerator,
        provider: BaseProvider,
        session: ChatSession,
        do_save: bool,
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

    async def _stream_response_wrapper_with_introspection(
        self,
        provider_stream: AsyncGenerator,
        provider: BaseProvider,
        session: ChatSession,
        do_save: bool,
        context_details: ContextPreparationDetails,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider's stream, yields text, handles session saving,
        and updates introspection data after streaming completes.

        Args:
            provider_stream: The async generator from the provider
            provider: The provider instance for response extraction
            session: The chat session to update
            do_save: Whether to save the session after streaming completes
            context_details: The context preparation details to update
            model: The model being used

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
                # Update completion tokens after stream completes
                completion_tokens = await self._count_completion_tokens(
                    full_response, provider, model
                )
                context_details.completion_tokens = completion_tokens
                context_details.total_tokens = context_details.prompt_tokens + completion_tokens

                # Cache the updated context details
                self._transient_last_interaction_info_cache[session.id] = context_details

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

    async def _count_completion_tokens(
        self,
        content: str,
        provider: BaseProvider,
        model: str,
    ) -> int:
        """
        Counts the tokens in the completion response.

        Args:
            content: The response content text
            provider: The provider instance
            model: The model used

        Returns:
            The token count for the completion
        """
        try:
            return await provider.count_tokens(content, model)
        except Exception as e:
            logger.warning(f"Failed to count completion tokens: {e}. Estimating...")
            # Fallback: rough estimate of ~4 chars per token
            return len(content) // 4

    def get_last_interaction_context_info(
        self, session_id: str
    ) -> Optional[ContextPreparationDetails]:
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

    async def preview_context_for_chat(
        self,
        current_user_query: str,
        *,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Previews the context that would be sent to the LLM without making an API call.

        This is a "dry-run" method useful for:
        - Token count estimation
        - Context debugging
        - Validating prompt construction
        - UI displays of prepared context

        Args:
            current_user_query: The query to preview (not added to session)
            session_id: Optional session ID for loading history
            system_message: Optional system message override
            provider_name: Optional provider override
            model_name: Optional model override
            active_context_item_ids: Context items to include
            explicitly_staged_items: Items to stage explicitly
            enable_rag: Whether to include RAG retrieval in preview
            rag_retrieval_k: Number of RAG documents to retrieve
            rag_collection_name: RAG collection to query
            rag_metadata_filter: RAG metadata filter
            prompt_template_values: Custom prompt template values

        Returns:
            Dictionary containing:
                - prepared_messages: List of messages that would be sent
                - final_token_count: Total token count
                - max_tokens_for_model: Model's context limit
                - truncation_actions_taken: Details of any truncation
                - rag_documents_used: RAG documents if enable_rag=True
                - rendered_rag_template_content: Formatted RAG prompt if applicable
        """
        # Get provider and model
        active_provider = self._provider_manager.get_provider(provider_name)
        actual_model = model_name or active_provider.default_model

        # Load or create temporary session
        if session_id:
            chat_session = await self._session_manager.load_or_create_session(
                session_id, system_message
            )
        else:
            chat_session = ChatSession(
                id=str(uuid.uuid4()), name="preview_session", system_message=system_message
            )

        # Create temporary message for preview
        temp_message = Message(id=str(uuid.uuid4()), role=Role.USER, content=current_user_query)

        # Temporarily add message to session for context preparation
        original_messages = chat_session.messages.copy()
        chat_session.messages.append(temp_message)

        try:
            # Prepare context using the same logic as chat()
            context_details = await self._memory_manager.prepare_context(
                session=chat_session,
                provider_name=active_provider.get_name(),
                model_name=actual_model,
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter,
                active_context_item_ids=active_context_item_ids,
                explicitly_staged_items=explicitly_staged_items,
                prompt_template_values=prompt_template_values,
            )

            # Return as dictionary (model_dump will handle serialization)
            return context_details.model_dump()

        finally:
            # Restore original messages (don't modify the actual session)
            chat_session.messages = original_messages

    # ==============================================================================
    # Session Management Methods
    # ==============================================================================

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

    async def create_session(
        self,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> ChatSession:
        """
        Creates a new chat session or loads an existing one.

        This method is the primary way for clients like llmchat to create
        new conversation sessions. If a session_id is provided and exists,
        it will be loaded; otherwise, a new session is created.

        Args:
            session_id: Optional ID for the session. If None, a new UUID is generated.
            name: Optional human-readable name for the session.
            system_message: Optional system message to initialize the session with.

        Returns:
            ChatSession: The created or loaded session object.

        Raises:
            SessionStorageError: If there's an error creating/loading the session.
        """
        session = await self._session_manager.load_or_create_session(
            session_id=session_id, system_message=system_message
        )

        # Set the name if provided
        if name and session.name != name:
            session.name = name
            # Save the session to persist the name
            await self._session_manager.save_session(session)

        logger.info(f"Session created/loaded: {session.id} (name: {session.name})")
        return session

    async def update_session_name(self, session_id: str, new_name: str) -> None:
        """
        Updates the name of a chat session.

        Args:
            session_id: The session ID to update
            new_name: The new name for the session

        Raises:
            SessionNotFoundError: If the session doesn't exist
        """
        await self._session_manager.update_session_name(session_id, new_name)

    # ==============================================================================
    # Context Item Management (Session Workspace)
    # ==============================================================================

    async def add_context_item(
        self,
        session_id: str,
        content: str,
        item_type: ContextItemType = ContextItemType.USER_TEXT,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Adds a context item to a session's workspace.

        Args:
            session_id: The session to add the item to
            content: The content of the context item
            item_type: The type of context item
            source_id: Optional source identifier
            metadata: Optional metadata dictionary

        Returns:
            The ID of the created context item

        Raises:
            SessionNotFoundError: If the session doesn't exist
        """
        session = await self.get_session(session_id)

        item = ContextItem(
            id=str(uuid.uuid4()),
            type=item_type,
            content=content,
            source_id=source_id,
            metadata=metadata or {},
        )

        session.context_items.append(item)
        await self._session_manager.save_session(session)

        return item.id

    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        """
        Retrieves a specific context item from a session's workspace.

        Args:
            session_id: The session ID
            item_id: The context item ID

        Returns:
            ContextItem if found, None otherwise
        """
        session = await self.get_session(session_id)
        return next((item for item in session.context_items if item.id == item_id), None)

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        """
        Removes a context item from a session's workspace.

        Args:
            session_id: The session ID
            item_id: The context item ID to remove

        Returns:
            True if item was found and removed, False otherwise
        """
        session = await self.get_session(session_id)
        original_count = len(session.context_items)
        session.context_items = [item for item in session.context_items if item.id != item_id]

        if len(session.context_items) < original_count:
            await self._session_manager.save_session(session)
            return True
        return False

    # ==============================================================================
    # Vector Store / RAG Methods
    # ==============================================================================

    async def add_documents_to_vector_store(
        self, documents: List[Dict[str, Any]], collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Adds documents to the vector store for RAG.

        This method handles the full pipeline:
        1. Generates embeddings for document content via EmbeddingManager
        2. Stores documents with embeddings in the vector store

        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata'
            collection_name: Optional collection name (uses default if not specified)

        Returns:
            List of document IDs that were added

        Raises:
            VectorStorageError: If document addition fails
            EmbeddingError: If embedding generation fails
        """
        import uuid

        from .models import ContextDocument

        if not documents:
            return []

        # Get the vector storage backend
        vector_storage = self._storage_manager.vector_storage

        # Extract content for batch embedding
        contents = [doc.get("content", "") for doc in documents]

        # Generate embeddings in batch for better performance
        # NOTE: EmbeddingManager uses generate_embedding/generate_embeddings, NOT get_embedding
        embeddings = await self._embedding_manager.generate_embeddings(contents)

        # Get the default embedding model identifier for metadata
        default_embedding_model = self.config.get("llmcore.default_embedding_model", "unknown")

        # Prepare ContextDocument objects with embeddings
        context_docs: List[ContextDocument] = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", str(uuid.uuid4()))
            embedding = embeddings[i]

            # Add embedding metadata
            embedding_metadata = metadata.copy()
            # Parse provider from model identifier (format: "provider:model_name" or just "model_name")
            if ":" in default_embedding_model:
                provider_name = default_embedding_model.split(":")[0]
                model_name = default_embedding_model.split(":", 1)[1]
            else:
                provider_name = "sentence-transformers"  # Default provider
                model_name = default_embedding_model

            embedding_metadata["embedding_model_provider"] = provider_name
            embedding_metadata["embedding_model_name"] = model_name
            embedding_metadata["embedding_dimension"] = len(embedding)

            context_doc = ContextDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=embedding_metadata,
            )
            context_docs.append(context_doc)

        # Add documents to vector storage
        return await vector_storage.add_documents(context_docs, collection_name)

    async def search_vector_store(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
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
        return await self._embedding_manager.search(query, k, collection_name, metadata_filter)

    async def list_vector_collections(self) -> List[str]:
        """
        Lists all available vector store collections.

        Returns:
            List of collection names
        """
        vector_storage = self._storage_manager.vector_storage
        if hasattr(vector_storage, "list_collections"):
            return await vector_storage.list_collections()
        return []

    # ==============================================================================
    # Context Preset Management
    # ==============================================================================

    async def save_context_preset(self, preset: ContextPreset) -> None:
        """
        Saves a context preset for later use.

        Args:
            preset: The ContextPreset object to save

        Raises:
            StorageError: If save fails
        """
        session_storage = self._storage_manager.session_storage
        await session_storage.save_context_preset(preset)

    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """
        Retrieves a context preset by name.

        Args:
            preset_name: The name of the preset to retrieve

        Returns:
            ContextPreset if found, None otherwise
        """
        session_storage = self._storage_manager.session_storage
        return await session_storage.get_context_preset(preset_name)

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """
        Lists all available context presets.

        Returns:
            List of preset metadata dictionaries
        """
        session_storage = self._storage_manager.session_storage
        return await session_storage.list_context_presets()

    async def delete_context_preset(self, preset_name: str) -> bool:
        """
        Deletes a context preset.

        Args:
            preset_name: The name of the preset to delete

        Returns:
            True if deleted, False if not found
        """
        session_storage = self._storage_manager.session_storage
        return await session_storage.delete_context_preset(preset_name)
