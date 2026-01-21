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
import time
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import aiofiles
import yaml

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
    CostEstimate,
    Message,
    ModelDetails,
    ModelValidationResult,
    PullProgress,
    PullResult,
    Role,
    SessionTokenStats,
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

    # =========================================================================
    # Statistics & Introspection
    # =========================================================================
    async def get_session_token_stats(self, session_id: str) -> SessionTokenStats:
        """
        Get cumulative token usage statistics for a session.

        Aggregates all token usage data from interactions within the session
        to provide summary statistics useful for monitoring and cost estimation.

        The method uses two data sources:
        1. Session metadata's "interactions" list (preferred, contains detailed data)
        2. Message token counts (fallback for sessions without interaction tracking)

        Args:
            session_id: Session to get stats for.

        Returns:
            SessionTokenStats with totals, averages, and timing info.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        Example:
            >>> stats = await llm.get_session_token_stats("session_123")
            >>> print(f"Total tokens: {stats.total_tokens:,}")
            >>> print(f"Interactions: {stats.interaction_count}")
            >>> print(f"Avg per interaction: {stats.avg_prompt_tokens:.0f} prompt")
        """
        # Get session asynchronously
        session = await self._session_manager.load_or_create_session(session_id)

        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Initialize stats
        stats = SessionTokenStats(
            session_id=session_id,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
        )

        # Try to use recorded interaction data from session metadata
        interaction_data = session.metadata.get("interactions", [])

        if interaction_data:
            # Use recorded interaction data (preferred)
            for interaction in interaction_data:
                prompt_tokens = interaction.get("prompt_tokens", 0)
                completion_tokens = interaction.get("completion_tokens", 0)
                cached_tokens = interaction.get("cached_tokens", 0)

                stats.total_prompt_tokens += prompt_tokens
                stats.total_completion_tokens += completion_tokens
                stats.total_cached_tokens += cached_tokens
                stats.interaction_count += 1

                # Track max values
                stats.max_prompt_tokens = max(stats.max_prompt_tokens, prompt_tokens)
                stats.max_completion_tokens = max(stats.max_completion_tokens, completion_tokens)

                # Track by model
                model = interaction.get("model", "unknown")
                if model not in stats.by_model:
                    stats.by_model[model] = {"prompt": 0, "completion": 0, "count": 0}
                stats.by_model[model]["prompt"] += prompt_tokens
                stats.by_model[model]["completion"] += completion_tokens
                stats.by_model[model]["count"] += 1

                # Track timing
                timestamp_str = interaction.get("timestamp")
                if timestamp_str:
                    try:
                        if timestamp_str.endswith("Z"):
                            ts = datetime.fromisoformat(timestamp_str[:-1] + "+00:00")
                        else:
                            ts = datetime.fromisoformat(timestamp_str)

                        if stats.first_interaction_at is None or ts < stats.first_interaction_at:
                            stats.first_interaction_at = ts
                        if stats.last_interaction_at is None or ts > stats.last_interaction_at:
                            stats.last_interaction_at = ts
                    except (ValueError, TypeError):
                        pass
        else:
            # Fallback: estimate from message token counts
            user_messages = [m for m in session.messages if m.role == Role.USER]
            assistant_messages = [m for m in session.messages if m.role == Role.ASSISTANT]

            for msg in session.messages:
                if msg.tokens:
                    if msg.role == Role.ASSISTANT:
                        stats.total_completion_tokens += msg.tokens
                        stats.max_completion_tokens = max(stats.max_completion_tokens, msg.tokens)
                    elif msg.role == Role.USER:
                        stats.total_prompt_tokens += msg.tokens
                        stats.max_prompt_tokens = max(stats.max_prompt_tokens, msg.tokens)

            stats.interaction_count = min(len(user_messages), len(assistant_messages))

            # Use message timestamps for timing
            if session.messages:
                sorted_msgs = sorted(session.messages, key=lambda m: m.timestamp)
                stats.first_interaction_at = sorted_msgs[0].timestamp
                stats.last_interaction_at = sorted_msgs[-1].timestamp

        # Calculate totals and averages
        stats.total_tokens = stats.total_prompt_tokens + stats.total_completion_tokens

        if stats.interaction_count > 0:
            stats.avg_prompt_tokens = stats.total_prompt_tokens / stats.interaction_count
            stats.avg_completion_tokens = stats.total_completion_tokens / stats.interaction_count

        return stats

    def estimate_cost(
        self,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> CostEstimate:
        """
        Estimate cost for given token counts.

        Uses the model card library for pricing data. If pricing is not
        available (e.g., local Ollama models), returns an estimate with
        pricing_source="unavailable".

        Args:
            provider_name: Provider name (e.g., "anthropic", "openai").
            model_name: Model name (e.g., "claude-sonnet-4", "gpt-4o").
            prompt_tokens: Input/prompt token count.
            completion_tokens: Output/completion token count.
            cached_tokens: Tokens served from cache (discount applied).
            reasoning_tokens: Reasoning/thinking tokens (special pricing).

        Returns:
            CostEstimate with cost breakdown and total.

        Example:
            >>> cost = llm.estimate_cost(
            ...     "anthropic", "claude-sonnet-4",
            ...     prompt_tokens=10000,
            ...     completion_tokens=2000,
            ...     cached_tokens=5000
            ... )
            >>> print(f"Estimated cost: {cost.format_cost()}")
            >>> print(f"  Input: ${cost.input_cost:.4f}")
            >>> print(f"  Output: ${cost.output_cost:.4f}")
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        card = registry.get(provider_name, model_name)

        # Initialize result with token counts
        result = CostEstimate(
            input_cost=0.0,
            output_cost=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            provider=provider_name,
            model_id=model_name,
        )

        if not card or not card.pricing:
            # No pricing available (e.g., local models)
            result.pricing_source = "unavailable"
            logger.debug(f"No pricing data for {provider_name}/{model_name}")
            return result

        pricing = card.pricing.per_million_tokens
        result.pricing_source = "model_card"
        result.currency = card.pricing.currency

        # Store pricing info
        result.input_price_per_million = pricing.input
        result.output_price_per_million = pricing.output
        result.cached_price_per_million = pricing.cached_input

        # Calculate input cost
        regular_input_tokens = max(0, prompt_tokens - cached_tokens)
        result.input_cost = (regular_input_tokens / 1_000_000) * pricing.input

        # Add cached token cost if applicable
        if cached_tokens > 0:
            if pricing.cached_input is not None:
                cached_cost = (cached_tokens / 1_000_000) * pricing.cached_input
            else:
                # If no cached price, charge full input price
                cached_cost = (cached_tokens / 1_000_000) * pricing.input
            result.input_cost += cached_cost

            # Calculate savings from caching
            full_input_cost = (cached_tokens / 1_000_000) * pricing.input
            if pricing.cached_input is not None:
                result.cached_discount = full_input_cost - (
                    (cached_tokens / 1_000_000) * pricing.cached_input
                )

        # Calculate output cost
        regular_output_tokens = max(0, completion_tokens - reasoning_tokens)
        result.output_cost = (regular_output_tokens / 1_000_000) * pricing.output

        # Calculate reasoning cost (if separate pricing exists)
        if reasoning_tokens > 0:
            if pricing.reasoning_output is not None:
                result.reasoning_cost = (reasoning_tokens / 1_000_000) * pricing.reasoning_output
            else:
                # If no separate reasoning price, use regular output price
                result.output_cost += (reasoning_tokens / 1_000_000) * pricing.output

        # Calculate total
        result.total_cost = result.input_cost + result.output_cost + result.reasoning_cost

        return result

    async def estimate_session_cost(self, session_id: str) -> CostEstimate:
        """
        Estimate total cost for all interactions in a session.

        Uses recorded interaction data and model card pricing to calculate
        cumulative session costs.

        Args:
            session_id: Session to estimate cost for.

        Returns:
            Aggregated CostEstimate for all interactions in the session.

        Raises:
            SessionNotFoundError: If session doesn't exist.

        Example:
            >>> cost = await llm.estimate_session_cost("session_123")
            >>> print(f"Session cost: {cost.format_cost()}")
            >>> print(f"Total tokens: {cost.prompt_tokens + cost.completion_tokens:,}")
        """
        # Get token stats first
        stats = await self.get_session_token_stats(session_id)

        # Get session to determine provider/model
        session = await self._session_manager.load_or_create_session(session_id)

        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Determine primary model used (model with most interactions)
        primary_model = "unknown"
        if stats.by_model:
            primary_model = max(stats.by_model.items(), key=lambda x: x[1].get("count", 0))[0]

        # Get provider from session metadata or use default
        provider = session.metadata.get("provider")
        if not provider:
            # Try to get from context info cache
            context_info = self._transient_last_interaction_info_cache.get(session_id)
            if context_info:
                provider = context_info.provider
            else:
                provider = self._default_provider

        # Calculate cost using the estimate_cost method
        return self.estimate_cost(
            provider_name=provider,
            model_name=primary_model,
            prompt_tokens=stats.total_prompt_tokens,
            completion_tokens=stats.total_completion_tokens,
            cached_tokens=stats.total_cached_tokens,
        )

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
    # Session Fork, Clone, and Message Operations (Phase 3)
    # ==============================================================================

    async def fork_session(
        self,
        session_id: str,
        *,
        new_name: Optional[str] = None,
        from_message_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        message_range: Optional[Tuple[int, int]] = None,
        include_context_items: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fork a session to create an independent copy with selected messages.

        Fork creates a new session with selected messages from the source.
        The forked session is completely independent - changes to either
        session do not affect the other.

        Fork modes (mutually exclusive, checked in order of priority):
        1. message_ids: Include only these specific messages
        2. message_range: Include messages in index range (0-based, inclusive)
        3. from_message_id: Include this message and all subsequent messages
        4. None of above: Full fork (all messages)

        Args:
            session_id: Source session to fork from
            new_name: Name for the new session. If None, generates name
                     like "source_name_fork_20260120_143052"
            from_message_id: Fork from this message onwards (inclusive)
            message_ids: List of specific message IDs to include
            message_range: Tuple of (start_index, end_index), 0-based inclusive
            include_context_items: Whether to copy workspace items to fork
            metadata: Additional metadata to add to forked session

        Returns:
            ID of the newly created session

        Example:
            # Full fork
            new_id = await llm.fork_session("session_123")

            # Fork from a specific point
            new_id = await llm.fork_session(
                "session_123",
                from_message_id="msg_005",
                new_name="divergent_exploration"
            )

            # Fork specific messages
            new_id = await llm.fork_session(
                "session_123",
                message_ids=["msg_001", "msg_003", "msg_005"],
                new_name="selected_messages"
            )

            # Fork a range
            new_id = await llm.fork_session(
                "session_123",
                message_range=(2, 7),  # Messages at indices 2-7
                new_name="conversation_segment"
            )

        Raises:
            SessionNotFoundError: If source session doesn't exist
            ValueError: If specified message IDs or range are invalid
        """
        from datetime import datetime as dt

        # Get source session
        source_session = await self._session_manager.get_session(session_id)
        if not source_session:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Determine which messages to include (sorted by timestamp)
        source_messages = sorted(source_session.messages, key=lambda m: m.timestamp)

        if message_ids:
            # Mode 1: Specific message IDs
            id_set = set(message_ids)
            messages_to_copy = [m for m in source_messages if m.id in id_set]

            if len(messages_to_copy) != len(message_ids):
                found_ids = {m.id for m in messages_to_copy}
                missing = id_set - found_ids
                raise ValueError(f"Message IDs not found: {missing}")

            fork_type = "specific_messages"

        elif message_range:
            # Mode 2: Index range (0-based, inclusive)
            start_idx, end_idx = message_range
            if start_idx < 0 or end_idx >= len(source_messages) or start_idx > end_idx:
                raise ValueError(
                    f"Invalid range ({start_idx}, {end_idx}) for session with "
                    f"{len(source_messages)} messages"
                )

            messages_to_copy = source_messages[start_idx : end_idx + 1]
            fork_type = "range"

        elif from_message_id:
            # Mode 3: From message onwards
            found_idx = None
            for i, msg in enumerate(source_messages):
                if msg.id == from_message_id:
                    found_idx = i
                    break

            if found_idx is None:
                raise ValueError(f"Message not found: {from_message_id}")

            messages_to_copy = source_messages[found_idx:]
            fork_type = "from_message"

        else:
            # Mode 4: Full fork
            messages_to_copy = source_messages
            fork_type = "full"

        # Generate name if not provided
        if not new_name:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            source_name = source_session.name or "session"
            new_name = f"{source_name}_fork_{timestamp}"

        # Create fork metadata
        fork_metadata: Dict[str, Any] = {
            "forked_from": session_id,
            "forked_at": dt.now(timezone.utc).isoformat(),
            "fork_type": fork_type,
            "source_message_count": len(source_messages),
            "forked_message_count": len(messages_to_copy),
        }

        if from_message_id:
            fork_metadata["fork_point_message_id"] = from_message_id
        if message_range:
            fork_metadata["fork_range"] = list(message_range)

        if metadata:
            fork_metadata.update(metadata)

        # Create new session
        new_session = await self.create_session(name=new_name)
        new_session.metadata = fork_metadata

        # Copy messages with new IDs
        for msg in messages_to_copy:
            new_message = Message(
                id=str(uuid.uuid4()),  # New ID for independence
                session_id=new_session.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,  # Preserve original timestamp
                tool_call_id=msg.tool_call_id,
                tokens=msg.tokens,
                metadata=msg.metadata.copy() if msg.metadata else {},
            )
            new_session.messages.append(new_message)

        # Copy context items if requested
        if include_context_items and source_session.context_items:
            for item in source_session.context_items:
                new_item = ContextItem(
                    id=str(uuid.uuid4()),  # New ID for independence
                    type=item.type,
                    source_id=item.source_id,
                    content=item.content,
                    tokens=item.tokens,
                    original_tokens=item.original_tokens,
                    is_truncated=item.is_truncated,
                    metadata=item.metadata.copy() if item.metadata else {},
                    timestamp=item.timestamp,
                )
                new_session.context_items.append(new_item)

        # Save the forked session
        await self._session_manager.save_session(new_session)

        logger.info(
            f"Forked session {session_id} to {new_session.id} "
            f"({len(messages_to_copy)} messages, type: {fork_type})"
        )

        return new_session.id

    async def clone_session(
        self,
        session_id: str,
        new_name: Optional[str] = None,
        *,
        include_messages: bool = True,
        include_context_items: bool = True,
    ) -> str:
        """
        Clone a session (full copy with new ID).

        Unlike fork, clone is always a complete copy. The main difference
        from fork is semantic - clone implies an exact duplicate, while
        fork implies divergence.

        Args:
            session_id: Source session to clone
            new_name: Name for cloned session (uses "source_clone" if None)
            include_messages: If False, creates empty session with same metadata
            include_context_items: Whether to copy workspace items

        Returns:
            ID of cloned session

        Example:
            # Full clone
            clone_id = await llm.clone_session("session_123")

            # Clone settings only (no messages)
            clone_id = await llm.clone_session(
                "session_123",
                include_messages=False
            )
        """
        from datetime import datetime as dt

        # Get source session
        source_session = await self._session_manager.get_session(session_id)
        if not source_session:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Generate name if not provided
        if not new_name:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            source_name = source_session.name or "session"
            new_name = f"{source_name}_clone_{timestamp}"

        # Create clone metadata
        clone_metadata: Dict[str, Any] = {
            "cloned_from": session_id,
            "cloned_at": dt.now(timezone.utc).isoformat(),
            "messages_included": include_messages,
            "context_items_included": include_context_items,
        }

        # Preserve source metadata with prefix
        if source_session.metadata:
            for key, value in source_session.metadata.items():
                if key not in ("cloned_from", "cloned_at"):
                    clone_metadata[f"source_{key}"] = value

        # Create new session
        new_session = await self.create_session(name=new_name)
        new_session.metadata = clone_metadata

        # Copy messages if requested
        if include_messages:
            for msg in source_session.messages:
                new_message = Message(
                    id=str(uuid.uuid4()),
                    session_id=new_session.id,
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    tool_call_id=msg.tool_call_id,
                    tokens=msg.tokens,
                    metadata=msg.metadata.copy() if msg.metadata else {},
                )
                new_session.messages.append(new_message)

        # Copy context items if requested
        if include_context_items and source_session.context_items:
            for item in source_session.context_items:
                new_item = ContextItem(
                    id=str(uuid.uuid4()),
                    type=item.type,
                    source_id=item.source_id,
                    content=item.content,
                    tokens=item.tokens,
                    original_tokens=item.original_tokens,
                    is_truncated=item.is_truncated,
                    metadata=item.metadata.copy() if item.metadata else {},
                    timestamp=item.timestamp,
                )
                new_session.context_items.append(new_item)

        # Save the cloned session
        await self._session_manager.save_session(new_session)

        logger.info(
            f"Cloned session {session_id} to {new_session.id} "
            f"(messages: {include_messages}, context: {include_context_items})"
        )

        return new_session.id

    async def delete_messages(
        self,
        session_id: str,
        message_ids: List[str],
    ) -> int:
        """
        Delete specific messages from a session.

        Args:
            session_id: Session containing the messages
            message_ids: List of message IDs to delete

        Returns:
            Number of messages actually deleted

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = await self._session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        id_set = set(message_ids)
        original_count = len(session.messages)

        session.messages = [m for m in session.messages if m.id not in id_set]

        deleted_count = original_count - len(session.messages)

        if deleted_count > 0:
            session.updated_at = datetime.now(timezone.utc)
            await self._session_manager.save_session(session)
            logger.info(f"Deleted {deleted_count} messages from session {session_id}")

        return deleted_count

    async def copy_messages_to_session(
        self,
        source_session_id: str,
        target_session_id: str,
        message_ids: List[str],
        *,
        preserve_timestamps: bool = True,
    ) -> List[str]:
        """
        Copy specific messages from one session to another.

        Args:
            source_session_id: Session to copy from
            target_session_id: Session to copy to
            message_ids: IDs of messages to copy
            preserve_timestamps: If True, keep original timestamps; otherwise use current time

        Returns:
            List of new message IDs created in target session

        Raises:
            SessionNotFoundError: If either session doesn't exist
            ValueError: If any message_ids are not found
        """
        from datetime import datetime as dt

        # Get source and target sessions
        source_session = await self._session_manager.get_session(source_session_id)
        if not source_session:
            raise SessionNotFoundError(f"Source session not found: {source_session_id}")

        target_session = await self._session_manager.get_session(target_session_id)
        if not target_session:
            raise SessionNotFoundError(f"Target session not found: {target_session_id}")

        # Find messages to copy
        id_set = set(message_ids)
        messages_to_copy = [m for m in source_session.messages if m.id in id_set]

        if len(messages_to_copy) != len(message_ids):
            found_ids = {m.id for m in messages_to_copy}
            missing = id_set - found_ids
            raise ValueError(f"Message IDs not found in source session: {missing}")

        # Copy messages with new IDs
        new_ids: List[str] = []
        for msg in messages_to_copy:
            new_id = str(uuid.uuid4())
            new_message = Message(
                id=new_id,
                session_id=target_session_id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp if preserve_timestamps else dt.now(timezone.utc),
                tool_call_id=msg.tool_call_id,
                tokens=msg.tokens,
                metadata={
                    **msg.metadata,
                    "copied_from_session": source_session_id,
                    "copied_from_message": msg.id,
                },
            )
            target_session.messages.append(new_message)
            new_ids.append(new_id)

        # Sort messages by timestamp
        target_session.messages.sort(key=lambda m: m.timestamp)
        target_session.updated_at = dt.now(timezone.utc)

        await self._session_manager.save_session(target_session)

        logger.info(
            f"Copied {len(new_ids)} messages from session {source_session_id} "
            f"to session {target_session_id}"
        )

        return new_ids

    async def get_messages_by_range(
        self,
        session_id: str,
        start_index: int,
        end_index: int,
    ) -> List[Message]:
        """
        Get messages from a session by index range.

        Messages are sorted by timestamp before indexing.

        Args:
            session_id: Session to get messages from
            start_index: Start index (0-based, inclusive)
            end_index: End index (0-based, inclusive)

        Returns:
            List of Message objects in the range

        Raises:
            SessionNotFoundError: If session doesn't exist
            ValueError: If range is invalid
        """
        session = await self._session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        messages = sorted(session.messages, key=lambda m: m.timestamp)
        msg_count = len(messages)

        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        if end_index < start_index:
            raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index})")
        if start_index >= msg_count:
            raise ValueError(
                f"start_index ({start_index}) out of range for session with {msg_count} messages"
            )

        # Clamp end_index to valid range
        end_index = min(end_index, msg_count - 1)

        return messages[start_index : end_index + 1]

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

    # ==========================================================================
    # Model Card Library Methods
    # ==========================================================================

    def get_model_card(
        self,
        provider_name: str,
        model_id: str,
    ) -> Optional["ModelCard"]:
        """
        Get model card for a specific model.

        Lookup order:
        1. User override (~/.config/llmcore/model_cards/)
        2. Built-in cards (llmcore/model_cards/default_cards/)
        3. None if not found

        Args:
            provider_name: Provider name (e.g., "openai", "anthropic")
            model_id: Model identifier or alias

        Returns:
            ModelCard if found, None otherwise

        Example:
            >>> card = llm.get_model_card("openai", "gpt-4o")
            >>> if card:
            ...     print(f"Context: {card.get_context_length()}")
            ...     print(f"Supports vision: {card.capabilities.vision}")
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        return registry.get(provider_name, model_id)

    def list_model_cards(
        self,
        provider_name: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
    ) -> List["ModelCardSummary"]:
        """
        List available model cards with optional filtering.

        Args:
            provider_name: Filter by provider (None for all)
            model_type: Filter by type ("chat", "embedding", etc.)
            tags: Filter by tags (any match)
            status: Filter by status ("active", "deprecated", etc.)

        Returns:
            List of ModelCardSummary objects

        Example:
            >>> # List all chat models
            >>> cards = llm.list_model_cards(model_type="chat")
            >>>
            >>> # List all OpenAI models
            >>> cards = llm.list_model_cards(provider_name="openai")
            >>>
            >>> # List models with specific tags
            >>> cards = llm.list_model_cards(tags=["reasoning", "vision"])
        """
        from .model_cards import ModelType, get_model_card_registry

        registry = get_model_card_registry()

        mt = None
        if model_type:
            try:
                mt = ModelType(model_type)
            except ValueError:
                logger.warning(f"Unknown model type: {model_type}")

        return registry.list_cards(
            provider=provider_name,
            model_type=mt,
            tags=tags,
            status=status,
        )

    def get_model_context_length(
        self,
        provider_name: str,
        model_id: str,
        *,
        fallback_to_provider: bool = True,
    ) -> int:
        """
        Get context length for a model.

        Lookup order:
        1. Model card (if available)
        2. Provider's get_max_context_length() (if fallback_to_provider=True)
        3. Default (4096)

        Args:
            provider_name: Provider name
            model_id: Model identifier
            fallback_to_provider: Query provider if not in card

        Returns:
            Maximum input context length in tokens

        Example:
            >>> context_len = llm.get_model_context_length("openai", "gpt-4o")
            >>> print(f"Max context: {context_len:,} tokens")
        """
        # Try model card first
        card = self.get_model_card(provider_name, model_id)
        if card:
            return card.get_context_length()

        # Fallback to provider
        if fallback_to_provider:
            try:
                provider = self._provider_manager.get_provider(provider_name)
                return provider.get_max_context_length(model_id)
            except Exception:
                pass

        return 4096  # Default

    def get_model_pricing(
        self,
        provider_name: str,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get pricing information for a model.

        Args:
            provider_name: Provider name
            model_id: Model identifier

        Returns:
            Dict with pricing info or None if not available

        Example:
            >>> pricing = llm.get_model_pricing("openai", "gpt-4o")
            >>> if pricing:
            ...     print(f"Input: ${pricing['input']}/1M tokens")
            ...     print(f"Output: ${pricing['output']}/1M tokens")
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        return registry.get_pricing(provider_name, model_id)

    def estimate_model_cost(
        self,
        provider_name: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> Optional[float]:
        """
        Estimate cost for a given token usage.

        Args:
            provider_name: Provider name
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (subset of input)

        Returns:
            Estimated cost in USD, or None if pricing not available

        Example:
            >>> cost = llm.estimate_model_cost(
            ...     "openai", "gpt-4o",
            ...     input_tokens=10000,
            ...     output_tokens=2000
            ... )
            >>> if cost:
            ...     print(f"Estimated cost: ${cost:.4f}")
        """
        card = self.get_model_card(provider_name, model_id)
        if card:
            return card.estimate_cost(input_tokens, output_tokens, cached_tokens)
        return None

    def save_model_card(
        self,
        card: "ModelCard",
        *,
        user_override: bool = True,
    ) -> pathlib.Path:
        """
        Save a model card to disk.

        Args:
            card: ModelCard to save
            user_override: If True, save to user directory; else to package

        Returns:
            Path where the card was saved

        Example:
            >>> from llmcore.model_cards import ModelCard, ModelContext
            >>> card = ModelCard(
            ...     model_id="my-custom-model",
            ...     provider="ollama",
            ...     model_type="chat",
            ...     context=ModelContext(max_input_tokens=8192),
            ... )
            >>> path = llm.save_model_card(card)
            >>> print(f"Saved to: {path}")
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        return registry.save_card(card, user_override=user_override)

    def get_model_card_providers(self) -> List[str]:
        """
        Get list of providers that have model cards registered.

        Returns:
            List of provider names

        Example:
            >>> providers = llm.get_model_card_providers()
            >>> print("Available providers:", providers)
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        return registry.get_providers()

    def get_model_card_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model card registry.

        Returns:
            Dict with registry statistics including:
            - total_cards: Total number of loaded cards
            - providers: Number of providers
            - aliases: Number of registered aliases
            - cards_by_provider: Breakdown by provider

        Example:
            >>> stats = llm.get_model_card_registry_stats()
            >>> print(f"Loaded {stats['total_cards']} model cards")
        """
        from .model_cards import get_model_card_registry

        registry = get_model_card_registry()
        return registry.stats()

    # ==========================================================================
    # Phase 1: Enhanced Model Management Methods
    # ==========================================================================

    async def get_provider_model_details(
        self,
        provider_name: Optional[str] = None,
        *,
        fetch_remote: bool = False,
        include_embeddings: bool = True,
    ) -> Dict[str, List[ModelDetails]]:
        """
        Get detailed model information for one or all providers.

        This is the primary method for model discovery. It aggregates information
        from multiple sources:
        1. Model card registry (authoritative metadata)
        2. Provider APIs (for Ollama local, OpenAI list, etc.)
        3. Configuration (user-specified models)

        Args:
            provider_name: Specific provider to query, or None for all loaded providers
            fetch_remote: If True, query provider APIs for latest data
                         - Ollama: Always queries local API (ignores this flag)
                         - OpenAI: Calls models.list() API
                         - Anthropic: Returns static list (no list API)
                         - Google: Calls list_models() API
            include_embeddings: Include embedding models in results

        Returns:
            Dict mapping provider names to lists of ModelDetails

        Example:
            >>> # Get all models for current provider
            >>> models = await llm.get_provider_model_details("ollama")
            >>> for model in models["ollama"]:
            ...     print(f"{model.id}: {model.context_length} tokens")
            >>>
            >>> # Get all models from all providers
            >>> all_models = await llm.get_provider_model_details()

        Raises:
            ConfigError: If provider_name is invalid
            ProviderError: If provider API call fails
        """
        from .model_cards import get_model_card_registry

        if provider_name and provider_name not in self.get_available_providers():
            raise ConfigError(f"Provider '{provider_name}' is not loaded")

        providers_to_query = [provider_name] if provider_name else self.get_available_providers()
        registry = get_model_card_registry()
        result: Dict[str, List[ModelDetails]] = {}

        for prov_name in providers_to_query:
            models: List[ModelDetails] = []

            # Special handling for Ollama - always query local API
            if prov_name.lower() == "ollama":
                try:
                    ollama_models = await self._get_ollama_models(prov_name)
                    models.extend(ollama_models)
                except Exception as e:
                    logger.warning(f"Failed to query Ollama API: {e}")
                    # Fall back to cards
                    models.extend(
                        self._get_models_from_cards(prov_name, registry, include_embeddings)
                    )
            else:
                # For other providers, use cards primarily
                models.extend(self._get_models_from_cards(prov_name, registry, include_embeddings))

                # If fetch_remote and provider supports listing, supplement
                if fetch_remote:
                    try:
                        remote_models = await self._fetch_remote_model_list(prov_name)
                        existing_ids = {m.id for m in models}
                        for rm in remote_models:
                            if rm.id not in existing_ids:
                                models.append(rm)
                    except Exception as e:
                        logger.debug(f"Remote model fetch not available for {prov_name}: {e}")

            # Also add models from config that might not be in cards
            try:
                config_models = self.get_models_for_provider(prov_name)
                existing_ids = {m.id for m in models}
                for model_id in config_models:
                    if model_id not in existing_ids:
                        # Create basic ModelDetails for config-only models
                        models.append(
                            ModelDetails(
                                id=model_id,
                                provider_name=prov_name,
                                context_length=self.get_model_context_length(prov_name, model_id),
                                metadata={"source": "config"},
                            )
                        )
            except Exception as e:
                logger.debug(f"Could not get config models for {prov_name}: {e}")

            result[prov_name] = models

        return result

    async def _get_ollama_models(self, provider_name: str) -> List[ModelDetails]:
        """
        Query local Ollama API for installed models.

        Args:
            provider_name: Should be "ollama"

        Returns:
            List of ModelDetails for locally installed models
        """
        try:
            provider = self._provider_manager.get_provider(provider_name)
        except Exception as e:
            raise ProviderError(provider_name, f"Failed to get provider: {e}")

        # The Ollama provider should have a client with list() method
        if not hasattr(provider, "_client") or provider._client is None:
            raise ProviderError(provider_name, "Ollama client not initialized")

        try:
            # Use the ollama library's list method
            # Response format varies by SDK version:
            # - Older: dict with "models" key
            # - Newer: ListResponse object with .models attribute
            list_response = await provider._client.list()

            # Handle both response formats
            if hasattr(list_response, "models"):
                # Newer SDK: ListResponse object
                models_data = list_response.models
            elif isinstance(list_response, dict):
                # Older SDK: dict format
                models_data = list_response.get("models", [])
            else:
                logger.warning(f"Unexpected Ollama list response type: {type(list_response)}")
                models_data = []
        except Exception as e:
            raise ProviderError(provider_name, f"Failed to list Ollama models: {e}")

        models: List[ModelDetails] = []
        for model_data in models_data:
            # Handle both Model object and dict formats
            if hasattr(model_data, "model"):
                # Newer SDK: Model object with .model attribute for name
                name = model_data.model
                # Details might be nested object or dict
                if hasattr(model_data, "details") and model_data.details:
                    details = model_data.details
                    family = (
                        getattr(details, "family", None)
                        if hasattr(details, "family")
                        else details.get("family")
                        if isinstance(details, dict)
                        else None
                    )
                    param_size = (
                        getattr(details, "parameter_size", None)
                        if hasattr(details, "parameter_size")
                        else details.get("parameter_size")
                        if isinstance(details, dict)
                        else None
                    )
                    quant_level = (
                        getattr(details, "quantization_level", None)
                        if hasattr(details, "quantization_level")
                        else details.get("quantization_level")
                        if isinstance(details, dict)
                        else None
                    )
                else:
                    family = param_size = quant_level = None
                size = getattr(model_data, "size", None)
                digest = getattr(model_data, "digest", None)
                modified_at = getattr(model_data, "modified_at", None)
                format_str = (
                    getattr(model_data.details, "format", None)
                    if hasattr(model_data, "details") and model_data.details
                    else None
                )
            else:
                # Older SDK: dict format
                name = model_data.get("name", "")
                details = model_data.get("details", {})
                family = details.get("family") if isinstance(details, dict) else None
                param_size = details.get("parameter_size") if isinstance(details, dict) else None
                quant_level = (
                    details.get("quantization_level") if isinstance(details, dict) else None
                )
                size = model_data.get("size")
                digest = model_data.get("digest")
                modified_at = model_data.get("modified_at")
                format_str = details.get("format") if isinstance(details, dict) else None

            if not name:
                continue

            # Try to get context length from model cards first
            context_length = self.get_model_context_length(provider_name, name)

            model = ModelDetails(
                id=name,
                provider_name=provider_name,
                display_name=name,
                family=family,
                parameter_count=param_size,
                quantization_level=quant_level,
                file_size_bytes=size,
                context_length=context_length,
                supports_streaming=True,
                supports_tools=True,  # Most modern Ollama models support tools
                metadata={
                    "digest": digest,
                    "modified_at": str(modified_at) if modified_at else None,
                    "format": format_str,
                    "source": "ollama_api",
                },
            )
            models.append(model)

        return models

    def _get_models_from_cards(
        self,
        provider_name: str,
        registry: Any,
        include_embeddings: bool = True,
    ) -> List[ModelDetails]:
        """
        Get models from the model card registry.

        Args:
            provider_name: Provider to get models for
            registry: ModelCardRegistry instance
            include_embeddings: Whether to include embedding models

        Returns:
            List of ModelDetails built from model cards
        """
        cards = registry.list_cards(provider=provider_name)
        models: List[ModelDetails] = []

        for card_summary in cards:
            # Filter out embeddings if requested
            if not include_embeddings and card_summary.model_type == "embedding":
                continue

            card = registry.get(provider_name, card_summary.model_id)
            if card:
                model = ModelDetails(
                    id=card.model_id,
                    provider_name=provider_name,
                    display_name=card.display_name or card.model_id,
                    context_length=card.context.max_input_tokens if card.context else 4096,
                    max_output_tokens=card.context.max_output_tokens if card.context else None,
                    supports_streaming=card.capabilities.streaming if card.capabilities else True,
                    supports_tools=card.capabilities.tool_use if card.capabilities else False,
                    supports_vision=card.capabilities.vision if card.capabilities else False,
                    supports_reasoning=card.capabilities.reasoning if card.capabilities else False,
                    family=card.architecture.family if card.architecture else None,
                    parameter_count=card.architecture.parameter_count
                    if card.architecture
                    else None,
                    model_type=card.model_type,
                    metadata={"source": "card"},
                )
                models.append(model)

        return models

    async def _fetch_remote_model_list(self, provider_name: str) -> List[ModelDetails]:
        """
        Fetch model list from provider's remote API.

        Currently a stub - can be extended for providers that support listing.

        Args:
            provider_name: Provider to fetch from

        Returns:
            List of ModelDetails from remote API
        """
        # This can be extended to support OpenAI's models.list(), etc.
        # For now, return empty list
        return []

    async def validate_model_for_provider(
        self,
        provider_name: str,
        model_name: str,
    ) -> ModelValidationResult:
        """
        Validate if a model is available for a provider.

        Performs multi-level validation:
        1. Check model card registry (with alias support)
        2. Check provider's known models (config + API)
        3. For Ollama, check local models
        4. Provide suggestions for similar models if not found

        Args:
            provider_name: Provider to validate against
            model_name: Model identifier to validate

        Returns:
            ModelValidationResult with:
                - is_valid: bool - Whether model is available
                - canonical_name: str - Correct model name (may differ in case)
                - suggestions: List[str] - Similar models if not found
                - error_message: Optional[str] - Human-readable error
                - model_details: Optional[ModelDetails] - If valid, full details

        Example:
            >>> result = await llm.validate_model_for_provider("openai", "gpt-5-turbo")
            >>> if not result.is_valid:
            ...     print(f"Model not found: {result.error_message}")
            ...     print(f"Did you mean: {', '.join(result.suggestions)}")
            >>> else:
            ...     print(f"Using model: {result.canonical_name}")
        """
        from .model_cards import get_model_card_registry

        if provider_name not in self.get_available_providers():
            return ModelValidationResult(
                is_valid=False,
                error_message=f"Provider '{provider_name}' is not loaded",
            )

        registry = get_model_card_registry()

        # Get available models for this provider
        try:
            models_dict = await self.get_provider_model_details(provider_name)
            available_models = models_dict.get(provider_name, [])
        except Exception as e:
            # Can't validate, allow with warning
            return ModelValidationResult(
                is_valid=True,
                canonical_name=model_name,
                error_message=f"Could not validate (allowing): {e}",
            )

        # If no models returned (dynamic provider), allow anything
        if not available_models:
            return ModelValidationResult(
                is_valid=True,
                canonical_name=model_name,
                error_message="Provider accepts any model name (dynamic)",
            )

        model_ids = [m.id for m in available_models]
        model_map = {m.id: m for m in available_models}

        # 1. Exact match
        if model_name in model_ids:
            return ModelValidationResult(
                is_valid=True,
                canonical_name=model_name,
                model_details=model_map[model_name],
            )

        # 2. Case-insensitive match
        for mid in model_ids:
            if model_name.lower() == mid.lower():
                return ModelValidationResult(
                    is_valid=True,
                    canonical_name=mid,  # Return correct casing
                    model_details=model_map[mid],
                    error_message=f"Note: Using '{mid}' (case corrected)",
                )

        # 3. Alias match (check model cards)
        card = registry.get(provider_name, model_name)
        if card:
            # Found via alias
            canonical = card.model_id
            # Try to find the corresponding ModelDetails
            details = model_map.get(canonical)
            return ModelValidationResult(
                is_valid=True,
                canonical_name=canonical,
                model_details=details,
                error_message=f"Resolved alias to '{canonical}'",
            )

        # 4. Not found - generate suggestions
        suggestions = self._generate_model_suggestions(model_name, model_ids)

        return ModelValidationResult(
            is_valid=False,
            suggestions=suggestions[:5],  # Top 5 suggestions
            error_message=f"Model '{model_name}' not found for provider '{provider_name}'",
        )

    def _generate_model_suggestions(
        self,
        query: str,
        available: List[str],
        max_suggestions: int = 5,
    ) -> List[str]:
        """
        Generate model name suggestions using fuzzy matching.

        Args:
            query: The model name that wasn't found
            available: List of available model IDs
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested model IDs, sorted by relevance
        """
        query_lower = query.lower()
        scored: List[Tuple[int, str]] = []

        for model_id in available:
            model_lower = model_id.lower()
            score = 0

            # Substring match
            if query_lower in model_lower:
                score += 50

            # Prefix match
            if model_lower.startswith(query_lower):
                score += 30

            # Word overlap
            query_words = set(query_lower.replace("-", " ").replace("_", " ").split())
            model_words = set(model_lower.replace("-", " ").replace("_", " ").split())
            overlap = len(query_words & model_words)
            score += overlap * 20

            # Levenshtein-like simple distance (for typos)
            if len(query_lower) > 3 and len(model_lower) > 3:
                common_prefix = 0
                for a, b in zip(query_lower, model_lower):
                    if a == b:
                        common_prefix += 1
                    else:
                        break
                score += common_prefix * 5

            if score > 0:
                scored.append((score, model_id))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        return [model_id for _, model_id in scored[:max_suggestions]]

    async def pull_model(
        self,
        provider_name: str,
        model_name: str,
        *,
        stream: bool = True,
        progress_callback: Optional[Callable[[PullProgress], None]] = None,
        insecure: bool = False,
    ) -> PullResult:
        """
        Download/pull a model from a provider's registry.

        Currently only supported for Ollama provider.

        Args:
            provider_name: Must be "ollama"
            model_name: Model to pull (e.g., "llama3.2:70b")
            stream: Whether to stream progress updates
            progress_callback: Called with progress updates if stream=True
            insecure: Allow insecure connections (Ollama)

        Returns:
            PullResult with success status, model name, error message, and duration

        Raises:
            ProviderError: If provider is not Ollama or pull fails

        Example:
            >>> def on_progress(p: PullProgress):
            ...     if p.percent_complete:
            ...         print(f"Progress: {p.percent_complete:.1f}%")
            >>>
            >>> result = await llm.pull_model(
            ...     "ollama",
            ...     "llama3.2:8b",
            ...     progress_callback=on_progress
            ... )
            >>> if result.success:
            ...     print(f"Pulled in {result.duration_seconds:.1f}s")
        """
        if provider_name.lower() != "ollama":
            raise ProviderError(
                provider_name,
                f"Model pulling is only supported for Ollama, not '{provider_name}'",
            )

        try:
            provider = self._provider_manager.get_provider(provider_name)
        except Exception as e:
            return PullResult(
                success=False,
                model_name=model_name,
                error_message=f"Failed to get provider: {e}",
            )

        if not hasattr(provider, "_client") or provider._client is None:
            return PullResult(
                success=False,
                model_name=model_name,
                error_message="Ollama client not initialized",
            )

        start_time = time.time()

        try:
            # Use the ollama library's pull method
            if stream and progress_callback:
                # Stream progress updates
                async for progress_data in await provider._client.pull(
                    model=model_name,
                    stream=True,
                    insecure=insecure,
                ):
                    progress = PullProgress(
                        status=progress_data.get("status", "downloading"),
                        digest=progress_data.get("digest"),
                        total_bytes=progress_data.get("total"),
                        completed_bytes=progress_data.get("completed"),
                        percent_complete=(
                            (progress_data.get("completed", 0) / progress_data.get("total", 1))
                            * 100
                            if progress_data.get("total")
                            else None
                        ),
                    )
                    progress_callback(progress)
            else:
                # Non-streaming pull
                await provider._client.pull(
                    model=model_name,
                    stream=False,
                    insecure=insecure,
                )

            duration = time.time() - start_time

            # Final success callback
            if progress_callback:
                progress_callback(
                    PullProgress(
                        status="success",
                        percent_complete=100.0,
                    )
                )

            return PullResult(
                success=True,
                model_name=model_name,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Provide helpful error messages
            if "not found" in error_msg.lower():
                error_msg = f"Model '{model_name}' not found in Ollama registry"
            elif "connection" in error_msg.lower():
                error_msg = f"Connection error: Is Ollama running? ({error_msg})"

            return PullResult(
                success=False,
                model_name=model_name,
                error_message=error_msg,
                duration_seconds=duration,
            )

    def update_config_add_model(
        self,
        provider_name: str,
        model_name: str,
        *,
        set_as_default: bool = False,
    ) -> bool:
        """
        Add a model to the provider's tracked models in config.

        Updates the in-memory config. Note: This does not persist to disk
        automatically. Use config persistence methods for that.

        Args:
            provider_name: Provider to add model to
            model_name: Model name to add
            set_as_default: If True, also set as default model

        Returns:
            True if added, False if already present

        Example:
            >>> # Add a newly pulled model to config
            >>> added = llm.update_config_add_model("ollama", "llama3.2:8b")
            >>> if added:
            ...     print("Model added to config")
        """
        if provider_name not in self.get_available_providers():
            logger.warning(f"Provider '{provider_name}' not in available providers")
            return False

        # Get current models from config
        current_models = []
        try:
            config_models = self.get_models_for_provider(provider_name)
            current_models = list(config_models) if config_models else []
        except Exception:
            pass

        # Check if already present
        if model_name in current_models:
            logger.debug(f"Model '{model_name}' already in config for {provider_name}")
            if set_as_default:
                # Still set as default even if already present
                self._set_default_model_for_provider(provider_name, model_name)
            return False

        # Add to list
        current_models.append(model_name)

        # Update config
        # Note: The actual config update mechanism depends on how config is managed
        # This is a best-effort implementation that works with common config patterns
        try:
            if hasattr(self.config, "set"):
                self.config.set(f"providers.{provider_name}.models", current_models)
            elif isinstance(self.config, dict):
                if "providers" not in self.config:
                    self.config["providers"] = {}
                if provider_name not in self.config["providers"]:
                    self.config["providers"][provider_name] = {}
                self.config["providers"][provider_name]["models"] = current_models
        except Exception as e:
            logger.warning(f"Could not update config: {e}")
            return False

        if set_as_default:
            self._set_default_model_for_provider(provider_name, model_name)

        logger.info(f"Added model '{model_name}' to {provider_name} config")
        return True

    def _set_default_model_for_provider(
        self,
        provider_name: str,
        model_name: str,
    ) -> None:
        """Set the default model for a provider in config."""
        try:
            if hasattr(self.config, "set"):
                self.config.set(f"providers.{provider_name}.default_model", model_name)
            elif isinstance(self.config, dict):
                if "providers" in self.config and provider_name in self.config["providers"]:
                    self.config["providers"][provider_name]["default_model"] = model_name
        except Exception as e:
            logger.warning(f"Could not set default model: {e}")

    # =========================================================================
    # EXPORT/IMPORT APIs (Phase 6)
    # =========================================================================

    def export_session(
        self,
        session_id: str,
        *,
        format: Literal["json", "yaml", "dict"] = "json",
        include_context_items: bool = True,
        include_metadata: bool = True,
        pretty: bool = True,
    ) -> Union[str, Dict[str, Any]]:
        """
        Export a session to a serialized format.

        Exports the session including messages, context items, and metadata
        in a format suitable for backup, transfer, or documentation.

        Args:
            session_id: ID of the session to export.
            format: Output format - "json", "yaml", or "dict" for raw dict.
            include_context_items: Whether to include workspace/context items.
            include_metadata: Whether to include session metadata.
            pretty: Pretty-print output (for json/yaml formats).

        Returns:
            For "json"/"yaml": Serialized string.
            For "dict": Raw dictionary.

        Raises:
            LLMCoreError: If session not found or serialization fails.
            ValueError: If format is invalid or YAML requested but not installed.

        Example:
            >>> # Export to JSON
            >>> json_data = llm.export_session("session_123")
            >>> with open("backup.json", "w") as f:
            ...     f.write(json_data)
            >>>
            >>> # Export to dict for programmatic use
            >>> data = llm.export_session("session_123", format="dict")
            >>> print(f"Messages: {len(data['messages'])}")
        """
        import json as json_module

        if format not in ("json", "yaml", "dict"):
            raise ValueError(f"Invalid format '{format}'. Must be 'json', 'yaml', or 'dict'.")

        # Get session from storage
        if self._storage_manager is None:
            raise LLMCoreError("Storage manager not initialized")

        session = self._storage_manager.get_session(session_id)
        if session is None:
            raise LLMCoreError(f"Session '{session_id}' not found")

        # Build export data structure
        export_data: Dict[str, Any] = {
            "llmcore_export_version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session": {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat().replace("+00:00", "Z"),
                "updated_at": session.updated_at.isoformat().replace("+00:00", "Z"),
            },
        }

        # Include metadata if requested
        if include_metadata and session.metadata:
            export_data["session"]["metadata"] = session.metadata

        # Export messages
        export_data["messages"] = []
        if session.messages:
            sorted_messages = sorted(session.messages, key=lambda m: m.timestamp)
            for msg in sorted_messages:
                msg_data = {
                    "id": msg.id,
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat().replace("+00:00", "Z"),
                }
                if msg.tool_call_id:
                    msg_data["tool_call_id"] = msg.tool_call_id
                if msg.tokens:
                    msg_data["tokens"] = msg.tokens
                if include_metadata and msg.metadata:
                    msg_data["metadata"] = msg.metadata
                export_data["messages"].append(msg_data)

        # Export context items if requested
        if include_context_items and session.context_items:
            export_data["context_items"] = []
            for item in session.context_items:
                item_data = {
                    "id": item.id,
                    "type": item.type.value if hasattr(item.type, "value") else str(item.type),
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat().replace("+00:00", "Z"),
                }
                if item.source_id:
                    item_data["source_id"] = item.source_id
                if item.tokens:
                    item_data["tokens"] = item.tokens
                if item.original_tokens:
                    item_data["original_tokens"] = item.original_tokens
                if item.is_truncated:
                    item_data["is_truncated"] = item.is_truncated
                if include_metadata and item.metadata:
                    item_data["metadata"] = item.metadata
                export_data["context_items"].append(item_data)

        # Return in requested format
        if format == "dict":
            return export_data

        if format == "json":
            indent = 2 if pretty else None
            return json_module.dumps(export_data, indent=indent, ensure_ascii=False, default=str)

        if format == "yaml":
            try:
                import yaml
            except ImportError:
                raise ValueError("YAML format requires PyYAML. Install with: pip install pyyaml")
            if pretty:
                return yaml.dump(
                    export_data,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            else:
                return yaml.dump(export_data, default_flow_style=True, allow_unicode=True)

        # Should not reach here
        raise ValueError(f"Invalid format: {format}")

    async def import_session(
        self,
        data: Union[str, Dict[str, Any]],
        *,
        format: Literal["json", "yaml", "dict"] = "json",
        new_name: Optional[str] = None,
        merge_into: Optional[str] = None,
    ) -> str:
        """
        Import a session from serialized data.

        Creates a new session from exported data, or merges into an existing
        session. Supports JSON, YAML, and dict formats.

        Args:
            data: Serialized session data (string for json/yaml, dict for dict).
            format: Input format - "json", "yaml", or "dict".
            new_name: Optional name for the imported session (uses original if None).
            merge_into: If provided, merge messages into this existing session ID
                        instead of creating a new session.

        Returns:
            ID of the imported or merged session.

        Raises:
            LLMCoreError: If import fails or merge target not found.
            ValueError: If format is invalid or data is malformed.

        Example:
            >>> # Import from JSON file
            >>> with open("backup.json", "r") as f:
            ...     json_data = f.read()
            >>> session_id = await llm.import_session(json_data)
            >>> print(f"Imported session: {session_id}")
            >>>
            >>> # Import and merge into existing session
            >>> session_id = await llm.import_session(json_data, merge_into="existing_123")
        """
        import json as json_module

        if format not in ("json", "yaml", "dict"):
            raise ValueError(f"Invalid format '{format}'. Must be 'json', 'yaml', or 'dict'.")

        # Parse input data
        if format == "dict":
            if not isinstance(data, dict):
                raise ValueError("For format='dict', data must be a dictionary")
            parsed_data = data
        elif format == "json":
            if not isinstance(data, str):
                raise ValueError("For format='json', data must be a string")
            try:
                parsed_data = json_module.loads(data)
            except json_module.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}")
        elif format == "yaml":
            if not isinstance(data, str):
                raise ValueError("For format='yaml', data must be a string")
            try:
                import yaml
            except ImportError:
                raise ValueError("YAML format requires PyYAML. Install with: pip install pyyaml")
            try:
                parsed_data = yaml.safe_load(data)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML data: {e}")
        else:
            raise ValueError(f"Invalid format: {format}")

        # Validate parsed data structure
        if not isinstance(parsed_data, dict):
            raise ValueError("Import data must be a dictionary/object")

        if "messages" not in parsed_data and "session" not in parsed_data:
            raise ValueError("Import data must contain 'messages' or 'session' key")

        # Extract session info
        session_info = parsed_data.get("session", {})
        original_name = session_info.get("name")
        messages_data = parsed_data.get("messages", [])
        context_items_data = parsed_data.get("context_items", [])

        if self._storage_manager is None:
            raise LLMCoreError("Storage manager not initialized")

        if merge_into:
            # Merge into existing session
            existing_session = self._storage_manager.get_session(merge_into)
            if existing_session is None:
                raise LLMCoreError(f"Target session '{merge_into}' not found for merge")

            # Add messages to existing session
            for msg_data in messages_data:
                role_str = msg_data.get("role", "user")
                try:
                    role = Role(role_str)
                except ValueError:
                    role = Role.USER

                message = Message(
                    id=str(uuid.uuid4()),  # Generate new ID to avoid conflicts
                    session_id=merge_into,
                    role=role,
                    content=msg_data.get("content", ""),
                    timestamp=datetime.fromisoformat(
                        msg_data.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                    tool_call_id=msg_data.get("tool_call_id"),
                    tokens=msg_data.get("tokens"),
                    metadata=msg_data.get("metadata", {}),
                )
                existing_session.messages.append(message)

            # Add context items
            for item_data in context_items_data:
                type_str = item_data.get("type", "user_text")
                try:
                    item_type = ContextItemType(type_str)
                except ValueError:
                    item_type = ContextItemType.USER_TEXT

                context_item = ContextItem(
                    id=str(uuid.uuid4()),  # Generate new ID
                    type=item_type,
                    source_id=item_data.get("source_id"),
                    content=item_data.get("content", ""),
                    tokens=item_data.get("tokens"),
                    original_tokens=item_data.get("original_tokens"),
                    is_truncated=item_data.get("is_truncated", False),
                    metadata=item_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        item_data.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                )
                existing_session.add_context_item(context_item)

            existing_session.updated_at = datetime.now(timezone.utc)
            self._storage_manager.save_session(existing_session)
            return merge_into

        else:
            # Create new session
            final_name = new_name or original_name
            new_session = await self.create_session(name=final_name)

            # Add messages
            for msg_data in messages_data:
                role_str = msg_data.get("role", "user")
                try:
                    role = Role(role_str)
                except ValueError:
                    role = Role.USER

                message = Message(
                    id=str(uuid.uuid4()),
                    session_id=new_session.id,
                    role=role,
                    content=msg_data.get("content", ""),
                    timestamp=datetime.fromisoformat(
                        msg_data.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                    tool_call_id=msg_data.get("tool_call_id"),
                    tokens=msg_data.get("tokens"),
                    metadata=msg_data.get("metadata", {}),
                )
                new_session.messages.append(message)

            # Add context items
            for item_data in context_items_data:
                type_str = item_data.get("type", "user_text")
                try:
                    item_type = ContextItemType(type_str)
                except ValueError:
                    item_type = ContextItemType.USER_TEXT

                context_item = ContextItem(
                    id=str(uuid.uuid4()),
                    type=item_type,
                    source_id=item_data.get("source_id"),
                    content=item_data.get("content", ""),
                    tokens=item_data.get("tokens"),
                    original_tokens=item_data.get("original_tokens"),
                    is_truncated=item_data.get("is_truncated", False),
                    metadata=item_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        item_data.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                )
                new_session.add_context_item(context_item)

            new_session.updated_at = datetime.now(timezone.utc)
            self._storage_manager.save_session(new_session)
            return new_session.id

    def export_context_items(
        self,
        session_id: str,
        item_ids: Optional[List[str]] = None,
        *,
        format: Literal["json", "yaml"] = "json",
        pretty: bool = True,
    ) -> str:
        """
        Export context items from a session.

        Exports workspace/context items in a portable format for backup
        or transfer to another session.

        Args:
            session_id: Session containing the items.
            item_ids: Specific item IDs to export (all items if None).
            format: Output format - "json" or "yaml".
            pretty: Pretty-print output.

        Returns:
            Serialized context items as a string.

        Raises:
            LLMCoreError: If session not found.
            ValueError: If format is invalid or YAML requested but not installed.

        Example:
            >>> # Export all context items
            >>> items_json = llm.export_context_items("session_123")
            >>> with open("context_backup.json", "w") as f:
            ...     f.write(items_json)
            >>>
            >>> # Export specific items
            >>> items_json = llm.export_context_items(
            ...     "session_123",
            ...     item_ids=["item_1", "item_2"]
            ... )
        """
        import json as json_module

        if format not in ("json", "yaml"):
            raise ValueError(f"Invalid format '{format}'. Must be 'json' or 'yaml'.")

        if self._storage_manager is None:
            raise LLMCoreError("Storage manager not initialized")

        session = self._storage_manager.get_session(session_id)
        if session is None:
            raise LLMCoreError(f"Session '{session_id}' not found")

        # Filter items if specific IDs requested
        items_to_export = session.context_items
        if item_ids is not None:
            item_ids_set = set(item_ids)
            items_to_export = [item for item in items_to_export if item.id in item_ids_set]

        # Build export structure
        export_data = {
            "llmcore_export_version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_session_id": session_id,
            "context_items": [],
        }

        for item in items_to_export:
            item_data = {
                "id": item.id,
                "type": item.type.value if hasattr(item.type, "value") else str(item.type),
                "content": item.content,
                "timestamp": item.timestamp.isoformat().replace("+00:00", "Z"),
            }
            if item.source_id:
                item_data["source_id"] = item.source_id
            if item.tokens:
                item_data["tokens"] = item.tokens
            if item.original_tokens:
                item_data["original_tokens"] = item.original_tokens
            if item.is_truncated:
                item_data["is_truncated"] = item.is_truncated
            if item.metadata:
                item_data["metadata"] = item.metadata
            export_data["context_items"].append(item_data)

        # Serialize to requested format
        if format == "json":
            indent = 2 if pretty else None
            return json_module.dumps(export_data, indent=indent, ensure_ascii=False, default=str)

        if format == "yaml":
            try:
                import yaml
            except ImportError:
                raise ValueError("YAML format requires PyYAML. Install with: pip install pyyaml")
            if pretty:
                return yaml.dump(
                    export_data,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            else:
                return yaml.dump(export_data, default_flow_style=True, allow_unicode=True)

        raise ValueError(f"Invalid format: {format}")
