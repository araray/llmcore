# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
"""

import asyncio
import importlib.resources
import json
import logging
import pathlib
import uuid # Ensure uuid is imported
from datetime import datetime, timezone # Ensure timezone is imported
from typing import (Any, AsyncGenerator, Dict, List, Optional, Tuple, Type,
                    Union)

import aiofiles # Keep for file operations in context item management

from .context.manager import ContextManager
from .embedding.manager import EmbeddingManager
from .exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
from .models import (ChatSession, ContextDocument, ContextItem,
                     ContextItemType, Message, Role, ContextPreparationDetails,
                     ContextPreset, ContextPresetItem) # Ensure all models are imported
from .providers.base import BaseProvider
from .providers.manager import ProviderManager
from .sessions.manager import SessionManager
from .storage.manager import StorageManager

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]
try:
    import tomllib # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib # Fallback for Python < 3.11
    except ImportError:
        tomllib = None # type: ignore [assignment]

# For type checking Ollama stream responses if the library is used
try:
    from ollama import ChatResponse as OllamaChatResponse
except ImportError:
    OllamaChatResponse = None # type: ignore [assignment]


logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, Retrieval Augmented
    Generation (RAG), context pool management (user-added context items),
    context preview, and management of saved context presets.

    It is initialized asynchronously using the `LLMCore.create()` classmethod.
    This class orchestrates various managers (provider, storage, session,
    context, embedding) to deliver its functionalities.
    """

    config: ConfyConfig
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _context_manager: ContextManager
    _embedding_manager: EmbeddingManager
    _transient_last_interaction_info_cache: Dict[str, ContextPreparationDetails]
    _transient_sessions_cache: Dict[str, ChatSession]

    def __init__(self):
        """
        Private constructor. Use the `LLMCore.create()` classmethod for
        asynchronous initialization.
        Initializes internal caches for transient session data.
        """
        self._transient_last_interaction_info_cache = {}
        self._transient_sessions_cache = {}
        # Initialization of managers is handled by the `create` method.
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

        This method handles the loading of configurations (using `confy`),
        and the initialization of all internal managers (ProviderManager,
        StorageManager, SessionManager, EmbeddingManager, ContextManager).

        Args:
            config_overrides: Dictionary of configuration values to override settings
                              from files or environment variables. Highest precedence.
                              Keys use dot-notation (e.g., "providers.openai.default_model").
            config_file_path: Path to a custom TOML or JSON configuration file.
            env_prefix: The prefix for environment variables that LLMCore will
                        consider for configuration overrides (default: "LLMCORE").
                        Set to `""` to consider all non-system env vars, or `None`
                        to disable environment variable loading.

        Returns:
            A fully initialized LLMCore instance.

        Raises:
            ConfigError: If essential configuration is missing, invalid, or if
                         dependencies like `tomli(b)` are not found.
            LLMCoreError: For general failures during manager initializations if not
                          more specifically caught (e.g., ProviderError, StorageError).
            ProviderError: If a configured provider fails to initialize.
            StorageError: If a configured storage backend fails to initialize.
            EmbeddingError: If the default embedding model fails to initialize.
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        # 1. Initialize self.config using confy.Config(...)
        try:
            from confy.loader import Config as ActualConfyConfig # Ensure it's the actual class
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required for loading default_config.toml.")

            default_config_dict = {}
            try:
                # Use importlib.resources for robust path handling within package
                if hasattr(importlib.resources, 'files'): # Python 3.9+
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # tomllib.load expects bytes
                        default_config_dict = tomllib.load(f)
                else: # Fallback for Python 3.8 (importlib.resources.read_text)
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8') # type: ignore
                    default_config_dict = tomllib.loads(default_config_content) # type: ignore
            except Exception as e:
                 raise ConfigError(f"Failed to load default configuration from llmcore.config: {e}")

            instance.config = ActualConfyConfig(
                defaults=default_config_dict,
                file_path=config_file_path,
                prefix=env_prefix,
                overrides_dict=config_overrides,
                mandatory=[] # No mandatory keys at this level, handled by components
            )
            logger.info("confy configuration loaded successfully.")
        except ImportError as e:
             # This typically means confy itself or a core dep is missing
             raise ConfigError(f"Configuration dependency (confy or toml parser) missing: {e}")
        except ConfigError: # Re-raise ConfigErrors from confy or default loading
            raise
        except Exception as e: # Catch any other unexpected error during config init
            raise ConfigError(f"LLMCore configuration initialization failed: {e}")

        # 2. Initialize Managers, passing self.config
        try:
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized.")
        except (ConfigError, ProviderError) as e: # Catch specific errors from ProviderManager
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True)
            raise
        except Exception as e: # Catch other unexpected errors
            raise LLMCoreError(f"ProviderManager initialization failed unexpectedly: {e}")

        try:
            instance._storage_manager = StorageManager(instance.config)
            await instance._storage_manager.initialize_storages() # This is async
            logger.info("StorageManager initialized.")
        except (ConfigError, StorageError) as e:
            logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True)
            raise
        except Exception as e:
            raise LLMCoreError(f"StorageManager initialization failed unexpectedly: {e}")

        try:
            # SessionManager depends on an initialized session storage backend
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized.")
        except StorageError as e: # If get_session_storage fails (e.g., not configured)
            raise LLMCoreError(f"SessionManager initialization failed due to storage issue: {e}")
        except Exception as e:
            raise LLMCoreError(f"SessionManager initialization failed unexpectedly: {e}")

        try:
            instance._embedding_manager = EmbeddingManager(instance.config)
            await instance._embedding_manager.initialize_embedding_model() # Initialize default model
            logger.info("EmbeddingManager initialized.")
        except (ConfigError, EmbeddingError) as e:
             logger.error(f"Failed to initialize EmbeddingManager: {e}", exc_info=True)
             raise
        except Exception as e:
            raise LLMCoreError(f"EmbeddingManager initialization failed unexpectedly: {e}")

        try:
            instance._context_manager = ContextManager(
                config=instance.config,
                provider_manager=instance._provider_manager,
                storage_manager=instance._storage_manager,
                embedding_manager=instance._embedding_manager
            )
            logger.info("ContextManager initialized.")
        except Exception as e:
            raise LLMCoreError(f"ContextManager initialization failed unexpectedly: {e}")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    # --- Core Chat Method ---

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
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None, # New parameter
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, user-added context, RAG, and explicitly staged items.

        Handles conversation history, optional Retrieval Augmented Generation (RAG)
        from a vector store, provider-specific token counting, context window
        management (including truncation), and optional MCP formatting.

        Args:
            message: The user's input message content.
            session_id: ID of the session to use/continue. If None, a temporary
                        (non-persistent) session is used for this call only.
            system_message: An optional system message for the LLM. Behavior when
                            provided for an existing session depends on context strategy
                            (e.g., it might replace or supplement existing system messages).
            provider_name: Override the default provider specified in the configuration.
            model_name: Override the default model for the selected provider.
            stream: If True, returns an async generator yielding response text chunks (str).
                    If False (default), returns the complete response content as a string.
            save_session: If True (default) and session_id is provided, the
                          conversation turn (user message + assistant response) is
                          saved to the persistent session storage. Ignored if
                          session_id is None.
            active_context_item_ids: A list of IDs for user-added context items (from the
                                     session's context pool, e.g., "workspace" items)
                                     to be considered for inclusion in the LLM's context.
            explicitly_staged_items: A list of `Message` or `ContextItem` objects
                                     that are explicitly chosen to be included in the
                                     context, potentially overriding or supplementing
                                     other context sources like history or RAG.
            enable_rag: If True, enables Retrieval Augmented Generation by searching
                        the vector store for context relevant to the message.
            rag_retrieval_k: Number of documents to retrieve for RAG. Overrides the
                             default from configuration if provided.
            rag_collection_name: Name of the vector store collection to use for RAG.
                                 Overrides the default from configuration if provided.
            rag_metadata_filter: Optional dictionary for metadata filtering in RAG.
            prompt_template_values: Optional dictionary of values for custom prompt template placeholders.
            **provider_kwargs: Additional keyword arguments passed directly to the
                               selected provider's chat completion API call
                               (e.g., temperature=0.7, max_tokens=100). Note: `max_tokens`
                               here usually refers to the *response* length limit, not
                               the context window limit.

        Returns:
            If stream=False: The full response content as a string.
            If stream=True: An asynchronous generator yielding response text chunks (str).

        Raises:
            ProviderError: If the LLM provider API call fails (e.g., authentication, rate limit).
            SessionNotFoundError: If a specified session_id is not found in storage (and not creating new).
            ConfigError: If configuration for the selected provider/model is invalid.
            VectorStorageError: If RAG retrieval from the vector store fails.
            EmbeddingError: If embedding generation fails for the RAG query.
            ContextLengthError: If the essential context (e.g., system prompt + user message)
                                exceeds the model's limit even after internal truncation attempts.
            LLMCoreError: For other library-specific errors.
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', save_session={save_session}, provider='{provider_actual_name}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}, "
            f"RAG_filter_keys={list(rag_metadata_filter.keys()) if rag_metadata_filter else 'None'}, " # Log keys for brevity
            f"prompt_values_count={len(prompt_template_values) if prompt_template_values else 0}, "
            f"active_user_items_count={len(active_context_item_ids) if active_context_item_ids else 0}, "
            f"explicitly_staged_items_count={len(explicitly_staged_items) if explicitly_staged_items else 0}"
        )

        try:
            # --- Session Handling (largely same as before) ---
            chat_session: ChatSession
            if session_id:
                if not save_session: # Transient session logic
                    if session_id in self._transient_sessions_cache:
                        chat_session = self._transient_sessions_cache[session_id]
                        logger.debug(f"Using transient session '{session_id}' from cache.")
                        # Update system message if provided for existing transient session
                        has_sys_msg = any(m.role == Role.SYSTEM for m in chat_session.messages)
                        current_sys_msg_content = next((m.content for m in chat_session.messages if m.role == Role.SYSTEM), None)
                        if system_message and (not has_sys_msg or current_sys_msg_content != system_message) :
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM] # Remove old
                            chat_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=session_id))
                            logger.debug(f"Updated/Added system message to cached transient session '{session_id}'.")
                    else: # New transient session
                        chat_session = ChatSession(id=session_id)
                        if system_message:
                            chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
                        self._transient_sessions_cache[session_id] = chat_session
                        logger.debug(f"Created new transient session '{session_id}' and cached it.")
                else: # Persistent session
                    chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
            else: # Stateless call (no session_id provided)
                chat_session = ChatSession(id=f"temp_stateless_{uuid.uuid4().hex[:8]}") # Temporary ID
                if system_message:
                    chat_session.add_message(message_content=system_message, role=Role.SYSTEM)

            # Add current user message to the session object (in memory)
            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}' (is_transient={not save_session and bool(session_id)})")

            # --- Prepare context using ContextManager ---
            # This now returns ContextPreparationDetails
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=chat_session, # Pass the potentially modified chat_session
                provider_name=provider_actual_name,
                model_name=target_model,
                active_context_item_ids=active_context_item_ids,
                explicitly_staged_items=explicitly_staged_items, # Pass new parameter
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter,
                prompt_template_values=prompt_template_values
            )
            # Extract necessary info from context_details
            context_payload = context_details.prepared_messages
            prepared_context_token_count = context_details.final_token_count
            # rag_documents_used_this_turn is available in context_details.rag_documents_used

            logger.info(f"Prepared context with {len(context_payload)} messages ({prepared_context_token_count} tokens) for model '{target_model}'.")

            # Cache the full context preparation details if a session_id is active
            if session_id:
                self._transient_last_interaction_info_cache[session_id] = context_details
                logger.debug(f"Stored full context preparation details for session '{session_id}'.")

            # --- Call Provider ---
            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload, model=target_model, stream=stream, **provider_kwargs
            )

            # --- Handle Response (Stream or Full) ---
            if stream:
                logger.debug(f"Processing stream response from provider '{provider_actual_name}'")
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # type: ignore
                # The wrapper now handles adding assistant message to session and saving
                return self._stream_response_wrapper(
                    provider_stream, active_provider, chat_session, (save_session and session_id is not None)
                )
            else: # Non-streaming
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(provider_actual_name, "Invalid response format (expected dict).")

                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)

                if full_response_content is None: # If content extraction failed
                     # Log warning, but still proceed to save an empty/error message if needed
                     logger.warning(f"Content extraction failed for non-streaming response from {provider_actual_name}. Response data: {str(response_data)[:200]}")
                     full_response_content = f"[LLMCore Error: Failed to extract content from provider response]"
                else:
                     logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                # Add assistant message to session object (in memory)
                assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")

                # Save session if persistent and requested
                if save_session and session_id: # Only save if session_id was provided and save_session is true
                    await self._session_manager.save_session(chat_session)

                return full_response_content

        except (SessionNotFoundError, StorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError) as e:
             logger.error(f"Chat failed: {e}") # Log with specific error type
             # Clean up transient session if it was created for this failed call
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
                 logger.debug(f"Cleared failed transient session '{session_id}' from cache.")
             raise # Re-raise the caught specific exception
        except Exception as e: # Catch any other unexpected errors
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             # Clean up transient session if it was created for this failed call
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
             raise LLMCoreError(f"Chat execution failed due to an unexpected error: {e}")


    async def preview_context_for_chat(
        self,
        current_user_query: str,
        *, # Force keyword-only arguments
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
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]: # Returns a dictionary representation of ContextPreparationDetails
        """
        Previews the context that would be prepared for an LLM chat call
        without making the actual API call to the LLM provider.

        This method simulates the context preparation process, including history
        selection, RAG (if enabled), inclusion of user-added ("workspace") items,
        explicitly staged items, and subsequent truncation strategies. It returns
        a detailed dictionary of what would be sent to the LLM.

        Args:
            current_user_query: The user's current message/query for which to
                                preview the context.
            session_id: Optional ID of an existing session to use as a base for
                        history and context items (from its pool/workspace).
                        If None, a temporary session context is simulated.
            system_message: An optional system message to consider for this preview.
                            If a `session_id` is provided and that session has a
                            system message, this argument might override or supplement it
                            based on the `ContextManager`'s internal logic for handling
                            system messages during context preparation. For a new or
                            stateless preview, this will be the primary system message.
            provider_name: Optional. Override the default provider for context calculation.
            model_name: Optional. Override the default model for the selected provider
                        for context calculation.
            active_context_item_ids: Optional list of IDs for user-added context items
                                     (from the session's pool/workspace) to be considered
                                     for inclusion.
            explicitly_staged_items: Optional list of `Message` or `ContextItem` objects
                                     to explicitly include in the previewed context. These
                                     are given priority based on `inclusion_priority` config.
            enable_rag: Boolean indicating whether to simulate RAG retrieval.
            rag_retrieval_k: Number of documents to retrieve for RAG simulation.
            rag_collection_name: Vector store collection name for RAG simulation.
            rag_metadata_filter: Optional metadata filter for RAG simulation.
            prompt_template_values: Optional dictionary of values for custom placeholders
                                    in the RAG prompt template.

        Returns:
            A dictionary representing the `ContextPreparationDetails` object, including:
            - 'prepared_messages': List of `Message` objects that would be sent.
            - 'final_token_count': Total token count for `prepared_messages`.
            - 'max_tokens_for_model': The target model's maximum context token limit.
            - 'rag_documents_used': List of `ContextDocument` if RAG was simulated.
            - 'rendered_rag_template_content': The RAG-rendered query, if applicable.
            - 'truncation_actions_taken': Dictionary detailing any truncations performed.

        Raises:
            LLMCoreError: If context preparation itself fails (e.g., due to config issues).
            ConfigError: If provider/model cannot be determined.
            Other exceptions from `ContextManager.prepare_context` if issues arise.
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}' for preview.")

        logger.info(f"Previewing context for query: '{current_user_query[:50]}...' "
                    f"(Provider: {provider_actual_name}, Model: {target_model}, Session: {session_id})")

        # Simulate session state for preview
        preview_session: ChatSession
        if session_id:
            # Try to load existing session. If it doesn't exist, SessionManager.load_or_create_session
            # will create a new (empty) ChatSession object with this ID.
            # The system_message provided to this preview method will be handled below.
            try:
                # Pass system_message_if_new=None because we'll handle system_message explicitly for preview
                loaded_session = await self._session_manager.load_or_create_session(session_id, system_message_if_new=None)
                preview_session = loaded_session

                # If a system_message is specifically provided for the preview,
                # ensure it's the one used, potentially overriding one from the loaded session for this preview.
                if system_message:
                    existing_sys_msg_idx = next((i for i, msg in enumerate(preview_session.messages) if msg.role == Role.SYSTEM), -1)
                    if existing_sys_msg_idx != -1:
                        # Modify existing system message for the preview
                        preview_session.messages[existing_sys_msg_idx].content = system_message
                        preview_session.messages[existing_sys_msg_idx].tokens = None # Force re-tokenization by ContextManager
                        logger.debug(f"Preview: Using provided system_message, overriding session's system message.")
                    else:
                        # Add as the first message if no system message existed
                        preview_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=preview_session.id))
                        logger.debug(f"Preview: Using provided system_message as no system message was in loaded session.")
                # If no system_message is provided for preview, any system message already in loaded_session will be used.

            except Exception as e_load:
                logger.warning(f"Could not load/create session '{session_id}' for preview: {e_load}. Simulating with a new temporary session structure.")
                preview_session = ChatSession(id=session_id or f"preview_temp_{uuid.uuid4().hex[:8]}")
                if system_message: # Add system message if provided for this new temp session
                    preview_session.add_message(message_content=system_message, role=Role.SYSTEM)
        else: # No session_id, create a fully temporary session for preview
            preview_session = ChatSession(id=f"preview_stateless_{uuid.uuid4().hex[:8]}")
            if system_message:
                preview_session.add_message(message_content=system_message, role=Role.SYSTEM)

        # Add the current_user_query as the latest user message to the preview_session's messages list.
        # This is crucial as prepare_context expects the query to be part of the session's messages.
        # This message object is temporary and only for this preview operation.
        preview_session.add_message(message_content=current_user_query, role=Role.USER)

        try:
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=preview_session, # Use the prepared preview_session
                provider_name=provider_actual_name,
                model_name=target_model,
                active_context_item_ids=active_context_item_ids,
                explicitly_staged_items=explicitly_staged_items,
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter,
                prompt_template_values=prompt_template_values
            )
            # Convert Pydantic model to dictionary for return, as per spec
            return context_details.model_dump(mode="json") # mode="json" handles datetimes etc.

        except Exception as e:
            logger.error(f"Error during context preview preparation: {e}", exc_info=True)
            # Ensure to wrap in LLMCoreError or a more specific one if applicable
            if isinstance(e, (ConfigError, ContextLengthError, EmbeddingError, ProviderError, VectorStorageError)):
                raise
            raise LLMCoreError(f"Context preview failed due to an internal error: {e}")


    async def _stream_response_wrapper(
        self, provider_stream: AsyncGenerator[Any, None], provider: BaseProvider,
        session: ChatSession, do_save_session: bool
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider's stream, yields text chunks, and handles session saving.
        (Ensured indentation fix is present)
        """
        full_response_content = ""; error_occurred = False; provider_name = provider.get_name()
        try:
            async for chunk in provider_stream:
                chunk_dict: Optional[Dict[str, Any]] = None
                if isinstance(chunk, dict): chunk_dict = chunk
                elif OllamaChatResponse and isinstance(chunk, OllamaChatResponse): # type: ignore
                    try: chunk_dict = chunk.model_dump()
                    except Exception as dump_err: logger.warning(f"Could not dump Ollama stream object: {dump_err}. Chunk: {chunk}"); continue
                else: logger.warning(f"Received non-dict/non-OllamaResponse chunk: {type(chunk)} - {chunk}"); continue
                if not chunk_dict: continue

                text_delta = self._extract_delta_content(chunk_dict, provider)
                error_message = chunk_dict.get('error')
                finish_reason = None
                if 'choices' in chunk_dict and chunk_dict['choices'] and isinstance(chunk_dict['choices'][0], dict):
                    finish_reason = chunk_dict['choices'][0].get('finish_reason')
                elif 'finish_reason' in chunk_dict:
                    finish_reason = chunk_dict.get('finish_reason')
                elif chunk_dict.get("type") == "message_delta" and chunk_dict.get("delta"):
                    finish_reason = chunk_dict["delta"].get("stop_reason")

                if text_delta: full_response_content += text_delta; yield text_delta
                if error_message: logger.error(f"Error during stream: {error_message}"); raise ProviderError(provider_name, error_message)
                if finish_reason and finish_reason not in ["stop", "length", None, "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_USE", "stop_token", "max_tokens", "NOT_SET", "OTHER"]:
                    logger.warning(f"Stream stopped due to reason: {finish_reason}")
        except Exception as e:
            error_occurred = True
            logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True)
            if isinstance(e, ProviderError): # Correctly indented re-raise
                raise
            raise ProviderError(provider_name, f"Stream processing error: {e}") # Correctly indented wrap
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if full_response_content or not error_occurred:
                assistant_msg = session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' (len: {len(full_response_content)}) added to session '{session.id}' (in-memory) after stream.")
            else:
                 logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response.")
            if do_save_session:
                try:
                    await self._session_manager.save_session(session)
                    logger.debug(f"Session '{session.id}' persisted after stream.")
                except Exception as save_e:
                     logger.error(f"Failed to save session {session.id} after stream: {save_e}", exc_info=True)

    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts text delta from stream chunk. (Implementation unchanged)"""
        provider_name = provider.get_name(); text_delta = ""
        try:
            if provider_name == "openai" or provider_name == "gemini":
                choices = chunk.get('choices', [])
                if choices and isinstance(choices[0], dict) and choices[0].get('delta'): text_delta = choices[0]['delta'].get('content', '') or ""
            elif provider_name == "anthropic":
                type_val = chunk.get("type")
                if type_val == "content_block_delta" and chunk.get('delta', {}).get('type') == "text_delta": text_delta = chunk.get('delta', {}).get('text', "") or ""
            elif provider_name == "ollama":
                message_chunk = chunk.get('message', {});
                if message_chunk and isinstance(message_chunk, dict): text_delta = message_chunk.get('content', '') or ""
                elif 'response' in chunk and isinstance(chunk['response'], str): text_delta = chunk.get('response', '') or ""
        except Exception as e: logger.warning(f"Error extracting delta from {provider_name} chunk: {e}. Chunk: {str(chunk)[:200]}"); text_delta = ""
        return text_delta or ""

    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> Optional[str]:
        """Extracts full response content. (Implementation unchanged)"""
        provider_name = provider.get_name(); full_response_content: Optional[str] = None
        try:
            if provider_name == "openai":
                choices = response_data.get('choices', [])
                if choices and isinstance(choices[0], dict) and choices[0].get('message'): full_response_content = choices[0]['message'].get('content')
            elif provider_name == "gemini":
                candidates = response_data.get('candidates', [])
                if candidates and isinstance(candidates[0], dict) and candidates[0].get('content'):
                    content_parts = candidates[0]['content'].get('parts', [])
                    if content_parts and isinstance(content_parts[0], dict): full_response_content = content_parts[0].get('text')
            elif provider_name == "anthropic":
                content_blocks = response_data.get('content', [])
                if content_blocks and isinstance(content_blocks, list) and content_blocks[0].get("type") == "text": full_response_content = content_blocks[0].get("text")
            elif provider_name == "ollama":
                message_part = response_data.get('message', {})
                if message_part and isinstance(message_part, dict): full_response_content = message_part.get('content')
                elif 'response' in response_data and isinstance(response_data['response'], str): full_response_content = response_data.get('response')
            if full_response_content is None and response_data: logger.warning(f"Could not extract content from {provider_name} response: {str(response_data)[:200]}"); return None
            return str(full_response_content) if full_response_content is not None else None
        except Exception as e: logger.error(f"Error extracting full content from {provider_name}: {e}. Response: {str(response_data)[:200]}", exc_info=True); return None

    # --- Session Management Methods ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieves a specific chat session object (including messages and context items).
        Checks transient cache first, then persistent storage.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The `ChatSession` object if found, otherwise None.

        Raises:
            StorageError: If interaction with the session storage fails.
        """
        logger.debug(f"LLMCore.get_session for ID: {session_id}")
        if session_id in self._transient_sessions_cache:
            logger.debug(f"Returning session '{session_id}' from transient cache.")
            return self._transient_sessions_cache[session_id]
        try:
            # load_or_create_session with system_message=None will load if exists.
            # If it doesn't exist in persistent storage, SessionManager's current
            # load_or_create_session will create a new *empty* one.
            # For get_session, we strictly want to return None if not found persistently.
            # This requires a change in SessionManager or a new method there.
            # For now, assuming SessionManager.load_or_create_session(..., system_message=None)
            # might return a new empty session if not found.
            # A more accurate get_session would be:
            # session = await self._session_manager._storage.get_session(session_id)
            # but that bypasses SessionManager logic.
            # Let's assume SessionManager.load_or_create_session is the entry point.
            # If it creates one, it's effectively "not found" for a pure "get" operation
            # if it wasn't in transient cache.
            # The original spec for SessionManager.load_or_create_session implies it creates.
            # The current implementation of LLMCore.get_session relies on this.
            # If SessionManager.load_or_create_session returns a *new* session when not found,
            # then this get_session will also return that new (empty) session.
            # This might not be the strict "get only if exists" behavior.
            # However, for now, maintaining consistency with previous logic.
            return await self._session_manager.load_or_create_session(session_id=session_id, system_message=None)
        except SessionNotFoundError: # This would be ideal if SessionManager raised it on "get miss"
            logger.warning(f"Session ID '{session_id}' not found by SessionManager.")
            return None
        except StorageError as e:
            logger.error(f"Storage error getting session '{session_id}': {e}")
            raise # Re-raise to be handled by caller

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists available persistent chat sessions (metadata only).

        Returns:
            A list of dictionaries, each containing basic info like 'id',
            'name', 'created_at', 'updated_at', 'message_count', 'context_item_count'.
            Does not include full messages or context item content.

        Raises:
            StorageError: If interaction with the session storage fails.
        """
        logger.debug("LLMCore.list_sessions called (persistent sessions only).")
        try:
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.list_sessions()
        except StorageError as e:
            logger.error(f"Storage error listing sessions: {e}")
            raise

    async def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session from both transient cache and persistent storage.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was found in either cache or persistent storage
            and deletion was attempted/successful from persistent storage,
            False otherwise.

        Raises:
            StorageError: If interaction with the session storage fails during deletion.
        """
        logger.debug(f"LLMCore.delete_session for ID: {session_id}")
        was_in_transient = self._transient_sessions_cache.pop(session_id, None) is not None
        if was_in_transient:
            logger.debug(f"Removed session '{session_id}' from transient cache during delete.")

        # Also clear any last interaction info for this session from its cache
        self._transient_last_interaction_info_cache.pop(session_id, None)

        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted_persistent = await session_storage.delete_session(session_id)
            if deleted_persistent:
                logger.info(f"Session '{session_id}' deleted from persistent storage.")
            elif not was_in_transient: # If not in transient and not deleted from persistent
                logger.warning(f"Session '{session_id}' not found for deletion in persistent storage.")
            return deleted_persistent or was_in_transient # True if deleted from either
        except StorageError as e:
            logger.error(f"Storage error deleting session '{session_id}': {e}")
            raise
        # Fallback if persistent deletion fails but was in transient (already popped)
        return was_in_transient


    # --- RAG / Vector Store Management Methods ---
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict]=None, doc_id: Optional[str]=None, collection_name: Optional[str]=None) -> str:
        """
        Adds a single document (text content) to the configured vector store.
        (Docstring and implementation unchanged)
        """
        logger.debug(f"Adding document to vector store (Collection: {collection_name or 'default'})...")
        try:
            embedding = await self._embedding_manager.generate_embedding(content)
            doc_metadata = metadata if metadata is not None else {}
            doc = ContextDocument(id=doc_id if doc_id else str(uuid.uuid4()), content=content, embedding=embedding, metadata=doc_metadata)
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents([doc], collection_name=collection_name)
            if not added_ids: raise VectorStorageError("Failed to add document, no ID returned.")
            logger.info(f"Document '{added_ids[0]}' added to vector store collection '{collection_name or vector_storage._default_collection_name}'.") # type: ignore
            return added_ids[0]
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to add document: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding document: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str]=None) -> List[str]:
        """
        Adds multiple documents to the configured vector store in a batch.
        (Docstring and implementation unchanged)
        """
        if not documents: return []
        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            contents = [doc_data["content"] for doc_data in documents if isinstance(doc_data.get("content"), str)]
            if len(contents) != len(documents): raise ValueError("All documents must have string 'content'.")
            embeddings = await self._embedding_manager.generate_embeddings(contents)
            if len(embeddings) != len(documents): raise EmbeddingError("Mismatch between texts and generated embeddings count.")

            vector_storage = self._storage_manager.get_vector_storage()
            resolved_collection_name = collection_name or vector_storage._default_collection_name # type: ignore

            docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata",{})) for d, emb in zip(documents, embeddings)]

            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)
            logger.info(f"Batch of {len(added_ids)} docs added/updated in collection '{resolved_collection_name}'.")
            return added_ids
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e: logger.error(f"Failed to add documents batch: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str]=None, filter_metadata: Optional[Dict]=None) -> List[ContextDocument]:
        """
        Performs a similarity search for relevant documents in the vector store.
        (Docstring and implementation unchanged)
        """
        if k <= 0: raise ValueError("'k' must be positive.")
        logger.debug(f"Searching vector store (k={k}, Collection: {collection_name or 'default'}) for query: '{query[:50]}...'")
        try:
            query_embedding = await self._embedding_manager.generate_embedding(query)
            vector_storage = self._storage_manager.get_vector_storage()
            results = await vector_storage.similarity_search(query_embedding=query_embedding, k=k, collection_name=collection_name, filter_metadata=filter_metadata)
            logger.info(f"Vector store search returned {len(results)} documents.")
            return results
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to search vector store: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error searching vector store: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def delete_documents_from_vector_store(self, document_ids: List[str], *, collection_name: Optional[str]=None) -> bool:
        """
        Deletes documents from the vector store by their IDs.
        (Docstring and implementation unchanged)
        """
        if not document_ids: logger.warning("delete_documents_from_vector_store called with empty ID list."); return True
        logger.debug(f"Deleting {len(document_ids)} documents from vector store (Collection: {collection_name or 'default'})...")
        try:
            vector_storage = self._storage_manager.get_vector_storage()
            success = await vector_storage.delete_documents(document_ids=document_ids, collection_name=collection_name)
            logger.info(f"Deletion attempt for {len(document_ids)} documents completed (Success: {success}).")
            return success
        except (VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to delete documents: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error deleting documents: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def list_rag_collections(self) -> List[str]:
        """
        Lists the names of all available RAG collections from the vector store.
        (Docstring and implementation unchanged)
        """
        logger.debug("LLMCore.list_rag_collections called.")
        try:
            return await self._storage_manager.list_vector_collection_names()
        except StorageError as e:
            logger.error(f"Storage error listing RAG collections: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing RAG collections: {e}", exc_info=True)
            raise LLMCoreError(f"Failed to list RAG collections: {e}")


    # --- User Context Item Management Methods (Session Pool / Workspace) ---
    async def add_text_context_item(self, session_id: str, content: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """
        Adds a text snippet as a context item to a session's context pool (workspace).
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required to add a context item.")
        # Determine if session is transient or needs loading/creation
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)

        item_id_actual = item_id or str(uuid.uuid4())
        item_metadata = metadata or {}
        if ignore_char_limit: # Store this preference in metadata
            item_metadata['ignore_char_limit'] = True

        item = ContextItem(
            id=item_id_actual, type=ContextItemType.USER_TEXT,
            source_id=item_id_actual, # For text items, source_id can be same as item_id or a label
            content=content, metadata=item_metadata
        )
        try: # Estimate tokens (can be refined by ContextManager if needed when preparing context)
            provider = self._provider_manager.get_default_provider()
            item.tokens = await provider.count_tokens(content, model=provider.default_model)
            item.original_tokens = item.tokens # Initially, tokens and original_tokens are same
        except Exception as e:
            logger.warning(f"Could not count tokens for user text item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None

        session.add_context_item(item)
        if not is_transient: # Only save to persistent storage if it's not a transient session
            await self._session_manager.save_session(session)
        logger.info(f"Added text context item '{item.id}' to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def add_file_context_item(self, session_id: str, file_path: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """
        Adds file content as a context item to a session's context pool (workspace).
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required.")
        path_obj = pathlib.Path(file_path).expanduser().resolve()
        if not path_obj.is_file(): raise FileNotFoundError(f"File not found: {path_obj}")

        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)

        try:
            async with aiofiles.open(path_obj, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
        except Exception as e:
            raise LLMCoreError(f"Failed to read file content from {path_obj}: {e}")

        file_metadata = metadata or {}
        file_metadata.setdefault("filename", path_obj.name)
        file_metadata.setdefault("original_path", str(path_obj))
        if ignore_char_limit:
            file_metadata['ignore_char_limit'] = True

        item_id_actual = item_id or str(uuid.uuid4())
        item = ContextItem(
            id=item_id_actual, type=ContextItemType.USER_FILE, source_id=str(path_obj),
            content=content, metadata=file_metadata
        )
        try:
            provider = self._provider_manager.get_default_provider()
            item.tokens = await provider.count_tokens(content, model=provider.default_model)
            item.original_tokens = item.tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for file item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None

        session.add_context_item(item)
        if not is_transient:
            await self._session_manager.save_session(session)
        logger.info(f"Added file context item '{item.id}' (from {path_obj}) to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def update_context_item(self, session_id: str, item_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        """
        Updates an existing context item in a session's context pool (workspace).
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)

        item_to_update = session.get_context_item(item_id)
        if not item_to_update:
            raise LLMCoreError(f"Context item '{item_id}' not found in session '{session_id}'.")

        updated = False
        if content is not None:
            item_to_update.content = content
            item_to_update.is_truncated = False # Reset truncation status on content update
            try: # Re-estimate tokens
                provider = self._provider_manager.get_default_provider()
                item_to_update.tokens = await provider.count_tokens(content, model=provider.default_model)
                item_to_update.original_tokens = item_to_update.tokens # Update original too
            except Exception:
                item_to_update.tokens = None; item_to_update.original_tokens = None
            updated = True

        if metadata is not None:
            # Preserve ignore_char_limit if not explicitly changed by new metadata
            existing_ignore_limit = item_to_update.metadata.get('ignore_char_limit')
            item_to_update.metadata.update(metadata)
            if 'ignore_char_limit' not in metadata and existing_ignore_limit is not None:
                item_to_update.metadata['ignore_char_limit'] = existing_ignore_limit
            updated = True

        if updated:
            item_to_update.timestamp = datetime.now(timezone.utc) # Update timestamp on any change
            if not is_transient:
                await self._session_manager.save_session(session)
        logger.info(f"Updated context item '{item_id}' in session '{session_id}' (is_transient={is_transient}).")
        return item_to_update

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        """
        Removes a context item from a session's context pool (workspace).
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        # Ensure session object is loaded/created to modify its context_items
        session_obj_to_modify: Optional[ChatSession] = self._transient_sessions_cache.get(session_id)
        if not session_obj_to_modify:
            session_obj_to_modify = await self._session_manager.load_or_create_session(session_id, system_message=None) # Load or create if not in transient

        if not session_obj_to_modify: # Should not happen if load_or_create_session works
             logger.warning(f"Session '{session_id}' not found for removing context item.")
             return False

        removed = session_obj_to_modify.remove_context_item(item_id)
        if removed:
            if not is_transient: # Only save if it's a persistent session
                await self._session_manager.save_session(session_obj_to_modify)
            logger.info(f"Removed context item '{item_id}' from session '{session_id}' (is_transient={is_transient}).")
        else:
            logger.warning(f"Context item '{item_id}' not found in session '{session_id}' for removal.")
        return removed

    async def get_session_context_items(self, session_id: str) -> List[ContextItem]:
        """
        Retrieves all context items from a given session's context pool (workspace).
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id) # Uses the updated get_session logic
        if not session:
            raise SessionNotFoundError(session_id=session_id, message="Session not found when trying to list its context items.")
        return session.context_items

    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        """
        Retrieves a specific context item from a session's context pool (workspace) by its ID.
        (Docstring and implementation unchanged)
        """
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session:
            return None
        return session.get_context_item(item_id)

    async def get_last_interaction_context_info(self, session_id: str) -> Optional[ContextPreparationDetails]:
        """
        Gets the cached `ContextPreparationDetails` object from the last context
        preparation for a given session. This includes the messages sent, token counts,
        RAG documents used, and truncation details.

        Args:
            session_id: The ID of the session.

        Returns:
            The `ContextPreparationDetails` object if available, otherwise None.
        """
        if not session_id:
            logger.warning("get_last_interaction_context_info called without session_id.")
            return None
        cached_info = self._transient_last_interaction_info_cache.get(session_id)
        if cached_info:
            logger.debug(f"Retrieved last interaction context details from transient cache for session '{session_id}'.")
        else:
            logger.debug(f"No interaction context details found in transient cache for session '{session_id}'.")
        return cached_info

    async def get_last_used_rag_documents(self, session_id: str) -> Optional[List[ContextDocument]]:
        """
        Gets the list of RAG documents that were used in the last chat turn
        for a given session, by accessing the cached context preparation details.

        Args:
            session_id: The ID of the session.

        Returns:
            A list of `ContextDocument` objects if RAG was used and info is cached,
            otherwise None or an empty list.
        """
        context_details = await self.get_last_interaction_context_info(session_id)
        if context_details:
            return context_details.rag_documents_used
        return None

    async def pin_rag_document_as_context_item(
        self,
        session_id: str,
        original_rag_doc_id: str,
        custom_item_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ContextItem:
        """
        Takes a RAG document that was used in the last interaction for a session,
        and "pins" its content as a new `ContextItem` of type `RAG_SNIPPET`
        into that session's context pool (workspace).

        Args:
            session_id: The ID of the session where the RAG document was used.
            original_rag_doc_id: The ID of the `ContextDocument` (from RAG results) to pin.
            custom_item_id: Optional custom ID for the new `ContextItem` being created.
            custom_metadata: Optional additional metadata for the new `ContextItem`.

        Returns:
            The newly created `ContextItem` representing the pinned RAG snippet.

        Raises:
            LLMCoreError: If no RAG info is cached for the session, or the specified
                          `original_rag_doc_id` is not found among the last used RAG documents.
            ValueError: If `session_id` or `original_rag_doc_id` is not provided.
        """
        if not session_id: raise ValueError("session_id is required for pinning RAG document.")
        if not original_rag_doc_id: raise ValueError("original_rag_doc_id is required for pinning.")

        # Determine if session is transient or needs loading/creation for modification
        is_transient_session = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient_session else await self._session_manager.load_or_create_session(session_id)

        context_details = await self.get_last_interaction_context_info(session_id) # Use updated method
        last_rag_docs = context_details.rag_documents_used if context_details else None

        if not last_rag_docs:
            raise LLMCoreError(f"No RAG documents from the last turn found in cache for session '{session_id}'. Cannot pin.")

        doc_to_pin: Optional[ContextDocument] = next((doc for doc in last_rag_docs if doc.id == original_rag_doc_id), None)
        if not doc_to_pin:
            raise LLMCoreError(f"RAG document with ID '{original_rag_doc_id}' not found among last used RAG documents for session '{session_id}'.")

        new_item_id = custom_item_id or str(uuid.uuid4())
        new_item_metadata = {
            "pinned_from_rag": True,
            "original_rag_doc_id": doc_to_pin.id,
            "original_rag_doc_metadata": doc_to_pin.metadata, # type: ignore # Pydantic handles dict
            "pinned_timestamp": datetime.now(timezone.utc).isoformat()
        }
        if custom_metadata:
            new_item_metadata.update(custom_metadata)

        # Create the new ContextItem representing the pinned RAG snippet
        pinned_item = ContextItem(
            id=new_item_id,
            type=ContextItemType.RAG_SNIPPET, # Mark as a pinned RAG snippet
            source_id=doc_to_pin.id, # Link back to the original RAG document ID
            content=doc_to_pin.content,
            metadata=new_item_metadata,
            timestamp=datetime.now(timezone.utc) # New timestamp for this item
        )
        try: # Estimate tokens for the pinned item
            provider = self._provider_manager.get_default_provider()
            pinned_item.tokens = await provider.count_tokens(pinned_item.content, model=provider.default_model)
            pinned_item.original_tokens = pinned_item.tokens
        except Exception as e:
            logger.warning(f"Could not count tokens for pinned RAG snippet '{pinned_item.id}': {e}. Tokens set to None."); pinned_item.tokens = None; pinned_item.original_tokens = None

        session.add_context_item(pinned_item) # Add to the session's context pool
        if not is_transient_session: # Save if it's a persistent session
            await self._session_manager.save_session(session)

        logger.info(f"Pinned RAG document '{original_rag_doc_id}' as new context item '{pinned_item.id}' in session '{session_id}' (is_transient={is_transient_session}).")
        return pinned_item

    # --- Provider Info Methods ---
    def get_available_providers(self) -> List[str]:
        """
        Lists the names of all successfully loaded and configured LLM provider instances.
        (Docstring and implementation unchanged)
        """
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """
        Lists available models for a specific configured provider instance.
        Note: This might return a cached/static list or perform an API call
              depending on the provider's implementation in `BaseProvider`.
        (Docstring and implementation unchanged)

        Args:
            provider_name: The name of the provider instance (as configured).

        Returns:
            A list of model name strings available for the provider.

        Raises:
            ConfigError: If the specified provider instance is not configured or found.
            ProviderError: If fetching models from the provider API fails.
        """
        logger.debug(f"LLMCore.get_models_for_provider for: {provider_name}")
        try:
            provider = self._provider_manager.get_provider(provider_name)
            return provider.get_available_models()
        except (ConfigError, ProviderError) as e: # Catch specific errors from get_provider or get_available_models
            logger.error(f"Error getting models for provider '{provider_name}': {e}")
            raise
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error getting models for provider '{provider_name}': {e}", exc_info=True)
            raise ProviderError(provider_name, f"Failed to retrieve models due to an unexpected error: {e}")

    # --- New Context Preset Management API Methods ---
    async def save_context_preset(self, preset_name: str, items: List[ContextPresetItem],
                                  description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextPreset:
        """
        Saves or updates a context preset.

        A context preset is a named collection of `ContextPresetItem` objects that can be
        persistently stored and later loaded to quickly populate a context.

        Args:
            preset_name: The unique name for the preset. This name will be used as its
                         identifier for loading, deleting, or renaming.
            items: A list of `ContextPresetItem` objects to include in the preset.
                   Each item defines its type (e.g., text content, file reference)
                   and necessary data.
            description: Optional user-provided description for the preset.
            metadata: Optional dictionary for storing additional, unstructured
                      metadata associated with the preset itself.

        Returns:
            The saved `ContextPreset` object, including any modifications made during
            saving (like updated timestamps).

        Raises:
            StorageError: If saving to the configured session storage backend fails.
            ValueError: If `preset_name` is invalid (e.g., empty or contains characters
                        unsuitable for storage keys/filenames, typically validated by
                        the `ContextPreset` Pydantic model).
        """
        logger.info(f"Saving context preset '{preset_name}' with {len(items)} items.")
        try:
            # Create the ContextPreset object. Pydantic model validation for 'name' will occur here.
            preset = ContextPreset(
                name=preset_name,
                items=items,
                description=description,
                metadata=metadata or {},
                updated_at=datetime.now(timezone.utc) # Ensure updated_at is current
                # created_at will be set by default_factory if new, or preserved by storage if updating
            )

            session_storage = self._storage_manager.get_session_storage()
            await session_storage.save_context_preset(preset)
            logger.info(f"Context preset '{preset.name}' saved successfully.")
            return preset
        except ValueError as ve: # Catch Pydantic validation error for name or other fields
            logger.error(f"Invalid data for context preset '{preset_name}': {ve}")
            raise # Re-raise ValueError to indicate bad input
        except StorageError as se:
            logger.error(f"Failed to save context preset '{preset_name}' to storage: {se}", exc_info=True)
            raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error saving context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error occurred while saving preset '{preset_name}': {e}")

    async def load_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """
        Loads a context preset by its unique name from storage.

        Args:
            preset_name: The name of the context preset to load.

        Returns:
            The `ContextPreset` object if found, otherwise None.

        Raises:
            StorageError: If loading from the configured session storage backend fails.
        """
        logger.info(f"Loading context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name)
            if preset:
                logger.info(f"Context preset '{preset_name}' loaded successfully with {len(preset.items)} items.")
            else:
                logger.info(f"Context preset '{preset_name}' not found in storage.")
            return preset
        except StorageError as se:
            logger.error(f"Failed to load context preset '{preset_name}' from storage: {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error occurred while loading preset '{preset_name}': {e}")

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """
        Lists metadata of all available context presets.

        Returns:
            A list of dictionaries, where each dictionary contains metadata for a preset
            (e.g., 'name', 'description', 'item_count', 'updated_at').
            The exact fields depend on the storage backend's implementation of
            `BaseSessionStorage.list_context_presets`.

        Raises:
            StorageError: If listing presets from the backend fails.
        """
        logger.info("Listing all available context presets.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            presets_meta = await session_storage.list_context_presets()
            logger.info(f"Found {len(presets_meta)} context presets.")
            return presets_meta
        except StorageError as se:
            logger.error(f"Failed to list context presets from storage: {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing context presets: {e}", exc_info=True)
            raise StorageError(f"Unexpected error occurred while listing presets: {e}")

    async def delete_context_preset(self, preset_name: str) -> bool:
        """
        Deletes a context preset by its name from storage.

        Args:
            preset_name: The name of the preset to delete.

        Returns:
            True if the preset was found and successfully deleted, False otherwise.

        Raises:
            StorageError: If deletion from the configured session storage backend fails.
        """
        logger.info(f"Deleting context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted = await session_storage.delete_context_preset(preset_name)
            if deleted:
                logger.info(f"Context preset '{preset_name}' deleted successfully from storage.")
            else:
                logger.warning(f"Context preset '{preset_name}' not found in storage for deletion.")
            return deleted
        except StorageError as se:
            logger.error(f"Failed to delete context preset '{preset_name}' from storage: {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error occurred while deleting preset '{preset_name}': {e}")

    async def update_context_preset_description(self, preset_name: str, new_description: str) -> bool:
        """
        Updates the description of an existing context preset.
        The `updated_at` timestamp of the preset will also be refreshed.

        Args:
            preset_name: The name of the context preset to update.
            new_description: The new description string for the preset.

        Returns:
            True if the preset was found and its description updated successfully,
            False if the preset was not found.

        Raises:
            StorageError: If updating the preset in the backend fails.
        """
        logger.info(f"Updating description for context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name) # Load existing
            if not preset:
                logger.warning(f"Context preset '{preset_name}' not found for updating description.")
                return False

            preset.description = new_description
            preset.updated_at = datetime.now(timezone.utc) # Explicitly update timestamp

            await session_storage.save_context_preset(preset) # Save the modified preset
            logger.info(f"Description for context preset '{preset_name}' updated successfully.")
            return True
        except StorageError as se:
            logger.error(f"Failed to update description for context preset '{preset_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating description for preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error updating preset description for '{preset_name}': {e}")

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """
        Renames an existing context preset.

        Args:
            old_name: The current name of the preset.
            new_name: The new name for the preset. The new name must be valid
                      (e.g., not empty, no forbidden characters for filenames/keys).

        Returns:
            True if the preset was successfully renamed, False if the `old_name`
            was not found or if the `new_name` already exists.

        Raises:
            ValueError: If `new_name` is invalid (e.g., empty or contains
                        characters unsuitable for storage keys/filenames). This validation
                        is typically handled by the storage backend or Pydantic model.
            StorageError: For other storage-related issues during the rename operation.
        """
        logger.info(f"Attempting to rename context preset from '{old_name}' to '{new_name}'.")
        if not new_name or not new_name.strip(): # Basic check, Pydantic model does more
            raise ValueError("New preset name cannot be empty or just whitespace.")
        # More specific validation for new_name (e.g. filesystem chars) should be handled by
        # the ContextPreset Pydantic model or the storage backend's rename_context_preset method.

        try:
            session_storage = self._storage_manager.get_session_storage()
            renamed_successfully = await session_storage.rename_context_preset(old_name, new_name)
            if renamed_successfully:
                logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}'.")
            else:
                # The storage backend's rename method should log specifics if old_name not found or new_name exists
                logger.warning(f"Failed to rename context preset '{old_name}' to '{new_name}' (check storage logs for details like 'not found' or 'new name exists').")
            return renamed_successfully
        except ValueError: # Re-raise ValueError (e.g., from Pydantic validation in storage layer)
            raise
        except StorageError as se:
            logger.error(f"Storage error occurred while renaming context preset '{old_name}' to '{new_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error occurred during preset rename: {e}")


    # --- Utility / Cleanup ---
    async def close(self):
        """
        Closes connections for storage backends, provider clients, and other
        managed resources. Should be called when the application using LLMCore
        is shutting down to ensure graceful resource release.
        (Docstring and implementation unchanged)
        """
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = [
            self._provider_manager.close_providers(),
            self._storage_manager.close_storages(),
            self._embedding_manager.close(), # Assuming EmbeddingManager also has a close method
        ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception):
                 logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)

        # Clear transient caches
        self._transient_last_interaction_info_cache.clear()
        self._transient_sessions_cache.clear()
        logger.info("LLMCore resources cleanup complete and transient caches cleared.")

    async def __aenter__(self):
        """Enter the runtime context related to this object for 'async with' usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object, ensuring cleanup."""
        await self.close()
