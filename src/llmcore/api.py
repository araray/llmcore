# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
"""

import asyncio
import logging
import importlib.resources
import pathlib
import json # For parsing stream chunks if needed
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Type

# Models and Exceptions
from .models import ChatSession, ContextDocument, Message, Role
from .exceptions import (
    LLMCoreError, ProviderError, SessionNotFoundError, ConfigError,
    StorageError, SessionStorageError, VectorStorageError,
    EmbeddingError, ContextLengthError, MCPError
)
# Storage
from .storage.manager import StorageManager # Use StorageManager
# Sessions
from .sessions.manager import SessionManager
# Context
from .context.manager import ContextManager
# Providers
from .providers.manager import ProviderManager # Use ProviderManager
from .providers.base import BaseProvider # Keep BaseProvider for type hinting

# confy and tomli
try:
    from confy.loader import Config as ConfyConfig
    import tomli
except ImportError as e:
    ConfyConfig = Dict[str, Any] # type: ignore
    logging.getLogger(__name__).warning(f"confy or tomli not imported: {e}")

logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, and
    Retrieval Augmented Generation (RAG) using configurable providers,
    storage backends, and embedding models.

    Instantiate using 'instance = await LLMCore.create(...)'.
    """

    config: ConfyConfig
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _context_manager: ContextManager
    # _embedding_manager: EmbeddingManager # Placeholder for Phase 2

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
        """
        Asynchronously creates and initializes an LLMCore instance.

        Loads configuration, initializes managers (Providers, Storage, Sessions, Context),
        and prepares the instance for use.

        Args:
            config_overrides: Dictionary of configuration overrides (highest precedence).
            config_file_path: Path to a custom TOML or JSON configuration file.
            env_prefix: Prefix for environment variable overrides.

        Returns:
            An initialized LLMCore instance.

        Raises:
            ConfigError: If essential configuration is missing or invalid.
            ImportError: If required dependencies are not installed.
            StorageError: If storage backend initialization fails.
            ProviderError: If provider initialization fails.
            LLMCoreError: For other initialization errors.
        """
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
                    # Fallback for older Python versions
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

        # --- Step 2: Initialize Provider Manager ---
        try:
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized successfully.")
        except (ConfigError, ProviderError) as e:
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True)
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Unexpected error initializing ProviderManager: {e}", exc_info=True)
            raise LLMCoreError(f"ProviderManager initialization failed: {e}")


        # --- Step 3: Initialize Storage Manager ---
        try:
            instance._storage_manager = StorageManager(instance.config)
            # Initialize the actual storage backends asynchronously
            await instance._storage_manager.initialize_storages()
            logger.info("StorageManager initialized successfully.")
        except (ConfigError, StorageError) as e:
            logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True)
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Unexpected error initializing StorageManager: {e}", exc_info=True)
            raise LLMCoreError(f"StorageManager initialization failed: {e}")

        # --- Step 4: Initialize Session Manager ---
        try:
            # Pass the initialized session storage from the manager
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized successfully.")
        except StorageError as e: # Catch if session storage wasn't configured/initialized
             logger.error(f"Cannot initialize SessionManager: {e}")
             raise LLMCoreError(f"SessionManager initialization failed due to storage issue: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing SessionManager: {e}", exc_info=True)
            raise LLMCoreError(f"SessionManager initialization failed: {e}")

        # --- Step 5: Initialize Embedding Manager (Placeholder) ---
        # instance._embedding_manager = EmbeddingManager(instance.config)
        logger.warning("EmbeddingManager initialization is pending (Phase 2).")

        # --- Step 6: Initialize Context Manager ---
        try:
            # Pass the ProviderManager to ContextManager
            instance._context_manager = ContextManager(instance.config, instance._provider_manager)
            logger.info("ContextManager initialized successfully.")
        except Exception as e:
            logger.error(f"Unexpected error initializing ContextManager: {e}", exc_info=True)
            raise LLMCoreError(f"ContextManager initialization failed: {e}")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    # --- Core Chat Method ---
    async def chat(
        self,
        message: str,
        *, # Force subsequent arguments to be keyword-only
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None, # Now used to select provider via manager
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        # RAG parameters
        enable_rag: bool = False, # Placeholder
        rag_retrieval_k: Optional[int] = None, # Placeholder
        rag_collection_name: Optional[str] = None, # Placeholder
        # Provider specific arguments (e.g., temperature, max_tokens for response)
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the configured LLM, managing context and session.

        Handles conversation history, optional Retrieval Augmented Generation (RAG)
        from a vector store, provider-specific token counting, context window
        management (including truncation), and optional MCP formatting.

        Args:
            message: The user's message content.
            session_id: ID of the session to use/continue. If None, a temporary
                        (non-persistent) session is used for this call only.
            system_message: An optional system message for the LLM. Behavior when
                            provided for an existing session depends on context strategy
                            (e.g., it might replace or supplement existing system messages).
            provider_name: Override the default provider specified in the configuration.
                           Uses the default provider if None.
            model_name: Override the default model for the selected provider.
            stream: If True, returns an async generator yielding response text chunks (str).
                    If False (default), returns the complete response content as a string.
            save_session: If True (default) and session_id is provided, the
                          conversation turn (user message + assistant response) is
                          saved to the persistent session storage. Ignored if
                          session_id is None.
            enable_rag: If True, enables Retrieval Augmented Generation by searching
                        the vector store for context relevant to the message. (Phase 2)
            rag_retrieval_k: Number of documents to retrieve for RAG. (Phase 2)
            rag_collection_name: Name of the vector store collection for RAG. (Phase 2)
            **provider_kwargs: Additional keyword arguments passed directly to the
                               selected provider's chat completion API call
                               (e.g., temperature=0.7, max_tokens=100).

        Returns:
            If stream=False: The full response content as a string.
            If stream=True: An asynchronous generator yielding response text chunks (str).

        Raises:
            ProviderError: If the LLM provider API call fails.
            SessionNotFoundError: If a specified session_id is not found.
            ConfigError: If configuration for the selected provider/model is invalid.
            VectorStoreError: If RAG retrieval fails. (Phase 2)
            EmbeddingError: If embedding generation fails. (Phase 2)
            ContextLengthError: If essential context exceeds model limits.
            MCPError: If MCP formatting fails. (Phase 3)
            LLMCoreError: For other library-specific errors.
        """
        # --- Provider Selection ---
        try:
            active_provider = self._provider_manager.get_provider(provider_name)
        except (ConfigError, ProviderError) as e:
             logger.error(f"Failed to get provider '{provider_name or 'default'}': {e}")
             raise # Re-raise config/provider errors

        target_model = model_name or active_provider.default_model # Use specified or provider's default
        if not target_model:
             raise ConfigError(f"Could not determine target model for provider '{active_provider.get_name()}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', provider='{active_provider.get_name()}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}"
        )

        try:
            # 1. Load or create session
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
                # Placeholder for Phase 2:
                # query_embedding = await self._embedding_manager.generate_embedding(message)
                # rag_docs = await self._storage_manager.get_vector_storage().similarity_search(...)
                # rag_results_content = [doc.content for doc in rag_docs]
            # --- End RAG Placeholder ---

            # 2. Prepare context using ContextManager
            # ContextManager now gets the provider instance via ProviderManager internally
            context_payload: List[Message] = await self._context_manager.prepare_context(
                session=chat_session,
                provider_name=active_provider.get_name(), # Pass name, CM gets instance
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
                # --- Streaming Path (Implemented) ---
                logger.debug(f"Processing stream response from provider '{active_provider.get_name()}'")

                async def stream_wrapper() -> AsyncGenerator[str, None]:
                    """Wraps the provider's stream, yields text chunks, and saves session."""
                    full_response_content = ""
                    error_occurred = False
                    try:
                        # Iterate through the raw chunks from the provider
                        async for chunk_dict in response_data_or_generator: # type: ignore
                            if not isinstance(chunk_dict, dict):
                                logger.warning(f"Received non-dict chunk in stream: {chunk_dict}")
                                continue # Skip non-dictionary chunks

                            # --- Extract text delta from chunk ---
                            # This logic needs to be provider-agnostic or handled within provider
                            # For now, assume a common pattern or add provider-specific parsing here
                            text_delta = ""
                            if active_provider.get_name() == "openai":
                                choices = chunk_dict.get('choices', [])
                                if choices and isinstance(choices, list) and choices[0]:
                                    delta = choices[0].get('delta', {})
                                    text_delta = delta.get('content', '') or ""
                            elif active_provider.get_name() == "anthropic":
                                # Anthropic streaming format might differ, adjust based on its SDK/API
                                # Example placeholder structure:
                                type = chunk_dict.get("type")
                                if type == "content_block_delta":
                                     delta = chunk_dict.get("delta", {})
                                     if delta.get("type") == "text_delta":
                                          text_delta = delta.get("text", "") or ""
                                elif type == "message_delta":
                                     # Handle other delta types if necessary
                                     pass
                            elif active_provider.get_name() == "ollama":
                                # Ollama format handled in its _process_stream, yields dicts
                                # Check 'message' structure first
                                message_chunk = chunk_dict.get('message', {})
                                if isinstance(message_chunk, dict):
                                     text_delta = message_chunk.get('content', '') or ""
                                # Fallback for older /generate endpoint format
                                elif 'response' in chunk_dict:
                                     text_delta = chunk_dict.get('response', '') or ""

                            # --- End text delta extraction ---

                            if text_delta:
                                full_response_content += text_delta
                                yield text_delta
                            # Check for potential error messages within the stream
                            if chunk_dict.get('error'):
                                 error_msg = f"Error during stream: {chunk_dict['error']}"
                                 logger.error(error_msg)
                                 raise ProviderError(active_provider.get_name(), error_msg)

                    except Exception as e:
                        error_occurred = True
                        logger.error(f"Error processing stream from {active_provider.get_name()}: {e}", exc_info=True)
                        # Decide whether to raise immediately or just log and stop yielding
                        # Raising here will stop the client from getting further chunks
                        raise ProviderError(active_provider.get_name(), f"Stream processing error: {e}")
                    finally:
                        logger.debug("Stream finished.")
                        # Save session after stream completes, even if an error occurred mid-stream
                        # (to capture the partial response if desired, or just the user message)
                        if save_session and session_id:
                            # Only add assistant message if stream didn't error out immediately
                            # and some content was received.
                            if full_response_content or not error_occurred:
                                assistant_msg = chat_session.add_message(
                                    message_content=full_response_content, role=Role.ASSISTANT
                                )
                                logger.debug(f"Assistant message '{assistant_msg.id}' (length: {len(full_response_content)}) added to session '{chat_session.id}' after stream.")
                            else:
                                 logger.debug(f"No assistant message added to session '{chat_session.id}' due to stream error or empty response.")
                            # Always save the session state (which includes the user message)
                            await self._session_manager.save_session(chat_session)

                return stream_wrapper()
                # --- End Streaming Path ---
            else:
                # --- Non-Streaming Path ---
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response from provider for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(active_provider.get_name(), "Invalid response format received from provider.")

                response_data = response_data_or_generator

                # --- Extract full content from response ---
                # Needs to be provider-agnostic or handled within provider
                full_response_content = ""
                if active_provider.get_name() == "openai":
                    choices = response_data.get('choices', [])
                    if choices and isinstance(choices, list) and choices[0]:
                        message_data = choices[0].get('message', {})
                        full_response_content = message_data.get('content', '') or ""
                elif active_provider.get_name() == "anthropic":
                    # Anthropic non-streamed format might differ
                    # Example placeholder:
                    content_blocks = response_data.get('content', [])
                    if content_blocks and isinstance(content_blocks, list):
                         # Assuming text content is in the first block
                         if content_blocks[0].get("type") == "text":
                              full_response_content = content_blocks[0].get("text", "") or ""
                elif active_provider.get_name() == "ollama":
                    message_part = response_data.get('message', {})
                    if isinstance(message_part, dict):
                        full_response_content = message_part.get('content', '') or ""
                    elif 'response' in response_data: # Fallback for /generate
                        full_response_content = response_data.get('response', '') or ""

                if not full_response_content and response_data:
                     logger.warning(f"Could not extract content from non-streaming response: {response_data}")
                     full_response_content = str(response_data) # Fallback

                logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                # Add assistant message and save session
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


    # --- Session Management Methods (Delegate to SessionManager/StorageManager) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a specific chat session object (including messages)."""
        logger.debug(f"LLMCore.get_session called for session_id: {session_id}")
        try:
            # SessionManager handles loading logic using the storage backend
            return await self._session_manager.load_or_create_session(session_id=session_id)
        except SessionNotFoundError:
             return None # Return None if not found, consistent with spec
        except StorageError as e:
             logger.error(f"Storage error getting session '{session_id}': {e}")
             raise # Re-raise storage errors

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists available persistent chat sessions (metadata only)."""
        logger.debug("LLMCore.list_sessions called.")
        try:
            # Delegate directly to the session storage backend via StorageManager
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.list_sessions()
        except StorageError as e:
            logger.error(f"Storage error listing sessions: {e}")
            raise

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a persistent chat session from storage."""
        logger.debug(f"LLMCore.delete_session called for session_id: {session_id}")
        try:
            # Delegate directly to the session storage backend via StorageManager
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.delete_session(session_id)
        except StorageError as e:
            logger.error(f"Storage error deleting session '{session_id}': {e}")
            raise

    # --- RAG / Vector Store Management Methods (Skeletons - unchanged, use StorageManager when implemented) ---
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict] = None, doc_id: Optional[str] = None, collection_name: Optional[str] = None) -> str:
        # Phase 2:
        # vector_storage = self._storage_manager.get_vector_storage()
        # embedding = await self._embedding_manager.generate_embedding(content)
        # doc = ContextDocument(...)
        # return await vector_storage.add_documents([doc], collection_name) -> return doc.id
        raise NotImplementedError("RAG methods require Phase 2 implementation.")
    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str] = None) -> List[str]: raise NotImplementedError("RAG methods require Phase 2 implementation.")
    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str] = None, filter_metadata: Optional[Dict] = None) -> List[ContextDocument]: raise NotImplementedError("RAG methods require Phase 2 implementation.")
    async def delete_documents_from_vector_store(self, document_ids: List[str], *, collection_name: Optional[str] = None) -> bool: raise NotImplementedError("RAG methods require Phase 2 implementation.")


    # --- Provider Info Methods (Delegate to ProviderManager) ---
    def get_available_providers(self) -> List[str]:
        """Lists the names of all successfully loaded LLM providers."""
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """
        Lists available models for a specific loaded provider.

        Note: This might return a cached list or perform an API call depending
              on the provider implementation's get_available_models method.
        """
        logger.debug(f"LLMCore.get_models_for_provider called for: {provider_name}")
        try:
            provider = self._provider_manager.get_provider(provider_name)
            # Provider's method might be sync or async - handle appropriately if needed
            # Assuming sync for now as per BaseProvider spec
            return provider.get_available_models()
        except (ConfigError, ProviderError) as e:
            logger.error(f"Error getting models for provider '{provider_name}': {e}")
            raise # Re-raise specific errors
        except Exception as e: # Catch potential errors from provider.get_available_models()
             logger.error(f"Unexpected error getting models for provider '{provider_name}': {e}", exc_info=True)
             raise ProviderError(provider_name, f"Failed to retrieve models: {e}")


    # --- Utility / Cleanup ---
    async def close(self):
        """Closes connections for storage backends and providers."""
        logger.info("LLMCore.close() called. Cleaning up resources...")
        # Close providers via ProviderManager
        if hasattr(self, '_provider_manager') and self._provider_manager:
            await self._provider_manager.close_providers()
        # Close storage backends via StorageManager
        if hasattr(self, '_storage_manager') and self._storage_manager:
            await self._storage_manager.close_storages()
        # (Close embedding manager when implemented)
        logger.info("LLMCore resources cleanup complete.")

    # --- Async Context Management ---
    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        # Initialization is handled by the `create` classmethod.
        # This method allows using `async with LLMCore.create(...) as llm:`
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        await self.close()
