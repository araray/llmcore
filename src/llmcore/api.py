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
from .storage.manager import StorageManager
# Sessions
from .sessions.manager import SessionManager
# Context
from .context.manager import ContextManager
# Providers
from .providers.manager import ProviderManager
from .providers.base import BaseProvider
# Embedding
from .embedding.manager import EmbeddingManager # Import EmbeddingManager

# confy and tomli/tomllib
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore
try:
    import tomllib # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib # Fallback for Python < 3.11
    except ImportError:
        tomllib = None # type: ignore

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
    _embedding_manager: EmbeddingManager # Added EmbeddingManager instance variable

    def __init__(self):
        """Private constructor. Use `create` classmethod for async initialization."""
        # Initialization logic moved to the `create` classmethod
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

        Loads configuration, initializes managers (Providers, Storage, Sessions,
        Embedding, Context), and prepares the instance for use.

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
            EmbeddingError: If embedding model initialization fails.
            LLMCoreError: For other initialization errors.
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        # --- Step 1: Initialize Config ---
        try:
            from confy.loader import Config as ActualConfyConfig
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required.")

            default_config_dict = {}
            try:
                # Load default config using importlib.resources
                if hasattr(importlib.resources, 'files'): # Python 3.9+
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # type: ignore
                        default_config_dict = tomllib.load(f)
                else: # Fallback for older Python versions (less likely given requires-python >= 3.9)
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8')
                    default_config_dict = tomllib.loads(default_config_content)
            except Exception as e:
                 raise ConfigError(f"Failed to load default configuration: {e}")

            instance.config = ActualConfyConfig(
                defaults=default_config_dict, file_path=config_file_path,
                prefix=env_prefix, overrides_dict=config_overrides, mandatory=[]
            )
            logger.info("confy configuration loaded successfully.")
        except ImportError as e:
             raise ConfigError(f"Configuration dependency missing: {e}")
        except ConfigError: raise
        except Exception as e:
            raise ConfigError(f"Configuration initialization failed: {e}")

        # --- Step 2: Initialize Provider Manager ---
        try:
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized successfully.")
        except (ConfigError, ProviderError) as e:
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True)
            raise
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
            raise
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

        # --- Step 5: Initialize Embedding Manager ---
        try:
            instance._embedding_manager = EmbeddingManager(instance.config)
            # Initialize the actual embedding model asynchronously
            await instance._embedding_manager.initialize_embedding_model()
            logger.info("EmbeddingManager initialized successfully.")
        except (ConfigError, EmbeddingError) as e:
             logger.error(f"Failed to initialize EmbeddingManager: {e}", exc_info=True)
             # If embedding fails, RAG won't work, but core chat might still be usable.
             # Decide whether to raise or just log a warning. Raising for clarity.
             raise
        except Exception as e:
            logger.error(f"Unexpected error initializing EmbeddingManager: {e}", exc_info=True)
            raise LLMCoreError(f"EmbeddingManager initialization failed: {e}")


        # --- Step 6: Initialize Context Manager ---
        try:
            # Pass ProviderManager, StorageManager, and EmbeddingManager
            instance._context_manager = ContextManager(
                config=instance.config,
                provider_manager=instance._provider_manager,
                storage_manager=instance._storage_manager, # Pass StorageManager
                embedding_manager=instance._embedding_manager # Pass EmbeddingManager
            )
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
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        # RAG parameters
        enable_rag: bool = False, # Now functional
        rag_retrieval_k: Optional[int] = None, # Functional
        rag_collection_name: Optional[str] = None, # Functional
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
                            provided for an existing session depends on context strategy.
            provider_name: Override the default provider specified in the configuration.
            model_name: Override the default model for the selected provider.
            stream: If True, returns an async generator yielding response text chunks (str).
                    If False (default), returns the complete response content as a string.
            save_session: If True (default) and session_id is provided, the
                          conversation turn is saved to persistent session storage.
            enable_rag: If True, enables Retrieval Augmented Generation.
            rag_retrieval_k: Number of documents to retrieve for RAG. Overrides config default.
            rag_collection_name: Vector store collection name for RAG. Overrides config default.
            **provider_kwargs: Additional keyword arguments passed directly to the
                               selected provider's chat completion API call.

        Returns:
            If stream=False: The full response content as a string.
            If stream=True: An asynchronous generator yielding response text chunks (str).

        Raises:
            ProviderError, SessionNotFoundError, ConfigError, VectorStorageError,
            EmbeddingError, ContextLengthError, MCPError, LLMCoreError.
        """
        # --- Provider Selection ---
        try:
            active_provider = self._provider_manager.get_provider(provider_name)
            provider_actual_name = active_provider.get_name() # Get actual name used
        except (ConfigError, ProviderError) as e:
             logger.error(f"Failed to get provider '{provider_name or 'default'}': {e}")
             raise

        target_model = model_name or active_provider.default_model
        if not target_model:
             raise ConfigError(f"Could not determine target model for provider '{provider_actual_name}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', provider='{provider_actual_name}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}"
        )

        try:
            # 1. Load or create session
            chat_session = await self._session_manager.load_or_create_session(
                session_id, system_message
            )
            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}'")

            # 2. Prepare context using ContextManager
            # ContextManager now handles RAG internally if enabled
            context_payload: List[Message] = await self._context_manager.prepare_context(
                session=chat_session,
                provider_name=provider_actual_name,
                model_name=target_model,
                rag_enabled=enable_rag, # Pass RAG flag
                rag_k=rag_retrieval_k, # Pass RAG parameters
                rag_collection=rag_collection_name,
                # Pass MCP flag if implemented
            )
            logger.info(f"Prepared context with {len(context_payload)} messages for model '{target_model}'.")

            # 3. Call provider's chat_completion
            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload,
                model=target_model,
                stream=stream,
                **provider_kwargs
            )

            # 4. Process response and save session (Stream/Non-Stream logic remains similar)
            if stream:
                # --- Streaming Path ---
                logger.debug(f"Processing stream response from provider '{provider_actual_name}'")
                return self._stream_response_wrapper(
                    response_data_or_generator, # type: ignore
                    active_provider,
                    chat_session,
                    save_session
                )
            else:
                # --- Non-Streaming Path ---
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(provider_actual_name, "Invalid response format.")

                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)
                logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                if save_session and session_id:
                    assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                    logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")
                    await self._session_manager.save_session(chat_session)

                return full_response_content

        except (SessionNotFoundError, SessionStorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError, MCPError) as e: # Added RAG errors
             logger.error(f"Chat failed: {e}")
             raise # Propagate specific, known errors
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             raise LLMCoreError(f"Chat execution failed: {e}")


    async def _stream_response_wrapper(
        self,
        provider_stream: AsyncGenerator[Dict[str, Any], None],
        provider: BaseProvider,
        session: ChatSession,
        save_session: bool
    ) -> AsyncGenerator[str, None]:
        """Wraps the provider's stream, yields text chunks, and saves session."""
        full_response_content = ""
        error_occurred = False
        provider_name = provider.get_name()
        session_id = session.id if session.id else None # Get ID for saving check

        try:
            async for chunk_dict in provider_stream:
                if not isinstance(chunk_dict, dict):
                    logger.warning(f"Received non-dict chunk in stream: {chunk_dict}")
                    continue

                text_delta = self._extract_delta_content(chunk_dict, provider)

                if text_delta:
                    full_response_content += text_delta
                    yield text_delta

                # Check for potential error messages within the stream
                if chunk_dict.get('error'):
                     error_msg = f"Error during stream: {chunk_dict['error']}"
                     logger.error(error_msg)
                     raise ProviderError(provider_name, error_msg)
                # Check for finish reason indicating an issue (e.g., safety)
                finish_reason = chunk_dict.get('finish_reason')
                if finish_reason and finish_reason not in ["stop", "length", None]: # Adjust based on provider reasons
                     error_msg = f"Stream stopped due to reason: {finish_reason}"
                     logger.warning(error_msg)
                     # Optionally raise or just stop yielding
                     raise ProviderError(provider_name, error_msg)


        except Exception as e:
            error_occurred = True
            logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True)
            if isinstance(e, ProviderError): raise # Re-raise provider errors
            raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if save_session and session_id:
                if full_response_content or not error_occurred:
                    assistant_msg = session.add_message(
                        message_content=full_response_content, role=Role.ASSISTANT
                    )
                    logger.debug(f"Assistant message '{assistant_msg.id}' (length: {len(full_response_content)}) added to session '{session_id}' after stream.")
                else:
                     logger.debug(f"No assistant message added to session '{session_id}' due to stream error or empty response.")
                # Always save the session state (which includes the user message)
                await self._session_manager.save_session(session)


    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts the text delta from a stream chunk based on the provider."""
        provider_name = provider.get_name()
        text_delta = ""
        try:
            if provider_name == "openai":
                choices = chunk.get('choices', [])
                if choices and isinstance(choices, list) and choices[0]:
                    delta = choices[0].get('delta', {})
                    text_delta = delta.get('content', '') or ""
            elif provider_name == "anthropic":
                type = chunk.get("type")
                if type == "content_block_delta":
                     delta = chunk.get("delta", {})
                     if delta.get("type") == "text_delta":
                          text_delta = delta.get("text", "") or ""
            elif provider_name == "ollama":
                # Handle official ollama library stream format
                message_chunk = chunk.get('message', {})
                if isinstance(message_chunk, dict):
                     text_delta = message_chunk.get('content', '') or ""
                # Fallback for older /generate endpoint format if needed
                elif 'response' in chunk:
                     text_delta = chunk.get('response', '') or ""
            elif provider_name == "gemini":
                 # Handle official google-genai stream format
                 message_chunk = chunk.get('message', {})
                 if isinstance(message_chunk, dict):
                      text_delta = message_chunk.get('content', '') or ""
                 # Fallback check on choices delta (mimicking OpenAI structure)
                 elif chunk.get('choices') and chunk['choices'][0].get('delta'):
                      text_delta = chunk['choices'][0]['delta'].get('content', '') or ""


        except Exception as e:
             logger.warning(f"Error extracting delta content from {provider_name} chunk: {e}. Chunk: {chunk}")
             text_delta = "" # Ensure empty string on error

        return text_delta


    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts the full response content from a non-streaming response dict."""
        provider_name = provider.get_name()
        full_response_content = ""
        try:
            if provider_name == "openai":
                choices = response_data.get('choices', [])
                if choices and isinstance(choices, list) and choices[0]:
                    message_data = choices[0].get('message', {})
                    full_response_content = message_data.get('content', '') or ""
            elif provider_name == "anthropic":
                content_blocks = response_data.get('content', [])
                if content_blocks and isinstance(content_blocks, list):
                     if content_blocks[0].get("type") == "text":
                          full_response_content = content_blocks[0].get("text", "") or ""
            elif provider_name == "ollama":
                # Handle official ollama library response format
                message_part = response_data.get('message', {})
                if isinstance(message_part, dict):
                    full_response_content = message_part.get('content', '') or ""
                elif 'response' in response_data: # Fallback for /generate
                    full_response_content = response_data.get('response', '') or ""
            elif provider_name == "gemini":
                 # Handle official google-genai response format
                 choices = response_data.get('choices', [])
                 if choices and choices[0].get('message'):
                      full_response_content = choices[0]['message'].get('content', '') or ""
                 # Add other potential extraction paths if the library format differs

            if not full_response_content and response_data:
                 logger.warning(f"Could not extract content from non-streaming {provider_name} response: {response_data}")
                 full_response_content = str(response_data) # Fallback

        except Exception as e:
             logger.error(f"Error extracting full content from {provider_name} response: {e}. Response: {response_data}", exc_info=True)
             full_response_content = f"Error: Could not parse response. {e}" # Provide error info

        return full_response_content


    # --- Session Management Methods (Delegate to SessionManager/StorageManager) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a specific chat session object (including messages)."""
        logger.debug(f"LLMCore.get_session called for session_id: {session_id}")
        try:
            return await self._session_manager.load_or_create_session(session_id=session_id)
        except SessionNotFoundError:
             return None
        except StorageError as e:
             logger.error(f"Storage error getting session '{session_id}': {e}")
             raise

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists available persistent chat sessions (metadata only)."""
        logger.debug("LLMCore.list_sessions called.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.list_sessions()
        except StorageError as e:
            logger.error(f"Storage error listing sessions: {e}")
            raise

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a persistent chat session from storage."""
        logger.debug(f"LLMCore.delete_session called for session_id: {session_id}")
        try:
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.delete_session(session_id)
        except StorageError as e:
            logger.error(f"Storage error deleting session '{session_id}': {e}")
            raise

    # --- RAG / Vector Store Management Methods (Now Implemented) ---

    async def add_document_to_vector_store(
        self,
        content: str,
        *,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None, # Allow user to provide ID
        collection_name: Optional[str] = None
    ) -> str:
        """
        Adds a single document (text content) to the configured vector store.

        Generates embedding using the configured embedding model and stores
        the document content, embedding, and metadata.

        Args:
            content: The text content of the document.
            metadata: Optional dictionary of metadata associated with the document.
            doc_id: Optional specific ID to assign to the document. If None,
                    an ID will be generated by the ContextDocument model.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            The ID assigned to the added document.

        Raises:
            EmbeddingError: If embedding generation fails.
            VectorStorageError: If adding the document to the store fails.
            ConfigError: If vector store or embedding model is not configured/initialized.
        """
        logger.debug(f"Adding document to vector store (Collection: {collection_name or 'default'})...")
        try:
            # 1. Generate embedding via EmbeddingManager
            embedding = await self._embedding_manager.generate_embedding(content)

            # 2. Create ContextDocument (ID generated if not provided)
            # Ensure metadata is a dict
            doc_metadata = metadata if metadata is not None else {}
            doc = ContextDocument(id=doc_id, content=content, embedding=embedding, metadata=doc_metadata)

            # 3. Get vector storage and add document
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents([doc], collection_name=collection_name)

            if not added_ids:
                 # This case should ideally not happen if add_documents raises on failure
                 raise VectorStorageError("Failed to add document, no ID returned.")

            logger.info(f"Document '{added_ids[0]}' added to vector store collection '{collection_name or 'default'}'.")
            return added_ids[0] # Return the actual ID used (could be generated)

        except (EmbeddingError, VectorStorageError, ConfigError) as e:
             logger.error(f"Failed to add document to vector store: {e}")
             raise # Re-raise specific errors
        except Exception as e:
             logger.error(f"Unexpected error adding document: {e}", exc_info=True)
             raise VectorStorageError(f"Unexpected error adding document: {e}")


    async def add_documents_to_vector_store(
        self,
        documents: List[Dict[str, Any]],
        *,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Adds multiple documents to the configured vector store in a batch.

        Args:
            documents: A list of dictionaries, each representing a document.
                       Expected format: {"content": str, "metadata": Optional[Dict], "id": Optional[str]}
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            A list of IDs assigned to the added documents.

        Raises:
            EmbeddingError: If embedding generation fails for any document.
            VectorStorageError: If adding documents to the store fails.
            ConfigError: If vector store or embedding model is not configured/initialized.
            ValueError: If input document format is invalid.
        """
        if not documents:
            return []

        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            # 1. Extract content and prepare for batch embedding
            contents = []
            context_docs_to_create = []
            for doc_data in documents:
                content = doc_data.get("content")
                if not isinstance(content, str):
                    raise ValueError(f"Invalid document data: 'content' field missing or not a string in {doc_data}")
                contents.append(content)
                # Store other data temporarily
                context_docs_to_create.append({
                    "id": doc_data.get("id"), # Allow None, ContextDocument generates ID
                    "content": content,
                    "metadata": doc_data.get("metadata", {}) # Ensure metadata is dict
                })

            # 2. Generate embeddings in batch via EmbeddingManager
            embeddings = await self._embedding_manager.generate_embeddings(contents)

            if len(embeddings) != len(documents):
                raise EmbeddingError(message="Mismatch between number of texts and generated embeddings.")

            # 3. Create list of ContextDocument objects with embeddings
            docs_to_add: List[ContextDocument] = []
            for i, doc_data in enumerate(context_docs_to_create):
                docs_to_add.append(
                    ContextDocument(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        embedding=embeddings[i],
                        metadata=doc_data["metadata"]
                    )
                )

            # 4. Get vector storage and add documents
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)

            logger.info(f"Batch of {len(added_ids)} documents added/updated in vector store collection '{collection_name or 'default'}'.")
            return added_ids

        except (EmbeddingError, VectorStorageError, ConfigError, ValueError) as e:
             logger.error(f"Failed to add documents batch to vector store: {e}")
             raise # Re-raise specific errors
        except Exception as e:
             logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True)
             raise VectorStorageError(f"Unexpected error adding documents batch: {e}")


    async def search_vector_store(
        self,
        query: str,
        *,
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[ContextDocument]:
        """
        Performs a similarity search for relevant documents in the vector store.

        Generates an embedding for the query text and searches the specified
        collection.

        Args:
            query: The text query to search for.
            k: The number of top similar documents to retrieve.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.
            filter_metadata: Optional dictionary to filter results based on metadata.

        Returns:
            A list of ContextDocument objects representing the search results.

        Raises:
            EmbeddingError: If embedding generation for the query fails.
            VectorStorageError: If the search operation fails.
            ConfigError: If vector store or embedding model is not configured/initialized.
            ValueError: If k is not a positive integer.
        """
        if k <= 0:
            raise ValueError("'k' must be a positive integer for search.")

        logger.debug(f"Searching vector store (k={k}, Collection: {collection_name or 'default'}) for query: '{query[:50]}...'")
        try:
            # 1. Generate query embedding via EmbeddingManager
            query_embedding = await self._embedding_manager.generate_embedding(query)

            # 2. Get vector storage and perform search
            vector_storage = self._storage_manager.get_vector_storage()
            results = await vector_storage.similarity_search(
                query_embedding=query_embedding,
                k=k,
                collection_name=collection_name,
                filter_metadata=filter_metadata
            )
            logger.info(f"Vector store search returned {len(results)} documents.")
            return results

        except (EmbeddingError, VectorStorageError, ConfigError) as e:
             logger.error(f"Failed to search vector store: {e}")
             raise # Re-raise specific errors
        except Exception as e:
             logger.error(f"Unexpected error searching vector store: {e}", exc_info=True)
             raise VectorStorageError(f"Unexpected error searching vector store: {e}")


    async def delete_documents_from_vector_store(
        self,
        document_ids: List[str],
        *,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Deletes documents from the vector store by their IDs.

        Args:
            document_ids: A list of document IDs to delete.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            True if deletion was attempted successfully (even if some IDs weren't found),
            False otherwise (indicating a storage backend error).

        Raises:
            VectorStorageError: If the deletion operation fails fundamentally.
            ConfigError: If vector store is not configured/initialized or collection is invalid.
            ValueError: If document_ids list is empty.
        """
        if not document_ids:
            # raise ValueError("document_ids list cannot be empty for deletion.")
            logger.warning("delete_documents_from_vector_store called with empty ID list.")
            return True # No operation needed, considered successful

        logger.debug(f"Deleting {len(document_ids)} documents from vector store (Collection: {collection_name or 'default'})...")
        try:
            vector_storage = self._storage_manager.get_vector_storage()
            success = await vector_storage.delete_documents(
                document_ids=document_ids,
                collection_name=collection_name
            )
            logger.info(f"Deletion attempt for {len(document_ids)} documents completed (Success: {success}).")
            return success
        except (VectorStorageError, ConfigError) as e:
             logger.error(f"Failed to delete documents from vector store: {e}")
             raise # Re-raise specific errors
        except Exception as e:
             logger.error(f"Unexpected error deleting documents: {e}", exc_info=True)
             raise VectorStorageError(f"Unexpected error deleting documents: {e}")


    # --- Provider Info Methods (Delegate to ProviderManager) ---
    def get_available_providers(self) -> List[str]:
        """Lists the names of all successfully loaded LLM providers."""
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Lists available models for a specific loaded provider."""
        logger.debug(f"LLMCore.get_models_for_provider called for: {provider_name}")
        try:
            provider = self._provider_manager.get_provider(provider_name)
            # Assuming sync for now as per BaseProvider spec
            # TODO: Consider if this should be async to allow API calls
            return provider.get_available_models()
        except (ConfigError, ProviderError) as e:
            logger.error(f"Error getting models for provider '{provider_name}': {e}")
            raise
        except Exception as e:
             logger.error(f"Unexpected error getting models for provider '{provider_name}': {e}", exc_info=True)
             raise ProviderError(provider_name, f"Failed to retrieve models: {e}")


    # --- Utility / Cleanup ---
    async def close(self):
        """Closes connections for storage backends, providers, and embedding manager."""
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = []
        # Close providers via ProviderManager
        if hasattr(self, '_provider_manager') and self._provider_manager:
            close_tasks.append(self._provider_manager.close_providers())
        # Close storage backends via StorageManager
        if hasattr(self, '_storage_manager') and self._storage_manager:
            close_tasks.append(self._storage_manager.close_storages())
        # Close embedding manager
        if hasattr(self, '_embedding_manager') and self._embedding_manager:
             close_tasks.append(self._embedding_manager.close())

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for result in results:
                 if isinstance(result, Exception):
                      logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)

        logger.info("LLMCore resources cleanup complete.")

    # --- Async Context Management ---
    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        # Initialization is handled by the `create` classmethod.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        await self.close()
