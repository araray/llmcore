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
from .embedding.manager import EmbeddingManager

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

# Import Ollama response type for stream checking if available
try:
    from ollama import ChatResponse as OllamaChatResponse
except ImportError:
    OllamaChatResponse = None # type: ignore


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
    _embedding_manager: EmbeddingManager

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
                if hasattr(importlib.resources, 'files'): # Python 3.9+
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # type: ignore
                        default_config_dict = tomllib.load(f)
                else: # Fallback unlikely given requires-python >= 3.9
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

        # --- Initialize Managers ---
        # Provider Manager
        try:
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized.")
        except (ConfigError, ProviderError) as e:
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True)
            raise
        except Exception as e:
            raise LLMCoreError(f"ProviderManager initialization failed: {e}")

        # Storage Manager
        try:
            instance._storage_manager = StorageManager(instance.config)
            await instance._storage_manager.initialize_storages()
            logger.info("StorageManager initialized.")
        except (ConfigError, StorageError) as e:
            logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True)
            raise
        except Exception as e:
            raise LLMCoreError(f"StorageManager initialization failed: {e}")

        # Session Manager
        try:
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized.")
        except StorageError as e:
             raise LLMCoreError(f"SessionManager initialization failed due to storage issue: {e}")
        except Exception as e:
            raise LLMCoreError(f"SessionManager initialization failed: {e}")

        # Embedding Manager
        try:
            instance._embedding_manager = EmbeddingManager(instance.config)
            await instance._embedding_manager.initialize_embedding_model()
            logger.info("EmbeddingManager initialized.")
        except (ConfigError, EmbeddingError) as e:
             logger.error(f"Failed to initialize EmbeddingManager: {e}", exc_info=True)
             raise # Embedding failure is critical for RAG, raise it.
        except Exception as e:
            raise LLMCoreError(f"EmbeddingManager initialization failed: {e}")

        # Context Manager
        try:
            instance._context_manager = ContextManager(
                config=instance.config,
                provider_manager=instance._provider_manager,
                storage_manager=instance._storage_manager,
                embedding_manager=instance._embedding_manager
            )
            logger.info("ContextManager initialized.")
        except Exception as e:
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
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
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
            provider_actual_name = active_provider.get_name()
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
            # Need to handle potential token counting errors here
            try:
                context_payload: List[Message] = await self._context_manager.prepare_context(
                    session=chat_session,
                    provider_name=provider_actual_name,
                    model_name=target_model,
                    rag_enabled=enable_rag,
                    rag_k=rag_retrieval_k,
                    rag_collection=rag_collection_name,
                )
            except ProviderError as token_error:
                # If token counting fails within prepare_context, re-raise as a chat failure
                logger.error(f"Token counting failed during context preparation: {token_error}")
                raise ProviderError(provider_actual_name, f"Token counting failed for message {user_msg_obj.id}: {token_error}")

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
                # --- Streaming Path ---
                logger.debug(f"Processing stream response from provider '{provider_actual_name}'")
                # Ensure the generator type hint matches what the wrapper expects
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # Use Any for broader compatibility initially
                return self._stream_response_wrapper(
                    provider_stream,
                    active_provider,
                    chat_session,
                    save_session
                )
            else:
                # --- Non-Streaming Path ---
                if not isinstance(response_data_or_generator, dict):
                     # This check was causing the issue with Ollama's ChatResponse object
                     # The provider should now always return a dict
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(provider_actual_name, "Invalid response format (expected dict).")

                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)
                # Check if extraction failed (returned None) vs. returning an empty string
                if full_response_content is None:
                     logger.warning(f"Could not extract content from non-streaming {provider_actual_name} response: {response_data}")
                     full_response_content = f"Response received, but content extraction failed. Data: {str(response_data)[:200]}..."
                else:
                     logger.debug(f"Received full response content (length: {len(full_response_content)}).")


                if save_session and session_id:
                    assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                    logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")
                    await self._session_manager.save_session(chat_session)

                return full_response_content

        except (SessionNotFoundError, SessionStorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError, MCPError) as e:
             logger.error(f"Chat failed: {e}")
             raise
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             raise LLMCoreError(f"Chat execution failed: {e}")


    async def _stream_response_wrapper(
        self,
        provider_stream: AsyncGenerator[Any, None], # Accept Any type initially
        provider: BaseProvider,
        session: ChatSession,
        save_session: bool
    ) -> AsyncGenerator[str, None]:
        """Wraps the provider's stream, yields text chunks, and saves session."""
        full_response_content = ""
        error_occurred = False
        provider_name = provider.get_name()
        # Use session.id directly; load_or_create ensures it exists even for temp sessions
        session_id_for_saving = session.id if save_session else None

        try:
            async for chunk in provider_stream:
                chunk_dict: Optional[Dict[str, Any]] = None

                # Convert chunk to dict if necessary (specifically for Ollama)
                if isinstance(chunk, dict):
                    chunk_dict = chunk
                elif OllamaChatResponse and isinstance(chunk, OllamaChatResponse):
                    # Convert Ollama response object to dict
                    try:
                        chunk_dict = chunk.model_dump()
                    except Exception as dump_err:
                        logger.warning(f"Could not dump Ollama stream object to dict: {dump_err}. Chunk: {chunk}")
                        continue # Skip this chunk if conversion fails
                else:
                    # Handle other potential non-dict types if necessary, or log warning
                    logger.warning(f"Received non-dict/non-OllamaResponse chunk in stream: {type(chunk)} - {chunk}")
                    continue # Skip unrecognized chunk types

                if not chunk_dict: # Skip if conversion failed or chunk was invalid
                    continue

                # Extract delta and check for errors/finish reasons within the chunk dict
                text_delta = self._extract_delta_content(chunk_dict, provider)
                error_message = chunk_dict.get('error')
                # Adjust finish reason check based on provider specifics if needed
                finish_reason = chunk_dict.get('finish_reason')

                if text_delta:
                    full_response_content += text_delta
                    yield text_delta

                if error_message:
                     error_msg = f"Error during stream: {error_message}"
                     logger.error(error_msg)
                     raise ProviderError(provider_name, error_msg)

                # Check for problematic finish reasons
                if finish_reason and finish_reason not in ["stop", "length", None]:
                     error_msg = f"Stream stopped due to reason: {finish_reason}"
                     logger.warning(error_msg)
                     # Don't raise here, let the stream finish naturally but log the warning
                     # raise ProviderError(provider_name, error_msg) # Stop the stream on problematic finish

        except Exception as e:
            error_occurred = True
            logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True)
            if isinstance(e, ProviderError): raise
            raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            # Save only if save_session was True and a session_id exists (persistent session)
            # Check if session_id is not None and potentially not temporary
            if save_session and session.id and not session.id.startswith("temp_"):
                if full_response_content or not error_occurred:
                    assistant_msg = session.add_message(
                        message_content=full_response_content, role=Role.ASSISTANT
                    )
                    logger.debug(f"Assistant message '{assistant_msg.id}' (length: {len(full_response_content)}) added to session '{session.id}' after stream.")
                else:
                     logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response.")
                # Always save the session state (which includes the user message)
                try:
                    await self._session_manager.save_session(session)
                except Exception as save_e:
                     logger.error(f"Failed to save session {session.id} after stream: {save_e}", exc_info=True)


    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts the text delta from a stream chunk based on the provider."""
        provider_name = provider.get_name()
        text_delta = ""
        try:
            if provider_name == "openai":
                choices = chunk.get('choices', [])
                if choices and choices[0].get('delta'):
                    text_delta = choices[0]['delta'].get('content', '') or ""
            elif provider_name == "anthropic":
                type = chunk.get("type")
                if type == "content_block_delta" and chunk.get('delta', {}).get('type') == "text_delta":
                    text_delta = chunk.get('delta', {}).get('text', "") or ""
                # Add handling for other Anthropic stream types if necessary
            elif provider_name == "ollama":
                # Handle official ollama library stream format (now expecting dict)
                message_chunk = chunk.get('message', {})
                if message_chunk:
                    text_delta = message_chunk.get('content', '') or ""
                elif 'response' in chunk: # Fallback for older generate stream? Unlikely with new lib.
                    text_delta = chunk.get('response', '') or ""
            elif provider_name == "gemini":
                 # Handle official google-genai stream format
                 # The most reliable way seems to be checking the 'message' structure if present,
                 # otherwise fallback to the OpenAI-like structure if yielded by our wrapper.
                 message_chunk = chunk.get('message', {})
                 if message_chunk and isinstance(message_chunk, dict):
                      text_delta = message_chunk.get('content', '') or ""
                 elif chunk.get('choices') and chunk['choices'][0].get('delta'):
                      text_delta = chunk['choices'][0]['delta'].get('content', '') or ""
                 # Direct text attribute might exist on the raw chunk object before dict conversion,
                 # but we standardized on dicts. If the dict conversion failed, text_delta remains "".

        except Exception as e:
             logger.warning(f"Error extracting delta content from {provider_name} chunk: {e}. Chunk: {chunk}")
             text_delta = ""

        return text_delta or "" # Ensure string return


    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> Optional[str]:
        """
        Extracts the full response content from a non-streaming response dict.
        Returns the content string, or None if extraction fails.
        Handles empty string ('') as valid content.
        """
        provider_name = provider.get_name()
        full_response_content: Optional[str] = None # Initialize to None
        try:
            if provider_name == "openai":
                choices = response_data.get('choices', [])
                if choices and choices[0].get('message'):
                    full_response_content = choices[0]['message'].get('content') # Allow None
            elif provider_name == "anthropic":
                content_blocks = response_data.get('content', [])
                if content_blocks and content_blocks[0].get("type") == "text":
                     full_response_content = content_blocks[0].get("text") # Allow None
            elif provider_name == "ollama":
                message_part = response_data.get('message', {})
                if message_part:
                    # Use .get() which returns None if 'content' is missing
                    full_response_content = message_part.get('content')
                elif 'response' in response_data: # Fallback for generate? Unlikely.
                    full_response_content = response_data.get('response') # Allow None
            elif provider_name == "gemini":
                 choices = response_data.get('choices', [])
                 if choices and choices[0].get('message'):
                      full_response_content = choices[0]['message'].get('content') # Allow None

            # If content was found but is None, convert it to an empty string
            if full_response_content is None and response_data:
                 # Check if the expected path exists but value is None/missing
                 if provider_name == "openai" and response_data.get('choices', [{}])[0].get('message'):
                     full_response_content = "" # Assume empty content if structure exists but content is None
                 elif provider_name == "anthropic" and response_data.get('content', [{}])[0].get("type") == "text":
                     full_response_content = ""
                 elif provider_name == "ollama" and response_data.get('message'):
                     full_response_content = ""
                 elif provider_name == "gemini" and response_data.get('choices', [{}])[0].get('message'):
                      full_response_content = ""
                 else:
                      # Content path not found, extraction truly failed
                      logger.warning(f"Could not extract content path from non-streaming {provider_name} response structure: {response_data}")
                      return None # Indicate failure

            # Ensure return is string or None
            return str(full_response_content) if full_response_content is not None else None

        except Exception as e:
             logger.error(f"Error extracting full content from {provider_name} response: {e}. Response: {response_data}", exc_info=True)
             return None # Indicate failure


    # --- Session Management Methods ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a specific chat session object (including messages)."""
        logger.debug(f"LLMCore.get_session called for session_id: {session_id}")
        try:
            # SessionManager handles loading logic using the storage backend
            # Allow SessionNotFoundError to propagate if session_id is specified but not found
            return await self._session_manager.load_or_create_session(session_id=session_id)
        except SessionNotFoundError:
             logger.warning(f"Session ID '{session_id}' not found.")
             return None # Return None if not found, consistent with spec
        except StorageError as e:
             logger.error(f"Storage error getting session '{session_id}': {e}")
             raise # Re-raise storage errors

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

    # --- RAG / Vector Store Management Methods ---
    async def add_document_to_vector_store(
        self,
        content: str,
        *,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
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
            embedding = await self._embedding_manager.generate_embedding(content)
            doc_metadata = metadata if metadata is not None else {}
            doc = ContextDocument(id=doc_id, content=content, embedding=embedding, metadata=doc_metadata)
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents([doc], collection_name=collection_name)
            if not added_ids:
                 raise VectorStorageError("Failed to add document, no ID returned.")
            logger.info(f"Document '{added_ids[0]}' added to vector store collection '{collection_name or 'default'}'.")
            return added_ids[0]
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e:
             logger.error(f"Failed to add document to vector store: {e}")
             raise
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
            EmbeddingError, VectorStorageError, ConfigError, ValueError.
        """
        if not documents: return []
        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            contents = []
            context_docs_to_create = []
            for doc_data in documents:
                content = doc_data.get("content")
                if not isinstance(content, str): raise ValueError(f"Invalid document data: 'content' missing/not string in {doc_data}")
                contents.append(content)
                context_docs_to_create.append({
                    "id": doc_data.get("id"),
                    "content": content,
                    "metadata": doc_data.get("metadata", {})
                })

            embeddings = await self._embedding_manager.generate_embeddings(contents)
            if len(embeddings) != len(documents): raise EmbeddingError("Mismatch between texts and generated embeddings.")

            docs_to_add: List[ContextDocument] = [
                ContextDocument(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    embedding=embeddings[i],
                    metadata=doc_data["metadata"]
                ) for i, doc_data in enumerate(context_docs_to_create)
            ]

            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)
            logger.info(f"Batch of {len(added_ids)} documents added/updated in vector store collection '{collection_name or 'default'}'.")
            return added_ids
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e:
             logger.error(f"Failed to add documents batch to vector store: {e}")
             raise
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

        Args:
            query: The text query to search for.
            k: The number of top similar documents to retrieve.
            collection_name: The target vector store collection. Uses default if None.
            filter_metadata: Optional dictionary to filter results based on metadata.

        Returns:
            A list of ContextDocument objects representing the search results.

        Raises:
            EmbeddingError, VectorStorageError, ConfigError, ValueError.
        """
        if k <= 0: raise ValueError("'k' must be a positive integer for search.")
        logger.debug(f"Searching vector store (k={k}, Collection: {collection_name or 'default'}) for query: '{query[:50]}...'")
        try:
            query_embedding = await self._embedding_manager.generate_embedding(query)
            vector_storage = self._storage_manager.get_vector_storage()
            results = await vector_storage.similarity_search(
                query_embedding=query_embedding, k=k,
                collection_name=collection_name, filter_metadata=filter_metadata
            )
            logger.info(f"Vector store search returned {len(results)} documents.")
            return results
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e:
             logger.error(f"Failed to search vector store: {e}")
             raise
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
            collection_name: The target vector store collection. Uses default if None.

        Returns:
            True if deletion was attempted successfully, False otherwise.

        Raises:
            VectorStorageError, ConfigError, ValueError.
        """
        if not document_ids:
            logger.warning("delete_documents_from_vector_store called with empty ID list.")
            return True
        logger.debug(f"Deleting {len(document_ids)} documents from vector store (Collection: {collection_name or 'default'})...")
        try:
            vector_storage = self._storage_manager.get_vector_storage()
            success = await vector_storage.delete_documents(
                document_ids=document_ids, collection_name=collection_name
            )
            logger.info(f"Deletion attempt for {len(document_ids)} documents completed (Success: {success}).")
            return success
        except (VectorStorageError, ConfigError, StorageError) as e:
             logger.error(f"Failed to delete documents from vector store: {e}")
             raise
        except Exception as e:
             logger.error(f"Unexpected error deleting documents: {e}", exc_info=True)
             raise VectorStorageError(f"Unexpected error deleting documents: {e}")


    # --- Provider Info Methods ---
    def get_available_providers(self) -> List[str]:
        """Lists the names of all successfully loaded LLM providers."""
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Lists available models for a specific loaded provider."""
        logger.debug(f"LLMCore.get_models_for_provider called for: {provider_name}")
        try:
            provider = self._provider_manager.get_provider(provider_name)
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
        close_tasks = [
            self._provider_manager.close_providers(),
            self._storage_manager.close_storages(),
            self._embedding_manager.close(),
        ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception):
                  logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)
        logger.info("LLMCore resources cleanup complete.")

    # --- Async Context Management ---
    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        # If initialization needs to be async and tied to context entry, move it here.
        # For now, assuming initialization happens via LLMCore.create()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        await self.close()
