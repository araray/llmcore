# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
"""

import asyncio
import logging
import importlib.resources
import pathlib # For file operations
import json # For parsing stream chunks if needed
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Type

# Models and Exceptions
from .models import ChatSession, ContextDocument, Message, Role, ContextItem, ContextItemType # Added ContextItem, ContextItemType
from .exceptions import (
    LLMCoreError, ProviderError, SessionNotFoundError, ConfigError,
    StorageError, SessionStorageError, VectorStorageError,
    EmbeddingError, ContextLengthError
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

try:
    from ollama import ChatResponse as OllamaChatResponse
except ImportError:
    OllamaChatResponse = None # type: ignore [assignment]


logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.
    Provides methods for chat completions, session management, RAG, and context pool management.
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
        (Documentation remains similar, details in implementation)
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        try:
            from confy.loader import Config as ActualConfyConfig # type: ignore[no-redef]
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required for loading default_config.toml.")

            default_config_dict = {}
            try:
                if hasattr(importlib.resources, 'files'):
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # type: ignore[union-attr]
                        default_config_dict = tomllib.load(f) # type: ignore[union-attr]
                else:
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8')
                    default_config_dict = tomllib.loads(default_config_content) # type: ignore[union-attr]
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

        # Initialize Managers (Provider, Storage, Session, Embedding, Context)
        # (Initialization logic for managers remains the same as previous version)
        try:
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized.")
        except (ConfigError, ProviderError) as e:
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"ProviderManager initialization failed: {e}")
        try:
            instance._storage_manager = StorageManager(instance.config)
            await instance._storage_manager.initialize_storages()
            logger.info("StorageManager initialized.")
        except (ConfigError, StorageError) as e:
            logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"StorageManager initialization failed: {e}")
        try:
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized.")
        except StorageError as e: raise LLMCoreError(f"SessionManager init failed due to storage: {e}")
        except Exception as e: raise LLMCoreError(f"SessionManager initialization failed: {e}")
        try:
            instance._embedding_manager = EmbeddingManager(instance.config)
            await instance._embedding_manager.initialize_embedding_model()
            logger.info("EmbeddingManager initialized.")
        except (ConfigError, EmbeddingError) as e:
             logger.error(f"Failed to initialize EmbeddingManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"EmbeddingManager initialization failed: {e}")
        try:
            instance._context_manager = ContextManager(
                config=instance.config,
                provider_manager=instance._provider_manager,
                storage_manager=instance._storage_manager,
                embedding_manager=instance._embedding_manager
            )
            logger.info("ContextManager initialized.")
        except Exception as e: raise LLMCoreError(f"ContextManager initialization failed: {e}")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

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
        active_context_item_ids: Optional[List[str]] = None, # New: IDs of enabled ContextItems
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, user-added context, and RAG.

        Args:
            message: User's message.
            session_id: Session ID. If None, chat is stateless (no history/context items saved).
            system_message: System message (primarily for new sessions).
            provider_name: Override default provider.
            model_name: Override provider's default model.
            stream: True for streaming response.
            save_session: True to save turn to persistent session (if session_id provided).
            active_context_item_ids: List of IDs of ContextItems from the session's
                                     `context_items` that are currently enabled by the user.
            enable_rag: True for standard RAG based on the user's message.
            rag_retrieval_k: Number of documents for standard RAG.
            rag_collection_name: Collection for standard RAG.
            **provider_kwargs: Additional arguments for the provider.

        Returns:
            Full response string or async generator of text chunks.
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', provider='{provider_actual_name}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}, "
            f"active_user_items_count={len(active_context_item_ids) if active_context_item_ids else 0}"
        )

        try:
            chat_session: ChatSession
            if session_id:
                chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
            else: # Stateless chat
                chat_session = ChatSession(id=f"temp_stateless_{uuid.uuid4().hex[:8]}")
                if system_message:
                    chat_session.add_message(message_content=system_message, role=Role.SYSTEM)

            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}'")

            context_payload: List[Message] = await self._context_manager.prepare_context(
                session=chat_session,
                provider_name=provider_actual_name,
                model_name=target_model,
                active_context_item_ids=active_context_item_ids,
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection
            )
            logger.info(f"Prepared context with {len(context_payload)} messages for model '{target_model}'.")

            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload, model=target_model, stream=stream, **provider_kwargs
            )

            if stream:
                logger.debug(f"Processing stream response from provider '{provider_actual_name}'")
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # type: ignore[assignment]
                # Pass chat_session and save_session flag to the wrapper
                return self._stream_response_wrapper(
                    provider_stream, active_provider, chat_session, (save_session and session_id is not None)
                )
            else:
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(provider_actual_name, "Invalid response format (expected dict).")
                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)
                if full_response_content is None:
                     full_response_content = f"Response received, but content extraction failed. Data: {str(response_data)[:200]}..."
                else:
                     logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                if save_session and session_id: # Only save if session_id was provided
                    assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                    logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")
                    await self._session_manager.save_session(chat_session)
                return full_response_content

        except (SessionNotFoundError, StorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError) as e:
             logger.error(f"Chat failed: {e}")
             raise
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             raise LLMCoreError(f"Chat execution failed: {e}")

    async def _stream_response_wrapper(
        self,
        provider_stream: AsyncGenerator[Any, None],
        provider: BaseProvider,
        session: ChatSession, # Now accepts ChatSession
        do_save_session: bool # Explicit flag for saving
    ) -> AsyncGenerator[str, None]:
        """Wraps provider's stream, yields text chunks, and saves session if requested."""
        full_response_content = ""
        error_occurred = False
        provider_name = provider.get_name()
        # session_id_for_saving = session.id if do_save_session else None # Use session.id

        try:
            async for chunk in provider_stream:
                chunk_dict: Optional[Dict[str, Any]] = None
                if isinstance(chunk, dict): chunk_dict = chunk
                elif OllamaChatResponse and isinstance(chunk, OllamaChatResponse):
                    try: chunk_dict = chunk.model_dump()
                    except Exception as dump_err: logger.warning(f"Could not dump Ollama stream object: {dump_err}. Chunk: {chunk}"); continue
                else: logger.warning(f"Received non-dict/non-OllamaResponse chunk: {type(chunk)} - {chunk}"); continue
                if not chunk_dict: continue

                text_delta = self._extract_delta_content(chunk_dict, provider)
                error_message = chunk_dict.get('error')
                finish_reason = chunk_dict.get('finish_reason')

                if text_delta: full_response_content += text_delta; yield text_delta
                if error_message: logger.error(f"Error during stream: {error_message}"); raise ProviderError(provider_name, error_message)
                if finish_reason and finish_reason not in ["stop", "length", None]: logger.warning(f"Stream stopped due to reason: {finish_reason}")
        except Exception as e:
            error_occurred = True
            logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True)
            if isinstance(e, ProviderError): raise
            raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if do_save_session: # Check the explicit flag
                if full_response_content or not error_occurred:
                    assistant_msg = session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                    logger.debug(f"Assistant message '{assistant_msg.id}' (len: {len(full_response_content)}) added to session '{session.id}' after stream.")
                else:
                     logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response.")
                try:
                    await self._session_manager.save_session(session)
                except Exception as save_e:
                     logger.error(f"Failed to save session {session.id} after stream: {save_e}", exc_info=True)

    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts text delta from stream chunk (implementation same as before)."""
        provider_name = provider.get_name()
        text_delta = ""
        try:
            if provider_name == "openai":
                choices = chunk.get('choices', [])
                if choices and choices[0].get('delta'): text_delta = choices[0]['delta'].get('content', '') or ""
            elif provider_name == "anthropic":
                type_val = chunk.get("type")
                if type_val == "content_block_delta" and chunk.get('delta', {}).get('type') == "text_delta":
                    text_delta = chunk.get('delta', {}).get('text', "") or ""
            elif provider_name == "ollama":
                message_chunk = chunk.get('message', {})
                if message_chunk: text_delta = message_chunk.get('content', '') or ""
                elif 'response' in chunk: text_delta = chunk.get('response', '') or ""
            elif provider_name == "gemini":
                 choices = chunk.get('choices', [])
                 if choices and choices[0].get('delta'): text_delta = choices[0]['delta'].get('content', '') or ""
                 elif chunk.get('message', {}).get('content'): text_delta = chunk.get('message', {}).get('content', '') or ""
        except Exception as e: logger.warning(f"Error extracting delta from {provider_name} chunk: {e}. Chunk: {str(chunk)[:200]}"); text_delta = ""
        return text_delta or ""

    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> Optional[str]:
        """Extracts full response content (implementation same as before)."""
        provider_name = provider.get_name()
        full_response_content: Optional[str] = None
        try:
            if provider_name == "openai":
                choices = response_data.get('choices', [])
                if choices and choices[0].get('message'): full_response_content = choices[0]['message'].get('content')
            elif provider_name == "anthropic":
                content_blocks = response_data.get('content', [])
                if content_blocks and content_blocks[0].get("type") == "text": full_response_content = content_blocks[0].get("text")
            elif provider_name == "ollama":
                message_part = response_data.get('message', {})
                if message_part: full_response_content = message_part.get('content')
                elif 'response' in response_data: full_response_content = response_data.get('response')
            elif provider_name == "gemini":
                 choices = response_data.get('choices', [])
                 if choices and choices[0].get('message'): full_response_content = choices[0]['message'].get('content')
            if full_response_content is None and response_data:
                 is_extraction_failure = True # Simplified check
                 if not is_extraction_failure: full_response_content = ""
                 else: logger.warning(f"Could not extract content from {provider_name} response: {str(response_data)[:200]}"); return None
            return str(full_response_content) if full_response_content is not None else None
        except Exception as e: logger.error(f"Error extracting full content from {provider_name}: {e}. Response: {str(response_data)[:200]}", exc_info=True); return None

    # --- Session Management Methods (get_session, list_sessions, delete_session remain same) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        logger.debug(f"LLMCore.get_session for ID: {session_id}")
        try: return await self._session_manager.load_or_create_session(session_id=session_id)
        except SessionNotFoundError: logger.warning(f"Session ID '{session_id}' not found."); return None
        except StorageError as e: logger.error(f"Storage error getting session '{session_id}': {e}"); raise

    async def list_sessions(self) -> List[Dict[str, Any]]:
        logger.debug("LLMCore.list_sessions called.")
        try: session_storage = self._storage_manager.get_session_storage(); return await session_storage.list_sessions()
        except StorageError as e: logger.error(f"Storage error listing sessions: {e}"); raise

    async def delete_session(self, session_id: str) -> bool:
        logger.debug(f"LLMCore.delete_session for ID: {session_id}")
        try: session_storage = self._storage_manager.get_session_storage(); return await session_storage.delete_session(session_id)
        except StorageError as e: logger.error(f"Storage error deleting session '{session_id}': {e}"); raise

    # --- RAG / Vector Store Management Methods (remain same) ---
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict]=None, doc_id: Optional[str]=None, collection_name: Optional[str]=None) -> str:
        logger.debug(f"Adding document to vector store (Collection: {collection_name or 'default'})...")
        try:
            embedding = await self._embedding_manager.generate_embedding(content)
            doc_metadata = metadata if metadata is not None else {}
            doc = ContextDocument(id=doc_id if doc_id else str(uuid.uuid4()), content=content, embedding=embedding, metadata=doc_metadata)
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents([doc], collection_name=collection_name)
            if not added_ids: raise VectorStorageError("Failed to add document, no ID returned.")
            logger.info(f"Document '{added_ids[0]}' added to vector store collection '{collection_name or 'default'}'.")
            return added_ids[0]
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to add document: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding document: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str]=None) -> List[str]:
        if not documents: return []
        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            contents = [doc_data["content"] for doc_data in documents if isinstance(doc_data.get("content"), str)]
            if len(contents) != len(documents): raise ValueError("All documents must have string 'content'.")
            embeddings = await self._embedding_manager.generate_embeddings(contents)
            if len(embeddings) != len(documents): raise EmbeddingError("Mismatch between texts and generated embeddings.")
            docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata",{})) for d, emb in zip(documents, embeddings)]
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)
            logger.info(f"Batch of {len(added_ids)} docs added/updated in collection '{collection_name or 'default'}'.")
            return added_ids
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e: logger.error(f"Failed to add documents batch: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str]=None, filter_metadata: Optional[Dict]=None) -> List[ContextDocument]:
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
        if not document_ids: logger.warning("delete_documents_from_vector_store called with empty ID list."); return True
        logger.debug(f"Deleting {len(document_ids)} documents from vector store (Collection: {collection_name or 'default'})...")
        try:
            vector_storage = self._storage_manager.get_vector_storage()
            success = await vector_storage.delete_documents(document_ids=document_ids, collection_name=collection_name)
            logger.info(f"Deletion attempt for {len(document_ids)} documents completed (Success: {success}).")
            return success
        except (VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to delete documents: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error deleting documents: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    # --- New Context Item Management Methods ---
    async def add_text_context_item(
        self,
        session_id: str,
        content: str,
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextItem:
        """Adds a user-provided text snippet to the session's context pool."""
        if not session_id: raise ValueError("session_id is required to add a context item.")
        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id, message="Session not found to add context item.")

        item = ContextItem(
            id=item_id or str(uuid.uuid4()),
            type=ContextItemType.USER_TEXT,
            source_id=item_id or "user_text_snippet", # Or a more descriptive source
            content=content,
            metadata=metadata or {}
        )
        # Tokenize to store token count (optional, but good for context manager)
        try:
            provider = self._provider_manager.get_default_provider() # Or use a specific/configured tokenizer provider
            item.tokens = await provider.count_tokens(content, model=provider.default_model)
        except Exception as e:
            logger.warning(f"Could not count tokens for user text item '{item.id}': {e}. Tokens set to None.")
            item.tokens = None

        session.add_context_item(item)
        await self._session_manager.save_session(session)
        logger.info(f"Added text context item '{item.id}' to session '{session_id}'.")
        return item

    async def add_file_context_item(
        self,
        session_id: str,
        file_path: str,
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextItem:
        """Adds content from a user-provided file to the session's context pool."""
        if not session_id: raise ValueError("session_id is required.")
        if not pathlib.Path(file_path).is_file(): raise FileNotFoundError(f"File not found: {file_path}")

        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id, message="Session not found to add file context item.")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise LLMCoreError(f"Failed to read file content from {file_path}: {e}")

        file_metadata = metadata or {}
        file_metadata.setdefault("filename", pathlib.Path(file_path).name)
        file_metadata.setdefault("original_path", file_path)

        item = ContextItem(
            id=item_id or str(uuid.uuid4()),
            type=ContextItemType.USER_FILE,
            source_id=file_path,
            content=content,
            metadata=file_metadata
        )
        try:
            provider = self._provider_manager.get_default_provider()
            item.tokens = await provider.count_tokens(content, model=provider.default_model)
        except Exception as e:
            logger.warning(f"Could not count tokens for file item '{item.id}': {e}. Tokens set to None.")
            item.tokens = None

        session.add_context_item(item)
        await self._session_manager.save_session(session)
        logger.info(f"Added file context item '{item.id}' (from {file_path}) to session '{session_id}'.")
        return item

    async def update_context_item(
        self,
        session_id: str,
        item_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
        # 'enabled' status is managed by client and passed via active_context_item_ids to chat()
    ) -> ContextItem:
        """Updates content or metadata of an existing user-added context item."""
        if not session_id: raise ValueError("session_id is required.")
        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id)

        item_to_update = session.get_context_item(item_id)
        if not item_to_update:
            raise LLMCoreError(f"Context item '{item_id}' not found in session '{session_id}'.")

        updated = False
        if content is not None:
            item_to_update.content = content
            try: # Recalculate tokens
                provider = self._provider_manager.get_default_provider()
                item_to_update.tokens = await provider.count_tokens(content, model=provider.default_model)
            except Exception: item_to_update.tokens = None
            updated = True
        if metadata is not None:
            item_to_update.metadata.update(metadata) # Merge new metadata
            updated = True

        if updated:
            item_to_update.timestamp = datetime.now(timezone.utc)
            await self._session_manager.save_session(session)
            logger.info(f"Updated context item '{item_id}' in session '{session_id}'.")
        return item_to_update

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        """Removes a user-added context item from the session's pool."""
        if not session_id: raise ValueError("session_id is required.")
        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id)

        removed = session.remove_context_item(item_id)
        if removed:
            await self._session_manager.save_session(session)
            logger.info(f"Removed context item '{item_id}' from session '{session_id}'.")
        else:
            logger.warning(f"Context item '{item_id}' not found in session '{session_id}' for removal.")
        return removed

    async def get_session_context_items(self, session_id: str) -> List[ContextItem]:
        """Lists all user-added context items for a given session."""
        if not session_id: raise ValueError("session_id is required.")
        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id)
        return session.context_items

    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        """Retrieves a specific user-added context item from a session."""
        if not session_id: raise ValueError("session_id is required.")
        session = await self._session_manager.load_or_create_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id)
        return session.get_context_item(item_id)

    # --- Provider Info Methods (get_available_providers, get_models_for_provider remain same) ---
    def get_available_providers(self) -> List[str]:
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        logger.debug(f"LLMCore.get_models_for_provider for: {provider_name}")
        try: provider = self._provider_manager.get_provider(provider_name); return provider.get_available_models()
        except (ConfigError, ProviderError) as e: logger.error(f"Error getting models for provider '{provider_name}': {e}"); raise
        except Exception as e: logger.error(f"Unexpected error for provider '{provider_name}': {e}", exc_info=True); raise ProviderError(provider_name, f"Failed to retrieve models: {e}")

    # --- Utility / Cleanup (close, __aenter__, __aexit__ remain same) ---
    async def close(self):
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = [
            self._provider_manager.close_providers(),
            self._storage_manager.close_storages(),
            self._embedding_manager.close(),
        ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception): logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)
        logger.info("LLMCore resources cleanup complete.")

    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc_val, exc_tb): await self.close()
