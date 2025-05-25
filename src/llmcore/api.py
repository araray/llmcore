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

import aiofiles

from .context.manager import ContextManager
from .embedding.manager import EmbeddingManager
from .exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
from .models import (ChatSession, ContextDocument, ContextItem,
                     ContextItemType, Message, Role, ContextPreparationDetails,
                     ContextPreset, ContextPresetItem) # Added ContextPreset, ContextPresetItem
from .providers.base import BaseProvider
from .providers.manager import ProviderManager
from .sessions.manager import SessionManager
from .storage.manager import StorageManager

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
    from ollama import ChatResponse as OllamaChatResponse  # type: ignore
except ImportError:
    OllamaChatResponse = None # type: ignore [assignment]


logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.
    Provides methods for chat completions, session management, RAG, context pool management,
    context preview, and context preset management.
    Instantiate using 'instance = await LLMCore.create(...)'.
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
        """Private constructor. Use `create` classmethod for async initialization."""
        self._transient_last_interaction_info_cache = {}
        self._transient_sessions_cache = {}
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
        (Implementation unchanged from previous versions)
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        try:
            from confy.loader import Config as ActualConfyConfig
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required for loading default_config.toml.")
            default_config_dict = {}
            try:
                if hasattr(importlib.resources, 'files'):
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: default_config_dict = tomllib.load(f)
                else:
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8') # type: ignore
                    default_config_dict = tomllib.loads(default_config_content) # type: ignore
            except Exception as e: raise ConfigError(f"Failed to load default configuration: {e}")
            instance.config = ActualConfyConfig(defaults=default_config_dict, file_path=config_file_path, prefix=env_prefix, overrides_dict=config_overrides, mandatory=[])
            logger.info("confy configuration loaded successfully.")
        except ImportError as e: raise ConfigError(f"Configuration dependency missing: {e}")
        except ConfigError: raise
        except Exception as e: raise ConfigError(f"Configuration initialization failed: {e}")

        try: instance._provider_manager = ProviderManager(instance.config); logger.info("ProviderManager initialized.")
        except (ConfigError, ProviderError) as e: logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"ProviderManager initialization failed: {e}")
        try:
            instance._storage_manager = StorageManager(instance.config); await instance._storage_manager.initialize_storages()
            logger.info("StorageManager initialized.")
        except (ConfigError, StorageError) as e: logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"StorageManager initialization failed: {e}")
        try:
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized.")
        except StorageError as e: raise LLMCoreError(f"SessionManager init failed due to storage: {e}")
        except Exception as e: raise LLMCoreError(f"SessionManager initialization failed: {e}")
        try:
            instance._embedding_manager = EmbeddingManager(instance.config); await instance._embedding_manager.initialize_embedding_model()
            logger.info("EmbeddingManager initialized.")
        except (ConfigError, EmbeddingError) as e: logger.error(f"Failed to initialize EmbeddingManager: {e}", exc_info=True); raise
        except Exception as e: raise LLMCoreError(f"EmbeddingManager initialization failed: {e}")
        try:
            instance._context_manager = ContextManager(config=instance.config, provider_manager=instance._provider_manager, storage_manager=instance._storage_manager, embedding_manager=instance._embedding_manager)
            logger.info("ContextManager initialized.")
        except Exception as e: raise LLMCoreError(f"ContextManager initialization failed: {e}")
        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    async def chat(
        self, message: str, *, session_id: Optional[str] = None, system_message: Optional[str] = None,
        provider_name: Optional[str] = None, model_name: Optional[str] = None, stream: bool = False,
        save_session: bool = True, active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False, rag_retrieval_k: Optional[int] = None, rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None, prompt_template_values: Optional[Dict[str, str]] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, user-added context, RAG, and explicitly staged items.
        (Implementation largely unchanged, uses ContextPreparationDetails internally)
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model: raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}'.")
        logger.debug(f"LLMCore.chat: session='{session_id}', provider='{provider_actual_name}', model='{target_model}', RAG={enable_rag}, staged_items={len(explicitly_staged_items) if explicitly_staged_items else 0}")
        try:
            chat_session: ChatSession
            if session_id:
                if not save_session:
                    if session_id in self._transient_sessions_cache: chat_session = self._transient_sessions_cache[session_id]
                    else:
                        chat_session = ChatSession(id=session_id)
                        if system_message: chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
                        self._transient_sessions_cache[session_id] = chat_session
                else: chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
            else:
                chat_session = ChatSession(id=f"temp_stateless_{uuid.uuid4().hex[:8]}")
                if system_message: chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=chat_session, provider_name=provider_actual_name, model_name=target_model,
                active_context_item_ids=active_context_item_ids, explicitly_staged_items=explicitly_staged_items,
                rag_enabled=enable_rag, rag_k=rag_retrieval_k, rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter, prompt_template_values=prompt_template_values
            )
            context_payload = context_details.prepared_messages
            logger.info(f"Prepared context with {len(context_payload)} messages ({context_details.final_token_count} tokens) for model '{target_model}'.")
            if session_id: self._transient_last_interaction_info_cache[session_id] = context_details
            response_data_or_generator = await active_provider.chat_completion(context=context_payload, model=target_model, stream=stream, **provider_kwargs)
            if stream:
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # type: ignore
                return self._stream_response_wrapper(provider_stream, active_provider, chat_session, (save_session and session_id is not None))
            else:
                if not isinstance(response_data_or_generator, dict): raise ProviderError(provider_actual_name, "Invalid response format (expected dict).")
                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)
                if full_response_content is None: full_response_content = f"Response received, but content extraction failed. Data: {str(response_data)[:200]}..."
                chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                if save_session and session_id: await self._session_manager.save_session(chat_session)
                return full_response_content
        except (SessionNotFoundError, StorageError, ProviderError, ContextLengthError, ConfigError, EmbeddingError, VectorStorageError) as e:
             logger.error(f"Chat failed: {e}");
             if session_id and not save_session and session_id in self._transient_sessions_cache: del self._transient_sessions_cache[session_id]
             raise
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             if session_id and not save_session and session_id in self._transient_sessions_cache: del self._transient_sessions_cache[session_id]
             raise LLMCoreError(f"Chat execution failed: {e}")

    async def preview_context_for_chat(
        self, current_user_query: str, *, session_id: Optional[str] = None, system_message: Optional[str] = None,
        provider_name: Optional[str] = None, model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False, rag_retrieval_k: Optional[int] = None, rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None, prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Previews the context that would be prepared for an LLM chat call.
        (Implementation largely unchanged, returns dict representation of ContextPreparationDetails)
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model: raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}' for preview.")
        logger.info(f"Previewing context for query: '{current_user_query[:50]}...' (Provider: {provider_actual_name}, Model: {target_model}, Session: {session_id})")
        preview_session: ChatSession
        if session_id:
            try:
                loaded_session = await self._session_manager.load_or_create_session(session_id, system_message_if_new=None)
                preview_session = loaded_session
                if system_message: # If system_message provided for preview, ensure it's primary
                    existing_sys_msg_idx = next((i for i, msg in enumerate(preview_session.messages) if msg.role == Role.SYSTEM), -1)
                    if existing_sys_msg_idx != -1: preview_session.messages[existing_sys_msg_idx].content = system_message; preview_session.messages[existing_sys_msg_idx].tokens = None
                    else: preview_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=preview_session.id))
            except Exception as e_load:
                logger.warning(f"Could not load session '{session_id}' for preview: {e_load}. Simulating new temp session.")
                preview_session = ChatSession(id=session_id or f"preview_temp_{uuid.uuid4().hex[:8]}")
                if system_message: preview_session.add_message(message_content=system_message, role=Role.SYSTEM)
        else:
            preview_session = ChatSession(id=f"preview_stateless_{uuid.uuid4().hex[:8]}")
            if system_message: preview_session.add_message(message_content=system_message, role=Role.SYSTEM)
        preview_session.add_message(message_content=current_user_query, role=Role.USER)
        try:
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=preview_session, provider_name=provider_actual_name, model_name=target_model,
                active_context_item_ids=active_context_item_ids, explicitly_staged_items=explicitly_staged_items,
                rag_enabled=enable_rag, rag_k=rag_retrieval_k, rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter, prompt_template_values=prompt_template_values
            )
            return context_details.model_dump(mode="json")
        except Exception as e: logger.error(f"Error during context preview preparation: {e}", exc_info=True); raise LLMCoreError(f"Context preview failed: {e}")

    async def _stream_response_wrapper(
        self, provider_stream: AsyncGenerator[Any, None], provider: BaseProvider,
        session: ChatSession, do_save_session: bool
    ) -> AsyncGenerator[str, None]:
        """Wraps provider's stream. (Implementation unchanged)"""
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
                text_delta = self._extract_delta_content(chunk_dict, provider); error_message = chunk_dict.get('error'); finish_reason = None
                if 'choices' in chunk_dict and chunk_dict['choices'] and isinstance(chunk_dict['choices'][0], dict): finish_reason = chunk_dict['choices'][0].get('finish_reason')
                elif 'finish_reason' in chunk_dict: finish_reason = chunk_dict.get('finish_reason')
                elif chunk_dict.get("type") == "message_delta" and chunk_dict.get("delta"): finish_reason = chunk_dict["delta"].get("stop_reason")
                if text_delta: full_response_content += text_delta; yield text_delta
                if error_message: logger.error(f"Error during stream: {error_message}"); raise ProviderError(provider_name, error_message)
                if finish_reason and finish_reason not in ["stop", "length", None, "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_USE", "stop_token", "max_tokens", "NOT_SET", "OTHER"]: logger.warning(f"Stream stopped due to reason: {finish_reason}")
        except Exception as e: error_occurred = True; logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True);
            if isinstance(e, ProviderError): raise; raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if full_response_content or not error_occurred: session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
            else: logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response.")
            if do_save_session:
                try: await self._session_manager.save_session(session); logger.debug(f"Session '{session.id}' persisted after stream.")
                except Exception as save_e: logger.error(f"Failed to save session {session.id} after stream: {save_e}", exc_info=True)

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

    # --- Session Management Methods (largely unchanged) ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a session. (Implementation unchanged)"""
        logger.debug(f"LLMCore.get_session for ID: {session_id}")
        if session_id in self._transient_sessions_cache: return self._transient_sessions_cache[session_id]
        try: return await self._session_manager.load_or_create_session(session_id=session_id, system_message=None)
        except SessionNotFoundError: logger.warning(f"Session ID '{session_id}' not found."); return None
        except StorageError as e: logger.error(f"Storage error getting session '{session_id}': {e}"); raise
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists metadata of persistent sessions. (Implementation unchanged)"""
        logger.debug("LLMCore.list_sessions called.");
        try: return await self._storage_manager.get_session_storage().list_sessions()
        except StorageError as e: logger.error(f"Storage error listing sessions: {e}"); raise
    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session. (Implementation unchanged)"""
        logger.debug(f"LLMCore.delete_session for ID: {session_id}")
        was_in_transient = self._transient_sessions_cache.pop(session_id, None) is not None
        self._transient_last_interaction_info_cache.pop(session_id, None)
        try: deleted_persistent = await self._storage_manager.get_session_storage().delete_session(session_id)
            if deleted_persistent: logger.info(f"Session '{session_id}' deleted from persistent storage.")
            elif not was_in_transient: logger.warning(f"Session '{session_id}' not found for deletion.")
            return deleted_persistent or was_in_transient
        except StorageError as e: logger.error(f"Storage error deleting session '{session_id}': {e}"); raise
        return was_in_transient

    # --- RAG / Vector Store Management Methods (unchanged) ---
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict]=None, doc_id: Optional[str]=None, collection_name: Optional[str]=None) -> str:
        """Adds a single document to the vector store. (Implementation unchanged)"""
        try:
            embedding = await self._embedding_manager.generate_embedding(content)
            doc = ContextDocument(id=doc_id or str(uuid.uuid4()), content=content, embedding=embedding, metadata=metadata or {})
            added_ids = await self._storage_manager.get_vector_storage().add_documents([doc], collection_name=collection_name)
            if not added_ids: raise VectorStorageError("Failed to add document, no ID returned.")
            return added_ids[0]
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to add document: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding document: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")
    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str]=None) -> List[str]:
        """Adds multiple documents to the vector store. (Implementation unchanged)"""
        if not documents: return []
        try:
            contents = [d["content"] for d in documents]; embeddings = await self._embedding_manager.generate_embeddings(contents)
            docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata",{})) for d, emb in zip(documents, embeddings)]
            return await self._storage_manager.get_vector_storage().add_documents(docs_to_add, collection_name=collection_name)
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e: logger.error(f"Failed to add documents batch: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")
    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str]=None, filter_metadata: Optional[Dict]=None) -> List[ContextDocument]:
        """Searches the vector store. (Implementation unchanged)"""
        if k <= 0: raise ValueError("'k' must be positive.")
        try:
            query_embedding = await self._embedding_manager.generate_embedding(query)
            return await self._storage_manager.get_vector_storage().similarity_search(query_embedding=query_embedding, k=k, collection_name=collection_name, filter_metadata=filter_metadata)
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to search vector store: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error searching vector store: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")
    async def delete_documents_from_vector_store(self, document_ids: List[str], *, collection_name: Optional[str]=None) -> bool:
        """Deletes documents from the vector store. (Implementation unchanged)"""
        if not document_ids: return True
        try: return await self._storage_manager.get_vector_storage().delete_documents(document_ids=document_ids, collection_name=collection_name)
        except (VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to delete documents: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error deleting documents: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")
    async def list_rag_collections(self) -> List[str]:
        """Lists RAG collections. (Implementation unchanged)"""
        try: return await self._storage_manager.list_vector_collection_names()
        except StorageError as e: logger.error(f"Storage error listing RAG collections: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error listing RAG collections: {e}", exc_info=True); raise LLMCoreError(f"Failed to list RAG collections: {e}")

    # --- User Context Item Management Methods (largely unchanged) ---
    async def add_text_context_item(self, session_id: str, content: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """Adds a text snippet as a context item to a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = self._transient_sessions_cache.get(session_id) or await self._session_manager.load_or_create_session(session_id)
        item_id_actual = item_id or str(uuid.uuid4()); item_metadata = metadata or {}
        if ignore_char_limit: item_metadata['ignore_char_limit'] = True
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_TEXT, source_id=item_id_actual, content=content, metadata=item_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Token count failed for text item '{item.id}': {e}"); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not (session_id in self._transient_sessions_cache): await self._session_manager.save_session(session)
        return item
    async def add_file_context_item(self, session_id: str, file_path: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """Adds file content as a context item to a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        path_obj = pathlib.Path(file_path).expanduser().resolve();
        if not path_obj.is_file(): raise FileNotFoundError(f"File not found: {path_obj}")
        session = self._transient_sessions_cache.get(session_id) or await self._session_manager.load_or_create_session(session_id)
        try:
            async with aiofiles.open(path_obj, "r", encoding="utf-8", errors="ignore") as f: content = await f.read()
        except Exception as e: raise LLMCoreError(f"Failed to read file content from {path_obj}: {e}")
        file_metadata = metadata or {}; file_metadata.setdefault("filename", path_obj.name); file_metadata.setdefault("original_path", str(path_obj))
        if ignore_char_limit: file_metadata['ignore_char_limit'] = True
        item_id_actual = item_id or str(uuid.uuid4())
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_FILE, source_id=str(path_obj), content=content, metadata=file_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Token count failed for file item '{item.id}': {e}"); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not (session_id in self._transient_sessions_cache): await self._session_manager.save_session(session)
        return item
    async def update_context_item(self, session_id: str, item_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        """Updates an existing context item in a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = self._transient_sessions_cache.get(session_id) or await self._session_manager.load_or_create_session(session_id)
        item_to_update = session.get_context_item(item_id)
        if not item_to_update: raise LLMCoreError(f"Context item '{item_id}' not found in session '{session_id}'.")
        updated = False
        if content is not None:
            item_to_update.content = content; item_to_update.is_truncated = False
            try: provider = self._provider_manager.get_default_provider(); item_to_update.tokens = await provider.count_tokens(content, model=provider.default_model); item_to_update.original_tokens = item_to_update.tokens
            except Exception: item_to_update.tokens = None; item_to_update.original_tokens = None
            updated = True
        if metadata is not None:
            existing_ignore_limit = item_to_update.metadata.get('ignore_char_limit')
            item_to_update.metadata.update(metadata)
            if 'ignore_char_limit' not in metadata and existing_ignore_limit is not None: item_to_update.metadata['ignore_char_limit'] = existing_ignore_limit
            updated = True
        if updated: item_to_update.timestamp = datetime.now(timezone.utc)
        if updated and not (session_id in self._transient_sessions_cache): await self._session_manager.save_session(session)
        return item_to_update
    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        """Removes a context item from a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session_obj_to_modify = self._transient_sessions_cache.get(session_id) or await self._session_manager.load_or_create_session(session_id, system_message=None)
        if not session_obj_to_modify: return False
        removed = session_obj_to_modify.remove_context_item(item_id)
        if removed and not (session_id in self._transient_sessions_cache): await self._session_manager.save_session(session_obj_to_modify)
        return removed
    async def get_session_context_items(self, session_id: str) -> List[ContextItem]:
        """Retrieves all context items for a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id, message="Session not found for listing context items.")
        return session.context_items
    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        """Retrieves a specific context item from a session's pool. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        return session.get_context_item(item_id) if session else None
    async def get_last_interaction_context_info(self, session_id: str) -> Optional[ContextPreparationDetails]:
        """Gets cached ContextPreparationDetails from the last interaction. (Implementation unchanged)"""
        if not session_id: return None
        return self._transient_last_interaction_info_cache.get(session_id)
    async def get_last_used_rag_documents(self, session_id: str) -> Optional[List[ContextDocument]]:
        """Gets RAG documents from last cached interaction details. (Implementation unchanged)"""
        context_details = await self.get_last_interaction_context_info(session_id)
        return context_details.rag_documents_used if context_details else None
    async def pin_rag_document_as_context_item( self, session_id: str, original_rag_doc_id: str, custom_item_id: Optional[str] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        """Pins a RAG document from last turn as a new context item. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        if not original_rag_doc_id: raise ValueError("original_rag_doc_id is required.")
        session = self._transient_sessions_cache.get(session_id) or await self._session_manager.load_or_create_session(session_id)
        context_details = await self.get_last_interaction_context_info(session_id)
        last_rag_docs = context_details.rag_documents_used if context_details else None
        if not last_rag_docs: raise LLMCoreError(f"No RAG documents from last turn in cache for session '{session_id}'.")
        doc_to_pin: Optional[ContextDocument] = next((doc for doc in last_rag_docs if doc.id == original_rag_doc_id), None)
        if not doc_to_pin: raise LLMCoreError(f"RAG document ID '{original_rag_doc_id}' not found among last used RAG docs.")
        new_item_id = custom_item_id or str(uuid.uuid4())
        new_item_metadata = {"pinned_from_rag": True, "original_rag_doc_id": doc_to_pin.id, "original_rag_doc_metadata": doc_to_pin.metadata, "pinned_timestamp": datetime.now(timezone.utc).isoformat()}
        if custom_metadata: new_item_metadata.update(custom_metadata)
        pinned_item = ContextItem(id=new_item_id, type=ContextItemType.RAG_SNIPPET, source_id=doc_to_pin.id, content=doc_to_pin.content, metadata=new_item_metadata, timestamp=datetime.now(timezone.utc))
        try: provider = self._provider_manager.get_default_provider(); pinned_item.tokens = await provider.count_tokens(pinned_item.content, model=provider.default_model); pinned_item.original_tokens = pinned_item.tokens
        except Exception as e: logger.warning(f"Token count failed for pinned RAG snippet '{pinned_item.id}': {e}"); pinned_item.tokens = None; pinned_item.original_tokens = None
        session.add_context_item(pinned_item)
        if not (session_id in self._transient_sessions_cache): await self._session_manager.save_session(session)
        return pinned_item

    # --- Provider Info Methods (unchanged) ---
    def get_available_providers(self) -> List[str]:
        """Lists names of loaded provider instances. (Implementation unchanged)"""
        return self._provider_manager.get_available_providers()
    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Lists available models for a provider. (Implementation unchanged)"""
        try: return self._provider_manager.get_provider(provider_name).get_available_models()
        except (ConfigError, ProviderError) as e: logger.error(f"Error getting models for provider '{provider_name}': {e}"); raise
        except Exception as e: logger.error(f"Unexpected error for provider '{provider_name}': {e}", exc_info=True); raise ProviderError(provider_name, f"Failed to retrieve models: {e}")

    # --- New Context Preset Management API Methods ---
    async def save_context_preset(self, preset_name: str, items: List[ContextPresetItem],
                                  description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextPreset:
        """
        Saves or updates a context preset.

        Args:
            preset_name: The unique name for the preset.
            items: A list of `ContextPresetItem` objects to include in the preset.
            description: Optional description for the preset.
            metadata: Optional dictionary for additional preset metadata.

        Returns:
            The saved `ContextPreset` object.

        Raises:
            StorageError: If saving to the backend fails.
            ValueError: If preset_name is invalid.
        """
        logger.info(f"Saving context preset '{preset_name}' with {len(items)} items.")
        # Pydantic model for ContextPreset will validate name internally on creation
        try:
            preset = ContextPreset(
                name=preset_name,
                items=items,
                description=description,
                metadata=metadata or {},
                updated_at=datetime.now(timezone.utc) # Ensure updated_at is fresh
            )
            # If loading an existing to update, created_at should be preserved.
            # This logic might be better handled if we first try to load, then update.
            # For a simple save/upsert, creating a new object with potentially new created_at is okay.
            # If it's an update, the storage layer's ON CONFLICT should handle preserving created_at.

            session_storage = self._storage_manager.get_session_storage()
            await session_storage.save_context_preset(preset)
            logger.info(f"Context preset '{preset.name}' saved successfully.")
            return preset
        except ValueError as ve: # Catch Pydantic validation error for name
            logger.error(f"Invalid preset name '{preset_name}': {ve}")
            raise
        except StorageError as se:
            logger.error(f"Failed to save context preset '{preset_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error saving preset '{preset_name}': {e}")

    async def load_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """
        Loads a context preset by its name.

        Args:
            preset_name: The name of the preset to load.

        Returns:
            The `ContextPreset` object if found, otherwise None.

        Raises:
            StorageError: If loading from the backend fails.
        """
        logger.info(f"Loading context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name)
            if preset:
                logger.info(f"Context preset '{preset_name}' loaded successfully with {len(preset.items)} items.")
            else:
                logger.info(f"Context preset '{preset_name}' not found.")
            return preset
        except StorageError as se:
            logger.error(f"Failed to load context preset '{preset_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error loading preset '{preset_name}': {e}")

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """
        Lists metadata of all available context presets.

        Returns:
            A list of dictionaries, each containing metadata for a preset
            (e.g., name, description, item_count, updated_at).

        Raises:
            StorageError: If listing from the backend fails.
        """
        logger.info("Listing all context presets.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            presets_meta = await session_storage.list_context_presets()
            logger.info(f"Found {len(presets_meta)} context presets.")
            return presets_meta
        except StorageError as se:
            logger.error(f"Failed to list context presets: {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing context presets: {e}", exc_info=True)
            raise StorageError(f"Unexpected error listing presets: {e}")

    async def delete_context_preset(self, preset_name: str) -> bool:
        """
        Deletes a context preset by its name.

        Args:
            preset_name: The name of the preset to delete.

        Returns:
            True if the preset was deleted, False if not found.

        Raises:
            StorageError: If deletion from the backend fails.
        """
        logger.info(f"Deleting context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted = await session_storage.delete_context_preset(preset_name)
            if deleted:
                logger.info(f"Context preset '{preset_name}' deleted successfully.")
            else:
                logger.warning(f"Context preset '{preset_name}' not found for deletion.")
            return deleted
        except StorageError as se:
            logger.error(f"Failed to delete context preset '{preset_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting context preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error deleting preset '{preset_name}': {e}")

    async def update_context_preset_description(self, preset_name: str, new_description: str) -> bool:
        """
        Updates the description of an existing context preset.

        Args:
            preset_name: The name of the preset to update.
            new_description: The new description string.

        Returns:
            True if the preset was found and updated, False otherwise.

        Raises:
            StorageError: If updating the backend fails.
        """
        logger.info(f"Updating description for context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name)
            if not preset:
                logger.warning(f"Context preset '{preset_name}' not found for updating description.")
                return False

            preset.description = new_description
            preset.updated_at = datetime.now(timezone.utc) # Update timestamp
            await session_storage.save_context_preset(preset) # Save the modified preset
            logger.info(f"Description for context preset '{preset_name}' updated successfully.")
            return True
        except StorageError as se:
            logger.error(f"Failed to update description for context preset '{preset_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating description for preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error updating preset description '{preset_name}': {e}")

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """
        Renames an existing context preset.

        Args:
            old_name: The current name of the preset.
            new_name: The new name for the preset.

        Returns:
            True if successful, False if old_name not found or new_name already exists.

        Raises:
            ValueError: If new_name is invalid.
            StorageError: For other storage-related issues.
        """
        logger.info(f"Renaming context preset from '{old_name}' to '{new_name}'.")
        # Basic validation for new_name (Pydantic model on ContextPreset also validates)
        if not new_name or not new_name.strip():
            raise ValueError("New preset name cannot be empty.")
        # Further validation for filesystem/DB key suitability is handled by ContextPreset model
        # and potentially by the storage backend's rename_context_preset implementation.

        try:
            session_storage = self._storage_manager.get_session_storage()
            renamed = await session_storage.rename_context_preset(old_name, new_name)
            if renamed:
                logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}'.")
            else:
                logger.warning(f"Failed to rename context preset '{old_name}' to '{new_name}' (e.g., not found or new name exists).")
            return renamed
        except ValueError: # Re-raise ValueError from storage layer (e.g., invalid new_name)
            raise
        except StorageError as se:
            logger.error(f"Storage error renaming context preset '{old_name}' to '{new_name}': {se}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error renaming preset: {e}")


    # --- Utility / Cleanup (unchanged) ---
    async def close(self):
        """Closes all managed resources. (Implementation unchanged)"""
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = [ self._provider_manager.close_providers(), self._storage_manager.close_storages(), self._embedding_manager.close(), ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception): logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)
        self._transient_last_interaction_info_cache.clear(); self._transient_sessions_cache.clear()
        logger.info("LLMCore resources cleanup complete.")
    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc_val, exc_tb): await self.close()
