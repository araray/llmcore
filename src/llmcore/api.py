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

from .context.manager import ContextManager
from .embedding.manager import EmbeddingManager
from .exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
from .models import (ChatSession, ContextDocument, ContextItem, # Added ContextPreparationDetails
                     ContextItemType, Message, Role, ContextPreparationDetails)
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
    and context preview.
    Instantiate using 'instance = await LLMCore.create(...)'.
    """

    config: ConfyConfig
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _context_manager: ContextManager
    _embedding_manager: EmbeddingManager
    _transient_last_interaction_info_cache: Dict[str, ContextPreparationDetails] # Store full details
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
        (Implementation unchanged from previous versions, focused on loading config and managers)
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        try:
            from confy.loader import Config as ActualConfyConfig
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required for loading default_config.toml.")

            default_config_dict = {}
            try:
                if hasattr(importlib.resources, 'files'): # Python 3.9+
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f:
                        default_config_dict = tomllib.load(f)
                else: # Fallback for Python 3.8 (importlib.resources.read_text)
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8') # type: ignore
                    default_config_dict = tomllib.loads(default_config_content) # type: ignore
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
            await instance._embedding_manager.initialize_embedding_model() # Ensure default model is ready
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
        *, # Force subsequent arguments to be keyword-only
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, user-added context, RAG, and explicitly staged items.
        (Docstring unchanged from previous version, already updated for explicitly_staged_items)
        """
        active_provider = self._provider_manager.get_provider(provider_name)
        provider_actual_name = active_provider.get_name()
        target_model = model_name or active_provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for provider '{provider_actual_name}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', save_session={save_session}, provider='{provider_actual_name}', "
            f"model='{target_model}', stream={stream}, RAG={enable_rag}, "
            f"RAG_filter={rag_metadata_filter}, prompt_values_count={len(prompt_template_values) if prompt_template_values else 0}, "
            f"active_user_items_count={len(active_context_item_ids) if active_context_item_ids else 0}, "
            f"explicitly_staged_items_count={len(explicitly_staged_items) if explicitly_staged_items else 0}"
        )

        try:
            chat_session: ChatSession
            if session_id:
                if not save_session: # Transient session logic
                    if session_id in self._transient_sessions_cache:
                        chat_session = self._transient_sessions_cache[session_id]
                        logger.debug(f"Using transient session '{session_id}' from cache.")
                        has_sys_msg = any(m.role == Role.SYSTEM for m in chat_session.messages)
                        current_sys_msg_content = next((m.content for m in chat_session.messages if m.role == Role.SYSTEM), None)
                        if system_message and (not has_sys_msg or current_sys_msg_content != system_message) :
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM]
                            chat_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=session_id))
                            logger.debug(f"Updated/Added system message to cached transient session '{session_id}'.")
                    else:
                        chat_session = ChatSession(id=session_id)
                        if system_message:
                            chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
                        self._transient_sessions_cache[session_id] = chat_session
                        logger.debug(f"Created new transient session '{session_id}' and cached it.")
                else: # Persistent session
                    chat_session = await self._session_manager.load_or_create_session(session_id, system_message)
            else: # Stateless call
                chat_session = ChatSession(id=f"temp_stateless_{uuid.uuid4().hex[:8]}")
                if system_message:
                    chat_session.add_message(message_content=system_message, role=Role.SYSTEM)

            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}' (is_transient={not save_session and bool(session_id)})")

            # --- Updated: Use ContextPreparationDetails from prepare_context ---
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=chat_session,
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
            context_payload = context_details.prepared_messages
            prepared_context_token_count = context_details.final_token_count
            # rag_documents_used_this_turn is available in context_details.rag_documents_used
            # --- End Update ---

            logger.info(f"Prepared context with {len(context_payload)} messages ({prepared_context_token_count} tokens) for model '{target_model}'.")

            if session_id: # Cache full interaction details
                self._transient_last_interaction_info_cache[session_id] = context_details
                logger.debug(f"Stored full context preparation details for session '{session_id}'.")

            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload, model=target_model, stream=stream, **provider_kwargs
            )

            if stream:
                logger.debug(f"Processing stream response from provider '{provider_actual_name}'")
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # type: ignore
                return self._stream_response_wrapper(
                    provider_stream, active_provider, chat_session, (save_session and session_id is not None)
                )
            else: # Non-streaming
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(provider_actual_name, "Invalid response format (expected dict).")
                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)
                if full_response_content is None:
                     full_response_content = f"Response received, but content extraction failed. Data: {str(response_data)[:200]}..."
                else:
                     logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")

                if save_session and session_id:
                    await self._session_manager.save_session(chat_session)
                return full_response_content

        except (SessionNotFoundError, StorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError) as e:
             logger.error(f"Chat failed: {e}")
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
                 logger.debug(f"Cleared failed transient session '{session_id}' from cache.")
             raise
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
             raise LLMCoreError(f"Chat execution failed: {e}")

    async def preview_context_for_chat(
        self,
        current_user_query: str,
        *,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None, # System message for this preview
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Previews the context that would be prepared for an LLM chat call without making the actual API call.

        This method simulates the context preparation process, including history selection,
        RAG (if enabled), inclusion of user-added and explicitly staged items, and truncation.
        It returns a detailed dictionary of what would be sent.

        Args:
            current_user_query: The user's current message/query for which to preview context.
            session_id: ID of the session to use as a base for history and context items.
                        If None, a temporary session context is simulated.
            system_message: An optional system message to consider for this preview. This will
                            be used as the primary system message if a new session is simulated,
                            or potentially supplement/replace an existing one in a loaded session
                            based on ContextManager's logic if it were a real chat.
            provider_name: Override the default provider for context calculation.
            model_name: Override the default model for the selected provider for context calculation.
            active_context_item_ids: IDs of user-added items from the session's pool to consider.
            explicitly_staged_items: Items to explicitly include in the previewed context.
            enable_rag: Whether to simulate RAG retrieval.
            rag_retrieval_k: Number of documents for RAG.
            rag_collection_name: Vector store collection for RAG.
            rag_metadata_filter: Metadata filter for RAG.
            prompt_template_values: Values for RAG prompt template.

        Returns:
            A dictionary containing detailed information about the prepared context:
            - 'prepared_messages': List of `Message` objects.
            - 'final_token_count': Total tokens for `prepared_messages`.
            - 'max_tokens_for_model': Model's context limit.
            - 'rag_documents_used': List of `ContextDocument` if RAG was used.
            - 'rendered_rag_template_content': The RAG-rendered query, if applicable.
            - 'truncation_actions_taken': Details of any truncations.

        Raises:
            LLMCoreError, ConfigError, etc., if context preparation itself fails.
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
            # Try to load existing session, or create a temporary one if it doesn't exist
            # This ensures we have a ChatSession object to work with
            try:
                loaded_session = await self._session_manager.load_or_create_session(session_id, system_message_if_new=None) # Don't pass system_message here yet
                preview_session = loaded_session # Use the loaded or newly created persistent-candidate session

                # If a system_message is provided for preview, and the loaded session has one,
                # we need to decide how to handle it. For preview, let's assume the provided
                # system_message for preview takes precedence or is added if none exists.
                # A more complex strategy could be to merge or replace.
                # For simplicity, if system_message is given, we ensure it's the first system message.
                if system_message:
                    existing_sys_msg_idx = -1
                    for i, msg in enumerate(preview_session.messages):
                        if msg.role == Role.SYSTEM:
                            existing_sys_msg_idx = i
                            break
                    if existing_sys_msg_idx != -1:
                        preview_session.messages[existing_sys_msg_idx].content = system_message
                        preview_session.messages[existing_sys_msg_idx].tokens = None # Force re-tokenization
                    else:
                        preview_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=preview_session.id))

            except Exception as e_load: # Catch potential storage errors
                logger.warning(f"Could not load session '{session_id}' for preview: {e_load}. Simulating with a new temporary session.")
                preview_session = ChatSession(id=session_id or f"preview_temp_{uuid.uuid4().hex[:8]}")
                if system_message: # Add system message if provided for this new temp session
                    preview_session.add_message(message_content=system_message, role=Role.SYSTEM)
        else: # No session_id, create a fully temporary session for preview
            preview_session = ChatSession(id=f"preview_stateless_{uuid.uuid4().hex[:8]}")
            if system_message:
                preview_session.add_message(message_content=system_message, role=Role.SYSTEM)

        # Add the current_user_query as the latest user message to the preview_session
        # This is crucial as prepare_context expects the query to be part of the session's messages.
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
            raise LLMCoreError(f"Context preview failed: {e}")


    async def _stream_response_wrapper(
        self,
        provider_stream: AsyncGenerator[Any, None],
        provider: BaseProvider,
        session: ChatSession,
        do_save_session: bool
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider's stream, yields text chunks, and handles session saving.
        (Implementation unchanged from previous versions)
        """
        full_response_content = ""
        error_occurred = False
        provider_name = provider.get_name()

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
            if isinstance(e, ProviderError): raise
            raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if full_response_content or not error_occurred:
                assistant_msg = session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' (len: {len(full_response_content)}) added to session '{session.id}' (in-memory) after stream.")
            else:
                 logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response and error flag.")
            if do_save_session:
                try:
                    await self._session_manager.save_session(session)
                    logger.debug(f"Session '{session.id}' persisted after stream.")
                except Exception as save_e:
                     logger.error(f"Failed to save session {session.id} to persistent storage after stream: {save_e}", exc_info=True)

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

    # --- Methods below are largely unchanged but shown for completeness ---
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieves a session. (Implementation unchanged)"""
        logger.debug(f"LLMCore.get_session for ID: {session_id}")
        if session_id in self._transient_sessions_cache:
            logger.debug(f"Returning session '{session_id}' from transient cache.")
            return self._transient_sessions_cache[session_id]
        try:
            return await self._session_manager.load_or_create_session(session_id=session_id, system_message=None)
        except SessionNotFoundError:
            logger.warning(f"Session ID '{session_id}' not found by SessionManager.")
            return None
        except StorageError as e: logger.error(f"Storage error getting session '{session_id}': {e}"); raise

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists metadata of all persistent sessions. (Implementation unchanged)"""
        logger.debug("LLMCore.list_sessions called (persistent sessions only).")
        try:
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.list_sessions()
        except StorageError as e: logger.error(f"Storage error listing sessions: {e}"); raise

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session. (Implementation unchanged)"""
        logger.debug(f"LLMCore.delete_session for ID: {session_id}")
        was_in_transient = self._transient_sessions_cache.pop(session_id, None) is not None
        if was_in_transient: logger.debug(f"Removed session '{session_id}' from transient cache.")
        self._transient_last_interaction_info_cache.pop(session_id, None)
        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted_persistent = await session_storage.delete_session(session_id)
            if deleted_persistent: logger.info(f"Session '{session_id}' deleted from persistent storage.")
            elif not was_in_transient: logger.warning(f"Session '{session_id}' not found for deletion in persistent storage.")
            return deleted_persistent or was_in_transient
        except StorageError as e: logger.error(f"Storage error deleting session '{session_id}': {e}"); raise
        return was_in_transient

    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict]=None, doc_id: Optional[str]=None, collection_name: Optional[str]=None) -> str:
        """Adds a single document to the vector store. (Implementation unchanged)"""
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
        """Adds multiple documents to the vector store. (Implementation unchanged)"""
        if not documents: return []
        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            contents = [doc_data["content"] for doc_data in documents if isinstance(doc_data.get("content"), str)]
            if len(contents) != len(documents): raise ValueError("All documents must have string 'content'.")
            embeddings = await self._embedding_manager.generate_embeddings(contents)
            if len(embeddings) != len(documents): raise EmbeddingError("Mismatch between texts and generated embeddings.")
            vector_storage = self._storage_manager.get_vector_storage()
            resolved_collection_name = collection_name or vector_storage._default_collection_name # type: ignore
            docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata",{})) for d, emb in zip(documents, embeddings)]
            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)
            logger.info(f"Batch of {len(added_ids)} docs added/updated in collection '{resolved_collection_name}'.")
            return added_ids
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e: logger.error(f"Failed to add documents batch: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str]=None, filter_metadata: Optional[Dict]=None) -> List[ContextDocument]:
        """Searches the vector store for relevant documents. (Implementation unchanged)"""
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
        """Deletes documents from the vector store by their IDs. (Implementation unchanged)"""
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
        """Lists the names of all available RAG collections. (Implementation unchanged)"""
        logger.debug("LLMCore.list_rag_collections called.")
        try: return await self._storage_manager.list_vector_collection_names()
        except StorageError as e: logger.error(f"Storage error listing RAG collections: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error listing RAG collections: {e}", exc_info=True); raise LLMCoreError(f"Failed to list RAG collections: {e}")

    async def add_text_context_item(self, session_id: str, content: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """Adds a text snippet as a context item to a session. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required to add a context item.")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
        item_id_actual = item_id or str(uuid.uuid4()); item_metadata = metadata or {}
        if ignore_char_limit: item_metadata['ignore_char_limit'] = True
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_TEXT, source_id=item_id_actual, content=content, metadata=item_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for user text item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Added text context item '{item.id}' to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def add_file_context_item(self, session_id: str, file_path: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        """Adds file content as a context item to a session. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        path_obj = pathlib.Path(file_path).expanduser().resolve()
        if not path_obj.is_file(): raise FileNotFoundError(f"File not found: {path_obj}")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
        try:
            async with aiofiles.open(path_obj, "r", encoding="utf-8", errors="ignore") as f: content = await f.read()
        except Exception as e: raise LLMCoreError(f"Failed to read file content from {path_obj}: {e}")
        file_metadata = metadata or {}; file_metadata.setdefault("filename", path_obj.name); file_metadata.setdefault("original_path", str(path_obj))
        if ignore_char_limit: file_metadata['ignore_char_limit'] = True
        item_id_actual = item_id or str(uuid.uuid4())
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_FILE, source_id=str(path_obj), content=content, metadata=file_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for file item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Added file context item '{item.id}' (from {path_obj}) to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def update_context_item(self, session_id: str, item_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        """Updates an existing context item in a session. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
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
        if updated and not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Updated context item '{item_id}' in session '{session_id}' (is_transient={is_transient}).")
        return item_to_update

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        """Removes a context item from a session. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        session_obj_to_modify: Optional[ChatSession] = self._transient_sessions_cache.get(session_id) if is_transient else await self._session_manager.load_or_create_session(session_id, system_message=None)
        if not session_obj_to_modify: logger.warning(f"Session '{session_id}' not found for removing context item."); return False
        removed = session_obj_to_modify.remove_context_item(item_id)
        if removed:
            if not is_transient: await self._session_manager.save_session(session_obj_to_modify)
            logger.info(f"Removed context item '{item_id}' from session '{session_id}' (is_transient={is_transient}).")
        else: logger.warning(f"Context item '{item_id}' not found in session '{session_id}' for removal.")
        return removed

    async def get_session_context_items(self, session_id: str) -> List[ContextItem]:
        """Retrieves all context items for a given session. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id, message="Session not found when trying to list its context items.")
        return session.context_items

    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        """Retrieves a specific context item from a session by its ID. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session: return None
        return session.get_context_item(item_id)

    async def get_last_interaction_context_info(self, session_id: str) -> Optional[ContextPreparationDetails]: # Changed return type
        """
        Gets cached `ContextPreparationDetails` object from the last context preparation for a session.
        """
        if not session_id:
            logger.warning("get_last_interaction_context_info called without session_id.")
            return None
        cached_info = self._transient_last_interaction_info_cache.get(session_id)
        if cached_info: logger.debug(f"Retrieved last interaction context details from transient cache for session '{session_id}'.")
        else: logger.debug(f"No interaction context details found in transient cache for session '{session_id}'.")
        return cached_info

    async def get_last_used_rag_documents(self, session_id: str) -> Optional[List[ContextDocument]]:
        """
        Gets RAG documents used in the last turn for a session from the cached context details.
        """
        context_details = await self.get_last_interaction_context_info(session_id)
        if context_details:
            return context_details.rag_documents_used
        return None

    async def pin_rag_document_as_context_item(
        self, session_id: str, original_rag_doc_id: str,
        custom_item_id: Optional[str] = None, custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ContextItem:
        """Pins a RAG document from the last turn as a new user-added context item. (Implementation unchanged)"""
        if not session_id: raise ValueError("session_id is required.")
        if not original_rag_doc_id: raise ValueError("original_rag_doc_id is required.")
        is_transient_session = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient_session else await self._session_manager.load_or_create_session(session_id)

        context_details = await self.get_last_interaction_context_info(session_id) # Use updated method
        last_rag_docs = context_details.rag_documents_used if context_details else None

        if not last_rag_docs: raise LLMCoreError(f"No RAG documents from last turn in cache for session '{session_id}'. Cannot pin.")
        doc_to_pin: Optional[ContextDocument] = next((doc for doc in last_rag_docs if doc.id == original_rag_doc_id), None)
        if not doc_to_pin: raise LLMCoreError(f"RAG document ID '{original_rag_doc_id}' not found among last used RAG docs for session '{session_id}'.")

        new_item_id = custom_item_id or str(uuid.uuid4())
        new_item_metadata = {
            "pinned_from_rag": True, "original_rag_doc_id": doc_to_pin.id,
            "original_rag_doc_metadata": doc_to_pin.metadata, # type: ignore
            "pinned_timestamp": datetime.now(timezone.utc).isoformat()
        }
        if custom_metadata: new_item_metadata.update(custom_metadata)
        pinned_item = ContextItem(id=new_item_id, type=ContextItemType.RAG_SNIPPET, source_id=doc_to_pin.id, content=doc_to_pin.content, metadata=new_item_metadata, timestamp=datetime.now(timezone.utc))
        try:
            provider = self._provider_manager.get_default_provider()
            pinned_item.tokens = await provider.count_tokens(pinned_item.content, model=provider.default_model)
            pinned_item.original_tokens = pinned_item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for pinned RAG snippet '{pinned_item.id}': {e}. Tokens set to None."); pinned_item.tokens = None; pinned_item.original_tokens = None
        session.add_context_item(pinned_item)
        if not is_transient_session: await self._session_manager.save_session(session)
        logger.info(f"Pinned RAG document '{original_rag_doc_id}' as new context item '{pinned_item.id}' in session '{session_id}' (is_transient={is_transient_session}).")
        return pinned_item

    def get_available_providers(self) -> List[str]:
        """Lists names of all successfully loaded provider instances. (Implementation unchanged)"""
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Lists available models for a specific provider. (Implementation unchanged)"""
        logger.debug(f"LLMCore.get_models_for_provider for: {provider_name}")
        try:
            provider = self._provider_manager.get_provider(provider_name)
            return provider.get_available_models()
        except (ConfigError, ProviderError) as e: logger.error(f"Error getting models for provider '{provider_name}': {e}"); raise
        except Exception as e: logger.error(f"Unexpected error for provider '{provider_name}': {e}", exc_info=True); raise ProviderError(provider_name, f"Failed to retrieve models: {e}")

    async def close(self):
        """Closes all managed resources. (Implementation unchanged)"""
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = [
            self._provider_manager.close_providers(),
            self._storage_manager.close_storages(),
            self._embedding_manager.close(),
        ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception): logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)
        self._transient_last_interaction_info_cache.clear()
        self._transient_sessions_cache.clear()
        logger.info("LLMCore resources cleanup complete.")

    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        await self.close()
