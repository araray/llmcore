# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and user-added context items.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]


from ..providers.manager import ProviderManager
from ..providers.base import BaseProvider
from ..storage.manager import StorageManager
from ..embedding.manager import EmbeddingManager
from ..models import ChatSession, Message, Role as LLMCoreRole, ContextDocument, ContextItem, ContextItemType
from ..exceptions import (
    ContextError, ContextLengthError, ConfigError, ProviderError,
    EmbeddingError, VectorStorageError
)

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results and user-added context items,
    and ensures the final payload adheres to token limits using configurable strategies.
    """

    def __init__(
        self,
        config: ConfyConfig,
        provider_manager: ProviderManager,
        storage_manager: StorageManager,
        embedding_manager: EmbeddingManager
        ):
        """
        Initializes the ContextManager.
        Args:
            config: The main LLMCore configuration object.
            provider_manager: The initialized ProviderManager instance.
            storage_manager: The initialized StorageManager instance.
            embedding_manager: The initialized EmbeddingManager instance.
        """
        self._config = config
        self._provider_manager = provider_manager
        self._storage_manager = storage_manager
        self._embedding_manager = embedding_manager

        cm_config = self._config.get('context_management', {})
        self._reserved_response_tokens: int = cm_config.get('reserved_response_tokens', 500)
        self._history_selection_strategy: str = cm_config.get('history_selection_strategy', 'last_n_tokens')
        self._truncation_priority_order: List[str] = self._parse_truncation_priority(cm_config.get('truncation_priority', 'history,rag,user_items'))
        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)
        self._default_rag_k: int = cm_config.get('rag_retrieval_k', 3)
        self._rag_combination_strategy: str = cm_config.get('rag_combination_strategy', 'prepend_system')
        self._user_retained_messages_count: int = cm_config.get('user_retained_messages_count', 5)
        self._prioritize_user_context_items: bool = cm_config.get('prioritize_user_context_items', True)

        logger.info("ContextManager initialized.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"user_retained_messages={self._user_retained_messages_count}, "
                     f"prioritize_user_items={self._prioritize_user_context_items}, "
                     f"truncation_priority_order={self._truncation_priority_order}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     f"rag_combo_strategy={self._rag_combination_strategy}")
        self._validate_strategies()

    def _parse_truncation_priority(self, priority_str: str) -> List[str]:
        """Parses the truncation_priority string into a list, validating known types."""
        valid_priorities = {"history", "rag", "user_items"}
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_priorities]
        if len(ordered_priorities) != len(priorities):
            logger.warning(f"Invalid items in truncation_priority: '{priority_str}'. Using valid ones: {ordered_priorities}")
        if not ordered_priorities:
            logger.warning("No valid truncation_priority items. Defaulting to 'history,rag,user_items'.")
            return ["history", "rag", "user_items"]
        return ordered_priorities

    def _validate_strategies(self):
        """Validates configured strategies and logs warnings if unsupported."""
        if self._history_selection_strategy not in ['last_n_tokens', 'last_n_messages']:
             logger.warning(f"Unsupported history_selection_strategy '{self._history_selection_strategy}'. Falling back to 'last_n_tokens'.")
             self._history_selection_strategy = 'last_n_tokens'
        elif self._history_selection_strategy == 'last_n_messages':
             logger.warning("'last_n_messages' history strategy is less precise for token limits. Consider 'last_n_tokens'.")

        if self._rag_combination_strategy not in ['prepend_system', 'prepend_user']:
             logger.warning(f"Unsupported rag_combination_strategy '{self._rag_combination_strategy}'. Falling back to 'prepend_system'.")
             self._rag_combination_strategy = 'prepend_system'

    async def _build_history_messages(
        self,
        history_messages_all_non_system: List[Message],
        provider: BaseProvider,
        target_model: str,
        budget: int
    ) -> Tuple[List[Message], int]:
        """
        Builds the chat history part of the context based on user_retained_messages and backfilling.

        Args:
            history_messages_all_non_system: All non-system messages from the session, sorted chronologically.
            provider: The LLM provider instance for token counting.
            target_model: The target model name.
            budget: The token budget available for history messages.

        Returns:
            A tuple: (list of selected history messages, total tokens used by these messages).
        """
        selected_history: List[Message] = []
        current_history_tokens = 0

        if budget <= 0 or not history_messages_all_non_system:
            return selected_history, current_history_tokens

        # Ensure messages are pre-tokenized or tokenize them now
        for msg in history_messages_all_non_system:
            if msg.tokens is None:
                msg.tokens = await provider.count_message_tokens([msg], target_model)

        # Stage 1: User-retained messages (latest N turns)
        retained_for_now: List[Message] = []
        tokens_for_retained = 0
        num_user_messages_retained = 0
        temp_retained_buffer: List[Message] = []

        # Iterate from newest to oldest to pick retained messages
        for msg in reversed(history_messages_all_non_system):
            msg_tokens = msg.tokens or 0 # Should have been tokenized
            if tokens_for_retained + msg_tokens <= budget:
                temp_retained_buffer.append(msg)
                tokens_for_retained += msg_tokens
                if msg.role == LLMCoreRole.USER:
                    num_user_messages_retained += 1
                if self._user_retained_messages_count > 0 and num_user_messages_retained >= self._user_retained_messages_count:
                    break
            else:
                break

        retained_for_now = sorted(temp_retained_buffer, key=lambda m: m.timestamp)
        selected_history.extend(retained_for_now)
        current_history_tokens = tokens_for_retained
        logger.debug(f"Retained {len(retained_for_now)} history messages ({current_history_tokens} tokens) based on user_retained_messages_count ({self._user_retained_messages_count}).")

        # Stage 2: Backfill with older messages if space allows
        remaining_history_budget_for_backfill = budget - current_history_tokens
        if remaining_history_budget_for_backfill > 0:
            ids_in_selected_history = {m.id for m in selected_history}
            older_history_candidates = [
                msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history
            ]
            # Sort older messages: newest of the old first, to prioritize more recent among the "older" ones
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True)

            temp_backfill_history: List[Message] = []
            tokens_for_backfill = 0
            for msg in older_history_candidates:
                msg_tokens = msg.tokens or 0 # Should have been tokenized
                if tokens_for_backfill + msg_tokens <= remaining_history_budget_for_backfill:
                    temp_backfill_history.append(msg)
                    tokens_for_backfill += msg_tokens
                else:
                    break

            # Insert backfilled messages before the retained ones to maintain overall chronological order
            selected_history = sorted(temp_backfill_history, key=lambda m: m.timestamp) + selected_history
            current_history_tokens += tokens_for_backfill
            logger.debug(f"Backfilled {len(temp_backfill_history)} older history messages ({tokens_for_backfill} tokens).")

        return selected_history, current_history_tokens


    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None
    ) -> Tuple[List[Message], Optional[List[ContextDocument]]]:
        """
        Prepares the context payload for the LLM.

        Args:
            session: The current ChatSession.
            provider_name: Name of the LLM provider.
            model_name: Specific model name.
            active_context_item_ids: IDs of user-added context items to include.
            rag_enabled: Whether standard RAG is enabled for this turn.
            rag_k: Number of documents for standard RAG.
            rag_collection: Collection for standard RAG.

        Returns:
            A tuple containing:
            - The list of `Message` objects to be sent to the LLM.
            - An optional list of `ContextDocument` objects that were used for RAG in this turn.
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for context (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' (Provider: {provider.get_name()}, RAG: {rag_enabled}). Session: {session.id}")

        max_context_tokens = provider.get_max_context_length(target_model)
        available_tokens_for_prompt = max_context_tokens - self._reserved_response_tokens
        if available_tokens_for_prompt <= 0:
             raise ContextError(f"Config error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_context_tokens}).")
        logger.debug(f"Max context: {max_context_tokens}, Reserved for response: {self._reserved_response_tokens}, Available for prompt: {available_tokens_for_prompt}")

        # --- Component Gathering & Initial Tokenization ---
        system_messages_hist: List[Message] = []
        active_user_item_messages: List[Message] = []
        rag_query_context_message_obj: Optional[Message] = None
        rag_documents_used_this_turn: Optional[List[ContextDocument]] = None # Store RAG docs used

        # 1. System Messages from history
        system_messages_hist = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
        for msg in system_messages_hist:
            if msg.tokens is None: # Pre-tokenize if not already done
                msg.tokens = await provider.count_message_tokens([msg], target_model)

        # 2. Standard RAG query results (if RAG enabled for the current turn)
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k_val = rag_k if rag_k is not None else self._default_rag_k
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    retrieved_rag_docs = await vector_storage.similarity_search(query_embedding=query_embedding, k=k_val, collection_name=rag_collection)

                    formatted_rag_query_context_str = self._format_rag_docs_for_context(retrieved_rag_docs)
                    if formatted_rag_query_context_str:
                        rag_query_context_message_obj = Message(role=LLMCoreRole.SYSTEM, content=formatted_rag_query_context_str, session_id=session.id, id="rag_query_context_marker")
                        rag_query_context_message_obj.tokens = await provider.count_message_tokens([rag_query_context_message_obj], target_model)
                        rag_documents_used_this_turn = retrieved_rag_docs # Store the docs
                    logger.info(f"Std RAG returned {len(retrieved_rag_docs)} docs; context string tokens: {rag_query_context_message_obj.tokens if rag_query_context_message_obj else 0}.")
                except Exception as e: logger.error(f"Std RAG failed: {e}. No RAG query context will be added.")
            else: logger.warning("Std RAG enabled, but no user message found in history to use as query.")

        # 3. Active User-Added Context Items
        if active_context_item_ids and session.context_items:
            temp_user_items = []
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and (item.type == ContextItemType.USER_TEXT or item.type == ContextItemType.USER_FILE or item.type == ContextItemType.RAG_SNIPPET):
                    temp_user_items.append(item)
            temp_user_items.sort(key=lambda x: x.timestamp) # Process oldest first or by a defined priority

            for item in temp_user_items:
                # Determine prefix based on item type
                if item.type == ContextItemType.USER_TEXT:
                    prefix = f"--- User-Provided Text Snippet: {item.metadata.get('name', item.id)} ---"
                    suffix = f"--- End User-Provided Text Snippet: {item.metadata.get('name', item.id)} ---"
                elif item.type == ContextItemType.USER_FILE:
                    prefix = f"--- User-Provided File Content: {item.metadata.get('filename', item.id)} ---"
                    suffix = f"--- End User-Provided File Content: {item.metadata.get('filename', item.id)} ---"
                elif item.type == ContextItemType.RAG_SNIPPET:
                    prefix = f"--- Pinned RAG Snippet (Source: {item.metadata.get('original_source', item.source_id)}): {item.metadata.get('name', item.id)} ---"
                    suffix = f"--- End Pinned RAG Snippet: {item.metadata.get('name', item.id)} ---"
                else: # Should not happen if item.type check above is comprehensive
                    prefix = f"--- User-Provided Context: {item.metadata.get('name', item.id)} ---"
                    suffix = f"--- End User-Provided Context: {item.metadata.get('name', item.id)} ---"

                content_str = f"{prefix}\n{item.content}\n{suffix}"
                user_item_msg = Message(role=LLMCoreRole.SYSTEM, content=content_str, session_id=session.id, id=f"user_item_{item.id}")
                if item.tokens is None: # Tokenize if not already done
                    item.tokens = await provider.count_message_tokens([user_item_msg], target_model)
                user_item_msg.tokens = item.tokens
                active_user_item_messages.append(user_item_msg)

        # 4. Chat History (non-system messages)
        history_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM]
        for msg in history_messages_all_non_system: # Pre-tokenize history
             if msg.tokens is None:
                 msg.tokens = await provider.count_message_tokens([msg], target_model)


        # --- Iterative Assembly & Truncation ---
        candidate_messages: List[Message] = []
        current_tokens = 0

        # Helper to add messages if budget allows
        async def _add_to_candidates_if_budget(messages_to_add: List[Message]) -> None:
            nonlocal current_tokens
            for msg in messages_to_add:
                msg_token_count = msg.tokens or 0 # Should be pre-tokenized
                if current_tokens + msg_token_count <= available_tokens_for_prompt:
                    candidate_messages.append(msg)
                    current_tokens += msg_token_count
                else:
                    logger.debug(f"Message '{msg.id}' (role: {msg.role}, tokens: {msg_token_count}) exceeds budget. Not added.")
                    # If it's a critical message (like latest user message), this will be handled by final check.
                    break # Stop adding from this list if one message breaks budget

        # 1. Add all system messages from history
        await _add_to_candidates_if_budget(system_messages_hist)

        # Components to be added based on priority and budget
        # Order: RAG query context, Prioritized User Items, History, Non-prioritized User Items

        # 2. Add RAG query context (if enabled and fits)
        if rag_query_context_message_obj:
            if self._prioritize_user_context_items and self._rag_combination_strategy == "prepend_system":
                # If user items are prioritized and RAG is prepended, RAG might come before user items
                await _add_to_candidates_if_budget([rag_query_context_message_obj])
            elif not self._prioritize_user_context_items and self._rag_combination_strategy == "prepend_system":
                # If user items not prioritized, RAG comes first
                await _add_to_candidates_if_budget([rag_query_context_message_obj])


        # 3. Add User Items (prioritized or not based on config)
        if self._prioritize_user_context_items:
            await _add_to_candidates_if_budget(active_user_item_messages)

        # If RAG is prepend_user, it should come just before the last user message from history.
        # This is complex to inject here; for now, prepend_system is simpler.
        # If RAG was not added yet (e.g. user items prioritized and RAG is prepend_system but now is its turn)
        if rag_query_context_message_obj and rag_query_context_message_obj not in candidate_messages:
             if self._rag_combination_strategy == "prepend_system": # Default placement if not already added
                await _add_to_candidates_if_budget([rag_query_context_message_obj])


        # 4. Add Chat History
        history_budget = available_tokens_for_prompt - current_tokens
        built_history, built_history_tokens = await self._build_history_messages(
            history_messages_all_non_system, provider, target_model, history_budget
        )
        # _add_to_candidates_if_budget will handle if built_history itself is too large
        await _add_to_candidates_if_budget(built_history)


        # 5. Add remaining User Items (if not prioritized and space allows)
        if not self._prioritize_user_context_items:
            # Filter out items already added if any logic error led to that
            remaining_user_items = [item_msg for item_msg in active_user_item_messages if item_msg not in candidate_messages]
            await _add_to_candidates_if_budget(remaining_user_items)

        # --- Final Truncation Loop if still over budget ---
        final_candidate_messages = list(candidate_messages) # Work on a copy

        for priority_type in self._truncation_priority_order:
            if current_tokens <= available_tokens_for_prompt: break

            logger.debug(f"Context over budget ({current_tokens}/{available_tokens_for_prompt}). Attempting truncation for type: '{priority_type}'.")

            items_removed_in_pass = False
            if priority_type == "rag" and rag_query_context_message_obj and rag_query_context_message_obj.id in [m.id for m in final_candidate_messages]:
                logger.debug(f"Truncating by priority '{priority_type}': Removing RAG query context.")
                final_candidate_messages = [m for m in final_candidate_messages if m.id != rag_query_context_message_obj.id]
                current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                rag_documents_used_this_turn = None # RAG context removed
                items_removed_in_pass = True

            elif priority_type == "user_items":
                # Remove user items one by one, e.g., oldest or largest first. For now, just first found from the end.
                # This assumes active_user_item_messages was sorted by preference (e.g. timestamp ascending)
                user_item_ids_in_final = {m.id for m in final_candidate_messages if m.id.startswith("user_item_")}
                # Iterate over a copy of active_user_item_messages to decide removal order (e.g. oldest first)
                sorted_active_user_items_by_preference = sorted(
                    [item_msg for item_msg in active_user_item_messages if item_msg.id in user_item_ids_in_final],
                    key=lambda x: x.timestamp # Example: remove oldest user items first
                )
                for item_to_remove in sorted_active_user_items_by_preference:
                    if current_tokens <= available_tokens_for_prompt: break
                    if item_to_remove.id in user_item_ids_in_final:
                        logger.debug(f"Truncating by priority '{priority_type}': Removing user item '{item_to_remove.id}'.")
                        final_candidate_messages = [m for m in final_candidate_messages if m.id != item_to_remove.id]
                        current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                        items_removed_in_pass = True


            elif priority_type == "history":
                history_in_final = sorted(
                    [m for m in final_candidate_messages if not (m.role == LLMCoreRole.SYSTEM or m.id.startswith("user_item_") or m.id == "rag_query_context_marker")],
                    key=lambda m: m.timestamp
                )
                # Ensure last user message is not removed if it's the only history
                last_user_msg_session = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)

                while len(history_in_final) > self._minimum_history_messages and current_tokens > available_tokens_for_prompt:
                    if not history_in_final: break
                    msg_to_remove = history_in_final.pop(0) # Remove oldest history message

                    # Critical: Do not remove the latest user message from the original session if it's part of history_in_final
                    if last_user_msg_session and msg_to_remove.id == last_user_msg_session.id and len(history_in_final) < self._minimum_history_messages : # Check if it's the last user message and about to go below min
                        logger.debug(f"Protected last user message '{msg_to_remove.id}' from truncation to maintain minimum history or its presence.")
                        history_in_final.insert(0, msg_to_remove) # Add it back
                        break # Cannot truncate history further without removing essential latest user turn

                    logger.debug(f"Truncating by priority '{priority_type}': Removing history message '{msg_to_remove.id}'.")
                    final_candidate_messages = [m for m in final_candidate_messages if m.id != msg_to_remove.id]
                    current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                    items_removed_in_pass = True
                    if current_tokens <= available_tokens_for_prompt: break

            if not items_removed_in_pass:
                 logger.debug(f"No items of priority type '{priority_type}' found to remove or already at minimums.")


        # Final check: ensure the last user message is present if there was one
        last_user_msg_from_session = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        if last_user_msg_from_session:
            is_last_user_msg_present_in_final_context = any(m.id == last_user_msg_from_session.id for m in final_candidate_messages)
            if not is_last_user_msg_present_in_final_context:
                error_detail_msg = "Context length too short. Latest user message could not be included after all truncation attempts."
                logger.error(error_detail_msg + f" Limit: {available_tokens_for_prompt}, Current after truncation: {current_tokens}")
                raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message=error_detail_msg)

        if current_tokens > available_tokens_for_prompt:
            logger.error(f"Final context ({current_tokens} tokens) still exceeds budget ({available_tokens_for_prompt}) despite all truncation efforts.")
            raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message="Context exceeds token limit after all truncation attempts.")

        logger.info(f"Final prepared context: {len(final_candidate_messages)} messages, {current_tokens} tokens for model '{target_model}'.")

        # Filter out the RAG marker message before returning, but return the docs used.
        final_payload_messages = [msg for msg in final_candidate_messages if msg.id != "rag_query_context_marker"]
        return final_payload_messages, rag_documents_used_this_turn


    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string for context injection."""
        if not documents: return ""
        # Sort documents by score if available (assuming lower is better, adjust if needed)
        # Some vector stores return distance (lower is better), others similarity (higher is better)
        # Assuming ChromaDB-like distance scores for now.
        sorted_documents = sorted(documents, key=lambda d: d.score if d.score is not None else float('inf'))

        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(sorted_documents):
            source_info = doc.metadata.get("source", f"Document {doc.id[:8]}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            content_snippet = doc.content.replace('\n', ' ').strip() # Normalize newlines for context block
            context_parts.append(f"\n[Source: {source_info} {score_info}]\n{content_snippet}")
        context_parts.append("--- End Retrieved Documents ---")
        return "\n".join(context_parts)
