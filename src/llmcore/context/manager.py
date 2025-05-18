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
        # history_selection_strategy is used within _build_history_messages
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
        if not ordered_priorities: # Fallback if all were invalid
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

        # Validation for _truncation_priority_order is handled by _parse_truncation_priority

    async def _build_history_messages(
        self,
        history_messages_all_non_system: List[Message],
        provider: BaseProvider,
        target_model: str,
        budget: int
    ) -> Tuple[List[Message], int]:
        """Builds the chat history part of the context based on user_retained_messages and backfilling."""
        selected_history: List[Message] = []
        current_history_tokens = 0

        if budget <= 0 or not history_messages_all_non_system:
            return selected_history, current_history_tokens

        # Stage 1: User-retained messages (latest N turns)
        # A "turn" is roughly a user message and an assistant response.
        # We aim to keep `_user_retained_messages_count` user messages and their preceding assistant messages if possible.
        retained_for_now: List[Message] = []
        tokens_for_retained = 0

        # Iterate from newest to oldest to pick retained messages
        num_user_messages_retained = 0
        temp_retained_buffer: List[Message] = []

        for msg in reversed(history_messages_all_non_system):
            msg_tokens = msg.tokens if msg.tokens is not None else await provider.count_message_tokens([msg], target_model)
            msg.tokens = msg_tokens # Cache

            if tokens_for_retained + msg_tokens <= budget:
                temp_retained_buffer.append(msg)
                tokens_for_retained += msg_tokens
                if msg.role == LLMCoreRole.USER:
                    num_user_messages_retained += 1
                if self._user_retained_messages_count > 0 and num_user_messages_retained >= self._user_retained_messages_count:
                    break
            else: # Not enough budget even for this message
                break

        retained_for_now = sorted(temp_retained_buffer, key=lambda m: m.timestamp) # Add in chronological order
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
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True) # Newest of the old first

            temp_backfill_history: List[Message] = []
            tokens_for_backfill = 0
            for msg in older_history_candidates: # Iterate from newest of the remaining older messages
                msg_tokens = msg.tokens if msg.tokens is not None else await provider.count_message_tokens([msg], target_model)
                msg.tokens = msg_tokens

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
    ) -> List[Message]:
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
        # 1. System Messages from history
        system_messages_hist = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
        for msg in system_messages_hist: # Pre-tokenize
            msg.tokens = msg.tokens or await provider.count_message_tokens([msg], target_model)

        # 2. Standard RAG query results (if RAG enabled for the current turn)
        formatted_rag_query_context_str = ""
        rag_docs_for_query: List[ContextDocument] = []
        rag_query_context_message_obj: Optional[Message] = None
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k_val = rag_k if rag_k is not None else self._default_rag_k
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    rag_docs_for_query = await vector_storage.similarity_search(query_embedding=query_embedding, k=k_val, collection_name=rag_collection)
                    formatted_rag_query_context_str = self._format_rag_docs_for_context(rag_docs_for_query)
                    if formatted_rag_query_context_str:
                        rag_query_context_message_obj = Message(role=LLMCoreRole.SYSTEM, content=formatted_rag_query_context_str, session_id=session.id, id="rag_query_context_marker")
                        rag_query_context_message_obj.tokens = await provider.count_message_tokens([rag_query_context_message_obj], target_model)
                    logger.info(f"Std RAG returned {len(rag_docs_for_query)} docs; context string tokens: {rag_query_context_message_obj.tokens if rag_query_context_message_obj else 0}.")
                except Exception as e: logger.error(f"Std RAG failed: {e}. No RAG query context.")
            else: logger.warning("Std RAG enabled, but no user message for query.")

        # 3. Active User-Added Context Items
        active_user_item_messages: List[Message] = []
        if active_context_item_ids and session.context_items:
            temp_user_items = []
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and (item.type == ContextItemType.USER_TEXT or item.type == ContextItemType.USER_FILE):
                    temp_user_items.append(item)
            temp_user_items.sort(key=lambda x: x.timestamp) # Process oldest first or by a defined priority

            for item in temp_user_items:
                content_str = (f"--- User-Provided Context: {item.metadata.get('name', item.id)} ---\n"
                               f"{item.content}\n"
                               f"--- End User-Provided Context: {item.metadata.get('name', item.id)} ---")
                user_item_msg = Message(role=LLMCoreRole.SYSTEM, content=content_str, session_id=session.id, id=f"user_item_{item.id}")
                user_item_msg.tokens = item.tokens or await provider.count_message_tokens([user_item_msg], target_model)
                item.tokens = user_item_msg.tokens # Cache on original item
                active_user_item_messages.append(user_item_msg)

        # 4. Chat History (non-system messages)
        history_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM]

        # --- Iterative Assembly & Truncation ---
        candidate_messages: List[Message] = []
        current_tokens = 0

        # Order of preference for adding components:
        # 1. System history messages
        # 2. RAG query context (if enabled and fits)
        # 3. Prioritized User-added items (if enabled and fits)
        # 4. Chat History (retained + backfill)
        # 5. Non-prioritized User-added items (if fits)

        # Add system history
        for msg in system_messages_hist:
            if current_tokens + (msg.tokens or 0) <= available_tokens_for_prompt:
                candidate_messages.append(msg); current_tokens += msg.tokens or 0
            else: break

        # Add RAG query context
        included_rag_query_message = False
        if rag_query_context_message_obj and rag_query_context_message_obj.tokens is not None:
            if current_tokens + rag_query_context_message_obj.tokens <= available_tokens_for_prompt:
                candidate_messages.append(rag_query_context_message_obj)
                current_tokens += rag_query_context_message_obj.tokens
                included_rag_query_message = True
            else: logger.debug("RAG query context too large, omitting.")

        # Add User Items (prioritized or not based on config)
        user_items_to_consider = list(active_user_item_messages) # Copy

        def _add_user_items_if_budget(budget_left: int) -> int:
            nonlocal current_tokens, user_items_to_consider
            tokens_added_this_pass = 0
            items_added_this_pass = []
            for item_msg in user_items_to_consider: # Assumes sorted by preference
                if item_msg.tokens is None: continue # Should have been tokenized
                if current_tokens + item_msg.tokens <= available_tokens_for_prompt and tokens_added_this_pass + item_msg.tokens <= budget_left :
                    candidate_messages.append(item_msg)
                    current_tokens += item_msg.tokens
                    tokens_added_this_pass += item_msg.tokens
                    items_added_this_pass.append(item_msg)
                else: break # No more budget for this item or overall
            user_items_to_consider = [item for item in user_items_to_consider if item not in items_added_this_pass] # Remove added
            return tokens_added_this_pass

        if self._prioritize_user_context_items:
            _add_user_items_if_budget(available_tokens_for_prompt - current_tokens)

        # Add Chat History
        history_budget = available_tokens_for_prompt - current_tokens
        built_history, built_history_tokens = await self._build_history_messages(
            history_messages_all_non_system, provider, target_model, history_budget
        )
        candidate_messages.extend(built_history)
        current_tokens += built_history_tokens

        # Add remaining User Items (if not prioritized and space allows)
        if not self._prioritize_user_context_items:
            _add_user_items_if_budget(available_tokens_for_prompt - current_tokens)

        # --- Final Truncation Loop if still over budget ---
        # This loop applies configured truncation priorities.
        final_candidate_messages = list(candidate_messages) # Work on a copy

        for priority_type in self._truncation_priority_order:
            if current_tokens <= available_tokens_for_prompt: break

            if priority_type == "rag" and included_rag_query_message and rag_query_context_message_obj:
                logger.debug(f"Truncating by priority '{priority_type}': Removing RAG query context.")
                final_candidate_messages = [m for m in final_candidate_messages if m.id != "rag_query_context_marker"]
                current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                included_rag_query_message = False # Mark as removed
                rag_docs_for_query = [] # No RAG docs are effectively used now
                if current_tokens <= available_tokens_for_prompt: break

            elif priority_type == "user_items":
                # Remove user items one by one, e.g., oldest or largest first. For now, just first found.
                temp_final_candidates = list(final_candidate_messages) # Iterate on copy for safe removal
                for i in range(len(temp_final_candidates) -1, -1, -1): # Iterate backwards to remove
                    msg = temp_final_candidates[i]
                    if msg.id.startswith("user_item_"):
                        logger.debug(f"Truncating by priority '{priority_type}': Removing user item '{msg.id}'.")
                        final_candidate_messages.pop(i)
                        current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                        if current_tokens <= available_tokens_for_prompt: break
                if current_tokens <= available_tokens_for_prompt: break

            elif priority_type == "history":
                # Remove history messages one by one, oldest first, respecting minimums
                # Re-filter history items from the current candidates
                history_in_final = sorted([m for m in final_candidate_messages if not (m.role == LLMCoreRole.SYSTEM or m.id.startswith("user_item_") or m.id == "rag_query_context_marker")], key=lambda m:m.timestamp)

                while len(history_in_final) > self._minimum_history_messages and current_tokens > available_tokens_for_prompt:
                    removed_msg = history_in_final.pop(0) # Remove oldest history
                    final_candidate_messages = [m for m in final_candidate_messages if m.id != removed_msg.id]
                    logger.debug(f"Truncating by priority '{priority_type}': Removing history message '{removed_msg.id}'.")
                    current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
                    if current_tokens <= available_tokens_for_prompt: break
                if current_tokens <= available_tokens_for_prompt: break

        if current_tokens > available_tokens_for_prompt:
            last_user_msg_session = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            is_last_user_msg_present = any(m.id == last_user_msg_session.id for m in final_candidate_messages if last_user_msg_session) if last_user_msg_session else True

            if not is_last_user_msg_present :
                 error_detail_msg = "Context length too short to include the latest user message after all truncation attempts."
                 logger.error(error_detail_msg + f" Limit: {available_tokens_for_prompt}, Actual: {current_tokens}")
                 raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message=error_detail_msg)
            else:
                 logger.error(f"Final context ({current_tokens} tokens) still exceeds budget ({available_tokens_for_prompt}) despite all truncation efforts.")
                 raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message="Context exceeds token limit after all truncation attempts.")

        logger.info(f"Final prepared context: {len(final_candidate_messages)} messages, {current_tokens} tokens for model '{target_model}'.")
        return [msg for msg in final_candidate_messages if msg.id != "rag_query_context_marker"] # Ensure marker is not sent


    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string for context injection."""
        if not documents: return ""
        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(documents):
            source_info = doc.metadata.get("source", f"Document {doc.id[:8]}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            content_snippet = doc.content.replace('\n', ' ').strip()
            context_parts.append(f"\n[Source: {source_info} {score_info}]\n{content_snippet}")
        context_parts.append("--- End Retrieved Documents ---")
        return "\n".join(context_parts)
