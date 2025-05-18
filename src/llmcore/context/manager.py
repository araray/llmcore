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
        self._truncation_priority: str = cm_config.get('truncation_priority', 'history')
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
                     f"truncation_priority={self._truncation_priority}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     f"rag_combo_strategy={self._rag_combination_strategy}")
        self._validate_strategies()

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

        if self._truncation_priority not in ['history', 'rag', 'user_items']:
             logger.warning(f"Unsupported truncation_priority '{self._truncation_priority}'. Falling back to 'history'.")
             self._truncation_priority = 'history'


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
        """
        Prepares the context payload (List[Message]) to be sent to the LLM provider.
        (Full docstring from previous turn applies, updated for clarity)

        Order of assembly and truncation:
        1. System messages from chat history.
        2. Standard RAG query results (if RAG enabled for the current turn).
        3. Active user-added context items (text, files).
        4. Chat history (user-retained latest messages, then backfilled older messages).
        5. If over budget, truncation occurs based on `truncation_priority`.
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

        # --- Component Gathering ---
        # 1. System Messages from history
        system_messages_hist = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]

        # 2. Standard RAG query results
        formatted_rag_query_context_str = ""
        rag_docs_for_query: List[ContextDocument] = [] # Keep track for potential truncation
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k_val = rag_k if rag_k is not None else self._default_rag_k
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    rag_docs_for_query = await vector_storage.similarity_search(
                        query_embedding=query_embedding, k=k_val, collection_name=rag_collection
                    )
                    formatted_rag_query_context_str = self._format_rag_docs_for_context(rag_docs_for_query)
                    logger.info(f"Standard RAG search returned {len(rag_docs_for_query)} documents.")
                except (EmbeddingError, VectorStorageError, ConfigError) as e:
                     logger.error(f"Standard RAG retrieval failed: {e}. Proceeding without RAG query context.")
            else: logger.warning("Standard RAG enabled, but no user message for query.")

        # 3. Active User-Added Context Items
        active_user_items: List[ContextItem] = []
        if active_context_item_ids and session.context_items:
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and (item.type == ContextItemType.USER_TEXT or item.type == ContextItemType.USER_FILE):
                    active_user_items.append(item)
            active_user_items.sort(key=lambda x: x.timestamp) # Oldest first, or configurable

        # 4. Chat History (non-system messages)
        history_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM]

        # --- Initial Assembly and Token Counting ---
        # This list will hold Message objects ready for the provider
        candidate_messages: List[Message] = []
        current_tokens = 0

        # Helper to add messages and count tokens
        async def _add_and_count(messages_to_add: List[Message], component_name: str) -> int:
            nonlocal current_tokens
            added_tokens = 0
            for msg in messages_to_add:
                msg_tokens = msg.tokens if msg.tokens is not None else await provider.count_message_tokens([msg], target_model)
                msg.tokens = msg_tokens # Cache token count on message object

                if current_tokens + msg_tokens <= available_tokens_for_prompt:
                    candidate_messages.append(msg)
                    current_tokens += msg_tokens
                    added_tokens += msg_tokens
                else:
                    logger.debug(f"Budget exceeded while adding {component_name}. Stopping addition of this component.")
                    return added_tokens # Return tokens added before exceeding budget
            logger.debug(f"Added {component_name} ({added_tokens} tokens). Current total: {current_tokens} tokens.")
            return added_tokens

        # Add System Messages from history
        await _add_and_count(system_messages_hist, "System History Messages")

        # Add RAG Query Context (as a system message)
        rag_query_context_message_obj: Optional[Message] = None
        if formatted_rag_query_context_str:
            rag_query_context_message_obj = Message(
                role=LLMCoreRole.SYSTEM, content=formatted_rag_query_context_str,
                session_id=session.id, id="rag_query_context_marker"
            )
            await _add_and_count([rag_query_context_message_obj], "RAG Query Context")
            if rag_query_context_message_obj not in candidate_messages: # It didn't fit
                rag_query_context_message_obj = None # Mark as not included
                rag_docs_for_query = [] # If context string doesn't fit, docs are effectively not used

        # Add Active User Items (as system messages)
        user_items_messages_added: List[Message] = []
        for item in active_user_items: # Iterate through sorted active items
            content_str = (f"--- User-Provided Context: {item.metadata.get('name', item.id)} ---\n"
                           f"{item.content}\n"
                           f"--- End User-Provided Context: {item.metadata.get('name', item.id)} ---")
            user_item_msg = Message(role=LLMCoreRole.SYSTEM, content=content_str, session_id=session.id, id=f"user_item_{item.id}")
            # We need to count this specific message's tokens, not the item's pre-counted tokens,
            # as the formatting adds to the count.
            # The _add_and_count helper will do this.
            if current_tokens + (await provider.count_message_tokens([user_item_msg], target_model)) <= available_tokens_for_prompt:
                 await _add_and_count([user_item_msg], f"User Item '{item.id}'")
                 if user_item_msg in candidate_messages:
                     user_items_messages_added.append(user_item_msg)
                 else: break # Stop if one doesn't fit
            else: break


        # Add Chat History (user_retained_messages_count and backfill)
        # This needs to be added carefully considering remaining budget.
        history_budget_after_fixed_items = available_tokens_for_prompt - current_tokens
        selected_history_for_candidate: List[Message] = []

        if history_budget_after_fixed_items > 0 and history_messages_all_non_system:
            # Stage 1: User-retained messages (latest N)
            # We take from the end of history_messages_all_non_system
            num_retained_actual = 0
            temp_retained_history: List[Message] = []
            temp_retained_tokens = 0

            for msg in reversed(history_messages_all_non_system):
                if num_retained_actual >= self._user_retained_messages_count and self._user_retained_messages_count > 0:
                    break # Got enough "retained" messages

                msg_tokens = msg.tokens if msg.tokens is not None else await provider.count_message_tokens([msg], target_model)
                msg.tokens = msg_tokens

                if temp_retained_tokens + msg_tokens <= history_budget_after_fixed_items:
                    temp_retained_history.append(msg)
                    temp_retained_tokens += msg_tokens
                    num_retained_actual +=1
                else:
                    break # Not enough budget even for retained messages

            selected_history_for_candidate.extend(reversed(temp_retained_history)) # Add in chronological order
            current_tokens += temp_retained_tokens
            logger.debug(f"Added {len(selected_history_for_candidate)} retained history messages ({temp_retained_tokens} tokens).")

            # Stage 2: Backfill with older messages
            history_budget_after_retained = available_tokens_for_prompt - current_tokens
            if history_budget_after_retained > 0:
                # Candidates for backfilling are those not already in selected_history_for_candidate
                ids_in_selected_history = {m.id for m in selected_history_for_candidate}
                older_history_candidates = [
                    msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history
                ]

                temp_backfill_history: List[Message] = []
                temp_backfill_tokens = 0
                for msg in reversed(older_history_candidates): # Iterate from newest of the remaining older messages
                    msg_tokens = msg.tokens if msg.tokens is not None else await provider.count_message_tokens([msg], target_model)
                    msg.tokens = msg_tokens
                    if temp_backfill_tokens + msg_tokens <= history_budget_after_retained:
                        temp_backfill_history.append(msg)
                        temp_backfill_tokens += msg_tokens
                    else:
                        break
                # Insert backfilled messages before the retained ones to maintain order
                candidate_messages.extend(sorted(temp_backfill_history, key=lambda m: m.timestamp))
                current_tokens += temp_backfill_tokens
                logger.debug(f"Backfilled {len(temp_backfill_history)} older history messages ({temp_backfill_tokens} tokens).")

        # Add the already selected history (retained ones) to the main candidate_messages list
        candidate_messages.extend(selected_history_for_candidate) # These were already counted towards current_tokens

        # --- Truncation Loop ---
        # Re-sort candidate_messages (excluding system messages at the start) to ensure chronological order for history part
        # System messages, RAG context, User items are usually prepended. History follows.
        # A more robust sort might be needed if insertion order is complex.
        # For now, assume system/RAG/user items are at the start, history at the end.

        # Separate prepended items from history for easier sorting/truncation of history
        prepended_items = [m for m in candidate_messages if m.role == LLMCoreRole.SYSTEM or m.id.startswith("user_item_") or m.id == "rag_query_context_marker"]
        history_items_in_candidate = [m for m in candidate_messages if m not in prepended_items]
        history_items_in_candidate.sort(key=lambda m: m.timestamp)

        candidate_messages = prepended_items + history_items_in_candidate
        current_tokens = await provider.count_message_tokens(candidate_messages, target_model) # Recalculate total

        logger.debug(f"Context before final truncation: {len(candidate_messages)} messages, {current_tokens} tokens (Budget: {available_tokens_for_prompt}).")

        while current_tokens > available_tokens_for_prompt:
            can_truncate_history = len([m for m in history_items_in_candidate if m.role != LLMCoreRole.SYSTEM]) > self._minimum_history_messages
            can_truncate_user_items = any(m.id.startswith("user_item_") for m in prepended_items)
            can_truncate_rag = rag_query_context_message_obj is not None and rag_query_context_message_obj in prepended_items

            truncated_this_iteration = False

            if self._truncation_priority == "history" and can_truncate_history:
                # Remove oldest history message
                for i, msg in enumerate(history_items_in_candidate):
                    if msg.role != LLMCoreRole.SYSTEM: # Should always be true here
                        logger.debug(f"Truncating (history priority): Oldest history message '{msg.id}' ({msg.tokens} tokens).")
                        history_items_in_candidate.pop(i)
                        truncated_this_iteration = True
                        break
            elif self._truncation_priority == "user_items" and can_truncate_user_items:
                # Remove oldest/largest user item (simplistic: remove first one found)
                for i, msg in enumerate(prepended_items):
                    if msg.id.startswith("user_item_"):
                        logger.debug(f"Truncating (user_items priority): User item '{msg.id}' ({msg.tokens} tokens).")
                        prepended_items.pop(i)
                        # Also remove from original active_user_items list to reflect it's not used
                        original_item_id = msg.id.replace("user_item_", "")
                        active_user_items = [item for item in active_user_items if item.id != original_item_id]
                        truncated_this_iteration = True
                        break
            elif self._truncation_priority == "rag" and can_truncate_rag:
                # Simplistic: remove the entire RAG context string if it's too much.
                # Finer-grained: truncate rag_docs_for_query and reformat.
                if rag_query_context_message_obj:
                    logger.debug(f"Truncating (rag priority): RAG query context ({rag_query_context_message_obj.tokens} tokens).")
                    prepended_items = [m for m in prepended_items if m.id != "rag_query_context_marker"]
                    rag_query_context_message_obj = None # Mark as removed
                    rag_docs_for_query = [] # No RAG docs used
                    truncated_this_iteration = True

            # Fallback truncation if priority item couldn't be truncated
            if not truncated_this_iteration:
                if can_truncate_history:
                    for i, msg in enumerate(history_items_in_candidate):
                        if msg.role != LLMCoreRole.SYSTEM:
                            logger.debug(f"Truncating (fallback): Oldest history message '{msg.id}' ({msg.tokens} tokens).")
                            history_items_in_candidate.pop(i)
                            truncated_this_iteration = True
                            break
                elif can_truncate_user_items: # Fallback to user items
                     for i, msg in enumerate(prepended_items):
                        if msg.id.startswith("user_item_"):
                            logger.debug(f"Truncating (fallback): User item '{msg.id}' ({msg.tokens} tokens).")
                            prepended_items.pop(i)
                            original_item_id = msg.id.replace("user_item_", "")
                            active_user_items = [item for item in active_user_items if item.id != original_item_id]
                            truncated_this_iteration = True
                            break
                elif can_truncate_rag and rag_query_context_message_obj: # Fallback to RAG
                    logger.debug(f"Truncating (fallback): RAG query context ({rag_query_context_message_obj.tokens} tokens).")
                    prepended_items = [m for m in prepended_items if m.id != "rag_query_context_marker"]
                    rag_query_context_message_obj = None
                    rag_docs_for_query = []
                    truncated_this_iteration = True

            if not truncated_this_iteration:
                logger.error(f"Cannot truncate context further. Current tokens {current_tokens} still exceed budget {available_tokens_for_prompt}.")
                break

            candidate_messages = prepended_items + history_items_in_candidate
            current_tokens = await provider.count_message_tokens(candidate_messages, target_model)
            logger.debug(f"Context after truncation iteration: {len(candidate_messages)} messages, {current_tokens} tokens.")

        if current_tokens > available_tokens_for_prompt:
            # Final check: ensure last user message is included if possible.
            # This is a simplified check; more robust logic might be needed.
            last_original_user_msg = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_original_user_msg and not any(m.id == last_original_user_msg.id for m in candidate_messages):
                 logger.error(f"Context length error: Last user message was truncated. Limit: {available_tokens_for_prompt}, Actual: {current_tokens}")
                 raise ContextLengthError(
                    model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens,
                    message="Context length too short to include essential messages after truncation."
                 )
            else: # Still over budget, but last user message might be there.
                 logger.error(f"Final context ({current_tokens} tokens) exceeds budget ({available_tokens_for_prompt}) despite truncation.")
                 raise ContextLengthError(
                    model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens,
                    message="Context exceeds token limit even after truncation attempts."
                 )

        logger.info(f"Final prepared context: {len(candidate_messages)} messages, {current_tokens} tokens for model '{target_model}'.")

        # Remove any special marker messages (like RAG marker if it was just for structure) before returning
        # For now, the RAG context is a system message, so it's fine.
        # If user_item messages were markers, they'd be removed here.
        return candidate_messages


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
