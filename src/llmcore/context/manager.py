# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and user-added context items.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]


from ..embedding.manager import EmbeddingManager
from ..exceptions import (ConfigError, ContextError, ContextLengthError,
                          EmbeddingError, ProviderError, VectorStorageError)
from ..models import (ChatSession, ContextDocument, ContextItem,
                      ContextItemType, Message)
from ..models import Role as LLMCoreRole
from ..providers.base import BaseProvider
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results and user-added context items,
    and ensures the final payload adheres to token limits using configurable strategies.
    Includes per-item truncation for user-added context.
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
        self._max_chars_per_user_item: int = cm_config.get('max_chars_per_user_item', 40000)

        logger.info("ContextManager initialized.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"user_retained_messages={self._user_retained_messages_count}, "
                     f"prioritize_user_items={self._prioritize_user_context_items}, "
                     f"truncation_priority_order={self._truncation_priority_order}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     f"rag_combo_strategy={self._rag_combination_strategy}, "
                     f"max_chars_per_user_item={self._max_chars_per_user_item}")
        self._validate_strategies()

    def _parse_truncation_priority(self, priority_str: str) -> List[str]:
        """
        Parses the truncation_priority string from config into a list of valid types.
        Valid types are "history", "rag", "user_items".
        Logs a warning if invalid items are found or if the list is empty, using defaults.
        """
        valid_priorities = {"history", "rag", "user_items"}
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_priorities]
        if len(ordered_priorities) != len(priorities):
            invalid_found = set(priorities) - set(ordered_priorities)
            logger.warning(f"Invalid items found in 'truncation_priority' config: {invalid_found}. "
                           f"Using valid ones: {ordered_priorities}")
        if not ordered_priorities:
            default_priority = ["history", "rag", "user_items"]
            logger.warning(f"No valid 'truncation_priority' items configured. Defaulting to: {default_priority}")
            return default_priority
        return ordered_priorities

    def _validate_strategies(self):
        """Validates configured strategies and logs warnings if unsupported, falling back to defaults."""
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
        Builds the chat history part of the context based on user_retained_messages and backfilling within the given token budget.
        Ensures messages have token counts before processing.

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

        # Ensure messages have token counts. This is crucial for budget calculations.
        for msg in history_messages_all_non_system:
            if msg.tokens is None:
                try:
                    msg.tokens = await provider.count_message_tokens([msg], target_model)
                except Exception as e_tok:
                    logger.error(f"Failed to count tokens for history message '{msg.id}': {e_tok}. Approximating.")
                    msg.tokens = len(msg.content) // 4 # Rough approximation

        # Stage 1: User-retained messages (latest N user messages and their preceding assistant messages)
        retained_for_now: List[Message] = []
        tokens_for_retained = 0
        num_user_messages_retained = 0
        temp_retained_buffer: List[Message] = [] # To hold messages for a potential turn

        # Iterate from newest to oldest to pick recent turns
        for i in range(len(history_messages_all_non_system) - 1, -1, -1):
            msg = history_messages_all_non_system[i]
            msg_tokens = msg.tokens or 0 # Should have tokens by now

            # Try to add the current message to the buffer
            if tokens_for_retained + msg_tokens <= budget:
                temp_retained_buffer.insert(0, msg) # Insert at beginning to maintain order
                tokens_for_retained += msg_tokens

                if msg.role == LLMCoreRole.USER:
                    num_user_messages_retained += 1
                    # If we've retained enough user messages, break
                    if self._user_retained_messages_count > 0 and \
                       num_user_messages_retained >= self._user_retained_messages_count:
                        break
            else:
                # Not enough budget for this message, so this turn (or part of it) cannot be included
                break # Stop collecting retained messages

        retained_for_now = temp_retained_buffer # temp_retained_buffer is already sorted chronologically
        selected_history.extend(retained_for_now)
        current_history_tokens = tokens_for_retained
        logger.debug(f"Retained {len(retained_for_now)} recent history messages ({current_history_tokens} tokens) "
                     f"based on user_retained_messages_count ({self._user_retained_messages_count}).")

        # Stage 2: Backfill with older messages if budget allows
        remaining_history_budget_for_backfill = budget - current_history_tokens
        if remaining_history_budget_for_backfill > 0:
            # Get messages not already selected, these are older
            ids_in_selected_history = {m.id for m in selected_history}
            older_history_candidates = [
                msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history
            ]
            # Sort older candidates from newest to oldest to prioritize more recent ones for backfill
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True)

            temp_backfill_history: List[Message] = []
            tokens_for_backfill = 0
            for msg in older_history_candidates:
                msg_tokens = msg.tokens or 0
                if tokens_for_backfill + msg_tokens <= remaining_history_budget_for_backfill:
                    temp_backfill_history.insert(0, msg) # Insert at beginning to maintain order
                    tokens_for_backfill += msg_tokens
                else:
                    break # No more budget for backfill

            # Prepend backfilled messages (which are now sorted) to the already selected history
            selected_history = temp_backfill_history + selected_history
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
        rag_collection: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None, # Added
        prompt_template_values: Optional[Dict[str, str]] = None # Added
    ) -> Tuple[List[Message], Optional[List[ContextDocument]], int]:
        """
        Prepares the context payload for the LLM, including history, RAG, and user-added items.
        User-added items are truncated if they exceed `max_chars_per_user_item`, unless
        `item.metadata.get('ignore_char_limit', False)` is True.

        Args:
            session: The current ChatSession object.
            provider_name: Name of the LLM provider.
            model_name: Specific model name for the provider.
            active_context_item_ids: List of IDs for user-added context items to include.
            rag_enabled: Whether to perform RAG.
            rag_k: Number of documents for RAG retrieval.
            rag_collection: Vector store collection for RAG.
            rag_metadata_filter: Optional dictionary for metadata filtering in RAG.
            prompt_template_values: Optional dictionary of values for custom prompt template placeholders.
                                    (Currently accepted, usage planned for Phase 4).

        Returns:
            A tuple: (final list of Messages for LLM, list of RAG docs used, total token count of final messages).
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for context (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' (Provider: {provider.get_name()}, RAG: {rag_enabled}). Session: {session.id}")
        if rag_metadata_filter:
            logger.debug(f"RAG metadata filter active: {rag_metadata_filter}")
        if prompt_template_values:
            logger.debug(f"Custom prompt template values provided: {prompt_template_values}")


        max_context_tokens = provider.get_max_context_length(target_model)
        available_tokens_for_prompt = max_context_tokens - self._reserved_response_tokens
        if available_tokens_for_prompt <= 0:
             raise ContextError(f"Config error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_context_tokens}).")
        logger.debug(f"Max context: {max_context_tokens}, Reserved for response: {self._reserved_response_tokens}, Available for prompt: {available_tokens_for_prompt}")

        # --- Initialize components of the context ---
        system_messages_hist: List[Message] = []
        active_user_item_messages: List[Message] = []
        rag_query_context_message_obj: Optional[Message] = None
        rag_documents_used_this_turn: Optional[List[ContextDocument]] = None

        # 1. Extract and tokenize historical system messages from the session
        system_messages_from_session = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
        for msg in system_messages_from_session:
            if msg.tokens is None: # Ensure tokens are counted
                msg.tokens = await provider.count_message_tokens([msg], target_model)
            system_messages_hist.append(msg) # Add to our list for context building

        # 2. Process User-Added Context Items (with potential per-item truncation)
        if active_context_item_ids and session.context_items:
            temp_user_items_to_process: List[ContextItem] = []
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and item.type in [ContextItemType.USER_TEXT, ContextItemType.USER_FILE, ContextItemType.RAG_SNIPPET]:
                    temp_user_items_to_process.append(item)
            temp_user_items_to_process.sort(key=lambda x: x.timestamp)

            for item in temp_user_items_to_process:
                item.is_truncated = False # Reset truncation flag
                ignore_this_item_char_limit = item.metadata.get('ignore_char_limit', False)

                if item.original_tokens is None:
                    temp_msg_for_orig_count = Message(role=LLMCoreRole.SYSTEM, content=item.content, session_id=session.id, id=f"temp_orig_count_{item.id}")
                    try:
                        item.original_tokens = await provider.count_message_tokens([temp_msg_for_orig_count], target_model)
                    except Exception as e_tok:
                        logger.error(f"Failed to count original tokens for item '{item.id}': {e_tok}. Approximating.")
                        item.original_tokens = len(item.content) // 4

                if not ignore_this_item_char_limit and len(item.content) > self._max_chars_per_user_item:
                    original_char_len = len(item.content)
                    item_type_value = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
                    truncated_content = item.content[:self._max_chars_per_user_item] + \
                                        f"\n[...content of item '{item.id}' (type: {item_type_value}) truncated due to character limit...]"
                    item.content = truncated_content
                    item.is_truncated = True

                    temp_msg_for_trunc_count = Message(role=LLMCoreRole.SYSTEM, content=item.content, session_id=session.id, id=f"temp_trunc_count_{item.id}")
                    try:
                        item.tokens = await provider.count_message_tokens([temp_msg_for_trunc_count], target_model)
                    except Exception as e_tok_trunc:
                        logger.error(f"Failed to count tokens for truncated item '{item.id}': {e_tok_trunc}. Approximating.")
                        item.tokens = len(item.content) // 4
                    logger.warning(
                        f"User context item '{item.id}' (type: {item_type_value}, original: {item.original_tokens} tokens, {original_char_len} chars) "
                        f"was truncated to {self._max_chars_per_user_item} chars (new content tokens: {item.tokens}). ignore_char_limit was False."
                    )
                elif item.tokens is None:
                    item.tokens = item.original_tokens

                if ignore_this_item_char_limit:
                    logger.info(f"User context item '{item.id}' (type: {item.type.value if isinstance(item.type, ContextItemType) else str(item.type)}) is ignoring character limit ({self._max_chars_per_user_item} chars). Full content included.")

                item_type_str = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
                source_desc = item.metadata.get('filename', item.id) if item.type == ContextItemType.USER_FILE else \
                              (item.metadata.get('original_source', item.source_id) if item.type == ContextItemType.RAG_SNIPPET else item.id)
                truncation_status_msg = " (TRUNCATED)" if item.is_truncated else ""

                prefix = f"--- User-Provided Context Item (ID: {item.id}, Type: {item_type_str}, Source: {source_desc}, Original Tokens: {item.original_tokens or 'N/A'}{truncation_status_msg}) ---"
                suffix = f"--- End User-Provided Context Item (ID: {item.id}) ---"
                content_str_for_llm = f"{prefix}\n{item.content}\n{suffix}"

                user_item_msg = Message(
                    role=LLMCoreRole.SYSTEM,
                    content=content_str_for_llm,
                    session_id=session.id,
                    id=f"user_item_{item.id}"
                )
                try:
                    user_item_msg.tokens = await provider.count_message_tokens([user_item_msg], target_model)
                except Exception as e_tok_final:
                    logger.error(f"Failed to count tokens for formatted user item message '{user_item_msg.id}': {e_tok_final}. Approximating.")
                    user_item_msg.tokens = len(user_item_msg.content) // 4
                active_user_item_messages.append(user_item_msg)

        # 3. Perform RAG if enabled
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k_val = rag_k if rag_k is not None else self._default_rag_k
                rag_collection_name_actual = rag_collection or self._storage_manager.get_vector_storage()._default_collection_name # type: ignore
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    retrieved_rag_docs = await vector_storage.similarity_search(
                        query_embedding=query_embedding,
                        k=k_val,
                        collection_name=rag_collection_name_actual,
                        filter_metadata=rag_metadata_filter # Pass the filter here
                    )

                    formatted_rag_query_context_str = self._format_rag_docs_for_context(retrieved_rag_docs)
                    if formatted_rag_query_context_str:
                        rag_query_context_message_obj = Message(role=LLMCoreRole.SYSTEM, content=formatted_rag_query_context_str, session_id=session.id, id="rag_query_context_marker")
                        rag_query_context_message_obj.tokens = await provider.count_message_tokens([rag_query_context_message_obj], target_model)
                        rag_documents_used_this_turn = retrieved_rag_docs
                    logger.info(f"Std RAG returned {len(retrieved_rag_docs)} docs; RAG context message tokens: {rag_query_context_message_obj.tokens if rag_query_context_message_obj else 0}.")
                except Exception as e: logger.error(f"Std RAG failed: {e}. No RAG query context will be added.")
            else: logger.warning("Std RAG enabled, but no user message found in history to use as query.")

        # 4. Get non-system messages from history and ensure they have token counts
        history_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM]
        for msg in history_messages_all_non_system:
             if msg.tokens is None: msg.tokens = await provider.count_message_tokens([msg], target_model)

        # --- Assemble candidate messages and manage token budget ---
        candidate_messages: List[Message] = []
        current_tokens = 0

        async def _add_to_candidates_if_budget(messages_to_add: List[Message]) -> None:
            nonlocal current_tokens
            for msg in messages_to_add:
                if msg.tokens is None:
                    try:
                        msg.tokens = await provider.count_message_tokens([msg], target_model)
                    except Exception as e_tok_add:
                        logger.error(f"Failed to count tokens for message '{msg.id}' during _add_to_candidates: {e_tok_add}. Approximating.")
                        msg.tokens = len(msg.content) // 4
                msg_token_count = msg.tokens or 0
                if current_tokens + msg_token_count <= available_tokens_for_prompt:
                    candidate_messages.append(msg)
                    current_tokens += msg_token_count
                else:
                    logger.debug(f"Message '{msg.id}' (role: {msg.role.value if isinstance(msg.role, LLMCoreRole) else str(msg.role)}, tokens: {msg_token_count}) exceeds budget ({current_tokens + msg_token_count} > {available_tokens_for_prompt}). Not added.")
                    break

        await _add_to_candidates_if_budget(system_messages_hist)
        if self._prioritize_user_context_items:
            await _add_to_candidates_if_budget(active_user_item_messages)
            if rag_query_context_message_obj and self._rag_combination_strategy == "prepend_system":
                await _add_to_candidates_if_budget([rag_query_context_message_obj])
        else:
            if rag_query_context_message_obj and self._rag_combination_strategy == "prepend_system":
                await _add_to_candidates_if_budget([rag_query_context_message_obj])
            await _add_to_candidates_if_budget(active_user_item_messages)

        history_budget = available_tokens_for_prompt - current_tokens
        built_history, built_history_tokens = await self._build_history_messages(
            history_messages_all_non_system, provider, target_model, history_budget
        )
        await _add_to_candidates_if_budget(built_history)

        # --- Final Truncation Pass based on _truncation_priority_order ---
        final_candidate_messages = list(candidate_messages)

        for priority_type_to_truncate in self._truncation_priority_order:
            if current_tokens <= available_tokens_for_prompt: break
            logger.debug(f"Context over budget ({current_tokens}/{available_tokens_for_prompt}). Attempting truncation for type: '{priority_type_to_truncate}'.")
            items_removed_in_this_pass = False
            temp_final_candidates = list(final_candidate_messages)

            if priority_type_to_truncate == "rag":
                if rag_query_context_message_obj and rag_query_context_message_obj.id in [m.id for m in temp_final_candidates]:
                    logger.debug(f"Truncating by priority '{priority_type_to_truncate}': Removing RAG query context message.")
                    temp_final_candidates = [m for m in temp_final_candidates if m.id != rag_query_context_message_obj.id]
                    rag_documents_used_this_turn = None
                    items_removed_in_this_pass = True
            elif priority_type_to_truncate == "user_items":
                user_item_message_ids_in_context = {m.id for m in temp_final_candidates if m.id.startswith("user_item_")}
                sorted_active_user_item_msgs_for_trunc = sorted(
                    [msg for msg in active_user_item_messages if msg.id in user_item_message_ids_in_context],
                    key=lambda m: session.get_context_item(m.id.replace("user_item_","")).timestamp if session.get_context_item(m.id.replace("user_item_","")) else datetime.min.replace(tzinfo=timezone.utc)
                )
                for item_msg_to_remove in sorted_active_user_item_msgs_for_trunc:
                    if current_tokens <= available_tokens_for_prompt: break
                    if item_msg_to_remove.id in user_item_message_ids_in_context:
                        logger.debug(f"Truncating by priority '{priority_type_to_truncate}': Removing user item message '{item_msg_to_remove.id}'.")
                        temp_final_candidates = [m for m in temp_final_candidates if m.id != item_msg_to_remove.id]
                        items_removed_in_this_pass = True
                        current_tokens = await provider.count_message_tokens(temp_final_candidates, target_model)
            elif priority_type_to_truncate == "history":
                history_in_final_context = sorted(
                    [m for m in temp_final_candidates if not (m.role == LLMCoreRole.SYSTEM or m.id.startswith("user_item_") or m.id == "rag_query_context_marker")],
                    key=lambda m: m.timestamp
                )
                last_user_msg_from_session = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
                while len(history_in_final_context) > self._minimum_history_messages and current_tokens > available_tokens_for_prompt:
                    if not history_in_final_context: break
                    msg_to_remove_from_history = history_in_final_context.pop(0)
                    if last_user_msg_from_session and msg_to_remove_from_history.id == last_user_msg_from_session.id and \
                       len(history_in_final_context) < self._minimum_history_messages :
                        logger.debug(f"Protected last user message '{msg_to_remove_from_history.id}' from history truncation as it would violate minimum history count.")
                        history_in_final_context.insert(0, msg_to_remove_from_history)
                        break
                    logger.debug(f"Truncating by priority '{priority_type_to_truncate}': Removing history message '{msg_to_remove_from_history.id}'.")
                    temp_final_candidates = [m for m in temp_final_candidates if m.id != msg_to_remove_from_history.id]
                    items_removed_in_this_pass = True
                    current_tokens = await provider.count_message_tokens(temp_final_candidates, target_model)
                    if current_tokens <= available_tokens_for_prompt: break

            if items_removed_in_this_pass:
                final_candidate_messages = temp_final_candidates
                current_tokens = await provider.count_message_tokens(final_candidate_messages, target_model)
            else:
                 logger.debug(f"No items of priority type '{priority_type_to_truncate}' found to remove or already at minimums for that type.")

        last_user_msg_session = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        if last_user_msg_session:
            is_last_user_msg_present_in_final_context = any(m.id == last_user_msg_session.id for m in final_candidate_messages)
            if not is_last_user_msg_present_in_final_context:
                error_detail_msg = "Context length too short. Latest user message could not be included after all truncation attempts."
                logger.error(error_detail_msg + f" Limit: {available_tokens_for_prompt}, Current after truncation: {current_tokens}")
                raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message=error_detail_msg)

        if current_tokens > available_tokens_for_prompt:
            logger.error(f"Final context ({current_tokens} tokens) still exceeds budget ({available_tokens_for_prompt}) despite all truncation efforts.")
            raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message="Context exceeds token limit after all truncation attempts.")

        logger.info(f"Final prepared context: {len(final_candidate_messages)} messages, {current_tokens} tokens for model '{target_model}'.")
        final_payload_messages = [msg for msg in final_candidate_messages if msg.id != "rag_query_context_marker"]
        final_token_count = await provider.count_message_tokens(final_payload_messages, target_model)
        return final_payload_messages, rag_documents_used_this_turn, final_token_count

    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string for context injection."""
        if not documents: return ""
        sorted_documents = sorted(documents, key=lambda d: d.score if d.score is not None else float('inf'))
        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(sorted_documents):
            source_info = doc.metadata.get("source", f"Document {doc.id[:8]}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            content_snippet = ' '.join(doc.content.splitlines()).strip()
            context_parts.append(f"\n[Source: {source_info} {score_info}]\n{content_snippet}")
        context_parts.append("--- End Retrieved Documents ---")
        return "\n".join(context_parts)
