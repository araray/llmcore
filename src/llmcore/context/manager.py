# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and user-added context items.
Includes per-item truncation for user-added context and prompt template rendering.
Supports explicitly staged items for prioritized inclusion in context.
The prepare_context method now returns a detailed ContextPreparationDetails object.
"""

import asyncio
import logging
import os # For path operations if needed for template loading
from pathlib import Path # For path operations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]


from ..embedding.manager import EmbeddingManager
from ..exceptions import (ConfigError, ContextError, ContextLengthError,
                          EmbeddingError, ProviderError, VectorStorageError)
from ..models import (ChatSession, ContextDocument, ContextItem, # Added ContextPreparationDetails
                      ContextItemType, Message, ContextPreparationDetails)
from ..models import Role as LLMCoreRole
from ..providers.base import BaseProvider
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results and user-added context items,
    renders prompt templates, handles explicitly staged items, and ensures the final
    payload adheres to token limits using configurable strategies.
    Includes per-item truncation for user-added context.
    Dynamically selects embedding models for RAG queries based on collection metadata.
    The `prepare_context` method returns a `ContextPreparationDetails` object.
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
        (Docstring and implementation largely unchanged from previous version,
         initializes configuration values for context strategies.)
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

        # --- MODIFIED DEFAULT INCLUSION PRIORITY ---
        self._inclusion_priority_order: List[str] = self._parse_inclusion_priority(
            cm_config.get('inclusion_priority', "system_history,explicitly_staged,user_items_active,history_chat,final_user_query")
        )
        # --- END MODIFICATION ---
        self._truncation_priority_order: List[str] = self._parse_truncation_priority(
            cm_config.get('truncation_priority', 'history_chat,user_items_active,rag_in_query,explicitly_staged')
        )

        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)
        self._default_rag_k: int = cm_config.get('rag_retrieval_k', 3)
        self._user_retained_messages_count: int = cm_config.get('user_retained_messages_count', 5)
        self._prioritize_user_context_items: bool = cm_config.get('prioritize_user_context_items', True)
        self._max_chars_per_user_item: int = cm_config.get('max_chars_per_user_item', 40000)

        self._prompt_template_content: str
        default_template_str = cm_config.get(
            'default_prompt_template',
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        prompt_template_path_str = cm_config.get('prompt_template_path', "")

        if prompt_template_path_str:
            base_path_for_template = Path.cwd()
            main_config_file_origin = getattr(self._config, '_config_file_path_loaded_from', None)
            if main_config_file_origin:
                main_config_path_obj = Path(main_config_file_origin)
                if main_config_path_obj.is_file():
                    base_path_for_template = main_config_path_obj.parent
            resolved_template_path = Path(prompt_template_path_str)
            if not resolved_template_path.is_absolute():
                resolved_template_path = (base_path_for_template / prompt_template_path_str).resolve()
            if resolved_template_path.is_file():
                try:
                    self._prompt_template_content = resolved_template_path.read_text(encoding='utf-8')
                    logger.info(f"Loaded RAG prompt template from: {resolved_template_path}")
                except Exception as e:
                    logger.warning(f"Failed to read template '{resolved_template_path}': {e}. Using default.")
                    self._prompt_template_content = default_template_str
            else:
                logger.warning(f"Template file not found: '{resolved_template_path}'. Using default.")
                self._prompt_template_content = default_template_str
        else:
            logger.info("No prompt_template_path. Using default_prompt_template string.")
            self._prompt_template_content = default_template_str

        logger.info("ContextManager initialized.")
        logger.debug(f"Inclusion priority: {self._inclusion_priority_order}")
        logger.debug(f"Truncation priority: {self._truncation_priority_order}")
        self._validate_strategies()

    def _parse_inclusion_priority(self, priority_str: str) -> List[str]:
        """Parses the inclusion_priority string. (Implementation unchanged)"""
        valid_inclusions = {"system_history", "explicitly_staged", "user_items_active", "history_chat", "final_user_query"}
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_inclusions]
        if len(ordered_priorities) != len(priorities):
            invalid_found = set(priorities) - set(ordered_priorities)
            logger.warning(f"Invalid items in 'inclusion_priority': {invalid_found}. Using valid: {ordered_priorities}")
        if not ordered_priorities:
            # --- MODIFIED DEFAULT FALLBACK INCLUSION PRIORITY ---
            default_priority = ["system_history", "explicitly_staged", "user_items_active", "history_chat", "final_user_query"]
            # --- END MODIFICATION ---
            logger.warning(f"No valid 'inclusion_priority' items. Defaulting to: {default_priority}")
            return default_priority
        return ordered_priorities

    def _parse_truncation_priority(self, priority_str: str) -> List[str]:
        """Parses the truncation_priority string. (Implementation unchanged)"""
        valid_priorities = {"history_chat", "rag_in_query", "user_items_active", "explicitly_staged"}
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_priorities]
        legacy_map = {"history": "history_chat", "rag": "rag_in_query", "user_items": "user_items_active"}
        mapped_legacy_priorities = []
        made_legacy_mapping = False
        for p_val in priorities: # Iterate original list to preserve user's intended order as much as possible
            if p_val in legacy_map:
                mapped_val = legacy_map[p_val]
                if mapped_val in valid_priorities: mapped_legacy_priorities.append(mapped_val); made_legacy_mapping = True
                else: logger.warning(f"Legacy truncation item '{p_val}' has no valid new mapping. Skipping.")
            elif p_val in valid_priorities: mapped_legacy_priorities.append(p_val)
            # Invalid items are implicitly dropped here, will be caught by length check later

        if made_legacy_mapping:
            logger.info(f"Legacy truncation_priority items detected. Mapped '{priority_str}' to '{','.join(mapped_legacy_priorities)}'.")
            priorities_to_use = mapped_legacy_priorities
        else:
            priorities_to_use = priorities # Use original if no legacy mapping occurred

        ordered_priorities = [p for p in priorities_to_use if p in valid_priorities] # Final filter

        if len(ordered_priorities) != len(priorities_to_use): # Check if any invalid items were present
            invalid_found = set(priorities_to_use) - set(ordered_priorities)
            logger.warning(f"Invalid items found in 'truncation_priority' config: {invalid_found}. "
                           f"Using valid ones: {ordered_priorities}")
        if not ordered_priorities:
            default_priority = ["history_chat", "user_items_active", "rag_in_query", "explicitly_staged"]
            logger.warning(f"No valid 'truncation_priority' items configured. Defaulting to: {default_priority}")
            return default_priority
        return ordered_priorities

    def _validate_strategies(self):
        """Validates configured strategies. (Implementation unchanged)"""
        if self._history_selection_strategy not in ['last_n_tokens', 'last_n_messages']:
             logger.warning(f"Unsupported history_selection_strategy '{self._history_selection_strategy}'. Falling back to 'last_n_tokens'.")
             self._history_selection_strategy = 'last_n_tokens'

    async def _format_and_tokenize_item_as_message(
        self, item: Union[Message, ContextItem], provider: BaseProvider,
        target_model: str, item_category: str
    ) -> Message:
        """Formats ContextItem to Message or tokenizes existing Message. (Implementation unchanged)"""
        if isinstance(item, Message):
            if item.tokens is None: item.tokens = await provider.count_message_tokens([item], target_model)
            return item
        content_for_llm = item.content; item.is_truncated = False
        ignore_this_item_char_limit = item.metadata.get('ignore_char_limit', False)
        if not ignore_this_item_char_limit and len(item.content) > self._max_chars_per_user_item:
            content_for_llm = item.content[:self._max_chars_per_user_item]; item.is_truncated = True
        item_type_str = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
        source_desc_parts = []
        if item.type == ContextItemType.USER_FILE and item.metadata.get("filename"): source_desc_parts.append(f"Filename: {item.metadata['filename']}")
        elif item.type == ContextItemType.RAG_SNIPPET and item.metadata.get("original_rag_doc_id"): source_desc_parts.append(f"OriginalRAGID: {item.metadata['original_rag_doc_id']}")
        if item.source_id and item.source_id not in (item.metadata.get("filename"), item.metadata.get("original_rag_doc_id")): source_desc_parts.append(f"SourceID: {item.source_id}")
        source_desc = ", ".join(source_desc_parts) if source_desc_parts else item.id
        header_category_map = {"explicitly_staged": "Staged Context Item", "user_items_active": "User-Provided Context Item"}
        header_prefix = header_category_map.get(item_category, "Context Item")
        trunc_status_msg = " (TRUNCATED by char limit)" if item.is_truncated else ""
        header = f"--- {header_prefix} (ID: {item.id}, Type: {item_type_str}, Source: {source_desc}{trunc_status_msg}) ---"
        footer = f"--- End {header_prefix} (ID: {item.id}) ---"
        formatted_content = f"{header}\n{content_for_llm}\n{footer}"
        #msg = Message(role=LLMCoreRole.SYSTEM, content=formatted_content, session_id=item.metadata.get("session_id_for_message", "context_item_session"), id=f"{item_category}_{item.id}")
        msg = Message(role=LLMCoreRole.USER, content=formatted_content, session_id=item.metadata.get("session_id_for_message", "context_item_session"), id=f"{item_category}_{item.id}")
        msg.tokens = await provider.count_message_tokens([msg], target_model)
        if item.original_tokens is None:
            if item.is_truncated:
                temp_orig_msg_content = f"{header}\n{item.content}\n{footer}"; temp_orig_msg = Message(role=LLMCoreRole.SYSTEM, content=temp_orig_msg_content, session_id=msg.session_id, id=f"{msg.id}_orig")
                item.original_tokens = await provider.count_message_tokens([temp_orig_msg], target_model)
            else: item.original_tokens = msg.tokens
        return msg

    async def _build_history_messages(
        self, history_messages_all_non_system: List[Message], provider: BaseProvider,
        target_model: str, budget: int
    ) -> Tuple[List[Message], int]:
        """Builds the chat history part of context. (Implementation unchanged)"""
        selected_history: List[Message] = []; current_history_tokens = 0
        if budget <= 0 or not history_messages_all_non_system: return selected_history, current_history_tokens
        for msg in history_messages_all_non_system:
            if msg.tokens is None:
                try: msg.tokens = await provider.count_message_tokens([msg], target_model)
                except Exception as e_tok: logger.error(f"Token count failed for history msg '{msg.id}': {e_tok}. Approx."); msg.tokens = len(msg.content) // 4
        retained_for_now: List[Message] = []; tokens_for_retained = 0; num_user_messages_retained = 0; temp_retained_buffer: List[Message] = []
        for i in range(len(history_messages_all_non_system) - 1, -1, -1):
            msg = history_messages_all_non_system[i]; msg_tokens = msg.tokens or 0
            if tokens_for_retained + msg_tokens <= budget:
                temp_retained_buffer.insert(0, msg); tokens_for_retained += msg_tokens
                if msg.role == LLMCoreRole.USER:
                    num_user_messages_retained += 1
                    if self._user_retained_messages_count > 0 and num_user_messages_retained >= self._user_retained_messages_count: break
            else: break
        retained_for_now = temp_retained_buffer; selected_history.extend(retained_for_now); current_history_tokens = tokens_for_retained
        remaining_history_budget_for_backfill = budget - current_history_tokens
        if remaining_history_budget_for_backfill > 0 and self._history_selection_strategy == "last_n_tokens":
            ids_in_selected_history = {m.id for m in selected_history}
            older_history_candidates = [msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history]
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True)
            temp_backfill_history: List[Message] = []; tokens_for_backfill = 0
            for msg in older_history_candidates:
                msg_tokens = msg.tokens or 0
                if tokens_for_backfill + msg_tokens <= remaining_history_budget_for_backfill:
                    temp_backfill_history.insert(0, msg); tokens_for_backfill += msg_tokens
                else: break
            selected_history = temp_backfill_history + selected_history; current_history_tokens += tokens_for_backfill
            logger.debug(f"History: Retained {len(retained_for_now)} recent, backfilled {len(temp_backfill_history)} older. Total history tokens: {current_history_tokens}")
        else: logger.debug(f"History: Retained {len(retained_for_now)} recent. Total history tokens: {current_history_tokens}")
        return selected_history, current_history_tokens

    def _render_prompt_template(
        self, rag_context_str: str, question_str: str, custom_template_values: Optional[Dict[str, str]]
    ) -> str:
        """Renders the loaded prompt template. (Implementation unchanged)"""
        rendered_prompt = self._prompt_template_content
        rendered_prompt = rendered_prompt.replace("{context}", rag_context_str)
        rendered_prompt = rendered_prompt.replace("{question}", question_str)
        if custom_template_values:
            for key, value in custom_template_values.items():
                rendered_prompt = rendered_prompt.replace(f"{{{key}}}", str(value))
        return rendered_prompt

    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        message_inclusion_map: Optional[Dict[str, bool]] = None, # Added parameter
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> ContextPreparationDetails: # Changed return type
        """
        Prepares the context payload for the LLM, returning detailed information.
        This now respects the message_inclusion_map to filter chat history before
        any other history selection logic is applied.

        Args:
            (Args docstring is extensive, see previous versions or api.py for full details)
            message_inclusion_map: A map where keys are message IDs and values are booleans.
                                   If a message ID from history is in the map with a value
                                   of `False`, it will be excluded from the context.
                                   If a message ID is not in the map, it is included by default.

        Returns:
            A `ContextPreparationDetails` object containing the final messages,
            token counts, RAG info, and truncation details.
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for context (provider: {provider.get_name()}).")

        max_model_tokens = provider.get_max_context_length(target_model)
        available_tokens_for_prompt = max_model_tokens - self._reserved_response_tokens
        if available_tokens_for_prompt <= 0:
             raise ContextError(f"Config error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_model_tokens}).")

        logger.debug(f"Preparing context for model '{target_model}'. Available prompt tokens: {available_tokens_for_prompt}")

        # Initialize for ContextPreparationDetails
        truncation_actions: Dict[str, Any] = {"details": []} # Store detailed truncation log

        components: Dict[str, List[Message]] = {cat: [] for cat in self._inclusion_priority_order}
        component_tokens: Dict[str, int] = {cat: 0 for cat in self._inclusion_priority_order}
        rag_documents_used_this_turn: Optional[List[ContextDocument]] = None
        rendered_rag_query_content: Optional[str] = None


        # --- 1. Gather and Prepare All Potential Components ---
        # System messages from session history
        for msg in session.messages:
            if msg.role == LLMCoreRole.SYSTEM:
                if msg.tokens is None: msg.tokens = await provider.count_message_tokens([msg], target_model)
                components["system_history"].append(msg)
                component_tokens["system_history"] += msg.tokens or 0

        # Explicitly staged items
        if explicitly_staged_items:
            for item_idx, item in enumerate(explicitly_staged_items):
                # Ensure item has a unique ID if it's a direct Message without one yet (though unlikely for staged)
                if isinstance(item, Message) and not hasattr(item, 'id'): item.id = f"staged_msg_{item_idx}"
                elif isinstance(item, ContextItem) and not hasattr(item, 'id'): item.id = f"staged_ctx_item_{item_idx}"

                formatted_msg = await self._format_and_tokenize_item_as_message(item, provider, target_model, "explicitly_staged")
                components["explicitly_staged"].append(formatted_msg)
                component_tokens["explicitly_staged"] += formatted_msg.tokens or 0

        # Active user items from session pool (workspace)
        if active_context_item_ids and session.context_items:
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item:
                    formatted_msg = await self._format_and_tokenize_item_as_message(item, provider, target_model, "user_items_active")
                    components["user_items_active"].append(formatted_msg)
                    component_tokens["user_items_active"] += formatted_msg.tokens or 0

        # Prepare Final User Query (with or without RAG)
        last_user_message_obj = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        if not last_user_message_obj:
            raise ContextError("Cannot prepare context without a current user query in the session.")
        actual_query_for_llm = last_user_message_obj.content
        final_user_query_content = actual_query_for_llm

        if rag_enabled:
            # (RAG retrieval logic)
            query_text_for_rag = actual_query_for_llm
            k_val = rag_k if rag_k is not None else self._default_rag_k
            actual_rag_collection_name = rag_collection or self._storage_manager.get_vector_storage()._default_collection_name # type: ignore
            query_embedding_model_identifier: Optional[str] = None
            try:
                collection_meta = await self._storage_manager.get_vector_storage().get_collection_metadata(actual_rag_collection_name)
                if collection_meta:
                    coll_emb_provider = collection_meta.get("embedding_model_provider"); coll_emb_model_name = collection_meta.get("embedding_model_name")
                    if coll_emb_provider and coll_emb_model_name: query_embedding_model_identifier = f"{coll_emb_provider}:{coll_emb_model_name}" if coll_emb_provider.lower() != "sentence-transformers" else coll_emb_model_name
            except Exception as e_meta: logger.error(f"RAG: Error retrieving collection metadata for '{actual_rag_collection_name}': {e_meta}. Falling back to default embedding model.")
            try:
                query_embedding = await self._embedding_manager.generate_embedding(query_text_for_rag, model_identifier=query_embedding_model_identifier)
                retrieved_rag_docs = await self._storage_manager.get_vector_storage().similarity_search(query_embedding=query_embedding, k=k_val, collection_name=actual_rag_collection_name, filter_metadata=rag_metadata_filter)
                rag_documents_used_this_turn = retrieved_rag_docs
                rag_formatted_docs_str = self._format_rag_docs_for_context(retrieved_rag_docs)
                final_user_query_content = self._render_prompt_template(rag_formatted_docs_str, actual_query_for_llm, prompt_template_values)
                rendered_rag_query_content = final_user_query_content # Store for return
                logger.info(f"RAG retrieved {len(retrieved_rag_docs)} docs. Query content rendered with template.")
            except Exception as e_rag:
                logger.error(f"RAG retrieval or processing failed: {e_rag}. Using plain user query.", exc_info=True)
                rag_documents_used_this_turn = []
                rendered_rag_query_content = None # No RAG template used

        final_query_message = Message(role=LLMCoreRole.USER, content=final_user_query_content, session_id=session.id, id=f"final_query_{last_user_message_obj.id}")
        final_query_message.tokens = await provider.count_message_tokens([final_query_message], target_model)
        components["final_user_query"] = [final_query_message]
        component_tokens["final_user_query"] = final_query_message.tokens or 0


        # --- 2. Initial Assembly & Budget for History ---
        history_chat_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM and msg.id != last_user_message_obj.id]

        # --- FEAT-01 Implementation: Filter history based on inclusion map ---
        if message_inclusion_map:
            initial_history_count = len(history_chat_messages_all_non_system)
            history_chat_messages_all_non_system = [
                msg for msg in history_chat_messages_all_non_system
                if message_inclusion_map.get(msg.id, True)  # Default to True (include) if not in map
            ]
            final_history_count = len(history_chat_messages_all_non_system)
            logger.info(
                f"Filtered chat history using message_inclusion_map. "
                f"Retained {final_history_count} of {initial_history_count} messages for context selection."
            )
        # --- End FEAT-01 Implementation ---

        final_payload_messages: List[Message] = []
        current_tokens_in_payload = 0

        tokens_for_non_history_components = 0
        for category_key in self._inclusion_priority_order:
            if category_key != "history_chat":
                tokens_for_non_history_components += component_tokens.get(category_key, 0)

        budget_for_history = available_tokens_for_prompt - tokens_for_non_history_components

        built_history_messages, built_history_tokens = await self._build_history_messages(history_chat_messages_all_non_system, provider, target_model, budget_for_history)
        components["history_chat"] = built_history_messages
        component_tokens["history_chat"] = built_history_tokens

        for category_key in self._inclusion_priority_order:
            for msg in components[category_key]:
                msg_tokens = msg.tokens or 0
                if current_tokens_in_payload + msg_tokens <= available_tokens_for_prompt:
                    final_payload_messages.append(msg)
                    current_tokens_in_payload += msg_tokens
                else: pass # Let truncation handle

        # --- 3. Truncation if Over Budget (existing logic adapted, now populates truncation_actions) ---
        if current_tokens_in_payload > available_tokens_for_prompt:
            logger.info(f"Context over budget ({current_tokens_in_payload}/{available_tokens_for_prompt}). Applying truncation (priority: {self._truncation_priority_order}).")

            for category_to_truncate in self._truncation_priority_order:
                if current_tokens_in_payload <= available_tokens_for_prompt: break
                tokens_to_free_needed = current_tokens_in_payload - available_tokens_for_prompt

                original_count_in_category = len(components[category_to_truncate])
                truncated_list, freed_tokens = self._truncate_message_list_from_start(components[category_to_truncate], tokens_to_free_needed)

                if freed_tokens > 0:
                    removed_count = original_count_in_category - len(truncated_list)
                    truncation_actions["details"].append(
                        f"Truncated '{category_to_truncate}': removed {removed_count} item(s), freed {freed_tokens} tokens."
                    )
                    if category_to_truncate not in truncation_actions: truncation_actions[category_to_truncate] = {}
                    truncation_actions[category_to_truncate]["removed_count"] = truncation_actions[category_to_truncate].get("removed_count",0) + removed_count
                    truncation_actions[category_to_truncate]["tokens_freed"] = truncation_actions[category_to_truncate].get("tokens_freed",0) + freed_tokens

                    components[category_to_truncate] = truncated_list
                    component_tokens[category_to_truncate] -= freed_tokens
                    current_tokens_in_payload -= freed_tokens
                    logger.debug(f"Truncated '{category_to_truncate}' by {freed_tokens} tokens.")

                # Special handling for rag_in_query
                if category_to_truncate == "rag_in_query" and rag_enabled and freed_tokens == 0: # If RAG didn't get truncated by list removal
                    # Check if removing RAG from query helps
                    plain_query_msg = Message(role=LLMCoreRole.USER, content=actual_query_for_llm, session_id=session.id, id=f"plain_query_{last_user_message_obj.id}")
                    plain_query_msg.tokens = await provider.count_message_tokens([plain_query_msg], target_model)
                    original_final_query_tokens = component_tokens.get("final_user_query", 0) # Should exist

                    if (current_tokens_in_payload - original_final_query_tokens + (plain_query_msg.tokens or 0)) <= available_tokens_for_prompt:
                        freed_by_removing_rag = original_final_query_tokens - (plain_query_msg.tokens or 0)
                        if freed_by_removing_rag > 0:
                            components["final_user_query"] = [plain_query_msg]
                            component_tokens["final_user_query"] = plain_query_msg.tokens or 0
                            current_tokens_in_payload -= freed_by_removing_rag
                            rag_documents_used_this_turn = []
                            rendered_rag_query_content = None # RAG template no longer used
                            logger.info(f"Truncated 'rag_in_query': Removed RAG context, using plain query. Freed approx {freed_by_removing_rag} tokens.")
                            truncation_actions["details"].append(f"Simplified 'rag_in_query': removed RAG, freed {freed_by_removing_rag} tokens.")
                            truncation_actions["rag_in_query_simplified"] = True
                        else: logger.debug("Tried to truncate RAG, but plain query is not smaller.")
                    else: logger.warning("Cannot truncate 'rag_in_query' further; plain query also exceeds budget.")

            # Re-assemble final_payload_messages after truncation
            final_payload_messages = []
            current_tokens_in_payload = 0 # Recalculate
            for category_key in self._inclusion_priority_order:
                for msg in components[category_key]:
                    msg_tokens = msg.tokens or 0
                    if current_tokens_in_payload + msg_tokens <= available_tokens_for_prompt:
                        final_payload_messages.append(msg)
                        current_tokens_in_payload += msg_tokens
                    else: break # Stop adding from this category if budget exceeded
                if current_tokens_in_payload > available_tokens_for_prompt:
                    logger.debug(f"Budget exceeded while processing category '{category_key}'. Further categories in inclusion_priority might be skipped.")
                    # This break is important: if a high-priority category fills the budget,
                    # lower-priority ones won't be added, which is correct.
                    # However, the truncation logic above should have already handled this.
                    # This re-assembly is more about respecting the order after truncation.

        if current_tokens_in_payload > available_tokens_for_prompt:
            logger.error(f"Context still too long ({current_tokens_in_payload}/{available_tokens_for_prompt}) after all truncation.")
            raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens_in_payload,
                                     message="Context exceeds token limit after all truncation attempts.")

        # Ensure minimum history messages (soft constraint)
        actual_history_in_payload = [m for m in final_payload_messages if m in components["history_chat"]]
        if len(actual_history_in_payload) < self._minimum_history_messages:
            logger.debug(f"Final payload has {len(actual_history_in_payload)} history_chat messages (min: {self._minimum_history_messages}).")

        final_token_count_for_payload = await provider.count_message_tokens(final_payload_messages, target_model)
        logger.info(f"Final prepared context: {len(final_payload_messages)} messages, {final_token_count_for_payload} tokens for model '{target_model}'.")

        return ContextPreparationDetails(
            prepared_messages=final_payload_messages,
            final_token_count=final_token_count_for_payload,
            max_tokens_for_model=max_model_tokens, # Use max_model_tokens, not available_tokens_for_prompt
            rag_documents_used=rag_documents_used_this_turn,
            rendered_rag_template_content=rendered_rag_query_content if rag_enabled else None,
            truncation_actions_taken=truncation_actions
        )

    def _truncate_message_list_from_start(self, messages: List[Message], tokens_to_free: int) -> Tuple[List[Message], int]:
        """Removes messages from START of list until tokens_to_free met. (Implementation unchanged)"""
        freed_so_far = 0; truncated_list = list(messages)
        while freed_so_far < tokens_to_free and truncated_list:
            msg_to_remove = truncated_list.pop(0); freed_so_far += msg_to_remove.tokens or 0
        return truncated_list, freed_so_far

    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats RAG documents into a string. (Implementation unchanged)"""
        if not documents: return ""
        sorted_documents = sorted(documents, key=lambda d: d.score if d.score is not None else float('inf'))
        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(sorted_documents):
            source_info_parts = []
            if doc.metadata and doc.metadata.get("source_file_path_relative"): source_info_parts.append(f"File: {doc.metadata.get('source_file_path_relative')}")
            elif doc.metadata and doc.metadata.get("source"): source_info_parts.append(f"Source: {doc.metadata.get('source')}")
            else: source_info_parts.append(f"DocID: {doc.id[:12]}")
            if doc.metadata and doc.metadata.get("start_line"): source_info_parts.append(f"Line: {doc.metadata.get('start_line')}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            header = f"Context Document {i+1}: [{', '.join(source_info_parts)}] {score_info}"
            content_snippet = ' '.join(doc.content.splitlines()).strip()
            context_parts.append(f"\n{header}\n{content_snippet}")
        context_parts.append("\n--- End Retrieved Documents ---")
        return "\n".join(context_parts)
