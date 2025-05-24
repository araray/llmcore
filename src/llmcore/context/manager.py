# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and user-added context items.
Includes per-item truncation for user-added context and prompt template rendering.
Supports explicitly staged items for prioritized inclusion in context.
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
    renders prompt templates, handles explicitly staged items, and ensures the final
    payload adheres to token limits using configurable strategies.
    Includes per-item truncation for user-added context.
    Dynamically selects embedding models for RAG queries based on collection metadata.
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
        self._config = config # Full ConfyConfig object
        self._provider_manager = provider_manager
        self._storage_manager = storage_manager
        self._embedding_manager = embedding_manager

        cm_config = self._config.get('context_management', {})
        self._reserved_response_tokens: int = cm_config.get('reserved_response_tokens', 500)
        self._history_selection_strategy: str = cm_config.get('history_selection_strategy', 'last_n_tokens')

        # New: Inclusion priority order
        self._inclusion_priority_order: List[str] = self._parse_inclusion_priority(
            cm_config.get('inclusion_priority', "system_history,final_user_query,explicitly_staged,user_items_active,history_chat")
        )
        # Updated: Truncation priority order (now includes 'explicitly_staged' and 'rag_in_query')
        self._truncation_priority_order: List[str] = self._parse_truncation_priority(
            cm_config.get('truncation_priority', 'history_chat,user_items_active,rag_in_query,explicitly_staged')
        )

        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)
        self._default_rag_k: int = cm_config.get('rag_retrieval_k', 3)
        # self._rag_combination_strategy: str = cm_config.get('rag_combination_strategy', 'prepend_system') # Less relevant with templates
        self._user_retained_messages_count: int = cm_config.get('user_retained_messages_count', 5)
        self._prioritize_user_context_items: bool = cm_config.get('prioritize_user_context_items', True) # Note: inclusion_priority is more direct
        self._max_chars_per_user_item: int = cm_config.get('max_chars_per_user_item', 40000)

        # --- Load Prompt Template (existing logic) ---
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
        # --- End Load Prompt Template ---

        logger.info("ContextManager initialized.")
        logger.debug(f"Inclusion priority: {self._inclusion_priority_order}")
        logger.debug(f"Truncation priority: {self._truncation_priority_order}")
        self._validate_strategies()

    def _parse_inclusion_priority(self, priority_str: str) -> List[str]:
        """Parses the inclusion_priority string from config into a list of valid types."""
        valid_inclusions = {"system_history", "explicitly_staged", "user_items_active", "history_chat", "final_user_query"}
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_inclusions]
        if len(ordered_priorities) != len(priorities):
            invalid_found = set(priorities) - set(ordered_priorities)
            logger.warning(f"Invalid items in 'inclusion_priority': {invalid_found}. Using valid: {ordered_priorities}")
        if not ordered_priorities:
            default_priority = ["system_history", "final_user_query", "explicitly_staged", "user_items_active", "history_chat"]
            logger.warning(f"No valid 'inclusion_priority' items. Defaulting to: {default_priority}")
            return default_priority
        return ordered_priorities

    def _parse_truncation_priority(self, priority_str: str) -> List[str]:
        """
        Parses the truncation_priority string, now supporting 'explicitly_staged' and 'rag_in_query'.
        """
        valid_priorities = {"history_chat", "rag_in_query", "user_items_active", "explicitly_staged"} # Updated valid set
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_priorities]

        # Legacy "history" and "rag" mapping
        legacy_map = {"history": "history_chat", "rag": "rag_in_query", "user_items": "user_items_active"}
        mapped_legacy_priorities = []
        made_legacy_mapping = False
        for p_idx, p_val in enumerate(priorities):
            if p_val in legacy_map:
                mapped_val = legacy_map[p_val]
                if mapped_val in valid_priorities:
                    mapped_legacy_priorities.append(mapped_val)
                    made_legacy_mapping = True
                else: # Should not happen if legacy_map keys are subset of valid_priorities
                    logger.warning(f"Legacy truncation item '{p_val}' has no valid new mapping. Skipping.")
            elif p_val in valid_priorities:
                 mapped_legacy_priorities.append(p_val) # Already a new valid item
            # Else: it's an invalid item, will be caught later

        if made_legacy_mapping:
            logger.info(f"Legacy truncation_priority items detected. Mapped '{priority_str}' to '{','.join(mapped_legacy_priorities)}'.")
            priorities = mapped_legacy_priorities
            ordered_priorities = [p for p in priorities if p in valid_priorities] # Re-filter based on new valid set

        if len(ordered_priorities) != len(priorities):
            invalid_found = set(priorities) - set(ordered_priorities)
            logger.warning(f"Invalid items found in 'truncation_priority' config: {invalid_found}. "
                           f"Using valid ones: {ordered_priorities}")
        if not ordered_priorities:
            default_priority = ["history_chat", "user_items_active", "rag_in_query", "explicitly_staged"] # Default order
            logger.warning(f"No valid 'truncation_priority' items configured. Defaulting to: {default_priority}")
            return default_priority
        return ordered_priorities

    def _validate_strategies(self):
        """Validates configured strategies and logs warnings if unsupported."""
        if self._history_selection_strategy not in ['last_n_tokens', 'last_n_messages']:
             logger.warning(f"Unsupported history_selection_strategy '{self._history_selection_strategy}'. Falling back to 'last_n_tokens'.")
             self._history_selection_strategy = 'last_n_tokens'
        # Other strategy validations if needed

    async def _format_and_tokenize_item_as_message(
        self,
        item: Union[Message, ContextItem],
        provider: BaseProvider,
        target_model: str,
        item_category: str # e.g., "explicitly_staged", "user_items_active"
    ) -> Message:
        """
        Formats a ContextItem into a Message object (SYSTEM role) or ensures tokens for an existing Message.
        """
        if isinstance(item, Message):
            if item.tokens is None:
                item.tokens = await provider.count_message_tokens([item], target_model)
            return item

        # It's a ContextItem
        # Apply per-item character truncation if not ignored
        content_for_llm = item.content
        item.is_truncated = False # Reset before check
        ignore_this_item_char_limit = item.metadata.get('ignore_char_limit', False)

        if not ignore_this_item_char_limit and len(item.content) > self._max_chars_per_user_item:
            content_for_llm = item.content[:self._max_chars_per_user_item]
            item.is_truncated = True
            logger.debug(f"ContextItem '{item.id}' (type: {item.type.value}) content truncated from {len(item.content)} to {len(content_for_llm)} chars (limit: {self._max_chars_per_user_item}).")

        item_type_str = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
        source_desc_parts = []
        if item.type == ContextItemType.USER_FILE and item.metadata.get("filename"):
            source_desc_parts.append(f"Filename: {item.metadata['filename']}")
        elif item.type == ContextItemType.RAG_SNIPPET and item.metadata.get("original_rag_doc_id"):
            source_desc_parts.append(f"OriginalRAGID: {item.metadata['original_rag_doc_id']}")
        if item.source_id and item.source_id not in (item.metadata.get("filename"), item.metadata.get("original_rag_doc_id")):
             source_desc_parts.append(f"SourceID: {item.source_id}")

        source_desc = ", ".join(source_desc_parts) if source_desc_parts else item.id

        header_category_map = {
            "explicitly_staged": "Staged Context Item",
            "user_items_active": "User-Provided Context Item"
        }
        header_prefix = header_category_map.get(item_category, "Context Item")

        trunc_status_msg = " (TRUNCATED by char limit)" if item.is_truncated else ""
        header = f"--- {header_prefix} (ID: {item.id}, Type: {item_type_str}, Source: {source_desc}{trunc_status_msg}) ---"
        footer = f"--- End {header_prefix} (ID: {item.id}) ---"

        formatted_content = f"{header}\n{content_for_llm}\n{footer}"

        # Create a new Message object for this formatted ContextItem
        # Using SYSTEM role for these items as they are providing context, not direct user input.
        msg = Message(
            role=LLMCoreRole.SYSTEM,
            content=formatted_content,
            session_id=item.metadata.get("session_id_for_message", "context_item_session"), # Needs a session_id
            id=f"{item_category}_{item.id}" # Unique ID for this message representation
        )
        msg.tokens = await provider.count_message_tokens([msg], target_model)

        # Update original_tokens on the ContextItem if not already set
        if item.original_tokens is None:
            # Estimate original tokens based on untruncated content if it was truncated
            if item.is_truncated:
                temp_orig_msg_content = f"{header}\n{item.content}\n{footer}" # Use full item.content
                temp_orig_msg = Message(role=LLMCoreRole.SYSTEM, content=temp_orig_msg_content, session_id=msg.session_id, id=f"{msg.id}_orig")
                item.original_tokens = await provider.count_message_tokens([temp_orig_msg], target_model)
            else:
                item.original_tokens = msg.tokens
        return msg

    async def _build_history_messages(
        self,
        history_messages_all_non_system: List[Message],
        provider: BaseProvider,
        target_model: str,
        budget: int
    ) -> Tuple[List[Message], int]:
        """
        Builds the chat history part of the context. (Existing logic largely reused)
        Ensures messages have token counts before processing.
        """
        selected_history: List[Message] = []
        current_history_tokens = 0

        if budget <= 0 or not history_messages_all_non_system:
            return selected_history, current_history_tokens

        for msg in history_messages_all_non_system:
            if msg.tokens is None:
                try:
                    msg.tokens = await provider.count_message_tokens([msg], target_model)
                except Exception as e_tok:
                    logger.error(f"Failed to count tokens for history message '{msg.id}': {e_tok}. Approximating.")
                    msg.tokens = len(msg.content) // 4

        retained_for_now: List[Message] = []
        tokens_for_retained = 0
        num_user_messages_retained = 0
        temp_retained_buffer: List[Message] = []

        for i in range(len(history_messages_all_non_system) - 1, -1, -1):
            msg = history_messages_all_non_system[i]
            msg_tokens = msg.tokens or 0

            if tokens_for_retained + msg_tokens <= budget:
                temp_retained_buffer.insert(0, msg)
                tokens_for_retained += msg_tokens
                if msg.role == LLMCoreRole.USER:
                    num_user_messages_retained += 1
                    if self._user_retained_messages_count > 0 and \
                       num_user_messages_retained >= self._user_retained_messages_count:
                        break
            else:
                break

        retained_for_now = temp_retained_buffer
        selected_history.extend(retained_for_now)
        current_history_tokens = tokens_for_retained

        remaining_history_budget_for_backfill = budget - current_history_tokens
        if remaining_history_budget_for_backfill > 0 and self._history_selection_strategy == "last_n_tokens": # Only backfill for last_n_tokens
            ids_in_selected_history = {m.id for m in selected_history}
            older_history_candidates = [
                msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history
            ]
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True)

            temp_backfill_history: List[Message] = []
            tokens_for_backfill = 0
            for msg in older_history_candidates:
                msg_tokens = msg.tokens or 0
                if tokens_for_backfill + msg_tokens <= remaining_history_budget_for_backfill:
                    temp_backfill_history.insert(0, msg)
                    tokens_for_backfill += msg_tokens
                else:
                    break
            selected_history = temp_backfill_history + selected_history
            current_history_tokens += tokens_for_backfill
            logger.debug(f"History: Retained {len(retained_for_now)} recent msgs, backfilled {len(temp_backfill_history)} older msgs. Total history tokens: {current_history_tokens}")
        else:
            logger.debug(f"History: Retained {len(retained_for_now)} recent msgs. Total history tokens: {current_history_tokens}")

        # Ensure minimum history messages if budget allows (after primary selection)
        if len(selected_history) < self._minimum_history_messages and budget > current_history_tokens:
            # This part needs careful integration with the above logic or to be a separate pass
            # For now, the _build_history_messages focuses on budget and user_retained_messages_count
            pass

        return selected_history, current_history_tokens

    def _render_prompt_template(
        self,
        rag_context_str: str,
        question_str: str,
        custom_template_values: Optional[Dict[str, str]]
    ) -> str:
        """Renders the loaded prompt template. (Existing logic)"""
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
        active_context_item_ids: Optional[List[str]] = None, # From session.context_items (workspace)
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None, # New parameter
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Message], Optional[List[ContextDocument]], int]:
        """
        Prepares the context payload for the LLM, incorporating explicitly staged items.
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Target model undetermined for context (provider: {provider.get_name()}).")

        max_context_tokens = provider.get_max_context_length(target_model)
        available_tokens_for_prompt = max_context_tokens - self._reserved_response_tokens
        if available_tokens_for_prompt <= 0:
             raise ContextError(f"Config error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_context_tokens}).")

        logger.debug(f"Preparing context for model '{target_model}'. Available prompt tokens: {available_tokens_for_prompt}")

        # --- 1. Gather and Prepare All Potential Components ---
        components: Dict[str, List[Message]] = {
            "system_history": [], "explicitly_staged": [], "user_items_active": [],
            "history_chat": [], "final_user_query": []
        }
        component_tokens: Dict[str, int] = {k: 0 for k in components}
        rag_documents_used_this_turn: Optional[List[ContextDocument]] = None

        # System messages from session history
        for msg in session.messages:
            if msg.role == LLMCoreRole.SYSTEM:
                if msg.tokens is None: msg.tokens = await provider.count_message_tokens([msg], target_model)
                components["system_history"].append(msg)
                component_tokens["system_history"] += msg.tokens or 0

        # Explicitly staged items
        if explicitly_staged_items:
            for item in explicitly_staged_items:
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
        final_user_query_content = actual_query_for_llm # Default to plain query

        if rag_enabled:
            query_text_for_rag = actual_query_for_llm
            k_val = rag_k if rag_k is not None else self._default_rag_k
            actual_rag_collection_name = rag_collection or self._storage_manager.get_vector_storage()._default_collection_name # type: ignore
            query_embedding_model_identifier: Optional[str] = None # Logic to determine this based on collection metadata
            try:
                collection_meta = await self._storage_manager.get_vector_storage().get_collection_metadata(actual_rag_collection_name)
                if collection_meta:
                    coll_emb_provider = collection_meta.get("embedding_model_provider")
                    coll_emb_model_name = collection_meta.get("embedding_model_name")
                    if coll_emb_provider and coll_emb_model_name:
                        query_embedding_model_identifier = f"{coll_emb_provider}:{coll_emb_model_name}" if coll_emb_provider.lower() != "sentence-transformers" else coll_emb_model_name
            except Exception as e_meta: logger.error(f"RAG: Error retrieving collection metadata for '{actual_rag_collection_name}': {e_meta}. Falling back to default embedding model.")

            try:
                query_embedding = await self._embedding_manager.generate_embedding(query_text_for_rag, model_identifier=query_embedding_model_identifier)
                retrieved_rag_docs = await self._storage_manager.get_vector_storage().similarity_search(
                    query_embedding=query_embedding, k=k_val, collection_name=actual_rag_collection_name, filter_metadata=rag_metadata_filter
                )
                rag_documents_used_this_turn = retrieved_rag_docs
                rag_formatted_docs_str = self._format_rag_docs_for_context(retrieved_rag_docs) # This might need budget later
                final_user_query_content = self._render_prompt_template(rag_formatted_docs_str, actual_query_for_llm, prompt_template_values)
                logger.info(f"RAG retrieved {len(retrieved_rag_docs)} docs. Query content rendered with template.")
            except Exception as e_rag:
                logger.error(f"RAG retrieval or processing failed: {e_rag}. Using plain user query.", exc_info=True)
                rag_documents_used_this_turn = [] # Mark as empty
                # final_user_query_content remains actual_query_for_llm

        final_query_message = Message(role=LLMCoreRole.USER, content=final_user_query_content, session_id=session.id, id=f"final_query_{last_user_message_obj.id}")
        final_query_message.tokens = await provider.count_message_tokens([final_query_message], target_model)
        components["final_user_query"].append(final_query_message)
        component_tokens["final_user_query"] = final_query_message.tokens or 0

        # --- 2. Initial Assembly based on Inclusion Priority & Budget for History ---
        final_payload_messages: List[Message] = []
        current_tokens_in_payload = 0

        # Calculate tokens used by high-priority, non-history components
        tokens_for_non_history_components = 0
        for category_key in self._inclusion_priority_order:
            if category_key != "history_chat": # Exclude history for this sum
                tokens_for_non_history_components += component_tokens.get(category_key, 0)

        budget_for_history = available_tokens_for_prompt - tokens_for_non_history_components

        history_chat_messages_all_non_system = [
            msg for msg in session.messages
            if msg.role != LLMCoreRole.SYSTEM and msg.id != last_user_message_obj.id
        ]
        built_history_messages, built_history_tokens = await self._build_history_messages(
            history_chat_messages_all_non_system, provider, target_model, budget_for_history
        )
        components["history_chat"] = built_history_messages
        component_tokens["history_chat"] = built_history_tokens

        # Assemble according to inclusion priority
        for category_key in self._inclusion_priority_order:
            for msg in components[category_key]:
                msg_tokens = msg.tokens or 0 # Should be tokenized by now
                if current_tokens_in_payload + msg_tokens <= available_tokens_for_prompt:
                    final_payload_messages.append(msg)
                    current_tokens_in_payload += msg_tokens
                else:
                    logger.debug(f"Component '{category_key}' message '{msg.id}' (tokens: {msg_tokens}) "
                                 f"exceeded budget during initial assembly. Total: {current_tokens_in_payload}, Available: {available_tokens_for_prompt}")
                    # Optionally break here or let truncation handle it.
                    # For now, let truncation handle it to respect priority fully.
                    # However, this means we might add something that gets immediately truncated.
                    # A better approach: if a category itself cannot fit, subsequent lower-priority categories are skipped.
                    # This is implicitly handled if we just add what fits.
                    pass # Let truncation handle overflow

        # --- 3. Truncation if Over Budget ---
        if current_tokens_in_payload > available_tokens_for_prompt:
            logger.info(f"Context over budget ({current_tokens_in_payload}/{available_tokens_for_prompt}). Applying truncation (priority: {self._truncation_priority_order}).")

            # Create a mutable list of messages with their categories for easier truncation
            categorized_messages_for_truncation: List[Tuple[str, Message]] = []
            temp_payload_for_cat_tagging: List[Message] = []
            # Re-assemble based on inclusion to tag messages correctly for truncation
            for cat_key_incl in self._inclusion_priority_order:
                for msg_incl in components[cat_key_incl]: # Use the potentially budget-fitted components
                    # Check if this message actually made it into final_payload_messages initially
                    # This is tricky if initial assembly was greedy.
                    # Simpler: assume components dictionary holds what *could* be included.
                    # We need to truncate from these source component lists.
                    pass # This logic needs refinement.

            # Simplified truncation: iterate through final_payload_messages and remove based on category.
            # This requires knowing the category of each message in final_payload_messages.
            # For now, we'll apply truncation to the *source* component lists and then re-assemble.

            original_component_tokens = component_tokens.copy() # For logging later

            for category_to_truncate in self._truncation_priority_order:
                if current_tokens_in_payload <= available_tokens_for_prompt: break

                tokens_to_free_needed = current_tokens_in_payload - available_tokens_for_prompt

                if category_to_truncate == "history_chat":
                    # Truncate from the start (oldest) of components["history_chat"]
                    truncated_list, freed_tokens = self._truncate_message_list_from_start(components["history_chat"], tokens_to_free_needed)
                    if freed_tokens > 0:
                        components["history_chat"] = truncated_list
                        component_tokens["history_chat"] -= freed_tokens
                        current_tokens_in_payload -= freed_tokens
                        logger.debug(f"Truncated 'history_chat' by {freed_tokens} tokens.")

                elif category_to_truncate == "user_items_active":
                    # Truncate from start (oldest by timestamp, or first added)
                    truncated_list, freed_tokens = self._truncate_message_list_from_start(components["user_items_active"], tokens_to_free_needed)
                    if freed_tokens > 0:
                        components["user_items_active"] = truncated_list
                        component_tokens["user_items_active"] -= freed_tokens
                        current_tokens_in_payload -= freed_tokens
                        logger.debug(f"Truncated 'user_items_active' by {freed_tokens} tokens.")

                elif category_to_truncate == "explicitly_staged":
                    truncated_list, freed_tokens = self._truncate_message_list_from_start(components["explicitly_staged"], tokens_to_free_needed)
                    if freed_tokens > 0:
                        components["explicitly_staged"] = truncated_list
                        component_tokens["explicitly_staged"] -= freed_tokens
                        current_tokens_in_payload -= freed_tokens
                        logger.debug(f"Truncated 'explicitly_staged' by {freed_tokens} tokens.")

                elif category_to_truncate == "rag_in_query" and rag_enabled:
                    # This means the final_user_query (which includes RAG) is too large.
                    # Simplest: remove RAG and use plain query.
                    plain_query_msg = Message(role=LLMCoreRole.USER, content=actual_query_for_llm, session_id=session.id, id=f"plain_query_{last_user_message_obj.id}")
                    plain_query_msg.tokens = await provider.count_message_tokens([plain_query_msg], target_model)

                    original_final_query_tokens = component_tokens["final_user_query"]

                    if (current_tokens_in_payload - original_final_query_tokens + (plain_query_msg.tokens or 0)) <= available_tokens_for_prompt:
                        freed_by_removing_rag = original_final_query_tokens - (plain_query_msg.tokens or 0)
                        if freed_by_removing_rag > 0 : # Only if it actually frees tokens
                            components["final_user_query"] = [plain_query_msg]
                            component_tokens["final_user_query"] = plain_query_msg.tokens or 0
                            current_tokens_in_payload -= freed_by_removing_rag
                            rag_documents_used_this_turn = [] # RAG was removed
                            logger.info(f"Truncated 'rag_in_query': Removed RAG context, using plain query. Freed approx {freed_by_removing_rag} tokens.")
                        else:
                            logger.debug("Tried to truncate RAG, but plain query is not smaller. No change.")
                    else:
                        logger.warning("Cannot truncate 'rag_in_query' further; even plain query exceeds budget with other components.")
                        # If even plain query doesn't fit, it's a ContextLengthError handled below.

            # Re-assemble final_payload_messages after truncation of components
            final_payload_messages = []
            current_tokens_in_payload = 0
            for category_key in self._inclusion_priority_order:
                for msg in components[category_key]:
                    msg_tokens = msg.tokens or 0
                    if current_tokens_in_payload + msg_tokens <= available_tokens_for_prompt:
                        final_payload_messages.append(msg)
                        current_tokens_in_payload += msg_tokens
                    else: # Should not happen if truncation worked, but as safeguard
                        logger.warning(f"Message '{msg.id}' from '{category_key}' still over budget after truncation pass. Final check will handle.")
                        break

        # Final check
        if current_tokens_in_payload > available_tokens_for_prompt:
            # This implies the essential parts (like final_user_query or system messages) are too large.
            logger.error(f"Context still too long after all truncation attempts. "
                         f"Final tokens: {current_tokens_in_payload}, Budget: {available_tokens_for_prompt}")
            raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens_in_payload,
                                     message="Context exceeds token limit after all truncation attempts. Essential components might be too large.")

        # Ensure minimum history messages are present if possible (final pass, not strictly budget-bound if space was made)
        # This is a soft constraint, should not cause error if budget is too tight.
        actual_history_chat_messages_in_payload = [m for m in final_payload_messages if m in components["history_chat"]]
        if len(actual_history_chat_messages_in_payload) < self._minimum_history_messages:
            logger.debug(f"Final payload has {len(actual_history_chat_messages_in_payload)} history_chat messages, "
                         f"less than minimum_history_messages ({self._minimum_history_messages}). Budget was likely too constrained.")


        final_token_count_for_payload = await provider.count_message_tokens(final_payload_messages, target_model)
        logger.info(f"Final prepared context: {len(final_payload_messages)} messages, {final_token_count_for_payload} tokens for model '{target_model}'.")

        return final_payload_messages, rag_documents_used_this_turn, final_token_count_for_payload

    def _truncate_message_list_from_start(self, messages: List[Message], tokens_to_free: int) -> Tuple[List[Message], int]:
        """Removes messages from the START of the list until tokens_to_free is met or list is empty."""
        freed_so_far = 0
        truncated_list = list(messages)

        while freed_so_far < tokens_to_free and truncated_list:
            msg_to_remove = truncated_list.pop(0) # Remove from start (oldest)
            freed_so_far += msg_to_remove.tokens or 0

        return truncated_list, freed_so_far

    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string. (Existing logic)"""
        if not documents: return ""
        sorted_documents = sorted(documents, key=lambda d: d.score if d.score is not None else float('inf'))
        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(sorted_documents):
            source_info_parts = []
            if doc.metadata and doc.metadata.get("source_file_path_relative"):
                source_info_parts.append(f"File: {doc.metadata.get('source_file_path_relative')}")
            elif doc.metadata and doc.metadata.get("source"):
                source_info_parts.append(f"Source: {doc.metadata.get('source')}")
            else: source_info_parts.append(f"DocID: {doc.id[:12]}")
            if doc.metadata and doc.metadata.get("start_line"):
                 source_info_parts.append(f"Line: {doc.metadata.get('start_line')}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            header = f"Context Document {i+1}: [{', '.join(source_info_parts)}] {score_info}"
            content_snippet = ' '.join(doc.content.splitlines()).strip()
            context_parts.append(f"\n{header}\n{content_snippet}")
        context_parts.append("\n--- End Retrieved Documents ---")
        return "\n".join(context_parts)
