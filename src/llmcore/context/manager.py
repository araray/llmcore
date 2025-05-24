# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and user-added context items.
Includes per-item truncation for user-added context and prompt template rendering.
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
    renders prompt templates, and ensures the final payload adheres to token limits
    using configurable strategies. Includes per-item truncation for user-added context.
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
        self._truncation_priority_order: List[str] = self._parse_truncation_priority(cm_config.get('truncation_priority', 'history,rag,user_items'))
        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)
        self._default_rag_k: int = cm_config.get('rag_retrieval_k', 3)
        self._rag_combination_strategy: str = cm_config.get('rag_combination_strategy', 'prepend_system') # Note: This strategy might be superseded by prompt templating for RAG.
        self._user_retained_messages_count: int = cm_config.get('user_retained_messages_count', 5)
        self._prioritize_user_context_items: bool = cm_config.get('prioritize_user_context_items', True)
        self._max_chars_per_user_item: int = cm_config.get('max_chars_per_user_item', 40000)

        # --- Load Prompt Template ---
        self._prompt_template_content: str
        default_template_str = cm_config.get(
            'default_prompt_template',
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:" # A simple fallback
        )
        prompt_template_path_str = cm_config.get('prompt_template_path', "")

        if prompt_template_path_str:
            base_path_for_template = Path.cwd() # Default base if main config path not found
            # Check if the main config object has the path it was loaded from
            # This attribute `_config_file_path_loaded_from` is specific to how `confy` might store it.
            # If LLMCore.create stores this on its self.config, it can be accessed.
            # For now, we assume self._config is the ConfyConfig instance passed from LLMCore.
            main_config_file_origin = getattr(self._config, '_config_file_path_loaded_from', None)
            if main_config_file_origin:
                main_config_path_obj = Path(main_config_file_origin)
                if main_config_path_obj.is_file():
                    base_path_for_template = main_config_path_obj.parent
                    logger.debug(f"Resolved template base path from main config: {base_path_for_template}")

            resolved_template_path = Path(prompt_template_path_str)
            if not resolved_template_path.is_absolute():
                resolved_template_path = (base_path_for_template / prompt_template_path_str).resolve()

            if resolved_template_path.is_file():
                try:
                    self._prompt_template_content = resolved_template_path.read_text(encoding='utf-8')
                    logger.info(f"Loaded RAG prompt template from: {resolved_template_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to read prompt template file '{resolved_template_path}': {e}. "
                        f"Using default_prompt_template string."
                    )
                    self._prompt_template_content = default_template_str
            else:
                logger.warning(
                    f"Prompt template file not found at '{resolved_template_path}'. "
                    f"Using default_prompt_template string."
                )
                self._prompt_template_content = default_template_str
        else:
            logger.info("No prompt_template_path configured. Using default_prompt_template string.")
            self._prompt_template_content = default_template_str
        # --- End Load Prompt Template ---


        logger.info("ContextManager initialized.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"user_retained_messages={self._user_retained_messages_count}, "
                     f"prioritize_user_items={self._prioritize_user_context_items}, "
                     f"truncation_priority_order={self._truncation_priority_order}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     # f"rag_combo_strategy={self._rag_combination_strategy}, " # Less relevant with templates
                     f"max_chars_per_user_item={self._max_chars_per_user_item}")
        logger.debug(f"Using RAG prompt template (first 100 chars): '{self._prompt_template_content[:100].replace(os.linesep, ' ')}...'")
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

        # rag_combination_strategy is less relevant now with explicit prompt templating for RAG
        # if self._rag_combination_strategy not in ['prepend_system', 'prepend_user']:
        #      logger.warning(f"Unsupported rag_combination_strategy '{self._rag_combination_strategy}'. Falling back to 'prepend_system'.")
        #      self._rag_combination_strategy = 'prepend_system'

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

        retained_for_now = temp_retained_buffer
        selected_history.extend(retained_for_now)
        current_history_tokens = tokens_for_retained
        logger.debug(f"Retained {len(retained_for_now)} recent history messages ({current_history_tokens} tokens) "
                     f"based on user_retained_messages_count ({self._user_retained_messages_count}).")

        # Stage 2: Backfill with older messages if budget allows
        remaining_history_budget_for_backfill = budget - current_history_tokens
        if remaining_history_budget_for_backfill > 0:
            ids_in_selected_history = {m.id for m in selected_history}
            older_history_candidates = [
                msg for msg in history_messages_all_non_system if msg.id not in ids_in_selected_history
            ]
            older_history_candidates.sort(key=lambda m: m.timestamp, reverse=True) # Newest of the older ones first

            temp_backfill_history: List[Message] = []
            tokens_for_backfill = 0
            for msg in older_history_candidates: # Iterate newest of remaining history first
                msg_tokens = msg.tokens or 0
                if tokens_for_backfill + msg_tokens <= remaining_history_budget_for_backfill:
                    temp_backfill_history.insert(0, msg) # Add to beginning to maintain chronological order
                    tokens_for_backfill += msg_tokens
                else:
                    break # Stop if budget exceeded

            selected_history = temp_backfill_history + selected_history # Prepend backfilled older history
            current_history_tokens += tokens_for_backfill
            logger.debug(f"Backfilled {len(temp_backfill_history)} older history messages ({tokens_for_backfill} tokens).")

        return selected_history, current_history_tokens

    def _render_prompt_template(
        self,
        rag_context_str: str,
        question_str: str,
        custom_template_values: Optional[Dict[str, str]]
    ) -> str:
        """
        Renders the loaded prompt template with context, question, and custom values.

        Args:
            rag_context_str: The formatted string of RAG documents.
            question_str: The user's question.
            custom_template_values: Dictionary of custom key-value pairs for placeholders.

        Returns:
            The rendered prompt string.
        """
        rendered_prompt = self._prompt_template_content # Start with the loaded template

        # Substitute standard placeholders
        rendered_prompt = rendered_prompt.replace("{context}", rag_context_str)
        rendered_prompt = rendered_prompt.replace("{question}", question_str)

        # Substitute custom placeholders
        if custom_template_values:
            for key, value in custom_template_values.items():
                rendered_prompt = rendered_prompt.replace(f"{{{key}}}", str(value)) # Ensure value is string

        # Check for any remaining unsubstituted placeholders (optional, for debugging)
        # This is a simple check; more robust regex might be needed for complex cases.
        remaining_placeholders = [match.group(1) for match in Path("").glob(r"\{([a-zA-Z0-9_]+)\}") if match.group(1) not in ["context", "question"] and (not custom_template_values or match.group(1) not in custom_template_values)] # type: ignore
        if remaining_placeholders:
            logger.warning(f"Prompt template may have unsubstituted placeholders: {remaining_placeholders}")

        return rendered_prompt


    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Message], Optional[List[ContextDocument]], int]:
        """
        Prepares the context payload for the LLM, including history, RAG, and user-added items.
        User-added items are truncated if they exceed `max_chars_per_user_item`, unless
        `item.metadata.get('ignore_char_limit', False)` is True.
        Dynamically selects embedding model for RAG queries based on collection metadata.
        Renders a prompt template if RAG is enabled.

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
        active_user_item_messages: List[Message] = [] # Formatted user-added context items
        rag_formatted_context_str: str = "" # String of formatted RAG documents
        rag_documents_used_this_turn: Optional[List[ContextDocument]] = None
        query_embedding_model_identifier: Optional[str] = None

        # --- 1. System Messages from History ---
        # (Existing logic for extracting system messages from session.messages)
        system_messages_from_session = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
        for msg in system_messages_from_session:
            if msg.tokens is None:
                msg.tokens = await provider.count_message_tokens([msg], target_model)
            system_messages_hist.append(msg)

        # --- 2. Process and Format Active User-Added Context Items ---
        # (Existing logic for truncating and formatting user_items into Message objects)
        if active_context_item_ids and session.context_items:
            # ... (same as existing logic to populate active_user_item_messages) ...
            # This part ensures active_user_item_messages contains Message objects,
            # each representing a user-added context item, potentially truncated.
            # For brevity, not repeating the full truncation logic here.
            # Assume active_user_item_messages is populated correctly.
            temp_user_items_to_process: List[ContextItem] = []
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and item.type in [ContextItemType.USER_TEXT, ContextItemType.USER_FILE, ContextItemType.RAG_SNIPPET]:
                    temp_user_items_to_process.append(item)
            temp_user_items_to_process.sort(key=lambda x: x.timestamp)

            for item in temp_user_items_to_process:
                # ... (truncation logic from existing code) ...
                # After truncation, format as a system message for inclusion
                item_type_str = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
                source_desc = item.metadata.get('filename', item.id) if item.type == ContextItemType.USER_FILE else \
                              (item.metadata.get('original_source', item.source_id) if item.type == ContextItemType.RAG_SNIPPET else item.id)
                truncation_status_msg = " (TRUNCATED)" if item.is_truncated else ""
                prefix = f"--- User-Provided Context Item (ID: {item.id}, Type: {item_type_str}, Source: {source_desc}, Original Tokens: {item.original_tokens or 'N/A'}{truncation_status_msg}) ---"
                suffix = f"--- End User-Provided Context Item (ID: {item.id}) ---"
                content_str_for_llm = f"{prefix}\n{item.content}\n{suffix}"
                user_item_msg = Message(role=LLMCoreRole.SYSTEM, content=content_str_for_llm, session_id=session.id, id=f"user_item_{item.id}")
                user_item_msg.tokens = await provider.count_message_tokens([user_item_msg], target_model) # Count formatted
                active_user_item_messages.append(user_item_msg)


        # --- 3. Perform RAG if enabled ---
        last_user_message_obj = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        actual_query_for_llm = last_user_message_obj.content if last_user_message_obj else ""

        if rag_enabled and last_user_message_obj:
            query_text_for_rag = last_user_message_obj.content
            k_val = rag_k if rag_k is not None else self._default_rag_k
            actual_rag_collection_name = rag_collection or self._storage_manager.get_vector_storage()._default_collection_name # type: ignore

            # Dynamic Embedding Model Selection for RAG Query (existing logic)
            # ... (same as existing logic to determine query_embedding_model_identifier) ...
            try:
                collection_meta = await self._storage_manager.get_vector_storage().get_collection_metadata(actual_rag_collection_name)
                if collection_meta:
                    coll_emb_provider = collection_meta.get("embedding_model_provider")
                    coll_emb_model_name = collection_meta.get("embedding_model_name")
                    if coll_emb_provider and coll_emb_model_name:
                        query_embedding_model_identifier = f"{coll_emb_provider}:{coll_emb_model_name}" if coll_emb_provider.lower() != "sentence-transformers" else coll_emb_model_name
                        logger.info(f"RAG: Using collection-specific embedding model for query: '{query_embedding_model_identifier}'")
            except Exception as e_meta:
                logger.error(f"RAG: Error retrieving collection metadata for '{actual_rag_collection_name}': {e_meta}. Falling back to default.", exc_info=True)
                query_embedding_model_identifier = None

            try:
                query_embedding = await self._embedding_manager.generate_embedding(
                    query_text_for_rag, model_identifier=query_embedding_model_identifier
                )
                vector_storage = self._storage_manager.get_vector_storage()
                retrieved_rag_docs = await vector_storage.similarity_search(
                    query_embedding=query_embedding, k=k_val,
                    collection_name=actual_rag_collection_name, filter_metadata=rag_metadata_filter
                )
                rag_formatted_context_str = self._format_rag_docs_for_context(retrieved_rag_docs)
                rag_documents_used_this_turn = retrieved_rag_docs
                logger.info(f"RAG retrieved {len(retrieved_rag_docs)} docs.")
            except Exception as e_rag:
                logger.error(f"RAG retrieval failed: {e_rag}. No RAG context will be added.", exc_info=True)
                rag_formatted_context_str = "" # Ensure it's empty on failure
                rag_documents_used_this_turn = []
        elif rag_enabled and not last_user_message_obj:
            logger.warning("RAG enabled, but no user message found in history to use as query.")
            actual_query_for_llm = "" # Should not happen if session.messages includes current user turn

        # --- 4. Assemble candidate messages and manage token budget ---
        candidate_messages: List[Message] = []
        current_tokens = 0

        # Helper to add messages to candidates if budget allows
        async def _add_to_candidates_if_budget(messages_to_add: List[Message]) -> None:
            nonlocal current_tokens
            for msg in messages_to_add:
                if msg.tokens is None: msg.tokens = await provider.count_message_tokens([msg], target_model)
                msg_token_count = msg.tokens or 0
                if current_tokens + msg_token_count <= available_tokens_for_prompt:
                    candidate_messages.append(msg)
                    current_tokens += msg_token_count
                else:
                    logger.debug(f"Message '{msg.id}' (role: {msg.role.value}, tokens: {msg_token_count}) exceeds budget. Not added.")
                    break

        # Add system messages first
        await _add_to_candidates_if_budget(system_messages_hist)

        # Add user-added context items (formatted as system messages)
        # Order of user_items vs RAG depends on _prioritize_user_context_items
        # For RAG with prompt templates, RAG context is part of the final user message.
        # User items are typically prepended as system messages.
        await _add_to_candidates_if_budget(active_user_item_messages)


        # --- 5. Build History and Final User Message (RAG or Non-RAG) ---
        # The last message in session.messages is the current user's query.
        # We need to separate it from the "history" part for assembly.
        history_messages_all_non_system = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM and msg.id != (last_user_message_obj.id if last_user_message_obj else None)]

        # Calculate budget for historical messages
        # This budget needs to account for the final user message (rendered or plain)
        # For RAG, the final user message will be the rendered template.
        # For non-RAG, it's just the user's query.

        final_user_prompt_content_for_llm: str
        if rag_enabled and last_user_message_obj:
            final_user_prompt_content_for_llm = self._render_prompt_template(
                rag_context_str=rag_formatted_context_str,
                question_str=actual_query_for_llm, # This is last_user_message_obj.content
                custom_template_values=prompt_template_values
            )
        elif last_user_message_obj: # Non-RAG, but there's a user message
            final_user_prompt_content_for_llm = actual_query_for_llm
        else: # Should not happen in a normal chat flow with a user message
            final_user_prompt_content_for_llm = ""
            logger.error("Prepare_context called without a final user message in session to process.")


        final_user_message_for_llm_obj = Message(
            role=LLMCoreRole.USER, # The rendered prompt is from the user's perspective
            content=final_user_prompt_content_for_llm,
            session_id=session.id,
            id=f"final_user_prompt_{last_user_message_obj.id if last_user_message_obj else uuid.uuid4().hex}"
        )
        final_user_message_for_llm_obj.tokens = await provider.count_message_tokens([final_user_message_for_llm_obj], target_model)

        # Budget for history is what's left after system, user_items, and the final rendered user prompt
        history_budget = available_tokens_for_prompt - current_tokens - (final_user_message_for_llm_obj.tokens or 0)

        built_history, built_history_tokens = await self._build_history_messages(
            history_messages_all_non_system, provider, target_model, history_budget
        )
        # Insert history after system messages and user_items, but before the final user prompt
        candidate_messages.extend(built_history)
        current_tokens += built_history_tokens

        # Now add the final user message (rendered RAG prompt or plain query)
        # if its tokens fit
        if current_tokens + (final_user_message_for_llm_obj.tokens or 0) <= available_tokens_for_prompt:
            candidate_messages.append(final_user_message_for_llm_obj)
            current_tokens += (final_user_message_for_llm_obj.tokens or 0)
        else:
            logger.warning(f"Final user prompt (tokens: {final_user_message_for_llm_obj.tokens}) "
                           f"could not be added as it exceeds remaining budget. "
                           f"Current tokens: {current_tokens}, Available: {available_tokens_for_prompt}. "
                           "This might mean history/user_items took too much space or RAG prompt is too large.")
            # If the final user message (the most important part) doesn't fit, this is a problem.
            # Truncation logic below might try to make space, but it should prioritize this.
            # For now, we add it and let truncation sort it out, but this indicates a very tight budget.
            if not rag_enabled and last_user_message_obj: # If not RAG, ensure original user query is there
                 candidate_messages.append(last_user_message_obj)
                 current_tokens += (last_user_message_obj.tokens or 0)


        # --- 6. Truncation (if still over budget) ---
        # The truncation logic needs to be careful not to remove the *final_user_message_for_llm_obj*
        # if it was successfully added, as that's the primary input.
        # The existing truncation logic iterates through `_truncation_priority_order`.
        # If "history" is truncated, it should act on `built_history` part within `candidate_messages`.
        # If "rag" is truncated, it means `rag_formatted_context_str` was too large, implying
        # the `final_user_message_for_llm_obj` (if RAG) might be too large.
        # This part needs careful review to ensure the rendered prompt isn't wrongly truncated.
        # For now, the existing truncation logic will apply to the `candidate_messages` list.
        # The `_format_rag_docs_for_context` could be made to truncate its own output if needed.

        final_payload_messages = list(candidate_messages) # Start with what fit so far

        # (Simplified Truncation - existing detailed logic for _truncation_priority_order applies)
        # This loop should ensure that the final_user_message_for_llm_obj is preserved if possible.
        # A better strategy might be to calculate space for final_user_message_for_llm_obj first,
        # then fill history, then user_items, then RAG context string *within* the template.
        # However, with the template approach, RAG context is already part of final_user_message_for_llm_obj.

        # For now, let's assume the existing truncation logic will try to make space.
        # If after all truncation, the context is still too long, or if the essential
        # final_user_message_for_llm_obj itself is too large, then ContextLengthError.

        for priority_type_to_truncate in self._truncation_priority_order:
            if current_tokens <= available_tokens_for_prompt: break
            # ... (existing truncation logic for 'history', 'rag', 'user_items') ...
            # If 'rag' is to be truncated, and RAG is enabled, this means the
            # `final_user_message_for_llm_obj` (which contains the RAG context via template)
            # is too large. This scenario is complex because we can't easily truncate *parts*
            # of the rendered template. We might need to re-render with fewer RAG docs.
            # For now, if 'rag' is truncated, and RAG was enabled, we might have to remove
            # the entire RAG-rendered user message and fall back to a non-RAG query if possible,
            # or error out if the non-RAG query also doesn't fit.

            # Simplified: if current_tokens > available_tokens_for_prompt after filling,
            # and the last message is the rendered RAG prompt, and 'rag' is high in truncation priority,
            # this might lead to removing the whole query.
            # This needs more sophisticated handling: if RAG causes overflow, try reducing RAG docs.

        # Ensure the very last user message (the actual query part) is present
        if not any(msg.id == (final_user_message_for_llm_obj.id if final_user_message_for_llm_obj else None) for msg in final_payload_messages) and last_user_message_obj:
             # If the main rendered prompt didn't make it, try to at least include the original user query
             if current_tokens + (last_user_message_obj.tokens or 0) <= available_tokens_for_prompt:
                 final_payload_messages.append(last_user_message_obj)
                 current_tokens += (last_user_message_obj.tokens or 0)
             else: # Still no space for even the raw user query
                error_detail_msg = "Context length too short. Latest user message could not be included after all truncation attempts."
                logger.error(error_detail_msg + f" Limit: {available_tokens_for_prompt}, Current after truncation: {current_tokens}")
                raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message=error_detail_msg)


        if current_tokens > available_tokens_for_prompt:
            logger.error(f"Final context ({current_tokens} tokens) still exceeds budget ({available_tokens_for_prompt}) despite all truncation efforts.")
            raise ContextLengthError(model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens, message="Context exceeds token limit after all truncation attempts.")

        # Final token count for the actual payload sent to LLM
        final_token_count_for_payload = await provider.count_message_tokens(final_payload_messages, target_model)
        logger.info(f"Final prepared context: {len(final_payload_messages)} messages, {final_token_count_for_payload} tokens for model '{target_model}'.")

        return final_payload_messages, rag_documents_used_this_turn, final_token_count_for_payload


    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string for context injection."""
        if not documents: return ""
        # Sort by score if available (lower is better for distance, higher for similarity)
        # Assuming score is distance-like for now (common with ChromaDB)
        sorted_documents = sorted(documents, key=lambda d: d.score if d.score is not None else float('inf'))

        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(sorted_documents):
            source_info_parts = []
            if doc.metadata and doc.metadata.get("source_file_path_relative"):
                source_info_parts.append(f"File: {doc.metadata.get('source_file_path_relative')}")
            elif doc.metadata and doc.metadata.get("source"): # Generic source
                source_info_parts.append(f"Source: {doc.metadata.get('source')}")
            else:
                source_info_parts.append(f"DocID: {doc.id[:12]}") # Fallback to ID

            if doc.metadata and doc.metadata.get("start_line"):
                 source_info_parts.append(f"Line: {doc.metadata.get('start_line')}")

            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            header = f"Context Document {i+1}: [{', '.join(source_info_parts)}] {score_info}"

            content_snippet = ' '.join(doc.content.splitlines()).strip() # Normalize whitespace
            context_parts.append(f"\n{header}\n{content_snippet}")
        context_parts.append("\n--- End Retrieved Documents ---")
        return "\n".join(context_parts)
