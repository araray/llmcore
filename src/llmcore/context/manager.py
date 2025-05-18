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

        # New config options from user request
        self._user_retained_messages_count: int = cm_config.get('user_retained_messages_count', 5)
        self._prioritize_user_context_items: bool = cm_config.get('prioritize_user_context_items', True)
        # self._enable_rag_for_history_overflow: bool = cm_config.get('enable_rag_for_history_overflow', False) # Future
        # self._rag_history_summary_model: Optional[str] = cm_config.get('rag_history_summary_model') # Future

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

        if self._truncation_priority not in ['history', 'rag', 'user_items']: # Added 'user_items'
             logger.warning(f"Unsupported truncation_priority '{self._truncation_priority}'. Falling back to 'history'.")
             self._truncation_priority = 'history'


    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None, # IDs of enabled ContextItems
        # RAG parameters for standard RAG queries
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None
    ) -> List[Message]:
        """
        Prepares the context payload (List[Message]) to be sent to the LLM provider.

        Handles history selection (including user_retained_messages_count and backfilling),
        incorporates active user-added context items, performs standard RAG queries,
        counts tokens, and truncates context based on configured strategies.

        Args:
            session: The current ChatSession.
            provider_name: The name of the target provider.
            model_name: The specific model name being used.
            active_context_item_ids: List of IDs of ContextItems from session.context_items
                                     that are currently enabled by the user.
            rag_enabled: Whether to perform standard RAG based on the latest user message.
            rag_k: Number of documents to retrieve for standard RAG.
            rag_collection: Vector store collection name for standard RAG.

        Returns:
            The prepared context payload as a list of `llmcore.models.Message` objects.

        Raises:
            ContextLengthError, ProviderError, ConfigError, EmbeddingError, VectorStorageError.
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Could not determine target model for context (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' (Provider: {provider.get_name()}, RAG: {rag_enabled}). Session: {session.id}")

        max_context_tokens = provider.get_max_context_length(target_model)
        available_tokens_for_prompt = max_context_tokens - self._reserved_response_tokens
        if available_tokens_for_prompt <= 0:
             raise ContextError(f"Config error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_context_tokens}).")

        logger.debug(f"Max context: {max_context_tokens}, Reserved for response: {self._reserved_response_tokens}, Available for prompt: {available_tokens_for_prompt}")

        # 1. Gather all potential context components
        system_messages = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
        history_messages_all = [msg for msg in session.messages if msg.role != LLMCoreRole.SYSTEM]

        # User-added context items (text, files)
        active_user_items_content: List[Tuple[ContextItem, str]] = [] # Store item and its content string
        if active_context_item_ids and session.context_items:
            for item_id in active_context_item_ids:
                item = session.get_context_item(item_id)
                if item and (item.type == ContextItemType.USER_TEXT or item.type == ContextItemType.USER_FILE):
                    # For USER_FILE, content is already loaded by LLMCore.add_file_context_item
                    content_str = f"--- User-Provided Context: {item.metadata.get('name', item.id)} ---\n" \
                                  f"{item.content}\n" \
                                  f"--- End User-Provided Context: {item.metadata.get('name', item.id)} ---"
                    active_user_items_content.append((item, content_str))

        # Standard RAG results (based on current user query)
        rag_docs_for_query: List[ContextDocument] = []
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k_val = rag_k if rag_k is not None else self._default_rag_k
                logger.info(f"Performing standard RAG search (k={k_val}, Collection: {rag_collection or 'default'}) for query: '{query_text[:50]}...'")
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    rag_docs_for_query = await vector_storage.similarity_search(
                        query_embedding=query_embedding, k=k_val, collection_name=rag_collection
                    )
                    logger.info(f"Standard RAG search returned {len(rag_docs_for_query)} documents.")
                except (EmbeddingError, VectorStorageError, ConfigError) as e:
                     logger.error(f"Standard RAG retrieval failed: {e}. Proceeding without RAG query context.")
                     rag_docs_for_query = []
            else:
                logger.warning("Standard RAG enabled, but no user message found in session to use as query.")

        # Format RAG results into a single string for token counting and potential inclusion
        formatted_rag_query_context_str = self._format_rag_docs_for_context(rag_docs_for_query) if rag_docs_for_query else ""

        # 2. Tokenize components (asynchronously)
        system_tokens = await provider.count_message_tokens(system_messages, target_model) if system_messages else 0

        user_items_tokens_map: Dict[str, int] = {} # item.id -> tokens
        total_user_items_tokens = 0
        for item, content_str in active_user_items_content:
            tokens = await provider.count_tokens(content_str, target_model)
            item.tokens = tokens # Cache on the item model if needed by client
            user_items_tokens_map[item.id] = tokens
            total_user_items_tokens += tokens

        rag_query_context_tokens = await provider.count_tokens(formatted_rag_query_context_str, target_model) if formatted_rag_query_context_str else 0

        logger.debug(f"Initial token counts: System={system_tokens}, UserItems={total_user_items_tokens}, RAGQueryContext={rag_query_context_tokens}")

        # 3. Build the context, prioritizing components and truncating
        final_messages: List[Message] = []
        current_tokens = 0

        # Add system messages first (always included if they fit)
        if system_tokens <= available_tokens_for_prompt:
            final_messages.extend(system_messages)
            current_tokens += system_tokens
        else:
            logger.warning(f"System messages ({system_tokens} tokens) exceed available prompt budget ({available_tokens_for_prompt}). Omitting system messages.")
            # Or truncate system messages if that's a desired strategy

        # Add RAG query context (if enabled and fits)
        # This usually comes after system messages, before user items or history.
        rag_query_context_message: Optional[Message] = None
        if formatted_rag_query_context_str and (current_tokens + rag_query_context_tokens <= available_tokens_for_prompt):
            rag_query_context_message = Message(
                role=LLMCoreRole.SYSTEM, # Or USER, depending on how it should be presented
                content=formatted_rag_query_context_str,
                session_id=session.id, # Associate with the session
                id="rag_query_context_marker" # Special ID
            )
            final_messages.append(rag_query_context_message)
            current_tokens += rag_query_context_tokens
        elif formatted_rag_query_context_str:
            logger.warning(f"Standard RAG context ({rag_query_context_tokens} tokens) too large to fit with system messages. Omitting.")


        # Add user-added context items (if prioritized and fit)
        included_user_items: List[ContextItem] = []
        if self._prioritize_user_context_items:
            sorted_user_items = sorted(active_user_items_content, key=lambda x: x[0].timestamp, reverse=True) # Example: newest first
            for item, content_str in sorted_user_items:
                item_tokens = user_items_tokens_map.get(item.id, 0)
                if current_tokens + item_tokens <= available_tokens_for_prompt:
                    # Present user items as system messages or a specific format
                    final_messages.append(Message(role=LLMCoreRole.SYSTEM, content=content_str, session_id=session.id, id=f"user_item_{item.id}"))
                    current_tokens += item_tokens
                    included_user_items.append(item)
                else:
                    logger.debug(f"User context item '{item.id}' ({item_tokens} tokens) doesn't fit. Skipping.")
                    # Could implement partial inclusion or truncation for user items if needed
                    break
            logger.info(f"Included {len(included_user_items)} user-added context items ({sum(user_items_tokens_map[i.id] for i in included_user_items)} tokens).")


        # History selection (Stage 1: user_retained_messages_count, Stage 2: backfill)
        # This considers the remaining token budget.
        history_token_budget = available_tokens_for_prompt - current_tokens
        selected_history_messages: List[Message] = []

        if history_token_budget > 0 and history_messages_all:
            # Stage 1: Prioritize user_retained_messages_count (latest N turns, user + assistant)
            retained_messages: List[Message] = []
            retained_tokens = 0

            # Get latest N non-system messages
            latest_n_history = history_messages_all[-self._user_retained_messages_count*2:] # Approx 2 messages per turn

            temp_retained_msgs = []
            for msg in reversed(latest_n_history):
                msg_tokens = await provider.count_message_tokens([msg], target_model)
                if retained_tokens + msg_tokens <= history_token_budget:
                    temp_retained_msgs.append(msg)
                    retained_tokens += msg_tokens
                else:
                    break
                if len(temp_retained_msgs) >= self._user_retained_messages_count * 2 and self._user_retained_messages_count > 0 : # Stop if we have enough turns
                     # This logic needs refinement to ensure complete turns are kept if possible.
                     # For simplicity, just taking N messages for now.
                     pass

            retained_messages = sorted(temp_retained_msgs, key=lambda m: m.timestamp)
            selected_history_messages.extend(retained_messages)
            current_history_tokens = retained_tokens

            logger.debug(f"Retained {len(retained_messages)} messages ({retained_tokens} tokens) based on user_retained_messages_count.")

            # Stage 2: Backfill with older messages if space allows
            remaining_history_budget = history_token_budget - current_history_tokens
            if remaining_history_budget > 0:
                older_history_candidates = [
                    msg for msg in history_messages_all
                    if msg.id not in {m.id for m in retained_messages}
                ]

                backfilled_messages = []
                backfilled_tokens = 0
                for msg in reversed(older_history_candidates): # Iterate from newest of the older messages
                    msg_tokens = await provider.count_message_tokens([msg], target_model)
                    if backfilled_tokens + msg_tokens <= remaining_history_budget:
                        backfilled_messages.append(msg)
                        backfilled_tokens += msg_tokens
                    else:
                        break
                selected_history_messages.extend(sorted(backfilled_messages, key=lambda m: m.timestamp))
                current_history_tokens += backfilled_tokens
                logger.debug(f"Backfilled {len(backfilled_messages)} older messages ({backfilled_tokens} tokens).")

        # Add selected history messages to final_messages
        # Ensure they are sorted correctly with other items if priorities change
        # For now, history comes after system, RAG query context, and prioritized user items.
        final_messages.extend(selected_history_messages)
        current_tokens += await provider.count_message_tokens(selected_history_messages, target_model) # Recalculate history tokens accurately

        # If user items were not prioritized, try to add them now if space allows
        if not self._prioritize_user_context_items:
            # This logic would be similar to the prioritized inclusion, but with remaining budget
            # For simplicity in this pass, we'll assume if not prioritized, they are not included
            # unless explicitly handled by truncation logic below.
            pass


        # Final Truncation if still over budget (should be rare if budgeting above is correct)
        # This simple truncation just removes from the start of combined non-system messages.
        # A more sophisticated truncation would use self._truncation_priority ('history', 'rag', 'user_items')
        final_messages_for_counting = [m for m in final_messages if m.id != "rag_query_context_marker"] # Exclude special markers for final count
        current_tokens = await provider.count_message_tokens(final_messages_for_counting, target_model)

        while current_tokens > available_tokens_for_prompt:
            logger.warning(f"Context still over budget ({current_tokens}/{available_tokens_for_prompt}). Performing final truncation.")
            # Simplistic truncation: remove oldest non-system, non-RAG-marker, non-user-item message
            # This needs to be smarter based on self._truncation_priority
            removed_something = False
            for i, msg in enumerate(final_messages):
                if msg.role != LLMCoreRole.SYSTEM and not msg.id.startswith("rag_") and not msg.id.startswith("user_item_"):
                    final_messages.pop(i)
                    removed_something = True
                    break
            if not removed_something: # Cannot truncate further essential messages
                logger.error(f"Cannot truncate context further. Current tokens {current_tokens} still exceed budget {available_tokens_for_prompt}.")
                break
            current_tokens = await provider.count_message_tokens(
                 [m for m in final_messages if m.id != "rag_query_context_marker"], target_model
            )

        if current_tokens > available_tokens_for_prompt:
            raise ContextLengthError(
                model_name=target_model, limit=available_tokens_for_prompt, actual=current_tokens,
                message="Final context still exceeds token limit after all truncation attempts."
            )

        logger.info(f"Final prepared context: {len(final_messages)} messages, {current_tokens} tokens for model '{target_model}'.")

        # Remove any special marker messages before returning
        return [msg for msg in final_messages if not msg.id.startswith("rag_query_context_marker")]


    def _format_rag_docs_for_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved RAG documents into a single string for context injection."""
        if not documents: return ""
        # Sort documents by score if available (assuming lower is better for some stores like Chroma L2)
        # For cosine similarity, higher is better. This needs to be provider-specific or configurable.
        # For now, let's assume they are already sorted by relevance by the vector store.
        context_parts = ["--- Retrieved Relevant Documents ---"]
        for i, doc in enumerate(documents):
            source_info = doc.metadata.get("source", f"Document {doc.id[:8]}")
            score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
            content_snippet = doc.content.replace('\n', ' ').strip()
            context_parts.append(f"\n[Source: {source_info} {score_info}]\n{content_snippet}")
        context_parts.append("--- End Retrieved Documents ---")
        return "\n".join(context_parts)
