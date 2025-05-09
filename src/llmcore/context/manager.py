# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, and RAG integration.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]


from ..providers.manager import ProviderManager
from ..providers.base import BaseProvider # ContextPayload will be updated to List[Message]
from ..storage.manager import StorageManager
from ..embedding.manager import EmbeddingManager
from ..models import ChatSession, Message, Role as LLMCoreRole, ContextDocument
from ..exceptions import (
    ContextError, ContextLengthError, ConfigError, ProviderError,
    EmbeddingError, VectorStorageError # MCPError removed
)


logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results, and ensures
    the final payload adheres to the token limits of the target model,
    using configurable strategies.
    Uses ProviderManager, StorageManager, and EmbeddingManager.
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

        # Load relevant settings from config with defaults
        cm_config = self._config.get('context_management', {})
        self._reserved_response_tokens: int = cm_config.get('reserved_response_tokens', 500)
        self._history_selection_strategy: str = cm_config.get('history_selection_strategy', 'last_n_tokens')
        self._truncation_priority: str = cm_config.get('truncation_priority', 'history')
        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)
        self._default_rag_k: int = cm_config.get('rag_retrieval_k', 3)
        self._rag_combination_strategy: str = cm_config.get('rag_combination_strategy', 'prepend_system')

        logger.info("ContextManager initialized with Provider, Storage, and Embedding Managers.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"truncation_priority={self._truncation_priority}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     f"rag_combo_strategy={self._rag_combination_strategy}")

        # Validate strategies and log warnings if unsupported/fallback needed
        if self._history_selection_strategy not in ['last_n_tokens', 'last_n_messages']:
             logger.warning(f"Unsupported history_selection_strategy '{self._history_selection_strategy}'. Falling back to 'last_n_tokens'.")
             self._history_selection_strategy = 'last_n_tokens'
        elif self._history_selection_strategy == 'last_n_messages':
             logger.warning("'last_n_messages' history strategy is less precise for token limits and may not be fully implemented. Consider 'last_n_tokens'.")

        if self._rag_combination_strategy not in ['prepend_system', 'prepend_user']:
             logger.warning(f"Unsupported rag_combination_strategy '{self._rag_combination_strategy}'. Falling back to 'prepend_system'.")
             self._rag_combination_strategy = 'prepend_system'

        if self._truncation_priority not in ['history', 'rag']:
             logger.warning(f"Unsupported truncation_priority '{self._truncation_priority}'. Falling back to 'history'.")
             self._truncation_priority = 'history'


    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        # RAG parameters
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None
    ) -> List[Message]: # Return type updated to List[Message]
        """
        Prepares the context payload (List[Message]) to be sent to the LLM provider.

        Handles history selection, RAG retrieval and injection, token counting,
        and context truncation based on configuration and parameters.

        Args:
            session: The current ChatSession containing the message history.
            provider_name: The name of the target provider.
            model_name: The specific model name being used.
            rag_enabled: Whether to perform RAG.
            rag_k: Number of documents to retrieve for RAG (overrides default).
            rag_collection: Vector store collection name for RAG (overrides default).

        Returns:
            The prepared context payload as a list of `llmcore.models.Message` objects.

        Raises:
            ContextLengthError: If context cannot be reduced below the model's limit.
            ProviderError: If provider interaction fails (token counting, limits).
            ConfigError: If configuration is invalid or provider/model not found.
            EmbeddingError: If RAG query embedding fails.
            VectorStorageError: If RAG search fails.
        """
        # --- Context Preparation Logic ---
        try:
            provider = self._provider_manager.get_provider(provider_name)
        except (ConfigError, ProviderError) as e:
             logger.error(f"ContextManager failed to get provider '{provider_name}': {e}")
             raise

        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Could not determine target model for context preparation (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' (Provider: {provider.get_name()}, RAG: {rag_enabled}).")

        try:
            max_context_tokens = provider.get_max_context_length(target_model)
        except Exception as e:
            logger.error(f"Failed to get max context length for model '{target_model}': {e}")
            raise ProviderError(provider.get_name(), f"Could not get max context length: {e}")

        available_tokens = max_context_tokens - self._reserved_response_tokens
        if available_tokens <= 0:
             raise ContextError(f"Configuration error: reserved_response_tokens ({self._reserved_response_tokens}) "
                                f"exceeds model context limit ({max_context_tokens}).")

        logger.debug(f"Max context: {max_context_tokens}, Reserved: {self._reserved_response_tokens}, Available for prompt: {available_tokens}")

        # --- RAG Retrieval ---
        retrieved_docs: List[ContextDocument] = []
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
            if last_user_message:
                query_text = last_user_message.content
                k = rag_k if rag_k is not None else self._default_rag_k
                logger.info(f"Performing RAG search (k={k}, Collection: {rag_collection or 'default'}) for query: '{query_text[:50]}...'")
                try:
                    query_embedding = await self._embedding_manager.generate_embedding(query_text)
                    vector_storage = self._storage_manager.get_vector_storage()
                    retrieved_docs = await vector_storage.similarity_search(
                        query_embedding=query_embedding, k=k, collection_name=rag_collection
                    )
                    logger.info(f"RAG search returned {len(retrieved_docs)} documents.")
                except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e:
                     logger.error(f"RAG retrieval failed: {e}. Proceeding without RAG context.")
                     retrieved_docs = []
                except Exception as e:
                     logger.error(f"Unexpected error during RAG retrieval: {e}", exc_info=True)
                     retrieved_docs = []
            else:
                logger.warning("RAG enabled, but no user message found in session to use as query.")
        # --- End RAG Retrieval ---

        temp_rag_context_str = self._format_rag_context(retrieved_docs) if retrieved_docs else ""
        rag_tokens = await provider.count_tokens(temp_rag_context_str, target_model) if temp_rag_context_str else 0

        history_token_budget = available_tokens - rag_tokens
        if history_token_budget < 0:
             logger.warning(f"RAG context alone ({rag_tokens} tokens) exceeds available budget ({available_tokens}).")
             history_token_budget = 0

        selected_history: List[Message] = []
        if history_token_budget > 0:
            if self._history_selection_strategy == 'last_n_tokens':
                selected_history, _ = await self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
            elif self._history_selection_strategy == 'last_n_messages':
                logger.warning("Using fallback 'last_n_tokens' strategy for 'last_n_messages' history selection.")
                selected_history, _ = await self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
            else: # Fallback
                selected_history, _ = await self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
        else:
            system_messages = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
            system_tokens = await provider.count_message_tokens(system_messages, target_model) if system_messages else 0
            if system_tokens <= history_token_budget:
                 selected_history = system_messages
            else:
                 selected_history = []
                 logger.warning("Not enough token budget even for system messages after RAG context.")


        # --- Combine and Truncate ---
        final_messages, final_token_count = await self._combine_and_truncate(
            provider=provider,
            model_name=target_model,
            token_budget=available_tokens,
            system_messages=[msg for msg in selected_history if msg.role == LLMCoreRole.SYSTEM],
            history_messages=[msg for msg in selected_history if msg.role != LLMCoreRole.SYSTEM],
            rag_context_str=temp_rag_context_str,
            rag_docs=retrieved_docs
        )

        logger.info(f"Final context: {len(final_messages)} messages, {final_token_count} tokens.")

        # --- Final Check ---
        last_original_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        if last_original_user_message:
            last_message_included = any(m.id == last_original_user_message.id for m in final_messages if m.role == LLMCoreRole.USER)
            if not last_message_included:
                 logger.error(f"Context length ({final_token_count} tokens) is too short for model '{target_model}' "
                              f"(limit {available_tokens}). Cannot fit essential messages.")
                 raise ContextLengthError(
                     model_name=target_model, limit=available_tokens, actual=final_token_count,
                     message="Context length too short to include essential messages."
                 )

        # Return the standard list of messages, removing the temporary RAG message marker
        final_messages_no_rag_marker = [msg for msg in final_messages if msg.id != "rag_context"]
        return final_messages_no_rag_marker

    # --- Helper methods ---
    def _format_rag_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved documents into a string for context injection or truncation checks."""
        if not documents: return ""
        context_parts = ["--- Retrieved Context ---"]
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i+1}")
            content_snippet = doc.content.replace('\n', ' ').strip()
            context_parts.append(f"\n[Source: {source}]\n{content_snippet}")
        return "\n".join(context_parts) + "\n--- End Context ---"

    async def _select_history_last_n_tokens(
        self,
        all_messages: List[Message],
        provider: BaseProvider,
        model_name: str,
        token_budget: int
    ) -> Tuple[List[Message], int]:
        """Selects history messages based on token count, prioritizing recent ones and system messages."""
        selected_messages: List[Message] = []
        current_tokens = 0
        system_messages = [msg for msg in all_messages if msg.role == LLMCoreRole.SYSTEM]
        try:
            system_tokens = await provider.count_message_tokens(system_messages, model_name) if system_messages else 0
        except Exception as e:
            raise ProviderError(provider.get_name(), f"Token counting failed for system messages: {e}")

        if system_tokens > token_budget:
             logger.warning(f"System messages ({system_tokens} tokens) alone exceed history budget ({token_budget}). Cannot include history.")
             return system_messages, system_tokens

        selected_messages.extend(system_messages)
        current_tokens += system_tokens

        non_system_messages = [msg for msg in all_messages if msg.role != LLMCoreRole.SYSTEM]
        for msg in reversed(non_system_messages):
            try:
                message_tokens = await provider.count_message_tokens([msg], model_name)
            except Exception as e:
                raise ProviderError(provider.get_name(), f"Token counting failed for message {msg.id}: {e}")

            if current_tokens + message_tokens <= token_budget:
                selected_messages.append(msg)
                current_tokens += message_tokens
            else:
                logger.debug(f"History token budget ({token_budget}) reached. Stopping selection.")
                break

        selected_messages.sort(key=lambda m: m.timestamp) # Ensure chronological order
        logger.debug(f"Selected {len(selected_messages)} history messages ({current_tokens} tokens) for budget {token_budget}.")
        return selected_messages, current_tokens

    async def _combine_and_truncate(
        self,
        provider: BaseProvider,
        model_name: str,
        token_budget: int,
        system_messages: List[Message],
        history_messages: List[Message],
        rag_context_str: str,
        rag_docs: List[ContextDocument]
    ) -> Tuple[List[Message], int]:
        """
        Combines history and RAG context, then truncates if necessary.
        Returns the final List[Message] (including a temporary RAG message marker if RAG is used)
        and the final token count.
        """
        combined_context: List[Message] = []
        rag_message_marker: Optional[Message] = None

        if rag_context_str:
            rag_message_marker = Message(role=LLMCoreRole.SYSTEM, content=rag_context_str, session_id="rag_context", id="rag_context")
            try:
                 rag_message_marker.tokens = await provider.count_tokens(rag_context_str, model_name)
            except Exception as e:
                 logger.error(f"Failed to count tokens for RAG context message: {e}")
                 rag_message_marker.tokens = 0 # Default to 0 if counting fails

            logger.debug(f"Combining context using strategy: {self._rag_combination_strategy}")
            if self._rag_combination_strategy == "prepend_system":
                combined_context.extend(system_messages)
                combined_context.append(rag_message_marker)
                combined_context.extend(history_messages)
            elif self._rag_combination_strategy == "prepend_user":
                 combined_context.extend(system_messages)
                 last_user_idx = -1
                 for i in range(len(history_messages) - 1, -1, -1):
                     if history_messages[i].role == LLMCoreRole.USER:
                         last_user_idx = i
                         break
                 if last_user_idx != -1:
                      combined_context.extend(history_messages[:last_user_idx])
                      combined_context.append(rag_message_marker)
                      combined_context.extend(history_messages[last_user_idx:])
                 else: # No user message in history_messages, append RAG after system
                      combined_context.extend(history_messages)
                      combined_context.append(rag_message_marker)
            else: # Default/Fallback to prepend_system
                 combined_context.extend(system_messages)
                 combined_context.append(rag_message_marker)
                 combined_context.extend(history_messages)
        else:
            combined_context.extend(system_messages)
            combined_context.extend(history_messages)

        try:
            current_tokens = await provider.count_message_tokens(combined_context, model_name)
        except Exception as e:
             raise ProviderError(provider.get_name(), f"Token counting failed for initial combined context: {e}")

        logger.debug(f"Combined context before truncation: {len(combined_context)} messages, {current_tokens} tokens (Budget: {token_budget}).")

        # Truncation loop
        while current_tokens > token_budget:
            # Identify messages eligible for truncation (non-system, non-RAG marker)
            history_msg_indices_for_truncation = [
                i for i, msg in enumerate(combined_context)
                if msg.role != LLMCoreRole.SYSTEM and msg.id != "rag_context"
            ]
            can_truncate_history = len(history_msg_indices_for_truncation) > self._minimum_history_messages
            can_truncate_rag = rag_message_marker is not None and rag_docs

            logger.debug(f"Truncation needed: {current_tokens}/{token_budget} tokens. Priority: {self._truncation_priority}. Can truncate history: {can_truncate_history}. Can truncate RAG: {can_truncate_rag}.")

            truncated_something = False
            if self._truncation_priority == "history" and can_truncate_history:
                if history_msg_indices_for_truncation:
                    idx_to_remove = history_msg_indices_for_truncation[0] # Oldest non-system, non-RAG
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message: {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True
            elif self._truncation_priority == "rag" and can_truncate_rag:
                rag_docs.pop() # Remove least relevant RAG doc
                if rag_docs:
                    new_rag_context_str = self._format_rag_context(rag_docs)
                    rag_message_marker.content = new_rag_context_str # type: ignore[union-attr]
                    try:
                        rag_message_marker.tokens = await provider.count_tokens(new_rag_context_str, model_name) # type: ignore[union-attr]
                    except Exception as e: logger.error(f"Failed to count tokens for truncated RAG context: {e}"); rag_message_marker.tokens = 0 # type: ignore[union-attr]
                    logger.debug(f"Truncated least relevant RAG document. New RAG context tokens: {rag_message_marker.tokens}") # type: ignore[union-attr]
                else: # All RAG docs removed
                    try: combined_context.remove(rag_message_marker) # type: ignore[arg-type]
                    except ValueError: logger.error("Failed to find RAG message marker for removal during truncation.")
                    rag_message_marker = None
                    logger.debug("Removed RAG context entirely after truncating all docs.")
                truncated_something = True
            elif can_truncate_history: # Fallback to history if priority was RAG but RAG couldn't be truncated
                 if history_msg_indices_for_truncation:
                    idx_to_remove = history_msg_indices_for_truncation[0]
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message (fallback): {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True
            elif can_truncate_rag: # Fallback to RAG if priority was history but history couldn't be truncated
                 rag_docs.pop()
                 if rag_docs:
                    new_rag_context_str = self._format_rag_context(rag_docs)
                    rag_message_marker.content = new_rag_context_str # type: ignore[union-attr]
                    try:
                        rag_message_marker.tokens = await provider.count_tokens(new_rag_context_str, model_name) # type: ignore[union-attr]
                    except Exception as e: logger.error(f"Failed to count tokens for truncated RAG context (fallback): {e}"); rag_message_marker.tokens = 0 # type: ignore[union-attr]
                    logger.debug(f"Truncated least relevant RAG document (fallback). New RAG context tokens: {rag_message_marker.tokens}") # type: ignore[union-attr]
                 else:
                    try: combined_context.remove(rag_message_marker) # type: ignore[arg-type]
                    except ValueError: logger.error("Failed to find RAG message marker for removal during fallback truncation.")
                    rag_message_marker = None
                    logger.debug("Removed RAG context entirely (fallback).")
                 truncated_something = True
            else:
                logger.error(f"Cannot truncate context further. Current tokens {current_tokens} exceed budget {token_budget}.")
                break # Exit loop if no truncation possible

            if truncated_something:
                try:
                    current_tokens = await provider.count_message_tokens(combined_context, model_name)
                    logger.debug(f"Context after truncation step: {len(combined_context)} messages, {current_tokens} tokens.")
                except Exception as e:
                    raise ProviderError(provider.get_name(), f"Token counting failed after truncation step: {e}")
            else:
                 logger.error("Truncation loop failed to remove any message or RAG content, but budget still exceeded.")
                 break # Should not happen if logic is correct and can_truncate flags are accurate

        if current_tokens > token_budget:
             raise ContextLengthError(
                 model_name=model_name, limit=token_budget, actual=current_tokens,
                 message="Context exceeds token limit even after truncation attempts."
             )

        return combined_context, current_tokens
