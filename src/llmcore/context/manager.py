# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and optional MCP formatting.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore

from ..providers.manager import ProviderManager
from ..providers.base import BaseProvider
from ..storage.manager import StorageManager
from ..embedding.manager import EmbeddingManager
from ..models import ChatSession, Message, Role, ContextDocument
from ..exceptions import (
    ContextError, ContextLengthError, ConfigError, ProviderError,
    EmbeddingError, VectorStorageError
)


logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results, and ensures
    the final payload adheres to the token limits of the target model,
    using configurable strategies. Uses ProviderManager, StorageManager,
    and EmbeddingManager.
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
             # Keep it set, but the selection function might fall back

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
        rag_collection: Optional[str] = None,
        # MCP parameters (placeholders for Phase 3)
        use_mcp: bool = False # TODO: Implement MCP logic in Phase 3 Task 3.4
    ) -> List[Message]:
        """
        Prepares the list of messages to be sent to the LLM provider.

        Handles history selection, RAG retrieval and injection, token counting,
        and context truncation based on configuration.

        Args:
            session: The current ChatSession containing the message history.
            provider_name: The name of the target provider.
            model_name: The specific model name being used.
            rag_enabled: Whether to perform RAG.
            rag_k: Number of documents to retrieve for RAG (overrides default).
            rag_collection: Vector store collection name for RAG (overrides default).
            use_mcp: Placeholder for MCP flag (Phase 3). If True, attempts MCP formatting.

        Returns:
            A list of Message objects representing the final context payload.

        Raises:
            ContextLengthError: If context cannot be reduced below the model's limit.
            ProviderError: If provider interaction fails (token counting, limits).
            ConfigError: If configuration is invalid or provider/model not found.
            EmbeddingError: If RAG query embedding fails.
            VectorStorageError: If RAG search fails.
        """
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
        rag_context_str = ""
        rag_tokens = 0
        if rag_enabled:
            last_user_message = next((msg for msg in reversed(session.messages) if msg.role == Role.USER), None)
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
                    if retrieved_docs:
                        rag_context_str = self._format_rag_context(retrieved_docs)
                        rag_tokens = provider.count_tokens(rag_context_str, target_model)
                        logger.debug(f"Formatted RAG context ({rag_tokens} tokens).")

                except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e:
                     logger.error(f"RAG retrieval failed: {e}. Proceeding without RAG context.")
                     retrieved_docs = []
                     rag_context_str = ""
                     rag_tokens = 0
                except Exception as e:
                     logger.error(f"Unexpected error during RAG retrieval: {e}", exc_info=True)
                     retrieved_docs = []
                     rag_context_str = ""
                     rag_tokens = 0
            else:
                logger.warning("RAG enabled, but no user message found in session to use as query.")
        # --- End RAG Retrieval ---

        history_token_budget = available_tokens - rag_tokens
        if history_token_budget < 0:
             logger.warning(f"RAG context alone ({rag_tokens} tokens) exceeds available budget ({available_tokens}). "
                            "Attempting to proceed with only RAG context (may fail).")
             history_token_budget = 0

        # --- History Selection ---
        selected_history: List[Message] = []
        history_tokens = 0
        if history_token_budget > 0:
            if self._history_selection_strategy == 'last_n_tokens':
                selected_history, history_tokens = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
            elif self._history_selection_strategy == 'last_n_messages':
                # Placeholder: Implement last_n_messages strategy or fall back
                logger.warning("Using fallback 'last_n_tokens' strategy for 'last_n_messages' history selection.")
                selected_history, history_tokens = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
                # selected_history, history_tokens = self._select_history_last_n_messages(
                #     session.messages, provider, target_model, history_token_budget, some_n_value
                # )
            else: # Fallback if somehow an invalid strategy was set
                selected_history, history_tokens = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
        else:
            # Keep only system messages if budget allows, otherwise empty history
            system_messages = [msg for msg in session.messages if msg.role == Role.SYSTEM]
            system_tokens = provider.count_message_tokens(system_messages, target_model) if system_messages else 0
            if system_tokens <= history_token_budget:
                 selected_history = system_messages
                 history_tokens = system_tokens
            else:
                 selected_history = []
                 history_tokens = 0
                 logger.warning("Not enough token budget even for system messages after RAG context.")

        # --- Combine and Truncate ---
        final_context, final_token_count = self._combine_and_truncate(
            provider=provider,
            model_name=target_model,
            token_budget=available_tokens,
            system_messages=[msg for msg in selected_history if msg.role == Role.SYSTEM],
            history_messages=[msg for msg in selected_history if msg.role != Role.SYSTEM],
            rag_context_str=rag_context_str,
            rag_docs=retrieved_docs
        )

        logger.info(f"Final context: {len(final_context)} messages, {final_token_count} tokens (Budget: {available_tokens}).")

        # Final check: Ensure the last user message is included if possible
        last_original_user_message = next((msg for msg in reversed(session.messages) if msg.role == Role.USER), None)
        if last_original_user_message:
            last_message_included = any(m.id == last_original_user_message.id for m in final_context if m.role == Role.USER)
            if not last_message_included:
                 logger.error(f"Context length ({final_token_count} tokens) is too short for model '{target_model}' "
                              f"(limit {available_tokens}). Cannot fit essential messages (system + last user).")
                 raise ContextLengthError(
                     model_name=target_model, limit=available_tokens, actual=final_token_count,
                     message="Context length too short to include essential messages."
                 )

        # --- MCP Formatting Placeholder (Phase 3 Task 3.4) ---
        if use_mcp:
            logger.warning("MCP formatting is not yet implemented in prepare_context.")
            # try:
            #     from modelcontextprotocol import Context as MCPContext, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
            #     # Map final_context and rag_docs to MCP objects
            #     # ... implementation ...
            #     mcp_context = MCPContext(...)
            #     logger.info("Formatted context using Model Context Protocol.")
            #     return mcp_context # Return MCP object instead of List[Message]
            # except ImportError:
            #     logger.error("MCP SDK 'modelcontextprotocol' not installed. Cannot format context.")
            #     raise ConfigError("MCP formatting enabled but SDK not installed.")
            # except Exception as mcp_e:
            #     logger.error(f"Failed to format context using MCP: {mcp_e}")
            #     raise MCPError(f"MCP formatting failed: {mcp_e}")
        # --- End MCP Placeholder ---

        return final_context # Return List[Message] if MCP not used or failed

    def _format_rag_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved documents into a string for context injection."""
        if not documents:
            return ""

        context_parts = ["--- Retrieved Context ---"]
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i+1}")
            content_snippet = doc.content.replace('\n', ' ').strip()
            # Optionally limit snippet length: content_snippet = content_snippet[:500]
            context_parts.append(f"\n[Source: {source}]\n{content_snippet}")
            # Consider adding score: f"(Score: {doc.score:.4f})" if available and meaningful

        return "\n".join(context_parts) + "\n--- End Context ---" # Add clear delimiters

    def _select_history_last_n_tokens(
        self,
        all_messages: List[Message],
        provider: BaseProvider,
        model_name: str,
        token_budget: int
    ) -> Tuple[List[Message], int]:
        """
        Selects history messages based on token count, prioritizing recent ones and system messages.
        This function focuses *only* on selecting history within its budget.
        """
        selected_messages: List[Message] = []
        current_tokens = 0

        system_messages = [msg for msg in all_messages if msg.role == Role.SYSTEM]
        try:
            system_tokens = provider.count_message_tokens(system_messages, model_name) if system_messages else 0
        except Exception as e:
            raise ProviderError(provider.get_name(), f"Token counting failed for system messages: {e}")

        if system_tokens > token_budget:
             logger.warning(f"System messages ({system_tokens} tokens) alone exceed history budget ({token_budget}). Only including system messages.")
             return system_messages, system_tokens # Let combine/truncate handle final fit

        selected_messages.extend(system_messages)
        current_tokens += system_tokens

        # Iterate through non-system messages in reverse (newest first)
        non_system_messages = [msg for msg in all_messages if msg.role != Role.SYSTEM]
        for msg in reversed(non_system_messages):
            try:
                # Count tokens for this single message
                message_tokens = provider.count_message_tokens([msg], model_name)
            except Exception as e:
                raise ProviderError(provider.get_name(), f"Token counting failed for message {msg.id}: {e}")

            if current_tokens + message_tokens <= token_budget:
                selected_messages.append(msg) # Add to the list (will sort later)
                current_tokens += message_tokens
            else:
                # Stop adding messages once budget is exceeded for this message
                logger.debug(f"History token budget ({token_budget}) reached. Stopping selection.")
                break

        # Sort the final list by timestamp to maintain chronological order
        selected_messages.sort(key=lambda m: m.timestamp)
        logger.debug(f"Selected {len(selected_messages)} history messages ({current_tokens} tokens) for budget {token_budget}.")
        return selected_messages, current_tokens

    # Placeholder for last_n_messages strategy (as per spec indicating it's not fully implemented/preferred)
    # def _select_history_last_n_messages(
    #     self,
    #     all_messages: List[Message],
    #     provider: BaseProvider,
    #     model_name: str,
    #     token_budget: int,
    #     n_messages: int # The 'n' value would need to come from config
    # ) -> Tuple[List[Message], int]:
    #     """Selects the last N messages, checking if they fit the token budget."""
    #     logger.warning("Using 'last_n_messages' strategy - token count may exceed budget.")
    #     system_messages = [msg for msg in all_messages if msg.role == Role.SYSTEM]
    #     non_system_messages = [msg for msg in all_messages if msg.role != Role.SYSTEM]
    #
    #     selected_non_system = non_system_messages[-n_messages:] # Get last N
    #     selected_messages = system_messages + selected_non_system
    #     selected_messages.sort(key=lambda m: m.timestamp)
    #
    #     try:
    #         current_tokens = provider.count_message_tokens(selected_messages, model_name)
    #     except Exception as e:
    #         raise ProviderError(provider.get_name(), f"Token counting failed for selected messages: {e}")
    #
    #     if current_tokens > token_budget:
    #          logger.error(f"'last_n_messages' ({n_messages}) resulted in {current_tokens} tokens, exceeding budget {token_budget}. "
    #                       "This strategy cannot guarantee fitting within limits.")
    #          # Option: Fallback to last_n_tokens? Or raise error? Let's raise for clarity.
    #          raise ContextLengthError(model_name, token_budget, current_tokens,
    #                                   message=f"'last_n_messages' ({n_messages}) exceeded token budget.")
    #
    #     logger.debug(f"Selected last {n_messages} messages ({current_tokens} tokens).")
    #     return selected_messages, current_tokens


    def _combine_and_truncate(
        self,
        provider: BaseProvider,
        model_name: str,
        token_budget: int,
        system_messages: List[Message],
        history_messages: List[Message], # Already selected history (excluding system)
        rag_context_str: str,
        rag_docs: List[ContextDocument]
    ) -> Tuple[List[Message], int]:
        """
        Combines history and RAG context based on strategy, then truncates if necessary.

        Args:
            provider: The LLM provider instance.
            model_name: The target model name.
            token_budget: The maximum allowed tokens for the combined context.
            system_messages: List of system messages selected.
            history_messages: List of user/assistant messages selected by history strategy.
            rag_context_str: The formatted string of retrieved RAG documents.
            rag_docs: The original list of retrieved ContextDocument objects (for RAG truncation).

        Returns:
            A tuple containing the final list of Message objects and their total token count.

        Raises:
            ContextLengthError: If the context cannot be truncated below the budget.
            ProviderError: If token counting fails during truncation.
        """
        combined_context: List[Message] = []
        rag_message: Optional[Message] = None

        # --- Context Combination ---
        if rag_context_str:
            # Create a temporary Message object for the RAG context
            rag_message = Message(role=Role.SYSTEM, content=rag_context_str, session_id="rag_context")
            try:
                 rag_message.tokens = provider.count_message_tokens([rag_message], model_name)
            except Exception as e:
                 logger.error(f"Failed to count tokens for RAG context message: {e}")
                 rag_message.tokens = 0 # Assume 0 if counting fails

            logger.debug(f"Combining context using strategy: {self._rag_combination_strategy}")
            if self._rag_combination_strategy == "prepend_system":
                # Add RAG context after original system messages, before history
                combined_context.extend(system_messages)
                combined_context.append(rag_message)
                combined_context.extend(history_messages)
            elif self._rag_combination_strategy == "prepend_user":
                 # Add RAG context just before the *last* user message in the selected history
                 combined_context.extend(system_messages)
                 # Find the index of the last user message
                 last_user_idx = -1
                 for i in range(len(history_messages) - 1, -1, -1):
                     if history_messages[i].role == Role.USER:
                         last_user_idx = i
                         break
                 if last_user_idx != -1:
                      # Insert RAG before the last user message
                      combined_context.extend(history_messages[:last_user_idx])
                      combined_context.append(rag_message)
                      combined_context.extend(history_messages[last_user_idx:])
                 else: # No user messages in history, append RAG after system/assistant
                      combined_context.extend(history_messages)
                      combined_context.append(rag_message)
            else: # Default/Fallback to prepend_system
                 combined_context.extend(system_messages)
                 combined_context.append(rag_message)
                 combined_context.extend(history_messages)
        else:
            # No RAG context, just use history
            combined_context.extend(system_messages)
            combined_context.extend(history_messages)

        # --- Truncation ---
        try:
            current_tokens = provider.count_message_tokens(combined_context, model_name)
        except Exception as e:
             raise ProviderError(provider.get_name(), f"Token counting failed for initial combined context: {e}")

        logger.debug(f"Combined context before truncation: {len(combined_context)} messages, {current_tokens} tokens (Budget: {token_budget}).")

        # Truncate based on priority
        while current_tokens > token_budget:
            # Determine if history/RAG can be truncated further
            # Count non-system, non-RAG messages available for truncation
            history_msg_indices = [i for i, msg in enumerate(combined_context) if msg.role != Role.SYSTEM and msg.id != "rag_context"]
            can_truncate_history = len(history_msg_indices) > self._minimum_history_messages

            # Check if RAG context exists and we have docs left to potentially shorten
            can_truncate_rag = rag_message is not None and rag_docs

            logger.warning(f"Truncation needed: {current_tokens}/{token_budget} tokens. "
                           f"Priority: {self._truncation_priority}. Can truncate history: {can_truncate_history}. Can truncate RAG: {can_truncate_rag}.")

            truncated_something = False
            if self._truncation_priority == "history" and can_truncate_history:
                # Remove oldest non-system, non-RAG message (first in the list of indices)
                if history_msg_indices:
                    idx_to_remove = history_msg_indices[0]
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message: {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True

            elif self._truncation_priority == "rag" and can_truncate_rag:
                # Remove the least relevant RAG document and regenerate RAG context string
                rag_docs.pop() # Remove last doc (assuming sorted by relevance initially)
                if rag_docs:
                    rag_context_str = self._format_rag_context(rag_docs)
                    rag_message.content = rag_context_str # Update content in the message object
                    try:
                        rag_message.tokens = provider.count_message_tokens([rag_message], model_name) # Recalculate tokens
                    except Exception as e:
                        logger.error(f"Failed to count tokens for truncated RAG context: {e}")
                        rag_message.tokens = 0 # Assume 0 on failure
                    logger.debug(f"Truncated least relevant RAG document. New RAG context tokens: {rag_message.tokens}")
                else:
                    # Remove the RAG message entirely if no docs left
                    try:
                        rag_message_index = combined_context.index(rag_message)
                        combined_context.pop(rag_message_index)
                    except ValueError: # Should not happen if rag_message exists
                         logger.error("Failed to find RAG message for removal during truncation.")
                    rag_message = None # Signal RAG context is gone
                    logger.debug("Removed RAG context entirely after truncating all docs.")
                truncated_something = True

            elif can_truncate_history: # Fallback to history if priority couldn't be met
                 if history_msg_indices:
                    idx_to_remove = history_msg_indices[0]
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message (fallback): {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True

            elif can_truncate_rag: # Fallback to RAG if history couldn't be truncated
                 rag_docs.pop()
                 if rag_docs:
                    rag_context_str = self._format_rag_context(rag_docs)
                    rag_message.content = rag_context_str
                    try:
                        rag_message.tokens = provider.count_message_tokens([rag_message], model_name)
                    except Exception as e:
                        logger.error(f"Failed to count tokens for truncated RAG context (fallback): {e}")
                        rag_message.tokens = 0
                    logger.debug(f"Truncated least relevant RAG document (fallback). New RAG context tokens: {rag_message.tokens}")
                 else:
                    try:
                        rag_message_index = combined_context.index(rag_message)
                        combined_context.pop(rag_message_index)
                    except ValueError:
                         logger.error("Failed to find RAG message for removal during fallback truncation.")
                    rag_message = None
                    logger.debug("Removed RAG context entirely (fallback).")
                 truncated_something = True
            else:
                # Cannot truncate further based on rules
                logger.error(f"Cannot truncate context further. Current tokens {current_tokens} exceed budget {token_budget}. "
                             f"Minimum history: {self._minimum_history_messages}, RAG docs left: {len(rag_docs) if rag_docs else 0}")
                break # Exit loop to raise ContextLengthError outside the loop

            # Recalculate total tokens ONLY if something was truncated
            if truncated_something:
                try:
                    current_tokens = provider.count_message_tokens(combined_context, model_name)
                    logger.debug(f"Context after truncation step: {len(combined_context)} messages, {current_tokens} tokens.")
                except Exception as e:
                    raise ProviderError(provider.get_name(), f"Token counting failed after truncation step: {e}")
            else:
                 # Safety break if no truncation happened in an iteration, means we are stuck
                 logger.error("Truncation loop failed to remove any message or RAG content, but budget still exceeded.")
                 break


        # Final check after truncation loop
        if current_tokens > token_budget:
             raise ContextLengthError(
                 model_name=model_name, limit=token_budget, actual=current_tokens,
                 message="Context exceeds token limit even after truncation attempts."
             )

        return combined_context, current_tokens
