# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, and potentially RAG integration
and MCP formatting in the future.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

# Use Type checking block for conditional imports if needed
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from ..providers.base import BaseProvider
#     from ..models import ChatSession, Message, Role
#     from confy.loader import Config as ConfyConfig

from ..providers.base import BaseProvider # Direct import should be fine
from ..models import ChatSession, Message, Role
from ..exceptions import ContextError, ContextLengthError, ConfigError

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore


logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates potential RAG results (future),
    and ensures the final payload adheres to the token limits of the target model.
    """

    def __init__(self, config: ConfyConfig):
        """
        Initializes the ContextManager.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
                    Reads settings from the `[context_management]` section.
        """
        self._config = config
        # Load relevant settings from config with defaults
        cm_config = self._config.get('context_management', {})
        self._reserved_response_tokens: int = cm_config.get('reserved_response_tokens', 500)
        self._history_selection_strategy: str = cm_config.get('history_selection_strategy', 'last_n_tokens')
        self._truncation_priority: str = cm_config.get('truncation_priority', 'history') # Placeholder for future use
        self._minimum_history_messages: int = cm_config.get('minimum_history_messages', 1)

        logger.info("ContextManager initialized.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"min_history={self._minimum_history_messages}")

        if self._history_selection_strategy not in ['last_n_tokens']: # Add more later e.g. 'last_n_messages'
             logger.warning(f"Unsupported history_selection_strategy '{self._history_selection_strategy}'. "
                            "Falling back to 'last_n_tokens'.")
             self._history_selection_strategy = 'last_n_tokens'


    async def prepare_context(
        self,
        session: ChatSession,
        provider: BaseProvider,
        model_name: Optional[str] = None,
        # RAG parameters (placeholders for Phase 2)
        rag_results: Optional[List[str]] = None,
        # MCP parameters (placeholders for Phase 3)
        use_mcp: bool = False
    ) -> List[Message]:
        """
        Prepares the list of messages to be sent to the LLM provider.

        Selects messages from the session history based on the configured strategy
        and token limits, potentially incorporating RAG results in the future.

        Args:
            session: The current ChatSession containing the message history.
            provider: The target BaseProvider instance for token counting and limits.
            model_name: The specific model name being used (needed for accurate limits/counting).
            rag_results: Placeholder for retrieved document contents (Phase 2).
            use_mcp: Placeholder for MCP flag (Phase 3).

        Returns:
            A list of Message objects representing the final context payload.

        Raises:
            ContextLengthError: If the context cannot be reduced below the model's limit
                                while preserving essential messages (system + latest user).
            ProviderError: If the provider fails during token counting or limit retrieval.
            ConfigError: If essential configuration is missing.
        """
        target_model = model_name or provider.default_model # Use provider's default if not specified
        if not target_model:
             raise ConfigError(f"Could not determine target model for context preparation (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' using provider '{provider.get_name()}'.")

        try:
            max_context_tokens = provider.get_max_context_length(target_model)
        except Exception as e:
            logger.error(f"Failed to get max context length for model '{target_model}' from provider '{provider.get_name()}': {e}")
            raise ProviderError(provider.get_name(), f"Could not get max context length: {e}")

        # Calculate available tokens for the prompt (context + history + potential RAG)
        available_tokens = max_context_tokens - self._reserved_response_tokens
        if available_tokens <= 0:
             logger.error(f"Reserved response tokens ({self._reserved_response_tokens}) exceed or equal "
                          f"max context length ({max_context_tokens}) for model '{target_model}'.")
             raise ContextError("Configuration error: reserved_response_tokens too high for model context limit.")

        logger.debug(f"Max context: {max_context_tokens}, Reserved: {self._reserved_response_tokens}, Available for prompt: {available_tokens}")

        # --- RAG Integration Placeholder (Phase 2) ---
        rag_tokens = 0
        rag_context_messages: List[Message] = []
        if rag_results:
            logger.warning("RAG integration is not yet fully implemented in prepare_context.")
            # Placeholder logic: Assume RAG results are prepended as a system message
            # rag_content = "\n\n".join(rag_results)
            # rag_msg = Message(role=Role.SYSTEM, content=f"Retrieved Context:\n{rag_content}", session_id=session.id)
            # rag_tokens = provider.count_tokens(rag_msg.content, target_model) # Count RAG content tokens
            # rag_context_messages.append(rag_msg)
            pass
        # --- End RAG Placeholder ---

        # Select messages based on strategy
        if self._history_selection_strategy == 'last_n_tokens':
            selected_history, current_tokens = self._select_history_last_n_tokens(
                session.messages, provider, target_model, available_tokens - rag_tokens
            )
        # Elif self._history_selection_strategy == 'last_n_messages':
        #     selected_history, current_tokens = self._select_history_last_n_messages(...)
        else:
            # Should not happen due to fallback in __init__, but defensively handle
            logger.warning(f"Unknown history strategy '{self._history_selection_strategy}', using 'last_n_tokens'.")
            selected_history, current_tokens = self._select_history_last_n_tokens(
                session.messages, provider, target_model, available_tokens - rag_tokens
            )

        # Combine RAG context (if any) and selected history
        final_context: List[Message] = rag_context_messages + selected_history
        final_token_count = current_tokens + rag_tokens

        logger.debug(f"Prepared context: {len(final_context)} messages, {final_token_count} tokens (limit: {available_tokens}).")

        # Final check: Ensure the essential messages (system + last user) didn't get truncated away
        # This is more relevant if truncation becomes very aggressive.
        has_system = any(m.role == Role.SYSTEM for m in final_context)
        has_user = any(m.role == Role.USER for m in final_context)
        last_original_message = session.messages[-1] if session.messages else None

        # Check if the very last user message is included (important for response generation)
        last_message_included = (
            last_original_message and
            last_original_message.role == Role.USER and
            any(m.id == last_original_message.id for m in final_context)
        )

        if not last_message_included and last_original_message and last_original_message.role == Role.USER:
             # This scenario indicates a potential problem where the context is so tight
             # that even the last user message couldn't fit after system messages.
             logger.error(f"Context length ({final_token_count} tokens) is too short for model '{target_model}' "
                          f"(limit {available_tokens}). Cannot fit essential messages (system + last user).")
             raise ContextLengthError(
                 model_name=target_model,
                 limit=available_tokens,
                 actual=final_token_count, # Or calculate tokens for system + last user
                 message="Context length too short to include essential messages."
             )

        # --- MCP Formatting Placeholder (Phase 3) ---
        if use_mcp:
            logger.warning("MCP formatting is not yet implemented in prepare_context.")
            # context_payload = self._format_mcp_context(final_context, rag_results)
            # return context_payload # Return MCP object
            pass
        # --- End MCP Placeholder ---

        return final_context # Return list of Message objects


    def _select_history_last_n_tokens(
        self,
        all_messages: List[Message],
        provider: BaseProvider,
        model_name: str,
        token_budget: int
    ) -> Tuple[List[Message], int]:
        """Selects messages based on token count, prioritizing recent ones and system messages."""

        selected_messages: List[Message] = []
        current_tokens = 0

        # Always include system messages first, calculate their token cost
        system_messages = [msg for msg in all_messages if msg.role == Role.SYSTEM]
        try:
            system_tokens = provider.count_message_tokens(system_messages, model_name) if system_messages else 0
        except Exception as e:
            logger.error(f"Failed to count tokens for system messages: {e}")
            raise ProviderError(provider.get_name(), f"Token counting failed for system messages: {e}")

        if system_tokens > token_budget:
             logger.error(f"System messages alone ({system_tokens} tokens) exceed token budget ({token_budget}) "
                          f"for model '{model_name}'.")
             raise ContextLengthError(
                 model_name=model_name,
                 limit=token_budget,
                 actual=system_tokens,
                 message="System messages exceed available token budget."
             )

        selected_messages.extend(system_messages)
        current_tokens += system_tokens
        logger.debug(f"Included {len(system_messages)} system messages ({system_tokens} tokens). "
                     f"Remaining budget: {token_budget - current_tokens} tokens.")

        # Add non-system messages from newest to oldest until budget is filled
        non_system_messages = [msg for msg in all_messages if msg.role != Role.SYSTEM]
        # Iterate in reverse (newest first)
        for msg in reversed(non_system_messages):
            try:
                # Count tokens for this single message (including overhead)
                # Note: count_message_tokens might be slightly inaccurate for single messages
                # depending on implementation, but should be close enough.
                # A more precise way might involve counting pairs, but adds complexity.
                # Let's use count_message_tokens on a list containing just this message.
                message_tokens = provider.count_message_tokens([msg], model_name)
                # Adjust for base overhead if count_message_tokens includes it
                # This depends heavily on the provider's count_message_tokens implementation.
                # Assuming count_message_tokens([msg]) gives a reasonable estimate for now.

            except Exception as e:
                logger.error(f"Failed to count tokens for message ID {msg.id}: {e}")
                raise ProviderError(provider.get_name(), f"Token counting failed for message {msg.id}: {e}")

            if current_tokens + message_tokens <= token_budget:
                selected_messages.append(msg) # Add to the list (will sort later)
                current_tokens += message_tokens
                logger.debug(f"Included message {msg.id} ({msg.role.value}, {message_tokens} tokens). "
                             f"Total tokens: {current_tokens}/{token_budget}")
            else:
                logger.debug(f"Skipping message {msg.id} ({msg.role.value}, {message_tokens} tokens). "
                             f"Exceeds budget ({current_tokens}/{token_budget}).")
                # If we skip the last user message, that's a problem handled in prepare_context
                break # Stop adding messages once budget is exceeded

        # Sort the final list by timestamp to maintain chronological order
        selected_messages.sort(key=lambda m: m.timestamp)

        # --- Basic Truncation (if still over budget after selection) ---
        # This loop removes the oldest *non-system* message until budget is met
        # or minimum history count is reached.
        final_token_count = provider.count_message_tokens(selected_messages, model_name)
        non_system_count = sum(1 for m in selected_messages if m.role != Role.SYSTEM)

        while final_token_count > token_budget and non_system_count > self._minimum_history_messages:
             logger.warning(f"Context ({final_token_count} tokens) still exceeds budget ({token_budget}) after initial selection. "
                            f"Attempting truncation (min history: {self._minimum_history_messages}).")
             # Find the oldest non-system message to remove
             oldest_non_system_index = -1
             for i, msg in enumerate(selected_messages):
                 if msg.role != Role.SYSTEM:
                     oldest_non_system_index = i
                     break

             if oldest_non_system_index != -1:
                 removed_msg = selected_messages.pop(oldest_non_system_index)
                 logger.debug(f"Truncated oldest non-system message: {removed_msg.id} ({removed_msg.role.value})")
                 final_token_count = provider.count_message_tokens(selected_messages, model_name)
                 non_system_count -= 1
             else:
                 # Should not happen if non_system_count > 0, but break defensively
                 logger.warning("Truncation loop encountered unexpected state: no non-system message found.")
                 break

        # Final check after truncation
        if final_token_count > token_budget:
             logger.error(f"Context could not be truncated sufficiently ({final_token_count} tokens) "
                          f"for model '{model_name}' (budget: {token_budget}). Minimum history kept: {non_system_count}.")
             raise ContextLengthError(
                 model_name=model_name,
                 limit=token_budget,
                 actual=final_token_count,
                 message="Context exceeds token limit even after truncation."
             )

        logger.debug(f"Final selected history: {len(selected_messages)} messages, {final_token_count} tokens.")
        return selected_messages, final_token_count
