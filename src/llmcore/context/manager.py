# src/llmcore/context/manager.py
"""
Context Management for LLMCore.

Handles the assembly of context payloads for LLM providers, managing
token limits, history selection, RAG integration, and optional MCP formatting.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore

# Use TYPE_CHECKING for MCP import to avoid runtime dependency
if TYPE_CHECKING:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any
        MCPMessage = Any
        MCPRole = Any
        RetrievedKnowledge = Any
        mcp_library_available = False
else:
    # At runtime, try to import, but don't fail if not present unless MCP is enabled later
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any
        MCPMessage = Any
        MCPRole = Any
        RetrievedKnowledge = Any
        mcp_library_available = False


from ..providers.manager import ProviderManager
from ..providers.base import BaseProvider, ContextPayload # ContextPayload is Union[List[Message], MCPContextObject]
from ..storage.manager import StorageManager
from ..embedding.manager import EmbeddingManager
from ..models import ChatSession, Message, Role as LLMCoreRole, ContextDocument
from ..exceptions import (
    ContextError, ContextLengthError, ConfigError, ProviderError,
    EmbeddingError, VectorStorageError, MCPError # Added MCPError
)


logger = logging.getLogger(__name__)

# Mapping from LLMCore Role Enum to MCP Role Enum (if MCP library is available)
LLMCORE_TO_MCP_ROLE_MAP: Dict[LLMCoreRole, Any] = {}
if mcp_library_available:
    LLMCORE_TO_MCP_ROLE_MAP = {
        LLMCoreRole.SYSTEM: MCPRole.SYSTEM,
        LLMCoreRole.USER: MCPRole.USER,
        LLMCoreRole.ASSISTANT: MCPRole.ASSISTANT,
    }


class ContextManager:
    """
    Manages the context window for LLM interactions.

    Selects messages from history, integrates RAG results, and ensures
    the final payload adheres to the token limits of the target model,
    using configurable strategies. Can optionally format the output using
    the Model Context Protocol (MCP) if enabled and the library is installed.
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

        # MCP specific config
        self._enable_mcp_globally: bool = self._config.get('llmcore.enable_mcp', False)
        self._mcp_version: Optional[str] = cm_config.get('mcp_version') # e.g., "v1"

        logger.info("ContextManager initialized with Provider, Storage, and Embedding Managers.")
        logger.debug(f"Context settings: reserved_tokens={self._reserved_response_tokens}, "
                     f"history_strategy={self._history_selection_strategy}, "
                     f"truncation_priority={self._truncation_priority}, "
                     f"min_history={self._minimum_history_messages}, "
                     f"default_rag_k={self._default_rag_k}, "
                     f"rag_combo_strategy={self._rag_combination_strategy}, "
                     f"enable_mcp={self._enable_mcp_globally}, mcp_version={self._mcp_version}")

        # Validate strategies and log warnings if unsupported/fallback needed
        # (Validation logic remains the same as before)
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
        rag_collection: Optional[str] = None,
        # MCP parameter (can be overridden per call, defaults to global config)
        use_mcp: Optional[bool] = None
    ) -> ContextPayload: # Return type updated to ContextPayload
        """
        Prepares the context payload (either List[Message] or MCP Context object)
        to be sent to the LLM provider.

        Handles history selection, RAG retrieval and injection, token counting,
        context truncation, and optional MCP formatting based on configuration and parameters.

        Args:
            session: The current ChatSession containing the message history.
            provider_name: The name of the target provider.
            model_name: The specific model name being used.
            rag_enabled: Whether to perform RAG.
            rag_k: Number of documents to retrieve for RAG (overrides default).
            rag_collection: Vector store collection name for RAG (overrides default).
            use_mcp: If True, attempt to format the context using MCP. If False, use
                     standard List[Message]. If None (default), uses the global
                     `llmcore.enable_mcp` configuration setting.

        Returns:
            The prepared context payload, either as a list of `llmcore.models.Message`
            objects or an `modelcontextprotocol.Context` object if MCP formatting
            is enabled and successful.

        Raises:
            ContextLengthError: If context cannot be reduced below the model's limit.
            ProviderError: If provider interaction fails (token counting, limits).
            ConfigError: If configuration is invalid or provider/model not found, or
                         if MCP is enabled but the SDK is not installed.
            EmbeddingError: If RAG query embedding fails.
            VectorStorageError: If RAG search fails.
            MCPError: If MCP formatting fails.
        """
        # --- Determine if MCP should be used ---
        # Priority: Function argument -> Provider-specific config -> Global config
        mcp_final_enabled = self._enable_mcp_globally # Start with global default
        provider_config = self._config.get(f'providers.{provider_name}', {})
        provider_mcp_setting = provider_config.get('use_mcp') # Check provider specific override
        if provider_mcp_setting is not None:
            mcp_final_enabled = provider_mcp_setting
        if use_mcp is not None: # Check function argument override (highest priority)
            mcp_final_enabled = use_mcp

        if mcp_final_enabled and not mcp_library_available:
            logger.error("MCP formatting requested but 'modelcontextprotocol' library is not installed.")
            raise ConfigError("MCP formatting enabled but 'modelcontextprotocol' SDK is not installed. Install with 'pip install llmcore[mcp]'.")

        # --- Context Preparation Logic (mostly unchanged) ---
        try:
            provider = self._provider_manager.get_provider(provider_name)
        except (ConfigError, ProviderError) as e:
             logger.error(f"ContextManager failed to get provider '{provider_name}': {e}")
             raise

        target_model = model_name or provider.default_model
        if not target_model:
             raise ConfigError(f"Could not determine target model for context preparation (provider: {provider.get_name()}).")

        logger.debug(f"Preparing context for model '{target_model}' (Provider: {provider.get_name()}, RAG: {rag_enabled}, MCP: {mcp_final_enabled}).")

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

        # --- RAG Retrieval (unchanged) ---
        retrieved_docs: List[ContextDocument] = []
        rag_context_str = "" # Keep this for potential inclusion in MCP or standard format
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
                    # Note: rag_context_str is formatted later if needed, or passed to MCP formatter
                except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e:
                     logger.error(f"RAG retrieval failed: {e}. Proceeding without RAG context.")
                     retrieved_docs = []
                except Exception as e:
                     logger.error(f"Unexpected error during RAG retrieval: {e}", exc_info=True)
                     retrieved_docs = []
            else:
                logger.warning("RAG enabled, but no user message found in session to use as query.")
        # --- End RAG Retrieval ---

        # --- History Selection (unchanged, operates on List[Message]) ---
        # Calculate RAG token impact *before* history selection if RAG is prepended early
        # For simplicity, we calculate RAG token cost based on the formatted string *if* not using MCP.
        # If using MCP, token counting might need adjustment based on MCP structure.
        # This part needs careful consideration based on how MCP handles token counting.
        # Assuming count_tokens works on the formatted string for now, even if MCP is used later.
        temp_rag_context_str = self._format_rag_context(retrieved_docs) if retrieved_docs else ""
        rag_tokens = provider.count_tokens(temp_rag_context_str, target_model) if temp_rag_context_str else 0

        history_token_budget = available_tokens - rag_tokens
        if history_token_budget < 0:
             logger.warning(f"RAG context alone ({rag_tokens} tokens) exceeds available budget ({available_tokens}).")
             history_token_budget = 0

        selected_history: List[Message] = []
        if history_token_budget > 0:
            # Select history messages (logic remains the same)
            if self._history_selection_strategy == 'last_n_tokens':
                selected_history, _ = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
            elif self._history_selection_strategy == 'last_n_messages':
                logger.warning("Using fallback 'last_n_tokens' strategy for 'last_n_messages' history selection.")
                selected_history, _ = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
            else: # Fallback
                selected_history, _ = self._select_history_last_n_tokens(
                    session.messages, provider, target_model, history_token_budget
                )
        else:
            system_messages = [msg for msg in session.messages if msg.role == LLMCoreRole.SYSTEM]
            system_tokens = provider.count_message_tokens(system_messages, target_model) if system_messages else 0
            if system_tokens <= history_token_budget:
                 selected_history = system_messages
            else:
                 selected_history = []
                 logger.warning("Not enough token budget even for system messages after RAG context.")

        # --- Combine and Truncate (operates on List[Message]) ---
        final_messages, final_token_count = self._combine_and_truncate(
            provider=provider,
            model_name=target_model,
            token_budget=available_tokens,
            system_messages=[msg for msg in selected_history if msg.role == LLMCoreRole.SYSTEM],
            history_messages=[msg for msg in selected_history if msg.role != LLMCoreRole.SYSTEM],
            rag_context_str=temp_rag_context_str, # Pass formatted string for truncation logic
            rag_docs=retrieved_docs # Pass original docs for RAG truncation logic
        )

        logger.info(f"Final context before potential MCP formatting: {len(final_messages)} messages, {final_token_count} tokens.")

        # --- Final Check (unchanged) ---
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

        # --- MCP Formatting ---
        if mcp_final_enabled:
            logger.debug("Attempting to format final context using Model Context Protocol.")
            try:
                # Map LLMCore Messages and RAG docs to MCP objects
                mcp_messages: List[MCPMessage] = []
                for msg in final_messages:
                    # Skip the temporary RAG context message if it exists in final_messages
                    if msg.id == "rag_context":
                        continue
                    mcp_role = LLMCORE_TO_MCP_ROLE_MAP.get(msg.role)
                    if mcp_role:
                        mcp_messages.append(MCPMessage(role=mcp_role, content=msg.content))
                    else:
                        logger.warning(f"Skipping message with unmappable role for MCP: {msg.role}")

                mcp_knowledge: List[RetrievedKnowledge] = []
                if retrieved_docs: # Use the original retrieved docs before potential truncation
                    for doc in retrieved_docs:
                        # Map metadata - MCP expects Dict[str, str] for source_metadata
                        # We need to convert our potentially complex metadata dict.
                        # For now, let's include 'source' if available, otherwise doc ID.
                        source_meta = {"doc_id": doc.id}
                        if isinstance(doc.metadata, dict):
                            source = doc.metadata.get("source")
                            if source and isinstance(source, str):
                                source_meta["source"] = source
                            # Add other simple string metadata fields if needed

                        mcp_knowledge.append(RetrievedKnowledge(
                            content=doc.content,
                            source_metadata=source_meta,
                            # score=doc.score # Add score if MCP schema supports it
                        ))

                # Create the MCP Context object
                mcp_context = MCPContextObject(
                    messages=mcp_messages,
                    retrieved_knowledge=mcp_knowledge if mcp_knowledge else None,
                    # version=self._mcp_version # Add version if specified
                )
                logger.info(f"Context successfully formatted using MCP (Messages: {len(mcp_messages)}, Knowledge: {len(mcp_knowledge)}).")
                # Return the MCP object instead of the list of messages
                return mcp_context

            except Exception as mcp_e:
                logger.error(f"Failed to format context using MCP: {mcp_e}", exc_info=True)
                raise MCPError(f"MCP formatting failed: {mcp_e}")
        else:
            # Return the standard list of messages if MCP is not enabled
            # Remove the temporary RAG message if it exists
            final_messages_no_rag_marker = [msg for msg in final_messages if msg.id != "rag_context"]
            return final_messages_no_rag_marker

    # --- Helper methods (_format_rag_context, _select_history_last_n_tokens, _combine_and_truncate) remain the same ---
    # They operate internally on List[Message] before the final optional MCP formatting step.

    def _format_rag_context(self, documents: List[ContextDocument]) -> str:
        """Formats retrieved documents into a string for context injection or truncation checks."""
        if not documents:
            return ""
        context_parts = ["--- Retrieved Context ---"]
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i+1}")
            content_snippet = doc.content.replace('\n', ' ').strip()
            context_parts.append(f"\n[Source: {source}]\n{content_snippet}")
        return "\n".join(context_parts) + "\n--- End Context ---"

    def _select_history_last_n_tokens(
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
            system_tokens = provider.count_message_tokens(system_messages, model_name) if system_messages else 0
        except Exception as e:
            raise ProviderError(provider.get_name(), f"Token counting failed for system messages: {e}")

        if system_tokens > token_budget:
             logger.warning(f"System messages ({system_tokens} tokens) alone exceed history budget ({token_budget}). Only including system messages.")
             return system_messages, system_tokens

        selected_messages.extend(system_messages)
        current_tokens += system_tokens

        non_system_messages = [msg for msg in all_messages if msg.role != LLMCoreRole.SYSTEM]
        for msg in reversed(non_system_messages):
            try:
                message_tokens = provider.count_message_tokens([msg], model_name)
            except Exception as e:
                raise ProviderError(provider.get_name(), f"Token counting failed for message {msg.id}: {e}")

            if current_tokens + message_tokens <= token_budget:
                selected_messages.append(msg)
                current_tokens += message_tokens
            else:
                logger.debug(f"History token budget ({token_budget}) reached. Stopping selection.")
                break

        selected_messages.sort(key=lambda m: m.timestamp)
        logger.debug(f"Selected {len(selected_messages)} history messages ({current_tokens} tokens) for budget {token_budget}.")
        return selected_messages, current_tokens


    def _combine_and_truncate(
        self,
        provider: BaseProvider,
        model_name: str,
        token_budget: int,
        system_messages: List[Message],
        history_messages: List[Message],
        rag_context_str: str, # Formatted RAG string used here for truncation logic
        rag_docs: List[ContextDocument] # Original docs needed for RAG truncation
    ) -> Tuple[List[Message], int]:
        """
        Combines history and RAG context based on strategy, then truncates if necessary.
        Returns the final List[Message] (including the temporary RAG message if applicable)
        and the final token count *before* potential MCP formatting.
        """
        combined_context: List[Message] = []
        rag_message: Optional[Message] = None # Temporary message holding formatted RAG string

        # --- Context Combination ---
        if rag_context_str:
            rag_message = Message(role=LLMCoreRole.SYSTEM, content=rag_context_str, session_id="rag_context", id="rag_context") # Assign ID for easier removal later
            try:
                 rag_message.tokens = provider.count_message_tokens([rag_message], model_name)
            except Exception as e:
                 logger.error(f"Failed to count tokens for RAG context message: {e}")
                 rag_message.tokens = 0

            logger.debug(f"Combining context using strategy: {self._rag_combination_strategy}")
            if self._rag_combination_strategy == "prepend_system":
                combined_context.extend(system_messages)
                combined_context.append(rag_message)
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
                      combined_context.append(rag_message)
                      combined_context.extend(history_messages[last_user_idx:])
                 else:
                      combined_context.extend(history_messages)
                      combined_context.append(rag_message)
            else: # Default/Fallback
                 combined_context.extend(system_messages)
                 combined_context.append(rag_message)
                 combined_context.extend(history_messages)
        else:
            combined_context.extend(system_messages)
            combined_context.extend(history_messages)

        # --- Truncation ---
        try:
            current_tokens = provider.count_message_tokens(combined_context, model_name)
        except Exception as e:
             raise ProviderError(provider.get_name(), f"Token counting failed for initial combined context: {e}")

        logger.debug(f"Combined context before truncation: {len(combined_context)} messages, {current_tokens} tokens (Budget: {token_budget}).")

        while current_tokens > token_budget:
            history_msg_indices = [i for i, msg in enumerate(combined_context) if msg.role != LLMCoreRole.SYSTEM and msg.id != "rag_context"]
            can_truncate_history = len(history_msg_indices) > self._minimum_history_messages
            can_truncate_rag = rag_message is not None and rag_docs

            logger.warning(f"Truncation needed: {current_tokens}/{token_budget} tokens. Priority: {self._truncation_priority}. Can truncate history: {can_truncate_history}. Can truncate RAG: {can_truncate_rag}.")

            truncated_something = False
            if self._truncation_priority == "history" and can_truncate_history:
                if history_msg_indices:
                    idx_to_remove = history_msg_indices[0]
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message: {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True
            elif self._truncation_priority == "rag" and can_truncate_rag:
                rag_docs.pop()
                if rag_docs:
                    rag_context_str = self._format_rag_context(rag_docs)
                    rag_message.content = rag_context_str
                    try: rag_message.tokens = provider.count_message_tokens([rag_message], model_name)
                    except Exception as e: logger.error(f"Failed to count tokens for truncated RAG context: {e}"); rag_message.tokens = 0
                    logger.debug(f"Truncated least relevant RAG document. New RAG context tokens: {rag_message.tokens}")
                else:
                    try: combined_context.remove(rag_message) # Remove by object identity
                    except ValueError: logger.error("Failed to find RAG message for removal during truncation.")
                    rag_message = None
                    logger.debug("Removed RAG context entirely after truncating all docs.")
                truncated_something = True
            elif can_truncate_history: # Fallback
                 if history_msg_indices:
                    idx_to_remove = history_msg_indices[0]
                    removed_msg = combined_context.pop(idx_to_remove)
                    logger.debug(f"Truncated oldest history message (fallback): {removed_msg.id} ({removed_msg.role.value})")
                    truncated_something = True
            elif can_truncate_rag: # Fallback
                 rag_docs.pop()
                 if rag_docs:
                    rag_context_str = self._format_rag_context(rag_docs)
                    rag_message.content = rag_context_str
                    try: rag_message.tokens = provider.count_message_tokens([rag_message], model_name)
                    except Exception as e: logger.error(f"Failed to count tokens for truncated RAG context (fallback): {e}"); rag_message.tokens = 0
                    logger.debug(f"Truncated least relevant RAG document (fallback). New RAG context tokens: {rag_message.tokens}")
                 else:
                    try: combined_context.remove(rag_message)
                    except ValueError: logger.error("Failed to find RAG message for removal during fallback truncation.")
                    rag_message = None
                    logger.debug("Removed RAG context entirely (fallback).")
                 truncated_something = True
            else:
                logger.error(f"Cannot truncate context further. Current tokens {current_tokens} exceed budget {token_budget}.")
                break

            if truncated_something:
                try:
                    current_tokens = provider.count_message_tokens(combined_context, model_name)
                    logger.debug(f"Context after truncation step: {len(combined_context)} messages, {current_tokens} tokens.")
                except Exception as e:
                    raise ProviderError(provider.get_name(), f"Token counting failed after truncation step: {e}")
            else:
                 logger.error("Truncation loop failed to remove any message or RAG content, but budget still exceeded.")
                 break

        if current_tokens > token_budget:
             raise ContextLengthError(
                 model_name=model_name, limit=token_budget, actual=current_tokens,
                 message="Context exceeds token limit even after truncation attempts."
             )

        # Return the list *including* the temporary RAG message if it still exists
        # The caller (prepare_context) will decide whether to format this list into MCP
        # or remove the temporary RAG message before returning the final List[Message].
        return combined_context, current_tokens
