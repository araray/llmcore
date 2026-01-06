# src/llmcore/memory/manager.py
"""
Memory Management for LLMCore.

This module provides the MemoryManager, which serves as the central, intelligent
retrieval and context preparation interface for the entire hierarchical memory system
(Semantic, Episodic, and Working Memory).

REFACTORED: This class is now a high-level orchestrator. It delegates the detailed
implementation of context assembly and truncation to the `context_builder` module,
and RAG-specific prompt formatting to the `rag_utils` module. Its primary
responsibilities are to manage memory retrieval strategies, handle configuration,
and coordinate the context preparation process.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..embedding.manager import EmbeddingManager
from ..exceptions import (ConfigError, ContextError, ContextLengthError)
from ..models import (ChatSession, ContextDocument, ContextItem,
                      ContextItemType, Message, ContextPreparationDetails,
                      Role as LLMCoreRole)
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from . import context_builder, rag_utils

if TYPE_CHECKING:
    try:
        from confy.loader import Config as ConfyConfig
    except ImportError:
        ConfyConfig = Dict[str, Any]

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages the memory system for LLM interactions.

    Orchestrates the retrieval of information from semantic and episodic memory
    and coordinates the construction of the final context payload by delegating
    to specialized utility modules.
    """

    def __init__(
        self,
        config: "ConfyConfig",
        provider_manager: ProviderManager,
        storage_manager: StorageManager,
        embedding_manager: EmbeddingManager
    ):
        """
        Initializes the MemoryManager.

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

        # Load and cache context management configurations
        self._cm_config = self._load_and_parse_cm_config()

        logger.info("MemoryManager initialized as orchestrator.")
        logger.debug(f"Inclusion priority: {self._cm_config['inclusion_priority_order']}")
        logger.debug(f"Truncation priority: {self._cm_config['truncation_priority_order']}")

    def _load_and_parse_cm_config(self) -> Dict[str, Any]:
        """Loads, parses, and validates the context_management section of the config."""
        cm_config_raw = self._config.get('context_management', {})

        # Parse priorities
        inclusion_priority = self._parse_priority(
            cm_config_raw.get('inclusion_priority', "system_history,explicitly_staged,user_items_active,history_chat,final_user_query"),
            {"system_history", "explicitly_staged", "user_items_active", "history_chat", "final_user_query"}
        )
        truncation_priority = self._parse_priority(
            cm_config_raw.get('truncation_priority', 'history_chat,user_items_active,rag_in_query,explicitly_staged'),
            {"history_chat", "rag_in_query", "user_items_active", "explicitly_staged"}
        )

        # Load RAG template
        default_template = cm_config_raw.get('default_prompt_template', "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        template_path = cm_config_raw.get('prompt_template_path', "")
        template_content = default_template
        if template_path:
            try:
                resolved_path = Path(os.path.expanduser(template_path))
                if resolved_path.is_file():
                    template_content = resolved_path.read_text(encoding='utf-8')
                    logger.info(f"Loaded RAG prompt template from: {resolved_path}")
                else:
                    logger.warning(f"Template file not found: '{resolved_path}'. Using default.")
            except Exception as e:
                logger.warning(f"Failed to read template '{template_path}': {e}. Using default.")

        return {
            "reserved_response_tokens": cm_config_raw.get('reserved_response_tokens', 500),
            "inclusion_priority_order": inclusion_priority,
            "truncation_priority_order": truncation_priority,
            "user_retained_messages_count": cm_config_raw.get('user_retained_messages_count', 5),
            "max_chars_per_user_item": cm_config_raw.get('max_chars_per_user_item', 40000),
            "default_rag_k": cm_config_raw.get('rag_retrieval_k', 3),
            "prompt_template_content": template_content,
        }

    def _parse_priority(self, priority_str: str, valid_items: set) -> List[str]:
        """Helper to parse and validate priority lists from config."""
        priorities = [p.strip().lower() for p in priority_str.split(',')]
        ordered_priorities = [p for p in priorities if p in valid_items]
        if len(ordered_priorities) != len(priorities):
            invalid = set(priorities) - set(ordered_priorities)
            logger.warning(f"Invalid items in priority config: {invalid}. Ignoring.")
        return ordered_priorities or list(valid_items)

    async def _get_precise_context_length(self, provider, model: str) -> int:
        """Gets precise context length for a model using dynamic introspection."""
        try:
            models_details = await provider.get_models_details()
            for detail in models_details:
                if detail.id == model:
                    logger.debug(f"Found precise context length for {model}: {detail.context_length} tokens")
                    return detail.context_length
            logger.debug(f"Model {model} not in details, using provider fallback.")
            return provider.get_max_context_length(model)
        except Exception as e:
            logger.warning(f"Dynamic model introspection failed for {model}: {e}. Using provider fallback.")
            return provider.get_max_context_length(model)

    async def retrieve_relevant_context(self, goal: str) -> List[ContextItem]:
        """
        Primary method for retrieving relevant context from all memory tiers.

        This is the central interface that generates and executes queries against
        Semantic Memory (vector store) and Episodic Memory (episode log) based
        on the agent's current goal.

        Args:
            goal: The agent's current high-level goal or objective.

        Returns:
            A list of ContextItem objects containing relevant information from
            all memory sources, ranked and filtered for relevance.
        """
        logger.debug(f"Retrieving relevant context for goal: '{goal[:100]}...'")
        context_items: List[ContextItem] = []

        # Query Semantic Memory (vector store)
        try:
            vector_storage = self._storage_manager.vector_storage
            if vector_storage:
                goal_embedding = await self._embedding_manager.generate_embedding(goal)
                semantic_results = await vector_storage.similarity_search(
                    query_embedding=goal_embedding, k=self._cm_config['default_rag_k']
                )
                for i, doc in enumerate(semantic_results):
                    context_items.append(ContextItem(
                        id=f"semantic_{doc.id}", type=ContextItemType.RAG_SNIPPET,
                        source_id=doc.id, content=doc.content,
                        metadata={**doc.metadata, "retrieval_score": doc.score, "retrieval_rank": i + 1, "memory_source": "semantic"}
                    ))
                logger.debug(f"Retrieved {len(semantic_results)} items from semantic memory")
        except Exception as e:
            logger.warning(f"Semantic memory search failed: {e}")

        # Placeholder for Episodic Memory query
        # ...

        context_items.sort(key=lambda x: x.metadata.get("retrieval_score", 0.0))
        logger.info(f"Retrieved {len(context_items)} total context items for goal")
        return context_items

    async def prepare_context(
        self,
        session: ChatSession,
        provider_name: str,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        message_inclusion_map: Optional[Dict[str, bool]] = None,
        rag_enabled: bool = False,
        rag_k: Optional[int] = None,
        rag_collection: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> ContextPreparationDetails:
        """
        Prepares the context payload by orchestrating RAG and context building.

        This method coordinates the retrieval of RAG documents, rendering of the
        final user query, and then delegates the complex assembly and truncation
        logic to the `context_builder` module.

        Args:
            (Same as original, see docstrings in `context_builder.py` for details)

        Returns:
            A `ContextPreparationDetails` object with the final context.
        """
        provider = self._provider_manager.get_provider(provider_name)
        target_model = model_name or provider.default_model
        if not target_model:
            raise ConfigError(f"Target model not determined for provider '{provider.get_name()}'.")

        max_model_tokens = await self._get_precise_context_length(provider, target_model)

        last_user_message = next((msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None)
        if not last_user_message:
            raise ContextError("Cannot prepare context without a user query in the session.")

        final_user_query_content = last_user_message.content
        rag_documents_used: Optional[List[ContextDocument]] = None

        if rag_enabled:
            try:
                vector_storage = self._storage_manager.vector_storage
                k = rag_k if rag_k is not None else self._cm_config['default_rag_k']
                query_embedding = await self._embedding_manager.generate_embedding(last_user_message.content)
                rag_documents_used = await vector_storage.similarity_search(
                    query_embedding, k, rag_collection, rag_metadata_filter
                )
                rag_context_str = rag_utils.format_rag_docs_for_context(rag_documents_used)
                final_user_query_content = rag_utils.render_prompt_template(
                    self._cm_config['prompt_template_content'], rag_context_str, last_user_message.content, prompt_template_values
                )
                logger.info(f"RAG retrieved {len(rag_documents_used)} docs.")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}. Proceeding without RAG.", exc_info=True)
                rag_documents_used = []

        context_details = await context_builder.build_context_payload(
            session=session,
            provider=provider,
            target_model=target_model,
            max_model_tokens=max_model_tokens,
            config=self._cm_config,
            active_context_item_ids=active_context_item_ids,
            explicitly_staged_items=explicitly_staged_items,
            message_inclusion_map=message_inclusion_map,
            final_user_query_content=final_user_query_content
        )

        # Augment details with RAG info
        context_details.rag_documents_used = rag_documents_used
        if rag_enabled:
            context_details.rendered_rag_template_content = final_user_query_content

        logger.info(f"Final context prepared for model '{target_model}': {context_details.final_token_count} tokens.")
        return context_details
