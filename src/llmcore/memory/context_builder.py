# src/llmcore/memory/context_builder.py
"""
Core context assembly and truncation logic for the MemoryManager.

This module is responsible for the detailed process of building the final
context payload to be sent to an LLM. It gathers various components like
system messages, chat history, and user-provided context items, assembles them
according to a defined priority, and truncates them intelligently if the
payload exceeds the model's token limit.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import ChatSession, ContextItem, ContextItemType, ContextPreparationDetails, Message
from ..models import Role as LLMCoreRole
from ..providers.base import BaseProvider

logger = logging.getLogger(__name__)


async def build_context_payload(
    session: ChatSession,
    provider: BaseProvider,
    target_model: str,
    max_model_tokens: int,
    config: dict[str, Any],
    active_context_item_ids: list[str] | None = None,
    explicitly_staged_items: list[Message | ContextItem] | None = None,
    message_inclusion_map: dict[str, bool] | None = None,
    final_user_query_content: str = "",
) -> ContextPreparationDetails:
    """
    Assembles and truncates the context payload, returning detailed information.

    This is the core function that implements the multi-stage process of
    assembling the final context. It prioritizes what to include, and then
    decides what to truncate if the model's token limit is exceeded.

    Args:
        session: The ChatSession containing messages and context items.
        provider: The LLM provider instance for token counting.
        target_model: The specific model name.
        max_model_tokens: The precise context length for the target model.
        config: A dictionary containing context management settings.
        active_context_item_ids: List of context item IDs to include from the session.
        explicitly_staged_items: Items to include with high priority.
        message_inclusion_map: A map to filter chat history messages.
        final_user_query_content: The final, potentially RAG-formatted, user query.

    Returns:
        A `ContextPreparationDetails` object containing the final messages,
        token counts, and truncation details.
    """
    reserved_response_tokens = config.get("reserved_response_tokens", 500)
    inclusion_priority = config.get("inclusion_priority_order", [])
    truncation_priority = config.get("truncation_priority_order", [])

    available_tokens_for_prompt = max_model_tokens - reserved_response_tokens
    truncation_actions: dict[str, Any] = {"details": []}

    all_categories = set(inclusion_priority) | set(truncation_priority)
    components: dict[str, list[Message]] = {cat: [] for cat in all_categories}
    component_tokens: dict[str, int] = dict.fromkeys(all_categories, 0)

    # 1. Gather and Prepare All Potential Components
    # System messages from session history
    for msg in session.messages:
        if msg.role == LLMCoreRole.SYSTEM:
            if msg.tokens is None:
                msg.tokens = await provider.count_message_tokens([msg], target_model)
            components["system_history"].append(msg)
            component_tokens["system_history"] += msg.tokens or 0

    # Explicitly staged items
    if explicitly_staged_items:
        for item in explicitly_staged_items:
            formatted_msg = await _format_and_tokenize_item_as_message(
                item, provider, target_model, "explicitly_staged", config
            )
            components["explicitly_staged"].append(formatted_msg)
            component_tokens["explicitly_staged"] += formatted_msg.tokens or 0

    # Active user items from session pool
    if active_context_item_ids and session.context_items:
        for item_id in active_context_item_ids:
            item = session.get_context_item(item_id)
            if item:
                formatted_msg = await _format_and_tokenize_item_as_message(
                    item, provider, target_model, "user_items_active", config
                )
                components["user_items_active"].append(formatted_msg)
                component_tokens["user_items_active"] += formatted_msg.tokens or 0

    # Final User Query
    last_user_message_obj = next(
        (msg for msg in reversed(session.messages) if msg.role == LLMCoreRole.USER), None
    )
    final_query_message = Message(
        role=LLMCoreRole.USER,
        content=final_user_query_content,
        session_id=session.id,
        id=f"final_query_{last_user_message_obj.id if last_user_message_obj else 'new'}",
    )
    final_query_message.tokens = await provider.count_message_tokens(
        [final_query_message], target_model
    )
    components["final_user_query"] = [final_query_message]
    component_tokens["final_user_query"] = final_query_message.tokens or 0

    # 2. Initial Assembly & Budget for History
    history_messages = [
        msg
        for msg in session.messages
        if msg.role != LLMCoreRole.SYSTEM
        and (not last_user_message_obj or msg.id != last_user_message_obj.id)
    ]
    if message_inclusion_map:
        history_messages = [
            msg for msg in history_messages if message_inclusion_map.get(msg.id, True)
        ]

    tokens_for_non_history = sum(
        component_tokens.get(cat, 0) for cat in inclusion_priority if cat != "history_chat"
    )
    budget_for_history = available_tokens_for_prompt - tokens_for_non_history

    built_history, built_tokens = await _build_history_messages(
        history_messages, provider, target_model, budget_for_history, config
    )
    components["history_chat"] = built_history
    component_tokens["history_chat"] = built_tokens

    # 3. Assemble Final Payload (Pre-truncation)
    final_payload_messages: list[Message] = []
    current_tokens = 0
    for category in inclusion_priority:
        for msg in components[category]:
            final_payload_messages.append(msg)
            current_tokens += msg.tokens or 0

    # 4. Truncation if Over Budget
    if current_tokens > available_tokens_for_prompt:
        logger.info(
            f"Context over budget ({current_tokens}/{available_tokens_for_prompt}). Applying truncation."
        )
        for category_to_truncate in truncation_priority:
            if current_tokens <= available_tokens_for_prompt:
                break

            tokens_to_free = current_tokens - available_tokens_for_prompt
            original_count = len(components[category_to_truncate])
            truncated_list, freed_tokens = _truncate_message_list_from_start(
                components[category_to_truncate], tokens_to_free
            )

            if freed_tokens > 0:
                removed_count = original_count - len(truncated_list)
                truncation_actions["details"].append(
                    f"Truncated '{category_to_truncate}': removed {removed_count} item(s), freed {freed_tokens} tokens."
                )
                components[category_to_truncate] = truncated_list
                current_tokens -= freed_tokens

        # Re-assemble final payload after truncation
        final_payload_messages = []
        current_tokens = 0
        for category in inclusion_priority:
            for msg in components[category]:
                final_payload_messages.append(msg)
                current_tokens += msg.tokens or 0

    # Final check
    if current_tokens > available_tokens_for_prompt:
        # This should be a rare case, but as a safeguard
        logger.error(
            f"Context still too long ({current_tokens}/{available_tokens_for_prompt}) after all truncation."
        )
        # We might need a final, more aggressive truncation here, or raise an error.
        # For now, we'll let the caller handle the oversized payload.

    final_token_count = await provider.count_message_tokens(final_payload_messages, target_model)

    return ContextPreparationDetails(
        prepared_messages=final_payload_messages,
        final_token_count=final_token_count,
        max_tokens_for_model=max_model_tokens,
        truncation_actions_taken=truncation_actions,
    )


async def _format_and_tokenize_item_as_message(
    item: Message | ContextItem,
    provider: BaseProvider,
    target_model: str,
    item_category: str,
    config: dict[str, Any],
) -> Message:
    """Formats a ContextItem into a Message or tokenizes an existing Message."""

    # Type validation: ensure item is Message or ContextItem
    if not isinstance(item, (Message, ContextItem)):
        raise TypeError(
            f"explicitly_staged_items must contain Message or ContextItem objects, "
            f"got {type(item).__name__}. Ensure all items are properly typed."
        )

    if isinstance(item, Message):
        if item.tokens is None:
            item.tokens = await provider.count_message_tokens([item], target_model)
        return item

    max_chars_per_item = config.get("max_chars_per_user_item", 40000)
    content_for_llm = item.content
    item.is_truncated = False
    if not item.metadata.get("ignore_char_limit", False) and len(item.content) > max_chars_per_item:
        content_for_llm = item.content[:max_chars_per_item]
        item.is_truncated = True

    item_type_str = item.type.value if isinstance(item.type, ContextItemType) else str(item.type)
    source_desc = f"SourceID: {item.source_id}" if item.source_id else f"ID: {item.id}"
    header = f"--- Context Item ({source_desc}, Type: {item_type_str}) ---"
    footer = f"--- End Context Item ({source_desc}) ---"
    formatted_content = f"{header}\n{content_for_llm}\n{footer}"

    msg = Message(
        role=LLMCoreRole.USER,  # Using USER role for context items to be clearly separated
        content=formatted_content,
        session_id=item.metadata.get("session_id_for_message", "context_item_session"),
        id=f"{item_category}_{item.id}",
    )
    msg.tokens = await provider.count_message_tokens([msg], target_model)
    return msg


async def _build_history_messages(
    history_messages: list[Message],
    provider: BaseProvider,
    target_model: str,
    budget: int,
    config: dict[str, Any],
) -> tuple[list[Message], int]:
    """Builds the chat history part of the context within a given token budget."""
    if budget <= 0 or not history_messages:
        return [], 0

    user_retained_count = config.get("user_retained_messages_count", 5)

    for msg in history_messages:
        if msg.tokens is None:
            msg.tokens = await provider.count_message_tokens([msg], target_model)

    selected_history: list[Message] = []
    current_tokens = 0

    # Prioritize N most recent user messages and their preceding assistant responses
    retained_for_now: list[Message] = []
    tokens_for_retained = 0
    num_user_retained = 0

    for i in range(len(history_messages) - 1, -1, -1):
        msg = history_messages[i]
        msg_tokens = msg.tokens or 0

        if tokens_for_retained + msg_tokens <= budget:
            retained_for_now.insert(0, msg)
            tokens_for_retained += msg_tokens
            if msg.role == LLMCoreRole.USER:
                num_user_retained += 1
                if user_retained_count > 0 and num_user_retained >= user_retained_count:
                    break
        else:
            break

    selected_history.extend(retained_for_now)
    current_tokens = tokens_for_retained

    # Backfill with older messages if budget allows
    remaining_budget = budget - current_tokens
    if remaining_budget > 0:
        ids_in_selected = {m.id for m in selected_history}
        older_candidates = [msg for msg in history_messages if msg.id not in ids_in_selected]
        older_candidates.sort(key=lambda m: m.timestamp, reverse=True)

        backfill_history: list[Message] = []
        tokens_for_backfill = 0
        for msg in older_candidates:
            msg_tokens = msg.tokens or 0
            if tokens_for_backfill + msg_tokens <= remaining_budget:
                backfill_history.insert(0, msg)
                tokens_for_backfill += msg_tokens
            else:
                break
        selected_history = backfill_history + selected_history
        current_tokens += tokens_for_backfill

    return selected_history, current_tokens


def _truncate_message_list_from_start(
    messages: list[Message], tokens_to_free: int
) -> tuple[list[Message], int]:
    """Removes messages from the beginning of a list to free up tokens."""
    freed_so_far = 0
    truncated_list = list(messages)
    while freed_so_far < tokens_to_free and truncated_list:
        msg_to_remove = truncated_list.pop(0)
        freed_so_far += msg_to_remove.tokens or 0
    return truncated_list, freed_so_far
