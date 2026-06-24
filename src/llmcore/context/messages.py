"""Message-list utilities for provider-safe context pruning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def sanitize_tool_message_pairs(messages: Sequence[T]) -> list[T]:
    """Remove or repair invalid tool-call history after context pruning.

    Provider-native tool calling generally requires each retained assistant
    ``tool_calls`` entry to be followed by exactly one retained ``tool`` result
    with a matching ``tool_call_id``. Compression and truncation can drop one
    side of that pair. This helper removes orphan tool results and strips
    dangling assistant tool calls while preserving normal message content.
    """
    result_ids_after = _tool_result_suffixes(messages)
    sanitized: list[T] = []
    open_call_ids: set[str] = set()
    fulfilled_call_ids: set[str] = set()

    for index, message in enumerate(messages):
        if _is_tool_result_message(message):
            result_id = _string_field(message, "tool_call_id", "call_id", "id")
            if result_id and result_id in open_call_ids and result_id not in fulfilled_call_ids:
                sanitized.append(message)
                fulfilled_call_ids.add(result_id)
            continue

        tool_calls = _tool_calls_from_message(message)
        if tool_calls:
            remaining_result_ids = result_ids_after[index + 1]
            filtered_calls = [
                call
                for call in tool_calls
                if (call_id := _string_field(call, "id", "tool_call_id", "call_id"))
                and call_id in remaining_result_ids
                and call_id not in fulfilled_call_ids
            ]
            if filtered_calls:
                message = _with_tool_calls(message, filtered_calls)
                open_call_ids.update(
                    call_id
                    for call in filtered_calls
                    if (call_id := _string_field(call, "id", "tool_call_id", "call_id"))
                )
            else:
                message = _without_tool_calls(message)

        sanitized.append(message)

    return sanitized


def _tool_result_suffixes(messages: Sequence[Any]) -> list[set[str]]:
    suffixes: list[set[str]] = [set() for _ in range(len(messages) + 1)]
    running: set[str] = set()
    for index in range(len(messages) - 1, -1, -1):
        suffixes[index + 1] = set(running)
        message = messages[index]
        if _is_tool_result_message(message):
            result_id = _string_field(message, "tool_call_id", "call_id", "id")
            if result_id:
                running.add(result_id)
        suffixes[index] = set(running)
    return suffixes


def _with_tool_calls(message: T, tool_calls: list[Any]) -> T:
    if isinstance(message, dict):
        copied = dict(message)
        if "tool_calls" in copied:
            copied["tool_calls"] = tool_calls
        else:
            metadata = dict(copied.get("metadata") or {})
            metadata["tool_calls"] = tool_calls
            copied["metadata"] = metadata
        return copied  # type: ignore[return-value]

    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        new_metadata = dict(metadata)
        new_metadata["tool_calls"] = tool_calls
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"metadata": new_metadata})  # type: ignore[return-value]
        message.metadata = new_metadata  # type: ignore[attr-defined]
    return message


def _without_tool_calls(message: T) -> T:
    if isinstance(message, dict):
        copied = dict(message)
        copied.pop("tool_calls", None)
        if isinstance(copied.get("metadata"), dict):
            metadata = dict(copied["metadata"])
            metadata.pop("tool_calls", None)
            copied["metadata"] = metadata
        return copied  # type: ignore[return-value]

    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict) and "tool_calls" in metadata:
        new_metadata = dict(metadata)
        new_metadata.pop("tool_calls", None)
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"metadata": new_metadata})  # type: ignore[return-value]
        message.metadata = new_metadata  # type: ignore[attr-defined]
    return message


def _tool_calls_from_message(message: Any) -> list[Any]:
    tool_calls = _field(message, "tool_calls")
    if tool_calls is None:
        metadata = _field(message, "metadata", default={})
        if isinstance(metadata, dict):
            tool_calls = metadata.get("tool_calls")
    if not isinstance(tool_calls, list | tuple):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def _is_tool_result_message(message: Any) -> bool:
    role = _normalized_role(message)
    if role == "tool":
        return True
    item_type = _string_field(message, "type", "kind")
    return item_type in {"tool_result", "activity_result"}


def _normalized_role(message: Any) -> str | None:
    role = _field(message, "role")
    if role is None:
        return None
    value = getattr(role, "value", role)
    return str(value).lower()


def _string_field(value: Any, *names: str) -> str | None:
    result = _field(value, *names)
    if result is None:
        return None
    text = str(result).strip()
    return text or None


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    metadata = None
    if isinstance(value, dict):
        metadata = value.get("metadata")
    elif hasattr(value, "metadata"):
        metadata = value.metadata
    if isinstance(metadata, dict):
        for name in names:
            if name in metadata:
                return metadata[name]
    return default


__all__ = ["sanitize_tool_message_pairs"]
