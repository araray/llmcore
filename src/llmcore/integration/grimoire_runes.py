"""Optional Grimoire RuneSpec adapters for llmcore tools.

This module is safe to import without Grimoire installed. Grimoire-specific
schema helpers are imported only when conversion functions are called.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from llmcore.models import Tool


@dataclass(frozen=True)
class RuneCommandTool:
    """Converted llmcore tool plus Grimoire provenance metadata."""

    tool: Tool
    metadata: dict[str, Any]
    implementation_key: str


def rune_command_tool_name(rune: Any, command: Any) -> str:
    """Return the stable llmcore tool name for a Grimoire rune command."""
    rune_id = str(getattr(rune, "id", "") or "")
    command_name = str(getattr(command, "name", "") or "")
    return f"{rune_id.replace('/', '__')}__{command_name}"


def rune_command_to_tool(rune: Any, command: Any) -> Tool:
    """Convert one Grimoire command to a provider-agnostic llmcore ``Tool``."""
    return Tool(
        name=rune_command_tool_name(rune, command),
        description=str(
            getattr(command, "summary", None)
            or f"{getattr(rune, 'name', getattr(rune, 'id', 'Rune'))}: {getattr(command, 'name', '')}"
        ),
        parameters=_command_parameters_schema(command),
    )


def rune_command_metadata(rune: Any, command: Any) -> dict[str, Any]:
    """Build llmcore runtime metadata from Grimoire rune/command fields."""
    risk_level = getattr(command, "risk_level", None) or getattr(rune, "risk_level", None)
    requires_approval = bool(getattr(command, "requires_approval", False)) or bool(
        getattr(rune, "requires_approval", False)
    )
    execution_target = getattr(command, "execution_target", None)
    permissions = [_enum_value(permission) for permission in getattr(rune, "permissions", [])]
    tags = [str(tag) for tag in getattr(rune, "tags", [])]
    side_effects = [str(effect) for effect in getattr(command, "side_effects", [])]

    metadata: dict[str, Any] = {
        "source": "grimoire.rune",
        "rune_id": str(getattr(rune, "id", "") or ""),
        "command_name": str(getattr(command, "name", "") or ""),
        "risk_level": _enum_value(risk_level) or "low",
        "requires_approval": requires_approval,
        "permissions": permissions,
        "tags": tags,
        "side_effects": side_effects,
    }
    if execution_target:
        metadata["execution_target"] = str(execution_target)
    content_hash = getattr(rune, "content_hash", None)
    if content_hash:
        metadata["content_hash"] = str(content_hash)
    mappings = getattr(rune, "mappings", None)
    if isinstance(mappings, dict) and mappings:
        metadata["mappings"] = {str(key): str(value) for key, value in mappings.items()}
    return metadata


def rune_commands_to_tools(
    runes: Iterable[Any],
    *,
    implementation_key_prefix: str = "grimoire.runes",
) -> list[RuneCommandTool]:
    """Convert Grimoire runes to llmcore tool definitions and metadata."""
    converted: list[RuneCommandTool] = []
    for rune in runes:
        for command in getattr(rune, "commands", []) or []:
            tool = rune_command_to_tool(rune, command)
            converted.append(
                RuneCommandTool(
                    tool=tool,
                    metadata=rune_command_metadata(rune, command),
                    implementation_key=f"{implementation_key_prefix}.{tool.name}",
                )
            )
    return converted


def register_rune_tools(
    tool_manager: Any,
    runes: Iterable[Any],
    implementations: Mapping[Any, Callable[..., Any]],
    *,
    replace: bool = True,
    implementation_key_prefix: str = "grimoire.runes",
) -> int:
    """Register Grimoire rune commands as llmcore runtime tools.

    Args:
        tool_manager: llmcore ``ToolManager`` or compatible object exposing
            ``register_runtime_tool``.
        runes: Iterable of Grimoire ``RuneSpec`` objects.
        implementations: Mapping from tool name, ``(rune_id, command_name)``,
            or ``"rune_id:command_name"`` to the callable implementation.
        replace: Whether existing runtime registrations may be replaced.
        implementation_key_prefix: Prefix for llmcore implementation keys.

    Returns:
        Number of registered command tools.

    Raises:
        ValueError: If any command lacks a supplied callable implementation.
    """
    count = 0
    for converted in rune_commands_to_tools(
        runes,
        implementation_key_prefix=implementation_key_prefix,
    ):
        implementation = _resolve_implementation(converted, implementations)
        if implementation is None:
            rune_id = converted.metadata["rune_id"]
            command_name = converted.metadata["command_name"]
            raise ValueError(f"No implementation supplied for Grimoire command {rune_id}:{command_name}")

        tool_manager.register_runtime_tool(
            converted.tool,
            converted.implementation_key,
            implementation,
            replace=replace,
            metadata=converted.metadata,
        )
        count += 1
    return count


def _resolve_implementation(
    converted: RuneCommandTool,
    implementations: Mapping[Any, Callable[..., Any]],
) -> Callable[..., Any] | None:
    rune_id = converted.metadata["rune_id"]
    command_name = converted.metadata["command_name"]
    candidates = (
        converted.tool.name,
        (rune_id, command_name),
        f"{rune_id}:{command_name}",
    )
    for key in candidates:
        implementation = implementations.get(key)
        if implementation is not None:
            return implementation
    return None


def _command_parameters_schema(command: Any) -> dict[str, Any]:
    try:
        from grimoire.runes.schema import command_parameters_schema
    except ImportError:
        return _fallback_command_parameters_schema(command)
    return command_parameters_schema(command)


def _fallback_command_parameters_schema(command: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in getattr(command, "params", []) or []:
        param_type = str(getattr(param, "type", None) or "string")
        schema: dict[str, Any] = {"type": "boolean" if param_type == "bool" else param_type}
        for attr, key in (
            ("description", "description"),
            ("enum", "enum"),
            ("pattern", "pattern"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("default", "default"),
        ):
            value = getattr(param, attr, None)
            if value is not None:
                schema[key] = value
        name = str(getattr(param, "name", "") or "")
        if not name:
            continue
        properties[name] = schema
        if bool(getattr(param, "required", False)):
            required.append(name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


__all__ = [
    "RuneCommandTool",
    "register_rune_tools",
    "rune_command_metadata",
    "rune_command_to_tool",
    "rune_command_tool_name",
    "rune_commands_to_tools",
]
