# src/llmcore/agents/cognitive/phases/_prompting.py
"""Shared registry→messages helper for the cognitive phases (0.52.0).

The phases no longer carry f-string prompt fallbacks or inline SYSTEM
strings — EVERY phase prompt (system + user, atomically) comes from the
prompt registry (grimoire adapter). This helper is the single conversion
point from the registry's role-structured render to llmcore ``Message``
objects, with a compatibility shim for legacy registries that only
implement ``render()``.

Fail-loud: rendering errors propagate. There is no silent fallback — a
broken template must abort the phase, not degrade it.
"""

from __future__ import annotations

import logging
from typing import Any

from llmcore.models import Message, Role

logger = logging.getLogger(__name__)

_ROLE_TO_ENUM: dict[str, Role] = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
}


def require_prompt_registry(prompt_registry: Any, phase: str) -> Any:
    """Assert a prompt registry is present (the control plane is mandatory).

    Raises:
        ValueError: When ``prompt_registry`` is None — construction paths
            (``SingleAgentMode``/``CognitiveCycle``) should have injected the
            grimoire-backed adapter; reaching here without one is a wiring bug.
    """
    if prompt_registry is None:
        raise ValueError(
            f"{phase}: prompt_registry is required (0.52.0) — llmcore agent "
            "prompts come from the grimoire control plane; construct agents "
            "via LLMCore.create_enhanced_agent_manager() or inject a registry"
        )
    return prompt_registry


def messages_from_registry(
    prompt_registry: Any,
    template_id: str,
    variables: dict[str, Any],
) -> list[Message]:
    """Render a template to llmcore ``Message`` objects.

    Prefers the role-structured ``render_messages()`` (grimoire adapter).
    Legacy registries exposing only ``render()`` get their string output
    wrapped as a single USER message (no system message — such registries
    predate the atomic system+user contract).
    """
    render_messages = getattr(prompt_registry, "render_messages", None)
    if callable(render_messages):
        rendered = render_messages(template_id, variables)
    else:  # legacy registry shim
        rendered = [
            {"role": "user", "content": prompt_registry.render(template_id, variables)}
        ]

    messages: list[Message] = []
    for item in rendered:
        role = _ROLE_TO_ENUM.get(str(item.get("role", "user")).lower(), Role.USER)
        content = str(item.get("content", "") or "")
        if content:
            messages.append(Message(role=role, content=content))
    if not messages:
        raise ValueError(f"Template {template_id!r} rendered no message content")
    return messages


def record_template_use(
    prompt_registry: Any,
    template_id: str,
    *,
    success: bool,
    tokens: int | None = None,
) -> None:
    """Best-effort usage recording (never raises)."""
    try:
        if hasattr(prompt_registry, "record_use") and hasattr(
            prompt_registry, "get_template"
        ):
            template = prompt_registry.get_template(template_id)
            version = getattr(template, "active_version", None)
            if version is not None:
                prompt_registry.record_use(
                    version_id=version.id, success=success, tokens=tokens
                )
    except Exception as exc:  # telemetry must never break a phase
        logger.debug("record_use failed for %s: %s", template_id, exc)


__all__ = ["messages_from_registry", "record_template_use", "require_prompt_registry"]
