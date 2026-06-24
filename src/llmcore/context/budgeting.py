"""Prompt budget helpers for provider context assembly.

The helpers in this module intentionally live in LLMCore because provider
context windows, output reserves, and tool schema payloads are LLM-facing
concerns. Configuration packages should only supply values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from llmcore.tokens import TokenCounter, get_counter


@dataclass(frozen=True)
class ContextBudget:
    """Token budget split between prompt payload and non-message reserves."""

    context_window_tokens: int
    reserved_output_tokens: int = 0
    tool_schema_tokens: int = 0
    safety_margin_tokens: int = 0

    @property
    def reserved_non_prompt_tokens(self) -> int:
        """Tokens reserved for output, tool schemas, and safety margin."""
        return (
            self.reserved_output_tokens
            + self.tool_schema_tokens
            + self.safety_margin_tokens
        )

    @property
    def prompt_tokens_available(self) -> int:
        """Tokens available for chat messages or synthesized prompt context."""
        return max(0, self.context_window_tokens - self.reserved_non_prompt_tokens)

    @property
    def reserve_overflow_tokens(self) -> int:
        """How many reserve tokens exceed the model context window."""
        return max(0, self.reserved_non_prompt_tokens - self.context_window_tokens)

    @property
    def reserve_overflows_context(self) -> bool:
        """Whether reserves alone exceed the model context window."""
        return self.reserve_overflow_tokens > 0

    def to_dict(self) -> dict[str, int | bool]:
        """Return a JSON-safe diagnostic representation."""
        return {
            "context_window_tokens": self.context_window_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "reserved_non_prompt_tokens": self.reserved_non_prompt_tokens,
            "prompt_tokens_available": self.prompt_tokens_available,
            "reserve_overflow_tokens": self.reserve_overflow_tokens,
            "reserve_overflows_context": self.reserve_overflows_context,
        }


def build_context_budget(
    *,
    context_window_tokens: int,
    reserved_output_tokens: int = 0,
    tool_schema_tokens: int = 0,
    safety_margin_tokens: int = 0,
) -> ContextBudget:
    """Build a normalized context budget from possibly untrusted inputs."""
    return ContextBudget(
        context_window_tokens=_nonnegative_int(context_window_tokens),
        reserved_output_tokens=_nonnegative_int(reserved_output_tokens),
        tool_schema_tokens=_nonnegative_int(tool_schema_tokens),
        safety_margin_tokens=_nonnegative_int(safety_margin_tokens),
    )


def estimate_tool_schema_tokens(
    tools: Sequence[Any] | None,
    *,
    model: str | None = None,
    counter: TokenCounter | None = None,
) -> int:
    """Estimate the token footprint of active provider tool schemas."""
    if not tools:
        return 0

    token_counter = counter or get_counter(model)
    payload = [_tool_schema_payload(tool) for tool in tools]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return token_counter.count(encoded)


def should_compress_prompt(
    *,
    prompt_tokens: int,
    prompt_budget_tokens: int,
    threshold: float,
) -> bool:
    """Return whether prompt/input tokens exceed the compression threshold."""
    budget = _nonnegative_int(prompt_budget_tokens)
    if budget <= 0:
        return _nonnegative_int(prompt_tokens) > 0
    normalized_threshold = max(0.0, min(float(threshold), 1.0))
    return _nonnegative_int(prompt_tokens) > budget * normalized_threshold


def _tool_schema_payload(tool: Any) -> Any:
    """Normalize supported tool shapes to the provider-visible schema payload."""
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump()
    elif hasattr(tool, "dict"):
        tool = tool.dict()

    if isinstance(tool, dict):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            return tool
        if {"name", "description", "parameters"} <= set(tool):
            return {"type": "function", "function": tool}
        return tool

    return {"type": type(tool).__name__, "repr": repr(tool)}


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ContextBudget",
    "build_context_budget",
    "estimate_tool_schema_tokens",
    "should_compress_prompt",
]
