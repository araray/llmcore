# src/llmcore/routing/transforms/magic.py
"""Strip routing markers before the prompt leaves the process.

A separate transform rather than part of the magic-string classifier, because
the stripping has to happen whether or not the marker was acted on. A marker
that reaches a provider is llmcore's internals leaking into someone's context
window — and in an agent harness the model will quote it back, which turns a
routing convention into a visible bug.

This transform is installed automatically whenever the ``magic_string``
classifier is in the chain, so a user cannot configure the leak by forgetting
it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import RoutingRequest, Target, TransformAction, TransformResult
from ..classifiers.free import (
    DEFAULT_MAGIC_PATTERN,
    compile_magic_pattern,
    strip_magic_strings,
)
from . import register_transform

__all__ = ["StripMagicStrings"]


@dataclass(slots=True)
class StripMagicStrings:
    """Removes routing markers from the prompt, system text and messages."""

    pattern: re.Pattern[str] = None  # type: ignore[assignment]
    name: str = "strip_magic"
    cost_hint: str = "free"

    def __post_init__(self) -> None:
        if self.pattern is None:
            self.pattern = re.compile(DEFAULT_MAGIC_PATTERN, re.IGNORECASE)

    async def apply(self, request: RoutingRequest, target: Target) -> TransformResult:
        changes: dict[str, Any] = {}

        if request.prompt and self.pattern.search(request.prompt):
            changes["prompt"] = strip_magic_strings(request.prompt, self.pattern)
        if request.system and self.pattern.search(request.system):
            changes["system"] = strip_magic_strings(request.system, self.pattern)

        touched_messages = False
        messages = []
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, str) and self.pattern.search(content):
                message = {**message, "content": strip_magic_strings(content, self.pattern)}
                touched_messages = True
            messages.append(message)
        if touched_messages:
            changes["messages"] = tuple(messages)

        if not changes:
            return TransformResult(action=TransformAction.ALLOW, source=self.name)
        return TransformResult(
            action=TransformAction.ALLOW,
            reason="removed routing marker(s) before egress",
            source=self.name,
            **changes,
        )


def _build(*, config: Mapping[str, Any]) -> StripMagicStrings:
    raw = config.get("magic_pattern") or config.get("pattern")
    return StripMagicStrings(pattern=compile_magic_pattern(raw))


register_transform("strip_magic", _build)
