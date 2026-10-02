# src/llmcore/routing/classifiers/typesafe_jev.py
"""Classify with a TypeSafe ``choice`` question.

This is the neatest reuse in the whole design, and it is why the request's
mention of "jev" was worth following up: TypeSafe's ``choice`` primitive
returns the picked option, a probability for *every* option, and a calibrated
confidence — which is exactly the shape of
:class:`~llmcore.routing.models.Classification`. No parsing of a model's prose
answer, no "reply with one word and nothing else", no failure mode where the
judge writes a paragraph.

It costs one cheap API call, so it sits after every free signal in the chain
and runs only when they all abstain.

Lane *descriptions* are the option descriptions, which makes them load-bearing
rather than decorative: a lane described as "multi-step reasoning, proofs,
architecture review" routes far better than a bare ``deep``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..models import Classification, RoutingRequest
from . import register_classifier

logger = logging.getLogger(__name__)

__all__ = ["TypeSafeClassifier"]

#: How much of the prompt to send. Classification needs the shape of the
#: request, not all of it, and a classifier that re-sends a 200k context to
#: save money would be self-defeating.
DEFAULT_MAX_CHARS = 4_000


@dataclass(slots=True)
class TypeSafeClassifier:
    """Asks TypeSafe which lane a request belongs in.

    Args:
        provider_getter: Callable returning the TypeSafe provider. Injected
            rather than imported so routing does not depend on a provider
            manager existing, which keeps this testable without credentials.
        lanes: The lane table; names and descriptions become the options.
        model: TypeSafe model id, or ``None`` for the provider's default.
        question: The instruction shown to the model.
        max_chars: Truncation budget for the prompt.
    """

    provider_getter: Any
    lanes: Mapping[str, Any] = field(default_factory=dict)
    model: str | None = None
    question: str = (
        "Which of these handling lanes should this request be routed to? "
        "Judge by what the request actually requires, not by its length."
    )
    max_chars: int = DEFAULT_MAX_CHARS
    name: str = "typesafe_jev"
    cost_hint: str = "api"
    authority: str = "inferred"

    async def classify(self, request: RoutingRequest) -> Classification | None:
        if len(self.lanes) < 2:
            # A choice needs options. One lane is not a decision.
            return None

        provider = self.provider_getter()
        if provider is None:
            return None

        criteria = {
            name: (getattr(lane, "description", None) or f"The '{name}' lane")
            for name, lane in self.lanes.items()
        }
        state = self._state(request)
        if not state:
            return None

        result = await provider.system_one(
            state,
            {"lane": {"type": "choice", "instructions": self.question, "criteria": criteria}},
            model=self.model,
        )
        answer = result.choices.get("lane")
        if answer is None:
            logger.debug("TypeSafe returned no 'lane' answer; abstaining.")
            return None

        return Classification(
            lane=answer.choice.strip().lower(),
            confidence=float(answer.confidence),
            scores={k: float(v) for k, v in (answer.probabilities or {}).items()},
            source=self.name,
            rationale=(
                f"TypeSafe choice over {len(criteria)} lanes "
                f"(confidence {answer.confidence:.2f})"
            ),
        )

    def _state(self, request: RoutingRequest) -> str:
        parts: list[str] = []
        if request.system:
            parts.append(f"[system] {request.system}")
        for message in request.messages[-4:]:
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(f"[{message.get('role', 'user')}] {content}")
        if request.prompt:
            parts.append(f"[user] {request.prompt}")
        if request.tools:
            parts.append(f"[tools available] {', '.join(request.tools)}")
        text = "\n".join(parts).strip()
        if len(text) > self.max_chars:
            # Keep both ends: the instruction is usually at the top and the
            # actual question at the bottom, and dropping either loses the
            # thing being classified.
            head = self.max_chars // 2
            tail = self.max_chars - head
            text = f"{text[:head]}\n[...truncated...]\n{text[-tail:]}"
        return text


def _build(*, config: Mapping[str, Any], lanes: Mapping[str, Any]) -> TypeSafeClassifier:
    getter = config.get("provider_getter")
    if getter is None:
        raise ValueError(
            "The 'typesafe_jev' classifier needs a TypeSafe provider; it is wired "
            "automatically by RoutingManager."
        )
    typesafe_config = dict(config.get("typesafe") or {})
    return TypeSafeClassifier(
        provider_getter=getter,
        lanes=lanes,
        model=typesafe_config.get("model"),
        question=str(typesafe_config.get("question") or TypeSafeClassifier.question),
        max_chars=int(typesafe_config.get("max_chars", DEFAULT_MAX_CHARS)),
    )


register_classifier("typesafe_jev", _build)
register_classifier("jev", _build)
