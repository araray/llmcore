# src/llmcore/routing/classifiers/llm_classifier.py
"""Classify by asking a cheap model, for people who want neither a local
model nor a second vendor.

This dogfoods the library: the classifier is itself an llmcore target, so it
inherits pools, failover, cost accounting and everything else. It is also the
least precise of the paid options, because a chat model asked to pick one word
sometimes picks several — so the parser is deliberately forgiving and
abstains rather than guessing when the answer does not match a lane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..models import Classification, RoutingRequest
from . import register_classifier

logger = logging.getLogger(__name__)

__all__ = ["LLMClassifier"]

DEFAULT_MAX_CHARS = 4_000

_PROMPT = """\
Classify the request below into exactly one handling lane.

Lanes:
{lanes}

Answer with the lane name alone. No explanation, no punctuation.

Request:
{state}
"""


@dataclass(slots=True)
class LLMClassifier:
    """Asks a cheap target which lane a request belongs in.

    Args:
        ask: ``async (prompt: str, target: str | None) -> str``. Injected by
            the routing manager so this module needs no import of the chat
            API, which would be circular.
        lanes: The lane table; names and descriptions go into the prompt.
        target: Spec string of the model to ask. Should be a cheap one — a
            classifier that costs as much as the call it is routing has no
            reason to exist.
    """

    ask: Any
    lanes: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    #: Confidence reported for a match. A fixed number because a chat model's
    #: self-reported certainty is not calibrated, and inventing a per-answer
    #: score would dress a guess up as a measurement. It sits just above the
    #: default floor so the answer is used, but a local encoder or TypeSafe —
    #: both of which report real distributions — can be preferred over it.
    confidence: float = 0.6
    name: str = "llm"
    cost_hint: str = "api"
    authority: str = "inferred"

    async def classify(self, request: RoutingRequest) -> Classification | None:
        if len(self.lanes) < 2:
            return None
        state = self._state(request)
        if not state:
            return None

        described = "\n".join(
            f"- {name}: {getattr(lane, 'description', None) or 'no description given'}"
            for name, lane in self.lanes.items()
        )
        answer = await self.ask(
            _PROMPT.format(lanes=described, state=state), target=self.target
        )
        lane = self._match(answer)
        if lane is None:
            logger.debug("LLM classifier answered %r, which matches no lane; abstaining.", answer)
            return None
        return Classification(
            lane=lane,
            confidence=self.confidence,
            source=self.name,
            rationale=f"{self.target or 'the configured classifier model'} answered {lane!r}",
        )

    def _match(self, answer: str | None) -> str | None:
        """Find a lane in the answer, tolerating a chatty model.

        Exact match first, then a contained lane name. Where two lane names
        both appear the answer is ambiguous and we abstain, because picking
        the first would be arbitrary and the free classifiers or the default
        pool are better than a coin flip.
        """
        if not answer:
            return None
        text = answer.strip().strip(".\"'`").lower()
        if text in self.lanes:
            return text
        hits = [name for name in self.lanes if name in text]
        if len(hits) == 1:
            return hits[0]
        return None

    def _state(self, request: RoutingRequest) -> str:
        parts = [request.system or ""]
        parts += [
            str(m.get("content", "")) for m in request.messages[-4:] if isinstance(m.get("content"), str)
        ]
        parts.append(request.prompt or "")
        text = "\n".join(part for part in parts if part).strip()
        return text[: self.max_chars]


def _build(*, config: Mapping[str, Any], lanes: Mapping[str, Any]) -> LLMClassifier:
    ask = config.get("ask")
    if ask is None:
        raise ValueError(
            "The 'llm' classifier needs a chat callable; it is wired automatically by "
            "RoutingManager."
        )
    llm_config = dict(config.get("llm") or {})
    return LLMClassifier(
        ask=ask,
        lanes=lanes,
        target=llm_config.get("target"),
        max_chars=int(llm_config.get("max_chars", DEFAULT_MAX_CHARS)),
        confidence=float(llm_config.get("confidence", 0.6)),
    )


register_classifier("llm", _build)
