# src/llmcore/routing/verifiers/typesafe_noul.py
"""Verify with a TypeSafe ``noul`` question.

A ``noul`` is a yes/no question whose answer is the *probability of yes*, which
is precisely what a cascade threshold needs. Asking a chat model "is this
answer sufficient?" gets prose that has to be parsed and a confidence that was
never calibrated; asking a ``noul`` gets a number.

This is the second half of the TypeSafe reuse (the first is the ``choice``
classifier): one provider already in llmcore supplies both the
before-the-answer and after-the-answer judgements the design needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import RoutingRequest, Verdict
from . import register_verifier

logger = logging.getLogger(__name__)

__all__ = ["TypeSafeVerifier"]

DEFAULT_QUESTION = (
    "Does this answer fully and correctly address the request? Judge only whether the "
    "request was satisfied, not whether the answer could be longer."
)
DEFAULT_MAX_CHARS = 6_000


@dataclass(slots=True)
class TypeSafeVerifier:
    """Asks TypeSafe whether an answer is sufficient.

    Args:
        provider_getter: Callable returning the TypeSafe provider.
        threshold: Probability at or above which the answer is kept. The
            cascade's own threshold, passed in so one number governs.
        question: The question put to the model.
        model: TypeSafe model id, or ``None`` for the provider's default.
    """

    provider_getter: Any
    threshold: float = 0.7
    question: str = DEFAULT_QUESTION
    model: str | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    name: str = "typesafe_jev"
    cost_hint: str = "api"

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        provider = self.provider_getter()
        if provider is None:
            return Verdict(
                sufficient=None, rationale="TypeSafe provider unavailable", source=self.name
            )
        if not (response or "").strip():
            return Verdict(
                sufficient=False, score=0.0, rationale="empty response", source=self.name
            )

        state = {
            "request": self._clip(self._request_text(request)),
            "answer": self._clip(response),
        }
        try:
            result = await provider.system_one(
                state,
                {"sufficient": {"type": "noul", "instructions": self.question}},
                model=self.model,
            )
        except Exception as exc:
            # "Could not judge", not "insufficient": a verifier outage must
            # not quietly escalate every request to the expensive rung.
            logger.warning("TypeSafe verifier call failed (%s); reporting unknown.", exc)
            return Verdict(
                sufficient=None, rationale=f"verifier call failed: {exc}", source=self.name
            )

        answer = result.nouls.get("sufficient")
        if answer is None:
            return Verdict(
                sufficient=None, rationale="no 'sufficient' answer returned", source=self.name
            )
        score = float(answer.noul)
        return Verdict(
            sufficient=score >= self.threshold,
            score=score,
            rationale=(
                f"TypeSafe noul put sufficiency at {score:.2f} against a threshold of "
                f"{self.threshold:.2f}"
            ),
            source=self.name,
        )

    @staticmethod
    def _request_text(request: RoutingRequest) -> str:
        parts = [request.system or ""]
        parts += [
            str(message.get("content", ""))
            for message in request.messages
            if isinstance(message.get("content"), str)
        ]
        parts.append(request.prompt or "")
        return "\n".join(part for part in parts if part).strip()

    def _clip(self, text: str) -> str:
        return text if len(text) <= self.max_chars else text[: self.max_chars] + "\n[...truncated]"


def _build(*, config: Mapping[str, Any]) -> TypeSafeVerifier:
    getter = config.get("provider_getter")
    if getter is None:
        raise ValueError(
            "The 'typesafe_jev' verifier needs a TypeSafe provider; it is wired automatically "
            "by RoutingManager."
        )
    return TypeSafeVerifier(
        provider_getter=getter,
        threshold=float(config.get("threshold", 0.7)),
        question=str(config.get("question") or DEFAULT_QUESTION),
        model=config.get("model"),
        max_chars=int(config.get("max_chars", DEFAULT_MAX_CHARS)),
    )


register_verifier("typesafe_jev", _build)
register_verifier("jev", _build)
