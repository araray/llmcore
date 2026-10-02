# src/llmcore/routing/verifiers/llm_judge.py
"""Verify by asking a cheap model to score the answer.

The fallback for people who want neither a local model nor a second vendor. It
is the weakest of the three verifiers and the module says so: a chat model
asked for a number returns a number-shaped thing, and its calibration is
unknown. An unparseable answer is reported as *unknown* rather than guessed at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import RoutingRequest, Verdict
from . import register_verifier

logger = logging.getLogger(__name__)

__all__ = ["LLMJudgeVerifier"]

_PROMPT = """\
Score how well the answer addresses the request, from 0.0 to 1.0.

Reply with the number alone. No explanation.

Request:
{request}

Answer:
{answer}
"""

_NUMBER = re.compile(r"(?:0?\.\d+|[01](?:\.0+)?|\d{1,3}\s*%)")


@dataclass(slots=True)
class LLMJudgeVerifier:
    """Asks a cheap target to score an answer between 0 and 1."""

    ask: Any
    threshold: float = 0.7
    target: str | None = None
    max_chars: int = 6_000
    name: str = "llm"
    cost_hint: str = "api"

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        if not (response or "").strip():
            return Verdict(
                sufficient=False, score=0.0, rationale="empty response", source=self.name
            )
        try:
            answer = await self.ask(
                _PROMPT.format(
                    request=self._clip(self._request_text(request)),
                    answer=self._clip(response),
                ),
                target=self.target,
            )
        except Exception as exc:
            logger.warning("LLM judge call failed (%s); reporting unknown.", exc)
            return Verdict(
                sufficient=None, rationale=f"verifier call failed: {exc}", source=self.name
            )

        score = self._parse(answer)
        if score is None:
            return Verdict(
                sufficient=None,
                rationale=f"judge answered {str(answer)[:60]!r}, which is not a score",
                source=self.name,
            )
        return Verdict(
            sufficient=score >= self.threshold,
            score=score,
            rationale=f"judge scored {score:.2f} against a threshold of {self.threshold:.2f}",
            source=self.name,
        )

    @staticmethod
    def _parse(answer: str | None) -> float | None:
        if not answer:
            return None
        match = _NUMBER.search(str(answer))
        if not match:
            return None
        text = match.group(0).strip()
        if text.endswith("%"):
            return min(1.0, max(0.0, float(text.rstrip("% ")) / 100.0))
        return min(1.0, max(0.0, float(text)))

    @staticmethod
    def _request_text(request: RoutingRequest) -> str:
        parts = [request.system or ""]
        parts += [
            str(m.get("content", "")) for m in request.messages if isinstance(m.get("content"), str)
        ]
        parts.append(request.prompt or "")
        return "\n".join(part for part in parts if part).strip()

    def _clip(self, text: str) -> str:
        return text if len(text) <= self.max_chars else text[: self.max_chars] + "\n[...truncated]"


def _build(*, config: Mapping[str, Any]) -> LLMJudgeVerifier:
    ask = config.get("ask")
    if ask is None:
        raise ValueError(
            "The 'llm' verifier needs a chat callable; it is wired automatically by "
            "RoutingManager."
        )
    return LLMJudgeVerifier(
        ask=ask,
        threshold=float(config.get("threshold", 0.7)),
        target=config.get("target"),
        max_chars=int(config.get("max_chars", 6_000)),
    )


register_verifier("llm", _build)
