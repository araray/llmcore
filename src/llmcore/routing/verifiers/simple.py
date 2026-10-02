# src/llmcore/routing/verifiers/simple.py
"""Verifiers that cost nothing: a user callable, and a few exact checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..models import RoutingRequest, Verdict
from . import register_verifier

logger = logging.getLogger(__name__)

__all__ = ["JsonSchemaVerifier", "NonEmptyVerifier", "ScriptVerifier"]


@dataclass(slots=True)
class ScriptVerifier:
    """Calls a user function and trusts its judgement.

    The most useful verifier there is, because the user's own check is usually
    *exact* where a model's is a guess: run the compiler, validate against the
    schema, execute the test. A cascade built on one of those is not a
    heuristic.

    The callable receives ``(request, response)`` and may return a
    :class:`~llmcore.routing.models.Verdict`, a ``bool``, a ``float`` score,
    or ``None`` for "cannot judge". Returning a bare ``bool`` is the common
    case and must not require importing llmcore's types.
    """

    func: Callable[[RoutingRequest, str], Any] | None = None
    threshold: float = 0.7
    name: str = "script"
    cost_hint: str = "free"

    @classmethod
    def from_spec(cls, spec: str, *, threshold: float = 0.7) -> ScriptVerifier:
        """Load ``module.path:function_name``."""
        if ":" not in spec:
            raise ValueError(f"Script verifier spec {spec!r} must be 'module.path:function_name'.")
        module_path, _, func_name = spec.partition(":")
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        if not callable(func):
            raise ValueError(f"{spec!r} is not callable.")
        return cls(func=func, threshold=threshold)

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        if self.func is None:
            return Verdict(sufficient=None, rationale="no script configured", source=self.name)
        try:
            result = self.func(request, response)
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:
            # A broken check is "cannot judge", not "insufficient". Escalating
            # on a crashing verifier would spend the expensive model on every
            # request until someone noticed.
            logger.warning("Script verifier raised (%s); reporting unknown.", exc)
            return Verdict(
                sufficient=None, rationale=f"script raised: {exc}", source=self.name
            )
        return self._coerce(result)

    def _coerce(self, result: Any) -> Verdict:
        if result is None:
            return Verdict(sufficient=None, rationale="script returned None", source=self.name)
        if isinstance(result, Verdict):
            return result
        if isinstance(result, bool):
            return Verdict(sufficient=result, source=self.name, rationale="script returned a bool")
        if isinstance(result, (int, float)):
            score = float(result)
            return Verdict(
                sufficient=score >= self.threshold,
                score=score,
                source=self.name,
                rationale=f"script scored {score:.2f} against a threshold of {self.threshold:.2f}",
            )
        if isinstance(result, Mapping):
            score = result.get("score")
            sufficient = result.get("sufficient")
            if sufficient is None and score is not None:
                sufficient = float(score) >= self.threshold
            return Verdict(
                sufficient=sufficient,
                score=float(score) if score is not None else None,
                rationale=result.get("rationale"),
                source=self.name,
            )
        logger.warning(
            "Script verifier returned %s, which is not a Verdict, bool, number or mapping; "
            "reporting unknown.",
            type(result).__name__,
        )
        return Verdict(sufficient=None, rationale="unrecognised return value", source=self.name)


@dataclass(slots=True)
class NonEmptyVerifier:
    """Fails an answer that is empty, truncated or a bare refusal to try.

    The floor below which no cascade should keep a response. Cheap enough to
    stack in front of a paid verifier, and it catches the failure that costs
    most: a cheap model returning nothing useful, which a cascade would
    otherwise keep because nothing was technically wrong.

    It reports ``None`` for anything that merely *looks* fine, rather than
    ``True`` — passing a non-empty answer would claim a judgement it has not
    made, and the cascade's ``on_unknown`` setting is the right place for that
    decision.
    """

    min_chars: int = 16
    name: str = "non_empty"
    cost_hint: str = "free"

    _EVASIONS = (
        re.compile(r"^\s*i (?:can'?t|cannot|am unable to) (?:help|assist|do that)", re.IGNORECASE),
        re.compile(r"^\s*(?:as an ai|i'?m (?:just )?an ai)\b", re.IGNORECASE),
    )

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        text = (response or "").strip()
        if not text:
            return Verdict(sufficient=False, score=0.0, rationale="empty response", source=self.name)
        if len(text) < self.min_chars:
            return Verdict(
                sufficient=False,
                score=0.0,
                rationale=f"response is {len(text)} characters, below {self.min_chars}",
                source=self.name,
            )
        for pattern in self._EVASIONS:
            if pattern.search(text):
                return Verdict(
                    sufficient=False,
                    score=0.0,
                    rationale="response declines to attempt the task",
                    source=self.name,
                )
        return Verdict(
            sufficient=None,
            rationale="nothing obviously wrong; no judgement on quality",
            source=self.name,
        )


@dataclass(slots=True)
class JsonSchemaVerifier:
    """Checks that the response is JSON, optionally against required keys.

    An exact check, which is the point. For structured output this replaces a
    model's opinion with a parser's answer, and it is the cheapest useful
    cascade anyone can configure.
    """

    required_keys: tuple[str, ...] = ()
    name: str = "json"
    cost_hint: str = "free"

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        text = (response or "").strip()
        # Tolerate a fenced block, which models add even when told not to.
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError) as exc:
            return Verdict(
                sufficient=False, score=0.0, rationale=f"not valid JSON: {exc}", source=self.name
            )
        if self.required_keys:
            if not isinstance(parsed, Mapping):
                return Verdict(
                    sufficient=False,
                    score=0.0,
                    rationale=f"expected a JSON object, got {type(parsed).__name__}",
                    source=self.name,
                )
            missing = [key for key in self.required_keys if key not in parsed]
            if missing:
                return Verdict(
                    sufficient=False,
                    score=0.0,
                    rationale=f"missing required key(s): {', '.join(missing)}",
                    source=self.name,
                )
        return Verdict(sufficient=True, score=1.0, rationale="valid JSON", source=self.name)


def _build_script(*, config: Mapping[str, Any]) -> ScriptVerifier:
    spec = config.get("script") or config.get("verifier_script")
    threshold = float(config.get("threshold", 0.7))
    if callable(spec):
        return ScriptVerifier(func=spec, threshold=threshold)
    if not spec:
        raise ValueError(
            'The "script" verifier needs routing.cascade.script = "my_module:check_answer".'
        )
    return ScriptVerifier.from_spec(str(spec), threshold=threshold)


def _build_non_empty(*, config: Mapping[str, Any]) -> NonEmptyVerifier:
    return NonEmptyVerifier(min_chars=int(config.get("min_chars", 16)))


def _build_json(*, config: Mapping[str, Any]) -> JsonSchemaVerifier:
    keys = config.get("required_keys") or ()
    return JsonSchemaVerifier(required_keys=tuple(str(key) for key in keys))


register_verifier("script", _build_script)
register_verifier("non_empty", _build_non_empty)
register_verifier("json", _build_json)
