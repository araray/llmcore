# src/llmcore/routing/classifiers/free.py
"""The four classifiers that cost nothing to run.

These cover the common case, and they come first for that reason: if the
caller already said what they wanted, paying a model to guess it is absurd.
They also make the whole feature usable with no optional dependencies, no
local model and no extra API call — which is the difference between a
routing feature people turn on and one they read about.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..models import Classification, RoutingRequest, Target
from . import register_classifier

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAGIC_PATTERN",
    "compile_magic_pattern",
    "HeuristicClassifier",
    "HintClassifier",
    "MagicStringClassifier",
    "ScriptClassifier",
    "strip_magic_strings",
]


# ---------------------------------------------------------------------------
# hint
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HintClassifier:
    """Reads an explicit instruction off the request.

    ``chat("...", lane="deep")``, ``complexity="high"``, ``effort="minimal"``
    or ``target="xai:grok-4.1"`` — whatever the caller stated. It always wins
    when present, and reports no confidence because there is nothing to be
    uncertain about: this is not a guess, it is an instruction.

    ``complexity`` is mapped through the lane table when a lane of that name
    exists, so ``complexity="high"`` lands in the user's own ``high`` lane
    rather than in a vocabulary llmcore invented. Where no such lane exists it
    is carried as an effort level instead, which is the other thing people
    mean by it.
    """

    name: str = "hint"
    cost_hint: str = "free"
    authority: str = "caller"
    lanes: Mapping[str, Any] = None  # type: ignore[assignment]

    #: Keys consulted on ``RoutingRequest.hints``, in precedence order.
    LANE_KEYS = ("lane", "route")
    COMPLEXITY_KEYS = ("complexity", "difficulty")
    EFFORT_KEYS = ("effort", "reasoning_effort", "thinking")

    def __post_init__(self) -> None:
        if self.lanes is None:
            self.lanes = {}

    async def classify(self, request: RoutingRequest) -> Classification | None:
        hints = request.hints or {}

        target = hints.get("target")
        if target:
            return Classification(
                target=Target.parse(target) if isinstance(target, str) else target,
                source=self.name,
                rationale="caller pinned a target",
            )

        for key in self.LANE_KEYS:
            lane = hints.get(key)
            if lane:
                return Classification(
                    lane=str(lane).strip().lower(),
                    effort=self._effort(hints),
                    source=self.name,
                    rationale=f"caller passed {key}={lane!r}",
                )

        for key in self.COMPLEXITY_KEYS:
            value = hints.get(key)
            if not value:
                continue
            text = str(value).strip().lower()
            if text in self.lanes:
                return Classification(
                    lane=text,
                    effort=self._effort(hints),
                    source=self.name,
                    rationale=f"caller passed {key}={text!r}, which names a lane",
                )
            # No lane by that name: it is an effort level, which is the other
            # thing "complexity" commonly means.
            return Classification(
                effort=text,
                source=self.name,
                rationale=f"caller passed {key}={text!r}; no lane of that name, read as effort",
            )

        effort = self._effort(hints)
        if effort:
            return Classification(
                effort=effort, source=self.name, rationale="caller set an effort level"
            )
        return None

    def _effort(self, hints: Mapping[str, Any]) -> str | None:
        for key in self.EFFORT_KEYS:
            value = hints.get(key)
            if value:
                return str(value).strip().lower()
        return None


# ---------------------------------------------------------------------------
# magic_string
# ---------------------------------------------------------------------------

#: ``[[lane:deep]]``, ``[[effort:max]]``. Double brackets because single ones
#: appear in ordinary prose and Markdown far too often to claim.
DEFAULT_MAGIC_PATTERN = r"\[\[(?P<key>lane|route|effort|complexity):(?P<value>[\w.-]+)\]\]"


def compile_magic_pattern(raw: Any) -> re.Pattern[str] | None:
    """Compile a user-supplied magic pattern, or return ``None`` for the default.

    Validates the named groups up front. The classifier reads ``key`` and
    ``value`` from every match, so a pattern without them raises an
    ``IndexError`` from inside ``classify()`` — on a live request, once per
    call, as a warning in a log nobody is reading. Checking here turns that
    into one clear configuration error at startup.
    """
    if not raw:
        return None
    if isinstance(raw, re.Pattern):
        pattern = raw
    else:
        try:
            pattern = re.compile(str(raw), re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"routing.classifier.magic_pattern is not a valid regex: {exc}"
            ) from exc
    missing = [name for name in ("key", "value") if name not in pattern.groupindex]
    if missing:
        raise ValueError(
            f"routing.classifier.magic_pattern must define named group(s) "
            f"{', '.join(missing)}. The default is {DEFAULT_MAGIC_PATTERN!r}, which reads "
            f"'[[lane:deep]]' as key='lane', value='deep'."
        )
    return pattern


def strip_magic_strings(text: str, pattern: re.Pattern[str]) -> str:
    """Remove every magic marker from ``text`` and tidy the seam.

    Separate from the classifier because the *stripping* has to happen whether
    or not the marker was acted on, and in both the prompt and the message
    history. A marker that reaches a provider is a leak of llmcore's internals
    into someone's context window, and in a harness it would be quoted back by
    the model.
    """
    cleaned = pattern.sub("", text)
    # Collapse the double space a removal in mid-sentence leaves behind,
    # without touching intentional formatting elsewhere.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


@dataclass(slots=True)
class MagicStringClassifier:
    """Reads a marker out of the prompt: ``[[lane:deep]]``.

    This is the mechanism by which a *model* can route itself. In an agent
    harness the harness owns the API call and llmcore cannot add a kwarg to
    it — but the model's own text passes through, so a marker in the prompt
    is the one channel that always exists. It is how "let the agent ask for
    the cheap model when the task is trivial" becomes possible without
    touching the harness.

    Its authority is ``prompt``, deliberately below ``caller`` and ``policy``:
    content is not trustworthy the way an argument is. In a RAG or
    tool-output path the text may have come from a retrieved document, and a
    marker there would otherwise be a one-line prompt injection — "route this
    to the most expensive model", or out of a private lane. A caller's
    ``lane=`` therefore always wins over a marker.

    The marker is always stripped before egress, acted on or not.
    """

    name: str = "magic_string"
    cost_hint: str = "free"
    authority: str = "prompt"
    pattern: re.Pattern[str] = None  # type: ignore[assignment]
    lanes: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.pattern is None:
            self.pattern = re.compile(DEFAULT_MAGIC_PATTERN, re.IGNORECASE)
        if self.lanes is None:
            self.lanes = {}

    async def classify(self, request: RoutingRequest) -> Classification | None:
        haystack = self._haystack(request)
        if not haystack:
            return None
        found: dict[str, str] = {}
        for match in self.pattern.finditer(haystack):
            found[match.group("key").lower()] = match.group("value").strip().lower()
        if not found:
            return None

        lane = found.get("lane") or found.get("route")
        complexity = found.get("complexity")
        if lane is None and complexity:
            lane = complexity if complexity in self.lanes else None
        effort = found.get("effort") or (complexity if lane != complexity else None)

        if lane is None and effort is None:
            return None
        return Classification(
            lane=lane,
            effort=effort,
            source=self.name,
            rationale=f"prompt contained a routing marker: {found}",
        )

    @staticmethod
    def _haystack(request: RoutingRequest) -> str:
        parts = [request.prompt or ""]
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeuristicClassifier:
    """Crude signals: length, code fences, and how much is being asked.

    Deliberately simple, and deliberately honest about it. These rules were
    not validated on anyone's traffic and no accuracy is claimed; what they
    are good for is the obvious cases — a prompt with three code blocks and a
    stack trace is not trivial.

    One trap is worth naming, because the obvious heuristic falls straight
    into it: **prompt length does not predict request complexity.** "Write a
    2000-word essay comparing two schools of jurisprudence" is ten tokens and
    is not a cheap request; neither is "tell me about the Azores in autumn".
    So length is only ever used to *veto* the trivial lane, never to select
    it — selecting it needs a recognisable simple-task verb.

    Lane names come from config. Where a configured lane is absent the
    heuristic abstains rather than inventing a destination, so a user who
    named their lanes ``fast``/``smart`` is not quietly routed to a
    non-existent ``trivial``.
    """

    name: str = "heuristic"
    cost_hint: str = "free"
    authority: str = "inferred"
    trivial_lane: str = "trivial"
    standard_lane: str = "standard"
    deep_lane: str = "deep"
    code_lane: str | None = None
    trivial_max_tokens: int = 60
    deep_min_tokens: int = 1_500
    lanes: Mapping[str, Any] = None  # type: ignore[assignment]

    _CODE_FENCE = re.compile(r"```")
    _DEEP_WORDS = re.compile(
        r"\b(prove|derive|refactor|architect|design|optimi[sz]e|debug|why does|trade-?offs?|"
        r"step by step|analy[sz]e)\b",
        re.IGNORECASE,
    )
    _TRIVIAL_WORDS = re.compile(
        r"\b(translate|summari[sz]e|rephrase|capital of|spell|format|list)\b", re.IGNORECASE
    )
    #: Verbs that ask for *output*, which is the thing prompt length cannot
    #: see. "Write a 2000-word essay on X" is ten tokens and not remotely
    #: trivial, so these veto the trivial lane however short the prompt is.
    _GENERATIVE_WORDS = re.compile(
        r"\b(write|draft|compose|essay|implement|build|create|generate|plan|outline|"
        r"story|poem|article|report|paragraph)\b",
        re.IGNORECASE,
    )

    def __post_init__(self) -> None:
        if self.lanes is None:
            self.lanes = {}

    async def classify(self, request: RoutingRequest) -> Classification | None:
        text = "\n".join(
            [request.prompt or ""]
            + [m.get("content", "") for m in request.messages if isinstance(m.get("content"), str)]
        ).strip()
        if not text:
            return None

        tokens = request.approx_tokens
        fences = len(self._CODE_FENCE.findall(text))
        deep_words = bool(self._DEEP_WORDS.search(text))
        trivial_words = bool(self._TRIVIAL_WORDS.search(text))
        generative = bool(self._GENERATIVE_WORDS.search(text))
        has_tools = bool(request.tools)

        if self.code_lane and fences >= 1 and self._available(self.code_lane):
            return self._result(
                self.code_lane, 0.6, f"{fences} code fence(s) in the prompt"
            )

        if tokens >= self.deep_min_tokens or fences >= 2 or (deep_words and tokens > 120):
            if self._available(self.deep_lane):
                reasons = []
                if tokens >= self.deep_min_tokens:
                    reasons.append(f"~{tokens} tokens")
                if fences >= 2:
                    reasons.append(f"{fences} code fences")
                if deep_words:
                    reasons.append("reasoning verbs")
                return self._result(self.deep_lane, 0.6, ", ".join(reasons))

        # Note what is *not* here: a short-prompt rule. Length alone never
        # reaches the trivial lane, because it does not mean what it looks
        # like it means -- "Tell me about the Azores in autumn" is fourteen
        # tokens and is an ordinary request, not a cheap one. A recognisable
        # simple-task verb is required, which makes this rule narrow and
        # right rather than broad and wrong.
        if (
            trivial_words
            and tokens <= self.trivial_max_tokens
            and not has_tools
            and not deep_words
            and not generative
            and self._available(self.trivial_lane)
        ):
            return self._result(
                self.trivial_lane, 0.65, f"~{tokens} tokens and a simple-task verb"
            )

        if self._available(self.standard_lane):
            # Low confidence on purpose: "nothing stood out" is a weak signal,
            # so a confidence floor above this lets a better classifier win.
            return self._result(self.standard_lane, 0.4, "no strong signal either way")
        return None

    def _available(self, lane: str | None) -> bool:
        return bool(lane) and (not self.lanes or lane in self.lanes)

    def _result(self, lane: str, confidence: float, rationale: str) -> Classification:
        return Classification(
            lane=lane, confidence=confidence, source=self.name, rationale=rationale
        )


# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScriptClassifier:
    """Calls a user-supplied function and trusts what it says.

    The escape hatch, and the one classifier that can encode knowledge llmcore
    cannot have: which of *your* customers get the expensive model, which
    internal tool names imply a long task, which repository a prompt mentions.

    The callable may be sync or async. It receives the
    :class:`~llmcore.routing.models.RoutingRequest` and may return a
    :class:`~llmcore.routing.models.Classification`, a lane name, a dict of
    fields, or ``None``. All four shapes are accepted because a one-line
    ``return "deep"`` should not require importing llmcore's types.
    """

    name: str = "script"
    cost_hint: str = "free"
    authority: str = "policy"
    func: Callable[[RoutingRequest], Any] | None = None

    @classmethod
    def from_spec(cls, spec: str) -> ScriptClassifier:
        """Load ``module:function`` or ``module.submodule:function``."""
        if ":" not in spec:
            raise ValueError(
                f"Script classifier spec {spec!r} must be 'module.path:function_name'."
            )
        module_path, _, func_name = spec.partition(":")
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        if not callable(func):
            raise ValueError(f"{spec!r} is not callable.")
        return cls(func=func)

    async def classify(self, request: RoutingRequest) -> Classification | None:
        if self.func is None:
            return None
        result = self.func(request)
        if hasattr(result, "__await__"):
            result = await result
        return self._coerce(result)

    def _coerce(self, result: Any) -> Classification | None:
        if result is None:
            return None
        if isinstance(result, Classification):
            return result if not result.is_empty else None
        if isinstance(result, str):
            return Classification(
                lane=result.strip().lower(), source=self.name, rationale="returned by script"
            )
        if isinstance(result, Mapping):
            target = result.get("target")
            return Classification(
                lane=(str(result["lane"]).strip().lower() if result.get("lane") else None),
                effort=(str(result["effort"]).strip().lower() if result.get("effort") else None),
                target=(
                    Target.parse(target) if isinstance(target, str) else target
                ),
                confidence=result.get("confidence"),
                rationale=result.get("rationale", "returned by script"),
                source=self.name,
            )
        logger.warning(
            "Script classifier returned %r, which is not a Classification, lane name or mapping; "
            "ignoring it.",
            type(result).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _build_hint(*, config: Mapping[str, Any], lanes: Mapping[str, Any]) -> HintClassifier:
    return HintClassifier(lanes=lanes)


def _build_magic(*, config: Mapping[str, Any], lanes: Mapping[str, Any]) -> MagicStringClassifier:
    raw = config.get("magic_pattern") or config.get("pattern")
    return MagicStringClassifier(pattern=compile_magic_pattern(raw), lanes=lanes)


def _build_heuristic(
    *, config: Mapping[str, Any], lanes: Mapping[str, Any]
) -> HeuristicClassifier:
    heuristic = dict(config.get("heuristic") or {})
    return HeuristicClassifier(
        trivial_lane=str(heuristic.get("trivial_lane", "trivial")),
        standard_lane=str(heuristic.get("standard_lane", "standard")),
        deep_lane=str(heuristic.get("deep_lane", "deep")),
        code_lane=(str(heuristic["code_lane"]) if heuristic.get("code_lane") else None),
        trivial_max_tokens=int(heuristic.get("trivial_max_tokens", 60)),
        deep_min_tokens=int(heuristic.get("deep_min_tokens", 1_500)),
        lanes=lanes,
    )


def _build_script(*, config: Mapping[str, Any], lanes: Mapping[str, Any]) -> ScriptClassifier:
    spec = config.get("script") or config.get("script_path")
    if callable(spec):
        return ScriptClassifier(func=spec)
    if not spec:
        raise ValueError(
            'The "script" classifier needs routing.classifier.script = '
            '"my_module:classify_request".'
        )
    return ScriptClassifier.from_spec(str(spec))


register_classifier("hint", _build_hint)
register_classifier("magic_string", _build_magic)
register_classifier("heuristic", _build_heuristic)
register_classifier("script", _build_script)
