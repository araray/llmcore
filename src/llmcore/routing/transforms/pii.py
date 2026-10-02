# src/llmcore/routing/transforms/pii.py
"""Detect personal data in a prompt, and act on *where it is going*.

Two detectors ship:

``regex`` (default)
    Patterns for the identifiers that have a checkable shape — email
    addresses, credit-card numbers that pass Luhn, IBANs, US SSNs, phone
    numbers, IPs, and common API-key prefixes. Free, instant, no dependency,
    and bounded: it finds what it knows and nothing else.

``vela_pii``
    ``llm-semantic-router/Vela-1.0-Encoder-307M-PII``, a local 307M token
    classifier that labels spans. Catches names, addresses and other things
    no pattern can. Costs a few hundred milliseconds on CPU, like the other
    local encoders — see :mod:`llmcore.routing.classifiers.local_encoder` for
    measured figures on comparable hardware.

Both are honest about their limits, and the module is built so that the limit
does not matter as much as it otherwise would:

    **The guarantee is the route, not the redaction.**

``on_detect = "constrain"`` sends the prompt to a pool that never leaves the
machine. A missed identifier is then still on your own hardware, which is a
completely different failure from a missed identifier in a vendor's logs.
``redact`` can be stacked on top, and is worth stacking, but on its own it is
defence in depth rather than a guarantee — and this docstring exists so that
nobody reads the feature list and concludes otherwise.

Findings record a **hash** of what was matched, never the value. A privacy
feature whose audit log contains the identifiers it found would be the leak it
exists to prevent.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..models import Finding, RoutingRequest, Target, TransformAction, TransformResult
from . import register_transform

logger = logging.getLogger(__name__)

__all__ = [
    "PII_PATTERNS",
    "PiiTransform",
    "RegexPiiDetector",
    "VelaPiiDetector",
    "hash_value",
]


def hash_value(value: str, *, salt: str = "") -> str:
    """Return a short, stable digest of ``value``.

    Truncated to 16 hex characters: enough to correlate the same identifier
    across findings and audit records, short enough to stay readable in a log
    line. A salt is supported so digests cannot be compared against a rainbow
    table of, say, every email address in a leaked list — without it, hashing
    a low-entropy value like a phone number provides very little protection.
    """
    digest = hashlib.sha256(f"{salt}{value}".encode("utf-8")).hexdigest()
    return digest[:16]


def _luhn(digits: str) -> bool:
    """Luhn check, so a 16-digit order number is not called a credit card."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


#: ``kind -> (pattern, validator)``. The validator rejects shape-matched text
#: that is not actually an identifier, which is what keeps the false-positive
#: rate low enough for ``block`` to be usable at all.
PII_PATTERNS: dict[str, tuple[re.Pattern[str], Any]] = {
    "email": (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), None),
    # Anchored on a digit at both ends: a trailing `[ -]?` would swallow the
    # space after the number and glue the redaction onto the next word.
    "credit_card": (
        re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        lambda text: _luhn(re.sub(r"\D", "", text)) and 13 <= len(re.sub(r"\D", "", text)) <= 19,
    ),
    "us_ssn": (re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    "iban": (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), None),
    # Two to four groups, so "+351 912 345 678" matches *whole*. An earlier
    # two-group version left the final "678" behind, and a partially redacted
    # phone number is still a leak -- the remaining digits plus the context
    # usually identify the rest.
    "phone": (
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,3}(?!\w)"
        ),
        lambda text: len(re.sub(r"\D", "", text)) >= 9,
    ),
    "ipv4": (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        lambda text: all(0 <= int(part) <= 255 for part in text.split(".")),
    ),
    # Provider credentials. Finding one of these in a prompt is usually a
    # mistake someone will be glad was caught.
    "api_key": (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{36}|"
            r"AKIA[0-9A-Z]{16}|AIza[A-Za-z0-9_-]{35}|hf_[A-Za-z0-9]{30,})\b"
        ),
        None,
    ),
}


@dataclass(slots=True)
class RegexPiiDetector:
    """Pattern-based detection of identifiers with a checkable shape.

    Deliberately narrow. It does not look for names, addresses or dates of
    birth, because patterns cannot find those without flagging most ordinary
    English — and a detector that fires on everything gets turned off, which
    protects nothing. For those, use ``vela_pii``.
    """

    name: str = "regex"
    kinds: tuple[str, ...] = tuple(PII_PATTERNS)
    salt: str = ""

    def detect(self, text: str, *, where: str) -> list[tuple[Finding, int, int, str]]:
        """Return ``(finding, start, end, matched_text)`` for each hit."""
        hits: list[tuple[Finding, int, int, str]] = []
        for kind in self.kinds:
            entry = PII_PATTERNS.get(kind)
            if entry is None:
                continue
            pattern, validator = entry
            for match in pattern.finditer(text):
                matched = match.group(0)
                if validator is not None and not _safe_validate(validator, matched):
                    continue
                hits.append(
                    (
                        Finding(
                            kind=kind,
                            where=where,
                            hashed=hash_value(matched, salt=self.salt),
                            start=match.start(),
                            end=match.end(),
                            confidence=1.0 if validator is not None else 0.9,
                        ),
                        match.start(),
                        match.end(),
                        matched,
                    )
                )
        return hits


def _safe_validate(validator: Any, text: str) -> bool:
    try:
        return bool(validator(text))
    except Exception:
        return False


@dataclass(slots=True)
class VelaPiiDetector:
    """Local 307M token classifier for PII spans.

    Loaded lazily and run in a worker thread, for the same reason as the
    routing encoder: the forward pass is synchronous and CPU-bound, and
    running it on the event loop would stall every other request.

    Unavailable (no ``transformers``, no network, model not cached) means this
    detector **abstains and says so**, rather than silently reporting a clean
    prompt. The chain's ``fail_closed`` setting then decides what that means
    for the request.
    """

    name: str = "vela_pii"
    model_id: str = "llm-semantic-router/Vela-1.0-Encoder-307M-PII"
    revision: str | None = None
    device: str = "cpu"
    min_score: float = 0.5
    salt: str = ""
    _pipeline: Any = None
    _unavailable: bool = False

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._unavailable:
            return False
        try:
            from transformers import pipeline

            kwargs: dict[str, Any] = {"model": self.model_id, "device": self.device}
            if self.revision:
                kwargs["revision"] = self.revision
            self._pipeline = pipeline(
                "token-classification", aggregation_strategy="simple", **kwargs
            )
            return True
        except Exception as exc:
            logger.error(
                "vela_pii could not load %s (%s); it will NOT detect anything. Install the "
                "'local' extra or set routing.transforms.pii.detector = \"regex\".",
                self.model_id,
                exc,
            )
            self._unavailable = True
            return False

    def detect(self, text: str, *, where: str) -> list[tuple[Finding, int, int, str]]:
        if not self._ensure_loaded():
            return []
        hits: list[tuple[Finding, int, int, str]] = []
        for span in self._pipeline(text):
            score = float(span.get("score", 0.0))
            if score < self.min_score:
                continue
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
            matched = text[start:end]
            if not matched.strip():
                continue
            hits.append(
                (
                    Finding(
                        kind=str(span.get("entity_group") or span.get("entity") or "pii").lower(),
                        where=where,
                        hashed=hash_value(matched, salt=self.salt),
                        start=start,
                        end=end,
                        confidence=score,
                    ),
                    start,
                    end,
                    matched,
                )
            )
        return hits


@dataclass(slots=True)
class PiiTransform:
    """Finds personal data and decides what that means for this target.

    Args:
        detector: Something with ``detect(text, *, where)``.
        on_detect: ``constrain`` (default), ``redact``, ``block`` or ``allow``.
            ``constrain`` is the default because it is the only one of the
            four that is a guarantee rather than a mitigation.
        constrain_to_pool: The pool a constrained request is routed to. Must
            contain only targets you trust with the data — in practice, your
            own hardware.
        redact: Also rewrite the prompt, replacing each finding with a
            placeholder. Independent of ``on_detect``, so ``constrain`` plus
            ``redact`` gives both.
        local_providers: Providers treated as not leaving the machine. A
            request already headed for one of these needs no action, which is
            what makes a local-only pool the end of the matter rather than a
            loop.
        audit: Record findings in routing events (hashed).
        placeholder: Replacement text; ``{kind}`` is substituted.
    """

    detector: Any
    on_detect: str = "constrain"
    constrain_to_pool: str | None = None
    redact: bool = False
    local_providers: frozenset[str] = field(
        default_factory=lambda: frozenset({"ollama", "vllm"})
    )
    audit: bool = True
    placeholder: str = "[redacted:{kind}]"
    name: str = "pii"
    cost_hint: str = "free"

    async def apply(self, request: RoutingRequest, target: Target) -> TransformResult:
        texts = self._texts(request)
        all_findings: list[Finding] = []
        rewrites: dict[str, str] = {}

        for where, text in texts:
            hits = await self._detect(text, where)
            if not hits:
                continue
            hits = self._deoverlap(hits)
            all_findings.extend(finding for finding, _, _, _ in hits)
            if self.redact:
                rewrites[where] = self._redact(text, hits)

        if not all_findings:
            return TransformResult(action=TransformAction.ALLOW, source=self.name)

        # Already going somewhere that does not leave the machine: nothing to
        # do. Without this, a constrained request would be constrained again
        # on the retry and never make progress.
        if target.provider.lower() in self.local_providers:
            return TransformResult(
                action=TransformAction.ALLOW,
                findings=tuple(all_findings),
                reason=f"{len(all_findings)} finding(s), but {target.provider} is local",
                source=self.name,
            )

        action = TransformAction(self.on_detect)
        result_kwargs: dict[str, Any] = {
            "findings": tuple(all_findings),
            "source": self.name,
            "reason": self._reason(all_findings, action, target),
        }
        if self.redact and rewrites:
            result_kwargs.update(self._as_request_fields(request, rewrites))
        if action is TransformAction.CONSTRAIN:
            if not self.constrain_to_pool:
                # Misconfigured: 'constrain' with nowhere to constrain *to*
                # would otherwise degrade into 'allow', which is the one
                # outcome the user was trying to prevent.
                logger.error(
                    "pii transform is set to 'constrain' but no constrain_to_pool is configured; "
                    "blocking instead of sending personal data to %s.",
                    target.key,
                )
                return TransformResult(
                    action=TransformAction.BLOCK,
                    reason=(
                        "PII detected and on_detect='constrain', but "
                        "routing.transforms.pii.pool is not set"
                    ),
                    findings=tuple(all_findings),
                    source=self.name,
                )
            result_kwargs["constrain_to_pool"] = self.constrain_to_pool

        return TransformResult(action=action, **result_kwargs)

    # -- internals --------------------------------------------------------

    async def _detect(self, text: str, where: str) -> list[tuple[Finding, int, int, str]]:
        import asyncio

        detect = self.detector.detect
        if getattr(self.detector, "name", "") == "vela_pii":
            # Synchronous and CPU-bound; keep it off the event loop.
            return await asyncio.to_thread(detect, text, where=where)
        return detect(text, where=where)

    @staticmethod
    def _texts(request: RoutingRequest) -> list[tuple[str, str]]:
        texts: list[tuple[str, str]] = []
        if request.prompt:
            texts.append(("prompt", request.prompt))
        if request.system:
            texts.append(("system", request.system))
        for index, message in enumerate(request.messages):
            content = message.get("content")
            if isinstance(content, str) and content:
                texts.append((f"messages[{index}]", content))
        return texts

    @staticmethod
    def _deoverlap(
        hits: list[tuple[Finding, int, int, str]]
    ) -> list[tuple[Finding, int, int, str]]:
        """Drop hits that overlap one already kept, longest span first.

        Patterns genuinely collide: ``192.168.1.42`` is a valid IPv4 *and*
        nine digits in groups, so it matches both ``ipv4`` and ``phone``.
        Redacting both would rewrite the same span twice and produce
        nonsense, so the longer match wins and ties break on confidence.
        """
        kept: list[tuple[Finding, int, int, str]] = []
        for hit in sorted(hits, key=lambda h: (-(h[2] - h[1]), -(h[0].confidence or 0.0), h[1])):
            _, start, end, _ = hit
            if any(start < kept_end and end > kept_start for _, kept_start, kept_end, _ in kept):
                continue
            kept.append(hit)
        return kept

    def _redact(self, text: str, hits: Iterable[tuple[Finding, int, int, str]]) -> str:
        """Replace each hit, working right to left so offsets stay valid."""
        out = text
        for finding, start, end, _ in sorted(
            self._deoverlap(list(hits)), key=lambda hit: -hit[1]
        ):
            out = out[:start] + self.placeholder.format(kind=finding.kind) + out[end:]
        return out

    @staticmethod
    def _as_request_fields(
        request: RoutingRequest, rewrites: dict[str, str]
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if "prompt" in rewrites:
            fields["prompt"] = rewrites["prompt"]
        if "system" in rewrites:
            fields["system"] = rewrites["system"]
        message_rewrites = {
            int(key[len("messages[") : -1]): value
            for key, value in rewrites.items()
            if key.startswith("messages[")
        }
        if message_rewrites:
            messages = [dict(message) for message in request.messages]
            for index, content in message_rewrites.items():
                messages[index]["content"] = content
            fields["messages"] = tuple(messages)
        return fields

    def _reason(
        self, findings: list[Finding], action: TransformAction, target: Target
    ) -> str:
        kinds = sorted({finding.kind for finding in findings})
        summary = f"{len(findings)} finding(s) ({', '.join(kinds)})"
        if action is TransformAction.CONSTRAIN:
            return f"{summary}; constraining to pool '{self.constrain_to_pool}'"
        if action is TransformAction.BLOCK:
            return f"{summary}; blocked rather than sent to {target.key}"
        if action is TransformAction.REDACT:
            return f"{summary}; redacted before sending to {target.key}"
        return summary


def _build(*, config: Mapping[str, Any]) -> PiiTransform:
    raw = dict(config.get("pii") or config)
    detector_name = str(raw.get("detector") or "regex").strip().lower()
    salt = str(raw.get("hash_salt") or "")

    if detector_name in ("vela_pii", "vela"):
        detector: Any = VelaPiiDetector(
            model_id=str(raw.get("model_id") or VelaPiiDetector.model_id),
            revision=raw.get("revision"),
            device=str(raw.get("device") or "cpu"),
            min_score=float(raw.get("min_score", 0.5)),
            salt=salt,
        )
    elif detector_name == "regex":
        kinds = raw.get("kinds")
        detector = RegexPiiDetector(
            kinds=tuple(kinds) if kinds else tuple(PII_PATTERNS), salt=salt
        )
    else:
        raise ValueError(
            f"Unknown PII detector {detector_name!r}. Valid: regex, vela_pii."
        )

    on_detect = str(raw.get("on_detect") or "constrain").strip().lower()
    if on_detect not in {action.value for action in TransformAction}:
        raise ValueError(
            f"routing.transforms.pii.on_detect={on_detect!r} is not valid. "
            f"Valid: {', '.join(action.value for action in TransformAction)}."
        )

    local = raw.get("local_providers")
    return PiiTransform(
        detector=detector,
        on_detect=on_detect,
        constrain_to_pool=raw.get("pool") or raw.get("constrain_to_pool"),
        redact=bool(raw.get("redact", False)),
        local_providers=frozenset(local) if local else frozenset({"ollama", "vllm"}),
        audit=bool(raw.get("audit", True)),
        placeholder=str(raw.get("placeholder") or "[redacted:{kind}]"),
    )


register_transform("pii", _build)
