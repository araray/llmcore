# src/llmcore/routing/evaluation.py
"""Measuring whether a classifier actually routes your traffic correctly.

Every accuracy claim about a prompt router is worthless without this, which is
why llmcore's documentation makes none. A classifier that is 90% accurate on
someone else's benchmark tells you nothing about your lanes, your prompts and
your models.

What this needs from you is a **labelled set**: your own prompts, each with the
lane it should have gone to. Nothing else substitutes for it. Fifty prompts is
enough to catch an obviously wrong chain; two hundred is enough to compare
classifiers with any confidence.

The one thing this module insists on, which a plain accuracy figure hides:

    **The two error directions are not symmetric.**

Routing too cheap produces a bad answer, which is a user-visible failure.
Routing too expensive only costs money. A classifier at 85% that errs upward
is usable; the same 85% erring downward may not be. So every report separates
them, and that separation is the number worth looking at.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Classification, RoutingRequest

logger = logging.getLogger(__name__)

__all__ = [
    "EvalCase",
    "EvalReport",
    "LaneResult",
    "evaluate",
    "load_cases",
]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled prompt.

    Attributes:
        prompt: What the user would send.
        expected: The lane it should route to. ``None`` means "no opinion is
            acceptable" — useful for prompts you genuinely do not mind about,
            which keeps them in the latency numbers without distorting accuracy.
        system: Optional system message, since it changes what a classifier sees.
        tools: Tool names available on the turn; a classifier may use them.
        note: Free text, echoed in the report for the rows you want to inspect.
    """

    prompt: str
    expected: str | None = None
    system: str | None = None
    tools: tuple[str, ...] = ()
    note: str = ""

    def to_request(self) -> RoutingRequest:
        return RoutingRequest(prompt=self.prompt, system=self.system, tools=self.tools)


@dataclass(slots=True)
class LaneResult:
    """What one classifier did on one case."""

    case: EvalCase
    predicted: str | None
    confidence: float | None
    source: str
    rationale: str | None
    seconds: float

    @property
    def correct(self) -> bool | None:
        """``None`` when the case is unlabelled or the classifier abstained."""
        if self.case.expected is None:
            return None
        if self.predicted is None:
            return None
        return self.predicted == self.case.expected


@dataclass(slots=True)
class EvalReport:
    """Aggregate results for one classifier over one set of cases.

    Attributes:
        name: Which classifier (or chain) was measured.
        results: Per-case outcomes.
        lane_order: Lanes from cheapest to dearest, which is what makes the
            *direction* of an error meaningful. Without it the report can say
            how often routing was wrong but not whether being wrong was
            expensive or dangerous.
    """

    name: str
    results: list[LaneResult] = field(default_factory=list)
    lane_order: tuple[str, ...] = ()

    # -- counts -----------------------------------------------------------

    @property
    def labelled(self) -> list[LaneResult]:
        return [r for r in self.results if r.case.expected is not None]

    @property
    def answered(self) -> list[LaneResult]:
        return [r for r in self.labelled if r.predicted is not None]

    @property
    def abstentions(self) -> int:
        """Cases the classifier declined. Not errors: the chain continues."""
        return len(self.labelled) - len(self.answered)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.answered if r.correct)

    @property
    def accuracy(self) -> float | None:
        """Accuracy over the cases it actually answered.

        Deliberately *not* counting abstentions as wrong. A classifier that
        abstains is passing the turn to the next one in the chain, which is
        the designed behaviour; scoring it as an error would make the most
        honest classifier look like the worst.
        """
        return (self.correct / len(self.answered)) if self.answered else None

    @property
    def coverage(self) -> float | None:
        """Share of labelled cases it had an opinion on."""
        return (len(self.answered) / len(self.labelled)) if self.labelled else None

    # -- the part that matters --------------------------------------------

    def _rank(self, lane: str | None) -> int | None:
        if lane is None or lane not in self.lane_order:
            return None
        return self.lane_order.index(lane)

    @property
    def too_cheap(self) -> list[LaneResult]:
        """Routed below the expected lane. **These produce bad answers.**"""
        out = []
        for r in self.answered:
            got, want = self._rank(r.predicted), self._rank(r.case.expected)
            if got is not None and want is not None and got < want:
                out.append(r)
        return out

    @property
    def too_expensive(self) -> list[LaneResult]:
        """Routed above the expected lane. These only cost money."""
        out = []
        for r in self.answered:
            got, want = self._rank(r.predicted), self._rank(r.case.expected)
            if got is not None and want is not None and got > want:
                out.append(r)
        return out

    @property
    def confusion(self) -> dict[tuple[str, str], int]:
        """``(expected, predicted) -> count``, for the rows worth reading."""
        counts: dict[tuple[str, str], int] = {}
        for r in self.answered:
            key = (r.case.expected or "?", r.predicted or "?")
            counts[key] = counts.get(key, 0) + 1
        return counts

    # -- cost -------------------------------------------------------------

    @property
    def latency_p50(self) -> float:
        times = sorted(r.seconds for r in self.results)
        return statistics.median(times) if times else 0.0

    @property
    def latency_p95(self) -> float:
        times = sorted(r.seconds for r in self.results)
        return times[max(0, int(0.95 * len(times)) - 1)] if times else 0.0

    def summary(self) -> str:
        """A readable block, with the asymmetry made explicit."""
        lines = [f"{self.name}"]
        if not self.labelled:
            lines.append("  no labelled cases — nothing to measure")
            return "\n".join(lines)

        accuracy = self.accuracy
        lines.append(
            f"  answered   {len(self.answered)}/{len(self.labelled)}"
            f" ({(self.coverage or 0) * 100:.0f}% coverage, {self.abstentions} abstentions)"
        )
        lines.append(
            f"  agreement  {self.correct}/{len(self.answered)}"
            + (f" ({accuracy * 100:.0f}%)" if accuracy is not None else "")
        )
        if self.lane_order:
            cheap, dear = len(self.too_cheap), len(self.too_expensive)
            lines.append(
                f"  too cheap  {cheap}  <- these produce bad answers"
            )
            lines.append(
                f"  too dear   {dear}  <- these only cost money"
            )
        lines.append(
            f"  latency    p50 {self.latency_p50 * 1000:.0f} ms, "
            f"p95 {self.latency_p95 * 1000:.0f} ms"
        )
        return "\n".join(lines)

    def misroutes(self, *, direction: str = "cheap", limit: int = 10) -> list[LaneResult]:
        """The individual rows to read, since aggregates do not tell you why."""
        rows = self.too_cheap if direction == "cheap" else self.too_expensive
        return rows[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.name,
            "labelled": len(self.labelled),
            "answered": len(self.answered),
            "abstentions": self.abstentions,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "too_cheap": len(self.too_cheap),
            "too_expensive": len(self.too_expensive),
            "latency_p50_ms": self.latency_p50 * 1000,
            "latency_p95_ms": self.latency_p95 * 1000,
            "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(self.confusion.items())},
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load labelled cases from JSONL, JSON or a plain text file.

    JSONL is the format to use, one object per line::

        {"prompt": "rename this variable", "expected": "trivial"}
        {"prompt": "prove this lemma", "expected": "deep", "note": "maths"}

    A plain ``.txt`` file is accepted too, one prompt per line and no labels —
    useful for measuring *coverage and latency* on real traffic before anyone
    has done the work of labelling it.
    """
    path = Path(path)
    text = path.read_text()

    if path.suffix == ".txt":
        return [EvalCase(prompt=line.strip()) for line in text.splitlines() if line.strip()]

    rows: list[Any]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else loaded.get("cases", [])

    cases: list[EvalCase] = []
    for row in rows:
        if isinstance(row, str):
            cases.append(EvalCase(prompt=row))
            continue
        cases.append(
            EvalCase(
                prompt=str(row["prompt"]),
                expected=row.get("expected") or row.get("lane"),
                system=row.get("system"),
                tools=tuple(row.get("tools") or ()),
                note=str(row.get("note") or ""),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


async def evaluate(
    classifier: Any,
    cases: Sequence[EvalCase],
    *,
    name: str | None = None,
    lane_order: Iterable[str] = (),
) -> EvalReport:
    """Run one classifier (or a whole chain) over *cases*.

    Args:
        classifier: Anything with ``classify(request)``, which includes a
            :class:`~llmcore.routing.classifiers.ClassifierChain`.
        cases: Labelled prompts.
        name: Label for the report.
        lane_order: Lanes cheapest-first. Supply it — without it the report can
            say how often routing was wrong but not whether being wrong was
            expensive or dangerous, and that distinction is the whole point.

    Returns:
        An :class:`EvalReport`.
    """
    report = EvalReport(
        name=name or getattr(classifier, "name", type(classifier).__name__),
        lane_order=tuple(lane_order),
    )
    for case in cases:
        started = time.perf_counter()
        result: Classification | None
        try:
            result = await classifier.classify(case.to_request())
        except Exception as exc:
            logger.warning("Classifier raised on %r: %s", case.prompt[:60], exc)
            result = None
        elapsed = time.perf_counter() - started

        report.results.append(
            LaneResult(
                case=case,
                predicted=result.lane if result else None,
                confidence=result.confidence if result else None,
                source=(result.source if result else ""),
                rationale=(result.rationale if result else None),
                seconds=elapsed,
            )
        )
    return report


async def compare(
    classifiers: Mapping[str, Any],
    cases: Sequence[EvalCase],
    *,
    lane_order: Iterable[str] = (),
) -> dict[str, EvalReport]:
    """Run several classifiers over the same cases.

    The comparison is the useful output: the question is never "is this
    classifier good" but "is it better than the free one that costs nothing".
    """
    order = tuple(lane_order)
    return {
        label: await evaluate(classifier, cases, name=label, lane_order=order)
        for label, classifier in classifiers.items()
    }
