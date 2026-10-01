# tests/routing/test_evaluation.py
"""The classifier evaluation harness.

llmcore makes no accuracy claim for any classifier. This is the machinery for
replacing that gap with a measurement on real traffic, and the thing it must
get right is **not** the accuracy number -- it is the separation of the two
error directions. Routing too cheap produces a bad answer; routing too
expensive only costs money. A report that collapses those into one percentage
is actively misleading, because the same percentage can mean "usable" or
"unusable" depending on which way it errs.
"""

from __future__ import annotations

import json

import pytest

from llmcore.routing.evaluation import (
    EvalCase,
    compare,
    evaluate,
    load_cases,
)
from llmcore.routing.models import Classification

LANES = ("trivial", "standard", "deep")


class Scripted:
    """A classifier that answers from a lookup, for exact assertions."""

    name = "scripted"
    cost_hint = "free"
    authority = "inferred"

    def __init__(self, answers: dict[str, str | None], *, raises: set[str] = frozenset()):
        self.answers = answers
        self.raises = raises

    async def classify(self, request):
        if request.prompt in self.raises:
            raise RuntimeError("classifier exploded")
        lane = self.answers.get(request.prompt)
        return Classification(lane=lane, source=self.name) if lane else None


def cases(*pairs: tuple[str, str | None]) -> list[EvalCase]:
    return [EvalCase(prompt=p, expected=e) for p, e in pairs]


# ---------------------------------------------------------------------------
# The asymmetry, which is the point
# ---------------------------------------------------------------------------


class TestErrorDirection:
    @pytest.mark.asyncio
    async def test_routing_below_the_expected_lane_is_too_cheap(self):
        report = await evaluate(
            Scripted({"a": "trivial"}), cases(("a", "deep")), lane_order=LANES
        )
        assert len(report.too_cheap) == 1
        assert not report.too_expensive

    @pytest.mark.asyncio
    async def test_routing_above_the_expected_lane_is_too_expensive(self):
        report = await evaluate(
            Scripted({"a": "deep"}), cases(("a", "trivial")), lane_order=LANES
        )
        assert len(report.too_expensive) == 1
        assert not report.too_cheap

    @pytest.mark.asyncio
    async def test_the_same_accuracy_can_mean_opposite_things(self):
        """Two classifiers, both 0% accurate, with completely different
        consequences. This is the whole reason the harness exists."""
        prompts = cases(("a", "standard"), ("b", "standard"))
        cheap = await evaluate(
            Scripted({"a": "trivial", "b": "trivial"}), prompts, lane_order=LANES
        )
        dear = await evaluate(
            Scripted({"a": "deep", "b": "deep"}), prompts, lane_order=LANES
        )
        assert cheap.accuracy == dear.accuracy == 0.0
        assert len(cheap.too_cheap) == 2 and not cheap.too_expensive
        assert len(dear.too_expensive) == 2 and not dear.too_cheap

    @pytest.mark.asyncio
    async def test_without_a_lane_order_direction_cannot_be_judged(self):
        """And the report says nothing rather than guessing."""
        report = await evaluate(Scripted({"a": "trivial"}), cases(("a", "deep")))
        assert not report.too_cheap and not report.too_expensive
        assert report.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_the_summary_names_the_consequence_of_each_direction(self):
        report = await evaluate(
            Scripted({"a": "trivial"}), cases(("a", "deep")), lane_order=LANES
        )
        summary = report.summary()
        assert "produce bad answers" in summary and "only cost money" in summary


# ---------------------------------------------------------------------------
# Abstentions
# ---------------------------------------------------------------------------


class TestAbstentions:
    @pytest.mark.asyncio
    async def test_an_abstention_is_not_scored_as_an_error(self):
        """A classifier that abstains is passing the turn to the next one in
        the chain, which is the designed behaviour. Scoring it as wrong would
        make the most honest classifier look like the worst."""
        report = await evaluate(
            Scripted({"a": "deep"}), cases(("a", "deep"), ("b", "trivial")), lane_order=LANES
        )
        assert report.accuracy == 1.0
        assert report.abstentions == 1
        assert report.coverage == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_coverage_and_accuracy_are_reported_separately(self):
        """A classifier can be accurate and useless, or broad and wrong."""
        report = await evaluate(
            Scripted({"a": "deep"}),
            cases(("a", "deep"), ("b", "deep"), ("c", "deep")),
            lane_order=LANES,
        )
        assert report.accuracy == 1.0 and report.coverage == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_a_classifier_that_raises_counts_as_an_abstention(self):
        """Matching what the chain does at runtime -- a broken classifier
        degrades routing rather than breaking the request."""
        report = await evaluate(
            Scripted({"a": "deep"}, raises={"a"}), cases(("a", "deep")), lane_order=LANES
        )
        assert report.abstentions == 1
        assert report.accuracy is None

    @pytest.mark.asyncio
    async def test_unlabelled_cases_are_excluded_from_accuracy(self):
        """But kept in the latency numbers, so unlabelled traffic is still
        useful for measuring cost before anyone has labelled it."""
        report = await evaluate(
            Scripted({"a": "deep", "b": "trivial"}),
            [EvalCase("a", "deep"), EvalCase("b")],
            lane_order=LANES,
        )
        assert len(report.labelled) == 1 and report.accuracy == 1.0
        assert len(report.results) == 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    @pytest.mark.asyncio
    async def test_the_confusion_matrix_shows_which_lanes_are_confused(self):
        report = await evaluate(
            Scripted({"a": "trivial", "b": "trivial"}),
            cases(("a", "deep"), ("b", "deep")),
            lane_order=LANES,
        )
        assert report.confusion[("deep", "trivial")] == 2

    @pytest.mark.asyncio
    async def test_misroutes_return_the_individual_rows(self):
        """Aggregates do not tell you why; the prompts do."""
        report = await evaluate(
            Scripted({"a": "trivial"}), cases(("a", "deep")), lane_order=LANES
        )
        rows = report.misroutes(direction="cheap")
        assert rows[0].case.prompt == "a" and rows[0].predicted == "trivial"

    @pytest.mark.asyncio
    async def test_latency_is_measured(self):
        report = await evaluate(Scripted({"a": "deep"}), cases(("a", "deep")))
        assert report.latency_p50 >= 0 and report.latency_p95 >= 0

    @pytest.mark.asyncio
    async def test_the_report_serialises(self):
        report = await evaluate(
            Scripted({"a": "trivial"}), cases(("a", "deep")), lane_order=LANES
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["too_cheap"] == 1 and payload["classifier"] == "scripted"

    @pytest.mark.asyncio
    async def test_an_empty_labelled_set_says_so_rather_than_dividing_by_zero(self):
        report = await evaluate(Scripted({}), [EvalCase("a")], lane_order=LANES)
        assert report.accuracy is None
        assert "nothing to measure" in report.summary()


class TestCompare:
    @pytest.mark.asyncio
    async def test_several_classifiers_run_over_the_same_cases(self):
        """The question is never 'is this good' but 'is it better than the
        free one'."""
        prompts = cases(("a", "deep"), ("b", "trivial"))
        reports = await compare(
            {
                "good": Scripted({"a": "deep", "b": "trivial"}),
                "bad": Scripted({"a": "trivial", "b": "trivial"}),
            },
            prompts,
            lane_order=LANES,
        )
        assert reports["good"].accuracy == 1.0
        assert reports["bad"].accuracy == 0.5
        assert len(reports["bad"].too_cheap) == 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_jsonl(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text(
            '{"prompt": "a", "expected": "deep"}\n{"prompt": "b", "lane": "trivial"}\n'
        )
        loaded = load_cases(path)
        assert [c.expected for c in loaded] == ["deep", "trivial"]

    def test_json_array_and_envelope(self, tmp_path):
        array = tmp_path / "a.json"
        array.write_text('[{"prompt": "a", "expected": "deep"}]')
        envelope = tmp_path / "b.json"
        envelope.write_text('{"cases": [{"prompt": "a", "expected": "deep"}]}')
        assert len(load_cases(array)) == len(load_cases(envelope)) == 1

    def test_plain_text_is_unlabelled_traffic(self, tmp_path):
        """For measuring coverage and latency on real prompts before deciding
        whether labelling them is worth it."""
        path = tmp_path / "traffic.txt"
        path.write_text("first prompt\n\nsecond prompt\n")
        loaded = load_cases(path)
        assert len(loaded) == 2 and all(c.expected is None for c in loaded)

    def test_system_and_tools_are_carried(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(
            '{"prompt": "a", "expected": "deep", "system": "be terse", "tools": ["search"]}\n'
        )
        case = load_cases(path)[0]
        assert case.system == "be terse" and case.tools == ("search",)
        assert case.to_request().tools == ("search",)


class TestTheShippedStarterSet:
    """It is referenced from the docs, so it has to stay loadable and sane."""

    @pytest.fixture(scope="class")
    def starter(self):
        import pathlib

        return load_cases(
            pathlib.Path(__file__).parents[2] / "docs/examples/lane_eval_starter.jsonl"
        )

    def test_it_loads(self, starter):
        assert len(starter) >= 25

    def test_every_case_is_labelled(self, starter):
        assert all(case.expected for case in starter)

    def test_it_covers_several_lanes(self, starter):
        assert len({case.expected for case in starter}) >= 4

    def test_it_contains_the_length_trap(self, starter):
        """A short prompt asking for long output, which is the mistake the
        heuristic was written to avoid making."""
        assert any(
            "2000-word" in case.prompt and case.expected != "trivial" for case in starter
        )
