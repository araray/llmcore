# tests/routing/test_lanes_and_classifiers.py
"""Lanes, the classifier chain, and the classifiers that need no network.

The behaviour that matters most here is precedence: who gets to decide, and
in what order. A router that lets a length heuristic overrule an explicit
``lane="private"`` is worse than no router.
"""

from __future__ import annotations

import re

import pytest

from llmcore.routing.classifiers import (
    ClassifierChain,
    available_classifiers,
    build_classifier,
    register_classifier,
)
from llmcore.routing.classifiers.free import (
    DEFAULT_MAGIC_PATTERN,
    HeuristicClassifier,
    HintClassifier,
    MagicStringClassifier,
    ScriptClassifier,
    strip_magic_strings,
)
from llmcore.routing.lanes import Lane, parse_lanes
from llmcore.routing.models import Classification, RoutingRequest, Target
from llmcore.routing.protocols import RequestClassifier, authority_rank, cost_rank

LANES = parse_lanes(
    {
        "trivial": {"pool": "cheap", "description": "Short factual questions and translation"},
        "standard": {"pool": "main", "description": "Ordinary everyday requests"},
        "deep": {"target": "anthropic:claude-opus-5-5", "description": "Multi-step reasoning"},
        "code": {"pool": "code", "description": "Source code"},
        "private": {"pool": "local_only", "description": "Personal or sensitive data"},
    }
)


def req(prompt: str = "hello", **kwargs) -> RoutingRequest:
    return RoutingRequest(prompt=prompt, **kwargs)


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


class TestLanes:
    def test_the_short_form_points_at_a_target(self):
        lane = Lane.from_config("deep", "anthropic:claude-opus-5-5?effort=max")
        assert lane.target.model == "claude-opus-5-5"
        assert lane.pool is None

    def test_the_pool_prefix_disambiguates(self):
        """`main` is a good pool name AND a good provider instance name."""
        assert Lane.from_config("x", "pool:main").pool == "main"
        assert Lane.from_config("y", "main").target.provider == "main"

    def test_the_long_form_carries_params_and_a_description(self):
        lane = Lane.from_config(
            "deep",
            {
                "target": "anthropic:claude-opus-5-5",
                "params": {"effort": "max"},
                "description": "Proofs and architecture review",
            },
        )
        assert lane.params == {"effort": "max"}
        assert lane.description == "Proofs and architecture review"

    def test_exactly_one_destination_is_required(self):
        with pytest.raises(ValueError, match="exactly one destination"):
            Lane.from_config("x", {"pool": "a", "target": "openai:gpt-4o"})
        with pytest.raises(ValueError, match="no destination"):
            Lane.from_config("x", {"description": "nothing to route to"})

    def test_one_malformed_lane_does_not_cost_the_others(self):
        """Routing without a lane still works, so degrading beats refusing to
        start."""
        lanes = parse_lanes({"good": "pool:main", "bad": 42, "alsobad": {"pool": "a", "target": "b"}})
        assert set(lanes) == {"good"}

    def test_destination_round_trips(self):
        for raw in ("pool:cheap", "openai:gpt-4o?effort=low"):
            assert Lane.from_config("x", raw).destination == raw


# ---------------------------------------------------------------------------
# hint
# ---------------------------------------------------------------------------


class TestHintClassifier:
    @pytest.mark.asyncio
    async def test_a_lane_argument_is_taken_at_face_value(self):
        result = await HintClassifier(lanes=LANES).classify(req(hints={"lane": "deep"}))
        assert result.lane == "deep"
        assert result.confidence is None, "an instruction is not a guess"

    @pytest.mark.asyncio
    async def test_a_pinned_target_bypasses_lanes_entirely(self):
        result = await HintClassifier(lanes=LANES).classify(
            req(hints={"target": "xai:grok-4.1?effort=high"})
        )
        assert result.target == Target.parse("xai:grok-4.1?effort=high")

    @pytest.mark.asyncio
    async def test_complexity_maps_to_a_lane_when_one_has_that_name(self):
        """'Focus groups for complexity levels' are just lanes named after
        complexity levels."""
        result = await HintClassifier(lanes=LANES).classify(req(hints={"complexity": "deep"}))
        assert result.lane == "deep"

    @pytest.mark.asyncio
    async def test_complexity_becomes_an_effort_level_otherwise(self):
        result = await HintClassifier(lanes=LANES).classify(req(hints={"complexity": "xhigh"}))
        assert result.lane is None and result.effort == "xhigh"

    @pytest.mark.asyncio
    async def test_no_hint_means_no_opinion(self):
        assert await HintClassifier(lanes=LANES).classify(req()) is None

    @pytest.mark.asyncio
    async def test_a_lane_and_an_effort_travel_together(self):
        result = await HintClassifier(lanes=LANES).classify(
            req(hints={"lane": "deep", "effort": "max"})
        )
        assert (result.lane, result.effort) == ("deep", "max")


# ---------------------------------------------------------------------------
# magic_string
# ---------------------------------------------------------------------------


class TestMagicStrings:
    @pytest.mark.asyncio
    async def test_a_marker_in_the_prompt_routes(self):
        """How a model inside a harness routes itself: its text is the only
        channel that passes through."""
        result = await MagicStringClassifier(lanes=LANES).classify(
            req("[[lane:deep]] Walk me through this proof.")
        )
        assert result.lane == "deep"

    @pytest.mark.asyncio
    async def test_markers_are_found_in_message_history_too(self):
        result = await MagicStringClassifier(lanes=LANES).classify(
            RoutingRequest(messages=({"role": "user", "content": "[[effort:minimal]] hi"},))
        )
        assert result.effort == "minimal"

    @pytest.mark.asyncio
    async def test_no_marker_means_no_opinion(self):
        assert await MagicStringClassifier(lanes=LANES).classify(req("ordinary text")) is None

    @pytest.mark.asyncio
    async def test_single_brackets_are_not_claimed(self):
        """They appear in prose and Markdown far too often."""
        assert (
            await MagicStringClassifier(lanes=LANES).classify(req("see [lane:deep] in the docs"))
            is None
        )

    @pytest.mark.asyncio
    async def test_the_pattern_is_configurable(self):
        classifier = MagicStringClassifier(
            pattern=re.compile(r"<<(?P<key>lane):(?P<value>\w+)>>"), lanes=LANES
        )
        assert (await classifier.classify(req("<<lane:code>> fix this"))).lane == "code"

    def test_stripping_removes_the_marker_and_tidies_the_seam(self):
        pattern = re.compile(DEFAULT_MAGIC_PATTERN, re.IGNORECASE)
        assert strip_magic_strings("[[lane:deep]] Explain this.", pattern) == "Explain this."
        assert strip_magic_strings("Explain [[effort:max]] this.", pattern) == "Explain this."

    def test_stripping_leaves_ordinary_text_alone(self):
        pattern = re.compile(DEFAULT_MAGIC_PATTERN, re.IGNORECASE)
        text = "Compare [a] and [b], then\n\nexplain."
        assert strip_magic_strings(text, pattern) == text


# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------


class TestHeuristic:
    @pytest.mark.asyncio
    async def test_a_simple_task_verb_reaches_the_trivial_lane(self):
        result = await HeuristicClassifier(lanes=LANES).classify(
            req("Translate 'hello' into French.")
        )
        assert result.lane == "trivial"

    @pytest.mark.asyncio
    async def test_a_short_prompt_asking_for_output_is_not_trivial(self):
        """Prompt length does not predict request complexity: this is ten
        tokens and is not a cheap request."""
        result = await HeuristicClassifier(lanes=LANES).classify(
            req("Write a 2000-word essay comparing two schools of jurisprudence")
        )
        assert result is None or result.lane != "trivial"

    @pytest.mark.asyncio
    async def test_reasoning_verbs_in_a_long_prompt_reach_the_deep_lane(self):
        result = await HeuristicClassifier(lanes=LANES).classify(
            req("Please refactor this module and explain the trade-offs. " + "context " * 200)
        )
        assert result.lane == "deep"

    @pytest.mark.asyncio
    async def test_a_very_long_prompt_reaches_the_deep_lane_on_size_alone(self):
        result = await HeuristicClassifier(lanes=LANES).classify(req("word " * 3_000))
        assert result.lane == "deep"

    @pytest.mark.asyncio
    async def test_code_fences_reach_the_code_lane_when_configured(self):
        classifier = HeuristicClassifier(lanes=LANES, code_lane="code")
        result = await classifier.classify(req("```python\nx = 1\n```\nwhy does this fail?"))
        assert result.lane == "code"

    @pytest.mark.asyncio
    async def test_it_abstains_rather_than_naming_a_lane_that_does_not_exist(self):
        """A user who named their lanes fast/smart must not be routed to a
        non-existent 'trivial'."""
        lanes = parse_lanes({"fast": "pool:a", "smart": "pool:b"})
        assert await HeuristicClassifier(lanes=lanes).classify(req("Translate 'hi'.")) is None

    @pytest.mark.asyncio
    async def test_the_no_signal_answer_is_low_confidence(self):
        """So a better classifier can outrank it through the floor."""
        result = await HeuristicClassifier(lanes=LANES).classify(
            req("Tell me about the weather patterns of the Azores in autumn months.")
        )
        assert result.lane == "standard" and result.confidence < 0.55

    @pytest.mark.asyncio
    async def test_tool_use_blocks_the_trivial_lane(self):
        result = await HeuristicClassifier(lanes=LANES).classify(
            req("list them", tools=("search_db",))
        )
        assert result is None or result.lane != "trivial"

    @pytest.mark.asyncio
    async def test_an_empty_request_gets_no_opinion(self):
        assert await HeuristicClassifier(lanes=LANES).classify(RoutingRequest()) is None


# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------


class TestScriptClassifier:
    @pytest.mark.asyncio
    async def test_a_bare_lane_name_is_accepted(self):
        """`return "deep"` should not require importing llmcore's types."""
        result = await ScriptClassifier(func=lambda request: "deep").classify(req())
        assert result.lane == "deep"

    @pytest.mark.asyncio
    async def test_a_dict_is_accepted(self):
        result = await ScriptClassifier(
            func=lambda request: {"lane": "code", "confidence": 0.9, "rationale": "why"}
        ).classify(req())
        assert (result.lane, result.confidence) == ("code", 0.9)

    @pytest.mark.asyncio
    async def test_a_classification_passes_through(self):
        result = await ScriptClassifier(
            func=lambda request: Classification(lane="deep", source="mine")
        ).classify(req())
        assert result.source == "mine"

    @pytest.mark.asyncio
    async def test_an_async_function_works(self):
        async def decide(request):
            return "private"

        assert (await ScriptClassifier(func=decide).classify(req())).lane == "private"

    @pytest.mark.asyncio
    async def test_none_means_no_opinion(self):
        assert await ScriptClassifier(func=lambda request: None).classify(req()) is None

    @pytest.mark.asyncio
    async def test_a_nonsense_return_is_ignored_not_raised(self):
        assert await ScriptClassifier(func=lambda request: 3.14).classify(req()) is None

    def test_a_spec_without_a_colon_says_what_it_wanted(self):
        with pytest.raises(ValueError, match="module.path:function_name"):
            ScriptClassifier.from_spec("my_module.classify")


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, name, result, *, cost_hint="free", authority="inferred", raises=False):
        self.name = name
        self.cost_hint = cost_hint
        self.authority = authority
        self._result = result
        self._raises = raises
        self.calls = 0

    async def classify(self, request):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._result


class TestClassifierChain:
    @pytest.mark.asyncio
    async def test_the_first_opinion_wins_and_the_rest_are_not_run(self):
        first = _Stub("a", Classification(lane="deep", source="a"))
        second = _Stub("b", Classification(lane="trivial", source="b"))
        chain = ClassifierChain([first, second], enforce_cost_order=False)
        assert (await chain.classify(req())).source == "a"
        assert second.calls == 0, "a decided chain must not pay for the rest"

    @pytest.mark.asyncio
    async def test_an_abstention_passes_the_turn_along(self):
        chain = ClassifierChain(
            [_Stub("a", None), _Stub("b", Classification(lane="trivial", source="b"))],
            enforce_cost_order=False,
        )
        assert (await chain.classify(req())).source == "b"

    @pytest.mark.asyncio
    async def test_a_low_confidence_opinion_is_treated_as_an_abstention(self):
        chain = ClassifierChain(
            [
                _Stub("a", Classification(lane="deep", confidence=0.2, source="a")),
                _Stub("b", Classification(lane="trivial", source="b")),
            ],
            min_confidence=0.55,
            enforce_cost_order=False,
        )
        assert (await chain.classify(req())).source == "b"

    @pytest.mark.asyncio
    async def test_an_opinion_without_a_confidence_is_trusted(self):
        """`hint` is certain by construction; demanding a number from it would
        be theatre."""
        chain = ClassifierChain(
            [_Stub("a", Classification(lane="deep", source="a"))], min_confidence=0.99
        )
        assert (await chain.classify(req())).lane == "deep"

    @pytest.mark.asyncio
    async def test_a_classifier_that_raises_is_skipped(self):
        """The user asked a question; 'my prompt router crashed' is never an
        acceptable answer to it."""
        chain = ClassifierChain(
            [
                _Stub("broken", None, raises=True),
                _Stub("b", Classification(lane="trivial", source="b")),
            ],
            enforce_cost_order=False,
        )
        assert (await chain.classify(req())).source == "b"

    @pytest.mark.asyncio
    async def test_an_empty_classification_is_an_abstention(self):
        chain = ClassifierChain([_Stub("a", Classification(source="a"))])
        assert await chain.classify(req()) is None

    @pytest.mark.asyncio
    async def test_no_opinion_anywhere_returns_none(self):
        assert await ClassifierChain([_Stub("a", None)]).classify(req()) is None

    def test_an_empty_chain_is_falsey(self):
        assert not ClassifierChain([])

    def test_the_chain_is_reordered_cheapest_first(self):
        """A classifier that costs an API call to save an API call is only
        worth running once the free signals have abstained."""
        chain = ClassifierChain(
            [
                _Stub("paid", None, cost_hint="api"),
                _Stub("local", None, cost_hint="local"),
                _Stub("free", None, cost_hint="free"),
            ]
        )
        assert chain.names() == ["free", "local", "paid"]

    def test_instructions_are_ordered_before_guesses(self):
        """Both are free, so cost alone cannot separate them -- and running
        the guess first would override what the caller asked for."""
        chain = ClassifierChain(
            [
                _Stub("heuristic", None, authority="inferred"),
                _Stub("magic", None, authority="prompt"),
                _Stub("hint", None, authority="caller"),
            ]
        )
        assert chain.names() == ["hint", "magic", "heuristic"]

    def test_the_configured_order_is_respected_within_a_band(self):
        chain = ClassifierChain(
            [_Stub("second", None, authority="caller"), _Stub("first", None, authority="caller")]
        )
        assert chain.names() == ["second", "first"]

    def test_reordering_can_be_turned_off(self):
        chain = ClassifierChain(
            [_Stub("paid", None, cost_hint="api"), _Stub("free", None, cost_hint="free")],
            enforce_cost_order=False,
        )
        assert chain.names() == ["paid", "free"]

    @pytest.mark.asyncio
    async def test_a_caller_outranks_a_marker_in_retrieved_content(self):
        """The injection case: a marker may have arrived in a retrieved
        document, so it must never overrule the caller."""
        chain = ClassifierChain(
            [c for c in (build_classifier(n, lanes=LANES) for n in ("magic_string", "hint")) if c]
        )
        result = await chain.classify(
            req("Summarise this record. [[lane:deep]]", hints={"lane": "private"})
        )
        assert result.lane == "private" and result.source == "hint"

    @pytest.mark.asyncio
    async def test_a_marker_still_decides_when_the_caller_said_nothing(self):
        chain = ClassifierChain(
            [c for c in (build_classifier(n, lanes=LANES) for n in ("magic_string", "hint")) if c]
        )
        result = await chain.classify(req("Summarise this. [[lane:deep]]"))
        assert result.lane == "deep" and result.source == "magic_string"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_the_builtins_are_registered(self):
        assert {"hint", "magic_string", "heuristic", "script"} <= set(available_classifiers())

    def test_builtins_satisfy_the_protocol(self):
        for name in ("hint", "magic_string", "heuristic"):
            classifier = build_classifier(name, lanes=LANES)
            assert isinstance(classifier, RequestClassifier)
            assert cost_rank(classifier.cost_hint) < 3
            assert authority_rank(classifier.authority) < 4

    def test_an_unknown_name_is_skipped_not_raised(self):
        """A typo in a chain should cost that entry, not the process."""
        assert build_classifier("nonsense", lanes=LANES) is None

    def test_a_classifier_needing_wiring_it_lacks_is_skipped(self):
        assert build_classifier("llm", lanes=LANES) is None
        assert build_classifier("typesafe_jev", lanes=LANES) is None

    def test_script_needs_a_target_and_says_so(self):
        assert build_classifier("script", lanes=LANES) is None

    def test_a_user_classifier_registers_like_a_builtin(self):
        register_classifier("test_custom", lambda **kw: _Stub("test_custom", None), replace=True)
        assert build_classifier("test_custom") is not None

    def test_registration_does_not_silently_shadow_a_builtin(self):
        with pytest.raises(ValueError, match="already registered"):
            register_classifier("hint", lambda **kw: None)
