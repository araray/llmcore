# tests/providers/test_anthropic_thinking.py
"""Extended thinking across Claude generations.

The defect this covers is recorded in `PROVIDER_MODERNIZATION_PLAN.md` and in
the routing spec §6.4: `thinking.budget_tokens` is **rejected with a 400** by
Claude 4.6 and later, and llmcore forwarded a caller's budget unchanged. So a
perfectly reasonable request turned into an API error purely because of which
model served it -- which is exactly the kind of thing a pool makes routine,
since pool members span generations.

The two generations express reasoning depth differently and a caller should
not have to know which: `effort="high"` has to mean the same thing either way.
"""

from __future__ import annotations

import pytest

from llmcore.providers.anthropic_provider import AnthropicProvider as A


class TestGenerationDetection:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-4-1", (4, 1)),
            ("claude-sonnet-4-0", (4, 0)),
            ("claude-haiku-4-5-20251001", (4, 5)),
            ("claude-opus-4-6", (4, 6)),
            ("claude-sonnet-5-5", (5, 5)),
            ("claude-fable-5-1", (5, 1)),
        ],
    )
    def test_generations_parse(self, model, expected):
        assert A._model_generation(model) == expected

    def test_an_unparseable_id_is_unknown(self):
        assert A._model_generation("some-custom-deployment") is None
        assert A._model_generation(None) is None

    @pytest.mark.parametrize(
        ("model", "adaptive"),
        [
            ("claude-opus-4-1", False),
            ("claude-sonnet-4-5", False),
            ("claude-haiku-4-5-20251001", False),
            ("claude-opus-4-6", True),      # the boundary
            ("claude-sonnet-5-5", True),
            ("claude-opus-5-5", True),
        ],
    )
    def test_the_boundary_is_4_6(self, model, adaptive):
        assert A._supports_adaptive_thinking(model) is adaptive

    def test_an_unknown_model_is_assumed_modern(self):
        """The pre-4.6 family is the shrinking set, so defaulting the other
        way would break every new model by default."""
        assert A._supports_adaptive_thinking("claude-something-new") is True


class TestTheBug:
    """budget_tokens sent to a model that rejects it."""

    def test_budget_tokens_is_dropped_for_4_6_and_later(self):
        thinking, effort = A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 8192}, None, "claude-sonnet-5-5"
        )
        assert "budget_tokens" not in thinking
        assert thinking["type"] == "adaptive"
        assert effort is None

    def test_dropping_it_is_logged_loudly(self, caplog):
        A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 8192}, None, "claude-opus-5-5"
        )
        assert "budget_tokens" in caplog.text and "400" in caplog.text

    def test_budget_tokens_is_untouched_where_it_is_accepted(self):
        thinking, _ = A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 8192}, None, "claude-opus-4-1"
        )
        assert thinking == {"type": "enabled", "budget_tokens": 8192}

    def test_adaptive_is_converted_for_a_model_that_cannot_take_it(self):
        thinking, _ = A._normalize_thinking({"type": "adaptive"}, None, "claude-opus-4-1")
        assert thinking["type"] == "enabled" and thinking["budget_tokens"] >= 1024


class TestEffortIsGenerationAgnostic:
    """`effort="high"` must mean the same thing on either generation."""

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),     # folded to the nearest supported rung
            ("max", "max"),
        ],
    )
    def test_modern_models_get_output_config_effort(self, effort, expected):
        thinking, mapped = A._normalize_thinking(None, effort, "claude-opus-5-5")
        assert thinking == {"type": "adaptive"}
        assert mapped == expected

    @pytest.mark.parametrize(
        ("effort", "minimum"),
        [("minimal", 1024), ("low", 2048), ("medium", 8192), ("high", 16384), ("max", 32768)],
    )
    def test_older_models_get_a_token_budget(self, effort, minimum):
        thinking, mapped = A._normalize_thinking(None, effort, "claude-opus-4-1")
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == minimum
        assert mapped is None, "pre-4.6 models have no output_config.effort"

    def test_more_effort_means_more_budget(self):
        budgets = [
            A._normalize_thinking(None, e, "claude-opus-4-1")[0]["budget_tokens"]
            for e in ("minimal", "low", "medium", "high", "xhigh", "max")
        ]
        assert budgets == sorted(budgets)

    def test_every_budget_clears_anthropics_minimum(self):
        for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
            thinking, _ = A._normalize_thinking(None, effort, "claude-opus-4-1")
            assert thinking["budget_tokens"] >= 1024

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # Measured: 4.7+ answers 400 to {"type": "disabled"} and names the
            # replacement itself -- "send thinking: {type: between_tools}".
            ("claude-opus-5-5", "between_tools"),
            ("claude-sonnet-5-5", "between_tools"),
            ("claude-opus-4-7", "between_tools"),
            ("claude-sonnet-4-6", "disabled"),
            ("claude-opus-4-1", "disabled"),
        ],
    )
    def test_effort_none_turns_thinking_off_however_the_model_spells_it(
        self, model, expected
    ):
        """"Do not think" has two spellings, and a caller should not have to
        know which. Returning adaptive-at-low would quietly spend reasoning
        tokens the caller explicitly asked not to spend."""
        thinking, mapped = A._normalize_thinking(None, "none", model)
        assert thinking == {"type": expected}
        assert mapped is None


class TestTheThreeMeasuredBoundaries:
    """They do not coincide, which is why one cutoff was wrong.

    Measured against the live API on 2026-10-01, one request per model:

    ===========  ==============================  ====================  =========
    generation   type=enabled (+budget_tokens)   type=adaptive         disabled
    ===========  ==============================  ====================  =========
    <= 4.5       200 (the only option)           400 not supported     200
    4.6          200                             200                   200
    >= 4.7       400 not supported               200                   400
    ===========  ==============================  ====================  =========
    """

    @pytest.mark.parametrize(
        ("model", "adaptive_ok", "rejects_budget"),
        [
            ("claude-haiku-4-5-20251001", False, False),
            ("claude-sonnet-4-5-20250929", False, False),
            ("claude-sonnet-4-6", True, False),
            ("claude-opus-4-6", True, False),
            ("claude-opus-4-7", True, True),
            ("claude-sonnet-5-5", True, True),
            ("claude-fable-5-1", True, True),
        ],
    )
    def test_the_boundaries_match_what_the_api_does(self, model, adaptive_ok, rejects_budget):
        assert A._supports_adaptive_thinking(model) is adaptive_ok
        assert A._rejects_budget_tokens(model) is rejects_budget

    def test_an_explicit_budget_survives_on_the_4_6_overlap(self):
        """4.6 accepts both forms, so converting a caller's precise budget
        there would be overriding a valid, deliberate choice."""
        thinking, mapped = A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 2048}, None, "claude-sonnet-4-6"
        )
        assert thinking == {"type": "enabled", "budget_tokens": 2048}
        assert mapped is None

    def test_the_same_budget_is_converted_from_4_7(self):
        thinking, _ = A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 2048}, None, "claude-opus-4-7"
        )
        assert thinking == {"type": "adaptive"}

    def test_an_unknown_model_gets_the_newest_behaviour_everywhere(self):
        """The older families are the shrinking set."""
        assert A._supports_adaptive_thinking("claude-brand-new") is True
        assert A._rejects_budget_tokens("claude-brand-new") is True
        assert A._no_thinking_block("claude-brand-new") == {"type": "between_tools"}


class TestBudgetMustFitMaxTokens:
    """Anthropic requires `max_tokens` **strictly greater** than the budget.

    Measured: budget 1024 with max_tokens 1024 is a 400; 1025 succeeds. And a
    budget below 1024 is rejected outright, so for small `max_tokens` there is
    no valid budget at all.
    """

    def test_equal_values_are_not_allowed(self):
        assert A._fit_budget(1024, 1024) is None

    def test_one_more_token_is_enough(self):
        assert A._fit_budget(1024, 1025) == 1024

    def test_a_large_budget_is_clamped_just_under_max_tokens(self):
        assert A._fit_budget(16384, 2048) == 2047

    def test_a_budget_that_already_fits_is_untouched(self):
        assert A._fit_budget(8192, 40000) == 8192

    def test_no_valid_budget_exists_below_the_floor(self):
        for max_tokens in (1, 64, 512, 1024):
            assert A._fit_budget(16384, max_tokens) is None

    def test_an_unknown_max_tokens_is_left_alone(self):
        """llmcore cannot clamp against a limit it was not told."""
        assert A._fit_budget(16384, None) == 16384

    def test_effort_high_with_a_small_max_tokens_disables_rather_than_fails(self):
        """Sending a request that cannot succeed is worse than answering
        without reasoning and saying so."""
        thinking, _ = A._normalize_thinking(
            None, "high", "claude-opus-4-1", max_tokens=512
        )
        assert thinking == {"type": "disabled"}

    def test_effort_high_with_room_is_clamped_not_dropped(self):
        thinking, _ = A._normalize_thinking(
            None, "high", "claude-opus-4-1", max_tokens=2048
        )
        assert thinking == {"type": "enabled", "budget_tokens": 2047}

    def test_a_callers_oversized_budget_is_clamped_too(self):
        thinking, _ = A._normalize_thinking(
            {"type": "enabled", "budget_tokens": 16384},
            None,
            "claude-opus-4-1",
            max_tokens=4096,
        )
        assert thinking == {"type": "enabled", "budget_tokens": 4095}

    def test_a_converted_adaptive_request_also_has_to_fit(self):
        thinking, _ = A._normalize_thinking(
            {"type": "adaptive"}, None, "claude-opus-4-1", max_tokens=600
        )
        assert thinking == {"type": "disabled"}


class TestNoSurprises:
    def test_asking_for_nothing_changes_nothing(self):
        assert A._normalize_thinking(None, None, "claude-sonnet-5-5") == (None, None)

    def test_an_unknown_effort_lands_in_the_middle(self):
        """Rather than raising: a pool member that does not know a vendor's
        private effort word should still answer."""
        _, mapped = A._normalize_thinking(None, "turbo", "claude-opus-5-5")
        assert mapped == "medium"

    def test_the_callers_thinking_dict_is_not_mutated(self):
        original = {"type": "enabled", "budget_tokens": 8192}
        A._normalize_thinking(original, None, "claude-sonnet-5-5")
        assert original == {"type": "enabled", "budget_tokens": 8192}

    def test_effort_is_advertised_as_a_supported_parameter(self):
        provider = A.__new__(A)
        params = provider.get_supported_parameters("claude-opus-5-5")
        assert "effort" in params
        assert "max" in params["effort"]["enum"]
