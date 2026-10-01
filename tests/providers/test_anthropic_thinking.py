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

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-4-1"])
    def test_effort_none_disables_thinking_on_both_generations(self, model):
        """Returning adaptive-at-low would quietly spend reasoning tokens the
        caller explicitly asked not to spend."""
        thinking, mapped = A._normalize_thinking(None, "none", model)
        assert thinking == {"type": "disabled"}
        assert mapped is None


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
