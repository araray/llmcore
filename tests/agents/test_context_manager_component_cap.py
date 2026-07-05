# tests/agents/test_context_manager_component_cap.py
"""Per-item token cap for OBSERVATION/TOOL_RESULT components (C-1 layer 3).

The legacy ContextManager hard-maps observations and tool results to
Priority.CRITICAL. These tests cover the pre-insertion cap that truncates
oversized items with an explicit ``[truncated N tokens]`` marker and demotes
them from CRITICAL to HIGH, plus strict back-compat: small items are
untouched and ``max_component_tokens <= 0`` restores the old behavior.
"""

from __future__ import annotations

from llmcore.agents.context import ContextManager, ContextManagerConfig
from llmcore.agents.context.context_manager import Priority


def _make_manager(**overrides) -> ContextManager:
    config = ContextManagerConfig(
        max_tokens=1_000_000,
        reserve_for_output=1000,
        **overrides,
    )
    return ContextManager(config=config)


def _combined_content(context) -> str:
    return "\n\n".join(message.get("content", "") for message in context.messages)


class TestComponentCapConfig:
    """Configuration surface of the per-item cap."""

    def test_default_cap_is_generous_and_enabled(self):
        config = ContextManagerConfig()
        assert config.max_component_tokens == 8000

    def test_negative_cap_normalizes_to_disabled(self):
        config = ContextManagerConfig(max_component_tokens=-5)
        assert config.max_component_tokens == 0


class TestGiantToolResultCapped:
    """A 500K-char tool result is truncated and demoted."""

    def test_giant_tool_result_is_truncated_and_demoted(self):
        manager = _make_manager(max_tool_result_chars=1_000_000)
        formatted = manager._format_tool_result({"tool": "dump", "output": "x" * 500_000})
        assert manager.token_counter.count(formatted) > manager.config.max_component_tokens

        content, priority, truncated = manager._apply_component_cap(formatted, Priority.CRITICAL)

        assert truncated is True
        assert priority == Priority.HIGH
        assert "[truncated " in content
        assert content.endswith(" tokens]")
        # Within the cap plus a small allowance for the marker itself.
        assert manager.token_counter.count(content) <= manager.config.max_component_tokens + 16

    def test_giant_tool_result_keeps_built_context_bounded(self):
        manager = _make_manager(max_tool_result_chars=1_000_000)

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Inspect output.",
            tool_results=[{"tool": "dump", "output": "x" * 500_000}],
        )

        combined = _combined_content(context)
        assert "## Tool Result" in combined
        assert "[truncated " in combined
        # ~125K tokens uncapped; the cap keeps the whole prompt near 8K.
        assert context.total_tokens < 10_000
        assert any("tool result" in warning.lower() for warning in context.warnings)

    def test_demoted_tool_result_can_be_dropped_under_budget_pressure(self):
        manager = ContextManager(
            config=ContextManagerConfig(
                max_tokens=600,
                reserve_for_output=100,
                max_component_tokens=450,
                max_tool_result_chars=1_000_000,
            )
        )

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Inspect output.",
            observations=["observed " * 20],
            tool_results=[{"tool": "dump", "output": "x" * 40_000}],
        )

        # The capped-and-demoted tool result no longer forces its way in.
        assert "tool_result" in context.excluded_components
        assert "observation" in context.included_components

    def test_giant_observations_are_truncated_and_demoted(self):
        manager = _make_manager(max_observations=200, max_component_tokens=1000)
        observations = ["o" * 400 for _ in range(200)]

        obs_block = "## Observations\n" + "\n".join(f"- {o}" for o in observations)
        _content, priority, truncated = manager._apply_component_cap(obs_block, Priority.CRITICAL)
        assert truncated is True
        assert priority == Priority.HIGH

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Observe.",
            observations=observations,
        )
        combined = _combined_content(context)
        assert "## Observations" in combined
        assert "[truncated " in combined
        assert any("observations" in warning.lower() for warning in context.warnings)


class TestBackCompat:
    """Items under the cap and cap-disabled runs behave exactly as before."""

    def test_small_tool_result_is_untouched(self):
        manager = _make_manager()
        result = {"tool": "read_state", "output": {"status": "ok"}}
        expected = manager._format_tool_result(result)

        content, priority, truncated = manager._apply_component_cap(expected, Priority.CRITICAL)
        assert content == expected
        assert priority == Priority.CRITICAL
        assert truncated is False

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Read state.",
            tool_results=[result],
        )
        combined = _combined_content(context)
        assert expected in combined
        assert "[truncated " not in combined
        assert not context.warnings

    def test_small_observations_are_untouched(self):
        manager = _make_manager()

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Observe.",
            observations=["file read ok", "tests green"],
        )

        combined = _combined_content(context)
        assert "## Observations\n- file read ok\n- tests green" in combined
        assert "[truncated " not in combined

    def test_cap_disabled_restores_old_behavior(self):
        manager = _make_manager(max_component_tokens=0, max_tool_result_chars=1_000_000)
        giant = "x" * 500_000

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Inspect output.",
            tool_results=[{"tool": "dump", "output": giant}],
        )

        combined = _combined_content(context)
        assert "[truncated " not in combined
        assert giant in combined
        assert context.total_tokens > 100_000
        assert not context.warnings
