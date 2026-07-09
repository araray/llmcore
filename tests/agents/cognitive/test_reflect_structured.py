# tests/agents/cognitive/test_reflect_structured.py
"""REFLECT restructure + expected_outcome tests (plan §2.6).

Structured reflection: providers that whitelist ``response_format`` get the
strict JSON contract (temperature 0.3) with the labeled-text parser as
fallback. Grounding overrides keep self-judgment honest: a failed action can
never complete a step nor raise progress. Reflect gating skips the LLM call
for no-action iterations (``on_action``, default) or successful actions
(``on_failure``). THINK's optional ``Expected:`` line flows through
ObserveInput into ``observe._check_expectation``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmcore.agents.cognitive.models import (
    EnhancedAgentState,
    PerceiveOutput,
    ReflectInput,
    ReflectOutput,
    ThinkOutput,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle
from llmcore.agents.cognitive.phases.reflect import _parse_reflection_json, reflect_phase
from llmcore.agents.cognitive.phases.think import _parse_think_response
from llmcore.config.agents_config import AgentsConfig, ReflectionConfig, ReflectMode
from llmcore.models import ToolCall, ToolResult

CYCLE_MODULE = "llmcore.agents.cognitive.phases.cycle"

CLEAN_JSON = (
    '{"evaluation": "The search worked", "progress": 60, "step_completed": true,'
    ' "plan_needs_update": false, "updated_plan": null,'
    ' "insights": ["rivers counted"], "next_focus": "summarize"}'
)

LABELED_TEXT = (
    "EVALUATION: partial success\n"
    "PROGRESS: 75%\n"
    "INSIGHTS:\n- learned something\n"
    "PLAN_UPDATE: no\n"
    "STEP_COMPLETED: yes\n"
    "NEXT_FOCUS: keep going"
)


def _reflect_input(**overrides) -> ReflectInput:
    defaults: dict = {
        "goal": "Count the rivers",
        "plan": ["search", "summarize"],
        "current_step_index": 0,
        "last_action": ToolCall(id="c1", name="search", arguments={"q": "rivers"}),
        "observation": "found 10 rivers",
        "iteration_number": 1,
    }
    defaults.update(overrides)
    return ReflectInput(**defaults)


def _json_provider(content: str, *, supports_json: bool = True) -> Mock:
    provider = Mock()
    provider.get_name = Mock(return_value="test_provider")
    provider.default_model = "test_model"
    supported = {"temperature": {"type": "number"}}
    if supports_json:
        supported["response_format"] = {"type": "object"}
    provider.get_supported_parameters = Mock(return_value=supported)
    provider.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 12},
        }
    )
    provider.extract_response_content = Mock(return_value=content)
    return provider


def _provider_manager(provider: Mock) -> Mock:
    manager = Mock()
    manager.get_provider = Mock(return_value=provider)
    return manager


# =============================================================================
# Strict JSON parser
# =============================================================================


class TestParseReflectionJson:
    def test_clean_json(self):
        output = _parse_reflection_json(CLEAN_JSON)

        assert output is not None
        assert output.evaluation == "The search worked"
        assert output.progress_estimate == pytest.approx(0.6)
        assert output.step_completed is True
        assert output.plan_needs_update is False
        assert output.updated_plan is None
        assert output.insights == ["rivers counted"]
        assert output.next_focus == "summarize"

    def test_code_fenced_json(self):
        output = _parse_reflection_json(f"```json\n{CLEAN_JSON}\n```")
        assert output is not None
        assert output.progress_estimate == pytest.approx(0.6)

    def test_bare_fence_json(self):
        output = _parse_reflection_json(f"```\n{CLEAN_JSON}\n```")
        assert output is not None

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "EVALUATION: ok\nPROGRESS: 50%",
            "[1, 2, 3]",
            '{"progress": 50}',  # missing evaluation
            '{"evaluation": "ok"}',  # missing progress
            '{"evaluation": "", "progress": 50}',  # empty evaluation
            '{"evaluation": "ok", "progress": "most"}',  # non-numeric progress
        ],
    )
    def test_unparseable_returns_none(self, text):
        assert _parse_reflection_json(text) is None

    def test_progress_clamped_to_bounds(self):
        over = _parse_reflection_json('{"evaluation": "e", "progress": 150}')
        under = _parse_reflection_json('{"evaluation": "e", "progress": -5}')
        assert over is not None and over.progress_estimate == 1.0
        assert under is not None and under.progress_estimate == 0.0

    def test_updated_plan_and_insights_coerced(self):
        output = _parse_reflection_json(
            '{"evaluation": "e", "progress": 10, "plan_needs_update": true,'
            ' "updated_plan": ["step 1", "step 2"], "insights": ["a", 2]}'
        )
        assert output is not None
        assert output.plan_needs_update is True
        assert output.updated_plan == ["step 1", "step 2"]
        assert output.insights == ["a", "2"]


# =============================================================================
# reflect_phase structured-output behavior
# =============================================================================


class TestReflectPhaseStructured:
    @pytest.mark.asyncio
    async def test_json_capable_provider_gets_response_format(
        self, bundled_prompt_registry
    ):
        provider = _json_provider(CLEAN_JSON, supports_json=True)
        state = EnhancedAgentState(goal="g")

        output = await reflect_phase(
            agent_state=state,
            reflect_input=_reflect_input(),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        kwargs = provider.chat_completion.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.3
        assert output.evaluation == "The search worked"
        assert output.progress_estimate == pytest.approx(0.6)
        assert output.tokens_used == 12
        assert state.progress_estimate == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_provider_without_json_mode_keeps_legacy_call(
        self, bundled_prompt_registry
    ):
        provider = _json_provider(LABELED_TEXT, supports_json=False)

        output = await reflect_phase(
            agent_state=EnhancedAgentState(goal="g"),
            reflect_input=_reflect_input(),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        kwargs = provider.chat_completion.call_args.kwargs
        assert "response_format" not in kwargs
        assert kwargs["temperature"] == 0.7
        assert output.progress_estimate == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_use_structured_output_false_disables_json_mode(
        self, bundled_prompt_registry
    ):
        provider = _json_provider(LABELED_TEXT, supports_json=True)
        config = AgentsConfig()
        config.reflection.use_structured_output = False

        await reflect_phase(
            agent_state=EnhancedAgentState(goal="g"),
            reflect_input=_reflect_input(),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=config,
        )

        assert "response_format" not in provider.chat_completion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_labeled_parser(
        self, bundled_prompt_registry
    ):
        provider = _json_provider(LABELED_TEXT, supports_json=True)

        output = await reflect_phase(
            agent_state=EnhancedAgentState(goal="g"),
            reflect_input=_reflect_input(),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        assert output.evaluation == "partial success"
        assert output.progress_estimate == pytest.approx(0.75)
        assert output.step_completed is True

    @pytest.mark.asyncio
    async def test_messages_carry_grounding_variables(self, bundled_prompt_registry):
        provider = _json_provider(CLEAN_JSON)

        await reflect_phase(
            agent_state=EnhancedAgentState(goal="g"),
            reflect_input=_reflect_input(action_success=True, matches_expectation=False),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        messages = provider.chat_completion.call_args.kwargs["context"]
        rendered = "\n".join(m.content for m in messages)
        assert "ACTION SUCCEEDED: true" in rendered
        assert "MATCHED EXPECTATION: false" in rendered


# =============================================================================
# Grounding overrides
# =============================================================================


class TestGroundingOverrides:
    @pytest.mark.asyncio
    async def test_failed_action_forces_incomplete_and_clamps_progress(
        self, bundled_prompt_registry
    ):
        # The model claims 90% + step done; the action FAILED.
        claim = '{"evaluation": "went great", "progress": 90, "step_completed": true}'
        provider = _json_provider(claim)
        state = EnhancedAgentState(goal="g")
        state.progress_estimate = 0.3

        output = await reflect_phase(
            agent_state=state,
            reflect_input=_reflect_input(action_success=False),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        assert output.step_completed is False
        assert output.progress_estimate == pytest.approx(0.3)
        assert state.progress_estimate == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_grounding_disabled_keeps_model_claims(self, bundled_prompt_registry):
        claim = '{"evaluation": "went great", "progress": 90, "step_completed": true}'
        provider = _json_provider(claim)
        state = EnhancedAgentState(goal="g")
        state.progress_estimate = 0.3
        config = AgentsConfig()
        config.reflection.ground_in_observations = False

        output = await reflect_phase(
            agent_state=state,
            reflect_input=_reflect_input(action_success=False),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=config,
        )

        assert output.step_completed is True
        assert output.progress_estimate == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_final_answer_sets_progress_complete(self, bundled_prompt_registry):
        claim = '{"evaluation": "slow going", "progress": 10, "step_completed": false}'
        provider = _json_provider(claim)
        state = EnhancedAgentState(goal="g")
        state.is_finished = True
        state.final_answer = "42"

        output = await reflect_phase(
            agent_state=state,
            reflect_input=_reflect_input(action_success=True),
            provider_manager=_provider_manager(provider),
            prompt_registry=bundled_prompt_registry,
            agents_config=AgentsConfig(),
        )

        assert output.progress_estimate == 1.0
        assert state.progress_estimate == 1.0


# =============================================================================
# Reflect gating (cycle-level)
# =============================================================================


def _tool_manager(result: ToolResult | None = None) -> Mock:
    manager = Mock()
    manager.is_tool_loaded = Mock(return_value=True)
    manager.get_tool_names = Mock(return_value=["calculator"])
    manager.get_tool_definitions = Mock(side_effect=lambda names=None: [])
    manager.execute_tool = AsyncMock(
        return_value=result or ToolResult(tool_call_id="c1", content="4", is_error=False)
    )
    return manager


def _reflect_mode_config(mode: ReflectMode) -> AgentsConfig:
    config = AgentsConfig()
    config.circuit_breaker.enabled = False
    config.reflection.mode = mode
    return config


class TestReflectGating:
    async def _run_iteration(
        self,
        bundled_prompt_registry,
        config: AgentsConfig,
        think_output: ThinkOutput,
        tool_result: ToolResult | None = None,
    ):
        tool_manager = _tool_manager(tool_result)
        provider_manager = Mock()
        cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=Mock(),
            storage_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
            agents_config=config,
        )
        state = EnhancedAgentState(goal="Test goal", session_id="s")
        state.progress_estimate = 0.4

        reflect_mock = AsyncMock(
            return_value=ReflectOutput(evaluation="llm reflection", progress_estimate=0.6)
        )
        with (
            patch(
                f"{CYCLE_MODULE}.perceive_phase",
                AsyncMock(return_value=PerceiveOutput(retrieved_context=[])),
            ),
            # "Test goal" is heuristically simple, so complex_only gating
            # already skips PLAN — the patch just hardens against future
            # planning-mode defaults.
            patch(f"{CYCLE_MODULE}.plan_phase", AsyncMock()),
            patch(f"{CYCLE_MODULE}.think_phase", AsyncMock(return_value=think_output)),
            patch(f"{CYCLE_MODULE}.reflect_phase", reflect_mock),
        ):
            iteration = await cycle.run_iteration(
                agent_state=state, session_id="s", skip_validation=True
            )
        return iteration, state, reflect_mock, provider_manager

    @pytest.mark.asyncio
    async def test_on_action_skips_llm_for_no_action_iterations(
        self, bundled_prompt_registry
    ):
        iteration, state, reflect_mock, provider_manager = await self._run_iteration(
            bundled_prompt_registry,
            _reflect_mode_config(ReflectMode.ON_ACTION),
            ThinkOutput(thought="pondering", proposed_action=None),
        )

        reflect_mock.assert_not_called()
        provider_manager.get_provider.assert_not_called()
        assert iteration.reflect_output is not None
        assert iteration.reflect_output.evaluation == "(reflection skipped: no action)"
        assert iteration.reflect_output.step_completed is False
        assert iteration.reflect_output.plan_needs_update is False
        # Progress unchanged.
        assert iteration.reflect_output.progress_estimate == pytest.approx(0.4)
        assert state.progress_estimate == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_on_action_reflects_when_action_proposed(self, bundled_prompt_registry):
        iteration, _, reflect_mock, _ = await self._run_iteration(
            bundled_prompt_registry,
            _reflect_mode_config(ReflectMode.ON_ACTION),
            ThinkOutput(
                thought="use tool",
                proposed_action=ToolCall(
                    id="c1", name="calculator", arguments={"expression": "2+2"}
                ),
            ),
        )

        reflect_mock.assert_called_once()
        reflect_input = reflect_mock.call_args.kwargs["reflect_input"]
        assert reflect_input.action_success is True
        assert reflect_input.follow_up_needed is False
        assert iteration.reflect_output.evaluation == "llm reflection"

    @pytest.mark.asyncio
    async def test_on_failure_skips_llm_for_successful_action(
        self, bundled_prompt_registry
    ):
        iteration, _, reflect_mock, _ = await self._run_iteration(
            bundled_prompt_registry,
            _reflect_mode_config(ReflectMode.ON_FAILURE),
            ThinkOutput(
                thought="use tool",
                proposed_action=ToolCall(
                    id="c1", name="calculator", arguments={"expression": "2+2"}
                ),
            ),
        )

        reflect_mock.assert_not_called()
        assert iteration.reflect_output.evaluation == (
            "(reflection skipped: action succeeded)"
        )
        # External signals: successful action with no follow-up = step done.
        assert iteration.reflect_output.step_completed is True

    @pytest.mark.asyncio
    async def test_on_failure_reflects_on_failed_action(self, bundled_prompt_registry):
        _, _, reflect_mock, _ = await self._run_iteration(
            bundled_prompt_registry,
            _reflect_mode_config(ReflectMode.ON_FAILURE),
            ThinkOutput(
                thought="use tool",
                proposed_action=ToolCall(
                    id="c1", name="calculator", arguments={"expression": "1/0"}
                ),
            ),
            tool_result=ToolResult(tool_call_id="c1", content="boom", is_error=True),
        )

        reflect_mock.assert_called_once()
        reflect_input = reflect_mock.call_args.kwargs["reflect_input"]
        assert reflect_input.action_success is False
        assert reflect_input.follow_up_needed is True

    @pytest.mark.asyncio
    async def test_always_mode_reflects_without_action(self, bundled_prompt_registry):
        _, _, reflect_mock, _ = await self._run_iteration(
            bundled_prompt_registry,
            _reflect_mode_config(ReflectMode.ALWAYS),
            ThinkOutput(thought="pondering", proposed_action=None),
        )

        reflect_mock.assert_called_once()


# =============================================================================
# expected_outcome end-to-end
# =============================================================================


REACT_WITH_EXPECTED = (
    "Thought: I should calculate the sum\n"
    "Action: calculator\n"
    'Action Input: {"expression": "2+2"}\n'
    "Expected: The result should be 4"
)


class TestExpectedOutcome:
    def test_parse_expected_line(self):
        output = _parse_think_response(
            response_text=REACT_WITH_EXPECTED, response_dict=None, tool_manager=object()
        )

        assert output.expected_outcome == "The result should be 4"
        assert output.proposed_action is not None
        assert output.proposed_action.name == "calculator"
        # Regression: the Expected line must NOT be swallowed into the
        # Action Input JSON.
        assert output.proposed_action.arguments == {"expression": "2+2"}

    def test_expected_absent_is_none(self):
        output = _parse_think_response(
            response_text=(
                "Thought: calculate\nAction: calculator\n"
                'Action Input: {"expression": "2+2"}'
            ),
            response_dict=None,
            tool_manager=object(),
        )

        assert output.expected_outcome is None
        assert output.proposed_action.arguments == {"expression": "2+2"}

    async def _run_end_to_end(self, bundled_prompt_registry, tool_content: str):
        provider = Mock()
        provider.get_name = Mock(return_value="test_provider")
        provider.default_model = "test_model"
        provider.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": REACT_WITH_EXPECTED}}],
                "usage": {"total_tokens": 20},
            }
        )
        provider.extract_response_content = Mock(return_value=REACT_WITH_EXPECTED)

        provider_manager = Mock()
        provider_manager.get_provider = Mock(return_value=provider)
        tool_manager = _tool_manager(
            ToolResult(tool_call_id="c1", content=tool_content, is_error=False)
        )
        config = AgentsConfig()
        config.circuit_breaker.enabled = False
        cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=Mock(),
            storage_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
            agents_config=config,
        )
        state = EnhancedAgentState(goal="What is 2+2?", session_id="s")

        with (
            patch(
                f"{CYCLE_MODULE}.perceive_phase",
                AsyncMock(return_value=PerceiveOutput(retrieved_context=[])),
            ),
            # "What is 2+2?" classifies simple → PLAN gated off; the patch
            # hardens against future planning-mode defaults.
            patch(f"{CYCLE_MODULE}.plan_phase", AsyncMock()),
            patch(
                f"{CYCLE_MODULE}.reflect_phase",
                AsyncMock(
                    return_value=ReflectOutput(evaluation="ok", progress_estimate=0.5)
                ),
            ),
        ):
            iteration = await cycle.run_iteration(
                agent_state=state, session_id="s", skip_validation=True
            )
        return iteration

    @pytest.mark.asyncio
    async def test_expected_outcome_flows_into_observe_match(
        self, bundled_prompt_registry
    ):
        iteration = await self._run_end_to_end(bundled_prompt_registry, tool_content="4")

        assert iteration.think_output.expected_outcome == "The result should be 4"
        assert iteration.observe_output.matches_expectation is True
        assert "Expected: The result should be 4" in iteration.observe_output.observation

    @pytest.mark.asyncio
    async def test_unmet_expectation_detected(self, bundled_prompt_registry):
        iteration = await self._run_end_to_end(bundled_prompt_registry, tool_content="5")

        assert iteration.observe_output.matches_expectation is False
        assert iteration.observe_output.follow_up_needed is True


# =============================================================================
# Config + model surface
# =============================================================================


class TestReflectionConfigSurface:
    def test_defaults(self):
        config = AgentsConfig()
        assert isinstance(config.reflection, ReflectionConfig)
        assert config.reflection.mode is ReflectMode.ON_ACTION
        assert config.reflection.use_structured_output is True
        assert config.reflection.ground_in_observations is True

    def test_reflect_mode_members(self):
        assert {m.value for m in ReflectMode} == {"always", "on_action", "on_failure"}

    def test_model_fields_default_none(self):
        assert ThinkOutput(thought="t").expected_outcome is None
        reflect_input = _reflect_input()
        assert reflect_input.action_success is None
        assert reflect_input.matches_expectation is None
        assert reflect_input.follow_up_needed is None
