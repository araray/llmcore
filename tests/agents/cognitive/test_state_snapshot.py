from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.cognitive.models import (
    ActOutput,
    ConfidenceLevel,
    CycleIteration,
    EnhancedAgentState,
    ObserveOutput,
    PerceiveOutput,
    ThinkInput,
    ThinkOutput,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle
from llmcore.agents.single_agent import AgentResult
from llmcore.agents.tools import ToolManager
from llmcore.config.agents_config import AgentsConfig
from llmcore.models import Tool, ToolCall, ToolResult


def _iteration(observation: str = "observed") -> CycleIteration:
    tool_call = ToolCall(id="call-1", name="inspect", arguments={"path": "/tmp/data"})
    iteration = CycleIteration(iteration_number=1)
    iteration.think_output = ThinkOutput(
        thought="Need to inspect the file",
        proposed_action=tool_call,
        reasoning_tokens=17,
    )
    iteration.act_output = ActOutput(
        tool_result=ToolResult(tool_call_id="call-1", content="result" * 40),
        execution_time_ms=12.5,
        success=True,
    )
    iteration.observe_output = ObserveOutput(observation=observation)
    iteration.total_tokens_used = 17
    iteration.mark_completed(success=True)
    return iteration


def test_resume_snapshot_is_json_safe_and_bounded() -> None:
    state = EnhancedAgentState(goal="Summarize files", session_id="session-1", context="ctx")
    state.plan = ["Inspect", "Summarize"]
    state.plan_steps_status = ["completed", "pending"]
    state.current_plan_step_index = 1
    state.progress_estimate = 0.5
    state.overall_confidence = ConfidenceLevel.HIGH
    state.set_working_memory("non_json", MagicMock(name="opaque"))
    state.add_iteration(_iteration("A" * 200))

    snapshot = state.to_resume_snapshot(max_observation_chars=40, max_string_chars=80)

    json.dumps(snapshot)
    assert snapshot["schema_version"] == "llmcore.enhanced_agent_state.v1"
    assert snapshot["current_plan_step"] == "Summarize"
    assert snapshot["metrics"]["iteration_count"] == 1
    assert snapshot["iterations"][0]["observation"]["truncated"] is True
    assert len(snapshot["iterations"][0]["observation"]["content"]) <= 40
    assert isinstance(snapshot["working_memory"]["non_json"], str)


def test_resume_snapshot_rehydrates_core_state() -> None:
    state = EnhancedAgentState(goal="Do work", session_id="session-2")
    state.plan = ["One", "Two"]
    state.plan_steps_status = ["completed", "pending"]
    state.current_plan_step_index = 1
    state.progress_estimate = 0.25
    state.final_answer = "done"
    state.is_finished = True
    state.add_iteration(_iteration())

    restored = EnhancedAgentState.from_resume_snapshot(state.to_resume_snapshot())

    assert restored.goal == "Do work"
    assert restored.session_id == "session-2"
    assert restored.plan == ["One", "Two"]
    assert restored.current_plan_step_index == 1
    assert restored.progress_estimate == 0.25
    assert restored.is_finished is True
    assert restored.metadata["_resume_snapshot_iterations"]


def test_context_compression_cooldown_prevents_thrashing() -> None:
    state = EnhancedAgentState(goal="Compress")
    assert state.should_compress_context(min_iterations_between=2) is True

    state.mark_context_compressed(reason="history_budget", tokens_before=2000, tokens_after=800)

    assert state.should_compress_context(min_iterations_between=2) is False
    state.add_iteration(_iteration())
    assert state.should_compress_context(min_iterations_between=2) is False
    state.add_iteration(_iteration())
    assert state.should_compress_context(min_iterations_between=2) is True


def test_cognitive_cycle_history_is_valid_bounded_json() -> None:
    state = EnhancedAgentState(goal="History")
    state.add_iteration(_iteration("B" * 500))
    cycle = CognitiveCycle(
        provider_manager=MagicMock(),
        memory_manager=MagicMock(),
        storage_manager=MagicMock(),
        tool_manager=MagicMock(),
        max_history_observation_chars=50,
    )

    history = cycle._build_history(state)
    parsed = json.loads(history)

    observation = parsed["recent_iterations"][0]["observation"]
    assert observation["truncated"] is True
    assert len(observation["content"]) <= 50


def test_agent_result_to_dict_includes_state_snapshot() -> None:
    state = EnhancedAgentState(goal="Result", session_id="session-3")
    state.add_iteration(_iteration("observed"))
    result = AgentResult(
        goal="Result",
        final_answer="done",
        success=True,
        iteration_count=0,
        total_tokens=0,
        total_time_seconds=0.1,
        session_id="session-3",
        agent_state=state,
    )

    data = result.to_dict()

    assert data["agent_state_snapshot"]["session_id"] == "session-3"
    assert data["agent_state_snapshot"]["schema_version"] == "llmcore.enhanced_agent_state.v1"
    assert data["iteration_summaries"][0]["action"]["name"] == "inspect"
    assert data["iteration_summaries"][0]["tool_result"]["execution_success"] is True


@pytest.mark.asyncio
async def test_cognitive_cycle_passes_tool_inventory_to_think(monkeypatch) -> None:
    from llmcore.agents.cognitive.phases import cycle as cycle_module

    tool_manager = ToolManager(MagicMock(), MagicMock())
    tool_manager.register_runtime_tool(
        Tool(
            name="inspect_file",
            description="Inspect a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        "tests.runtime.inspect_file",
        lambda: "ok",
    )
    state = EnhancedAgentState(goal="Inspect", session_id="session-4")
    state.plan = ["Inspect file"]
    state.plan_steps_status = ["pending"]
    captured = {}

    async def fake_perceive_phase(**kwargs):
        return PerceiveOutput(retrieved_context=[])

    async def fake_think_phase(**kwargs):
        captured["available_tools"] = kwargs["think_input"].available_tools
        captured["agents_config"] = kwargs["agents_config"]
        return ThinkOutput(
            thought="done",
            is_final_answer=True,
            final_answer="done",
        )

    monkeypatch.setattr(cycle_module, "perceive_phase", fake_perceive_phase)
    monkeypatch.setattr(cycle_module, "think_phase", fake_think_phase)
    agents_config = AgentsConfig()
    cycle = CognitiveCycle(
        provider_manager=MagicMock(),
        memory_manager=MagicMock(),
        storage_manager=MagicMock(),
        tool_manager=tool_manager,
        agents_config=agents_config,
    )

    await cycle.run_iteration(agent_state=state, session_id="session-4")

    assert captured["agents_config"] is agents_config
    assert captured["available_tools"] == [
        {
            "name": "inspect_file",
            "description": "Inspect a file",
            "implementation_key": "tests.runtime.inspect_file",
            "parameter_names": ["path"],
            "required_parameters": ["path"],
        }
    ]
    assert tool_manager.get_tool_definitions()[0].parameters["properties"]["path"]["type"] == "string"


@pytest.mark.asyncio
async def test_think_phase_can_cap_native_provider_tool_schemas() -> None:
    from llmcore.agents.cognitive.phases.think import think_phase

    provider_manager = MagicMock()
    provider = MagicMock()
    provider.default_model = "test-model"
    provider.get_name.return_value = "test-provider"
    provider.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "Thought: done\nFinal Answer: done"}}]}
    )
    provider.extract_response_content.return_value = "Thought: done\nFinal Answer: done"
    provider_manager.get_provider.return_value = provider

    tool_manager = ToolManager(MagicMock(), MagicMock())
    tool_manager.register_runtime_tool(
        Tool(
            name="calculate_sum",
            description="Add numbers together",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        ),
        "tests.runtime.calculate_sum",
        lambda: "0",
    )
    tool_manager.register_runtime_tool(
        Tool(
            name="inspect_file",
            description="Inspect file contents and metadata",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        "tests.runtime.inspect_file.native_cap",
        lambda: "ok",
    )

    agents_config = AgentsConfig()
    agents_config.tool_inventory.max_native_tool_schemas = 1
    state = EnhancedAgentState(goal="Inspect file", session_id="session-5")

    await think_phase(
        agent_state=state,
        think_input=ThinkInput(
            goal="Inspect file contents",
            current_step="Inspect the target file path",
            available_tools=tool_manager.get_tool_inventory(),
        ),
        provider_manager=provider_manager,
        memory_manager=MagicMock(),
        tool_manager=tool_manager,
        agents_config=agents_config,
    )

    native_tools = provider.chat_completion.await_args.kwargs["tools"]
    assert [tool.name for tool in native_tools] == ["inspect_file"]
