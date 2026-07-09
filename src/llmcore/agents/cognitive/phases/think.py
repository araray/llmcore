# src/llmcore/agents/cognitive/phases/think.py
"""
THINK Phase Implementation.

The THINK phase uses the ReAct (Reasoning + Acting) framework to decide on
the next action. It considers the current goal, plan, history, and available
tools to make an informed decision.

Key Features:
- Uses prompt templates from the prompt library
- Implements ReAct format (Thought → Action)
- Extracts confidence levels
- Supports final answer detection
- Structured output parsing

References:
    - Technical Spec: Section 5.3.3 (THINK Phase)
    - Dossier: Step 2.5 (Cognitive Phases - THINK)
    - ReAct Paper: https://arxiv.org/abs/2210.03629
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from ..models import (
    ConfidenceLevel,
    EnhancedAgentState,
    PlanStepSpec,
    TerminationReason,
    ThinkInput,
    ThinkOutput,
)

if TYPE_CHECKING:
    from ....config.agents_config import AgentsConfig
    from ....memory.manager import MemoryManager
    from ....models import Message
    from ....providers.manager import ProviderManager
    from ...tools import ToolManager
    from ..models import EnhancedAgentState

from ...activities.parser import ActivityRequestParser
from ._prompting import messages_from_registry, record_template_use, require_prompt_registry

logger = logging.getLogger(__name__)


#: Deterministic corrective thought fed back when a finish call carries no
#: usable answer — the model is re-prompted instead of silently looping.
FINISH_WITHOUT_ANSWER_THOUGHT = (
    "finish called without an answer — provide the complete answer in the `answer` argument"
)


# =============================================================================
# THINK PHASE FUNCTION
# =============================================================================


async def think_phase(
    agent_state: EnhancedAgentState,
    think_input: ThinkInput,
    provider_manager: "ProviderManager",
    memory_manager: "MemoryManager",
    tool_manager: "ToolManager",
    prompt_registry: Any | None = None,  # PromptRegistry
    tracer: Any | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
    agents_config: Optional["AgentsConfig"] = None,
) -> ThinkOutput:
    """
    Execute the THINK phase of the cognitive cycle.

    Uses the ReAct framework to:
    1. Load the thinking prompt template
    2. Build context from state and memory
    3. Call the LLM for reasoning
    4. Parse Thought and Action
    5. Determine confidence level
    6. Record metrics

    G3 Enhancement: If native tool calling fails, falls back to activity-based
    execution for models without function calling support.

    Args:
        agent_state: Current enhanced agent state
        think_input: Input configuration for thinking
        provider_manager: Provider manager for LLM calls
        memory_manager: Memory manager for context
        tool_manager: Tool manager for available tools
        prompt_registry: Prompt registry (REQUIRED as of 0.52.0 — raises
            ValueError when None; a render failure aborts the phase)
        tracer: Optional OpenTelemetry tracer
        provider_name: Optional provider override
        model_name: Optional model override
        agents_config: Optional agents configuration (G3)

    Returns:
        ThinkOutput with reasoning and proposed action

    Example:
        >>> think_input = ThinkInput(
        ...     goal="Calculate factorial of 10",
        ...     current_step="Determine calculation approach",
        ...     available_tools=[t.model_dump() for t in tool_manager.get_tool_definitions()]
        ... )
        >>>
        >>> output = await think_phase(
        ...     agent_state=state,
        ...     think_input=think_input,
        ...     provider_manager=provider_manager,
        ...     memory_manager=memory_manager,
        ...     tool_manager=tool_manager
        ... )
        >>>
        >>> if output.proposed_action:
        ...     print(f"Proposed: {output.proposed_action.name}")
    """
    from ....tracing import add_span_attributes, create_span, record_span_exception

    # Load agents config if not provided (G3)
    if agents_config is None:
        from ....config.agents_config import AgentsConfig

        agents_config = AgentsConfig()

    with create_span(tracer, "cognitive.think") as span:
        require_prompt_registry(prompt_registry, "THINK")
        logger.debug("Starting THINK phase")

        convergence = getattr(agents_config, "convergence", None)

        structured_action = _tool_call_from_plan_step(think_input.current_step_spec)
        if structured_action is not None and structured_action.name in _finish_tool_names(
            convergence
        ):
            # A finish-named plan step is a terminal answer, not a tool call.
            answer = _finish_answer_from_arguments(structured_action.arguments)
            if _finish_answer_acceptable(answer, convergence):
                output = ThinkOutput(
                    thought="The current plan step provides the final answer directly.",
                    proposed_action=None,
                    is_final_answer=True,
                    final_answer=answer,
                    final_answer_source="finish_tool",
                    confidence=ConfidenceLevel.HIGH,
                )
                agent_state.is_finished = True
                agent_state.final_answer = answer
                agent_state.termination_reason = TerminationReason.FINISH_TOOL.value
                agent_state.overall_confidence = output.confidence
                if span:
                    add_span_attributes(
                        span,
                        {
                            "think.structured_plan_step": True,
                            "think.is_final": True,
                            "think.final_answer_source": "finish_tool",
                        },
                    )
                return output
            # Empty finish answer in the plan step: fall through to a normal
            # THINK so the model produces a real answer.
            logger.info("Plan-step finish call without an answer — running normal THINK")
            structured_action = None

        if structured_action is not None:
            output = ThinkOutput(
                thought=(
                    "Using the structured tool intent supplied by the current plan step."
                ),
                proposed_action=structured_action,
                confidence=ConfidenceLevel.HIGH,
            )
            agent_state.pending_tool_call = structured_action
            agent_state.overall_confidence = output.confidence
            if span:
                add_span_attributes(
                    span,
                    {
                        "think.structured_plan_step": True,
                        "think.proposed_tool": structured_action.name,
                    },
                )
            return output

        # 1. Render the THINK messages (system + user) from the registry.
        #    Built BEFORE the LLM try/except: a broken template must abort
        #    the phase (fail-loud), never degrade it.
        messages = _generate_thinking_messages(
            think_input=think_input, agent_state=agent_state, prompt_registry=prompt_registry
        )

        try:
            # 2. Build provider-native tool definitions
            tool_definitions = _select_native_tool_definitions(
                tool_manager=tool_manager,
                think_input=think_input,
                agents_config=agents_config,
            )

            # 3. Call LLM
            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            # Convert Tool objects to provider-compatible format
            tools_param = tool_definitions if tool_definitions else None

            # =================================================================
            # G3 Phase 6: Proactive activity check (pre-enabled by capability check)
            # =================================================================
            use_activity_execution = agent_state.get_working_memory("use_activity_execution", False)

            if use_activity_execution:
                logger.info("Proactive activity execution - skipping native tools")
                return await _think_phase_with_activities(
                    agent_state=agent_state,
                    think_input=think_input,
                    provider_manager=provider_manager,
                    provider=provider,
                    target_model=target_model,
                    prompt_registry=prompt_registry,
                    tool_manager=tool_manager,
                    agents_config=agents_config,
                    tracer=tracer,
                    span=span,
                )

            # =================================================================
            # G3 Phase 6: Try native tools first, fall back to activities
            # =================================================================
            use_activity_fallback = False
            response = None
            response_content = ""

            try:
                if callable(getattr(type(provider_manager), "chat_completion_with_retry", None)):
                    response = await provider_manager.chat_completion_with_retry(
                        provider,
                        context=messages,
                        model=target_model,
                        stream=False,
                        tools=tools_param,
                        tracer=tracer,
                        operation="cognitive.think",
                        temperature=0.7,
                    )
                else:
                    response = await provider.chat_completion(
                        context=messages,
                        model=target_model,
                        stream=False,
                        tools=tools_param,
                        temperature=0.7,
                    )
                # Extract response content
                response_content = provider.extract_response_content(response)

            except Exception as tool_error:
                error_msg = str(tool_error).lower()

                # Check if this is a tool support error (G3 Phase 6)
                is_tool_error = any(
                    phrase in error_msg
                    for phrase in [
                        "does not support tools",
                        "does not support function",
                        "tools are not supported",
                        "function calling not supported",
                        "tool_calls",
                        "tool use",
                    ]
                )

                if is_tool_error and agents_config.activities.enabled:
                    logger.info(
                        f"Native tools failed for {target_model}, "
                        f"attempting activity fallback: {tool_error}"
                    )
                    use_activity_fallback = True
                else:
                    # Re-raise if not a tool support issue or activities disabled
                    raise

            # =================================================================
            # G3 Phase 6: Activity Fallback Execution
            # =================================================================
            if use_activity_fallback:
                output = await _think_phase_with_activities(
                    agent_state=agent_state,
                    think_input=think_input,
                    provider_manager=provider_manager,
                    provider=provider,
                    target_model=target_model,
                    prompt_registry=prompt_registry,
                    tool_manager=tool_manager,
                    agents_config=agents_config,
                    tracer=tracer,
                    span=span,
                )
                return output

            # 4. Parse response (normal path)
            output = _parse_think_response(
                response_text=response_content,
                response_dict=response,
                tool_manager=tool_manager,
                convergence=convergence,
            )

            # 5. Update agent state
            if output.proposed_action:
                agent_state.pending_tool_call = output.proposed_action

            if output.is_final_answer:
                agent_state.is_finished = True
                agent_state.final_answer = output.final_answer
                agent_state.termination_reason = (
                    TerminationReason.FINISH_TOOL.value
                    if output.final_answer_source == "finish_tool"
                    else TerminationReason.FINAL_ANSWER_TEXT.value
                )

            agent_state.overall_confidence = output.confidence

            # 6. Record prompt usage metrics (best-effort)
            usage = response.get("usage", {}) if isinstance(response, dict) else None
            total_tokens = usage.get("total_tokens") if usage else None
            record_template_use(
                prompt_registry,
                "thinking_prompt",
                success=output.proposed_action is not None or output.is_final_answer,
                tokens=total_tokens,
            )

            # 7. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "think.has_action": output.proposed_action is not None,
                        "think.is_final": output.is_final_answer,
                        "think.final_answer_source": output.final_answer_source or "",
                        "think.confidence": output.confidence.value,
                        "think.provider": provider.get_name(),
                        "think.model": target_model,
                        "think.activity_fallback": False,
                    },
                )

            logger.info(
                f"THINK phase complete: "
                f"{'final answer' if output.is_final_answer else 'action proposed'}, "
                f"confidence={output.confidence.value}"
            )

            return output

        except Exception as e:
            logger.error(f"THINK phase failed: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

            # Return error output
            return ThinkOutput(
                thought=f"Error in thinking: {e!s}",
                proposed_action=None,
                is_final_answer=False,
                confidence=ConfidenceLevel.LOW,
            )


# =============================================================================
# ACTIVITY FALLBACK (G3 Phase 6)
# =============================================================================


async def _think_phase_with_activities(
    agent_state: EnhancedAgentState,
    think_input: ThinkInput,
    provider_manager: "ProviderManager",
    provider: Any,
    target_model: str,
    prompt_registry: Any,
    tool_manager: "ToolManager",
    agents_config: "AgentsConfig",
    tracer: Any | None,
    span: Any | None,
) -> ThinkOutput:
    """
    Fallback think phase using activity system instead of native tools.

    This prompts the model to output activities in XML format instead of
    using native function calling. Both the activity system prompt and the
    per-iteration execution prompt come from the prompt registry (rendering
    errors propagate — no inline fallback).

    Args:
        agent_state: Current agent state
        think_input: Think phase input
        provider_manager: Provider manager for retry-aware LLM calls
        provider: LLM provider
        target_model: Target model name
        prompt_registry: Prompt registry (grimoire adapter)
        agents_config: Agents configuration
        tracer: Optional tracer
        span: Optional tracing span

    Returns:
        ThinkOutput with activity-based action
    """
    from ....models import Message, Role
    from ....tracing import add_span_attributes
    logger.info("Using activity fallback for think phase")

    # Get built-in activity names plus runtime tools registered for this run.
    available_activities = _activity_protocol_names(tool_manager)

    # Pre-formatted prompt sections; the spell interpolates them verbatim.
    activities_section = ""
    if available_activities:
        activities_section = (
            f"\nAVAILABLE ACTIVITIES: {', '.join(available_activities)}\n"
            "IMPORTANT: You MUST use one of the activities listed above. "
            "Do not invent activity names.\n"
        )
    history_section = f"\n\nRECENT HISTORY:\n{think_input.history}" if think_input.history else ""
    context_section = f"\n\nRELEVANT CONTEXT:\n{think_input.context}" if think_input.context else ""

    system_messages = messages_from_registry(prompt_registry, "activity_system", {})
    user_messages = messages_from_registry(
        prompt_registry,
        "activity_execute",
        {
            "goal": think_input.goal,
            "current_step": think_input.current_step,
            "activities_section": activities_section,
            "history_section": history_section,
            "context_section": context_section,
        },
    )

    # Build messages with activity system prompt
    messages = [
        Message(role=Role.SYSTEM, content=system_messages[0].content),
        *user_messages,
    ]

    # Call LLM without tools
    if callable(getattr(type(provider_manager), "chat_completion_with_retry", None)):
        response = await provider_manager.chat_completion_with_retry(
            provider,
            context=messages,
            model=target_model,
            stream=False,
            tracer=tracer,
            operation="cognitive.think.activity",
            temperature=0.7,
            # No tools parameter - using activity system
        )
    else:
        response = await provider.chat_completion(
            context=messages,
            model=target_model,
            stream=False,
            temperature=0.7,
        )

    response_content = provider.extract_response_content(response)

    # Parse activities from response
    parser = ActivityRequestParser()
    parse_result = parser.parse(response_content)

    # Determine output
    is_final = parser.is_final_answer(response_content)
    final_answer_text = None

    if is_final:
        final_answer_text = parser.extract_final_answer(response_content)
        agent_state.is_finished = True
        agent_state.final_answer = final_answer_text
        agent_state.termination_reason = TerminationReason.FINAL_ANSWER_TEXT.value

    # Runtime tools can use the same XML protocol, then execute through the
    # normal ToolManager ACT path.
    proposed_action = None
    protocol_tool_call = _tool_call_from_activity_request(parse_result.requests, tool_manager)
    if protocol_tool_call is not None and not is_final:
        agent_state.set_working_memory("using_activity_fallback", False)
        agent_state.set_working_memory("pending_activities_text", None)
        agent_state.set_working_memory("parsed_activity_requests", None)
        proposed_action = protocol_tool_call
    else:
        # Store built-in activity state in working memory for act_phase.
        agent_state.set_working_memory("using_activity_fallback", True)
        agent_state.set_working_memory("pending_activities_text", response_content)
        agent_state.set_working_memory("parsed_activity_requests", parse_result.requests)

        # Create a pseudo-ToolCall for the first activity (for compatibility).
        if parse_result.has_requests and not is_final:
            first_activity = parse_result.requests[0]
            from ....models import ToolCall

            proposed_action = ToolCall(
                id=f"activity_{first_activity.activity}",
                name=f"activity:{first_activity.activity}",
                arguments=first_activity.parameters,
            )

    if proposed_action is not None:
        agent_state.pending_tool_call = proposed_action

    # Add tracing
    if span:
        add_span_attributes(
            span,
            {
                "think.activity_fallback": True,
                "think.activities_found": len(parse_result.requests),
                "think.is_final": is_final,
                "think.model": target_model,
            },
        )

    logger.info(
        f"Activity fallback complete: activities={len(parse_result.requests)}, is_final={is_final}"
    )

    return ThinkOutput(
        thought=response_content[:500] if response_content else "Processing via activities...",
        proposed_action=proposed_action,
        is_final_answer=is_final,
        final_answer=final_answer_text,
        final_answer_source="activity" if is_final else None,
        confidence=ConfidenceLevel.MEDIUM,
        using_activity_fallback=True,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _select_native_tool_definitions(
    tool_manager: "ToolManager",
    think_input: ThinkInput,
    agents_config: "AgentsConfig",
) -> list[Any]:
    """Select full native tool schemas for the provider request.

    By default this returns every loaded tool definition to preserve historical
    behavior. When ``agents.tool_inventory.max_native_tool_schemas`` is set, a
    small lexical scorer chooses the tools most relevant to the current goal and
    plan step. Prompt inventory remains separate from provider-native schemas.
    """
    tool_definitions = tool_manager.get_tool_definitions()
    tool_inventory_config = getattr(agents_config, "tool_inventory", None)
    if tool_inventory_config is None or not getattr(tool_inventory_config, "enabled", True):
        return tool_definitions

    max_native_tool_schemas = getattr(tool_inventory_config, "max_native_tool_schemas", None)
    if max_native_tool_schemas is None:
        return tool_definitions

    limit = max(0, int(max_native_tool_schemas))
    if limit >= len(tool_definitions):
        return tool_definitions
    if limit == 0:
        logger.debug("Native provider tool schemas disabled by tool inventory config")
        return []

    ranked = sorted(
        enumerate(tool_definitions),
        key=lambda item: (-_score_tool_definition(item[1], think_input), item[0]),
    )
    selected_names = [name for _, tool in ranked[:limit] if (name := _tool_name(tool))]

    try:
        selected = tool_manager.get_tool_definitions(selected_names)
    except TypeError:
        selected_indexes = {index for index, _ in ranked[:limit]}
        selected = [tool for index, tool in enumerate(tool_definitions) if index in selected_indexes]

    logger.debug(
        "Selected %d/%d native tool schemas for THINK provider call",
        len(selected),
        len(tool_definitions),
    )
    return selected


def _score_tool_definition(tool: Any, think_input: ThinkInput) -> int:
    """Score a tool against the active goal/current step for schema capping."""
    query_text = f"{think_input.goal} {think_input.current_step}".lower()
    query_tokens = _tokenize(query_text)
    name = _tool_name(tool).lower()
    description = _tool_description(tool).lower()
    parameter_names = _tool_parameter_names(tool)
    tool_tokens = _tokenize(f"{name} {description} {' '.join(parameter_names)}")

    score = len(query_tokens & tool_tokens)
    if name and name.replace("_", " ") in query_text:
        score += 5
    for token in query_tokens:
        if token in name:
            score += 2
    if name in {"finish", "final_answer", "respond_to_user", "human_approval"}:
        score += 1
    return score


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9_]+", text.lower()):
        parts = raw_token.split("_")
        for part in [*parts, raw_token]:
            if len(part) >= 3:
                tokens.add(part)
    return tokens


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        function_def = tool.get("function")
        if isinstance(function_def, dict):
            return str(function_def.get("name") or "")
        return str(tool.get("name") or "")
    return str(getattr(tool, "name", "") or "")


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        function_def = tool.get("function")
        if isinstance(function_def, dict):
            return str(function_def.get("description") or "")
        return str(tool.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def _tool_parameter_names(tool: Any) -> list[str]:
    parameters: Any = None
    if isinstance(tool, dict):
        function_def = tool.get("function")
        if isinstance(function_def, dict):
            parameters = function_def.get("parameters")
        else:
            parameters = tool.get("parameters")
        if parameters is None and isinstance(tool.get("parameter_names"), list):
            return [str(name) for name in tool["parameter_names"]]
    else:
        parameters = getattr(tool, "parameters", None)

    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        return []
    return [str(name) for name in properties]


def _activity_protocol_names(tool_manager: "ToolManager") -> list[str]:
    """Return built-in activity names plus loaded ToolManager tool names."""
    names: list[str] = []
    seen: set[str] = set()

    for name in _built_in_activity_names():
        if name not in seen:
            names.append(name)
            seen.add(name)

    for name in _loaded_tool_names(tool_manager):
        if name not in seen:
            names.append(name)
            seen.add(name)

    return names


def _tool_call_from_activity_request(requests: list[Any], tool_manager: "ToolManager"):
    """Convert a fallback XML request into a real ToolCall when it names a loaded tool."""
    loaded_tools = set(_loaded_tool_names(tool_manager))
    if not loaded_tools:
        return None
    built_in_activities = set(_built_in_activity_names())

    for request in requests:
        activity_name = str(getattr(request, "activity", "") or "")
        if activity_name in built_in_activities or activity_name not in loaded_tools:
            continue

        from ....models import ToolCall

        parameters = getattr(request, "parameters", {})
        return ToolCall(
            id=f"activity_tool_{activity_name}",
            name=activity_name,
            arguments=dict(parameters) if isinstance(parameters, dict) else {},
        )

    return None


def _built_in_activity_names() -> list[str]:
    from ...activities.registry import ActivityRegistry

    return ActivityRegistry().list_names()


def _loaded_tool_names(tool_manager: "ToolManager") -> list[str]:
    """Return loaded ToolManager tool names without requiring a concrete implementation."""
    try:
        if callable(getattr(type(tool_manager), "get_tool_inventory", None)):
            inventory = tool_manager.get_tool_inventory()
            names = [str(item.get("name") or "") for item in inventory if isinstance(item, dict)]
            return [name for name in names if name]
    except Exception:
        logger.debug("Unable to read tool inventory for activity protocol", exc_info=True)

    try:
        tool_definitions = tool_manager.get_tool_definitions()
    except Exception:
        logger.debug("Unable to read tool definitions for activity protocol", exc_info=True)
        return []

    return [name for tool in tool_definitions if (name := _tool_name(tool))]


def _tool_call_from_plan_step(step_spec: PlanStepSpec | None):
    """Convert a structured plan step into a direct tool call when possible."""
    if step_spec is None or not step_spec.tool_name:
        return None

    from ....models import ToolCall

    arguments = step_spec.input if isinstance(step_spec.input, dict) else {}
    return ToolCall(
        id=f"plan_step_{step_spec.index}_{step_spec.tool_name}",
        name=step_spec.tool_name,
        arguments=dict(arguments),
    )


def _generate_thinking_messages(
    think_input: ThinkInput, agent_state: EnhancedAgentState, prompt_registry: Any
) -> list["Message"]:
    """
    Render the THINK phase messages (system + user) from the prompt registry.

    Rendering errors propagate — a broken template aborts the phase instead
    of degrading it (0.52.0 control plane, no silent fallback).

    Args:
        think_input: Thinking input configuration
        agent_state: Current agent state
        prompt_registry: Prompt registry (grimoire adapter)

    Returns:
        Role-structured messages for the LLM call
    """
    del agent_state  # Reserved for future state-aware prompt variables.

    return messages_from_registry(
        prompt_registry,
        "thinking_prompt",
        {
            "goal": think_input.goal,
            "current_step": think_input.current_step,
            "history": think_input.history or "No previous actions.",
            "context": think_input.context or "",
            "tools": _format_tools(think_input.available_tools),
            "remaining_steps": "unlimited"
            if think_input.remaining_steps is None
            else str(think_input.remaining_steps),
        },
    )


def _format_tools(tool_definitions: list[dict[str, Any]]) -> str:
    """
    Format tool definitions for prompt.

    Handles multiple tool definition formats:
    - OpenAI format: {"type": "function", "function": {"name": ..., "description": ...}}
    - Direct format: {"name": ..., "description": ..., "parameters": ...}
    - Pydantic Tool.model_dump(): {"name": ..., "description": ..., "parameters": ...}

    Args:
        tool_definitions: List of tool definition dictionaries

    Returns:
        Formatted string listing available tools
    """
    if not tool_definitions:
        return "No tools available."

    lines = []
    for tool in tool_definitions:
        # Handle OpenAI function-calling format (nested under "function" key)
        if "function" in tool and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name", "unknown")
            desc = tool["function"].get("description", "No description")
            params = tool["function"].get("parameters", {})
        # Handle direct/Pydantic format (name/description at top level)
        elif "name" in tool:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            params = tool.get("parameters", {})
        else:
            logger.warning(f"Unknown tool definition format: {list(tool.keys())}")
            continue

        # Build parameter summary if available
        param_summary = ""
        param_names = []
        if params and isinstance(params, dict):
            properties = params.get("properties", {})
            if properties:
                param_names = list(properties.keys())[:5]
        elif isinstance(tool.get("parameter_names"), list):
            param_names = [str(name) for name in tool["parameter_names"][:5]]
        if param_names:
            param_summary = f" (params: {', '.join(param_names)})"

        lines.append(f"- {name}: {desc}{param_summary}")

    return "\n".join(lines) if lines else "No tools available."


def _parse_think_response(
    response_text: str,
    response_dict: dict[str, Any] | None,
    tool_manager: "ToolManager",
    convergence: Any | None = None,
) -> ThinkOutput:
    """
    Parse the LLM response into structured ThinkOutput.

    Args:
        response_text: Extracted text content from the LLM response
        response_dict: Original response dict for token usage extraction
        tool_manager: Tool manager for validation
        convergence: Optional ``ConvergenceConfig`` governing finish-tool
            interception (defaults apply when None)

    Returns:
        Parsed ThinkOutput
    """

    # Initialize output
    thought = ""
    proposed_action = None
    is_final_answer = False
    final_answer = None
    final_answer_source = None
    confidence = ConfidenceLevel.MEDIUM

    # Extract Thought
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract the optional Expected line (2.6) — what a successful result
    # looks like. Native tool-call responses usually lack it; None is fine.
    expected_match = re.search(
        r"Expected:\s*(.+?)(?=\n(?:Thought|Action|Final Answer)|\Z)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    expected_outcome = expected_match.group(1).strip() if expected_match else None
    if not expected_outcome:
        expected_outcome = None

    native_tool_call = _extract_native_tool_call(response_dict)
    if native_tool_call is not None and native_tool_call.name in _finish_tool_names(convergence):
        # Convergence: a native finish call terminates the run — it must
        # never surface as a proposed action for VALIDATE/ACT to churn on.
        answer = _finish_answer_from_arguments(native_tool_call.arguments)
        if _finish_answer_acceptable(answer, convergence):
            is_final_answer = True
            final_answer = answer
            final_answer_source = "finish_tool"
            confidence = ConfidenceLevel.HIGH
        else:
            # Empty/too-short answer: stay non-final with a deterministic
            # corrective thought so the next iteration re-prompts properly.
            thought = FINISH_WITHOUT_ANSWER_THOUGHT
            confidence = ConfidenceLevel.LOW
    elif native_tool_call is not None:
        proposed_action = native_tool_call
        confidence = _determine_confidence(thought, response_text)
    else:
        # Check for Final Answer
        final_answer_match = re.search(
            r"Final Answer:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE
        )

        if final_answer_match:
            is_final_answer = True
            final_answer = final_answer_match.group(1).strip()
            final_answer_source = "text"
            confidence = ConfidenceLevel.HIGH
        else:
            # Extract Action
            action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", response_text, re.IGNORECASE)

            # Stop at the optional Expected line (2.6) so it is never
            # swallowed into the JSON arguments.
            action_input_match = re.search(
                r"Action Input:\s*(.+?)(?=\nExpected:|\Z)",
                response_text,
                re.DOTALL | re.IGNORECASE,
            )

            if action_match:
                action_name = action_match.group(1).strip()
                action_input = ""

                if action_input_match:
                    action_input = action_input_match.group(1).strip()
                    # Try to parse as JSON
                    try:
                        action_args = json.loads(action_input)
                    except json.JSONDecodeError:
                        # Use as plain string
                        action_args = {"input": action_input}
                else:
                    action_args = {}

                # Create ToolCall
                from ....models import ToolCall

                proposed_action = ToolCall(
                    id=f"call_{len(thought)}",  # Simple ID generation
                    name=action_name,
                    arguments=action_args,
                )

            # Determine confidence from thought content
            confidence = _determine_confidence(thought, response_text)

    # Get token count if available from response dict
    reasoning_tokens = None
    if response_dict and isinstance(response_dict, dict):
        usage = response_dict.get("usage", {})
        if usage:
            reasoning_tokens = usage.get("total_tokens")

    return ThinkOutput(
        thought=thought or "Processing next action...",
        proposed_action=proposed_action,
        is_final_answer=is_final_answer,
        final_answer=final_answer,
        final_answer_source=final_answer_source,
        confidence=confidence,
        reasoning_tokens=reasoning_tokens,
        expected_outcome=expected_outcome,
    )


def _finish_tool_names(convergence: Any | None) -> list[str]:
    """Return the configured finish-tool names (defaults without a real config)."""
    names = getattr(convergence, "finish_tool_names", None)
    if isinstance(names, (list, tuple, set)):
        return [str(name) for name in names]
    return ["finish", "final_answer"]


def _finish_answer_from_arguments(arguments: Any) -> str:
    """Extract the final answer from finish-tool arguments.

    Prefers the schema's ``answer`` key, falling back to ``input`` because
    ``_coerce_tool_arguments`` wraps bare-string arguments as ``{"input": ...}``.
    """
    if not isinstance(arguments, dict):
        return "" if arguments is None else str(arguments)
    answer = arguments.get("answer")
    if answer is None:
        answer = arguments.get("input")
    if answer is None:
        return ""
    return answer if isinstance(answer, str) else str(answer)


def _finish_answer_acceptable(answer: str, convergence: Any | None) -> bool:
    """Check a finish answer against the convergence config's minimums."""
    require_nonempty = getattr(convergence, "require_nonempty_answer", True)
    if not isinstance(require_nonempty, bool):
        require_nonempty = True
    if not require_nonempty:
        return True
    try:
        min_chars = max(1, int(getattr(convergence, "min_answer_chars", 1)))
    except (TypeError, ValueError):
        min_chars = 1
    return len(answer.strip()) >= min_chars


def _extract_native_tool_call(response_dict: dict[str, Any] | None) -> Any | None:
    """Extract the first provider-native tool call from a chat response."""
    if not isinstance(response_dict, dict):
        return None

    for index, raw_tool_call in enumerate(_iter_native_tool_calls(response_dict)):
        tool_call = _native_tool_call_to_model(raw_tool_call, index=index)
        if tool_call is not None:
            return tool_call

    return None


def _iter_native_tool_calls(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return native tool-call dictionaries from common provider response shapes."""
    tool_calls: list[dict[str, Any]] = []

    direct_calls = response_dict.get("tool_calls")
    if isinstance(direct_calls, list):
        tool_calls.extend(call for call in direct_calls if isinstance(call, dict))

    message = response_dict.get("message")
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        tool_calls.extend(call for call in message["tool_calls"] if isinstance(call, dict))

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_message = choice.get("message") or choice.get("delta")
            if isinstance(choice_message, dict) and isinstance(
                choice_message.get("tool_calls"), list
            ):
                tool_calls.extend(
                    call for call in choice_message["tool_calls"] if isinstance(call, dict)
                )

    return tool_calls


def _native_tool_call_to_model(raw_tool_call: dict[str, Any], *, index: int) -> Any | None:
    """Convert one provider-native tool-call dict into llmcore's ToolCall model."""
    function = raw_tool_call.get("function")
    if not isinstance(function, dict):
        function = {}

    name = function.get("name") or raw_tool_call.get("name")
    if not name:
        return None

    raw_arguments = function.get("arguments", raw_tool_call.get("arguments", {}))
    arguments = _coerce_tool_arguments(raw_arguments)

    from ....models import ToolCall

    return ToolCall(
        id=str(raw_tool_call.get("id") or f"call_native_{index}"),
        name=str(name),
        arguments=arguments,
    )


def _coerce_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Normalize provider-native tool arguments into a dictionary."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"input": raw_arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"input": parsed}
    return {"input": raw_arguments}


def _determine_confidence(thought: str, full_text: str) -> ConfidenceLevel:
    """
    Determine confidence level from thought content.

    Args:
        thought: The thought text
        full_text: Full response text

    Returns:
        ConfidenceLevel
    """
    # Keywords indicating different confidence levels
    high_confidence_keywords = ["confident", "certain", "sure", "definitely", "clearly"]

    low_confidence_keywords = [
        "uncertain",
        "unsure",
        "maybe",
        "might",
        "possibly",
        "not sure",
        "unclear",
        "confused",
    ]

    text_lower = (thought + " " + full_text).lower()

    # Check for explicit confidence markers
    for keyword in high_confidence_keywords:
        if keyword in text_lower:
            return ConfidenceLevel.HIGH

    for keyword in low_confidence_keywords:
        if keyword in text_lower:
            return ConfidenceLevel.LOW

    # Default to medium
    return ConfidenceLevel.MEDIUM


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["think_phase"]
