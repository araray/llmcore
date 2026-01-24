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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..models import ConfidenceLevel, EnhancedAgentState, ThinkInput, ThinkOutput

if TYPE_CHECKING:
    from ....config.agents_config import AgentsConfig
    from ....memory.manager import MemoryManager
    from ....models import Message, Role, ToolCall
    from ....providers.manager import ProviderManager
    from ...tools import ToolManager
    from ..models import EnhancedAgentState

from ...activities.parser import ActivityRequestParser
from ...activities.prompts import ACTIVITY_SYSTEM_PROMPT, generate_activity_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# THINK PHASE FUNCTION
# =============================================================================


async def think_phase(
    agent_state: EnhancedAgentState,
    think_input: ThinkInput,
    provider_manager: "ProviderManager",
    memory_manager: "MemoryManager",
    tool_manager: "ToolManager",
    prompt_registry: Optional[Any] = None,  # PromptRegistry
    tracer: Optional[Any] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
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
        prompt_registry: Optional prompt registry
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
    from ....models import Message, Role
    from ....tracing import add_span_attributes, create_span, record_span_exception

    # Load agents config if not provided (G3)
    if agents_config is None:
        from ....config.agents_config import AgentsConfig

        agents_config = AgentsConfig()

    with create_span(tracer, "cognitive.think") as span:
        try:
            logger.debug("Starting THINK phase")

            # 1. Generate thinking prompt
            thinking_prompt = _generate_thinking_prompt(
                think_input=think_input, agent_state=agent_state, prompt_registry=prompt_registry
            )

            # 2. Build tool definitions
            tool_definitions = tool_manager.get_tool_definitions()

            # 3. Call LLM
            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are an autonomous AI agent using the ReAct framework. "
                    "Think step-by-step and use tools effectively.",
                ),
                Message(role=Role.USER, content=thinking_prompt),
            ]

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
                    provider=provider,
                    target_model=target_model,
                    prompt_registry=prompt_registry,
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
                    provider=provider,
                    target_model=target_model,
                    prompt_registry=prompt_registry,
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
            )

            # 5. Update agent state
            if output.proposed_action:
                agent_state.pending_tool_call = output.proposed_action

            if output.is_final_answer:
                agent_state.is_finished = True
                agent_state.final_answer = output.final_answer

            agent_state.overall_confidence = output.confidence

            # 6. Record metrics
            if prompt_registry and hasattr(prompt_registry, "record_use"):
                try:
                    template = prompt_registry.get_template("thinking_prompt")
                    if template.active_version:
                        # Extract token usage from response dict
                        usage = response.get("usage", {}) if isinstance(response, dict) else None
                        total_tokens = usage.get("total_tokens") if usage else None
                        prompt_registry.record_use(
                            version_id=template.active_version.id,
                            success=output.proposed_action is not None or output.is_final_answer,
                            tokens=total_tokens,
                        )
                except Exception as e:
                    logger.warning(f"Failed to record prompt metrics: {e}")

            # 7. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "think.has_action": output.proposed_action is not None,
                        "think.is_final": output.is_final_answer,
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
                thought=f"Error in thinking: {str(e)}",
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
    provider: Any,
    target_model: str,
    prompt_registry: Optional[Any],
    agents_config: "AgentsConfig",
    tracer: Optional[Any],
    span: Optional[Any],
) -> ThinkOutput:
    """
    Fallback think phase using activity system instead of native tools.

    This prompts the model to output activities in XML format instead of
    using native function calling.

    Args:
        agent_state: Current agent state
        think_input: Think phase input
        provider: LLM provider
        target_model: Target model name
        prompt_registry: Optional prompt registry
        agents_config: Agents configuration
        tracer: Optional tracer
        span: Optional tracing span

    Returns:
        ThinkOutput with activity-based action
    """
    from ....models import Message, Role
    from ....tracing import add_span_attributes
    from ...activities.registry import ActivityRegistry

    logger.info("Using activity fallback for think phase")

    # Get available activity names from registry
    registry = ActivityRegistry()
    available_activities = registry.list_names()

    # Generate activity-aware prompt with available activities
    activity_prompt = generate_activity_prompt(
        goal=think_input.goal,
        current_step=think_input.current_step,
        history=think_input.history,
        context=think_input.context,
        available_activities=available_activities,
    )

    # Build messages with activity system prompt
    messages = [
        Message(
            role=Role.SYSTEM,
            content=ACTIVITY_SYSTEM_PROMPT,
        ),
        Message(role=Role.USER, content=activity_prompt),
    ]

    # Call LLM without tools
    response = await provider.chat_completion(
        context=messages,
        model=target_model,
        stream=False,
        temperature=0.7,
        # No tools parameter - using activity system
    )

    response_content = provider.extract_response_content(response)

    # Parse activities from response
    parser = ActivityRequestParser()
    parse_result = parser.parse(response_content)

    # Store activity state in working memory for act_phase
    agent_state.set_working_memory("using_activity_fallback", True)
    agent_state.set_working_memory("pending_activities_text", response_content)
    agent_state.set_working_memory("parsed_activity_requests", parse_result.requests)

    # Determine output
    is_final = parser.is_final_answer(response_content)
    final_answer_text = None

    if is_final:
        final_answer_text = parser.extract_final_answer(response_content)
        agent_state.is_finished = True
        agent_state.final_answer = final_answer_text

    # Create a pseudo-ToolCall for the first activity (for compatibility)
    proposed_action = None
    if parse_result.has_requests and not is_final:
        first_activity = parse_result.requests[0]
        from ....models import ToolCall

        proposed_action = ToolCall(
            id=f"activity_{first_activity.activity}",
            name=f"activity:{first_activity.activity}",
            arguments=first_activity.parameters,
        )
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
        confidence=ConfidenceLevel.MEDIUM,
        using_activity_fallback=True,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_thinking_prompt(
    think_input: ThinkInput, agent_state: EnhancedAgentState, prompt_registry: Optional[Any]
) -> str:
    """
    Generate the thinking prompt using prompt library or fallback.

    Args:
        think_input: Thinking input configuration
        agent_state: Current agent state
        prompt_registry: Optional prompt registry

    Returns:
        Formatted thinking prompt
    """
    # Format tool definitions as string
    tools_str = _format_tools(think_input.available_tools)

    # Try to use prompt library
    if prompt_registry:
        try:
            return prompt_registry.render(
                template_id="thinking_prompt",
                variables={
                    "goal": think_input.goal,
                    "current_step": think_input.current_step,
                    "history": think_input.history or "No previous actions.",
                    "context": think_input.context or "",
                    "tools": tools_str,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to use prompt registry: {e}, falling back")

    # Fallback prompt
    prompt = f"""You are solving this task:

GOAL: {think_input.goal}

CURRENT STEP: {think_input.current_step}
"""

    if think_input.history:
        prompt += f"\n\nRECENT HISTORY:\n{think_input.history}"

    if think_input.context:
        prompt += f"\n\nRELEVANT CONTEXT:\n{think_input.context}"

    prompt += f"""

AVAILABLE TOOLS:
{tools_str}

Use the ReAct format:

Thought: [Your reasoning about what to do next]
Action: [Tool name]
Action Input: [Tool arguments]

OR if the task is complete:

Thought: [Final reasoning]
Final Answer: [Complete answer to the goal]

Respond now:
"""

    return prompt


def _format_tools(tool_definitions: List[Dict[str, Any]]) -> str:
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
        if params and isinstance(params, dict):
            properties = params.get("properties", {})
            if properties:
                param_names = list(properties.keys())[:5]  # Limit to first 5 params
                if param_names:
                    param_summary = f" (params: {', '.join(param_names)})"

        lines.append(f"- {name}: {desc}{param_summary}")

    return "\n".join(lines) if lines else "No tools available."


def _parse_think_response(
    response_text: str,
    response_dict: Optional[Dict[str, Any]],
    tool_manager: "ToolManager",
) -> ThinkOutput:
    """
    Parse the LLM response into structured ThinkOutput.

    Args:
        response_text: Extracted text content from the LLM response
        response_dict: Original response dict for token usage extraction
        tool_manager: Tool manager for validation

    Returns:
        Parsed ThinkOutput
    """

    # Initialize output
    thought = ""
    proposed_action = None
    is_final_answer = False
    final_answer = None
    confidence = ConfidenceLevel.MEDIUM

    # Extract Thought
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    if thought_match:
        thought = thought_match.group(1).strip()

    # Check for Final Answer
    final_answer_match = re.search(
        r"Final Answer:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE
    )

    if final_answer_match:
        is_final_answer = True
        final_answer = final_answer_match.group(1).strip()
        confidence = ConfidenceLevel.HIGH
    else:
        # Extract Action
        action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", response_text, re.IGNORECASE)

        action_input_match = re.search(
            r"Action Input:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE
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
        confidence=confidence,
        reasoning_tokens=reasoning_tokens,
    )


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
