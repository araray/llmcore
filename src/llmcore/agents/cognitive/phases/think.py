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
    from ....memory.manager import MemoryManager
    from ....models import Message, Role, ToolCall
    from ....providers.manager import ProviderManager
    from ...tools import ToolManager

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

    Returns:
        ThinkOutput with reasoning and proposed action

    Example:
        >>> think_input = ThinkInput(
        ...     goal="Calculate factorial of 10",
        ...     current_step="Determine calculation approach",
        ...     available_tools=tool_manager.get_tool_definitions()
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

            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False,
                tools=tools_param,
                temperature=0.7,
            )

            # Extract response content
            response_content = provider.extract_response_content(response)

            # 4. Parse response
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
    """Format tool definitions for prompt."""
    if not tool_definitions:
        return "No tools available."

    lines = []
    for tool in tool_definitions:
        name = tool.get("function", {}).get("name", "unknown")
        desc = tool.get("function", {}).get("description", "")
        lines.append(f"- {name}: {desc}")

    return "\n".join(lines)


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
