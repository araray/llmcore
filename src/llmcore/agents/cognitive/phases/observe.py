# src/llmcore/agents/cognitive/phases/observe.py
"""
OBSERVE Phase Implementation.

The OBSERVE phase processes the results of an executed action. It:
- Analyzes the tool result
- Compares against expected outcomes
- Extracts key insights
- Determines if follow-up is needed
- Prepares observations for reflection

This phase transforms raw execution results into structured observations
that can be used for learning and planning.

References:
    - Technical Spec: Section 5.3.6 (OBSERVE Phase)
    - Dossier: Step 2.6 (Cognitive Phases - OBSERVE)
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..models import EnhancedAgentState, ObserveInput, ObserveOutput

if TYPE_CHECKING:
    from ....models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# OBSERVE PHASE FUNCTION
# =============================================================================


async def observe_phase(
    agent_state: EnhancedAgentState, observe_input: ObserveInput, tracer: Optional[Any] = None
) -> ObserveOutput:
    """
    Execute the OBSERVE phase of the cognitive cycle.

    Processes action results by:
    1. Analyzing the tool result
    2. Comparing against expectations
    3. Extracting key insights
    4. Determining follow-up needs
    5. Formatting observations for reflection

    Args:
        agent_state: Current enhanced agent state
        observe_input: Input with action and result
        tracer: Optional OpenTelemetry tracer

    Returns:
        ObserveOutput with processed observations

    Example:
        >>> observe_input = ObserveInput(
        ...     action_taken=ToolCall(name="calculator", arguments={"expr": "2+2"}),
        ...     action_result=ToolResult(content="4", is_error=False),
        ...     expected_outcome="The result should be 4"
        ... )
        >>>
        >>> output = await observe_phase(
        ...     agent_state=state,
        ...     observe_input=observe_input
        ... )
        >>>
        >>> print(output.observation)
        >>> print(f"Matches expectation: {output.matches_expectation}")
    """
    from ....tracing import add_span_attributes, create_span

    with create_span(tracer, "cognitive.observe") as span:
        logger.debug(f"Starting OBSERVE phase for action: {observe_input.action_taken.name}")

        # 1. Analyze the result
        observation = _analyze_result(
            action=observe_input.action_taken,
            result=observe_input.action_result,
            expected=observe_input.expected_outcome,
        )

        # 2. Check if result matches expectation
        matches_expectation = None
        if observe_input.expected_outcome:
            matches_expectation = _check_expectation(
                result=observe_input.action_result, expected=observe_input.expected_outcome
            )

        # 3. Extract insights
        insights = _extract_insights(
            action=observe_input.action_taken,
            result=observe_input.action_result,
            matches_expectation=matches_expectation,
        )

        # 4. Determine if follow-up is needed
        follow_up_needed = _needs_follow_up(
            result=observe_input.action_result, matches_expectation=matches_expectation
        )

        # 5. Create output
        output = ObserveOutput(
            observation=observation,
            matches_expectation=matches_expectation,
            insights=insights,
            follow_up_needed=follow_up_needed,
        )

        # 6. Add tracing
        if span:
            add_span_attributes(
                span,
                {
                    "observe.tool_name": observe_input.action_taken.name,
                    "observe.is_error": observe_input.action_result.is_error,
                    "observe.matches_expectation": matches_expectation
                    if matches_expectation is not None
                    else "unknown",
                    "observe.insights_count": len(insights),
                    "observe.follow_up_needed": follow_up_needed,
                },
            )

        logger.info(
            f"OBSERVE phase complete: {observe_input.action_taken.name}, "
            f"match={matches_expectation}, follow_up={follow_up_needed}"
        )

        return output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _analyze_result(action: "ToolCall", result: "ToolResult", expected: Optional[str]) -> str:
    """
    Analyze the tool result and create a structured observation.

    Args:
        action: The action that was taken
        result: The result of the action
        expected: Optional expected outcome

    Returns:
        Structured observation string
    """
    observation_parts = []

    # Basic action description
    observation_parts.append(f"Executed {action.name} with arguments: {action.arguments}")

    # Result analysis
    if result.is_error:
        observation_parts.append(f"Result: ERROR - {result.content}")
    else:
        # Truncate long results
        content = result.content
        if len(content) > 500:
            content = content[:500] + "... (truncated)"
        observation_parts.append(f"Result: {content}")

    # Expectation comparison
    if expected:
        observation_parts.append(f"Expected: {expected}")

    return "\n".join(observation_parts)


def _check_expectation(result: "ToolResult", expected: str) -> bool:
    """
    Check if the result matches the expected outcome.

    Uses multiple strategies:
    1. If result is an error, it doesn't match
    2. Check if the expected value appears in the result
    3. Fall back to word overlap for complex expectations

    Args:
        result: The tool result
        expected: Expected outcome description

    Returns:
        True if matches, False otherwise
    """
    # If result is an error, it doesn't match expectations
    if result.is_error:
        return False

    result_lower = result.content.lower().strip()
    expected_lower = expected.lower().strip()

    # Strategy 1: Direct value match
    # Extract key values from expected (numbers, specific words)
    import re
    expected_values = re.findall(r'\b(\d+|true|false|yes|no|success|fail)\b', expected_lower)
    result_values = re.findall(r'\b(\d+|true|false|yes|no|success|fail)\b', result_lower)
    
    # If expected contains specific values, check if they appear in result
    if expected_values:
        for val in expected_values:
            if val in result_values or val in result_lower:
                return True
    
    # Strategy 2: Result content appears in expected or vice versa
    if result_lower in expected_lower or expected_lower in result_lower:
        return True
    
    # Strategy 3: Word overlap for complex expectations
    # Check if key words from expected are in result
    expected_words = set(expected_lower.split())
    result_words = set(result_lower.split())

    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "be", "should", "will", "would", "could"}
    expected_words -= stop_words
    result_words -= stop_words

    # Calculate overlap
    if not expected_words:
        return True  # If no meaningful expected words, assume match
        
    overlap = len(expected_words & result_words)
    total = len(expected_words)

    # If more than 30% of expected words are in result, consider it a match
    # (lowered from 50% to be more lenient)
    return (overlap / total) > 0.3 if total > 0 else True


def _extract_insights(
    action: "ToolCall", result: "ToolResult", matches_expectation: Optional[bool]
) -> list:
    """
    Extract key insights from the action and result.

    Args:
        action: The action taken
        result: The result
        matches_expectation: Whether result matched expectation

    Returns:
        List of insight strings
    """
    insights = []

    # Insight about success/failure
    if result.is_error:
        insights.append(f"Tool {action.name} failed: {result.content[:100]}")
    else:
        insights.append(f"Tool {action.name} succeeded")

    # Insight about expectation matching
    if matches_expectation is not None:
        if matches_expectation:
            insights.append("Result matched expectations")
        else:
            insights.append("Result did not match expectations - may need adjustment")

    # Tool-specific insights
    if action.name in ["execute_shell", "execute_python"]:
        if not result.is_error and result.content:
            insights.append("Code execution produced output")
        elif not result.is_error and not result.content:
            insights.append("Code execution produced no output")

    if action.name in ["save_file", "create_directory"]:
        if not result.is_error:
            insights.append("File system operation completed")

    return insights


def _needs_follow_up(result: "ToolResult", matches_expectation: Optional[bool]) -> bool:
    """
    Determine if follow-up action is needed.

    Args:
        result: The tool result
        matches_expectation: Whether result matched expectation

    Returns:
        True if follow-up needed, False otherwise
    """
    # Errors always need follow-up
    if result.is_error:
        return True

    # Unexpected results may need follow-up
    if matches_expectation is False:
        return True

    # Otherwise, probably okay to continue
    return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["observe_phase"]
