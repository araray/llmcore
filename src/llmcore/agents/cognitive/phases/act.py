# src/llmcore/agents/cognitive/phases/act.py
"""
ACT Phase Implementation.

The ACT phase executes the chosen tool call after validation. It handles:
- Tool execution through ToolManager
- Activity execution for models without native tools (G3 Phase 6)
- Sandbox integration (if active)
- Error handling and retries
- Execution metrics (time, success)
- Output tracking

This is the only phase where actual execution happens - all other phases
are planning, reasoning, or reflection.

References:
    - Technical Spec: Section 5.3.5 (ACT Phase)
    - Dossier: Step 2.6 (Cognitive Phases - ACT)
    - G3_COMPLETE_IMPLEMENTATION_PLAN.md
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from ..models import ActInput, ActOutput, EnhancedAgentState, ValidationResult

if TYPE_CHECKING:
    from ....models import ToolCall, ToolResult
    from ....config.agents_config import AgentsConfig
    from ...tools import ToolManager
    from ...sandbox import SandboxProvider

logger = logging.getLogger(__name__)


# =============================================================================
# ACT PHASE FUNCTION
# =============================================================================


async def act_phase(
    agent_state: EnhancedAgentState,
    act_input: ActInput,
    tool_manager: "ToolManager",
    tracer: Optional[Any] = None,
    max_retries: int = 0,
    agents_config: Optional["AgentsConfig"] = None,
    sandbox: Optional["SandboxProvider"] = None,
) -> ActOutput:
    """
    Execute the ACT phase of the cognitive cycle.

    Executes a tool call by:
    1. Checking if activity fallback is active (G3 Phase 6)
    2. Checking validation status
    3. Executing tool through ToolManager or ActivityLoop
    4. Measuring execution time
    5. Handling errors with optional retries
    6. Recording execution metrics

    Args:
        agent_state: Current enhanced agent state
        act_input: Input with tool call to execute
        tool_manager: Tool manager for execution
        tracer: Optional OpenTelemetry tracer
        max_retries: Maximum retry attempts on failure (default: 0)
        agents_config: Optional agents configuration (G3)
        sandbox: Optional sandbox provider (G3)

    Returns:
        ActOutput with execution results

    Example:
        >>> act_input = ActInput(
        ...     tool_call=ToolCall(name="calculator", arguments={"expr": "2+2"}),
        ...     validation_result=ValidationResult.APPROVED
        ... )
        >>>
        >>> output = await act_phase(
        ...     agent_state=state,
        ...     act_input=act_input,
        ...     tool_manager=tool_manager
        ... )
        >>>
        >>> if output.success:
        ...     print(f"Result: {output.tool_result.content}")
    """
    from ....models import ToolResult
    from ....tracing import add_span_attributes, create_span, record_span_exception

    with create_span(tracer, "cognitive.act") as span:
        try:
            # =================================================================
            # G3 Phase 6: Check for Activity Fallback
            # =================================================================
            using_activity_fallback = agent_state.get_working_memory(
                "using_activity_fallback", False
            )

            if using_activity_fallback:
                return await _act_phase_with_activities(
                    agent_state=agent_state,
                    tool_manager=tool_manager,
                    sandbox=sandbox,
                    tracer=tracer,
                    span=span,
                )

            # =================================================================
            # Normal Tool Execution Path
            # =================================================================
            logger.debug(f"Starting ACT phase: {act_input.tool_call.name}")

            # 1. Check validation (if provided)
            if act_input.validation_result == ValidationResult.REJECTED:
                logger.warning("Attempted to execute rejected action")
                return ActOutput(
                    tool_result=ToolResult(
                        tool_call_id=act_input.tool_call.id,
                        content="Action was rejected by validation",
                        is_error=True,
                    ),
                    execution_time_ms=0.0,
                    success=False,
                )

            if act_input.validation_result == ValidationResult.REQUIRES_HUMAN_APPROVAL:
                logger.info("Action requires human approval - pausing")
                return ActOutput(
                    tool_result=ToolResult(
                        tool_call_id=act_input.tool_call.id,
                        content="Action requires human approval",
                        is_error=False,
                    ),
                    execution_time_ms=0.0,
                    success=False,
                )

            # 2. Execute tool with retry logic
            start_time = time.time()
            tool_result = None
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt} of {max_retries}")

                    # Execute through ToolManager
                    # Note: ToolManager handles sandbox routing automatically
                    tool_result = await tool_manager.execute_tool(tool_call=act_input.tool_call)

                    # Success - break retry loop
                    if not tool_result.is_error:
                        break
                    else:
                        last_error = tool_result.content
                        if attempt < max_retries:
                            logger.warning(
                                f"Tool execution failed (attempt {attempt + 1}): {last_error}"
                            )

                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Tool execution error (attempt {attempt + 1}): {e}")

                    if attempt == max_retries:
                        # Create error result
                        tool_result = ToolResult(
                            tool_call_id=act_input.tool_call.id,
                            content=f"Execution failed: {str(e)}",
                            is_error=True,
                        )

            # 3. Calculate execution time
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            # 4. Determine success
            success = tool_result is not None and not tool_result.is_error

            # 5. Update agent state
            agent_state.total_tool_calls += 1

            # 6. Create output
            output = ActOutput(
                tool_result=tool_result, execution_time_ms=execution_time_ms, success=success
            )

            # 7. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "act.tool_name": act_input.tool_call.name,
                        "act.success": success,
                        "act.execution_time_ms": execution_time_ms,
                        "act.retries": max_retries,
                        "act.activity_fallback": False,
                    },
                )

            logger.info(
                f"ACT phase complete: {act_input.tool_call.name}, "
                f"success={success}, time={execution_time_ms:.1f}ms"
            )

            return output

        except Exception as e:
            logger.error(f"ACT phase failed: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

            # Return error output
            from ....models import ToolResult

            return ActOutput(
                tool_result=ToolResult(
                    tool_call_id=act_input.tool_call.id,
                    content=f"ACT phase error: {str(e)}",
                    is_error=True,
                ),
                execution_time_ms=0.0,
                success=False,
            )


# =============================================================================
# ACTIVITY FALLBACK (G3 Phase 6)
# =============================================================================


async def _act_phase_with_activities(
    agent_state: EnhancedAgentState,
    tool_manager: "ToolManager",
    sandbox: Optional["SandboxProvider"],
    tracer: Optional[Any],
    span: Optional[Any],
) -> ActOutput:
    """
    Execute activities through the activity system (fallback for non-tool models).

    Args:
        agent_state: Current agent state
        tool_manager: Tool manager (for context)
        sandbox: Optional sandbox provider
        tracer: Optional tracer
        span: Optional tracing span

    Returns:
        ActOutput with activity execution results
    """
    from ....models import ToolResult
    from ....tracing import add_span_attributes
    from ...activities.loop import ActivityLoop, ActivityLoopConfig
    from ...activities.registry import ExecutionContext

    logger.info("Executing via activity fallback")

    start_time = time.time()

    # Get pending activities from working memory
    pending_text = agent_state.get_working_memory("pending_activities_text", "")
    parsed_requests = agent_state.get_working_memory("parsed_activity_requests", [])

    if not pending_text and not parsed_requests:
        logger.warning("No pending activities to execute")
        return ActOutput(
            tool_result=ToolResult(
                tool_call_id="activity_fallback",
                content="No activities to execute",
                is_error=False,
            ),
            execution_time_ms=0.0,
            success=True,
        )

    # Create activity loop
    activity_loop = ActivityLoop(
        config=ActivityLoopConfig(
            max_per_iteration=10,
            stop_on_error=False,
        ),
        sandbox=sandbox,
    )

    # Create execution context
    context = ExecutionContext(
        sandbox=sandbox,
        tool_manager=tool_manager,
    )

    try:
        # Process activities
        result = await activity_loop.process_output(pending_text, context=context)

        # Calculate execution time
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        # Determine success
        success = not result.has_errors if hasattr(result, 'has_errors') else True

        # Build result content
        observation = result.observation if result.observation else "Activities executed"

        # Clear activity state
        agent_state.set_working_memory("using_activity_fallback", False)
        agent_state.set_working_memory("pending_activities_text", None)
        agent_state.set_working_memory("parsed_activity_requests", None)

        # Check if this was a final answer
        if result.is_final_answer:
            agent_state.is_finished = True
            agent_state.final_answer = observation

        # Update agent state
        agent_state.total_tool_calls += len(result.executions) if result.executions else 1

        # Add tracing
        if span:
            add_span_attributes(
                span,
                {
                    "act.activity_fallback": True,
                    "act.activities_executed": len(result.executions) if result.executions else 0,
                    "act.success": success,
                    "act.execution_time_ms": execution_time_ms,
                    "act.is_final": result.is_final_answer,
                },
            )

        logger.info(
            f"Activity fallback complete: "
            f"activities={len(result.executions) if result.executions else 0}, "
            f"success={success}, time={execution_time_ms:.1f}ms"
        )

        return ActOutput(
            tool_result=ToolResult(
                tool_call_id="activity_fallback",
                content=observation,
                is_error=not success,
            ),
            execution_time_ms=execution_time_ms,
            success=success,
        )

    except Exception as e:
        logger.error(f"Activity fallback execution failed: {e}", exc_info=True)

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        return ActOutput(
            tool_result=ToolResult(
                tool_call_id="activity_fallback",
                content=f"Activity execution failed: {str(e)}",
                is_error=True,
            ),
            execution_time_ms=execution_time_ms,
            success=False,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["act_phase"]
