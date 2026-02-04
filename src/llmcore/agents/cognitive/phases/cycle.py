# src/llmcore/agents/cognitive/cycle.py
"""
Cognitive Cycle Orchestrator.

The CognitiveCycle class orchestrates the complete 8-phase cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

It provides a high-level interface for running complete iterations and manages
the coordination between all phases, including error handling, tracing, and
state management.

References:
    - Technical Spec: Section 5.3.9 (Cognitive Cycle Orchestrator)
    - Dossier: Step 2.7 (Cognitive Cycle Orchestrator)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import AsyncIterator, Callable

from ..models import (
    ActInput,
    ConfidenceLevel,
    CycleIteration,
    EnhancedAgentState,
    ObserveInput,
    # Phase inputs
    PerceiveInput,
    PlanInput,
    ReflectInput,
    ThinkInput,
    UpdateInput,
    ValidateInput,
    ValidateOutput,
    # Enums
    ValidationResult,
)
from .act import act_phase
from .observe import observe_phase
from .perceive import perceive_phase
from .plan import plan_phase
from .reflect import reflect_phase
from .think import think_phase
from .update import update_phase
from .validate import validate_phase

if TYPE_CHECKING:
    from ....config.agents_config import AgentsConfig
    from ....memory.manager import MemoryManager
    from ....providers.manager import ProviderManager
    from ....storage.manager import StorageManager
    from ..sandbox import SandboxProvider
    from ..tools import ToolManager

from ...resilience.circuit_breaker import (
    AgentCircuitBreaker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STREAMING RESULT MODEL
# =============================================================================


@dataclass
class StreamingIterationResult:
    """
    Result yielded after each cognitive cycle iteration during streaming.

    Contains all relevant information for real-time progress display,
    including the current iteration state, progress estimates, and
    action/observation summaries.

    Attributes:
        iteration: Current iteration number (1-indexed)
        max_iterations: Maximum allowed iterations
        progress: Estimated progress toward goal (0.0 to 1.0)
        is_complete: Whether the task is complete
        is_final: Whether this is the last update (complete or stopped)
        status: Current status string
        current_phase: Name of the current/last phase executed
        message: Human-readable summary of this iteration
        action_name: Name of the action taken (if any)
        action_summary: Brief description of the action
        observation_summary: Brief description of the result/observation
        step_completed: Whether the current plan step was completed
        plan_step: Current plan step being worked on
        error: Error message if iteration failed
        tokens_used: Tokens used in this iteration
        duration_ms: Duration of this iteration in milliseconds
        stop_reason: Reason for stopping (if stopped early)
    """

    iteration: int
    max_iterations: int
    progress: float
    is_complete: bool = False
    is_final: bool = False
    status: str = "in_progress"
    current_phase: str = "unknown"
    message: str = ""
    action_name: str | None = None
    action_summary: str | None = None
    observation_summary: str | None = None
    step_completed: bool = False
    plan_step: str | None = None
    error: str | None = None
    tokens_used: int = 0
    duration_ms: float = 0.0
    stop_reason: str | None = None


# =============================================================================
# COGNITIVE CYCLE ORCHESTRATOR
# =============================================================================


class CognitiveCycle:
    """
    Orchestrates the complete 8-phase cognitive cycle.

    The CognitiveCycle manages the execution of all phases in sequence,
    handles errors, coordinates state updates, and provides a clean
    interface for agent execution.

    Example:
        >>> cycle = CognitiveCycle(
        ...     provider_manager=provider_manager,
        ...     memory_manager=memory_manager,
        ...     storage_manager=storage_manager,
        ...     tool_manager=tool_manager
        ... )
        >>>
        >>> # Run single iteration
        >>> await cycle.run_iteration(
        ...     agent_state=state,
        ...     session_id="session-123"
        ... )
        >>>
        >>> # Run until completion
        >>> final_result = await cycle.run_until_complete(
        ...     agent_state=state,
        ...     session_id="session-123",
        ...     max_iterations=10
        ... )

    Attributes:
        provider_manager: Provider manager for LLM calls
        memory_manager: Memory manager for context
        storage_manager: Storage manager for episodic memory
        tool_manager: Tool manager for actions
        prompt_registry: Optional prompt registry
    """

    def __init__(
        self,
        provider_manager: "ProviderManager",
        memory_manager: "MemoryManager",
        storage_manager: "StorageManager",
        tool_manager: "ToolManager",
        prompt_registry: Any | None = None,
        tracer: Any | None = None,
    ):
        """
        Initialize the cognitive cycle orchestrator.

        Args:
            provider_manager: Provider manager for LLM calls
            memory_manager: Memory manager for context
            storage_manager: Storage manager for episodic memory
            tool_manager: Tool manager for actions
            prompt_registry: Optional prompt registry
            tracer: Optional OpenTelemetry tracer
        """
        self.provider_manager = provider_manager
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager
        self.prompt_registry = prompt_registry
        self.tracer = tracer

    async def run_iteration(
        self,
        agent_state: EnhancedAgentState,
        session_id: str,
        sandbox: Optional["SandboxProvider"] = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        skip_validation: bool = False,
        approval_callback: Callable[[str], bool] | None = None,
    ) -> CycleIteration:
        """
        Run a single complete cognitive iteration.

        Executes all 8 phases in sequence:
        1. PERCEIVE: Gather inputs
        2. PLAN: Create/update strategic plan (first iteration only or on update)
        3. THINK: Reason about next action
        4. VALIDATE: Verify action safety
        5. ACT: Execute action
        6. OBSERVE: Process results
        7. REFLECT: Evaluate and learn
        8. UPDATE: Apply changes

        Args:
            agent_state: Current agent state
            session_id: Session ID for memory
            sandbox: Optional active sandbox
            provider_name: Optional provider override
            model_name: Optional model override

        Returns:
            Completed CycleIteration with all phase outputs

        Raises:
            Exception: If critical phase fails
        """
        from ...tracing import add_span_attributes, create_span, record_span_exception

        with create_span(self.tracer, "cognitive.iteration") as span:
            # Start new iteration
            iteration_number = agent_state.iteration_count + 1
            iteration = agent_state.start_iteration(iteration_number)

            try:
                logger.info(f"Starting iteration {iteration_number}")

                # Store skip_validation in working memory for activity fallback HITL
                agent_state.set_working_memory("skip_validation", skip_validation)

                # Store approval_callback in working memory for activity fallback HITL
                agent_state.set_working_memory("approval_callback", approval_callback)

                # ============================================================
                # Phase 1: PERCEIVE
                # ============================================================
                perceive_input = PerceiveInput(goal=agent_state.goal, force_refresh=False)

                iteration.perceive_output = await perceive_phase(
                    agent_state=agent_state,
                    perceive_input=perceive_input,
                    memory_manager=self.memory_manager,
                    sandbox=sandbox,
                    tracer=self.tracer,
                )

                # ============================================================
                # Phase 2: PLAN (first iteration or if plan needs update)
                # ============================================================
                should_plan = (
                    iteration_number == 1
                    or len(agent_state.plan) == 0
                    or agent_state.get_working_memory("plan_needs_update", False)
                )

                if should_plan:
                    plan_input = PlanInput(
                        goal=agent_state.goal,
                        context="\n".join(iteration.perceive_output.retrieved_context),
                        existing_plan=agent_state.plan if len(agent_state.plan) > 0 else None,
                    )

                    iteration.plan_output = await plan_phase(
                        agent_state=agent_state,
                        plan_input=plan_input,
                        provider_manager=self.provider_manager,
                        prompt_registry=self.prompt_registry,
                        tracer=self.tracer,
                        provider_name=provider_name,
                        model_name=model_name,
                    )

                # ============================================================
                # Phase 3: THINK
                # ============================================================
                current_step = (
                    agent_state.plan[agent_state.current_plan_step_index]
                    if agent_state.current_plan_step_index < len(agent_state.plan)
                    else "Complete the goal"
                )

                think_input = ThinkInput(
                    goal=agent_state.goal,
                    current_step=current_step,
                    history=self._build_history(agent_state),
                    context="\n".join(iteration.perceive_output.retrieved_context),
                    available_tools=[
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in self.tool_manager.get_tool_definitions()
                    ],
                )

                iteration.think_output = await think_phase(
                    agent_state=agent_state,
                    think_input=think_input,
                    provider_manager=self.provider_manager,
                    memory_manager=self.memory_manager,
                    tool_manager=self.tool_manager,
                    prompt_registry=self.prompt_registry,
                    tracer=self.tracer,
                    provider_name=provider_name,
                    model_name=model_name,
                )

                # If final answer, skip remaining phases
                if iteration.think_output.is_final_answer:
                    logger.info("Final answer provided, completing iteration")
                    agent_state.complete_iteration(success=True)
                    return iteration

                # ============================================================
                # Phase 4: VALIDATE
                # ============================================================
                if iteration.think_output.proposed_action:
                    if skip_validation:
                        # Auto-approve when skip_validation is True
                        logger.info("Skipping validation (auto-approve enabled)")
                        iteration.validate_output = ValidateOutput(
                            result=ValidationResult.APPROVED,
                            confidence=ConfidenceLevel.HIGH,
                            concerns=[],
                            suggestions=[],
                            requires_human_approval=False,
                        )
                    else:
                        validate_input = ValidateInput(
                            goal=agent_state.goal,
                            proposed_action=iteration.think_output.proposed_action,
                            reasoning=iteration.think_output.thought,
                            risk_tolerance="medium",  # Could be configurable
                        )

                        iteration.validate_output = await validate_phase(
                            agent_state=agent_state,
                            validate_input=validate_input,
                            provider_manager=self.provider_manager,
                            prompt_registry=self.prompt_registry,
                            tracer=self.tracer,
                            provider_name=provider_name,
                            model_name=model_name,
                        )

                    # ========================================================
                    # Phase 5: ACT
                    # ========================================================
                    act_input = ActInput(
                        tool_call=iteration.think_output.proposed_action,
                        validation_result=iteration.validate_output.result,
                    )

                    iteration.act_output = await act_phase(
                        agent_state=agent_state,
                        act_input=act_input,
                        tool_manager=self.tool_manager,
                        tracer=self.tracer,
                    )

                    # ========================================================
                    # Phase 6: OBSERVE
                    # ========================================================
                    observe_input = ObserveInput(
                        action_taken=iteration.think_output.proposed_action,
                        action_result=iteration.act_output.tool_result,
                        expected_outcome=None,  # Could extract from think_output
                    )

                    iteration.observe_output = await observe_phase(
                        agent_state=agent_state, observe_input=observe_input, tracer=self.tracer
                    )
                else:
                    logger.warning("No action proposed by THINK phase")

                # ============================================================
                # Phase 7: REFLECT
                # ============================================================
                reflect_input = ReflectInput(
                    goal=agent_state.goal,
                    plan=agent_state.plan,
                    current_step_index=agent_state.current_plan_step_index,
                    last_action=iteration.think_output.proposed_action
                    if iteration.think_output.proposed_action
                    else agent_state.pending_tool_call,
                    observation=iteration.observe_output.observation
                    if iteration.observe_output
                    else "No observation",
                    iteration_number=iteration_number,
                )

                iteration.reflect_output = await reflect_phase(
                    agent_state=agent_state,
                    reflect_input=reflect_input,
                    provider_manager=self.provider_manager,
                    prompt_registry=self.prompt_registry,
                    tracer=self.tracer,
                    provider_name=provider_name,
                    model_name=model_name,
                )

                # ============================================================
                # Phase 8: UPDATE
                # ============================================================
                update_input = UpdateInput(
                    reflection=iteration.reflect_output, current_state=agent_state
                )

                iteration.update_output = await update_phase(
                    agent_state=agent_state,
                    update_input=update_input,
                    storage_manager=self.storage_manager,
                    session_id=session_id,
                    tracer=self.tracer,
                )

                # ============================================================
                # Complete iteration
                # ============================================================
                agent_state.complete_iteration(success=True)

                if span:
                    add_span_attributes(
                        span,
                        {
                            "iteration.number": iteration_number,
                            "iteration.success": True,
                            "iteration.phases_completed": len(iteration.phases_completed),
                            "iteration.final_answer": iteration.think_output.is_final_answer
                            if iteration.think_output
                            else False,
                        },
                    )

                logger.info(f"Iteration {iteration_number} completed successfully")
                return iteration

            except Exception as e:
                logger.error(f"Iteration {iteration_number} failed: {e}", exc_info=True)
                iteration.mark_completed(success=False)
                agent_state.complete_iteration(success=False)

                if span:
                    record_span_exception(span, e)

                raise

    async def run_until_complete(
        self,
        agent_state: EnhancedAgentState,
        session_id: str,
        max_iterations: int = 10,
        sandbox: Optional["SandboxProvider"] = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        skip_validation: bool = False,
        agents_config: Optional["AgentsConfig"] = None,
        approval_callback: Callable[[str], bool] | None = None,
    ) -> str:
        """
        Run cognitive iterations until task is complete or max iterations reached.

        Args:
            agent_state: Enhanced agent state
            session_id: Session ID for memory
            max_iterations: Maximum iterations to run
            sandbox: Optional active sandbox
            provider_name: Optional provider override
            model_name: Optional model override
            skip_validation: If True, auto-approve all actions (bypass VALIDATE phase)
            agents_config: Optional agents configuration (G3)

        Returns:
            Final answer or status message

        Example:
            >>> state = EnhancedAgentState(goal="Calculate 10!")
            >>> result = await cycle.run_until_complete(
            ...     agent_state=state,
            ...     session_id="session-123",
            ...     max_iterations=10
            ... )
            >>> print(result)
        """
        # =================================================================
        # G3 Phase 5: Load Configuration and Initialize Circuit Breaker
        # =================================================================
        if agents_config is None:
            from ...config.agents_config import AgentsConfig

            agents_config = AgentsConfig()

        cb_config = agents_config.circuit_breaker

        # Initialize circuit breaker (G3 Phase 5)
        circuit_breaker: AgentCircuitBreaker | None = None
        if cb_config.enabled:
            circuit_breaker = AgentCircuitBreaker(
                max_iterations=min(max_iterations, cb_config.max_iterations),
                max_same_errors=cb_config.max_same_errors,
                max_execution_time_seconds=cb_config.max_execution_time_seconds,
                max_total_cost=cb_config.max_total_cost,
                progress_stall_threshold=cb_config.progress_stall_threshold,
                progress_stall_tolerance=cb_config.progress_stall_tolerance,
            )
            circuit_breaker.start()

        logger.info(
            f"Starting cognitive cycle: max_iterations={max_iterations}, "
            f"skip_validation={skip_validation}, "
            f"circuit_breaker={'enabled' if circuit_breaker else 'disabled'}"
        )

        actual_iterations = 0
        stopped_early = False
        stop_reason = None
        last_error: str | None = None
        accumulated_cost = 0.0

        for iteration_num in range(max_iterations):
            # Check if already finished
            if agent_state.is_finished:
                logger.info(f"Task completed in {iteration_num} iterations")
                return agent_state.final_answer or "Task completed successfully"

            # Run iteration
            try:
                iteration = await self.run_iteration(
                    agent_state=agent_state,
                    session_id=session_id,
                    sandbox=sandbox,
                    provider_name=provider_name,
                    model_name=model_name,
                    skip_validation=skip_validation,
                    approval_callback=approval_callback,
                )
                actual_iterations = iteration_num + 1
                last_error = None  # Clear error on success

                # Track cost if available
                if hasattr(iteration, "total_cost") and iteration.total_cost:
                    accumulated_cost += iteration.total_cost

                # Accumulate tokens from think phase if available
                if iteration.think_output and iteration.think_output.reasoning_tokens:
                    iteration.total_tokens_used += iteration.think_output.reasoning_tokens

                # =================================================================
                # G3 Phase 5: Circuit Breaker Check After Successful Iteration
                # =================================================================
                if circuit_breaker is not None:
                    # Estimate progress (0.0 to 1.0)
                    progress = getattr(agent_state, "progress_estimate", 0.0)
                    if progress == 0.0:
                        # Fallback: estimate based on iterations
                        progress = actual_iterations / max_iterations

                    # Extract step_completed from reflect output for accurate stall detection
                    step_completed = None
                    if iteration.reflect_output:
                        step_completed = getattr(iteration.reflect_output, "step_completed", None)

                    cb_result = circuit_breaker.check(
                        iteration=actual_iterations,
                        progress=progress,
                        error=None,
                        cost=accumulated_cost,
                        step_completed=step_completed,
                    )

                    if cb_result.tripped:
                        logger.warning(
                            f"Circuit breaker tripped: {cb_result.reason.value} - "
                            f"{cb_result.message}"
                        )
                        return (
                            f"Execution stopped by circuit breaker.\n"
                            f"Reason: {cb_result.reason.value}\n"
                            f"Details: {cb_result.message}\n"
                            f"Iterations completed: {actual_iterations}\n"
                            f"Progress: {progress:.1%}"
                        )

                # Check if we should stop
                if iteration.update_output and not iteration.update_output.should_continue:
                    stopped_early = True
                    # Determine stop reason
                    if agent_state.awaiting_human_approval:
                        stop_reason = "human_approval_required"
                    else:
                        stop_reason = "update_stopped"
                    logger.info(f"Stopping after {actual_iterations} iterations ({stop_reason})")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration_num + 1} failed: {e}")
                last_error = str(e)
                actual_iterations = iteration_num + 1

                # =================================================================
                # G3 Phase 5: Circuit Breaker Check After Error
                # =================================================================
                if circuit_breaker is not None:
                    progress = getattr(agent_state, "progress_estimate", 0.0)

                    cb_result = circuit_breaker.check(
                        iteration=actual_iterations,
                        progress=progress,
                        error=last_error,
                        cost=accumulated_cost,
                    )

                    if cb_result.tripped:
                        logger.warning(
                            f"Circuit breaker tripped on error: {cb_result.reason.value} - "
                            f"{cb_result.message}"
                        )
                        return (
                            f"Execution stopped by circuit breaker.\n"
                            f"Reason: {cb_result.reason.value}\n"
                            f"Details: {cb_result.message}\n"
                            f"Last error: {last_error[:200] if last_error else 'None'}\n"
                            f"Iterations completed: {actual_iterations}"
                        )

                # If circuit breaker hasn't tripped, the error is fatal
                # (unlike the old behavior which silently returned)
                if circuit_breaker is None:
                    return f"Task failed: {e!s}"

        # Check if task completed during the last iteration
        if agent_state.is_finished:
            return agent_state.final_answer or "Task completed"

        # Determine result based on how loop ended
        if stopped_early:
            if stop_reason == "human_approval_required":
                approval_prompt = (
                    agent_state.pending_approval_prompt or "Approval needed for proposed action"
                )
                logger.warning(f"Human approval required after {actual_iterations} iteration(s)")
                return (
                    f"Human approval required after {actual_iterations} iteration(s). "
                    f"Progress: {agent_state.progress_estimate:.1%}. {approval_prompt}"
                )
            else:
                # Stopped by UPDATE phase (e.g., progress stalled, reflection decided to stop)
                logger.info(
                    f"Task incomplete after {actual_iterations} iteration(s) (update stopped)"
                )
                return (
                    f"Task incomplete after {actual_iterations} iteration(s). "
                    f"Progress: {agent_state.progress_estimate:.1%}"
                )

        # Actually hit max iterations
        logger.warning(f"Max iterations ({max_iterations}) reached without completion")
        return (
            f"Task incomplete after {max_iterations} iterations (limit reached). "
            f"Progress: {agent_state.progress_estimate:.1%}"
        )

    async def run_streaming(
        self,
        agent_state: EnhancedAgentState,
        session_id: str,
        max_iterations: int = 10,
        sandbox: Optional["SandboxProvider"] = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        skip_validation: bool = False,
        agents_config: Optional["AgentsConfig"] = None,
        approval_callback: Callable[[str], bool] | None = None,
    ) -> AsyncIterator[StreamingIterationResult]:
        """
        Run cognitive iterations with streaming updates after each iteration.

        This method yields a StreamingIterationResult after each completed
        iteration, enabling real-time progress display in UIs/CLIs.

        Args:
            agent_state: Enhanced agent state
            session_id: Session ID for memory
            max_iterations: Maximum iterations to run
            sandbox: Optional active sandbox
            provider_name: Optional provider override
            model_name: Optional model override
            skip_validation: If True, auto-approve all actions
            agents_config: Optional agents configuration
            approval_callback: Optional callback for approval

        Yields:
            StreamingIterationResult after each iteration

        Example:
            >>> async for update in cycle.run_streaming(state, session_id, max_iterations=10):
            ...     print(f"Iteration {update.iteration}: {update.progress:.0%}")
            ...     if update.action_name:
            ...         print(f"  Action: {update.action_name}")
            ...     if update.is_final:
            ...         print(f"Final status: {update.status}")
        """
        # Load configuration and initialize circuit breaker (same as run_until_complete)
        if agents_config is None:
            from ....config.agents_config import AgentsConfig

            agents_config = AgentsConfig()

        cb_config = agents_config.circuit_breaker

        circuit_breaker: AgentCircuitBreaker | None = None
        if cb_config.enabled:
            circuit_breaker = AgentCircuitBreaker(
                max_iterations=min(max_iterations, cb_config.max_iterations),
                max_same_errors=cb_config.max_same_errors,
                max_execution_time_seconds=cb_config.max_execution_time_seconds,
                max_total_cost=cb_config.max_total_cost,
                progress_stall_threshold=cb_config.progress_stall_threshold,
                progress_stall_tolerance=cb_config.progress_stall_tolerance,
            )
            circuit_breaker.start()

        logger.info(
            f"Starting streaming cognitive cycle: max_iterations={max_iterations}, "
            f"skip_validation={skip_validation}"
        )

        accumulated_cost = 0.0

        for iteration_num in range(max_iterations):
            # Check if already finished before starting iteration
            if agent_state.is_finished:
                logger.info(f"Task completed in {iteration_num} iterations")
                yield StreamingIterationResult(
                    iteration=iteration_num,
                    max_iterations=max_iterations,
                    progress=1.0,
                    is_complete=True,
                    is_final=True,
                    status="complete",
                    current_phase="complete",
                    message=agent_state.final_answer or "Task completed successfully",
                )
                return

            # Run single iteration
            iteration_result: CycleIteration | None = None
            error_msg: str | None = None

            try:
                iteration_result = await self.run_iteration(
                    agent_state=agent_state,
                    session_id=session_id,
                    sandbox=sandbox,
                    provider_name=provider_name,
                    model_name=model_name,
                    skip_validation=skip_validation,
                    approval_callback=approval_callback,
                )

                # Track cost if available
                if hasattr(iteration_result, "total_cost") and iteration_result.total_cost:
                    accumulated_cost += iteration_result.total_cost

            except Exception as e:
                logger.error(f"Iteration {iteration_num + 1} failed: {e}")
                error_msg = str(e)

                # Check circuit breaker on error
                if circuit_breaker is not None:
                    progress = getattr(agent_state, "progress_estimate", 0.0)
                    cb_result = circuit_breaker.check(
                        iteration=iteration_num + 1,
                        progress=progress,
                        error=error_msg,
                        cost=accumulated_cost,
                    )

                    if cb_result.tripped:
                        logger.warning(
                            f"Circuit breaker tripped on error: {cb_result.reason.value}"
                        )
                        yield StreamingIterationResult(
                            iteration=iteration_num + 1,
                            max_iterations=max_iterations,
                            progress=progress,
                            is_complete=False,
                            is_final=True,
                            status="circuit_breaker_tripped",
                            current_phase="error",
                            message=f"Circuit breaker: {cb_result.message}",
                            error=error_msg,
                            stop_reason=cb_result.reason.value,
                        )
                        return

                # Yield error update
                yield StreamingIterationResult(
                    iteration=iteration_num + 1,
                    max_iterations=max_iterations,
                    progress=getattr(agent_state, "progress_estimate", 0.0),
                    is_complete=False,
                    is_final=circuit_breaker is None,  # Fatal if no circuit breaker
                    status="error",
                    current_phase="error",
                    message=f"Iteration failed: {error_msg[:100]}",
                    error=error_msg,
                )

                if circuit_breaker is None:
                    return  # Fatal error without circuit breaker
                continue  # Continue to next iteration with circuit breaker

            # Extract information from successful iteration
            actual_iteration = iteration_num + 1
            progress = getattr(agent_state, "progress_estimate", 0.0)

            # Get action information
            action_name: str | None = None
            action_summary: str | None = None
            if iteration_result.think_output and iteration_result.think_output.proposed_action:
                action = iteration_result.think_output.proposed_action
                action_name = action.name
                # Create a brief summary of the action arguments
                if hasattr(action, "arguments") and action.arguments:
                    args_str = str(action.arguments)
                    action_summary = args_str[:100] + "..." if len(args_str) > 100 else args_str

            # Get observation summary
            observation_summary: str | None = None
            if iteration_result.observe_output:
                obs = iteration_result.observe_output.observation
                observation_summary = obs[:150] + "..." if len(obs) > 150 else obs

            # Get step completion info
            step_completed = False
            if iteration_result.reflect_output:
                step_completed = iteration_result.reflect_output.step_completed

            # Get current plan step
            plan_step: str | None = None
            if agent_state.plan and agent_state.current_plan_step_index < len(agent_state.plan):
                plan_step = agent_state.plan[agent_state.current_plan_step_index]

            # Determine current phase (last completed phase)
            current_phase = "update"  # Default: full iteration completed
            if iteration_result.think_output and iteration_result.think_output.is_final_answer:
                current_phase = "think"  # Ended early with final answer

            # Check if task is complete
            is_complete = agent_state.is_finished
            if iteration_result.think_output and iteration_result.think_output.is_final_answer:
                is_complete = True

            # Build message
            if is_complete:
                message = agent_state.final_answer or "Task completed"
            elif action_name:
                message = f"Executed: {action_name}"
                if observation_summary:
                    message += f" → {observation_summary[:50]}"
            else:
                message = f"Iteration {actual_iteration} completed"

            # Check circuit breaker
            should_stop = False
            stop_reason: str | None = None

            if circuit_breaker is not None and not is_complete:
                cb_step_completed = None
                if iteration_result.reflect_output:
                    cb_step_completed = getattr(
                        iteration_result.reflect_output, "step_completed", None
                    )

                cb_result = circuit_breaker.check(
                    iteration=actual_iteration,
                    progress=progress,
                    error=None,
                    cost=accumulated_cost,
                    step_completed=cb_step_completed,
                )

                if cb_result.tripped:
                    should_stop = True
                    stop_reason = cb_result.reason.value
                    message = f"Circuit breaker: {cb_result.message}"
                    logger.warning(f"Circuit breaker tripped: {stop_reason}")

            # Check if UPDATE phase says to stop
            if (
                iteration_result.update_output
                and not iteration_result.update_output.should_continue
            ):
                should_stop = True
                if agent_state.awaiting_human_approval:
                    stop_reason = "human_approval_required"
                else:
                    stop_reason = "update_stopped"

            # Yield the iteration result
            yield StreamingIterationResult(
                iteration=actual_iteration,
                max_iterations=max_iterations,
                progress=progress,
                is_complete=is_complete,
                is_final=is_complete or should_stop,
                status="complete" if is_complete else ("stopped" if should_stop else "in_progress"),
                current_phase=current_phase,
                message=message,
                action_name=action_name,
                action_summary=action_summary,
                observation_summary=observation_summary,
                step_completed=step_completed,
                plan_step=plan_step,
                tokens_used=iteration_result.total_tokens_used,
                duration_ms=iteration_result.duration_ms,
                stop_reason=stop_reason,
            )

            if is_complete or should_stop:
                return

        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached without completion")
        yield StreamingIterationResult(
            iteration=max_iterations,
            max_iterations=max_iterations,
            progress=getattr(agent_state, "progress_estimate", 0.0),
            is_complete=False,
            is_final=True,
            status="max_iterations",
            current_phase="complete",
            message=f"Max iterations ({max_iterations}) reached",
            stop_reason="max_iterations",
        )

    def _build_history(self, agent_state: EnhancedAgentState) -> str:
        """
        Build history string from recent iterations.

        Args:
            agent_state: Current agent state

        Returns:
            Formatted history string
        """
        if not agent_state.iterations:
            return "No previous actions"

        # Get last 3 iterations
        recent_iterations = agent_state.iterations[-3:]

        history_parts = []
        for iteration in recent_iterations:
            if iteration.think_output and iteration.think_output.proposed_action:
                action = iteration.think_output.proposed_action
                history_parts.append(f"Iteration {iteration.iteration_number}: {action.name}")

                if iteration.observe_output:
                    # Truncate observation
                    obs = iteration.observe_output.observation
                    if len(obs) > 200:
                        obs = obs[:200] + "..."
                    history_parts.append(f"  Result: {obs}")

        return "\n".join(history_parts) if history_parts else "No previous actions"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["CognitiveCycle", "StreamingIterationResult"]
