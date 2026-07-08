# src/llmcore/agents/cognitive/phases/cycle.py
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

import inspect
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..models import (
    ActInput,
    ConfidenceLevel,
    CycleIteration,
    EnhancedAgentState,
    ObserveInput,
    # Phase inputs
    PerceiveInput,
    PlanInput,
    PlanStepSpec,
    ReflectInput,
    # Enums
    TerminationReason,
    ThinkInput,
    UpdateInput,
    ValidateInput,
    ValidateOutput,
    ValidationResult,
)
from ._prompting import messages_from_registry
from .act import act_phase
from .observe import observe_phase
from .perceive import perceive_phase
from .plan import plan_phase
from .reflect import reflect_phase
from .think import (
    _extract_native_tool_call,
    _finish_answer_acceptable,
    _finish_answer_from_arguments,
    _finish_tool_names,
    think_phase,
)
from .update import update_phase
from .validate import deterministic_precheck, validate_phase

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
        termination_reason: Why the run terminated (TerminationReason value),
            populated on every final update; ``stop_reason`` mirrors it where
            no more specific legacy value applies
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
    termination_reason: str | None = None


# =============================================================================
# CONVERGENCE HELPERS
# =============================================================================


def _resolve_convergence(agents_config: Any) -> Any:
    """Return a real ``ConvergenceConfig`` from a config object.

    Mock/legacy config objects without a typed ``convergence`` section get
    the defaults — the convergence invariant must not depend on duck-typed
    attributes evaluating truthy.
    """
    from ....config.agents_config import ConvergenceConfig

    convergence = getattr(agents_config, "convergence", None)
    if isinstance(convergence, ConvergenceConfig):
        return convergence
    return ConvergenceConfig()


def _provider_accepts_tool_choice(provider: Any) -> bool:
    """Feature-detect ``tool_choice`` support on ``provider.chat_completion``."""
    try:
        signature = inspect.signature(provider.chat_completion)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if "tool_choice" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


# =============================================================================
# COGNITIVE CYCLE ORCHESTRATOR
# =============================================================================


class CognitiveCycle:
    """
    Orchestrates the complete 8-phase cognitive cycle.

    The CognitiveCycle manages the execution of all phases in sequence,
    handles errors, coordinates state updates, and provides a clean
    interface for agent execution.

    Context Synthesis:
        The cycle supports optional ``ContextSynthesizer`` for sophisticated
        multi-source context assembly in the PERCEIVE phase. When provided,
        the synthesizer gathers context from multiple sources (goals, recent
        history, RAG, skills, episodic memory) in parallel, scores and
        prioritizes chunks, and fits them into the token budget.

        If no synthesizer is provided, the cycle falls back to direct
        ``MemoryManager`` retrieval for backward compatibility.

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
        >>>
        >>> # With context synthesis (recommended for autonomous operation)
        >>> from llmcore.agents.cognitive.phases import create_default_synthesizer
        >>> synthesizer = create_default_synthesizer(
        ...     goal_manager=goal_manager,
        ...     skill_loader=skill_loader,
        ... )
        >>> cycle = CognitiveCycle(
        ...     provider_manager=provider_manager,
        ...     memory_manager=memory_manager,
        ...     storage_manager=storage_manager,
        ...     tool_manager=tool_manager,
        ...     context_synthesizer=synthesizer,
        ... )

    Attributes:
        provider_manager: Provider manager for LLM calls
        memory_manager: Memory manager for context
        storage_manager: Storage manager for episodic memory
        tool_manager: Tool manager for actions
        prompt_registry: Prompt registry (required, grimoire adapter)
        context_synthesizer: Optional ContextSynthesizer for PERCEIVE phase
    """

    def __init__(
        self,
        provider_manager: "ProviderManager",
        memory_manager: "MemoryManager",
        storage_manager: "StorageManager",
        tool_manager: "ToolManager",
        prompt_registry: Any | None = None,
        tracer: Any | None = None,
        context_synthesizer: Any | None = None,
        agents_config: Optional["AgentsConfig"] = None,
        max_history_iterations: int = 3,
        max_history_observation_chars: int = 3000,
    ):
        """
        Initialize the cognitive cycle orchestrator.

        Args:
            provider_manager: Provider manager for LLM calls.
            memory_manager: Memory manager for context retrieval.
            storage_manager: Storage manager for episodic memory.
            tool_manager: Tool manager for actions.
            prompt_registry: Prompt registry (REQUIRED as of 0.52.0 — the
                grimoire-backed adapter supplying every phase prompt).
            tracer: Optional OpenTelemetry tracer.
            context_synthesizer: Optional ContextSynthesizer for sophisticated
                multi-source context assembly in the PERCEIVE phase. When
                provided, enables synthesis mode with prioritized context
                gathering from goals, recent history, RAG, skills, and
                episodic memory. When None, falls back to direct
                MemoryManager retrieval.
            agents_config: Optional agent system configuration. Defaults to
                AgentsConfig() for direct CognitiveCycle use.

        Raises:
            ValueError: When ``prompt_registry`` is None — the grimoire
                control plane is mandatory (0.52.0).
        """
        if prompt_registry is None:
            raise ValueError(
                "prompt_registry is required (0.52.0): llmcore agent prompts "
                "come from the grimoire control plane"
            )
        self.provider_manager = provider_manager
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager
        self.prompt_registry = prompt_registry
        self.tracer = tracer
        self.context_synthesizer = context_synthesizer
        if agents_config is None:
            from ....config.agents_config import AgentsConfig

            agents_config = AgentsConfig()
        self.agents_config = agents_config
        self.max_history_iterations = max(1, int(max_history_iterations))
        self.max_history_observation_chars = max(1, int(max_history_observation_chars))

    async def run_iteration(
        self,
        agent_state: EnhancedAgentState,
        session_id: str,
        sandbox: Optional["SandboxProvider"] = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        skip_validation: bool = False,
        approval_callback: Callable[[str], bool] | None = None,
        remaining_iterations: int | None = None,
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
            remaining_iterations: Iterations left in the run's budget; surfaced
                to THINK as ``remaining_steps`` (None = unlimited/unknown)

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
                    context_synthesizer=self.context_synthesizer,
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
                current_step_spec = _current_plan_step_spec(agent_state)

                tool_inventory_config = self.agents_config.tool_inventory
                if (
                    tool_inventory_config.enabled
                    and callable(getattr(type(self.tool_manager), "get_tool_inventory", None))
                ):
                    available_tools = self.tool_manager.get_tool_inventory(
                        max_description_chars=tool_inventory_config.max_description_chars,
                        include_parameters=tool_inventory_config.include_parameters,
                    )
                else:
                    available_tools = [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in self.tool_manager.get_tool_definitions()
                    ]

                think_input = ThinkInput(
                    goal=agent_state.goal,
                    current_step=current_step,
                    current_step_spec=current_step_spec,
                    history=self._build_history(agent_state),
                    context="\n".join(iteration.perceive_output.retrieved_context),
                    available_tools=available_tools,
                    remaining_steps=remaining_iterations,
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
                    agents_config=self.agents_config,
                )

                # If final answer, skip remaining phases
                if iteration.think_output.is_final_answer:
                    logger.info("Final answer provided, completing iteration")
                    iteration.update_token_totals_from_phases()
                    agent_state.complete_iteration(success=True)
                    return iteration

                # ============================================================
                # Phase 4: VALIDATE
                # ============================================================
                if iteration.think_output.proposed_action:
                    if skip_validation:
                        # skip_validation skips only the LLM judge (2.3): the
                        # deterministic guards (registry membership + dangerous
                        # patterns) still run unless explicitly disabled.
                        precheck_output = None
                        validation_config = getattr(self.agents_config, "validation", None)
                        deterministic_guards = getattr(
                            validation_config, "deterministic_guards", True
                        )
                        if not isinstance(deterministic_guards, bool):
                            deterministic_guards = True
                        if deterministic_guards:
                            precheck_output = deterministic_precheck(
                                iteration.think_output.proposed_action,
                                self.tool_manager,
                                goal=agent_state.goal,
                                reasoning=iteration.think_output.thought,
                            )
                        if precheck_output is not None:
                            # A guard fired: USE its output, mirroring
                            # validate_phase's state side-effects.
                            logger.info(
                                "Deterministic guard fired under skip_validation: %s",
                                precheck_output.result.value,
                            )
                            agent_state.pending_validation = ValidateInput(
                                goal=agent_state.goal,
                                proposed_action=iteration.think_output.proposed_action,
                                reasoning=iteration.think_output.thought,
                            )
                            agent_state.validation_history.append(precheck_output)
                            if precheck_output.requires_human_approval:
                                agent_state.awaiting_human_approval = True
                                agent_state.pending_approval_prompt = (
                                    precheck_output.approval_prompt
                                )
                            iteration.validate_output = precheck_output
                        else:
                            # Clean action: only the LLM judge is skipped.
                            logger.info("Skipping validation (auto-approve enabled)")
                            iteration.validate_output = ValidateOutput(
                                result=ValidationResult.APPROVED,
                                confidence=ConfidenceLevel.HIGH,
                                concerns=[],
                                suggestions=["LLM validation skipped"],
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
                            tool_manager=self.tool_manager,
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
                        agents_config=self.agents_config,
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
                iteration.update_token_totals_from_phases()
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

        convergence = _resolve_convergence(agents_config)
        # Convergence interventions need at least one normal iteration ahead
        # of the finalize slot. Single-iteration budgets (wairu's bounded
        # ``run(max_iterations=1)`` outer-loop driving pattern) keep legacy
        # semantics — the outer driver owns convergence there.
        convergence_active = max_iterations > max(1, convergence.finalize_when_remaining)
        finalize_attempted = False

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

            # Convergence (2.2): spend the last budgeted iteration on a
            # finalize synthesis pass instead of a normal iteration.
            remaining_iterations = max_iterations - iteration_num
            if (
                convergence_active
                and convergence.forced_finalize_enabled
                and not finalize_attempted
                and convergence.finalize_when_remaining > 0
                and remaining_iterations <= convergence.finalize_when_remaining
            ):
                finalize_attempted = True
                logger.info(
                    "Iteration budget nearly exhausted (%d remaining) — forcing finalization",
                    remaining_iterations,
                )
                await self._force_finalize(
                    agent_state,
                    reason=TerminationReason.FORCED_FINALIZE,
                    convergence=convergence,
                    provider_name=provider_name,
                    model_name=model_name,
                )
                break

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
                    remaining_iterations=remaining_iterations,
                )
                actual_iterations = iteration_num + 1
                last_error = None  # Clear error on success

                # Track cost if available
                if hasattr(iteration, "total_cost") and iteration.total_cost:
                    accumulated_cost += iteration.total_cost

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
                        agent_state.termination_reason = TerminationReason.CIRCUIT_BREAKER.value
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
                        agent_state.termination_reason = (
                            TerminationReason.HUMAN_APPROVAL_REQUIRED.value
                        )
                    else:
                        stop_reason = "update_stopped"
                        if not agent_state.is_finished and not agent_state.termination_reason:
                            agent_state.termination_reason = TerminationReason.UPDATE_STOPPED.value
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
                        agent_state.termination_reason = TerminationReason.CIRCUIT_BREAKER.value
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
                    agent_state.termination_reason = TerminationReason.ERROR.value
                    return f"Task failed: {e!s}"

        # Check if task completed during the last iteration
        if agent_state.is_finished:
            return agent_state.final_answer or "Task completed"

        # Convergence (2.2): the loop must not exit un-converged on the
        # max-iterations / update-stopped paths — synthesize an answer from
        # the accumulated observations. Human-approval and circuit-breaker
        # exits (and hard errors) remain the only un-converged exits.
        if (
            convergence_active
            and convergence.synthesis_on_exhaustion
            and not finalize_attempted
            and (not stopped_early or stop_reason == "update_stopped")
        ):
            finalize_attempted = True
            logger.info("Loop exited un-finished — attempting synthesis fallback")
            if await self._force_finalize(
                agent_state,
                reason=TerminationReason.SYNTHESIS_FALLBACK,
                convergence=convergence,
                provider_name=provider_name,
                model_name=model_name,
            ):
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
        if not agent_state.termination_reason:
            agent_state.termination_reason = TerminationReason.MAX_ITERATIONS.value
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

        convergence = _resolve_convergence(agents_config)
        # See run_until_complete: single-iteration budgets (wairu's bounded
        # outer-loop driving pattern) keep legacy semantics.
        convergence_active = max_iterations > max(1, convergence.finalize_when_remaining)
        finalize_attempted = False

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
                    stop_reason=agent_state.termination_reason,
                    termination_reason=agent_state.termination_reason,
                )
                return

            # Convergence (2.2): spend the last budgeted iteration on a
            # finalize synthesis pass instead of a normal iteration.
            remaining_iterations = max_iterations - iteration_num
            if (
                convergence_active
                and convergence.forced_finalize_enabled
                and not finalize_attempted
                and convergence.finalize_when_remaining > 0
                and remaining_iterations <= convergence.finalize_when_remaining
            ):
                finalize_attempted = True
                logger.info(
                    "Iteration budget nearly exhausted (%d remaining) — forcing finalization",
                    remaining_iterations,
                )
                finalized = await self._force_finalize(
                    agent_state,
                    reason=TerminationReason.FORCED_FINALIZE,
                    convergence=convergence,
                    provider_name=provider_name,
                    model_name=model_name,
                )
                if finalized:
                    yield StreamingIterationResult(
                        iteration=iteration_num,
                        max_iterations=max_iterations,
                        progress=1.0,
                        is_complete=True,
                        is_final=True,
                        status="complete",
                        current_phase="finalize",
                        message=agent_state.final_answer or "Task completed",
                        stop_reason=agent_state.termination_reason,
                        termination_reason=agent_state.termination_reason,
                    )
                    return
                break  # fall through to the max-iterations terminal update

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
                    remaining_iterations=remaining_iterations,
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
                        agent_state.termination_reason = TerminationReason.CIRCUIT_BREAKER.value
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
                            termination_reason=agent_state.termination_reason,
                        )
                        return

                # Yield error update
                if circuit_breaker is None:
                    agent_state.termination_reason = TerminationReason.ERROR.value
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
                    stop_reason=agent_state.termination_reason
                    if circuit_breaker is None
                    else None,
                    termination_reason=agent_state.termination_reason
                    if circuit_breaker is None
                    else None,
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
                    agent_state.termination_reason = TerminationReason.CIRCUIT_BREAKER.value
                    logger.warning(f"Circuit breaker tripped: {stop_reason}")

            # Check if UPDATE phase says to stop
            if (
                iteration_result.update_output
                and not iteration_result.update_output.should_continue
            ):
                should_stop = True
                if agent_state.awaiting_human_approval:
                    stop_reason = "human_approval_required"
                    if not is_complete:
                        agent_state.termination_reason = (
                            TerminationReason.HUMAN_APPROVAL_REQUIRED.value
                        )
                else:
                    stop_reason = "update_stopped"
                    if not is_complete and not agent_state.termination_reason:
                        agent_state.termination_reason = TerminationReason.UPDATE_STOPPED.value

            # Convergence (2.2): an update-stopped exit must not leave the
            # run un-converged — synthesize from the accumulated work.
            if (
                convergence_active
                and should_stop
                and not is_complete
                and stop_reason == "update_stopped"
                and convergence.synthesis_on_exhaustion
                and not finalize_attempted
            ):
                finalize_attempted = True
                logger.info("UPDATE stopped the loop un-finished — attempting synthesis fallback")
                if await self._force_finalize(
                    agent_state,
                    reason=TerminationReason.SYNTHESIS_FALLBACK,
                    convergence=convergence,
                    provider_name=provider_name,
                    model_name=model_name,
                ):
                    is_complete = True
                    progress = 1.0
                    message = agent_state.final_answer or "Task completed"
                    stop_reason = agent_state.termination_reason

            is_final = is_complete or should_stop

            # Yield the iteration result
            yield StreamingIterationResult(
                iteration=actual_iteration,
                max_iterations=max_iterations,
                progress=progress,
                is_complete=is_complete,
                is_final=is_final,
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
                stop_reason=stop_reason
                or (agent_state.termination_reason if is_final else None),
                termination_reason=agent_state.termination_reason if is_final else None,
            )

            if is_complete or should_stop:
                return

        # Convergence (2.2): loop exhausted without an answer — synthesis
        # fallback before conceding to the legacy max-iterations terminal.
        if (
            convergence_active
            and convergence.synthesis_on_exhaustion
            and not finalize_attempted
            and not agent_state.is_finished
        ):
            finalize_attempted = True
            logger.info("Loop exhausted un-finished — attempting synthesis fallback")
            if await self._force_finalize(
                agent_state,
                reason=TerminationReason.SYNTHESIS_FALLBACK,
                convergence=convergence,
                provider_name=provider_name,
                model_name=model_name,
            ):
                yield StreamingIterationResult(
                    iteration=max_iterations,
                    max_iterations=max_iterations,
                    progress=1.0,
                    is_complete=True,
                    is_final=True,
                    status="complete",
                    current_phase="finalize",
                    message=agent_state.final_answer or "Task completed",
                    stop_reason=agent_state.termination_reason,
                    termination_reason=agent_state.termination_reason,
                )
                return

        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached without completion")
        if not agent_state.termination_reason:
            agent_state.termination_reason = TerminationReason.MAX_ITERATIONS.value
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
            termination_reason=agent_state.termination_reason,
        )

    async def _force_finalize(
        self,
        agent_state: EnhancedAgentState,
        *,
        reason: TerminationReason,
        convergence: Any | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> bool:
        """Synthesize a final answer when the loop cannot exit converged.

        Renders the ``finalize_prompt`` template (vars: goal/history/context/
        reason), calls the provider with only the ``finish`` tool —
        ``tool_choice="required"`` where the provider's ``chat_completion``
        signature accepts it — and parses either a native finish call or the
        full response text as the final answer.

        NEVER raises in the exhaustion position: on any failure the state is
        left un-finished with ``termination_reason=ERROR`` and the caller
        falls through to the legacy terminal paths.

        Args:
            agent_state: State to finalize (mutated on success).
            reason: FORCED_FINALIZE (in-loop) or SYNTHESIS_FALLBACK (exhaustion).
            convergence: Resolved ConvergenceConfig; defaults from
                ``self.agents_config`` when omitted.
            provider_name: Optional provider override.
            model_name: Optional model override.

        Returns:
            True when a final answer was set on the state.
        """
        if convergence is None:
            convergence = _resolve_convergence(self.agents_config)

        logger.info("Forcing finalization (%s)", reason.value)
        try:
            # Raised history bounds: the synthesis pass is the last chance to
            # use the run's observations, so give it more than THINK's default.
            history = self._build_history(
                agent_state,
                max_iterations=max(5, self.max_history_iterations),
                max_observation_chars=max(4000, self.max_history_observation_chars),
            )
            messages = messages_from_registry(
                self.prompt_registry,
                "finalize_prompt",
                {
                    "goal": agent_state.goal,
                    "history": history,
                    "context": agent_state.context or "",
                    "reason": reason.value,
                },
            )

            finish_tools = self._finish_tool_definitions(convergence)
            provider = self.provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            call_kwargs: dict[str, Any] = {}
            if finish_tools and _provider_accepts_tool_choice(provider):
                call_kwargs["tool_choice"] = "required"

            if callable(getattr(type(self.provider_manager), "chat_completion_with_retry", None)):
                response = await self.provider_manager.chat_completion_with_retry(
                    provider,
                    context=messages,
                    model=target_model,
                    stream=False,
                    tools=finish_tools or None,
                    tracer=self.tracer,
                    operation="cognitive.finalize",
                    temperature=convergence.finalize_temperature,
                    **call_kwargs,
                )
            else:
                response = await provider.chat_completion(
                    context=messages,
                    model=target_model,
                    stream=False,
                    tools=finish_tools or None,
                    temperature=convergence.finalize_temperature,
                    **call_kwargs,
                )

            response_content = provider.extract_response_content(response)

            # Prefer a native finish call; fall back to the full text.
            answer = ""
            native_tool_call = _extract_native_tool_call(
                response if isinstance(response, dict) else None
            )
            if native_tool_call is not None and native_tool_call.name in _finish_tool_names(
                convergence
            ):
                answer = _finish_answer_from_arguments(native_tool_call.arguments)
            if not answer.strip():
                answer = str(response_content or "").strip()

            if not _finish_answer_acceptable(answer, convergence):
                logger.error("Forced finalization produced an empty answer (%s)", reason.value)
                agent_state.termination_reason = TerminationReason.ERROR.value
                return False

            agent_state.final_answer = answer
            agent_state.is_finished = True
            agent_state.termination_reason = reason.value
            logger.info("Forced finalization succeeded (%s)", reason.value)
            return True

        except Exception as exc:
            # The exhaustion position must never raise — record the failure
            # and let the caller fall through to the legacy terminal paths.
            logger.error("Forced finalization failed (%s): %s", reason.value, exc, exc_info=True)
            agent_state.termination_reason = TerminationReason.ERROR.value
            return False

    def _finish_tool_definitions(self, convergence: Any | None = None) -> list[Any]:
        """Return the finish tool definition(s) for the finalize provider call."""
        names = _finish_tool_names(convergence)
        if not names:
            return []
        try:
            definitions = self.tool_manager.get_tool_definitions(names)
        except TypeError:  # legacy managers without subset support
            try:
                definitions = [
                    tool
                    for tool in self.tool_manager.get_tool_definitions()
                    if str(getattr(tool, "name", "")) in names
                ]
            except Exception:
                logger.debug("Unable to load finish tool definitions", exc_info=True)
                return []
        except Exception:
            logger.debug("Unable to load finish tool definitions", exc_info=True)
            return []
        return list(definitions) if isinstance(definitions, list) else []

    def _build_history(
        self,
        agent_state: EnhancedAgentState,
        *,
        max_iterations: int | None = None,
        max_observation_chars: int | None = None,
    ) -> str:
        """
        Build a bounded JSON history summary from recent iterations.

        Tool results and observations are truncated before serialization, so the
        returned value remains valid JSON and can be parsed by downstream callers.
        The instance bounds apply unless a caller (e.g. ``_force_finalize``)
        raises them explicitly.
        """
        effective_iterations = (
            self.max_history_iterations if max_iterations is None else max(1, int(max_iterations))
        )
        effective_observation_chars = (
            self.max_history_observation_chars
            if max_observation_chars is None
            else max(1, int(max_observation_chars))
        )
        summaries = agent_state.recent_history_summaries(
            max_iterations=effective_iterations,
            max_observation_chars=effective_observation_chars,
            max_tool_result_chars=effective_observation_chars,
        )
        if not summaries:
            return "No previous actions"

        return json.dumps({"recent_iterations": summaries}, ensure_ascii=False)


def _current_plan_step_spec(agent_state: EnhancedAgentState) -> PlanStepSpec | None:
    """Return the structured spec for the current plan step when available."""
    raw_specs = agent_state.metadata.get("plan_step_specs")
    if not isinstance(raw_specs, list):
        return None

    step_index = agent_state.current_plan_step_index
    if step_index < 0 or step_index >= len(raw_specs):
        return None

    raw_spec = raw_specs[step_index]
    try:
        return PlanStepSpec.model_validate(raw_spec)
    except Exception:
        logger.debug("Ignoring invalid structured plan step at index %s", step_index)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["CognitiveCycle", "StreamingIterationResult"]
