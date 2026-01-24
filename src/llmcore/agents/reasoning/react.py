# src/llmcore/agents/reasoning/react.py
"""
ReAct Reasoner Implementation.

Implements the ReAct (Reasoning + Acting) framework from Yao et al. 2022.
This is the primary reasoning framework for tool-using agents.

Pattern:
    THOUGHT → ACTION → OBSERVATION → (repeat until done)

Research Reference:
    Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
    - Interleaved reasoning + acting outperforms either alone
    - Dominant pattern for knowledge-intensive tasks
    - Achieves state-of-the-art on HotPotQA, FEVER, ALFWorld

Usage:
    from llmcore.agents.reasoning import ReActReasoner

    reasoner = ReActReasoner(llm_provider)
    result = await reasoner.reason(
        goal="Search for log files in /var/log",
        context=agent_context
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ConfigDict = dict

    def Field(*args, **kwargs):
        return kwargs.get("default")


if TYPE_CHECKING:
    from llmcore.agents.activities.loop import ActivityLoop
    from llmcore.agents.activities.schema import ActivityRequest, ActivityResult
    from llmcore.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ReActStep(str, Enum):
    """Types of steps in ReAct reasoning."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


class ReActStatus(str, Enum):
    """Status of ReAct reasoning."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    EARLY_TERMINATION = "early_termination"


@dataclass
class ReActThought:
    """A thought step in ReAct reasoning."""

    content: str
    reasoning_type: str = "general"  # general, analysis, planning, reflection
    confidence: float = 0.5
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActAction:
    """An action step in ReAct reasoning."""

    activity: str
    parameters: Dict[str, Any]
    reason: str = ""
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActObservation:
    """An observation step in ReAct reasoning."""

    content: str
    success: bool
    activity: str = ""
    duration_ms: int = 0
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActTrajectory:
    """Complete reasoning trajectory."""

    steps: List[Union[ReActThought, ReActAction, ReActObservation]] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: ReActStatus = ReActStatus.IN_PROGRESS
    iterations: int = 0
    total_duration_ms: int = 0
    total_actions: int = 0
    successful_actions: int = 0

    def add_thought(self, thought: ReActThought) -> None:
        """Add a thought step."""
        self.steps.append(thought)

    def add_action(self, action: ReActAction) -> None:
        """Add an action step."""
        self.steps.append(action)
        self.total_actions += 1

    def add_observation(self, observation: ReActObservation) -> None:
        """Add an observation step."""
        self.steps.append(observation)
        if observation.success:
            self.successful_actions += 1

    def get_thoughts(self) -> List[ReActThought]:
        """Get all thought steps."""
        return [s for s in self.steps if isinstance(s, ReActThought)]

    def get_actions(self) -> List[ReActAction]:
        """Get all action steps."""
        return [s for s in self.steps if isinstance(s, ReActAction)]

    def get_observations(self) -> List[ReActObservation]:
        """Get all observation steps."""
        return [s for s in self.steps if isinstance(s, ReActObservation)]

    def format_history(self, max_steps: Optional[int] = None) -> str:
        """Format trajectory as history for LLM context."""
        lines = []
        steps = self.steps[-max_steps:] if max_steps else self.steps

        for step in steps:
            if isinstance(step, ReActThought):
                lines.append(f"Thought: {step.content}")
            elif isinstance(step, ReActAction):
                lines.append(f"Action: {step.activity}({step.parameters})")
                if step.reason:
                    lines.append(f"  Reason: {step.reason}")
            elif isinstance(step, ReActObservation):
                status = "✓" if step.success else "✗"
                lines.append(f"Observation [{status}]: {step.content[:500]}")

        return "\n".join(lines)

    @property
    def success_rate(self) -> float:
        """Calculate action success rate."""
        if self.total_actions == 0:
            return 1.0
        return self.successful_actions / self.total_actions


if PYDANTIC_AVAILABLE:

    class ReActResult(BaseModel):
        """Result of ReAct reasoning."""

        success: bool = Field(description="Whether reasoning succeeded")
        final_answer: Optional[str] = Field(None, description="Final answer if successful")
        status: ReActStatus = Field(description="Termination status")
        trajectory: Optional[Dict[str, Any]] = Field(None, description="Full reasoning trajectory")
        iterations: int = Field(0, description="Number of iterations")
        duration_ms: int = Field(0, description="Total duration in ms")
        error: Optional[str] = Field(None, description="Error message if failed")

        model_config = ConfigDict(use_enum_values=True)
else:

    @dataclass
    class ReActResult:
        """Result of ReAct reasoning."""

        success: bool
        final_answer: Optional[str] = None
        status: ReActStatus = ReActStatus.FAILURE
        trajectory: Optional[Dict[str, Any]] = None
        iterations: int = 0
        duration_ms: int = 0
        error: Optional[str] = None


# =============================================================================
# ReAct Prompt Templates
# =============================================================================


REACT_SYSTEM_PROMPT = """You are an AI agent that reasons step-by-step to solve tasks.

You follow the ReAct pattern:
1. THOUGHT: Analyze the situation, plan your approach, or reflect on observations
2. ACTION: Execute an activity to gather information or make progress
3. OBSERVATION: (Provided by the system) Results of your action

You MUST structure your response with explicit markers:

THOUGHT: [Your reasoning here]

Then either take an action using XML format:
<activity_request>
    <activity>[activity_name]</activity>
    <parameters>
        <param_name>param_value</param_name>
    </parameters>
    <reason>[Why this action]</reason>
</activity_request>

Or provide a final answer:
<final_answer>[Your complete answer]</final_answer>

Guidelines:
- Think before each action
- Use observations to inform next steps
- Stop when the goal is achieved
- Admit if you cannot complete the task"""

REACT_USER_PROMPT = """Goal: {goal}

{available_activities}

{context}

{history}

Now, think step-by-step and take the next action (or provide final answer if done)."""


# =============================================================================
# ReAct Reasoner Implementation
# =============================================================================


class ReActConfig:
    """Configuration for ReAct reasoning."""

    def __init__(
        self,
        max_iterations: int = 10,
        max_thoughts_per_iteration: int = 1,
        max_actions_per_iteration: int = 3,
        timeout_seconds: float = 300.0,
        include_history_steps: int = 10,
        temperature: float = 0.7,
        stop_on_final_answer: bool = True,
        stop_on_repeated_failure: bool = True,
        max_repeated_failures: int = 3,
    ):
        self.max_iterations = max_iterations
        self.max_thoughts_per_iteration = max_thoughts_per_iteration
        self.max_actions_per_iteration = max_actions_per_iteration
        self.timeout_seconds = timeout_seconds
        self.include_history_steps = include_history_steps
        self.temperature = temperature
        self.stop_on_final_answer = stop_on_final_answer
        self.stop_on_repeated_failure = stop_on_repeated_failure
        self.max_repeated_failures = max_repeated_failures


class ReActReasoner:
    """
    ReAct reasoning framework implementation.

    Implements the Thought → Action → Observation loop for
    tool-using language agents.

    Args:
        llm_provider: LLM provider for generating thoughts/actions
        activity_loop: Activity loop for executing actions
        config: ReAct configuration
    """

    def __init__(
        self,
        llm_provider: Optional["BaseLLMProvider"] = None,
        activity_loop: Optional["ActivityLoop"] = None,
        config: Optional[ReActConfig] = None,
    ):
        self.llm_provider = llm_provider
        self.activity_loop = activity_loop
        self.config = config or ReActConfig()

        # Tracking
        self._current_trajectory: Optional[ReActTrajectory] = None
        self._consecutive_failures = 0
        self._last_error: Optional[str] = None

    async def reason(
        self,
        goal: str,
        context: Optional[str] = None,
        available_activities: Optional[str] = None,
        initial_observations: Optional[List[str]] = None,
        on_step: Optional[
            Callable[[Union[ReActThought, ReActAction, ReActObservation]], None]
        ] = None,
    ) -> ReActResult:
        """
        Execute ReAct reasoning loop.

        Args:
            goal: The goal to achieve
            context: Additional context for the task
            available_activities: Formatted list of available activities
            initial_observations: Initial observations to include
            on_step: Callback for each reasoning step

        Returns:
            ReActResult with final answer or error
        """
        start_time = time.time()
        trajectory = ReActTrajectory()
        self._current_trajectory = trajectory
        self._consecutive_failures = 0

        # Add initial observations
        if initial_observations:
            for obs in initial_observations:
                trajectory.add_observation(
                    ReActObservation(
                        content=obs,
                        success=True,
                        activity="initial",
                        iteration=0,
                    )
                )

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout_seconds:
                    trajectory.status = ReActStatus.TIMEOUT
                    return self._create_result(
                        trajectory=trajectory,
                        success=False,
                        error="Reasoning timeout",
                        start_time=start_time,
                    )

                # Check repeated failures
                if self.config.stop_on_repeated_failure:
                    if self._consecutive_failures >= self.config.max_repeated_failures:
                        trajectory.status = ReActStatus.EARLY_TERMINATION
                        return self._create_result(
                            trajectory=trajectory,
                            success=False,
                            error=f"Too many consecutive failures: {self._last_error}",
                            start_time=start_time,
                        )

                trajectory.iterations = iteration

                # Generate thought and action
                llm_output = await self._generate_step(
                    goal=goal,
                    context=context,
                    available_activities=available_activities,
                    trajectory=trajectory,
                )

                # Parse thought
                thought = self._extract_thought(llm_output, iteration)
                if thought:
                    trajectory.add_thought(thought)
                    if on_step:
                        on_step(thought)

                # Check for final answer
                final_answer = self._extract_final_answer(llm_output)
                if final_answer and self.config.stop_on_final_answer:
                    trajectory.final_answer = final_answer
                    trajectory.status = ReActStatus.SUCCESS
                    return self._create_result(
                        trajectory=trajectory,
                        success=True,
                        final_answer=final_answer,
                        start_time=start_time,
                    )

                # Execute actions
                observation = await self._execute_actions(
                    llm_output=llm_output,
                    iteration=iteration,
                    trajectory=trajectory,
                    on_step=on_step,
                )

                # Handle observation
                if observation:
                    trajectory.add_observation(observation)
                    if on_step:
                        on_step(observation)

                    # Track failures
                    if not observation.success:
                        self._consecutive_failures += 1
                        self._last_error = observation.content[:200]
                    else:
                        self._consecutive_failures = 0
                else:
                    # No action taken - might be stuck
                    logger.warning(f"Iteration {iteration}: No action taken")

            # Max iterations reached
            trajectory.status = ReActStatus.MAX_ITERATIONS
            return self._create_result(
                trajectory=trajectory,
                success=False,
                error="Maximum iterations reached",
                start_time=start_time,
            )

        except Exception as e:
            logger.exception(f"ReAct reasoning error: {e}")
            trajectory.status = ReActStatus.FAILURE
            return self._create_result(
                trajectory=trajectory,
                success=False,
                error=str(e),
                start_time=start_time,
            )

    async def _generate_step(
        self,
        goal: str,
        context: Optional[str],
        available_activities: Optional[str],
        trajectory: ReActTrajectory,
    ) -> str:
        """Generate the next thought/action from LLM."""
        if not self.llm_provider:
            raise ValueError("LLM provider required for ReAct reasoning")

        # Build prompt
        history = trajectory.format_history(self.config.include_history_steps)

        user_prompt = REACT_USER_PROMPT.format(
            goal=goal,
            available_activities=available_activities or "No specific activities available.",
            context=context or "No additional context.",
            history=f"Previous steps:\n{history}" if history else "Starting fresh.",
        )

        # Call LLM
        response = await self.llm_provider.chat_async(
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
        )

        # Extract content from response
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict):
            return response.get("content", str(response))
        else:
            return str(response)

    async def _execute_actions(
        self,
        llm_output: str,
        iteration: int,
        trajectory: ReActTrajectory,
        on_step: Optional[Callable] = None,
    ) -> Optional[ReActObservation]:
        """Execute actions from LLM output."""
        if not self.activity_loop:
            logger.warning("No activity loop configured, cannot execute actions")
            return None

        # Process through activity loop
        loop_result = await self.activity_loop.process_output(llm_output)

        # Convert to observations
        if loop_result.executions:
            exec_info = loop_result.executions[0]  # Primary execution

            # Create action record
            action = ReActAction(
                activity=exec_info.request.activity,
                parameters=exec_info.request.parameters,
                reason=exec_info.request.reason or "",
                iteration=iteration,
            )
            trajectory.add_action(action)
            if on_step:
                on_step(action)

            # Create observation
            return ReActObservation(
                content=loop_result.observation,
                success=exec_info.success,
                activity=exec_info.request.activity,
                duration_ms=exec_info.result.duration_ms if exec_info.result else 0,
                iteration=iteration,
            )

        return None

    def _extract_thought(self, llm_output: str, iteration: int) -> Optional[ReActThought]:
        """Extract thought from LLM output."""
        import re

        # Look for explicit THOUGHT: marker
        thought_match = re.search(
            r"(?:THOUGHT|Thought|thought)[:\s]+(.+?)(?=\n(?:ACTION|Action|action|<activity_request>|<final_answer>)|$)",
            llm_output,
            re.DOTALL | re.IGNORECASE,
        )

        if thought_match:
            content = thought_match.group(1).strip()
            if content:
                return ReActThought(
                    content=content,
                    iteration=iteration,
                )

        # If no explicit marker, look for reasoning before action
        action_start = llm_output.find("<activity_request>")
        final_start = llm_output.find("<final_answer>")

        marker_pos = min(
            action_start if action_start >= 0 else len(llm_output),
            final_start if final_start >= 0 else len(llm_output),
        )

        if marker_pos > 20:  # At least some content before action
            content = llm_output[:marker_pos].strip()
            # Remove any remaining markers
            content = re.sub(r"^(THOUGHT|Thought|thought)[:\s]*", "", content)
            if content and len(content) > 10:
                return ReActThought(
                    content=content,
                    iteration=iteration,
                )

        return None

    def _extract_final_answer(self, llm_output: str) -> Optional[str]:
        """Extract final answer from LLM output."""
        import re

        # XML-style final answer
        match = re.search(
            r"<final_answer>(.*?)</final_answer>",
            llm_output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Marker-style final answer
        match = re.search(
            r"FINAL\s*ANSWER[:\s]+(.+?)(?:\n\n|$)",
            llm_output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _create_result(
        self,
        trajectory: ReActTrajectory,
        success: bool,
        start_time: float,
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ReActResult:
        """Create ReActResult from trajectory."""
        duration_ms = int((time.time() - start_time) * 1000)
        trajectory.total_duration_ms = duration_ms

        return ReActResult(
            success=success,
            final_answer=final_answer or trajectory.final_answer,
            status=trajectory.status,
            trajectory={
                "steps": len(trajectory.steps),
                "thoughts": len(trajectory.get_thoughts()),
                "actions": trajectory.total_actions,
                "successful_actions": trajectory.successful_actions,
                "success_rate": trajectory.success_rate,
                "history": trajectory.format_history(),
            },
            iterations=trajectory.iterations,
            duration_ms=duration_ms,
            error=error,
        )

    @property
    def current_trajectory(self) -> Optional[ReActTrajectory]:
        """Get current reasoning trajectory."""
        return self._current_trajectory


# =============================================================================
# Convenience Functions
# =============================================================================


async def reason_with_react(
    goal: str,
    llm_provider: "BaseLLMProvider",
    activity_loop: Optional["ActivityLoop"] = None,
    context: Optional[str] = None,
    max_iterations: int = 10,
    timeout_seconds: float = 300.0,
) -> ReActResult:
    """
    Convenience function for ReAct reasoning.

    Args:
        goal: Goal to achieve
        llm_provider: LLM provider
        activity_loop: Activity loop for execution
        context: Additional context
        max_iterations: Maximum iterations
        timeout_seconds: Timeout in seconds

    Returns:
        ReActResult
    """
    config = ReActConfig(
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )
    reasoner = ReActReasoner(
        llm_provider=llm_provider,
        activity_loop=activity_loop,
        config=config,
    )
    return await reasoner.reason(goal=goal, context=context)


__all__ = [
    # Enums
    "ReActStep",
    "ReActStatus",
    # Data models
    "ReActThought",
    "ReActAction",
    "ReActObservation",
    "ReActTrajectory",
    "ReActResult",
    # Config
    "ReActConfig",
    # Reasoner
    "ReActReasoner",
    # Convenience
    "reason_with_react",
]
