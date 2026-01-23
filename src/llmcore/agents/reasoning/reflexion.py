# src/llmcore/agents/reasoning/reflexion.py
"""
Reflexion Reasoner Implementation.

Implements the Reflexion framework from Shinn et al. 2023 for
learning from failures within a session.

Pattern:
    ReAct → Evaluate → Reflect → Retry (with reflection memory)

Research Reference:
    Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
    - 91% on HumanEval vs 80% GPT-4 alone
    - Verbal self-reflection stored in episodic memory
    - Requires clear evaluation criteria

Usage:
    from llmcore.agents.reasoning import ReflexionReasoner

    reasoner = ReflexionReasoner(llm_provider, activity_loop)
    result = await reasoner.reason_with_reflection(
        goal="Write a function to calculate fibonacci",
        max_trials=3
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
    Tuple,
    Union,
)

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


from .react import (
    ReActConfig,
    ReActReasoner,
    ReActResult,
    ReActStatus,
    ReActTrajectory,
)

if TYPE_CHECKING:
    from llmcore.agents.activities.loop import ActivityLoop
    from llmcore.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TrialOutcome(str, Enum):
    """Outcome of a trial."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class Reflection:
    """A reflection on a failed trial."""

    trial_number: int
    outcome: TrialOutcome
    what_went_wrong: str
    what_to_improve: str
    lessons_learned: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def format(self) -> str:
        """Format reflection for LLM context."""
        lessons = "\n".join(f"  - {l}" for l in self.lessons_learned)
        return f"""Trial {self.trial_number} Reflection:
Outcome: {self.outcome.value}
What went wrong: {self.what_went_wrong}
What to improve: {self.what_to_improve}
Lessons learned:
{lessons}"""


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial_number: int
    react_result: ReActResult
    outcome: TrialOutcome
    reflection: Optional[Reflection] = None
    duration_ms: int = 0

    @property
    def success(self) -> bool:
        """Whether trial was successful."""
        return self.outcome == TrialOutcome.SUCCESS


if PYDANTIC_AVAILABLE:

    class ReflexionResult(BaseModel):
        """Result of Reflexion reasoning."""

        success: bool = Field(description="Whether reasoning succeeded")
        final_answer: Optional[str] = Field(None, description="Final answer")
        trials: int = Field(0, description="Number of trials attempted")
        successful_trial: Optional[int] = Field(None, description="Which trial succeeded")
        reflections: List[Dict[str, Any]] = Field(
            default_factory=list, description="All reflections"
        )
        total_duration_ms: int = Field(0, description="Total duration")
        error: Optional[str] = Field(None, description="Error if failed")

        class Config:
            use_enum_values = True
else:

    @dataclass
    class ReflexionResult:
        """Result of Reflexion reasoning."""

        success: bool
        final_answer: Optional[str] = None
        trials: int = 0
        successful_trial: Optional[int] = None
        reflections: List[Dict[str, Any]] = field(default_factory=list)
        total_duration_ms: int = 0
        error: Optional[str] = None


# =============================================================================
# Reflexion Prompt Templates
# =============================================================================


REFLECTION_PROMPT = """You are analyzing a failed attempt at completing a task.

Task: {goal}

Previous Attempt Summary:
{attempt_summary}

Outcome: {outcome}

Now reflect on what went wrong and how to improve:

1. What specifically went wrong in this attempt?
2. What should be done differently next time?
3. What lessons can be learned?

Format your response as:
WHAT_WENT_WRONG: [concise description]
WHAT_TO_IMPROVE: [specific actionable improvement]
LESSONS_LEARNED:
- [lesson 1]
- [lesson 2]
- [lesson 3]"""


RETRY_PROMPT_ADDITION = """
=== IMPORTANT: Learning from Previous Attempts ===

You have attempted this task before and failed. Use these reflections to improve:

{reflections}

Apply these lessons in your new attempt. Do NOT repeat the same mistakes.
=== End of Reflections ===
"""


# =============================================================================
# Reflexion Reasoner Implementation
# =============================================================================


class ReflexionConfig:
    """Configuration for Reflexion reasoning."""

    def __init__(
        self,
        max_trials: int = 3,
        react_config: Optional[ReActConfig] = None,
        reflection_temperature: float = 0.5,
        require_evaluation: bool = True,
        evaluation_prompt: Optional[str] = None,
        stop_on_success: bool = True,
        include_all_reflections: bool = True,
    ):
        self.max_trials = max_trials
        self.react_config = react_config or ReActConfig()
        self.reflection_temperature = reflection_temperature
        self.require_evaluation = require_evaluation
        self.evaluation_prompt = evaluation_prompt
        self.stop_on_success = stop_on_success
        self.include_all_reflections = include_all_reflections


class ReflexionReasoner:
    """
    Reflexion reasoning framework implementation.

    Wraps ReAct with trial-based learning from failures.
    Each failed trial generates a reflection that informs
    the next attempt.

    Args:
        llm_provider: LLM provider for reasoning and reflection
        activity_loop: Activity loop for executing actions
        config: Reflexion configuration
        evaluator: Optional custom evaluator function
    """

    def __init__(
        self,
        llm_provider: Optional["BaseLLMProvider"] = None,
        activity_loop: Optional["ActivityLoop"] = None,
        config: Optional[ReflexionConfig] = None,
        evaluator: Optional[Callable[[str, ReActResult], TrialOutcome]] = None,
    ):
        self.llm_provider = llm_provider
        self.activity_loop = activity_loop
        self.config = config or ReflexionConfig()
        self.evaluator = evaluator

        # Create ReAct reasoner
        self.react_reasoner = ReActReasoner(
            llm_provider=llm_provider,
            activity_loop=activity_loop,
            config=self.config.react_config,
        )

        # Trial history
        self._trials: List[TrialResult] = []
        self._reflections: List[Reflection] = []

    async def reason_with_reflection(
        self,
        goal: str,
        context: Optional[str] = None,
        available_activities: Optional[str] = None,
        on_trial: Optional[Callable[[TrialResult], None]] = None,
        on_reflection: Optional[Callable[[Reflection], None]] = None,
    ) -> ReflexionResult:
        """
        Execute Reflexion reasoning loop.

        Args:
            goal: The goal to achieve
            context: Additional context
            available_activities: Available activities
            on_trial: Callback after each trial
            on_reflection: Callback after each reflection

        Returns:
            ReflexionResult with final answer or reflections
        """
        start_time = time.time()
        self._trials = []
        self._reflections = []

        for trial_num in range(1, self.config.max_trials + 1):
            logger.info(f"Starting trial {trial_num}/{self.config.max_trials}")

            # Build context with reflections
            trial_context = self._build_trial_context(context)

            # Run ReAct
            react_result = await self.react_reasoner.reason(
                goal=goal,
                context=trial_context,
                available_activities=available_activities,
            )

            # Evaluate outcome
            outcome = await self._evaluate_outcome(goal, react_result)

            # Create trial result
            trial = TrialResult(
                trial_number=trial_num,
                react_result=react_result,
                outcome=outcome,
                duration_ms=react_result.duration_ms,
            )
            self._trials.append(trial)

            if on_trial:
                on_trial(trial)

            # Check for success
            if outcome == TrialOutcome.SUCCESS and self.config.stop_on_success:
                logger.info(f"Trial {trial_num} succeeded!")
                return self._create_result(
                    success=True,
                    final_answer=react_result.final_answer,
                    successful_trial=trial_num,
                    start_time=start_time,
                )

            # Generate reflection for non-successful trials
            if trial_num < self.config.max_trials:
                reflection = await self._generate_reflection(
                    goal=goal,
                    trial=trial,
                )
                trial.reflection = reflection
                self._reflections.append(reflection)

                if on_reflection:
                    on_reflection(reflection)

                logger.info(f"Trial {trial_num} reflection: {reflection.what_to_improve}")

        # All trials exhausted
        best_trial = self._find_best_trial()

        return self._create_result(
            success=False,
            final_answer=best_trial.react_result.final_answer if best_trial else None,
            start_time=start_time,
            error="All trials exhausted without success",
        )

    def _build_trial_context(self, base_context: Optional[str]) -> str:
        """Build context including reflections from previous trials."""
        parts = []

        if base_context:
            parts.append(base_context)

        if self._reflections and self.config.include_all_reflections:
            reflections_text = "\n\n".join(r.format() for r in self._reflections)
            parts.append(RETRY_PROMPT_ADDITION.format(reflections=reflections_text))

        return "\n\n".join(parts) if parts else ""

    async def _evaluate_outcome(self, goal: str, result: ReActResult) -> TrialOutcome:
        """Evaluate the outcome of a trial."""
        # Use custom evaluator if provided
        if self.evaluator:
            return self.evaluator(goal, result)

        # Default evaluation based on ReAct status
        if result.status == ReActStatus.SUCCESS:
            return TrialOutcome.SUCCESS
        elif result.status in [ReActStatus.MAX_ITERATIONS, ReActStatus.TIMEOUT]:
            return TrialOutcome.PARTIAL
        elif result.status == ReActStatus.EARLY_TERMINATION:
            return TrialOutcome.FAILURE
        else:
            return TrialOutcome.ERROR

    async def _generate_reflection(
        self,
        goal: str,
        trial: TrialResult,
    ) -> Reflection:
        """Generate reflection on a failed trial."""
        if not self.llm_provider:
            # Generate basic reflection without LLM
            return Reflection(
                trial_number=trial.trial_number,
                outcome=trial.outcome,
                what_went_wrong="Unable to generate detailed reflection (no LLM)",
                what_to_improve="Try a different approach",
                lessons_learned=["Consider alternative methods"],
            )

        # Build attempt summary
        trajectory_info = trial.react_result.trajectory or {}
        attempt_summary = f"""
Iterations: {trial.react_result.iterations}
Duration: {trial.duration_ms}ms
Actions taken: {trajectory_info.get("actions", 0)}
Success rate: {trajectory_info.get("success_rate", 0):.0%}
Final status: {trial.react_result.status}

Action history:
{trajectory_info.get("history", "No history available")}

Error (if any): {trial.react_result.error or "None"}
"""

        # Generate reflection via LLM
        prompt = REFLECTION_PROMPT.format(
            goal=goal,
            attempt_summary=attempt_summary,
            outcome=trial.outcome.value,
        )

        try:
            response = await self.llm_provider.chat_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.reflection_temperature,
            )

            content = response.content if hasattr(response, "content") else str(response)

            # Parse reflection
            return self._parse_reflection(content, trial.trial_number, trial.outcome)

        except Exception as e:
            logger.warning(f"Failed to generate reflection: {e}")
            return Reflection(
                trial_number=trial.trial_number,
                outcome=trial.outcome,
                what_went_wrong=f"Trial failed: {trial.react_result.error}",
                what_to_improve="Try alternative approach",
                lessons_learned=[f"Error occurred: {e}"],
            )

    def _parse_reflection(
        self,
        content: str,
        trial_number: int,
        outcome: TrialOutcome,
    ) -> Reflection:
        """Parse reflection from LLM response."""
        import re

        # Extract what went wrong
        wrong_match = re.search(
            r"WHAT_WENT_WRONG:\s*(.+?)(?=WHAT_TO_IMPROVE|LESSONS_LEARNED|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        what_went_wrong = wrong_match.group(1).strip() if wrong_match else "Unknown issue"

        # Extract what to improve
        improve_match = re.search(
            r"WHAT_TO_IMPROVE:\s*(.+?)(?=LESSONS_LEARNED|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        what_to_improve = improve_match.group(1).strip() if improve_match else "Try differently"

        # Extract lessons learned
        lessons_match = re.search(
            r"LESSONS_LEARNED:\s*(.+?)$",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        lessons = []
        if lessons_match:
            lessons_text = lessons_match.group(1)
            # Parse bullet points
            lessons = [
                l.strip().lstrip("- •").strip()
                for l in lessons_text.split("\n")
                if l.strip() and not l.strip().startswith("WHAT_")
            ]

        if not lessons:
            lessons = [what_to_improve]

        return Reflection(
            trial_number=trial_number,
            outcome=outcome,
            what_went_wrong=what_went_wrong,
            what_to_improve=what_to_improve,
            lessons_learned=lessons,
        )

    def _find_best_trial(self) -> Optional[TrialResult]:
        """Find the best trial result."""
        if not self._trials:
            return None

        # Prioritize by outcome
        outcome_priority = {
            TrialOutcome.SUCCESS: 0,
            TrialOutcome.PARTIAL: 1,
            TrialOutcome.FAILURE: 2,
            TrialOutcome.ERROR: 3,
        }

        return min(
            self._trials,
            key=lambda t: (
                outcome_priority.get(t.outcome, 99),
                -t.react_result.iterations,  # More iterations = more effort
            ),
        )

    def _create_result(
        self,
        success: bool,
        start_time: float,
        final_answer: Optional[str] = None,
        successful_trial: Optional[int] = None,
        error: Optional[str] = None,
    ) -> ReflexionResult:
        """Create ReflexionResult."""
        duration_ms = int((time.time() - start_time) * 1000)

        return ReflexionResult(
            success=success,
            final_answer=final_answer,
            trials=len(self._trials),
            successful_trial=successful_trial,
            reflections=[
                {
                    "trial": r.trial_number,
                    "outcome": r.outcome.value,
                    "what_went_wrong": r.what_went_wrong,
                    "what_to_improve": r.what_to_improve,
                    "lessons": r.lessons_learned,
                }
                for r in self._reflections
            ],
            total_duration_ms=duration_ms,
            error=error,
        )

    @property
    def trials(self) -> List[TrialResult]:
        """Get all trial results."""
        return self._trials.copy()

    @property
    def reflections(self) -> List[Reflection]:
        """Get all reflections."""
        return self._reflections.copy()


# =============================================================================
# Convenience Functions
# =============================================================================


async def reason_with_reflexion(
    goal: str,
    llm_provider: "BaseLLMProvider",
    activity_loop: Optional["ActivityLoop"] = None,
    context: Optional[str] = None,
    max_trials: int = 3,
    react_max_iterations: int = 10,
) -> ReflexionResult:
    """
    Convenience function for Reflexion reasoning.

    Args:
        goal: Goal to achieve
        llm_provider: LLM provider
        activity_loop: Activity loop for execution
        context: Additional context
        max_trials: Maximum trial attempts
        react_max_iterations: Max iterations per ReAct trial

    Returns:
        ReflexionResult
    """
    react_config = ReActConfig(max_iterations=react_max_iterations)
    reflexion_config = ReflexionConfig(
        max_trials=max_trials,
        react_config=react_config,
    )

    reasoner = ReflexionReasoner(
        llm_provider=llm_provider,
        activity_loop=activity_loop,
        config=reflexion_config,
    )

    return await reasoner.reason_with_reflection(goal=goal, context=context)


__all__ = [
    # Enums
    "TrialOutcome",
    # Data models
    "Reflection",
    "TrialResult",
    "ReflexionResult",
    # Config
    "ReflexionConfig",
    # Reasoner
    "ReflexionReasoner",
    # Convenience
    "reason_with_reflexion",
]
