# src/llmcore/agents/activities/loop.py
"""
Activity Loop.

Processes LLM output through the activity system, executing activities and
building observations for the next cognitive iteration.

Loop Flow:
    LLM OUTPUT → PARSE ACTIVITIES → EXECUTE → FORMAT OBSERVATION → CONTINUE/DONE

The Activity Loop is the bridge between the cognitive cycle's THINK phase output
and the ACT/OBSERVE phases. It enables model-agnostic tool execution by parsing
structured text requests and executing them through the activity system.

References:
    - Master Plan: Section 12 (Activity Loop Design)
    - Technical Spec: Section 5.4.5 (Loop Integration)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from collections.abc import Callable

from .executor import ActivityExecutor
from .parser import ActivityRequestParser, ParseResult
from .registry import ActivityRegistry, ExecutionContext, get_default_registry
from .schema import (
    ActivityExecution,
    ActivityLoopResult,
    ActivityRequest,
    ActivityResult,
    ActivityStatus,
)

if TYPE_CHECKING:
    from ..sandbox import SandboxProvider

logger = logging.getLogger(__name__)


# =============================================================================
# ACTIVITY LOOP CONFIGURATION
# =============================================================================


@dataclass
class ActivityLoopConfig:
    """Configuration for the activity loop."""

    # Limits
    max_per_iteration: int = 10  # Max activities per LLM output
    max_total: int = 100  # Max total activities per session

    # Behavior
    stop_on_error: bool = False  # Stop processing on first error
    parallel_execution: bool = False  # Execute activities in parallel

    # Timeouts
    default_timeout_seconds: int = 60
    total_timeout_seconds: int = 300

    # Output formatting
    max_observation_length: int = 4000
    include_reasoning: bool = True


# =============================================================================
# ACTIVITY LOOP
# =============================================================================


class ActivityLoop:
    """
    Process LLM output through the activity system.

    The ActivityLoop extracts activity requests from LLM output, executes them,
    and formats observations for the next cognitive iteration.

    Example:
        >>> loop = ActivityLoop()
        >>> result = await loop.process_output('''
        ...     I'll search for log files.
        ...     <activity_request>
        ...         <activity>file_search</activity>
        ...         <parameters>
        ...             <path>/var/log</path>
        ...         </parameters>
        ...     </activity_request>
        ... ''')
        >>> print(result.observation)
    """

    def __init__(
        self,
        registry: ActivityRegistry | None = None,
        executor: ActivityExecutor | None = None,
        parser: ActivityRequestParser | None = None,
        config: ActivityLoopConfig | None = None,
        sandbox: SandboxProvider | None = None,
    ):
        """
        Initialize the activity loop.

        Args:
            registry: Activity registry
            executor: Activity executor
            parser: Request parser
            config: Loop configuration
            sandbox: Sandbox provider for isolated execution
        """
        self.registry = registry or get_default_registry()
        self.executor = executor or ActivityExecutor(registry=self.registry, sandbox=sandbox)
        self.parser = parser or ActivityRequestParser()
        self.config = config or ActivityLoopConfig()
        self.sandbox = sandbox

        # Session tracking
        self._total_executed: int = 0
        self._session_start: float | None = None

    def start_session(self) -> None:
        """Start a new activity session."""
        self._total_executed = 0
        self._session_start = time.time()
        logger.info("Activity loop session started")

    def reset(self) -> None:
        """Reset the activity loop state."""
        self._total_executed = 0
        self._session_start = None

    async def process_output(
        self,
        llm_output: str,
        context: ExecutionContext | None = None,
        approval_callback: Callable[[str], bool] | None = None,
    ) -> ActivityLoopResult:
        """
        Process LLM output through the activity system.

        Args:
            llm_output: Raw LLM output text
            context: Execution context
            approval_callback: HITL approval callback

        Returns:
            ActivityLoopResult with execution results and observation
        """
        if self._session_start is None:
            self.start_session()

        context = context or ExecutionContext()
        result = ActivityLoopResult()

        # Check total timeout
        if self._session_start:
            elapsed = time.time() - self._session_start
            if elapsed > self.config.total_timeout_seconds:
                result.should_continue = False
                result.observation = "Session timeout exceeded"
                return result

        # Check total activity limit
        if self._total_executed >= self.config.max_total:
            result.should_continue = False
            result.observation = f"Maximum total activities ({self.config.max_total}) reached"
            return result

        # Check for final answer
        if self.parser.is_final_answer(llm_output):
            answer = self.parser.extract_final_answer(llm_output)
            result.should_continue = False
            result.is_final_answer = True
            result.observation = answer or llm_output
            return result

        # Parse activity requests
        parse_result = self.parser.parse(llm_output)
        result.remaining_text = parse_result.remaining_text
        result.parse_errors = parse_result.errors

        if not parse_result.has_requests:
            # No activities found - might be reasoning or final answer
            result.should_continue = self._should_continue(parse_result)
            result.observation = parse_result.remaining_text
            return result

        # Execute activities
        requests_to_execute = parse_result.requests[: self.config.max_per_iteration]

        if self.config.parallel_execution:
            executions = await self._execute_parallel(
                requests_to_execute, context, approval_callback
            )
        else:
            executions = await self._execute_sequential(
                requests_to_execute, context, approval_callback
            )

        result.executions = executions
        self._total_executed += len(executions)

        # Format observation
        result.observation = self._format_observation(executions)
        result.should_continue = self._determine_continuation(executions)

        return result

    async def _execute_sequential(
        self,
        requests: list[ActivityRequest],
        context: ExecutionContext,
        approval_callback: Callable[[str], bool] | None,
    ) -> list[ActivityExecution]:
        """Execute activities sequentially."""
        executions: list[ActivityExecution] = []

        for request in requests:
            execution = ActivityExecution(request=request)

            try:
                result = await self.executor.execute(request, context, approval_callback)
                execution.result = result

                if self.config.stop_on_error and result.status == ActivityStatus.FAILED:
                    executions.append(execution)
                    break

            except Exception as e:
                logger.error(f"Activity execution error: {e}", exc_info=True)
                execution.result = ActivityResult(
                    activity=request.activity,
                    status=ActivityStatus.FAILED,
                    error=str(e),
                )

            executions.append(execution)

        return executions

    async def _execute_parallel(
        self,
        requests: list[ActivityRequest],
        context: ExecutionContext,
        approval_callback: Callable[[str], bool] | None,
    ) -> list[ActivityExecution]:
        """Execute activities in parallel."""

        async def execute_one(request: ActivityRequest) -> ActivityExecution:
            execution = ActivityExecution(request=request)
            try:
                result = await self.executor.execute(request, context, approval_callback)
                execution.result = result
            except Exception as e:
                execution.result = ActivityResult(
                    activity=request.activity,
                    status=ActivityStatus.FAILED,
                    error=str(e),
                )
            return execution

        executions = await asyncio.gather(
            *[execute_one(r) for r in requests],
            return_exceptions=False,
        )
        return list(executions)

    def _should_continue(self, parse_result: ParseResult) -> bool:
        """Determine if loop should continue when no activities found."""
        remaining = parse_result.remaining_text.lower()

        # Check for completion indicators
        completion_phrases = [
            "task complete",
            "done",
            "finished",
            "accomplished",
            "no further actions",
            "nothing more to do",
        ]
        for phrase in completion_phrases:
            if phrase in remaining:
                return False

        # Default: continue if there's reasoning text
        return len(remaining.strip()) > 0

    def _determine_continuation(self, executions: list[ActivityExecution]) -> bool:
        """Determine if loop should continue after executions."""
        if not executions:
            return True

        # Check for terminal activities
        for execution in executions:
            if execution.request.activity == "final_answer":
                return False
            if execution.request.activity == "ask_human":
                return False  # Wait for human input

        # Continue unless all failed
        all_failed = all(e.result and e.result.status == ActivityStatus.FAILED for e in executions)
        return not all_failed

    def _format_observation(self, executions: list[ActivityExecution]) -> str:
        """Format execution results as observation text."""
        if not executions:
            return "No activities executed."

        observations: list[str] = []

        for execution in executions:
            if execution.result is None:
                observations.append(f"Activity '{execution.request.activity}': No result")
                continue

            obs = execution.result.to_observation(
                max_length=self.config.max_observation_length // len(executions)
            )
            observations.append(obs)

        combined = "\n\n---\n\n".join(observations)

        # Truncate if too long
        if len(combined) > self.config.max_observation_length:
            combined = (
                combined[: self.config.max_observation_length]
                + f"\n... [truncated {len(combined) - self.config.max_observation_length} chars]"
            )

        return combined

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def total_executed(self) -> int:
        """Get total activities executed in session."""
        return self._total_executed

    @property
    def remaining_budget(self) -> int:
        """Get remaining activity budget."""
        return max(0, self.config.max_total - self._total_executed)

    @property
    def session_elapsed(self) -> float:
        """Get elapsed session time in seconds."""
        if self._session_start is None:
            return 0.0
        return time.time() - self._session_start


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def process_llm_output(
    llm_output: str,
    context: ExecutionContext | None = None,
    **kwargs,
) -> ActivityLoopResult:
    """
    Process LLM output through activity loop.

    Convenience function for one-off processing.

    Args:
        llm_output: LLM output text
        context: Execution context
        **kwargs: Additional arguments for ActivityLoop

    Returns:
        ActivityLoopResult
    """
    loop = ActivityLoop(**kwargs)
    return await loop.process_output(llm_output, context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ActivityLoop",
    "ActivityLoopConfig",
    "process_llm_output",
]
