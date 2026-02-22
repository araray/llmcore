# src/llmcore/agents/resilience/circuit_breaker.py
"""
Agent Circuit Breaker for LLMCore Agent System.

This module implements circuit breaker patterns to prevent runaway agent execution:
- Maximum iteration limits
- Repeated error detection
- Time and cost limits
- Progress stall detection

The circuit breaker "trips" when limits are exceeded, stopping agent execution
gracefully with a clear reason.

Usage:
    from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker

    breaker = AgentCircuitBreaker(
        max_iterations=15,
        max_same_errors=3,
        max_execution_time_seconds=300,
    )

    breaker.start()

    for iteration in range(100):
        result = breaker.check(
            iteration=iteration,
            progress=current_progress,
            error=last_error if failed else None,
            cost=iteration_cost,
        )

        if result.tripped:
            print(f"Circuit breaker tripped: {result.reason}")
            break

Integration Point:
    Call check() at the end of each CognitiveCycle iteration.

Author: llmcore team
Date: 2026-01-21
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Use Pydantic if available
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================


class TripReason(str, Enum):
    """Reasons why a circuit breaker might trip."""

    MAX_ITERATIONS = "max_iterations"
    REPEATED_ERROR = "repeated_error"
    TIME_LIMIT = "time_limit"
    COST_LIMIT = "cost_limit"
    PROGRESS_STALL = "progress_stall"
    MANUAL_STOP = "manual_stop"


class CircuitState(str, Enum):
    """State of the circuit breaker."""

    OPEN = "open"  # Normal operation, not tripped
    TRIPPED = "tripped"  # Breaker has been triggered
    RESET = "reset"  # Breaker was reset, ready for use


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_hash: str  # Hash of error message for comparison
    error_message: str  # Original error message
    iteration: int  # Iteration when error occurred
    timestamp: datetime  # When error occurred
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressRecord:
    """Record of progress at a point in time."""

    iteration: int
    progress: float  # 0.0 to 1.0
    timestamp: datetime


if PYDANTIC_AVAILABLE:

    class CircuitBreakerResult(BaseModel):
        """Result of a circuit breaker check."""

        tripped: bool = False
        reason: TripReason | None = None
        message: str = ""

        # Current state info
        current_iteration: int = 0
        elapsed_time_seconds: float = 0.0
        total_cost: float = 0.0
        error_count: int = 0
        same_error_count: int = 0
        progress_stall_iterations: int = 0

        # Recommendations
        should_retry: bool = False
        suggested_action: str = ""
else:

    @dataclass
    class CircuitBreakerResult:
        """Result of a circuit breaker check (dataclass version)."""

        tripped: bool = False
        reason: TripReason | None = None
        message: str = ""

        current_iteration: int = 0
        elapsed_time_seconds: float = 0.0
        total_cost: float = 0.0
        error_count: int = 0
        same_error_count: int = 0
        progress_stall_iterations: int = 0

        should_retry: bool = False
        suggested_action: str = ""


if PYDANTIC_AVAILABLE:

    class CircuitBreakerConfig(BaseModel):
        """Configuration for the circuit breaker."""

        max_iterations: int = Field(default=15, ge=1, le=1000)
        max_same_errors: int = Field(default=3, ge=1, le=100)
        max_execution_time_seconds: int = Field(default=300, ge=1)
        max_total_cost: float = Field(default=1.0, ge=0.0)
        progress_stall_threshold: int = Field(default=5, ge=1, le=50)
        progress_stall_tolerance: float = Field(default=0.01, ge=0.0, le=1.0)
else:

    @dataclass
    class CircuitBreakerConfig:
        """Configuration for the circuit breaker (dataclass version)."""

        max_iterations: int = 15
        max_same_errors: int = 3
        max_execution_time_seconds: int = 300
        max_total_cost: float = 1.0
        progress_stall_threshold: int = 5
        progress_stall_tolerance: float = 0.01


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class AgentCircuitBreaker:
    """
    Circuit breaker for agent execution control.

    Monitors agent execution and trips when limits are exceeded to prevent:
    - Infinite loops
    - Runaway costs
    - Repeated failures
    - Stalled progress

    Args:
        max_iterations: Maximum number of iterations before tripping
        max_same_errors: Maximum times the same error can occur
        max_execution_time_seconds: Maximum total execution time
        max_total_cost: Maximum total cost in dollars
        progress_stall_threshold: Iterations without progress before tripping
        progress_stall_tolerance: Minimum progress change to count as "progress"
    """

    def __init__(
        self,
        max_iterations: int = 15,
        max_same_errors: int = 3,
        max_execution_time_seconds: int = 300,
        max_total_cost: float = 1.0,
        progress_stall_threshold: int = 5,
        progress_stall_tolerance: float = 0.01,
        config: CircuitBreakerConfig | None = None,
    ):
        # Use config if provided, otherwise use individual params
        if config:
            self.config = config
        else:
            self.config = CircuitBreakerConfig(
                max_iterations=max_iterations,
                max_same_errors=max_same_errors,
                max_execution_time_seconds=max_execution_time_seconds,
                max_total_cost=max_total_cost,
                progress_stall_threshold=progress_stall_threshold,
                progress_stall_tolerance=progress_stall_tolerance,
            )

        # State tracking
        self._state: CircuitState = CircuitState.RESET
        self._start_time: datetime | None = None
        self._total_cost: float = 0.0

        # Error tracking
        self._errors: list[ErrorRecord] = []
        self._error_counts: Counter = Counter()  # error_hash -> count

        # Progress tracking
        self._progress_history: list[ProgressRecord] = []
        self._last_significant_progress: ProgressRecord | None = None

        # Trip info
        self._trip_reason: TripReason | None = None
        self._trip_message: str = ""

    @classmethod
    def from_config(cls, config: CircuitBreakerConfig) -> AgentCircuitBreaker:
        """Create circuit breaker from configuration object."""
        return cls(config=config)

    def start(self) -> None:
        """
        Start the circuit breaker for a new agent run.

        Must be called before check().
        """
        self._state = CircuitState.OPEN
        self._start_time = datetime.now()
        self._total_cost = 0.0
        self._errors = []
        self._error_counts = Counter()
        self._progress_history = []
        self._last_significant_progress = None
        self._trip_reason = None
        self._trip_message = ""

        logger.debug(f"Circuit breaker started with config: {self.config}")

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.RESET
        self._start_time = None
        self._total_cost = 0.0
        self._errors = []
        self._error_counts = Counter()
        self._progress_history = []
        self._last_significant_progress = None
        self._trip_reason = None
        self._trip_message = ""

        logger.debug("Circuit breaker reset")

    def check(
        self,
        iteration: int,
        progress: float = 0.0,
        error: str | None = None,
        cost: float = 0.0,
        context: dict[str, Any] | None = None,
        step_completed: bool | None = None,
    ) -> CircuitBreakerResult:
        """
        Check if the circuit breaker should trip.

        Call this at the end of each agent iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            progress: Current progress (0.0 to 1.0)
            error: Error message if this iteration failed, None if successful
            cost: Cost of this iteration in dollars
            context: Additional context for error tracking
            step_completed: Whether the current step was explicitly completed.
                If True, resets the progress stall counter even if progress
                value hasn't changed significantly. This prevents false stall
                trips when the agent reports low progress but is actually
                making progress on steps.

        Returns:
            CircuitBreakerResult indicating if breaker tripped and why
        """
        if self._state == CircuitState.RESET:
            raise RuntimeError("Circuit breaker not started. Call start() first.")

        if self._state == CircuitState.TRIPPED:
            return self._make_result(tripped=True)

        # Update cost tracking
        self._total_cost += cost

        # Record progress
        self._progress_history.append(
            ProgressRecord(
                iteration=iteration,
                progress=progress,
                timestamp=datetime.now(),
            )
        )

        # Record error if present
        if error:
            self._record_error(error, iteration, context or {})

        # Check all limits (pass step_completed for stall detection)
        result = self._check_limits(iteration, progress, step_completed)

        if result.tripped:
            self._state = CircuitState.TRIPPED
            self._trip_reason = result.reason
            self._trip_message = result.message
            logger.warning(f"Circuit breaker tripped: {result.reason.value} - {result.message}")

        return result

    def force_trip(self, reason: str = "Manual stop requested") -> CircuitBreakerResult:
        """
        Manually trip the circuit breaker.

        Args:
            reason: Reason for manual trip

        Returns:
            CircuitBreakerResult with MANUAL_STOP reason
        """
        self._state = CircuitState.TRIPPED
        self._trip_reason = TripReason.MANUAL_STOP
        self._trip_message = reason

        return self._make_result(
            tripped=True,
            reason=TripReason.MANUAL_STOP,
            message=reason,
        )

    def _record_error(self, error: str, iteration: int, context: dict[str, Any]) -> None:
        """Record an error occurrence."""

        # Hash error for comparison (normalize whitespace)
        error_normalized = " ".join(error.lower().split())
        error_hash = hashlib.md5(error_normalized.encode()).hexdigest()[:16]

        record = ErrorRecord(
            error_hash=error_hash,
            error_message=error,
            iteration=iteration,
            timestamp=datetime.now(),
            context=context,
        )

        self._errors.append(record)
        self._error_counts[error_hash] += 1

        logger.debug(
            f"Recorded error (hash={error_hash}, count={self._error_counts[error_hash]}): {error[:50]}"
        )

    def _check_limits(
        self, iteration: int, progress: float, step_completed: bool | None = None
    ) -> CircuitBreakerResult:
        """Check all circuit breaker limits."""

        # Check 1: Maximum iterations
        if iteration >= self.config.max_iterations:
            return self._make_result(
                tripped=True,
                reason=TripReason.MAX_ITERATIONS,
                message=f"Maximum iterations ({self.config.max_iterations}) exceeded",
                suggested_action="Increase max_iterations if task requires more steps",
            )

        # Check 2: Repeated errors
        max_error_count = max(self._error_counts.values()) if self._error_counts else 0
        if max_error_count >= self.config.max_same_errors:
            # Find the error that triggered this
            most_common_hash = self._error_counts.most_common(1)[0][0]
            error_sample = next(
                (e.error_message for e in self._errors if e.error_hash == most_common_hash),
                "Unknown error",
            )

            return self._make_result(
                tripped=True,
                reason=TripReason.REPEATED_ERROR,
                message=f"Same error occurred {max_error_count} times: {error_sample[:100]}",
                suggested_action="Fix the underlying issue or use different approach",
                should_retry=False,  # Don't retry same error
            )

        # Check 3: Time limit
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            if elapsed >= self.config.max_execution_time_seconds:
                return self._make_result(
                    tripped=True,
                    reason=TripReason.TIME_LIMIT,
                    message=f"Execution time ({elapsed:.0f}s) exceeded limit ({self.config.max_execution_time_seconds}s)",
                    suggested_action="Increase time limit or simplify task",
                )

        # Check 4: Cost limit
        if self._total_cost >= self.config.max_total_cost:
            return self._make_result(
                tripped=True,
                reason=TripReason.COST_LIMIT,
                message=f"Total cost (${self._total_cost:.4f}) exceeded limit (${self.config.max_total_cost})",
                suggested_action="Increase cost limit or use cheaper model",
            )

        # Check 5: Progress stall
        stall_iterations = self._check_progress_stall(progress, step_completed)
        if stall_iterations >= self.config.progress_stall_threshold:
            return self._make_result(
                tripped=True,
                reason=TripReason.PROGRESS_STALL,
                message=f"No progress for {stall_iterations} iterations (stuck at {progress * 100:.0f}%)",
                suggested_action="Task may be stuck, consider different approach",
                should_retry=True,  # May work with different strategy
            )

        # All checks passed
        return self._make_result(
            tripped=False,
            current_iteration=iteration,
            same_error_count=max_error_count,
            progress_stall_iterations=stall_iterations,
        )

    def _check_progress_stall(
        self, current_progress: float, step_completed: bool | None = None
    ) -> int:
        """
        Check how many iterations have passed without significant progress.

        Args:
            current_progress: Current progress value (0.0 to 1.0)
            step_completed: Whether the current step was explicitly completed.
                If True, resets the stall counter even if progress hasn't
                changed numerically. This prevents false stall detection
                when the model explicitly indicates step completion.

        Returns:
            Number of stalled iterations (0 if no stall or if step was completed)
        """
        if not self._progress_history:
            return 0

        # If step was explicitly completed, reset stall tracking
        # This is the key fix for false positives - when the model says
        # a step is done, we should trust that over the progress number
        if step_completed is True:
            self._last_significant_progress = self._progress_history[-1]
            logger.debug("Step explicitly completed, resetting progress stall tracking")
            return 0

        # Update significant progress tracking
        if self._last_significant_progress is None:
            self._last_significant_progress = self._progress_history[0]

        # Check if current progress is significantly higher
        progress_delta = current_progress - self._last_significant_progress.progress

        if progress_delta >= self.config.progress_stall_tolerance:
            # Made progress, update marker
            self._last_significant_progress = self._progress_history[-1]
            return 0

        # Calculate stall duration
        stall_start = self._last_significant_progress.iteration
        current_iteration = self._progress_history[-1].iteration

        return current_iteration - stall_start

    def _make_result(
        self,
        tripped: bool,
        reason: TripReason | None = None,
        message: str = "",
        suggested_action: str = "",
        should_retry: bool = False,
        current_iteration: int = 0,
        same_error_count: int = 0,
        progress_stall_iterations: int = 0,
    ) -> CircuitBreakerResult:
        """Create a CircuitBreakerResult with current state."""

        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()

        return CircuitBreakerResult(
            tripped=tripped,
            reason=reason or self._trip_reason,
            message=message or self._trip_message,
            current_iteration=current_iteration,
            elapsed_time_seconds=elapsed,
            total_cost=self._total_cost,
            error_count=len(self._errors),
            same_error_count=same_error_count,
            progress_stall_iterations=progress_stall_iterations,
            should_retry=should_retry,
            suggested_action=suggested_action,
        )

    # =========================================================================
    # Status and Introspection
    # =========================================================================

    @property
    def is_tripped(self) -> bool:
        """Check if breaker is currently tripped."""
        return self._state == CircuitState.TRIPPED

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def total_cost(self) -> float:
        """Get total cost so far."""
        return self._total_cost

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if not self._start_time:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    @property
    def error_summary(self) -> dict[str, int]:
        """Get summary of errors by count."""
        summary = {}
        for error_hash, count in self._error_counts.most_common():
            # Find a sample error message
            sample = next(
                (e.error_message[:100] for e in self._errors if e.error_hash == error_hash),
                "Unknown",
            )
            summary[sample] = count
        return summary

    def get_status(self) -> dict[str, Any]:
        """Get full status as dictionary."""
        return {
            "state": self._state.value,
            "tripped": self.is_tripped,
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
            "trip_message": self._trip_message,
            "elapsed_seconds": self.elapsed_time,
            "total_cost": self._total_cost,
            "error_count": len(self._errors),
            "unique_errors": len(self._error_counts),
            "progress_records": len(self._progress_history),
            "config": {
                "max_iterations": self.config.max_iterations,
                "max_same_errors": self.config.max_same_errors,
                "max_execution_time_seconds": self.config.max_execution_time_seconds,
                "max_total_cost": self.config.max_total_cost,
                "progress_stall_threshold": self.config.progress_stall_threshold,
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_circuit_breaker(
    preset: str = "default",
    **overrides,
) -> AgentCircuitBreaker:
    """
    Create a circuit breaker with preset configuration.

    Presets:
        - "default": Balanced defaults (15 iterations, 5min, $1)
        - "strict": Conservative limits (5 iterations, 1min, $0.10)
        - "permissive": Relaxed limits (50 iterations, 30min, $10)
        - "development": Very relaxed for testing

    Args:
        preset: Preset name
        **overrides: Override specific config values
    """
    presets = {
        "default": CircuitBreakerConfig(
            max_iterations=15,
            max_same_errors=3,
            max_execution_time_seconds=300,
            max_total_cost=1.0,
            progress_stall_threshold=5,
        ),
        "strict": CircuitBreakerConfig(
            max_iterations=5,
            max_same_errors=2,
            max_execution_time_seconds=60,
            max_total_cost=0.10,
            progress_stall_threshold=3,
        ),
        "permissive": CircuitBreakerConfig(
            max_iterations=50,
            max_same_errors=5,
            max_execution_time_seconds=1800,
            max_total_cost=10.0,
            progress_stall_threshold=10,
        ),
        "development": CircuitBreakerConfig(
            max_iterations=100,
            max_same_errors=10,
            max_execution_time_seconds=3600,
            max_total_cost=100.0,
            progress_stall_threshold=20,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset]

    # Apply overrides
    if overrides:
        config_dict = {
            "max_iterations": config.max_iterations,
            "max_same_errors": config.max_same_errors,
            "max_execution_time_seconds": config.max_execution_time_seconds,
            "max_total_cost": config.max_total_cost,
            "progress_stall_threshold": config.progress_stall_threshold,
            "progress_stall_tolerance": config.progress_stall_tolerance,
        }
        config_dict.update(overrides)
        config = CircuitBreakerConfig(**config_dict)

    return AgentCircuitBreaker(config=config)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Running Circuit Breaker self-tests...\n")

    passed = 0
    failed = 0

    # Test 1: Max iterations
    print("Test 1: Max iterations limit")
    breaker = AgentCircuitBreaker(max_iterations=5)
    breaker.start()

    for i in range(10):
        result = breaker.check(iteration=i, progress=0.1 * i)
        if result.tripped:
            if result.reason == TripReason.MAX_ITERATIONS and i == 5:
                print(f"  ✅ Tripped at iteration {i}: {result.reason.value}")
                passed += 1
            else:
                print(f"  ❌ Unexpected trip at {i}: {result.reason}")
                failed += 1
            break
    else:
        print("  ❌ Should have tripped")
        failed += 1

    # Test 2: Repeated errors
    print("\nTest 2: Repeated error detection")
    breaker = AgentCircuitBreaker(max_same_errors=3, max_iterations=100)
    breaker.start()

    for i in range(10):
        result = breaker.check(
            iteration=i, progress=0.0, error="Same error message" if i < 5 else None
        )
        if result.tripped:
            if result.reason == TripReason.REPEATED_ERROR:
                print(f"  ✅ Tripped at iteration {i}: {result.reason.value}")
                passed += 1
            else:
                print(f"  ❌ Wrong reason: {result.reason}")
                failed += 1
            break
    else:
        print("  ❌ Should have tripped")
        failed += 1

    # Test 3: Progress stall
    print("\nTest 3: Progress stall detection")
    breaker = AgentCircuitBreaker(
        progress_stall_threshold=3,
        max_iterations=100,
    )
    breaker.start()

    for i in range(10):
        result = breaker.check(iteration=i, progress=0.5)  # No progress
        if result.tripped:
            if result.reason == TripReason.PROGRESS_STALL:
                print(f"  ✅ Tripped at iteration {i}: {result.reason.value}")
                passed += 1
            else:
                print(f"  ❌ Wrong reason: {result.reason}")
                failed += 1
            break
    else:
        print("  ❌ Should have tripped")
        failed += 1

    # Test 4: Cost limit
    print("\nTest 4: Cost limit")
    breaker = AgentCircuitBreaker(max_total_cost=0.05, max_iterations=100)
    breaker.start()

    for i in range(10):
        result = breaker.check(iteration=i, progress=0.1 * i, cost=0.02)
        if result.tripped:
            if result.reason == TripReason.COST_LIMIT:
                print(
                    f"  ✅ Tripped at iteration {i}: {result.reason.value} (cost=${breaker.total_cost:.2f})"
                )
                passed += 1
            else:
                print(f"  ❌ Wrong reason: {result.reason}")
                failed += 1
            break
    else:
        print("  ❌ Should have tripped")
        failed += 1

    # Test 5: Normal completion (no trip)
    print("\nTest 5: Normal completion without tripping")
    breaker = AgentCircuitBreaker(max_iterations=10)
    breaker.start()

    all_ok = True
    for i in range(5):
        result = breaker.check(iteration=i, progress=0.2 * i)
        if result.tripped:
            print(f"  ❌ Unexpected trip at {i}")
            failed += 1
            all_ok = False
            break

    if all_ok:
        print("  ✅ Completed 5 iterations without tripping")
        passed += 1

    # Test 6: Factory presets
    print("\nTest 6: Factory presets")
    try:
        strict = create_circuit_breaker("strict")
        assert strict.config.max_iterations == 5

        permissive = create_circuit_breaker("permissive")
        assert permissive.config.max_iterations == 50

        custom = create_circuit_breaker("default", max_iterations=25)
        assert custom.config.max_iterations == 25

        print("  ✅ Factory presets work correctly")
        passed += 1
    except Exception as e:
        print(f"  ❌ Factory error: {e}")
        failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! ✅")
    else:
        print("Some tests failed ❌")
        exit(1)
