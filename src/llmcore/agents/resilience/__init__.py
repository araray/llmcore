# src/llmcore/agents/resilience/__init__.py
"""
Resilience module for agent execution control.

Provides circuit breakers, early termination detection, and other
mechanisms to prevent runaway agent execution.
"""

from .circuit_breaker import (
    AgentCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResult,
    CircuitState,
    TripReason,
    create_circuit_breaker,
)

__all__ = [
    "AgentCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerResult",
    "CircuitState",
    "TripReason",
    "create_circuit_breaker",
]
