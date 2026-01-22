# src/llmcore/agents/routing/__init__.py
"""
Model routing and capability checking module.

Provides intelligent model selection, capability verification,
and fallback chains for agent execution.
"""

from .capability_checker import (
    Capability,
    CapabilityChecker,
    CapabilityIssue,
    CompatibilityResult,
    IssueSeverity,
)

__all__ = [
    "Capability",
    "CapabilityChecker",
    "CapabilityIssue",
    "CompatibilityResult",
    "IssueSeverity",
]
