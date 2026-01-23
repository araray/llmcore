# src/llmcore/agents/routing/__init__.py
"""
Model routing and capability checking module.

Provides intelligent model selection, capability verification,
fallback chains, and cost-optimized routing for agent execution.

Features:
    - Three-tier model routing (FAST/BALANCED/CAPABLE)
    - Capability-aware model selection
    - Fallback chains with retry logic
    - Rate limit tracking and availability management
    - 35-85% cost savings on simple tasks

Usage:
    from llmcore.agents.routing import ModelRouter, CapabilityChecker

    router = ModelRouter()
    selection = router.select_model(
        complexity=GoalComplexity.SIMPLE,
        context=RoutingContext(estimated_tokens=1000)
    )
    print(f"Selected: {selection.model} (tier={selection.tier})")
"""

from .capability_checker import (
    Capability,
    CapabilityChecker,
    CapabilityIssue,
    CompatibilityResult,
    IssueSeverity,
)
from .model_router import (
    # Registry
    DEFAULT_MODELS,
    ModelCapability,
    ModelFallbackChain,
    # Data models
    ModelInfo,
    # Router
    ModelRouter,
    ModelSelection,
    # Enums
    ModelTier,
    RoutingContext,
    TierConfig,
    # Convenience
    get_default_router,
    select_model_for_complexity,
)

__all__ = [
    # Capability checker
    "Capability",
    "CapabilityChecker",
    "CapabilityIssue",
    "CompatibilityResult",
    "IssueSeverity",
    # Model router enums
    "ModelTier",
    "ModelCapability",
    # Model router data models
    "ModelInfo",
    "TierConfig",
    "ModelSelection",
    "RoutingContext",
    # Model router classes
    "ModelRouter",
    "ModelFallbackChain",
    # Registry
    "DEFAULT_MODELS",
    # Convenience functions
    "get_default_router",
    "select_model_for_complexity",
]
