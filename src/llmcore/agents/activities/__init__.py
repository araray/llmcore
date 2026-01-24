# src/llmcore/agents/activities/__init__.py
"""
Activity System for Model-Agnostic Tool Execution.

The Activity System provides a structured text-based protocol for LLMs to request
and execute actions, enabling tool use with any instruction-following model
regardless of native tool support.

Key Components:
    - Schema: Data models for activities, requests, and results
    - Parser: XML request extraction from LLM output
    - Registry: Activity catalog and discovery
    - Executor: Activity execution with validation and HITL
    - Loop: Integration with cognitive cycle

Example Usage:
    >>> from llmcore.agents.activities import (
    ...     ActivityLoop, ActivityRequest, ActivityRegistry
    ... )
    >>>
    >>> # Process LLM output
    >>> loop = ActivityLoop()
    >>> result = await loop.process_output('''
    ...     <activity_request>
    ...         <activity>file_read</activity>
    ...         <parameters><path>/etc/hosts</path></parameters>
    ...     </activity_request>
    ... ''')
    >>> print(result.observation)

Research Foundation:
    - ReAct: Yao et al., "ReAct: Synergizing Reasoning and Acting" (2022)
    - CoALA: Sumers et al., "Cognitive Architectures for Language Agents" (2023)

References:
    - Master Plan: Part III (Activity-Based Architecture)
    - Technical Spec: Section 5.4 (Activity System)
"""

from __future__ import annotations

# Executor
from .executor import (
    ActivityExecutor,
    ActivityValidator,
    HITLApprover,
    HITLDecision,
    HITLManagerAdapter,
    ValidationResult,
    create_hitl_approver,
)

# Loop
from .loop import (
    ActivityLoop,
    ActivityLoopConfig,
    process_llm_output,
)

# Parser
from .parser import (
    ActivityRequestParser,
    ParseResult,
    has_activity_request,
    parse_activity_requests,
)

# Prompts (G3 Phase 6)
from .prompts import (
    ACTIVITY_SYSTEM_PROMPT,
    generate_activity_prompt,
    get_activity_system_prompt,
)

# Registry
from .registry import (
    ActivityHandler,
    ActivityRegistry,
    ExecutionContext,
    RegisteredActivity,
    get_builtin_activities,
    get_default_registry,
    reset_default_registry,
)

# Schema
from .schema import (
    # Enums
    ActivityCategory,
    ActivityDefinition,
    ActivityExecution,
    ActivityLoopResult,
    ActivityRequest,
    ActivityResult,
    ActivityStatus,
    ExecutionTarget,
    # Data models
    ParameterSchema,
    ParameterType,
    RiskLevel,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.0.0"

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Enums
    "ActivityCategory",
    "RiskLevel",
    "ExecutionTarget",
    "ActivityStatus",
    "ParameterType",
    # Schema
    "ParameterSchema",
    "ActivityDefinition",
    "ActivityRequest",
    "ActivityResult",
    "ActivityExecution",
    "ActivityLoopResult",
    # Parser
    "ParseResult",
    "ActivityRequestParser",
    "parse_activity_requests",
    "has_activity_request",
    # Prompts (G3 Phase 6)
    "ACTIVITY_SYSTEM_PROMPT",
    "generate_activity_prompt",
    "get_activity_system_prompt",
    # Registry
    "ActivityRegistry",
    "RegisteredActivity",
    "ExecutionContext",
    "ActivityHandler",
    "get_builtin_activities",
    "get_default_registry",
    "reset_default_registry",
    # Executor
    "ValidationResult",
    "ActivityValidator",
    "HITLDecision",
    "HITLApprover",
    "HITLManagerAdapter",
    "ActivityExecutor",
    "create_hitl_approver",
    # Loop
    "ActivityLoopConfig",
    "ActivityLoop",
    "process_llm_output",
]


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================


def format_activities_for_prompt(
    categories: list = None,
    max_risk: RiskLevel = None,
    include_examples: bool = False,
) -> str:
    """
    Format available activities for inclusion in LLM prompts.

    Args:
        categories: Limit to specific categories (None for all)
        max_risk: Maximum risk level to include
        include_examples: Include usage examples

    Returns:
        Formatted string for prompt

    Example:
        >>> prompt_section = format_activities_for_prompt(
        ...     categories=[ActivityCategory.FILE_OPERATIONS],
        ...     max_risk=RiskLevel.MEDIUM,
        ... )
    """
    registry = get_default_registry()
    return registry.format_for_prompt(
        categories=categories,
        max_risk=max_risk,
        include_examples=include_examples,
    )


def list_available_activities(enabled_only: bool = True) -> list:
    """
    List all available activity names.

    Args:
        enabled_only: Only return enabled activities

    Returns:
        List of activity names
    """
    registry = get_default_registry()
    return registry.list_names(enabled_only=enabled_only)


def get_activity_info(name: str) -> dict:
    """
    Get information about a specific activity.

    Args:
        name: Activity name

    Returns:
        Dict with activity information or empty dict if not found
    """
    registry = get_default_registry()
    registered = registry.get(name)
    if not registered:
        return {}

    defn = registered.definition
    return {
        "name": defn.name,
        "category": defn.category.value,
        "description": defn.description,
        "risk_level": defn.risk_level.value,
        "parameters": [
            {
                "name": p.name,
                "type": p.type.value,
                "required": p.required,
                "description": p.description,
                "default": p.default,
            }
            for p in defn.parameters
        ],
        "requires_sandbox": defn.requires_sandbox,
        "timeout_seconds": defn.timeout_seconds,
        "enabled": registered.enabled,
        "source": registered.source,
    }
