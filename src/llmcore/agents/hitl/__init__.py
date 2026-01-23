# src/llmcore/agents/hitl/__init__.py
"""
Human-In-The-Loop (HITL) System for LLMCore Agents.

Provides human oversight for agent activities through:
- Risk assessment and classification
- Approval workflow management
- Scope-based auto-approval
- State persistence
- Audit logging

Components:
- HITLManager: Central coordinator
- RiskAssessor: Evaluates activity risk
- ApprovalScopeManager: Manages approval scopes
- HITLCallback: UI integration interface

Usage:
    >>> from llmcore.agents.hitl import HITLManager, HITLConfig
    >>>
    >>> # Create manager
    >>> manager = HITLManager(
    ...     config=HITLConfig(enabled=True, global_risk_threshold="medium")
    ... )
    >>>
    >>> # Check if activity needs approval
    >>> decision = await manager.check_approval(
    ...     activity_type="bash_exec",
    ...     parameters={"command": "ls -la"},
    ... )
    >>>
    >>> if decision.is_approved:
    ...     # Execute activity
    ...     pass
    ... else:
    ...     print(f"Rejected: {decision.reason}")

References:
    - Master Plan: Section 20-22 (HITL System)
    - Technical Spec: Section 5.5 (HITL Architecture)
"""

from __future__ import annotations

# Models
from .models import (
    # Enums
    ApprovalScope,
    ApprovalStatus,
    HITLEventType,
    TimeoutPolicy,
    # Risk
    RiskAssessment,
    RiskFactor,
    # Request/Response
    ActivityInfo,
    HITLDecision,
    HITLRequest,
    HITLResponse,
    # Scopes
    PersistentScope,
    ScopeGrant,
    SessionScope,
    ToolScope,
    # Config
    HITLConfig,
    # Audit
    HITLAuditEvent,
)

# Risk Assessor
from .risk_assessor import (
    DangerousPattern,
    ResourceScope,
    RiskAssessor,
    RiskLevel,
    create_risk_assessor,
    quick_assess,
)

# Scope Manager
from .scope import (
    ApprovalScopeManager,
    ScopeConditionMatcher,
)

# State Persistence
from .state import (
    FileHITLStore,
    HITLStateStore,
    InMemoryHITLStore,
)

# Callbacks
from .callbacks import (
    AutoApproveCallback,
    ConsoleHITLCallback,
    HITLCallback,
    QueueHITLCallback,
)

# Manager
from .manager import (
    HITLAuditLogger,
    HITLManager,
    create_hitl_manager,
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
    # Manager (primary API)
    "HITLManager",
    "create_hitl_manager",
    "HITLAuditLogger",
    # Config
    "HITLConfig",
    # Enums
    "ApprovalScope",
    "ApprovalStatus",
    "HITLEventType",
    "RiskLevel",
    "TimeoutPolicy",
    # Risk Assessment
    "RiskAssessor",
    "RiskAssessment",
    "RiskFactor",
    "DangerousPattern",
    "ResourceScope",
    "create_risk_assessor",
    "quick_assess",
    # Scope Management
    "ApprovalScopeManager",
    "ScopeConditionMatcher",
    "ScopeGrant",
    "ToolScope",
    "SessionScope",
    "PersistentScope",
    # State Persistence
    "HITLStateStore",
    "InMemoryHITLStore",
    "FileHITLStore",
    # Callbacks
    "HITLCallback",
    "ConsoleHITLCallback",
    "AutoApproveCallback",
    "QueueHITLCallback",
    # Request/Response
    "HITLRequest",
    "HITLResponse",
    "HITLDecision",
    "ActivityInfo",
    # Audit
    "HITLAuditEvent",
]
