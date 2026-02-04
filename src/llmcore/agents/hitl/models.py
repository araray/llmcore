# src/llmcore/agents/hitl/models.py
"""
Human-In-The-Loop (HITL) Data Models.

Defines the core data structures for the HITL approval system:
- Request/Response models for approval workflow
- Risk assessment data structures
- Approval scope definitions
- Configuration models

References:
    - Master Plan: Section 20-22 (HITL System)
    - Technical Spec: Section 29.6 (Phase 5)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone, UTC
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if pydantic not available

    BaseModel = object  # type: ignore

    def Field(*args, **kwargs):  # type: ignore
        return kwargs.get("default")


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"  # Waiting for human response
    APPROVED = "approved"  # Approved by human
    REJECTED = "rejected"  # Rejected by human
    MODIFIED = "modified"  # Approved with modifications
    TIMEOUT = "timeout"  # Timed out waiting
    AUTO_APPROVED = "auto_approved"  # Auto-approved by scope/policy
    ESCALATED = "escalated"  # Escalated to admin


class TimeoutPolicy(str, Enum):
    """Policy for handling approval timeouts."""

    REJECT = "reject"  # Reject on timeout (default for HIGH risk)
    APPROVE = "approve"  # Approve on timeout (for LOW risk)
    ESCALATE = "escalate"  # Escalate to admin
    RETRY = "retry"  # Re-prompt user


class ApprovalScope(str, Enum):
    """Scope of approval grant."""

    SINGLE = "single"  # One action only
    TOOL = "tool"  # All uses of specific tool
    PATTERN = "pattern"  # Actions matching pattern
    CATEGORY = "category"  # All tools in category
    SESSION = "session"  # All actions in session
    PERSISTENT = "persistent"  # Persisted across sessions


class HITLEventType(str, Enum):
    """Types of HITL audit events."""

    REQUEST_CREATED = "request_created"
    REQUEST_APPROVED = "request_approved"
    REQUEST_REJECTED = "request_rejected"
    REQUEST_MODIFIED = "request_modified"
    REQUEST_TIMEOUT = "request_timeout"
    REQUEST_ESCALATED = "request_escalated"
    SCOPE_GRANTED = "scope_granted"
    SCOPE_REVOKED = "scope_revoked"


# =============================================================================
# RISK ASSESSMENT MODELS
# =============================================================================


class RiskFactor(BaseModel):
    """A single risk factor with its contribution to overall risk."""

    name: str = Field(..., description="Factor name")
    level: str = Field(..., description="Risk level contribution")
    reason: str = Field("", description="Reason for this factor")
    weight: float = Field(1.0, description="Weight for risk calculation")


class RiskAssessment(BaseModel):
    """Complete risk assessment for an activity."""

    overall_level: str = Field(
        ..., description="Overall risk level (none/low/medium/high/critical)"
    )
    factors: list[RiskFactor] = Field(default_factory=list, description="Contributing risk factors")
    requires_approval: bool = Field(False, description="Whether approval is required")
    reason: str = Field("", description="Human-readable reason")
    confidence: float = Field(1.0, description="Confidence in assessment (0-1)")
    dangerous_patterns: list[str] = Field(
        default_factory=list, description="Dangerous patterns detected"
    )

    @property
    def is_high_risk(self) -> bool:
        """Check if risk level is HIGH or CRITICAL."""
        return self.overall_level in ("high", "critical")


# =============================================================================
# APPROVAL REQUEST/RESPONSE
# =============================================================================


class ActivityInfo(BaseModel):
    """Minimal activity information for HITL request."""

    activity_type: str = Field(..., description="Activity type name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Activity parameters")
    reason: str | None = Field(None, description="Reasoning for this activity")


class HITLRequest(BaseModel):
    """A request for human approval."""

    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request ID")
    activity: ActivityInfo = Field(..., description="Activity requiring approval")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    context_summary: str = Field("", description="Brief context for decision-making")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation time")
    expires_at: datetime | None = Field(None, description="Request expiration time")
    status: ApprovalStatus = Field(ApprovalStatus.PENDING, description="Current status")
    session_id: str | None = Field(None, description="Session this request belongs to")
    user_id: str | None = Field(None, description="User who will approve")
    priority: int = Field(0, description="Priority (higher = more urgent)")

    def set_expiration(self, timeout_seconds: int) -> None:
        """Set expiration based on timeout."""
        self.expires_at = datetime.now(UTC) + timedelta(seconds=timeout_seconds)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        # Handle both timezone-aware and naive datetimes
        now = datetime.now(UTC)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return now > expires

    @property
    def time_remaining_seconds(self) -> int:
        """Get seconds remaining before expiration."""
        if self.expires_at is None:
            return -1
        # Handle both timezone-aware and naive datetimes
        now = datetime.now(UTC)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        delta = expires - now
        return max(0, int(delta.total_seconds()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "activity": {
                "activity_type": self.activity.activity_type,
                "parameters": self.activity.parameters,
                "reason": self.activity.reason,
            },
            "risk_assessment": {
                "overall_level": self.risk_assessment.overall_level,
                "factors": [
                    {"name": f.name, "level": f.level, "reason": f.reason, "weight": f.weight}
                    for f in self.risk_assessment.factors
                ],
                "requires_approval": self.risk_assessment.requires_approval,
                "reason": self.risk_assessment.reason,
                "dangerous_patterns": self.risk_assessment.dangerous_patterns,
            },
            "context_summary": self.context_summary,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HITLRequest:
        """Create from dictionary."""
        activity_data = data.get("activity", {})
        risk_data = data.get("risk_assessment", {})

        return cls(
            request_id=data["request_id"],
            activity=ActivityInfo(
                activity_type=activity_data.get("activity_type", "unknown"),
                parameters=activity_data.get("parameters", {}),
                reason=activity_data.get("reason"),
            ),
            risk_assessment=RiskAssessment(
                overall_level=risk_data.get("overall_level", "medium"),
                factors=[
                    RiskFactor(
                        name=f.get("name", ""),
                        level=f.get("level", "low"),
                        reason=f.get("reason", ""),
                        weight=f.get("weight", 1.0),
                    )
                    for f in risk_data.get("factors", [])
                ],
                requires_approval=risk_data.get("requires_approval", True),
                reason=risk_data.get("reason", ""),
                dangerous_patterns=risk_data.get("dangerous_patterns", []),
            ),
            context_summary=data.get("context_summary", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            status=ApprovalStatus(data.get("status", "pending")),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            priority=data.get("priority", 0),
        )


class HITLResponse(BaseModel):
    """Human response to an approval request."""

    request_id: str = Field(..., description="Request this responds to")
    # Support both 'approved' (bool) and 'status' (enum) for flexibility
    approved: bool = Field(True, description="Whether approved (legacy)")
    status: ApprovalStatus | None = Field(None, description="Detailed status")
    modified_parameters: dict[str, Any] | None = Field(
        None, description="Modified parameters if approved with changes"
    )
    # Alias for modified_parameters
    modified_params: dict[str, Any] | None = Field(
        None, description="Modified parameters (alias)"
    )
    feedback: str | None = Field(None, description="Human feedback/reason")
    responded_at: datetime = Field(default_factory=datetime.now, description="Response time")
    response_time_ms: int = Field(0, description="Response time in milliseconds")
    responder_id: str = Field("", description="Who responded")
    # Accept both ApprovalScope enum (legacy) and ScopeGrant object (new)
    scope_grant: Any | None = Field(None, description="Scope to grant for similar actions")

    def __init__(self, **data):
        """Initialize with flexible parameter handling."""
        # Handle modified_params alias
        if "modified_params" in data and data["modified_params"] is not None:
            if data.get("modified_parameters") is None:
                data["modified_parameters"] = data["modified_params"]
        # Handle status -> approved mapping
        if "status" in data and data["status"] is not None:
            status = data["status"]
            if isinstance(status, ApprovalStatus):
                data["approved"] = status in (
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.AUTO_APPROVED,
                    ApprovalStatus.MODIFIED,
                )
        super().__init__(**data)

    def get_scope_grant_as_model(self) -> ScopeGrant | None:
        """Get scope_grant as ScopeGrant model, converting from enum if needed."""
        if self.scope_grant is None:
            return None
        if isinstance(self.scope_grant, ScopeGrant):
            return self.scope_grant
        if isinstance(self.scope_grant, ApprovalScope):
            # Convert enum to ScopeGrant with defaults
            return ScopeGrant(
                tool_name="*",  # Unknown tool, will need context
                scope_type=self.scope_grant,
                max_risk_level="medium",
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "request_id": self.request_id,
            "approved": self.approved,
            "modified_parameters": self.modified_parameters or self.modified_params,
            "feedback": self.feedback,
            "responded_at": self.responded_at.isoformat(),
            "responder_id": self.responder_id,
            "response_time_ms": self.response_time_ms,
        }
        if self.status is not None:
            result["status"] = self.status.value
        if self.scope_grant is not None:
            if isinstance(self.scope_grant, ScopeGrant):
                result["scope_grant"] = self.scope_grant.to_dict()
            elif isinstance(self.scope_grant, ApprovalScope):
                result["scope_grant"] = self.scope_grant.value
            else:
                result["scope_grant"] = str(self.scope_grant)
        return result


class HITLDecision(BaseModel):
    """Final decision from HITL process."""

    status: ApprovalStatus = Field(..., description="Decision status")
    reason: str = Field("", description="Reason for decision")
    modified_parameters: dict[str, Any] | None = Field(
        None, description="Modified parameters if applicable"
    )
    request: HITLRequest | None = Field(None, description="Original request")
    response: HITLResponse | None = Field(None, description="Human response")
    scope_id: str | None = Field(None, description="Approval scope if used")

    @property
    def is_approved(self) -> bool:
        """Check if action was approved."""
        return self.status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.AUTO_APPROVED,
            ApprovalStatus.MODIFIED,
        )


# =============================================================================
# APPROVAL SCOPE MODELS
# =============================================================================


class ToolScope(BaseModel):
    """Approval scope for a specific tool."""

    tool_name: str = Field(..., description="Tool/activity name")
    approved: bool = Field(True, description="Whether approved")
    conditions: dict[str, Any] | None = Field(
        None, description="Conditions for approval (e.g., path_pattern)"
    )
    granted_at: datetime = Field(default_factory=datetime.now, description="When scope was granted")
    granted_by: str = Field("", description="Who granted the scope")
    max_risk_level: str = Field("medium", description="Maximum risk level allowed")


class SessionScope(BaseModel):
    """Approvals valid for current session only."""

    session_id: str = Field(..., description="Session ID")
    approved_tools: list[ToolScope] = Field(default_factory=list, description="Tools with scope")
    approved_patterns: list[str] = Field(
        default_factory=list, description="Approved patterns (e.g., 'read any file in /workspace')"
    )
    session_approval: bool = Field(False, description="Full session approval granted")
    expires_at: datetime | None = Field(None, description="Session scope expiration")
    created_at: datetime = Field(default_factory=datetime.now, description="Scope creation time")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "approved_tools": [
                {
                    "tool_name": t.tool_name,
                    "approved": t.approved,
                    "conditions": t.conditions,
                    "granted_at": t.granted_at.isoformat(),
                    "granted_by": t.granted_by,
                    "max_risk_level": t.max_risk_level,
                }
                for t in self.approved_tools
            ],
            "approved_patterns": self.approved_patterns,
            "session_approval": self.session_approval,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionScope:
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            approved_tools=[
                ToolScope(
                    tool_name=t["tool_name"],
                    approved=t.get("approved", True),
                    conditions=t.get("conditions"),
                    granted_at=datetime.fromisoformat(t["granted_at"])
                    if "granted_at" in t
                    else datetime.now(),
                    granted_by=t.get("granted_by", ""),
                    max_risk_level=t.get("max_risk_level", "medium"),
                )
                for t in data.get("approved_tools", [])
            ],
            approved_patterns=data.get("approved_patterns", []),
            session_approval=data.get("session_approval", False),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
        )


class PersistentScope(BaseModel):
    """Approvals persisted across sessions."""

    user_id: str = Field(..., description="User ID")
    approved_tools: list[ToolScope] = Field(
        default_factory=list, description="Persistent tool approvals"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "approved_tools": [
                {
                    "tool_name": t.tool_name,
                    "approved": t.approved,
                    "conditions": t.conditions,
                    "granted_at": t.granted_at.isoformat(),
                    "granted_by": t.granted_by,
                    "max_risk_level": t.max_risk_level,
                }
                for t in self.approved_tools
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ScopeGrant(BaseModel):
    """
    Request to grant an approval scope.

    Used by callbacks to communicate user's desire to grant a scope
    for future similar activities.
    """

    tool_name: str = Field(..., description="Tool/activity name to grant scope for")
    scope_type: ApprovalScope = Field(ApprovalScope.SESSION, description="Type of scope to grant")
    max_risk_level: str = Field("medium", description="Maximum risk level allowed under this scope")
    conditions: dict[str, Any] | None = Field(
        None, description="Optional conditions (e.g., path_pattern)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "scope_type": self.scope_type.value,
            "max_risk_level": self.max_risk_level,
            "conditions": self.conditions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScopeGrant:
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            scope_type=ApprovalScope(data.get("scope_type", "session")),
            max_risk_level=data.get("max_risk_level", "medium"),
            conditions=data.get("conditions"),
        )


# =============================================================================
# CONFIGURATION
# =============================================================================


class HITLStorageConfig(BaseModel):
    """
    Storage backend configuration for HITL system.

    Supports multiple storage backends:
    - memory: In-memory storage (testing/ephemeral)
    - file: JSON file-based persistence
    - sqlite: SQLite database (development/single-user)
    - postgres: PostgreSQL database (production/multi-user)

    Example:
        >>> # SQLite backend
        >>> storage = HITLStorageConfig(backend="sqlite", sqlite_path="/tmp/hitl.db")
        >>>
        >>> # PostgreSQL backend
        >>> storage = HITLStorageConfig(
        ...     backend="postgres",
        ...     postgres_url="postgresql://user:pass@localhost/db"
        ... )
    """

    backend: str = Field(
        "memory",
        description="Storage backend type: memory, file, sqlite, postgres",
    )

    # File backend options
    file_path: str | None = Field(
        None,
        description="Path for file-based storage (used when backend='file')",
    )

    # SQLite options
    sqlite_path: str = Field(
        "~/.local/share/llmcore/hitl.db",
        description="Path to SQLite database file",
    )

    # PostgreSQL options
    postgres_url: str | None = Field(
        None,
        description="PostgreSQL connection URL",
    )
    postgres_min_pool_size: int = Field(
        2,
        description="Minimum connection pool size for PostgreSQL",
    )
    postgres_max_pool_size: int = Field(
        10,
        description="Maximum connection pool size for PostgreSQL",
    )
    postgres_table_prefix: str = Field(
        "",
        description="Table name prefix for multi-tenancy",
    )

    def validate_backend(self) -> None:
        """
        Validate that required fields are set for the selected backend.

        Raises:
            ValueError: If required configuration is missing
        """
        if self.backend == "postgres" and not self.postgres_url:
            raise ValueError("postgres_url is required when backend='postgres'")
        if self.backend == "file" and not self.file_path:
            raise ValueError("file_path is required when backend='file'")


class HITLConfig(BaseModel):
    """HITL system configuration."""

    enabled: bool = Field(True, description="Whether HITL is enabled")
    global_risk_threshold: str = Field(
        "medium", description="Minimum risk level requiring approval"
    )

    # Timeout settings
    default_timeout_seconds: int = Field(300, description="Default approval timeout (5 minutes)")
    timeout_policy: TimeoutPolicy = Field(
        TimeoutPolicy.REJECT, description="Default timeout handling policy"
    )
    timeout_policies_by_risk: dict[str, TimeoutPolicy] = Field(
        default_factory=lambda: {
            "low": TimeoutPolicy.APPROVE,
            "medium": TimeoutPolicy.REJECT,
            "high": TimeoutPolicy.REJECT,
            "critical": TimeoutPolicy.REJECT,
        },
        description="Timeout policy per risk level",
    )

    # Tool classifications
    safe_tools: list[str] = Field(
        default_factory=lambda: ["final_answer", "respond_to_user", "think_aloud"],
        description="Tools that never require approval",
    )
    low_risk_tools: list[str] = Field(
        default_factory=lambda: ["file_read", "file_search", "list_directory"],
        description="Low risk tools",
    )
    high_risk_tools: list[str] = Field(
        default_factory=lambda: ["bash_exec", "python_exec", "file_delete"],
        description="High risk tools",
    )
    critical_tools: list[str] = Field(
        default_factory=lambda: ["execute_sudo", "drop_database"],
        description="Critical risk tools",
    )

    # Batch approval
    batch_similar_requests: bool = Field(True, description="Batch similar requests together")
    batch_window_seconds: int = Field(5, description="Window for batching similar requests")

    # Audit
    audit_logging_enabled: bool = Field(True, description="Enable audit logging")
    audit_log_path: str | None = Field(None, description="Path for audit log file")

    # Storage backend configuration (NEW for Phase 7.2)
    storage: HITLStorageConfig = Field(
        default_factory=HITLStorageConfig,
        description="Storage backend configuration",
    )


# =============================================================================
# AUDIT EVENT
# =============================================================================


class HITLAuditEvent(BaseModel):
    """Audit event for HITL actions."""

    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Event ID")
    event_type: HITLEventType = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event time")
    request_id: str | None = Field(None, description="Related request ID")
    user_id: str | None = Field(None, description="User involved")
    session_id: str | None = Field(None, description="Session ID")
    activity_type: str | None = Field(None, description="Activity type")
    risk_level: str | None = Field(None, description="Risk level")
    decision: str | None = Field(None, description="Decision made")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "activity_type": self.activity_type,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "details": self.details,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ApprovalStatus",
    "TimeoutPolicy",
    "ApprovalScope",
    "HITLEventType",
    # Risk
    "RiskFactor",
    "RiskAssessment",
    # Request/Response
    "ActivityInfo",
    "HITLRequest",
    "HITLResponse",
    "HITLDecision",
    # Scopes
    "ToolScope",
    "SessionScope",
    "PersistentScope",
    "ScopeGrant",
    # Config
    "HITLStorageConfig",
    "HITLConfig",
    # Audit
    "HITLAuditEvent",
]
