# tests/agents/hitl/test_models.py
"""
Tests for HITL data models.

Tests:
- Enum serialization
- Request/Response models
- Risk assessment models
- Config validation
- Serialization/deserialization
"""

from datetime import datetime, timedelta, timezone

from llmcore.agents.hitl import (
    ActivityInfo,
    ApprovalScope,
    ApprovalStatus,
    HITLConfig,
    HITLDecision,
    HITLEventType,
    HITLRequest,
    HITLResponse,
    PersistentScope,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    SessionScope,
    TimeoutPolicy,
    ToolScope,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEnums:
    """Test enum serialization and values."""

    def test_approval_status_values(self):
        """ApprovalStatus should have expected values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.MODIFIED.value == "modified"
        assert ApprovalStatus.TIMEOUT.value == "timeout"
        assert ApprovalStatus.AUTO_APPROVED.value == "auto_approved"

    def test_timeout_policy_values(self):
        """TimeoutPolicy should have expected values."""
        assert TimeoutPolicy.REJECT.value == "reject"
        assert TimeoutPolicy.APPROVE.value == "approve"
        assert TimeoutPolicy.ESCALATE.value == "escalate"
        assert TimeoutPolicy.RETRY.value == "retry"

    def test_approval_scope_values(self):
        """ApprovalScope should have expected values."""
        assert ApprovalScope.SINGLE.value == "single"
        assert ApprovalScope.TOOL.value == "tool"
        assert ApprovalScope.PATTERN.value == "pattern"
        assert ApprovalScope.CATEGORY.value == "category"
        assert ApprovalScope.SESSION.value == "session"

    def test_hitl_event_type_values(self):
        """HITLEventType should have expected values."""
        assert HITLEventType.REQUEST_CREATED.value == "request_created"
        assert HITLEventType.REQUEST_APPROVED.value == "request_approved"
        assert HITLEventType.REQUEST_REJECTED.value == "request_rejected"


# =============================================================================
# RISK FACTOR TESTS
# =============================================================================


class TestRiskFactor:
    """Test RiskFactor model."""

    def test_risk_factor_creation(self):
        """Should create risk factor with all fields."""
        factor = RiskFactor(name="tool_risk", level="high", weight=1.0, reason="High risk tool")
        assert factor.name == "tool_risk"
        assert factor.level == "high"
        assert factor.weight == 1.0
        assert factor.reason == "High risk tool"

    def test_risk_factor_defaults(self):
        """Should use default values."""
        factor = RiskFactor(name="test", level="low")
        assert factor.weight == 1.0
        assert factor.reason == ""

    def test_risk_factor_serialization(self):
        """Should serialize to dict via model_dump."""
        factor = RiskFactor(name="test", level="medium", weight=0.5, reason="test reason")
        data = factor.model_dump()
        assert data["name"] == "test"
        assert data["level"] == "medium"
        assert data["weight"] == 0.5
        assert data["reason"] == "test reason"

    def test_risk_factor_from_dict(self):
        """Should deserialize from dict via Pydantic constructor."""
        data = {"name": "test", "level": "high", "weight": 2.0, "reason": "high risk"}
        factor = RiskFactor(**data)
        assert factor.name == "test"
        assert factor.level == "high"
        assert factor.weight == 2.0


# =============================================================================
# RISK ASSESSMENT TESTS
# =============================================================================


class TestRiskAssessment:
    """Test RiskAssessment model."""

    def test_risk_assessment_creation(self):
        """Should create risk assessment."""
        factor = RiskFactor(name="test", level="medium")
        assessment = RiskAssessment(
            overall_level="medium",
            factors=[factor],
            requires_approval=True,
            dangerous_patterns=["rm -rf"],
        )
        assert assessment.overall_level == "medium"
        assert len(assessment.factors) == 1
        assert assessment.requires_approval
        assert "rm -rf" in assessment.dangerous_patterns

    def test_risk_assessment_defaults(self):
        """Should use default values."""
        assessment = RiskAssessment(overall_level="low")
        assert assessment.factors == []
        assert not assessment.requires_approval
        assert assessment.dangerous_patterns == []

    def test_risk_assessment_serialization(self):
        """Should round-trip through serialization."""
        factor = RiskFactor(name="test", level="high", weight=1.5, reason="test")
        original = RiskAssessment(
            overall_level="high",
            factors=[factor],
            requires_approval=True,
            dangerous_patterns=["pattern1", "pattern2"],
        )
        data = original.model_dump()
        restored = RiskAssessment(**data)

        assert restored.overall_level == original.overall_level
        assert restored.requires_approval == original.requires_approval
        assert len(restored.factors) == len(original.factors)
        assert restored.dangerous_patterns == original.dangerous_patterns


# =============================================================================
# ACTIVITY INFO TESTS
# =============================================================================


class TestActivityInfo:
    """Test ActivityInfo model."""

    def test_activity_info_creation(self):
        """Should create activity info."""
        info = ActivityInfo(
            activity_type="bash_exec", parameters={"command": "ls"}, reason="List files"
        )
        assert info.activity_type == "bash_exec"
        assert info.parameters["command"] == "ls"
        assert info.reason == "List files"

    def test_activity_info_defaults(self):
        """Should use default values."""
        info = ActivityInfo(activity_type="test", parameters={})
        assert info.reason is None

    def test_activity_info_serialization(self):
        """Should serialize correctly."""
        info = ActivityInfo(
            activity_type="file_write",
            parameters={"path": "/tmp/test.txt", "content": "hello"},
            reason="Write test file",
        )
        data = info.model_dump()
        assert data["activity_type"] == "file_write"
        assert data["parameters"]["path"] == "/tmp/test.txt"


# =============================================================================
# HITL REQUEST TESTS
# =============================================================================


class TestHITLRequest:
    """Test HITLRequest model."""

    def test_request_creation(self):
        """Should create request with generated ID."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium")
        request = HITLRequest(
            activity=activity, risk_assessment=risk, session_id="sess-123", user_id="user-456"
        )
        assert request.request_id  # Auto-generated
        assert request.activity.activity_type == "test"
        assert request.status == ApprovalStatus.PENDING
        assert request.session_id == "sess-123"

    def test_request_expiration(self):
        """Should handle expiration correctly."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="low")
        request = HITLRequest(activity=activity, risk_assessment=risk)

        # Set expiration
        request.set_expiration(60)
        assert request.expires_at is not None
        assert not request.is_expired
        assert request.time_remaining_seconds > 0

    def test_request_expired(self):
        """Should detect expired requests."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="low")
        request = HITLRequest(activity=activity, risk_assessment=risk)

        # Set past expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert request.is_expired
        assert request.time_remaining_seconds == 0

    def test_request_serialization(self):
        """Should round-trip through serialization."""
        activity = ActivityInfo(activity_type="bash_exec", parameters={"cmd": "ls"})
        risk = RiskAssessment(overall_level="high", requires_approval=True)
        original = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            context_summary="Test context",
            session_id="sess-1",
            user_id="user-1",
        )
        original.set_expiration(300)

        data = original.to_dict()
        restored = HITLRequest.from_dict(data)

        assert restored.request_id == original.request_id
        assert restored.activity.activity_type == original.activity.activity_type
        assert restored.status == original.status


# =============================================================================
# HITL RESPONSE TESTS
# =============================================================================


class TestHITLResponse:
    """Test HITLResponse model."""

    def test_response_creation(self):
        """Should create response."""
        response = HITLResponse(
            request_id="req-123", approved=True, responder_id="user-456", feedback="Looks good"
        )
        assert response.request_id == "req-123"
        assert response.approved
        assert response.responder_id == "user-456"

    def test_response_with_modifications(self):
        """Should handle modified parameters."""
        response = HITLResponse(
            request_id="req-123", approved=True, modified_parameters={"path": "/workspace/safe.txt"}
        )
        assert response.modified_parameters["path"] == "/workspace/safe.txt"

    def test_response_with_scope_grant(self):
        """Should handle scope grants."""
        response = HITLResponse(request_id="req-123", approved=True, scope_grant=ApprovalScope.TOOL)
        assert response.scope_grant == ApprovalScope.TOOL

    def test_response_serialization(self):
        """Should serialize with enums."""
        response = HITLResponse(
            request_id="req-123", approved=True, scope_grant=ApprovalScope.SESSION
        )
        data = response.to_dict()
        assert data["scope_grant"] == "session"


# =============================================================================
# HITL DECISION TESTS
# =============================================================================


class TestHITLDecision:
    """Test HITLDecision model."""

    def test_decision_approved(self):
        """Should correctly identify approved status."""
        decision = HITLDecision(status=ApprovalStatus.APPROVED, reason="User approved")
        assert decision.is_approved
        assert decision.status == ApprovalStatus.APPROVED

    def test_decision_auto_approved(self):
        """Auto-approved should be considered approved."""
        decision = HITLDecision(status=ApprovalStatus.AUTO_APPROVED, reason="Low risk")
        assert decision.is_approved

    def test_decision_modified_approved(self):
        """Modified should be considered approved."""
        decision = HITLDecision(
            status=ApprovalStatus.MODIFIED,
            reason="Changed parameters",
            modified_parameters={"path": "/safe/path"},
        )
        assert decision.is_approved
        assert decision.modified_parameters["path"] == "/safe/path"

    def test_decision_rejected(self):
        """Rejected should not be approved."""
        decision = HITLDecision(status=ApprovalStatus.REJECTED, reason="Too risky")
        assert not decision.is_approved

    def test_decision_timeout(self):
        """Timeout should not be approved."""
        decision = HITLDecision(status=ApprovalStatus.TIMEOUT, reason="No response")
        assert not decision.is_approved


# =============================================================================
# SCOPE MODELS TESTS
# =============================================================================


class TestScopeModels:
    """Test scope-related models."""

    def test_tool_scope_creation(self):
        """Should create tool scope."""
        scope = ToolScope(
            tool_name="bash_exec",
            max_risk_level=RiskLevel.MEDIUM,
            conditions={"path_pattern": "/workspace/*"},
            granted_by="user-1",
        )
        assert scope.tool_name == "bash_exec"
        assert scope.max_risk_level == RiskLevel.MEDIUM
        assert scope.conditions["path_pattern"] == "/workspace/*"

    def test_session_scope_creation(self):
        """Should create session scope."""
        scope = SessionScope(session_id="sess-123")
        assert scope.session_id == "sess-123"
        assert scope.approved_tools == []
        assert scope.approved_patterns == []
        assert not scope.session_approval

    def test_session_scope_serialization(self):
        """Should serialize session scope."""
        tool = ToolScope(tool_name="test", max_risk_level=RiskLevel.LOW)
        scope = SessionScope(session_id="sess-123", approved_tools=[tool], session_approval=False)
        data = scope.model_dump()
        assert data["session_id"] == "sess-123"
        assert len(data["approved_tools"]) == 1
        assert data["approved_tools"][0]["tool_name"] == "test"

    def test_persistent_scope_creation(self):
        """Should create persistent scope."""
        scope = PersistentScope(user_id="user-123", approved_tools=[])
        assert scope.user_id == "user-123"
        assert scope.approved_tools == []


# =============================================================================
# HITL CONFIG TESTS
# =============================================================================


class TestHITLConfig:
    """Test HITLConfig model."""

    def test_config_defaults(self):
        """Should have sensible defaults."""
        config = HITLConfig()
        assert config.enabled
        assert config.global_risk_threshold == "medium"
        assert config.default_timeout_seconds == 300
        assert config.timeout_policy == TimeoutPolicy.REJECT

    def test_config_custom(self):
        """Should accept custom values."""
        config = HITLConfig(
            enabled=False,
            global_risk_threshold="high",
            default_timeout_seconds=600,
            timeout_policy=TimeoutPolicy.APPROVE,
        )
        assert not config.enabled
        assert config.global_risk_threshold == "high"
        assert config.default_timeout_seconds == 600
        assert config.timeout_policy == TimeoutPolicy.APPROVE

    def test_config_safe_tools(self):
        """Should configure safe tools."""
        config = HITLConfig(safe_tools=["respond", "think", "final_answer"])
        assert "respond" in config.safe_tools
        assert "think" in config.safe_tools

    def test_config_timeout_by_risk(self):
        """Should configure timeout policies by risk."""
        config = HITLConfig(
            timeout_policies_by_risk={
                "low": TimeoutPolicy.APPROVE,
                "medium": TimeoutPolicy.REJECT,
                "high": TimeoutPolicy.REJECT,
                "critical": TimeoutPolicy.REJECT,
            }
        )
        assert config.timeout_policies_by_risk["low"] == TimeoutPolicy.APPROVE

    def test_config_negative_timeout(self):
        """Negative timeout is currently accepted (no validation)."""
        # Note: If validation is needed, add a Pydantic validator to HITLConfig
        config = HITLConfig(default_timeout_seconds=-1)
        assert config.default_timeout_seconds == -1


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameters(self):
        """Should handle empty parameters."""
        info = ActivityInfo(activity_type="test", parameters={})
        assert info.parameters == {}

    def test_none_values(self):
        """Should handle None values gracefully."""
        response = HITLResponse(
            request_id="test", approved=True, modified_parameters=None, scope_grant=None
        )
        assert response.modified_parameters is None
        assert response.scope_grant is None

    def test_serialization_with_none(self):
        """Should serialize None values correctly."""
        response = HITLResponse(request_id="test", approved=False, feedback=None)
        data = response.to_dict()
        assert data["request_id"] == "test"

    def test_uuid_generation(self):
        """Request IDs should be unique."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="low")
        r1 = HITLRequest(activity=activity, risk_assessment=risk)
        r2 = HITLRequest(activity=activity, risk_assessment=risk)
        assert r1.request_id != r2.request_id
