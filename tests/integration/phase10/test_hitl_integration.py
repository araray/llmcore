"""
Phase 10 Integration Tests: HITL (Human-in-the-Loop) Workflows.

Tests that validate the llmcore HITL approval workflows function correctly
using the actual llmcore API.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from llmcore.agents.hitl.manager import (
    HITLManager,
    HITLConfig,
    HITLRequest,
    HITLResponse,
    HITLDecision,
    HITLStorageConfig,
    InMemoryHITLStore,
    FileHITLStore,
    RiskLevel,
    ApprovalStatus,
    create_hitl_manager,
)
from llmcore.agents.hitl.models import (
    ActivityInfo,
    RiskAssessment,
    RiskFactor,
)


class TestHITLManagerIntegration:
    """Test core HITL manager functionality."""

    def test_hitl_manager_creation(self) -> None:
        """Test that HITL manager can be created."""
        config = HITLConfig(
            enabled=True,
            default_timeout_seconds=60,
        )

        manager = HITLManager(config=config)

        assert manager is not None

    def test_hitl_config_defaults(self) -> None:
        """Test HITL config has sensible defaults."""
        config = HITLConfig()

        assert config.enabled == True  # Default enabled
        assert config.default_timeout_seconds > 0

    def test_hitl_manager_with_custom_config(self) -> None:
        """Test HITL manager accepts custom configuration."""
        config = HITLConfig(
            enabled=True,
            global_risk_threshold=RiskLevel.MEDIUM,
            default_timeout_seconds=120,
            safe_tools=["read_file", "list_files"],
            high_risk_tools=["delete_file"],
        )

        manager = HITLManager(config=config)

        assert manager.config.default_timeout_seconds == 120
        assert "read_file" in manager.config.safe_tools

    @pytest.mark.asyncio
    async def test_hitl_manager_initialize(self) -> None:
        """Test HITL manager async initialization."""
        config = HITLConfig(enabled=True)
        manager = HITLManager(config=config)

        await manager.initialize()

        try:
            # Manager should be initialized
            assert manager is not None
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_hitl_manager_safe_tool_check(self) -> None:
        """Test checking if a tool is safe."""
        config = HITLConfig(
            enabled=True,
            safe_tools=["read_file", "list_directory"],
        )
        manager = HITLManager(config=config)
        await manager.initialize()

        try:
            assert manager.is_safe_tool("read_file") == True
            assert manager.is_safe_tool("delete_file") == False
        finally:
            await manager.close()


class TestHITLModels:
    """Test HITL model structures."""

    def test_activity_info_creation(self) -> None:
        """Test creating ActivityInfo."""
        activity = ActivityInfo(
            activity_type="tool_call",
            parameters={"file": "/tmp/test.txt"},
            reason="Read a test file",
        )

        assert activity.activity_type == "tool_call"
        assert activity.parameters["file"] == "/tmp/test.txt"

    def test_risk_factor_creation(self) -> None:
        """Test creating RiskFactor."""
        factor = RiskFactor(
            name="file_deletion",
            level=RiskLevel.HIGH,
            reason="Deletes files permanently",
            weight=0.8,
        )

        assert factor.name == "file_deletion"
        assert factor.level == RiskLevel.HIGH

    def test_risk_assessment_creation(self) -> None:
        """Test creating RiskAssessment."""
        factor = RiskFactor(
            name="test_factor",
            level=RiskLevel.LOW,
        )

        assessment = RiskAssessment(
            overall_level=RiskLevel.LOW,
            factors=[factor],
            requires_approval=False,
        )

        assert assessment.overall_level == RiskLevel.LOW
        assert len(assessment.factors) == 1
        assert assessment.requires_approval == False

    def test_hitl_response_creation(self) -> None:
        """Test creating HITLResponse."""
        response = HITLResponse(
            request_id="test-request-id",
            approved=True,
            status=ApprovalStatus.APPROVED,
            feedback="Approved for execution",
        )

        assert response.approved == True
        assert response.status == ApprovalStatus.APPROVED
        assert response.feedback == "Approved for execution"


class TestHITLConfig:
    """Test HITL configuration options."""

    def test_config_safe_tools(self) -> None:
        """Test configuring safe tools."""
        config = HITLConfig(
            safe_tools=["read_file", "get_time", "list_files"],
        )

        assert len(config.safe_tools) == 3
        assert "read_file" in config.safe_tools

    def test_config_high_risk_tools(self) -> None:
        """Test configuring high-risk tools."""
        config = HITLConfig(
            high_risk_tools=["delete_file", "execute_shell"],
        )

        assert len(config.high_risk_tools) == 2
        assert "delete_file" in config.high_risk_tools

    def test_config_timeout_settings(self) -> None:
        """Test timeout configuration."""
        config = HITLConfig(
            default_timeout_seconds=300,
        )

        assert config.default_timeout_seconds == 300

    def test_config_risk_threshold(self) -> None:
        """Test risk threshold configuration."""
        config = HITLConfig(
            global_risk_threshold=RiskLevel.HIGH,
        )

        assert config.global_risk_threshold == RiskLevel.HIGH


class TestHITLStorage:
    """Test HITL state storage backends."""

    def test_in_memory_store_creation(self) -> None:
        """Test creating in-memory store."""
        store = InMemoryHITLStore()
        assert store is not None

    def test_file_store_creation(self, tmp_path: Path) -> None:
        """Test creating file-based store."""
        store_path = tmp_path / "hitl_state.json"
        store = FileHITLStore(storage_path=str(store_path))
        assert store is not None


class TestHITLRiskLevels:
    """Test risk level definitions."""

    def test_risk_level_values(self) -> None:
        """Test that all risk levels are defined."""
        levels = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM,
                  RiskLevel.HIGH, RiskLevel.CRITICAL]

        assert len(levels) == 5

    def test_risk_level_names(self) -> None:
        """Test risk level names."""
        assert RiskLevel.NONE.name == "NONE"
        assert RiskLevel.LOW.name == "LOW"
        assert RiskLevel.MEDIUM.name == "MEDIUM"
        assert RiskLevel.HIGH.name == "HIGH"
        assert RiskLevel.CRITICAL.name == "CRITICAL"


class TestApprovalStatus:
    """Test approval status definitions."""

    def test_approval_status_values(self) -> None:
        """Test that all approval statuses are defined."""
        statuses = [ApprovalStatus.PENDING, ApprovalStatus.APPROVED,
                    ApprovalStatus.REJECTED, ApprovalStatus.TIMEOUT]

        assert len(statuses) >= 3  # At minimum: pending, approved, rejected


class TestCreateHITLManagerFactory:
    """Test the HITL manager factory function."""

    def test_create_default_manager(self) -> None:
        """Test creating manager with defaults."""
        manager = create_hitl_manager()
        assert manager is not None

    def test_create_disabled_manager(self) -> None:
        """Test creating disabled manager."""
        manager = create_hitl_manager(enabled=False)
        assert manager is not None

    def test_create_manager_with_persist_path(self, tmp_path: Path) -> None:
        """Test creating manager with persistence."""
        persist_path = str(tmp_path / "hitl_state.json")
        manager = create_hitl_manager(
            persist_path=persist_path,
            storage_backend="file",
        )
        assert manager is not None


class TestHITLManagerAsync:
    """Test async HITL manager operations."""

    @pytest.mark.asyncio
    async def test_get_pending_requests(self) -> None:
        """Test getting pending requests."""
        config = HITLConfig(enabled=True)
        manager = HITLManager(config=config)
        await manager.initialize()

        try:
            pending = await manager.get_pending_requests()
            assert isinstance(pending, (list, tuple, dict))
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_get_statistics(self) -> None:
        """Test getting manager statistics."""
        config = HITLConfig(enabled=True)
        manager = HITLManager(config=config)
        await manager.initialize()

        try:
            stats = manager.get_statistics()
            assert isinstance(stats, dict)
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        """Test cleaning up expired requests."""
        config = HITLConfig(enabled=True)
        manager = HITLManager(config=config)
        await manager.initialize()

        try:
            cleaned = await manager.cleanup_expired()
            assert isinstance(cleaned, int)
            assert cleaned >= 0
        finally:
            await manager.close()


class TestHITLSessionApprovals:
    """Test session-level approval management."""

    @pytest.mark.asyncio
    async def test_grant_session_approval_for_tool(self) -> None:
        """Test granting session approval for a specific tool."""
        config = HITLConfig(enabled=True)
        manager = HITLManager(config=config)
        await manager.initialize()

        try:
            # Grant approval for a specific tool
            scope_id = manager.grant_session_approval(
                tool_name="read_file",
                max_risk_level=RiskLevel.LOW,
            )
            assert scope_id is not None
        finally:
            await manager.close()


class TestHITLDecisionModel:
    """Test HITL decision structures."""

    def test_decision_approved(self) -> None:
        """Test creating an approved decision."""
        decision = HITLDecision(
            status=ApprovalStatus.APPROVED,
        )

        assert decision.status == ApprovalStatus.APPROVED

    def test_decision_rejected_with_reason(self) -> None:
        """Test creating a rejected decision with reason."""
        decision = HITLDecision(
            status=ApprovalStatus.REJECTED,
            reason="Operation too risky",
        )

        assert decision.status == ApprovalStatus.REJECTED
        assert decision.reason == "Operation too risky"
