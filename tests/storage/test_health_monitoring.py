# tests/storage/test_health_monitoring.py
"""
Tests for the Health Monitoring System (Phase 1 - PRIMORDIUM).

Tests cover:
- Health check execution and result handling
- Circuit breaker state machine
- Health monitor background monitoring
- Composite health manager
- Health check functions for different backends
"""

import sys
from pathlib import Path

# Add storage module to path for direct imports (avoids llmcore import chain issues)
_storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
if str(_storage_path) not in sys.path:
    sys.path.insert(0, str(_storage_path))

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import time

from health import (
    HealthStatus,
    CircuitState,
    HealthCheckResult,
    StorageHealthReport,
    HealthConfig,
    CircuitBreaker,
    StorageHealthMonitor,
    StorageHealthManager,
    create_postgres_health_check,
    create_sqlite_health_check,
    create_chromadb_health_check,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def health_config() -> HealthConfig:
    """Create a test health config with short intervals."""
    return HealthConfig(
        check_interval_seconds=0.1,  # Fast for testing
        check_timeout_seconds=1.0,
        failure_threshold=3,
        recovery_timeout_seconds=0.5,
        half_open_max_requests=1,
        latency_warning_ms=100.0,
        latency_critical_ms=500.0,
        history_size=10,
        enabled=True,
        log_health_checks=False  # Reduce test noise
    )


@pytest.fixture
def healthy_check_fn() -> AsyncMock:
    """Create a health check function that always returns healthy."""
    async def check():
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
            details={"test": "healthy"}
        )
    return check


@pytest.fixture
def unhealthy_check_fn() -> AsyncMock:
    """Create a health check function that always returns unhealthy."""
    async def check():
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            latency_ms=50.0,
            error_message="Test failure"
        )
    return check


@pytest.fixture
def failing_check_fn():
    """Create a health check function that raises exceptions."""
    async def check():
        raise ConnectionError("Database connection failed")
    return check


# =============================================================================
# UNIT TESTS - HEALTH STATUS
# =============================================================================

class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test that all expected status values exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"
        assert HealthStatus.CIRCUIT_OPEN.value == "circuit_open"


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_state_values(self):
        """Test that all expected state values exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


# =============================================================================
# UNIT TESTS - HEALTH CHECK RESULT
# =============================================================================

class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_healthy_result(self):
        """Test creating a healthy result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            latency_ms=5.5,
            details={"query": "SELECT 1"}
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 5.5
        assert result.error_message is None
        assert result.details["query"] == "SELECT 1"
        assert result.timestamp is not None

    def test_unhealthy_result(self):
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            latency_ms=1000.0,
            error_message="Connection timeout"
        )

        assert result.status == HealthStatus.UNHEALTHY
        assert result.error_message == "Connection timeout"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
            details={"test": True}
        )

        d = result.to_dict()

        assert d["status"] == "healthy"
        assert d["latency_ms"] == 10.0
        assert d["details"]["test"] is True
        assert "timestamp" in d


# =============================================================================
# UNIT TESTS - HEALTH CONFIG
# =============================================================================

class TestHealthConfig:
    """Tests for HealthConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealthConfig()

        assert config.check_interval_seconds == 30.0
        assert config.check_timeout_seconds == 5.0
        assert config.failure_threshold == 3
        assert config.recovery_timeout_seconds == 60.0
        assert config.enabled is True

    def test_custom_config(self, health_config):
        """Test custom configuration values."""
        assert health_config.check_interval_seconds == 0.1
        assert health_config.failure_threshold == 3


# =============================================================================
# UNIT TESTS - CIRCUIT BREAKER
# =============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self, health_config) -> CircuitBreaker:
        """Create a circuit breaker with test config."""
        return CircuitBreaker(health_config)

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, breaker):
        """Test that circuit starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_allow_request_when_closed(self, breaker):
        """Test that requests are allowed when circuit is closed."""
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_record_success_resets_failures(self, breaker):
        """Test that success resets failure count."""
        # Record some failures
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker._failure_count == 2

        # Record success
        await breaker.record_success()
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        """Test that circuit opens after failure threshold."""
        # Record failures up to threshold
        for _ in range(3):  # failure_threshold = 3
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_blocks_requests_when_open(self, breaker):
        """Test that requests are blocked when circuit is open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        assert await breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, breaker):
        """Test that circuit transitions to half-open after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(0.6)  # recovery_timeout_seconds = 0.5

        # Next request should be allowed (half-open)
        result = await breaker.allow_request()
        assert result is True
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self, breaker):
        """Test that success in half-open transitions to closed."""
        # Get to half-open state
        for _ in range(3):
            await breaker.record_failure()
        await asyncio.sleep(0.6)
        await breaker.allow_request()  # Triggers transition to half-open

        # Record success
        await breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, breaker):
        """Test that failure in half-open transitions back to open."""
        # Get to half-open state
        for _ in range(3):
            await breaker.record_failure()
        await asyncio.sleep(0.6)
        await breaker.allow_request()  # Triggers transition to half-open

        # Record failure
        await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self, breaker):
        """Test manual circuit reset."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0


# =============================================================================
# UNIT TESTS - STORAGE HEALTH MONITOR
# =============================================================================

class TestStorageHealthMonitor:
    """Tests for StorageHealthMonitor class."""

    @pytest.mark.asyncio
    async def test_check_health_success(self, health_config, healthy_check_fn):
        """Test successful health check."""
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        result = await monitor.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert monitor.is_healthy is True
        assert monitor._total_checks == 1

    @pytest.mark.asyncio
    async def test_check_health_failure(self, health_config, failing_check_fn):
        """Test health check with exception."""
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=failing_check_fn,
            config=health_config
        )

        result = await monitor.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Database connection failed" in result.error_message
        assert monitor._total_failures == 1

    @pytest.mark.asyncio
    async def test_check_health_timeout(self, health_config):
        """Test health check timeout."""
        async def slow_check():
            await asyncio.sleep(5)  # Longer than timeout
            return HealthCheckResult(status=HealthStatus.HEALTHY, latency_ms=5000)

        config = HealthConfig(check_timeout_seconds=0.1, log_health_checks=False)
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=slow_check,
            config=config
        )

        result = await monitor.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_latency_degradation_warning(self, health_config):
        """Test that high latency is flagged as degraded."""
        async def slow_check():
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=600.0  # Above critical threshold of 500ms
            )

        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=slow_check,
            config=health_config
        )

        result = await monitor.check_health()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_history_retention(self, health_config, healthy_check_fn):
        """Test that history is retained up to limit."""
        health_config.history_size = 5
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        # Run more checks than history size
        for _ in range(10):
            await monitor.check_health()

        assert len(monitor._check_history) == 5
        assert monitor._total_checks == 10

    @pytest.mark.asyncio
    async def test_get_report(self, health_config, healthy_check_fn):
        """Test generating health report."""
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        await monitor.check_health()
        report = monitor.get_report()

        assert isinstance(report, StorageHealthReport)
        assert report.backend_name == "test_backend"
        assert report.backend_type == "session"
        assert report.total_checks == 1
        assert report.uptime_percentage == 100.0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_config, healthy_check_fn):
        """Test starting and stopping background monitoring."""
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        await monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        # Let it run briefly
        await asyncio.sleep(0.2)

        await monitor.stop()
        assert monitor._running is False

        # Should have run at least one check
        assert monitor._total_checks >= 1


# =============================================================================
# UNIT TESTS - STORAGE HEALTH MANAGER
# =============================================================================

class TestStorageHealthManager:
    """Tests for StorageHealthManager class."""

    @pytest.mark.asyncio
    async def test_register_monitor(self, health_config, healthy_check_fn):
        """Test registering a health monitor."""
        manager = StorageHealthManager()
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        manager.register_monitor(monitor)

        assert "test_backend" in manager._monitors

    @pytest.mark.asyncio
    async def test_unregister_monitor(self, health_config, healthy_check_fn):
        """Test unregistering a health monitor."""
        manager = StorageHealthManager()
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )

        manager.register_monitor(monitor)
        manager.unregister_monitor("test_backend")

        assert "test_backend" not in manager._monitors

    @pytest.mark.asyncio
    async def test_is_healthy_all_healthy(self, health_config, healthy_check_fn):
        """Test is_healthy when all backends are healthy."""
        manager = StorageHealthManager()

        for name in ["session", "vector"]:
            monitor = StorageHealthMonitor(
                backend_name=name,
                backend_type=name,
                check_fn=healthy_check_fn,
                config=health_config
            )
            manager.register_monitor(monitor)
            await monitor.check_health()  # Populate history

        assert manager.is_healthy() is True

    @pytest.mark.asyncio
    async def test_is_healthy_specific_backend(self, health_config, healthy_check_fn, failing_check_fn):
        """Test is_healthy for specific backend."""
        manager = StorageHealthManager()

        # Add healthy monitor
        healthy_monitor = StorageHealthMonitor(
            backend_name="healthy_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )
        manager.register_monitor(healthy_monitor)
        await healthy_monitor.check_health()

        # Add unhealthy monitor
        unhealthy_monitor = StorageHealthMonitor(
            backend_name="unhealthy_backend",
            backend_type="vector",
            check_fn=failing_check_fn,
            config=health_config
        )
        manager.register_monitor(unhealthy_monitor)
        await unhealthy_monitor.check_health()

        assert manager.is_healthy("healthy_backend") is True
        assert manager.is_healthy("unhealthy_backend") is False
        assert manager.is_healthy() is False  # Overall is unhealthy

    @pytest.mark.asyncio
    async def test_get_report_all(self, health_config, healthy_check_fn):
        """Test getting report for all backends."""
        manager = StorageHealthManager()

        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=healthy_check_fn,
            config=health_config
        )
        manager.register_monitor(monitor)
        await monitor.check_health()

        report = manager.get_report()

        assert "overall_healthy" in report
        assert "backends" in report
        assert "test_backend" in report["backends"]

    @pytest.mark.asyncio
    async def test_start_stop_all(self, health_config, healthy_check_fn):
        """Test starting and stopping all monitors."""
        manager = StorageHealthManager()

        for name in ["session", "vector"]:
            monitor = StorageHealthMonitor(
                backend_name=name,
                backend_type=name,
                check_fn=healthy_check_fn,
                config=health_config
            )
            manager.register_monitor(monitor)

        await manager.start_all()
        assert manager._started is True

        await asyncio.sleep(0.15)

        await manager.stop_all()
        assert manager._started is False


# =============================================================================
# UNIT TESTS - HEALTH CHECK FACTORY FUNCTIONS
# =============================================================================

class TestHealthCheckFactories:
    """Tests for health check factory functions."""

    @pytest.mark.asyncio
    async def test_create_postgres_health_check(self):
        """Test creating PostgreSQL health check function."""
        # Create mock pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone = AsyncMock(return_value=(1,))
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_pool.connection = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))
        mock_pool.get_stats = MagicMock(return_value={"pool_size": 5})

        check_fn = await create_postgres_health_check(mock_pool)
        result = await check_fn()

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_create_sqlite_health_check(self):
        """Test creating SQLite health check function."""
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        check_fn = await create_sqlite_health_check(mock_conn)
        result = await check_fn()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_create_chromadb_health_check(self):
        """Test creating ChromaDB health check function."""
        mock_client = MagicMock()
        mock_client.list_collections = MagicMock(return_value=[])

        check_fn = await create_chromadb_health_check(mock_client)
        result = await check_fn()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["collection_count"] == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHealthMonitoringIntegration:
    """Integration tests for health monitoring workflow."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, health_config, failing_check_fn):
        """Test that failing health checks open the circuit breaker."""
        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=failing_check_fn,
            config=health_config
        )

        # Run health checks to trigger circuit breaker
        for _ in range(5):
            await monitor.check_health()

        # Circuit should be open
        assert monitor.circuit_state == CircuitState.OPEN

        report = monitor.get_report()
        assert report.status == HealthStatus.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_recovery_workflow(self, health_config):
        """Test recovery from failure to healthy."""
        call_count = 0

        async def flaky_check():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Temporary failure")
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=10.0
            )

        monitor = StorageHealthMonitor(
            backend_name="test_backend",
            backend_type="session",
            check_fn=flaky_check,
            config=health_config
        )

        # Fail initially
        for _ in range(3):
            await monitor.check_health()

        assert monitor.circuit_state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.6)

        # Next check should transition to half-open and succeed
        await monitor.check_health()

        # Should be back to healthy
        assert monitor.is_healthy is True
