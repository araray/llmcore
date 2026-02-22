# src/llmcore/storage/health.py
"""
Health Check and Circuit Breaker Infrastructure for LLMCore Storage.

This module provides production-grade reliability features for the storage layer:

1. Health Checks: Periodic connectivity and latency monitoring
2. Circuit Breaker: Fail-fast pattern to prevent cascade failures
3. Connection Pool Monitoring: Track pool utilization and health
4. Health Status API: Unified health reporting for all backends

Design Philosophy:
- Observable by default (structured logging, metrics hooks)
- Graceful degradation (circuit breaker prevents resource exhaustion)
- Non-intrusive (health checks don't block normal operations)
- Configurable thresholds (tune for your deployment)

Usage:
    # Create health monitor for a storage backend
    health = StorageHealthMonitor(
        backend_name="postgres_sessions",
        check_fn=check_postgres_health,
        config=HealthConfig(
            check_interval_seconds=30,
            failure_threshold=3,
            recovery_timeout_seconds=60
        )
    )

    # Start background monitoring
    await health.start()

    # Check if backend is healthy before operations
    if health.is_healthy:
        await storage.save_session(session)
    else:
        raise StorageUnavailableError("Session storage is unhealthy")
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# HEALTH STATUS TYPES
# =============================================================================


class HealthStatus(str, Enum):
    """Health status of a storage backend."""

    HEALTHY = "healthy"  # All checks passing
    DEGRADED = "degraded"  # Some checks failing, but operational
    UNHEALTHY = "unhealthy"  # Critical checks failing
    UNKNOWN = "unknown"  # No health data yet
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker is open


class CircuitState(str, Enum):
    """Circuit breaker state machine states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if backend recovered


@dataclass
class HealthCheckResult:
    """Result of a single health check execution."""

    status: HealthStatus
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/API responses."""
        return {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "details": self.details,
        }


@dataclass
class StorageHealthReport:
    """Comprehensive health report for a storage backend."""

    backend_name: str
    backend_type: str  # "session" or "vector"
    status: HealthStatus
    circuit_state: CircuitState
    last_check: HealthCheckResult | None
    consecutive_failures: int
    total_checks: int
    total_failures: int
    uptime_percentage: float
    average_latency_ms: float
    last_successful_check: datetime | None
    last_failed_check: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/API responses."""
        return {
            "backend_name": self.backend_name,
            "backend_type": self.backend_type,
            "status": self.status.value,
            "circuit_state": self.circuit_state.value,
            "last_check": self.last_check.to_dict() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "uptime_percentage": round(self.uptime_percentage, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "last_successful_check": self.last_successful_check.isoformat()
            if self.last_successful_check
            else None,
            "last_failed_check": self.last_failed_check.isoformat()
            if self.last_failed_check
            else None,
        }


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""

    # Health check intervals
    check_interval_seconds: float = 30.0
    check_timeout_seconds: float = 5.0

    # Circuit breaker thresholds
    failure_threshold: int = 3  # Consecutive failures to open circuit
    recovery_timeout_seconds: float = 60.0  # Time before trying half-open
    half_open_max_requests: int = 1  # Requests allowed in half-open state

    # Latency thresholds
    latency_warning_ms: float = 500.0  # Log warning if latency exceeds this
    latency_critical_ms: float = 2000.0  # Consider degraded if latency exceeds this

    # History settings
    history_size: int = 100  # Number of check results to retain

    # Feature flags
    enabled: bool = True
    log_health_checks: bool = True
    emit_metrics: bool = True


DEFAULT_HEALTH_CONFIG = HealthConfig()


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker implementation for storage operations.

    State Machine:
        CLOSED -> OPEN: After failure_threshold consecutive failures
        OPEN -> HALF_OPEN: After recovery_timeout_seconds
        HALF_OPEN -> CLOSED: On successful request
        HALF_OPEN -> OPEN: On failed request

    Usage:
        breaker = CircuitBreaker(config)

        if breaker.allow_request():
            try:
                result = await do_operation()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise CircuitOpenError()
    """

    def __init__(self, config: HealthConfig):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_requests = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    async def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout_seconds:
                        logger.info("Circuit breaker transitioning to HALF_OPEN state")
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_requests = 0
                        return True
                return False

            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_requests < self.config.half_open_max_requests:
                    self._half_open_requests += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self._failure_count = 0
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning to CLOSED state (recovered)")
                self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "Circuit breaker transitioning to OPEN state (half-open test failed)"
                )
                self._state = CircuitState.OPEN
                self._half_open_requests = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit breaker transitioning to OPEN state "
                        f"(failure threshold {self.config.failure_threshold} reached)"
                    )
                    self._state = CircuitState.OPEN

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_requests = 0
            logger.info("Circuit breaker manually reset to CLOSED state")


# =============================================================================
# HEALTH MONITOR
# =============================================================================

# Type alias for health check function
HealthCheckFn = Callable[[], Coroutine[Any, Any, HealthCheckResult]]


class StorageHealthMonitor:
    """
    Background health monitor for a storage backend.

    Periodically executes health checks and manages circuit breaker state.
    Provides a unified API for querying backend health status.
    """

    def __init__(
        self,
        backend_name: str,
        backend_type: str,
        check_fn: HealthCheckFn,
        config: HealthConfig | None = None,
    ):
        """
        Initialize health monitor.

        Args:
            backend_name: Identifier for this backend (e.g., "postgres_sessions")
            backend_type: Type of backend ("session" or "vector")
            check_fn: Async function that performs health check
            config: Health monitoring configuration
        """
        self.backend_name = backend_name
        self.backend_type = backend_type
        self._check_fn = check_fn
        self.config = config or DEFAULT_HEALTH_CONFIG

        self._circuit_breaker = CircuitBreaker(self.config)
        self._check_history: list[HealthCheckResult] = []
        self._task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._total_checks = 0
        self._total_failures = 0
        self._total_latency_ms = 0.0
        self._last_successful_check: datetime | None = None
        self._last_failed_check: datetime | None = None

    @property
    def is_healthy(self) -> bool:
        """
        Check if the backend is considered healthy.

        The backend is healthy if:
        - No checks have run yet (optimistic default), OR
        - The latest check shows HEALTHY/DEGRADED status AND
          the circuit breaker is not OPEN (CLOSED or HALF_OPEN is acceptable)

        HALF_OPEN is acceptable because it means we're testing recovery,
        and the latest check was successful.
        """
        if not self._check_history:
            return True  # Assume healthy until proven otherwise

        latest = self._check_history[-1]
        return (
            latest.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            and not self._circuit_breaker.is_open
        )

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_breaker.state

    @property
    def consecutive_failures(self) -> int:
        """Get count of consecutive failures."""
        return self._circuit_breaker._failure_count

    async def check_health(self) -> HealthCheckResult:
        """
        Execute a single health check.

        Returns:
            HealthCheckResult with status and latency.
        """
        start_time = time.perf_counter()

        # Check if circuit should transition from OPEN to HALF_OPEN based on time
        # This allows health checks to test recovery after the timeout period
        if self._circuit_breaker.is_open:
            # Call allow_request to potentially trigger OPEN -> HALF_OPEN transition
            await self._circuit_breaker.allow_request()

        try:
            # Execute the health check with timeout
            result = await asyncio.wait_for(
                self._check_fn(), timeout=self.config.check_timeout_seconds
            )

            # Record success
            await self._circuit_breaker.record_success()
            self._last_successful_check = result.timestamp

            # Check for latency degradation
            if result.latency_ms >= self.config.latency_critical_ms:
                result.status = HealthStatus.DEGRADED
                result.details["latency_warning"] = f"High latency: {result.latency_ms:.0f}ms"
            elif result.latency_ms >= self.config.latency_warning_ms:
                if self.config.log_health_checks:
                    logger.warning(
                        f"Health check latency warning for {self.backend_name}: "
                        f"{result.latency_ms:.0f}ms"
                    )

        except TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error_message=f"Health check timed out after {self.config.check_timeout_seconds}s",
            )
            await self._circuit_breaker.record_failure()
            self._last_failed_check = result.timestamp
            self._total_failures += 1

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, error_message=str(e)
            )
            await self._circuit_breaker.record_failure()
            self._last_failed_check = result.timestamp
            self._total_failures += 1

            if self.config.log_health_checks:
                logger.error(f"Health check failed for {self.backend_name}: {e}", exc_info=True)

        # Update statistics
        self._total_checks += 1
        self._total_latency_ms += result.latency_ms

        # Store in history (with size limit)
        self._check_history.append(result)
        if len(self._check_history) > self.config.history_size:
            self._check_history.pop(0)

        # Log if enabled
        if self.config.log_health_checks:
            level = logging.DEBUG if result.status == HealthStatus.HEALTHY else logging.WARNING
            logger.log(
                level,
                f"Health check {self.backend_name}: status={result.status.value}, "
                f"latency={result.latency_ms:.0f}ms",
            )

        return result

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.debug(f"Starting health monitoring for {self.backend_name}")

        while self._running:
            try:
                await self.check_health()
            except Exception as e:
                logger.error(f"Unexpected error in health monitoring loop: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

        logger.debug(f"Stopped health monitoring for {self.backend_name}")

    async def start(self) -> None:
        """Start background health monitoring."""
        if not self.config.enabled:
            logger.info(f"Health monitoring disabled for {self.backend_name}")
            return

        if self._running:
            logger.warning(f"Health monitoring already running for {self.backend_name}")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_report(self) -> StorageHealthReport:
        """
        Generate comprehensive health report.

        Returns:
            StorageHealthReport with all health metrics.
        """
        # Determine overall status
        if self._circuit_breaker.is_open:
            status = HealthStatus.CIRCUIT_OPEN
        elif not self._check_history:
            status = HealthStatus.UNKNOWN
        else:
            status = self._check_history[-1].status

        # Calculate uptime percentage
        successful_checks = self._total_checks - self._total_failures
        uptime_pct = (
            (successful_checks / self._total_checks * 100) if self._total_checks > 0 else 100.0
        )

        # Calculate average latency
        avg_latency = (
            (self._total_latency_ms / self._total_checks) if self._total_checks > 0 else 0.0
        )

        return StorageHealthReport(
            backend_name=self.backend_name,
            backend_type=self.backend_type,
            status=status,
            circuit_state=self._circuit_breaker.state,
            last_check=self._check_history[-1] if self._check_history else None,
            consecutive_failures=self.consecutive_failures,
            total_checks=self._total_checks,
            total_failures=self._total_failures,
            uptime_percentage=uptime_pct,
            average_latency_ms=avg_latency,
            last_successful_check=self._last_successful_check,
            last_failed_check=self._last_failed_check,
        )


# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================


async def create_postgres_health_check(pool: Any) -> HealthCheckFn:
    """
    Create a health check function for PostgreSQL.

    Args:
        pool: AsyncConnectionPool from psycopg_pool

    Returns:
        Async function that performs health check.
    """

    async def check() -> HealthCheckResult:
        start = time.perf_counter()
        try:
            async with pool.connection() as conn:
                result = await conn.execute("SELECT 1 as health_check")
                row = await result.fetchone()

                latency_ms = (time.perf_counter() - start) * 1000

                if row and row[0] == 1:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency_ms,
                        details={
                            "query": "SELECT 1",
                            "pool_size": pool.get_stats().get("pool_size", "unknown"),
                        },
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error_message="Health check query returned unexpected result",
                    )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, error_message=str(e)
            )

    return check


async def create_sqlite_health_check(conn: Any) -> HealthCheckFn:
    """
    Create a health check function for SQLite.

    Args:
        conn: aiosqlite.Connection

    Returns:
        Async function that performs health check.
    """

    async def check() -> HealthCheckResult:
        start = time.perf_counter()
        try:
            cursor = await conn.execute("SELECT 1 as health_check")
            row = await cursor.fetchone()

            latency_ms = (time.perf_counter() - start) * 1000

            if row and row[0] == 1:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    details={"query": "SELECT 1"},
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    error_message="Health check query returned unexpected result",
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, error_message=str(e)
            )

    return check


async def create_chromadb_health_check(client: Any) -> HealthCheckFn:
    """
    Create a health check function for ChromaDB.

    Args:
        client: ChromaDB client instance

    Returns:
        Async function that performs health check.
    """

    async def check() -> HealthCheckResult:
        start = time.perf_counter()
        try:
            # ChromaDB heartbeat/list_collections as health check
            collections = client.list_collections()

            latency_ms = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"collection_count": len(collections)},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, error_message=str(e)
            )

    return check


# =============================================================================
# COMPOSITE HEALTH MANAGER
# =============================================================================


class StorageHealthManager:
    """
    Manages health monitoring for all storage backends.

    Provides unified health status across session and vector storage,
    and aggregates health reports for observability.
    """

    def __init__(self):
        """Initialize the health manager."""
        self._monitors: dict[str, StorageHealthMonitor] = {}
        self._started = False

    def register_monitor(self, monitor: StorageHealthMonitor) -> None:
        """
        Register a health monitor.

        Args:
            monitor: StorageHealthMonitor to register
        """
        self._monitors[monitor.backend_name] = monitor
        logger.debug(f"Registered health monitor: {monitor.backend_name}")

    def unregister_monitor(self, backend_name: str) -> None:
        """
        Unregister a health monitor.

        Args:
            backend_name: Name of the backend to unregister
        """
        if backend_name in self._monitors:
            del self._monitors[backend_name]
            logger.debug(f"Unregistered health monitor: {backend_name}")

    async def start_all(self) -> None:
        """Start all registered health monitors."""
        if self._started:
            return

        for monitor in self._monitors.values():
            await monitor.start()

        self._started = True
        logger.debug(f"Started {len(self._monitors)} health monitors")

    async def stop_all(self) -> None:
        """Stop all registered health monitors."""
        for monitor in self._monitors.values():
            await monitor.stop()

        self._started = False
        logger.info("Stopped all health monitors")

    def is_healthy(self, backend_name: str | None = None) -> bool:
        """
        Check if backends are healthy.

        Args:
            backend_name: Specific backend to check, or None for all

        Returns:
            True if specified backend(s) are healthy
        """
        if backend_name:
            monitor = self._monitors.get(backend_name)
            return monitor.is_healthy if monitor else True

        return all(m.is_healthy for m in self._monitors.values())

    def get_report(self, backend_name: str | None = None) -> dict[str, Any]:
        """
        Get health report for backend(s).

        Args:
            backend_name: Specific backend, or None for all

        Returns:
            Health report dictionary
        """
        if backend_name:
            monitor = self._monitors.get(backend_name)
            if monitor:
                return monitor.get_report().to_dict()
            return {"error": f"Unknown backend: {backend_name}"}

        return {
            "overall_healthy": self.is_healthy(),
            "backends": {
                name: monitor.get_report().to_dict() for name, monitor in self._monitors.items()
            },
        }

    async def run_health_check(self, backend_name: str) -> HealthCheckResult | None:
        """
        Manually trigger a health check for a specific backend.

        Args:
            backend_name: Backend to check

        Returns:
            HealthCheckResult or None if backend not found
        """
        monitor = self._monitors.get(backend_name)
        if monitor:
            return await monitor.check_health()
        return None
