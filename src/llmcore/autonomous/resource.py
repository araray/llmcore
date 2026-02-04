# src/llmcore/autonomous/resource.py
"""
Resource Monitoring System for Autonomous Operation.

Monitors system resources (CPU, memory, temperature, API costs) and
enforces constraints for safe operation on resource-constrained devices.

Critical for:
    - Raspberry Pi thermal management
    - API cost control
    - Battery-powered operation
    - Long-running autonomous agents

Example:
    monitor = ResourceMonitor(
        constraints=ResourceConstraints(
            max_cpu_percent=60,
            max_memory_percent=60,
            max_temperature_c=65,
            max_daily_cost_usd=3.0
        )
    )

    await monitor.start()

    # Check before operations
    if monitor.can_proceed():
        # Do work
        pass
    else:
        # Wait for resources
        await monitor.wait_for_resources(timeout=300)

    # Record API usage
    monitor.record_usage(tokens=1500, cost_usd=0.002)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ResourceConstraints:
    """
    Configurable resource limits.

    Attributes:
        max_cpu_percent: Maximum CPU usage (0-100)
        max_memory_percent: Maximum memory usage (0-100)
        max_temperature_c: Maximum CPU temperature (Celsius)
        max_hourly_cost_usd: Maximum hourly API spend
        max_daily_cost_usd: Maximum daily API spend
        max_hourly_tokens: Maximum hourly token usage
        max_daily_tokens: Maximum daily token usage
        min_request_interval_ms: Minimum time between API requests
        min_battery_percent: Minimum battery before pausing (if on battery)
        min_disk_free_gb: Minimum free disk space

    Example:
        # Conservative constraints for Raspberry Pi
        constraints = ResourceConstraints(
            max_cpu_percent=60,
            max_memory_percent=60,
            max_temperature_c=65,
            max_daily_cost_usd=3.0
        )
    """

    # Hardware constraints
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_temperature_c: float | None = 75.0

    # API cost constraints
    max_hourly_cost_usd: float = 1.0
    max_daily_cost_usd: float = 10.0
    max_hourly_tokens: int = 100_000
    max_daily_tokens: int = 1_000_000

    # Rate limiting
    min_request_interval_ms: int = 100

    # Power constraints
    min_battery_percent: float | None = None

    # Disk constraints
    min_disk_free_gb: float = 1.0


@dataclass
class ResourceUsage:
    """
    Current resource usage snapshot.

    Captured periodically by ResourceMonitor.
    """

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Hardware
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    temperature_c: float | None = None
    disk_free_gb: float = 0.0

    # API usage (current period)
    tokens_this_hour: int = 0
    tokens_today: int = 0
    cost_this_hour_usd: float = 0.0
    cost_today_usd: float = 0.0
    requests_this_hour: int = 0

    # Battery
    battery_percent: float | None = None
    battery_plugged: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "temperature_c": self.temperature_c,
            "disk_free_gb": self.disk_free_gb,
            "tokens_this_hour": self.tokens_this_hour,
            "tokens_today": self.tokens_today,
            "cost_this_hour_usd": self.cost_this_hour_usd,
            "cost_today_usd": self.cost_today_usd,
            "requests_this_hour": self.requests_this_hour,
            "battery_percent": self.battery_percent,
            "battery_plugged": self.battery_plugged,
        }


class ConstraintViolation(Enum):
    """Types of constraint violations."""

    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    TEMPERATURE_HIGH = "temperature_high"
    COST_HOURLY_EXCEEDED = "cost_hourly_exceeded"
    COST_DAILY_EXCEEDED = "cost_daily_exceeded"
    TOKENS_HOURLY_EXCEEDED = "tokens_hourly_exceeded"
    TOKENS_DAILY_EXCEEDED = "tokens_daily_exceeded"
    BATTERY_LOW = "battery_low"
    DISK_LOW = "disk_low"


@dataclass
class ResourceStatus:
    """
    Overall resource status with violation detection.

    Provides both fine-grained violation list and convenience
    properties for decision-making (can_proceed, should_throttle).
    """

    usage: ResourceUsage
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def is_constrained(self) -> bool:
        """Any constraints violated?"""
        return len(self.violations) > 0

    @property
    def can_proceed(self) -> bool:
        """
        Can we proceed with operations?

        Returns False only for "hard" violations that must stop operations:
        - Daily cost exceeded
        - Daily tokens exceeded
        - Battery critically low
        - Disk full
        """
        critical = {
            ConstraintViolation.COST_DAILY_EXCEEDED,
            ConstraintViolation.TOKENS_DAILY_EXCEEDED,
            ConstraintViolation.BATTERY_LOW,
            ConstraintViolation.DISK_LOW,
        }
        return not any(v in critical for v in self.violations)

    @property
    def should_throttle(self) -> bool:
        """
        Should we slow down operations?

        Returns True for "soft" violations where we should reduce load:
        - CPU high
        - Memory high
        - Temperature high
        - Hourly limits approached
        """
        return self.is_constrained and self.can_proceed


# =============================================================================
# ResourceMonitor
# =============================================================================


class ResourceMonitor:
    """
    Monitors system resources and enforces constraints.

    Runs a background loop that periodically checks resources and
    tracks API usage. Provides methods to check if operations should
    proceed and to wait for resources to become available.

    Example:
        monitor = ResourceMonitor(
            constraints=ResourceConstraints(
                max_cpu_percent=60,
                max_temperature_c=65
            ),
            check_interval=timedelta(seconds=30)
        )

        await monitor.start()

        # Main loop
        while running:
            if not monitor.can_proceed():
                logger.warning("Waiting for resources...")
                await monitor.wait_for_resources(timeout=300)

            # Do work...
            response = await llm.complete(...)
            monitor.record_usage(
                tokens=response.total_tokens,
                cost_usd=response.cost
            )

        await monitor.stop()
    """

    def __init__(
        self,
        constraints: ResourceConstraints | None = None,
        check_interval: timedelta = timedelta(seconds=30),
    ):
        """
        Initialize ResourceMonitor.

        Args:
            constraints: Resource limits to enforce
            check_interval: How often to check resources
        """
        self.constraints = constraints or ResourceConstraints()
        self.check_interval = check_interval

        self._status: ResourceStatus | None = None
        self._running = False
        self._loop_task: asyncio.Task | None = None

        # Usage tracking (reset periodically)
        self._hourly_tokens = 0
        self._hourly_cost = 0.0
        self._hourly_requests = 0
        self._daily_tokens = 0
        self._daily_cost = 0.0

        self._last_hour = datetime.utcnow().hour
        self._last_day = datetime.utcnow().date()

        # Callbacks
        self._on_violation: list[Callable] = []

    async def start(self) -> None:
        """Start resource monitoring."""
        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Resource monitor started (interval: {self.check_interval.total_seconds()}s)")

    async def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitor stopped")

    def on_violation(self, callback: Callable) -> None:
        """
        Register callback for constraint violations.

        Args:
            callback: Function(violations: List[ConstraintViolation])
        """
        self._on_violation.append(callback)

    async def _monitor_loop(self) -> None:
        """Periodic resource check loop."""
        while self._running:
            try:
                self._status = await self._check_resources()

                if self._status.is_constrained:
                    logger.warning(
                        f"Resource constraints violated: "
                        f"{[v.value for v in self._status.violations]}"
                    )

                    # Notify callbacks
                    for callback in self._on_violation:
                        try:
                            callback(self._status.violations)
                        except Exception:
                            pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")

            await asyncio.sleep(self.check_interval.total_seconds())

    async def _check_resources(self) -> ResourceStatus:
        """Check current resource status."""
        usage = await self._get_usage()
        violations = self._check_violations(usage)
        return ResourceStatus(usage=usage, violations=violations)

    async def _get_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed, resource monitoring limited")
            return ResourceUsage(
                tokens_this_hour=self._hourly_tokens,
                tokens_today=self._daily_tokens,
                cost_this_hour_usd=self._hourly_cost,
                cost_today_usd=self._daily_cost,
                requests_this_hour=self._hourly_requests,
            )

        # Reset counters if needed
        now = datetime.utcnow()
        if now.hour != self._last_hour:
            self._hourly_tokens = 0
            self._hourly_cost = 0.0
            self._hourly_requests = 0
            self._last_hour = now.hour

        if now.date() != self._last_day:
            self._daily_tokens = 0
            self._daily_cost = 0.0
            self._last_day = now.date()

        # CPU (brief sample)
        cpu = psutil.cpu_percent(interval=0.5)

        # Memory
        mem = psutil.virtual_memory()

        # Temperature
        temp = await self._get_temperature()

        # Disk
        try:
            disk = psutil.disk_usage("/")
            disk_free_gb = disk.free / (1024**3)
        except Exception:
            disk_free_gb = 999.0  # Assume OK if can't check

        # Battery
        battery_percent = None
        battery_plugged = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
                battery_plugged = battery.power_plugged
        except Exception:
            pass

        return ResourceUsage(
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_used_mb=mem.used / (1024**2),
            memory_available_mb=mem.available / (1024**2),
            temperature_c=temp,
            disk_free_gb=disk_free_gb,
            tokens_this_hour=self._hourly_tokens,
            tokens_today=self._daily_tokens,
            cost_this_hour_usd=self._hourly_cost,
            cost_today_usd=self._daily_cost,
            requests_this_hour=self._hourly_requests,
            battery_percent=battery_percent,
            battery_plugged=battery_plugged,
        )

    async def _get_temperature(self) -> float | None:
        """Get CPU temperature (platform-specific)."""
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if not temps:
                # Try Raspberry Pi thermal zone
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp") as f:
                        return float(f.read().strip()) / 1000.0
                except Exception:
                    pass
                return None

            # Look for common sensor names
            for name in [
                "coretemp",
                "cpu_thermal",
                "cpu-thermal",
                "acpitz",
                "k10temp",
            ]:
                if temps.get(name):
                    return temps[name][0].current

            # Fallback: first available
            for entries in temps.values():
                if entries:
                    return entries[0].current

        except Exception:
            pass

        return None

    def _check_violations(self, usage: ResourceUsage) -> list[ConstraintViolation]:
        """Check for constraint violations."""
        violations: list[ConstraintViolation] = []
        c = self.constraints

        # CPU
        if usage.cpu_percent > c.max_cpu_percent:
            violations.append(ConstraintViolation.CPU_HIGH)

        # Memory
        if usage.memory_percent > c.max_memory_percent:
            violations.append(ConstraintViolation.MEMORY_HIGH)

        # Temperature
        if c.max_temperature_c and usage.temperature_c:
            if usage.temperature_c > c.max_temperature_c:
                violations.append(ConstraintViolation.TEMPERATURE_HIGH)

        # Hourly cost
        if usage.cost_this_hour_usd >= c.max_hourly_cost_usd:
            violations.append(ConstraintViolation.COST_HOURLY_EXCEEDED)

        # Daily cost
        if usage.cost_today_usd >= c.max_daily_cost_usd:
            violations.append(ConstraintViolation.COST_DAILY_EXCEEDED)

        # Hourly tokens
        if usage.tokens_this_hour >= c.max_hourly_tokens:
            violations.append(ConstraintViolation.TOKENS_HOURLY_EXCEEDED)

        # Daily tokens
        if usage.tokens_today >= c.max_daily_tokens:
            violations.append(ConstraintViolation.TOKENS_DAILY_EXCEEDED)

        # Battery (only if on battery power)
        if c.min_battery_percent and usage.battery_percent is not None:
            if usage.battery_percent < c.min_battery_percent:
                if not usage.battery_plugged:
                    violations.append(ConstraintViolation.BATTERY_LOW)

        # Disk
        if usage.disk_free_gb < c.min_disk_free_gb:
            violations.append(ConstraintViolation.DISK_LOW)

        return violations

    def record_usage(
        self,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """
        Record API usage for tracking.

        Call this after each API request.

        Args:
            tokens: Tokens used in request
            cost_usd: Cost of request
        """
        self._hourly_tokens += tokens
        self._hourly_cost += cost_usd
        self._hourly_requests += 1
        self._daily_tokens += tokens
        self._daily_cost += cost_usd

    def can_proceed(self) -> bool:
        """
        Check if operations can proceed.

        Returns:
            True if no critical constraints violated
        """
        if self._status is None:
            return True
        return self._status.can_proceed

    def should_throttle(self) -> bool:
        """
        Check if operations should slow down.

        Returns:
            True if soft constraints violated
        """
        if self._status is None:
            return False
        return self._status.should_throttle

    async def wait_for_resources(
        self,
        timeout: float = 300,
        check_interval: float = 10,
    ) -> bool:
        """
        Wait until resources are available.

        Args:
            timeout: Maximum time to wait (seconds)
            check_interval: How often to check (seconds)

        Returns:
            True if resources became available, False if timeout
        """
        start = datetime.utcnow()
        while (datetime.utcnow() - start).total_seconds() < timeout:
            # Force resource check
            self._status = await self._check_resources()

            if self.can_proceed():
                return True

            logger.debug(
                f"Waiting for resources... "
                f"(violations: "
                f"{[v.value for v in self._status.violations]})"
            )
            await asyncio.sleep(check_interval)

        return False

    def get_status(self) -> dict[str, Any]:
        """Get current resource status as dictionary."""
        if self._status is None:
            return {"status": "not_started"}

        return {
            "timestamp": self._status.usage.timestamp.isoformat(),
            "can_proceed": self._status.can_proceed,
            "should_throttle": self._status.should_throttle,
            "violations": [v.value for v in self._status.violations],
            "hardware": {
                "cpu_percent": self._status.usage.cpu_percent,
                "memory_percent": self._status.usage.memory_percent,
                "temperature_c": self._status.usage.temperature_c,
                "disk_free_gb": self._status.usage.disk_free_gb,
            },
            "api_usage": {
                "tokens_this_hour": self._status.usage.tokens_this_hour,
                "tokens_today": self._status.usage.tokens_today,
                "cost_this_hour_usd": (self._status.usage.cost_this_hour_usd),
                "cost_today_usd": self._status.usage.cost_today_usd,
                "requests_this_hour": (self._status.usage.requests_this_hour),
            },
            "limits": {
                "max_cpu_percent": self.constraints.max_cpu_percent,
                "max_memory_percent": self.constraints.max_memory_percent,
                "max_temperature_c": self.constraints.max_temperature_c,
                "max_hourly_cost_usd": (self.constraints.max_hourly_cost_usd),
                "max_daily_cost_usd": self.constraints.max_daily_cost_usd,
            },
        }

    @property
    def current_usage(self) -> ResourceUsage | None:
        """Get current resource usage snapshot."""
        return self._status.usage if self._status else None
