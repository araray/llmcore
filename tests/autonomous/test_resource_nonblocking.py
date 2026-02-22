# tests/autonomous/test_resource_nonblocking.py
"""
Tests verifying ResourceMonitor uses non-blocking CPU sampling.

The original code called ``psutil.cpu_percent(interval=0.5)`` directly
in an async method, which blocked the event loop for 0.5s and prevented
timely cancellation on Ctrl+C.  The fix wraps the call in
``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.autonomous.resource import ResourceConstraints, ResourceMonitor


class TestNonBlockingCpuSampling:
    """Verify that _get_usage delegates CPU sampling to a thread."""

    @pytest.mark.asyncio
    async def test_cpu_percent_runs_in_thread(self):
        """psutil.cpu_percent is called via asyncio.to_thread, not directly."""
        monitor = ResourceMonitor(constraints=ResourceConstraints())

        mock_psutil = MagicMock()
        mock_psutil.cpu_percent = MagicMock(return_value=42.0)
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=55.0, used=4 * 1024**3, available=4 * 1024**3
        )
        mock_psutil.disk_usage.return_value = MagicMock(free=50 * 1024**3)
        mock_psutil.sensors_battery.return_value = None

        with (
            patch("llmcore.autonomous.resource.psutil", mock_psutil, create=True),
            patch(
                "llmcore.autonomous.resource.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=42.0,
            ) as mock_to_thread,
        ):
            # Patch the import inside _get_usage
            with patch.dict(
                "sys.modules",
                {"psutil": mock_psutil},
            ):
                usage = await monitor._get_usage()

            # Verify asyncio.to_thread was called with psutil.cpu_percent
            mock_to_thread.assert_awaited_once_with(mock_psutil.cpu_percent, 0.5)
            assert usage.cpu_percent == 42.0

    @pytest.mark.asyncio
    async def test_monitor_loop_cancellation(self):
        """Monitor loop can be cancelled promptly (no 0.5s block)."""
        monitor = ResourceMonitor(
            constraints=ResourceConstraints(),
        )
        await monitor.start()

        # Give it a tick to start
        await asyncio.sleep(0.05)

        # Cancel — should not hang
        await asyncio.wait_for(monitor.stop(), timeout=2.0)
        assert not monitor._running
