# src/llmcore/routing/state.py
"""Where per-target health lives.

The default store is in-process and needs no infrastructure, because llmcore
is a library: requiring Redis to fail over a 429 would be a worse bug than the
429. Spec §7 records this as the one thing deliberately *not* borrowed from
LiteLLM's router.

Three decisions worth naming:

* **Keyed by ``Target.key``, not by target.** The key excludes request
  parameters, so ``?effort=low`` and ``?effort=max`` share one record. They
  share one endpoint, one rate limit and one wallet, so they must share one
  cooldown — otherwise a 429 on one effort level leaves routing cheerfully
  hammering the same endpoint at another.
* **Async methods that never await.** The shape matches
  :class:`~llmcore.routing.protocols.RoutingStateStore` so a networked store
  is a drop-in, not a rewrite of every caller.
* **Locked.** Several coroutines record outcomes against the same target
  concurrently, and in-flight counters in particular are read-modify-write.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator

from .models import FailureKind, Outcome, TargetHealth

if TYPE_CHECKING:
    from .models import Balance

logger = logging.getLogger(__name__)

__all__ = ["InMemoryRoutingState"]


class InMemoryRoutingState:
    """Per-process health, cooldowns, latency and in-flight counters.

    Implements :class:`~llmcore.routing.protocols.RoutingStateStore`.

    Args:
        cooldowns: Per-:class:`~llmcore.routing.models.FailureKind` cooldown
            overrides, normally
            :attr:`~llmcore.routing.settings.RoutingSettings.cooldowns`. When
            omitted, :data:`~llmcore.routing.models.DEFAULT_COOLDOWNS` apply.
    """

    def __init__(self, cooldowns: dict[FailureKind, float | None] | None = None) -> None:
        self._health: dict[str, TargetHealth] = {}
        self._cooldowns = dict(cooldowns or {})
        self._lock = asyncio.Lock()

    # -- reads ------------------------------------------------------------

    async def get_health(self, target_key: str) -> TargetHealth:
        """Return health for ``target_key``, creating a fresh record if new."""
        async with self._lock:
            return self._health.setdefault(target_key, TargetHealth(target_key=target_key))

    def health_sync(self, target_key: str) -> TargetHealth:
        """Non-awaiting read, for selection.

        Selection scores every candidate in a pool; taking the lock per
        candidate would serialise routing behind itself for no benefit, since
        ``dict`` lookup and the attribute reads that follow are atomic enough
        for a scoring pass. A slightly stale cooldown costs one wasted attempt
        that the attempt loop already handles; a contended lock costs every
        request.
        """
        return self._health.setdefault(target_key, TargetHealth(target_key=target_key))

    async def snapshot(self) -> dict[str, TargetHealth]:
        """Return every known health record."""
        async with self._lock:
            return dict(self._health)

    def snapshot_sync(self) -> dict[str, TargetHealth]:
        """Non-awaiting :meth:`snapshot`, for synchronous reporting calls."""
        return dict(self._health)

    # -- writes -----------------------------------------------------------

    async def record(
        self, target_key: str, outcome: Outcome, *, now: datetime | None = None
    ) -> None:
        """Fold ``outcome`` into the stored health for ``target_key``.

        Args:
            target_key: The endpoint identity, i.e. :attr:`Target.key`.
            outcome: What happened.
            now: Injectable clock, so a cooldown can be asserted against a
                fixed moment. Selection takes the same argument, and the two
                have to agree or a test is measuring the wall clock.
        """
        async with self._lock:
            health = self._health.setdefault(target_key, TargetHealth(target_key=target_key))
            health.record(outcome, cooldowns=self._cooldowns, now=now)

    async def record_balance(
        self, target_key: str, balance: Balance | None, *, now: datetime | None = None
    ) -> None:
        """Attach a probed balance to a target.

        A *known* zero balance also starts an insufficient-credit cooldown:
        the probe has told us the next call will fail, so spending it to find
        out would be wasteful.
        """
        if balance is None:
            return
        async with self._lock:
            health = self._health.setdefault(target_key, TargetHealth(target_key=target_key))
            health.balance = balance
            if balance.is_known and balance.amount is not None and balance.amount <= 0:
                health.record(
                    Outcome(
                        target_key=target_key,
                        ok=False,
                        failure=FailureKind.INSUFFICIENT_CREDIT,
                        error="balance probe reported an empty account",
                    ),
                    cooldowns=self._cooldowns,
                    now=now,
                )

    async def clear(self, target_key: str | None = None) -> None:
        """Forget health for one target, or all of them.

        Needed for the honest case where a user has topped up an account or
        fixed a key and does not want to restart the process to escape an
        ``unusable`` mark.
        """
        async with self._lock:
            if target_key is None:
                self._health.clear()
            else:
                self._health.pop(target_key, None)

    # -- in-flight --------------------------------------------------------

    @asynccontextmanager
    async def in_flight(self, target_key: str) -> AsyncIterator[TargetHealth]:
        """Count one in-flight request against ``target_key``.

        A context manager rather than a pair of methods because
        ``least_busy`` is only as good as its decrements: one early return
        that forgets to decrement leaves a target looking permanently busy
        and quietly removes it from the rotation.
        """
        health = await self.get_health(target_key)
        async with self._lock:
            health.in_flight += 1
        try:
            yield health
        finally:
            async with self._lock:
                health.in_flight = max(0, health.in_flight - 1)
