# src/llmcore/runtimes/testing.py
"""An in-process fake runtime (spec phase R1).

Mirrors ``llmcore.media.testing.FakeMediaProvider``: the subsystem's own tests
exercise the manager, attachment and reaping without network, credentials or
spend. Shipped rather than kept in ``tests/`` so downstream consumers can test
their own runtime-dependent code the same way.

It records every call, which is what makes the safety invariants testable —
"construction must not provision" is only a real guarantee if something can
assert that ``up`` was never reached.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from .models import ModelSpec, Plan, Quantization, RuntimeHandle, RuntimePhase, RuntimeStatus

__all__ = ["FakeRuntime"]


class FakeRuntime:
    """A :class:`~llmcore.runtimes.protocols.ComputeRuntime` that spends nothing.

    Args:
        name: Backend name.
        sku: SKU to report in plans.
        fits: Whether plans report the model as fitting.
        fail_on_up: Raise from :meth:`up` to exercise fail-closed paths.
        base_url: Endpoint to report for provisioned runtimes.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        sku: str = "L4",
        fits: bool = True,
        fail_on_up: bool = False,
        base_url: str = "http://127.0.0.1:8111/v1",
    ) -> None:
        self.name = name
        self._sku = sku
        self._fits = fits
        self._fail_on_up = fail_on_up
        self._base_url = base_url

        #: Every call made, as ``(method, kwargs)``.
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: Runtimes currently "running".
        self.running: dict[str, RuntimeHandle] = {}
        #: Names released via :meth:`down`.
        self.released: list[str] = []

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    def called(self, method: str) -> int:
        """How many times *method* was called."""
        return sum(1 for m, _ in self.calls if m == method)

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Return a plan. Free and read-only, as the protocol requires."""
        self._record("estimate", repo_id=spec.repo_id, ctx=spec.context_length)
        required = 16.0 if self._fits else 999.0
        return Plan(
            spec=spec,
            sku=self._sku,
            recipe="vllm",
            quantization=spec.quantization or Quantization.NONE,
            vram_required_gb=required,
            vram_available_gb=22.5,
            context_length=spec.context_length,
            fits=self._fits,
            notes=("fake sizing",),
            estimated_cost_per_hour=0.0,
        )

    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle:
        """Pretend to provision. Raises when ``fail_on_up`` is set."""
        self._record("up", name=name, repo_id=plan.spec.repo_id, sku=plan.sku)
        if self._fail_on_up:
            raise RuntimeError("fake bootstrap failure")
        handle = RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id=f"fake-{name}",
            base_url=self._base_url,
            served_model=plan.spec.repo_id,
            api_style="openai",
            recipe=plan.recipe,
            sku=plan.sku,
            phase=RuntimePhase.READY,
            started_at=datetime.now(timezone.utc),
        )
        self.running[name] = handle
        return handle

    async def status(self, name: str | None = None) -> list[RuntimeStatus]:
        """Report on fake runtimes."""
        self._record("status", name=name)
        handles = self.running.values() if name is None else (
            [self.running[name]] if name in self.running else []
        )
        return [RuntimeStatus.from_handle(h) for h in handles]

    async def logs(
        self, name: str, *, component: str = "server", tail: int = 100
    ) -> AsyncIterator[str]:
        """Yield two fake log lines."""
        self._record("logs", name=name, component=component, tail=tail)
        for line in (f"[{component}] fake log 1", f"[{component}] fake log 2"):
            yield line

    async def down(self, name: str, *, release: bool = True) -> None:
        """Stop a fake runtime. Idempotent, as the protocol requires."""
        self._record("down", name=name, release=release)
        handle = self.running.pop(name, None)
        if handle is not None:
            handle.phase = RuntimePhase.STOPPED
        if release:
            self.released.append(name)

    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle:
        """Adopt a fake external runtime."""
        self._record("adopt", external_id=external_id, name=name)
        handle = RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id=external_id,
            base_url=self._base_url,
            served_model="adopted/model",
            phase=RuntimePhase.READY,
        )
        self.running[name] = handle
        return handle
