# src/llmcore/runtimes/protocols.py
"""The runtime backend contract (spec phase R1).

One protocol, several possible backends: Colab first, then SSH, RunPod, Modal.
Backends are discovered structurally rather than by inheritance, matching how
the media subsystem discovers capability adapters.

Note which operations spend money. ``estimate`` is read-only and free, and the
split exists so sizing can be inspected and approved before anything is
provisioned. ``up`` and ``adopt`` are the only methods that can start or take
ownership of billable compute.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import ModelSpec, Plan, RuntimeHandle, RuntimeStatus

__all__ = ["ComputeRuntime"]


@runtime_checkable
class ComputeRuntime(Protocol):
    """A backend that can provision GPU compute and serve a model on it."""

    #: Backend name, used in config and in handles (e.g. ``"colab"``).
    name: str

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Size *spec* without provisioning anything.

        Must be free and read-only. Callers rely on being able to size a model
        before deciding whether to spend.
        """
        ...

    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle:
        """Provision compute and serve ``plan.spec``. **This spends money.**

        Must **fail closed**: if bootstrap fails at any point, release the
        compute before raising. Never leave a VM assigned after an error.
        """
        ...

    async def status(self, name: str | None = None) -> list[RuntimeStatus]:
        """Report on one runtime, or all of them when *name* is ``None``."""
        ...

    def logs(self, name: str, *, component: str = "server", tail: int = 100) -> AsyncIterator[str]:
        """Stream log lines for *name*.

        A plain method returning an async iterator, not a coroutine, so callers
        write ``async for line in runtime.logs(...)``.
        """
        ...

    async def down(self, name: str, *, release: bool = True) -> None:
        """Stop *name*. With *release*, give the compute back and stop billing.

        Must be idempotent: tearing down something already gone is how recovery
        from a half-failed start works, so it cannot raise.
        """
        ...

    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle:
        """Take ownership of compute llmcore did not start.

        The recovery path for a leaked runtime: something is billing and llmcore
        has no record of it, so adopting it is what makes it killable.
        """
        ...
