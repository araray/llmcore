# src/llmcore/runtimes/diagnostics.py
"""Preflight checks for the runtimes subsystem.

Ported in spirit from ``agent-lens colab doctor``, which llmcore had no answer
to. Without it, llmcore discovers each problem one at a time and only when it
is in the way -- a missing CLI at provision time, an absent key at sizing time,
an unwritable state directory at the moment it needs to record a VM it has just
started. That last one is the dangerous ordering: the state file is what makes
a runtime findable and killable, so failing to write it *after* compute is
assigned is how a VM gets lost.

Every check here is **read-only and free**. The point is to be runnable before
deciding to spend, and while the subsystem is switched off.

Backends opt in by defining ``async def diagnose(self) -> list[Check]``.
Discovery is structural, like the backend protocol itself, so a backend without
one is simply not asked rather than being an error.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Check", "Severity", "run_diagnostics"]


class Severity:
    """How much a failed check matters.

    Not an enum of states but of *consequences*, because that is what decides
    whether to act: ``BLOCKS`` means provisioning cannot work, ``DEGRADES``
    means it can but something will be missing, and ``MONEY`` means something
    may be costing money right now -- which outranks both.
    """

    OK = "ok"
    DEGRADES = "degrades"
    BLOCKS = "blocks"
    MONEY = "money"


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic result.

    Attributes:
        name: Short label, e.g. ``"colab: cli"``.
        severity: One of :class:`Severity`.
        detail: What was found.
        fix: What to do about it, when there is something to do.
    """

    name: str
    severity: str
    detail: str
    fix: str | None = None

    @property
    def ok(self) -> bool:
        return self.severity == Severity.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "detail": self.detail,
            "fix": self.fix,
        }


def _state_dir_check(manager: Any) -> Check:
    """Whether a runtime could be *recorded*, which is what makes it killable."""
    try:
        directory = Path(str(manager.state.directory)).expanduser()
    except Exception as exc:  # pragma: no cover - defensive
        return Check(
            "state: directory", Severity.BLOCKS, f"could not be resolved: {exc}"
        )

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".llmcore-doctor-probe"
        probe.write_text("probe")
        probe.unlink()
    except Exception as exc:
        return Check(
            "state: directory",
            # Blocks rather than degrades, and deliberately so: llmcore writes
            # state *before* provisioning precisely so a crash cannot lose a
            # VM. If that write cannot happen, provisioning must not start.
            Severity.BLOCKS,
            f"{directory} is not writable ({type(exc).__name__}: {exc})",
            "point runtimes.state_dir somewhere writable",
        )
    return Check("state: directory", Severity.OK, f"{directory} is writable")


def _enabled_check(manager: Any) -> Check:
    if manager.enabled:
        return Check("subsystem", Severity.OK, "runtimes.enabled is true")
    return Check(
        "subsystem",
        Severity.DEGRADES,
        "runtimes.enabled is false, so nothing can be provisioned",
        "set [runtimes] enabled = true (sizing and status work regardless)",
    )


async def _billing_check(manager: Any) -> list[Check]:
    """Anything running costs money, and that outranks every other finding."""
    try:
        statuses = await manager.status()
    except Exception as exc:
        return [
            Check(
                "billing",
                Severity.DEGRADES,
                f"could not be determined ({type(exc).__name__}: {exc})",
                "run `llmcore-runtimes status` for the detail",
            )
        ]

    live = [s for s in statuses if getattr(s.phase, "is_billing", False)]
    orphans = [s for s in statuses if s.error and "orphan" in str(s.error)]
    checks: list[Check] = []
    if orphans:
        checks.append(
            Check(
                "billing: orphans",
                Severity.MONEY,
                f"{len(orphans)} runtime(s) exist that llmcore did not start: "
                + ", ".join(sorted(s.name for s in orphans)),
                "adopt one to make it killable, then `down` it",
            )
        )
    tracked = [s for s in live if s not in orphans]
    if tracked:
        spent = ", ".join(
            f"{s.name} ({s.spend_so_far})" if s.spend_so_far else s.name
            for s in sorted(tracked, key=lambda s: s.name)
        )
        checks.append(
            Check("billing: running", Severity.MONEY, f"{len(tracked)} running: {spent}")
        )
    if not checks:
        checks.append(
            Check("billing", Severity.OK, "nothing is running, as far as llmcore sees")
        )
    return checks


def _daily_spend_checks(manager: Any) -> list[Check]:
    """Today's spend per unit, against whatever cap is configured.

    Reported even with no cap set, because the number is the point: a cap
    nobody has chosen yet is best chosen from a week of real figures rather
    than guessed. The agent circuit breaker's ``$1.00`` default is what
    guessing looks like.
    """
    ledger = getattr(manager, "ledger", None)
    caps = getattr(manager, "_daily_caps", {}) or {}
    if ledger is None:
        return []

    checks: list[Check] = []
    units = set(caps)
    try:
        units |= {e.unit for e in ledger.entries()}
    except Exception as exc:  # pragma: no cover - defensive
        return [Check("spend: today", Severity.DEGRADES, f"ledger unreadable: {exc}")]

    for unit in sorted(units, key=str):
        try:
            total = ledger.total(unit)
        except Exception as exc:  # pragma: no cover - defensive
            checks.append(
                Check(f"spend: {unit}", Severity.DEGRADES, f"could not total: {exc}")
            )
            continue
        cap = caps.get(unit)
        if cap is None:
            checks.append(
                Check(
                    f"spend: {unit} today",
                    Severity.OK,
                    f"{total.rendered()} across {total.runs} runtime(s); no daily cap set",
                    "set runtimes.budget.max_usd_per_day once you know your normal",
                )
            )
            continue
        if total.amount >= cap:
            checks.append(
                Check(
                    f"spend: {unit} today",
                    Severity.BLOCKS,
                    f"{total.rendered()} of {unit.amount(cap)} — provisioning is "
                    f"refused until the window rolls over",
                    "raise the cap deliberately, or wait",
                )
            )
        elif total.amount >= cap * 0.8:
            checks.append(
                Check(
                    f"spend: {unit} today",
                    Severity.MONEY,
                    f"{total.rendered()} of {unit.amount(cap)}",
                )
            )
        else:
            checks.append(
                Check(
                    f"spend: {unit} today",
                    Severity.OK,
                    f"{total.rendered()} of {unit.amount(cap)}",
                )
            )
    return checks


async def run_diagnostics(manager: Any, *, backend: str | None = None) -> list[Check]:
    """Run every check, generic ones first, then each backend's own.

    Args:
        manager: A ``RuntimeManager``.
        backend: Only diagnose this backend. ``None`` does all of them,
            because a problem with the backend you are *not* using today is
            still worth knowing before you need it.
    """
    checks: list[Check] = [_enabled_check(manager), _state_dir_check(manager)]
    checks.extend(await _billing_check(manager))
    checks.extend(_daily_spend_checks(manager))

    names = [backend] if backend else manager.backends
    for name in names:
        try:
            runtime = manager._backend_for(name)
        except Exception as exc:
            checks.append(
                Check(f"{name}: backend", Severity.BLOCKS, f"not registered: {exc}")
            )
            continue
        diagnose = getattr(runtime, "diagnose", None)
        if not callable(diagnose):
            checks.append(
                Check(
                    f"{name}: checks",
                    Severity.OK,
                    "this backend publishes no checks of its own",
                )
            )
            continue
        try:
            checks.extend(await diagnose())
        except Exception as exc:
            checks.append(
                Check(
                    f"{name}: checks",
                    Severity.DEGRADES,
                    f"the backend's own checks failed to run "
                    f"({type(exc).__name__}: {exc})",
                )
            )
    return checks


# ---------------------------------------------------------------------------
# Helpers the backends share
# ---------------------------------------------------------------------------


def key_check(label: str, env_var: str, *, value: str | None) -> Check:
    """Whether a credential is present. Never reports the value."""
    if value:
        return Check(f"{label}: credential", Severity.OK, f"{env_var} is set")
    return Check(
        f"{label}: credential",
        Severity.BLOCKS,
        f"{env_var} is not set",
        f"export {env_var}, or point *.api_key_env_var at the variable you use",
    )


def binary_check(label: str, path: str) -> Check:
    """Whether an external CLI the backend drives is actually runnable."""
    resolved = shutil.which(path) or (path if os.path.isfile(path) else None)
    if resolved and os.access(resolved, os.X_OK):
        return Check(f"{label}: cli", Severity.OK, f"{resolved} is executable")
    return Check(
        f"{label}: cli",
        Severity.BLOCKS,
        f"{path!r} was not found on PATH and is not an executable file",
        "install it, or set the backend's cli_path to its location",
    )


async def reachable_check(
    label: str,
    what: str,
    probe: Any,
    *,
    # A bounded probe is the point of this helper: `doctor` must finish even
    # when a backend does not answer, so the deadline belongs here rather
    # than at every call site.
    timeout: float = 20.0,  # noqa: ASYNC109
) -> Check:
    """Whether a read-only endpoint answers, without spending anything."""
    try:
        result = await asyncio.wait_for(probe(), timeout=timeout)
    except asyncio.TimeoutError:
        return Check(
            f"{label}: {what}", Severity.DEGRADES, f"timed out after {timeout:g}s"
        )
    except Exception as exc:
        return Check(
            f"{label}: {what}",
            Severity.DEGRADES,
            f"unreachable ({type(exc).__name__}: {exc})",
        )
    return Check(f"{label}: {what}", Severity.OK, str(result))
