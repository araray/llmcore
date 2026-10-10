# src/llmcore/runtimes/ledger.py
"""A daily spend ledger and cap for provisioned compute.

The gap this fills was found by writing the agent-lens migration guide:
agent-lens keeps a spend ledger and a ``budget_units_per_day`` cap, and llmcore
had **no per-day budget at all**. Its ceilings were per *runtime*
(``max_spend``) and per *turn* (``[routing.budget]``) — so ten runtimes, each
politely under its own ceiling, could spend ten times what anyone intended, and
nothing would object.

**One cap cannot span two units.** Colab bills compute units with no published
dollar value; gpu.ai and DeepInfra bill dollars. So the ledger totals per unit
and the cap is configured per unit, for the same reason
:class:`~llmcore.runtimes.models.CostUnit` exists: there is no defensible
exchange rate, and inventing one to produce a single tidy number would be the
cost model's recurring bug in a new place.

**Unknown spend is not zero.** An entry whose amount could not be determined
is recorded with ``amount=None`` and counted as an *unpriced run*, not as
``0.00``. A day of unpriceable runtimes must not read as a free day, so a total
carrying unpriced entries says so and the cap refuses to claim it is being
honoured.

The file is append-only JSON lines under the runtime state directory, so it is
readable with ``cat`` and survives a crash mid-write as everything up to the
last complete line.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import CostUnit

logger = logging.getLogger(__name__)

__all__ = ["DailyTotal", "SpendEntry", "SpendLedger"]


@dataclass(frozen=True, slots=True)
class SpendEntry:
    """One runtime's spend, as recorded when it stopped.

    Attributes:
        at: When this was recorded, in UTC.
        runtime: Backend name.
        name: The runtime's name.
        amount: What it cost, or ``None`` when that could not be determined.
        unit: What ``amount`` is denominated in.
        external_id: Backend-side id, for tracing a figure back.
    """

    at: datetime
    runtime: str
    name: str
    amount: float | None
    unit: CostUnit
    external_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "runtime": self.runtime,
            "name": self.name,
            "amount": self.amount,
            "unit": str(self.unit),
            "external_id": self.external_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpendEntry | None:
        """Parse one line, or ``None`` if it cannot be trusted.

        A malformed line is skipped rather than raising: the ledger's job is to
        total what is known, and refusing to read the whole file because one
        line is broken would lose every good entry with it.
        """
        try:
            moment = datetime.fromisoformat(str(payload["at"]))
        except Exception:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        amount = payload.get("amount")
        if amount is not None and not isinstance(amount, (int, float)):
            amount = None
        try:
            unit = CostUnit(str(payload.get("unit", CostUnit.COMPUTE_UNIT)))
        except ValueError:
            unit = CostUnit.COMPUTE_UNIT
        return cls(
            at=moment,
            runtime=str(payload.get("runtime", "")),
            name=str(payload.get("name", "")),
            amount=float(amount) if amount is not None else None,
            unit=unit,
            external_id=str(payload.get("external_id", "")),
        )


@dataclass(frozen=True, slots=True)
class DailyTotal:
    """What has been spent in one unit over a window.

    Attributes:
        unit: The unit totalled.
        amount: Sum of the entries that had an amount.
        runs: How many runtimes contributed.
        unpriced_runs: How many of those had no determinable cost.
        since: Start of the window.
    """

    unit: CostUnit
    amount: float
    runs: int
    unpriced_runs: int
    since: datetime

    @property
    def is_partial(self) -> bool:
        """Whether ``amount`` is a floor rather than a total."""
        return self.unpriced_runs > 0

    def rendered(self) -> str:
        """The total in its own unit, flagged when incomplete."""
        text = self.unit.amount(self.amount)
        if self.is_partial:
            text += f" plus {self.unpriced_runs} unpriced run(s)"
        return text


@dataclass(slots=True)
class SpendLedger:
    """Append-only record of what provisioned compute has cost.

    Attributes:
        path: The JSON-lines file. Created on first write.
    """

    path: Path
    _entries: list[SpendEntry] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(self, entry: SpendEntry) -> None:
        """Append *entry*. Never raises -- a ledger write must not undo a stop.

        This is called from teardown paths. If recording the cost of releasing
        a runtime could fail the release, the ledger would be a way to keep
        paying for something, which is the opposite of its purpose.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")
            if self._entries is not None:
                self._entries.append(entry)
        except Exception as exc:
            logger.warning(
                "Could not write the spend ledger at %s (%s: %s). The runtime "
                "action itself was unaffected.",
                self.path,
                type(exc).__name__,
                exc,
            )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def entries(self, *, refresh: bool = False) -> list[SpendEntry]:
        """Every readable entry, oldest first."""
        if self._entries is not None and not refresh:
            return list(self._entries)
        out: list[SpendEntry] = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._entries = []
            return []
        except Exception as exc:
            logger.warning("Could not read the spend ledger: %s", exc)
            return []
        skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = SpendEntry.from_dict(json.loads(line))
            except Exception:
                entry = None
            if entry is None:
                skipped += 1
                continue
            out.append(entry)
        if skipped:
            logger.debug("Skipped %d unreadable ledger line(s).", skipped)
        out.sort(key=lambda e: e.at)
        self._entries = out
        return list(out)

    def total(
        self,
        unit: CostUnit,
        *,
        since: datetime | None = None,
        now: datetime | None = None,
    ) -> DailyTotal:
        """Total one unit since *since*, defaulting to the start of today UTC.

        UTC rather than local time, deliberately: a cap that moves with the
        machine's timezone is a cap whose reset time nobody can state, and the
        backends all report in UTC.
        """
        moment = now or datetime.now(timezone.utc)
        start = since or datetime.combine(
            moment.astimezone(timezone.utc).date(), datetime.min.time(), timezone.utc
        )
        amount = 0.0
        runs = 0
        unpriced = 0
        for entry in self.entries():
            if entry.unit is not unit or entry.at < start:
                continue
            runs += 1
            if entry.amount is None:
                unpriced += 1
            else:
                amount += entry.amount
        return DailyTotal(
            unit=unit, amount=amount, runs=runs, unpriced_runs=unpriced, since=start
        )

    def prune(self, *, keep_days: int = 90, now: datetime | None = None) -> int:
        """Drop entries older than *keep_days*. Returns how many were removed.

        Rewritten atomically through a temporary file, so a crash mid-prune
        leaves the old ledger intact rather than a half-written one.
        """
        moment = now or datetime.now(timezone.utc)
        cutoff = moment - timedelta(days=max(0, int(keep_days)))
        kept = [e for e in self.entries(refresh=True) if e.at >= cutoff]
        removed = len(self._entries or []) - len(kept)
        if removed <= 0:
            return 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(self.path.parent), delete=False
            )
            with handle as out:
                for entry in kept:
                    out.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")
            os.replace(handle.name, self.path)
            self._entries = kept
        except Exception as exc:
            logger.warning("Could not prune the spend ledger: %s", exc)
            return 0
        return removed

    def days(self, *, limit: int = 7) -> dict[date, dict[CostUnit, DailyTotal]]:
        """Per-day totals per unit, most recent *limit* days that have entries."""
        buckets: dict[date, dict[CostUnit, list[SpendEntry]]] = {}
        for entry in self.entries():
            day = entry.at.astimezone(timezone.utc).date()
            buckets.setdefault(day, {}).setdefault(entry.unit, []).append(entry)
        out: dict[date, dict[CostUnit, DailyTotal]] = {}
        for day in sorted(buckets, reverse=True)[: max(0, int(limit))]:
            out[day] = {}
            for unit, rows in buckets[day].items():
                priced = [r.amount for r in rows if r.amount is not None]
                out[day][unit] = DailyTotal(
                    unit=unit,
                    amount=sum(priced),
                    runs=len(rows),
                    unpriced_runs=len(rows) - len(priced),
                    since=datetime.combine(day, datetime.min.time(), timezone.utc),
                )
        return out
