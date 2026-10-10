# src/llmcore/routing/budget.py
"""A per-turn step and spend budget (cost-control design, layer 4).

The agent circuit breaker bounds an agent run. The chat path had no per-turn
budget at all -- and ``chat()`` is what the proxy exposes to agent harnesses,
which is where the expensive turns in the measurements actually came from.

**Why steps and not prompt complexity.** Measured on this project's own
traffic: routing by predicted prompt complexity is worth about **2.2% of
spend**, while turns of more than 200 steps are **62%** of it. Cost is roughly
``steps x context x cached_rate``, and context per step is close to flat -- so
the quantity worth bounding is the one that is observable directly, cheaply and
exactly. A prompt's eventual turn length is not predictable from its text; the
classifier work reached the same conclusion from the other side, finding its
label predicted session length rather than required capability.

**Three decisions the design left open, and what they resolved to.**

1. *Where does the accumulator live?* Caller-owned and explicitly fed, like
   :class:`~llmcore.agents.resilience.circuit_breaker.AgentCircuitBreaker`.
   Routing sees a request, not a turn, and only the caller knows where one
   begins. Building this on the routing event spine was the tempting
   alternative and is wrong: ``emit_attempt`` is guarded by ``has_sinks()``, so
   the budget would silently stop counting whenever nobody was listening -- a
   spend guard must not depend on observability being switched on.
2. *Is mid-turn ``constrain`` safe?* No, so it is not offered. Changing model
   inside a conversation changes behaviour, and with preserved-thinking models
   it invalidates reasoning blocks already in the history. :data:`BudgetAction`
   therefore stops at ``stop``; a caller who wants a cheaper target for a
   *later* turn can choose one itself, with the state this object reports.
3. *Who sets the budget?* ``caller > policy > config``, mirroring the
   classifier chain's authority order.

**Unknown cost is not zero.** A step whose target cannot be priced is counted
as a step and recorded as unpriced, and :attr:`BudgetVerdict.cost_is_partial`
says so. A budget that quietly treated unpriced calls as free would never trip
its spend ceiling -- which is the exact failure this subsystem's cost model has
had at seven separate layers.

There are **no default limits**. An unset dial is unbounded. The agent circuit
breaker shipped with a ``$1.00`` default that would have cut off **50.7% of
this project's normal turns**, and a number nobody chose is worse than no
number at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BudgetAction",
    "BudgetPolicy",
    "BudgetState",
    "BudgetVerdict",
    "TurnBudget",
]


class BudgetAction(StrEnum):
    """What to do when a budget is breached.

    Attributes:
        REPORT: Record it and change nothing. For measuring before enforcing.
        WARN: Surface the projection to the caller, but serve the request.
        STOP: Refuse the next step.
    """

    REPORT = "report"
    WARN = "warn"
    STOP = "stop"


class BudgetState(StrEnum):
    """Where a turn stands against its budget.

    Attributes:
        OK: Under the warning threshold.
        WARNING: Past ``warn_at_fraction`` of a limit, or projected to exceed
            one, but not yet over.
        EXCEEDED: A limit has been reached.
    """

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """The dials. An unset limit is unbounded.

    Attributes:
        max_steps: Steps a turn may take. The unit the measurements are in,
            and model-independent.
        max_cost_usd: Dollars a turn may spend. What an operator actually
            budgets, but unenforceable against targets with no price.
        warn_at_fraction: Fraction of a limit at which the state becomes
            ``WARNING``. Also applied to the *projection*, so a turn heading
            for the cap is flagged before it arrives.
        action: What a breach does.
    """

    max_steps: int | None = None
    max_cost_usd: float | None = None
    warn_at_fraction: float = 0.8
    action: BudgetAction = BudgetAction.WARN

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive, or None for unbounded.")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive, or None for unbounded.")
        if not 0.0 < self.warn_at_fraction <= 1.0:
            raise ValueError("warn_at_fraction must be in (0, 1].")

    @property
    def is_bounded(self) -> bool:
        """Whether this policy constrains anything at all."""
        return self.max_steps is not None or self.max_cost_usd is not None

    @classmethod
    def from_config(cls, get: Any) -> BudgetPolicy:
        """Build a policy from ``config.get``, leaving unset dials unbounded."""

        def number(key: str) -> Any:
            value = get(key, None)
            # Type-checked before coercion, not coerced hopefully: ``float()``
            # of a mock or a stray object succeeds and would install a budget
            # nobody configured.
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        steps = number("routing.budget.max_steps")
        cost = number("routing.budget.max_cost_usd")
        warn = number("routing.budget.warn_at_fraction")
        raw_action = get("routing.budget.action", None)
        try:
            action = BudgetAction(str(raw_action)) if raw_action else BudgetAction.WARN
        except ValueError:
            logger.warning(
                "Unknown routing.budget.action %r; using %s.",
                raw_action,
                BudgetAction.WARN,
            )
            action = BudgetAction.WARN
        return cls(
            max_steps=int(steps) if steps else None,
            max_cost_usd=float(cost) if cost else None,
            warn_at_fraction=float(warn) if warn and 0 < warn <= 1 else 0.8,
            action=action,
        )


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """A point-in-time reading of a turn against its budget.

    Attributes:
        state: Where the turn stands.
        action: What the policy says to do about it. ``REPORT`` even when
            exceeded, if that is what was configured.
        reason: Why, when the state is not ``OK``.
        steps: Steps taken so far.
        cost_usd: Dollars accounted for so far. See ``cost_is_partial``.
        projected_cost_usd: Cost at ``max_steps``, extrapolated from the mean
            cost per step. ``None`` without both a step limit and a priced
            step to extrapolate from.
        cost_is_partial: Some step could not be priced, so ``cost_usd`` is a
            floor rather than a total. A spend ceiling cannot be enforced
            while this is true, and the verdict says so rather than implying
            the limit is being honoured.
        should_stop: Convenience: the policy says to refuse the next step.
    """

    state: BudgetState
    action: BudgetAction
    reason: str | None = None
    steps: int = 0
    cost_usd: float = 0.0
    projected_cost_usd: float | None = None
    cost_is_partial: bool = False

    @property
    def should_stop(self) -> bool:
        """Whether the caller must not take another step."""
        return self.state is BudgetState.EXCEEDED and self.action is BudgetAction.STOP

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping, for an event or a usage block."""
        return {
            "state": str(self.state),
            "action": str(self.action),
            "reason": self.reason,
            "steps": self.steps,
            "cost_usd": round(self.cost_usd, 6),
            "projected_cost_usd": (
                round(self.projected_cost_usd, 6)
                if self.projected_cost_usd is not None
                else None
            ),
            "cost_is_partial": self.cost_is_partial,
        }


@dataclass(slots=True)
class TurnBudget:
    """Accumulates one turn's steps and spend, and reports on them.

    Mutable and single-threaded by design: it belongs to one turn, and the
    caller feeds it. Usage::

        budget = TurnBudget(BudgetPolicy(max_steps=200, action=BudgetAction.STOP))

        verdict = budget.check()
        if verdict.should_stop:
            raise RuntimeError(verdict.reason)

        answer = await llm.chat(...)
        budget.record(cost_usd=priced_or_none, input_tokens=..., output_tokens=...)

    Attributes:
        policy: The dials this turn is held to.
        steps: Completed steps.
        cost_usd: Dollars accounted for.
        unpriced_steps: Steps whose cost was unknown.
        input_tokens: Prompt tokens across the turn, cache reads included.
        cached_tokens: The cache-read share, where the provider reports it.
        output_tokens: Completion tokens across the turn.
    """

    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    steps: int = 0
    cost_usd: float = 0.0
    unpriced_steps: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        cost_usd: float | None = None,
        input_tokens: int | None = None,
        cached_tokens: int | None = None,
        output_tokens: int | None = None,
        steps: int = 1,
    ) -> None:
        """Add one completed step.

        Args:
            cost_usd: What the step cost, or ``None`` when the target could
                not be priced. ``None`` is **not** read as zero: the step still
                counts, and the turn is marked as only partially priced.
            input_tokens: Prompt tokens, including cache reads.
            cached_tokens: The cache-read share, when reported.
            output_tokens: Completion tokens.
            steps: How many steps this represents. Defaults to one.
        """
        self.steps += int(steps)
        if cost_usd is None:
            self.unpriced_steps += int(steps)
        else:
            self.cost_usd += float(cost_usd)
        self.input_tokens += int(input_tokens or 0)
        self.cached_tokens += int(cached_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    @property
    def cost_is_partial(self) -> bool:
        """Whether any step went unpriced, making ``cost_usd`` a floor."""
        return self.unpriced_steps > 0

    @property
    def cost_per_step(self) -> float | None:
        """Mean cost of the steps that could be priced, or ``None``."""
        priced = self.steps - self.unpriced_steps
        if priced <= 0:
            return None
        return self.cost_usd / priced

    def projected_cost_usd(self) -> float | None:
        """Cost at ``max_steps``, from the mean cost per priced step.

        ``None`` without a step limit to project to, or without a priced step
        to project from. Extrapolating from zero observations would produce a
        confident ``0.00``, which reads as "this turn is free".
        """
        if self.policy.max_steps is None:
            return None
        per_step = self.cost_per_step
        if per_step is None:
            return None
        return per_step * self.policy.max_steps

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def check(self) -> BudgetVerdict:
        """Read the turn against its policy, without changing anything."""
        projected = self.projected_cost_usd()
        base = {
            "action": self.policy.action,
            "steps": self.steps,
            "cost_usd": self.cost_usd,
            "projected_cost_usd": projected,
            "cost_is_partial": self.cost_is_partial,
        }

        if self.policy.max_steps is not None and self.steps >= self.policy.max_steps:
            return BudgetVerdict(
                state=BudgetState.EXCEEDED,
                reason=(
                    f"step budget reached ({self.steps}/{self.policy.max_steps} steps)"
                ),
                **base,
            )

        if self.policy.max_cost_usd is not None and self.cost_usd >= self.policy.max_cost_usd:
            # Reported as exceeded even when partial: the floor alone is over
            # the ceiling, so the conclusion does not depend on the gap.
            return BudgetVerdict(
                state=BudgetState.EXCEEDED,
                reason=(
                    f"spend budget reached (${self.cost_usd:,.2f} of "
                    f"${self.policy.max_cost_usd:,.2f}"
                    + (
                        f", plus {self.unpriced_steps} unpriced step(s))"
                        if self.cost_is_partial
                        else ")"
                    )
                ),
                **base,
            )

        warnings: list[str] = []
        fraction = self.policy.warn_at_fraction
        if self.policy.max_steps is not None:
            if self.steps >= self.policy.max_steps * fraction:
                warnings.append(
                    f"{self.steps} of {self.policy.max_steps} steps used"
                )
        if self.policy.max_cost_usd is not None:
            if self.cost_usd >= self.policy.max_cost_usd * fraction:
                warnings.append(
                    f"${self.cost_usd:,.2f} of ${self.policy.max_cost_usd:,.2f} spent"
                )
            if projected is not None and projected >= self.policy.max_cost_usd:
                warnings.append(
                    f"on course for ${projected:,.2f} by step "
                    f"{self.policy.max_steps}, over the ${self.policy.max_cost_usd:,.2f} "
                    f"ceiling"
                )
        if self.policy.max_cost_usd is not None and self.cost_is_partial:
            warnings.append(
                f"{self.unpriced_steps} step(s) could not be priced, so the spend "
                f"ceiling cannot be enforced"
            )

        if warnings:
            return BudgetVerdict(
                state=BudgetState.WARNING, reason="; ".join(warnings), **base
            )
        return BudgetVerdict(state=BudgetState.OK, **base)

    def with_policy(self, policy: BudgetPolicy) -> TurnBudget:
        """Return a copy held to *policy*, keeping what has been spent.

        How ``caller > policy > config`` is applied: a per-request override
        replaces the dials without resetting the meter, so a caller cannot
        clear a turn's accumulated spend by changing the limit.
        """
        return replace(self, policy=policy)
