# src/llmcore/routing/lanes.py
"""Lanes: named destinations a classifier can pick.

The single most useful idea in the prior art (Arch-Router, arXiv:2506.16655)
is that a router should choose a **lane**, not a model. The classifier never
learns a vendor's model ids, so swapping ``gpt-5.4`` for something newer is a
config edit and not a retraining job.

That one indirection generalises every variant in the original request —
complexity tiers, speed tiers, domain routing, a privacy class — into one
mechanism. "Focus groups for complexity levels" are lanes whose names happen
to be complexity levels, and nothing in this module knows or cares::

    [routing.lanes]
    trivial  = "pool:cheap"
    standard = "pool:main"
    deep     = "anthropic:claude-opus-5-5?effort=max"
    private  = "pool:local_only"

Lane names are the user's own words. A lane may also carry parameters it
imposes on whatever serves it (``effort``, ``max_tokens``), which is how
"route this to the deep lane" can mean "and think harder" without the
classifier knowing what effort means for a given vendor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from .models import Target

logger = logging.getLogger(__name__)

__all__ = ["POOL_PREFIX", "Lane", "parse_lanes"]

#: A lane destination beginning with this names a pool rather than a target.
#: Needed because ``main`` is a perfectly good pool name *and* a perfectly
#: good provider instance name, so the two must be distinguishable.
POOL_PREFIX = "pool:"


@dataclass(frozen=True, slots=True)
class Lane:
    """A named destination, pointing at a pool or a single target.

    Attributes:
        name: The lane's name, as a classifier returns it and as the proxy
            exposes it (``model="lane:deep"``).
        pool: The pool that serves this lane, if it points at one.
        target: The single target that serves this lane, if it points at one.
        params: Parameters the lane imposes, applied over the target's own but
            under an explicit per-call argument (see spec §6.1).
        description: Free text. Not decoration: the zero-shot classifiers
            score a prompt against lane *descriptions*, so this is the lane's
            training data. ``deep`` with a good description routes better than
            ``deep`` alone.
    """

    name: str
    pool: str | None = None
    target: Target | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Lane.name is required.")
        if (self.pool is None) == (self.target is None):
            raise ValueError(
                f"Lane '{self.name}' must name exactly one destination: either a pool "
                f'(pool = "cheap" or "pool:cheap") or a target ("openai:gpt-5.4").'
            )

    @property
    def destination(self) -> str:
        """The destination as it would be written in config."""
        if self.pool:
            return f"{POOL_PREFIX}{self.pool}"
        assert self.target is not None
        return self.target.spec()

    @classmethod
    def from_config(cls, name: str, raw: Any) -> Lane:
        """Build a lane from its config value.

        Accepts the short form and the long one, because the short form is
        what people write and the long one is what they grow into::

            deep = "anthropic:claude-opus-5-5?effort=max"

            [routing.lanes.deep]
            target = "anthropic:claude-opus-5-5"
            params = { effort = "max" }
            description = "Multi-step reasoning, proofs, architecture review"
        """
        if isinstance(raw, str):
            return cls._from_destination(name, raw, {}, None)

        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Lane '{name}' must be a destination string or a table, got {type(raw).__name__}."
            )

        params = dict(raw.get("params") or {})
        description = raw.get("description")
        pool = raw.get("pool")
        target = raw.get("target") or raw.get("destination")

        if pool and target:
            raise ValueError(
                f"Lane '{name}' sets both pool and target; it must have exactly one destination."
            )
        if pool:
            return cls(name=name, pool=str(pool), params=params, description=description)
        if target:
            return cls._from_destination(name, target, params, description)
        raise ValueError(
            f'Lane \'{name}\' has no destination. Set pool = "cheap" or '
            f'target = "openai:gpt-5.4".'
        )

    @classmethod
    def _from_destination(
        cls,
        name: str,
        destination: Any,
        params: dict[str, Any],
        description: str | None,
    ) -> Lane:
        if isinstance(destination, Target):
            return cls(name=name, target=destination, params=params, description=description)
        text = str(destination).strip()
        if text.lower().startswith(POOL_PREFIX):
            pool = text[len(POOL_PREFIX) :].strip()
            if not pool:
                raise ValueError(f"Lane '{name}' names an empty pool.")
            return cls(name=name, pool=pool, params=params, description=description)
        return cls(
            name=name,
            target=Target.parse(text),
            params=params,
            description=description,
        )


def parse_lanes(raw: Mapping[str, Any] | None) -> dict[str, Lane]:
    """Build the lane table from ``[routing.lanes]``.

    A lane that cannot be parsed is logged and skipped rather than raising:
    one malformed lane should cost that lane, not the whole process. Routing
    without a lane still works — it falls through to the default pool — so
    degrading here is strictly better than refusing to start.
    """
    lanes: dict[str, Lane] = {}
    for name, value in (raw or {}).items():
        try:
            lanes[str(name).strip().lower()] = Lane.from_config(str(name).strip().lower(), value)
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring malformed lane '%s': %s", name, exc)
    return lanes
