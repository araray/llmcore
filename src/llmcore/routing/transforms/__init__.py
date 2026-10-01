# src/llmcore/routing/transforms/__init__.py
"""Prompt transforms: what must not leave this machine.

A transform sees the request **and the target it is about to go to**, which is
what makes this more than a redaction pass: the same prompt can go verbatim to
a model on your own GPU and be redacted, rerouted or refused on its way to a
vendor.

The central claim of this layer, stated plainly because the documentation must
not oversell it:

    **Redaction is not a guarantee. Routing is.**

A detector that misses one identifier has leaked it, and no detector catches
everything. So the strong primitive here is ``constrain``: when a prompt
contains personal data, change *where it goes* — to a pool that never leaves
the machine. Redaction stays available and can be stacked on top as defence in
depth, but it is the weaker half and is documented as such.

Actions, in increasing severity:

``allow``
    Nothing found, or nothing worth acting on.
``redact``
    Send a rewritten prompt. Defence in depth, not a guarantee.
``constrain``
    Send the prompt, but only to a pool that satisfies the policy. This is the
    contribution: detection changes the destination.
``block``
    Do not send it at all. The honest answer when a prompt must not be
    processed and no acceptable target exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ..models import TransformAction, TransformResult

if TYPE_CHECKING:
    from ..models import RoutingRequest, Target
    from ..protocols import PromptTransform

logger = logging.getLogger(__name__)

__all__ = [
    "TransformChain",
    "available_transforms",
    "build_transform",
    "register_transform",
]

_FACTORIES: dict[str, Callable[..., PromptTransform]] = {}


def register_transform(
    name: str, factory: Callable[..., PromptTransform], *, replace: bool = False
) -> None:
    """Register a transform factory under ``name``."""
    key = name.strip().lower()
    if key in _FACTORIES and not replace:
        raise ValueError(f"Transform '{key}' is already registered; pass replace=True to swap it.")
    _FACTORIES[key] = factory


def available_transforms() -> list[str]:
    """Return every registered transform name."""
    return sorted(_FACTORIES)


def build_transform(
    name: str, *, config: Mapping[str, Any] | None = None
) -> PromptTransform | None:
    """Build one transform by name.

    Unlike a classifier, a transform that cannot be built is **loud**. A
    classifier that fails to load costs routing quality; a *privacy* transform
    that fails to load silently costs the user the protection they configured,
    and they would have no way to know. So this logs an error rather than a
    warning, and :class:`TransformChain` can be configured to fail closed.
    """
    key = name.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        logger.error(
            "Unknown transform '%s'; it will NOT run. Available: %s",
            name,
            ", ".join(available_transforms()),
        )
        return None
    try:
        return factory(config=dict(config or {}))
    except Exception as exc:
        logger.error(
            "Transform '%s' could not be built (%s); it will NOT run. If it was configured for "
            "privacy, treat this as a policy failure rather than a warning.",
            key,
            exc,
        )
        return None


class TransformChain:
    """Applies transforms in order, carrying the rewrite forward.

    Args:
        transforms: The chain, in configured order. Order matters here in a
            way it does not for classifiers: each transform sees the previous
            one's rewrite, so a redaction upstream changes what a detector
            downstream sees.
        fail_closed: What to do when a transform raises. ``True`` turns the
            error into a ``block``, which is the right default for anything
            protecting data: a detector that crashed has not cleared the
            prompt, and treating a crash as "nothing found" is how a privacy
            feature becomes decoration. ``False`` logs and continues, for
            chains doing something non-protective.
    """

    def __init__(
        self, transforms: Sequence[PromptTransform], *, fail_closed: bool = True
    ) -> None:
        self._transforms = tuple(transforms)
        self._fail_closed = fail_closed

    @property
    def transforms(self) -> tuple[PromptTransform, ...]:
        return self._transforms

    def __bool__(self) -> bool:
        return bool(self._transforms)

    def names(self) -> list[str]:
        return [getattr(t, "name", type(t).__name__) for t in self._transforms]

    async def apply(
        self, request: RoutingRequest, target: Target
    ) -> tuple[RoutingRequest, tuple[TransformResult, ...]]:
        """Run the chain and return the final request plus every result.

        Stops early on ``block``: once a transform has refused, running the
        rest would be pointless and might log findings about a prompt that is
        not going anywhere.
        """
        from dataclasses import replace as _replace

        current = request
        results: list[TransformResult] = []
        for transform in self._transforms:
            name = getattr(transform, "name", type(transform).__name__)
            try:
                result = await transform.apply(current, target)
            except Exception:
                logger.error("Transform '%s' raised.", name, exc_info=True)
                if self._fail_closed:
                    results.append(
                        TransformResult(
                            action=TransformAction.BLOCK,
                            reason=(
                                f"transform '{name}' failed and the chain is configured to fail "
                                f"closed; the prompt has not been cleared"
                            ),
                            source=name,
                        )
                    )
                    return current, tuple(results)
                continue

            results.append(result)
            changes: dict[str, Any] = {}
            if result.prompt is not None:
                changes["prompt"] = result.prompt
            if result.messages is not None:
                changes["messages"] = result.messages
            if result.system is not None:
                changes["system"] = result.system
            if changes:
                current = _replace(current, **changes)
            if result.action is TransformAction.BLOCK:
                break
        return current, tuple(results)


def _register_builtins() -> None:
    from . import pii  # noqa: F401
    from . import magic  # noqa: F401


_register_builtins()
