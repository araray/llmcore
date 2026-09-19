# src/llmcore/shared_events.py
"""
Federated event spine shared across the wairu ecosystem (plan S-3 / P-01).

This module is intentionally **stdlib-only** so that every repo in the
ecosystem (wairu, llmcore, semantiscan, grimoire) can depend on it without
introducing dependency cycles or third-party requirements.

It provides three small building blocks:

1. ``UnifiedEvent`` — a minimal, serialization-friendly event record that all
   repos can emit and consume regardless of their internal event models.
2. Correlation helpers — a :mod:`contextvars`-based correlation id that
   flows naturally across ``asyncio`` tasks and threads started with a copied
   context, so events produced anywhere inside one logical operation (e.g.
   one wairu fly iteration) share the same ``correlation_id``.
3. A tiny process-global sink registry — consumers call
   :func:`register_sink` to receive every emitted event. Sink exceptions are
   swallowed (with a debug log) so one misbehaving sink can never break an
   emitter.

Emitting is designed to be cheap when nobody is listening: check
:func:`has_sinks` before building a payload if construction is non-trivial.

Usage:
    >>> from llmcore import shared_events
    >>>
    >>> def my_sink(event: shared_events.UnifiedEvent) -> None:
    ...     print(event.source, event.type, event.correlation_id)
    >>>
    >>> shared_events.register_sink(my_sink)
    >>> with shared_events.correlation_context() as cid:
    ...     shared_events.emit(
    ...         shared_events.new_event("llmcore", "llm.call", {"model": "gpt-4o"})
    ...     )

References:
    - IMPROVEMENT_PLAN_2026-07.md Section 5.2 (S-3 federated event spine)
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED EVENT
# =============================================================================


@dataclass
class UnifiedEvent:
    """A minimal cross-repo event record.

    Attributes:
        source: Emitting system (e.g. ``"llmcore"``, ``"semantiscan"``,
            ``"wairu"``).
        type: Dotted event type (e.g. ``"cognitive.phase_completed"``).
        payload: Event-specific data; should be JSON-serializable.
        correlation_id: Id correlating events of one logical operation.
        event_id: Unique id for this event.
        timestamp: Unix timestamp (seconds) when the event was created.
    """

    source: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    event_id: str = field(default_factory=lambda: f"uev-{uuid.uuid4().hex[:16]}")
    timestamp: float = field(default_factory=time.time)


def new_event(
    source: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> UnifiedEvent:
    """Build a :class:`UnifiedEvent`, defaulting correlation from context.

    Args:
        source: Emitting system name.
        event_type: Dotted event type.
        payload: Event-specific data (defaults to empty dict).
        correlation_id: Explicit correlation id; when ``None`` the current
            contextvar value (see :func:`get_correlation_id`) is used.

    Returns:
        A populated UnifiedEvent.
    """
    return UnifiedEvent(
        source=source,
        type=event_type,
        payload=payload if payload is not None else {},
        correlation_id=correlation_id if correlation_id is not None else get_correlation_id(),
    )


# =============================================================================
# CORRELATION CONTEXT
# =============================================================================

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llmcore_shared_correlation_id", default=None
)


def new_correlation_id() -> str:
    """Generate a fresh correlation id."""
    return f"corr-{uuid.uuid4().hex[:16]}"


def set_correlation_id(correlation_id: str | None) -> contextvars.Token[str | None]:
    """Set the current correlation id.

    Args:
        correlation_id: The id to set (``None`` clears it for this context).

    Returns:
        A contextvars Token that can be passed to :func:`reset_correlation_id`.
    """
    return _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    """Return the correlation id for the current context (or ``None``)."""
    return _correlation_id.get()


def reset_correlation_id(token: contextvars.Token[str | None]) -> None:
    """Restore the correlation id captured by a :func:`set_correlation_id` call."""
    _correlation_id.reset(token)


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Iterator[str]:
    """Scope a correlation id to a ``with`` block.

    Args:
        correlation_id: Id to activate; a fresh one is generated when ``None``.

    Yields:
        The active correlation id.

    The previous correlation id (if any) is restored on exit, so nested
    scopes behave as expected.
    """
    cid = correlation_id if correlation_id is not None else new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


# =============================================================================
# SINK REGISTRY
# =============================================================================

Sink = Callable[[UnifiedEvent], None]

_sinks: list[Sink] = []
_sinks_lock = threading.Lock()


def register_sink(sink: Sink) -> None:
    """Register a sink to receive every emitted :class:`UnifiedEvent`.

    Registering the same callable twice is a no-op (delivered once).
    """
    with _sinks_lock:
        if sink not in _sinks:
            _sinks.append(sink)


def unregister_sink(sink: Sink) -> None:
    """Remove a previously registered sink (no-op if not registered)."""
    with _sinks_lock:
        try:
            _sinks.remove(sink)
        except ValueError:
            pass


def has_sinks() -> bool:
    """Return True when at least one sink is registered.

    Emitters should use this as a cheap guard before building expensive
    payloads.
    """
    return bool(_sinks)


def emit(event: UnifiedEvent) -> None:
    """Deliver *event* to all registered sinks.

    Sink exceptions are swallowed with a debug log so a faulty sink cannot
    break the emitting code path. No-op (and near-free) when no sinks are
    registered.
    """
    if not _sinks:
        return
    with _sinks_lock:
        sinks = tuple(_sinks)
    for sink in sinks:
        try:
            sink(event)
        except Exception:
            logger.debug("shared_events sink %r raised; event dropped for it", sink, exc_info=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Sink",
    "UnifiedEvent",
    "correlation_context",
    "emit",
    "get_correlation_id",
    "has_sinks",
    "new_correlation_id",
    "new_event",
    "register_sink",
    "reset_correlation_id",
    "set_correlation_id",
    "unregister_sink",
]
