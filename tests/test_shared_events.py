# tests/test_shared_events.py
"""
Tests for the federated event spine (llmcore.shared_events, plan S-3).

Covers:
- UnifiedEvent defaults and factory behavior
- Correlation contextvar helpers (set/get/reset, context manager, nesting)
- Contextvar isolation across asyncio tasks
- Sink registry: registration, deduplication, unregistration, faulty sinks
"""

from __future__ import annotations

import asyncio

import pytest

from llmcore import shared_events
from llmcore.shared_events import (
    UnifiedEvent,
    correlation_context,
    emit,
    get_correlation_id,
    has_sinks,
    new_correlation_id,
    new_event,
    register_sink,
    reset_correlation_id,
    set_correlation_id,
    unregister_sink,
)


@pytest.fixture(autouse=True)
def _clean_spine():
    """Ensure each test starts and ends with no sinks and no correlation id."""
    token = set_correlation_id(None)
    saved = list(shared_events._sinks)
    for sink in saved:
        unregister_sink(sink)
    yield
    for sink in list(shared_events._sinks):
        unregister_sink(sink)
    for sink in saved:
        register_sink(sink)
    reset_correlation_id(token)


# =============================================================================
# UNIFIED EVENT
# =============================================================================


class TestUnifiedEvent:
    def test_defaults(self) -> None:
        event = UnifiedEvent(source="llmcore", type="llm.call")
        assert event.source == "llmcore"
        assert event.type == "llm.call"
        assert event.payload == {}
        assert event.correlation_id is None
        assert event.event_id.startswith("uev-")
        assert event.timestamp > 0

    def test_event_ids_unique(self) -> None:
        a = UnifiedEvent(source="s", type="t")
        b = UnifiedEvent(source="s", type="t")
        assert a.event_id != b.event_id

    def test_new_event_captures_context_correlation(self) -> None:
        with correlation_context("corr-abc") as cid:
            event = new_event("wairu", "iteration.start", {"n": 1})
        assert cid == "corr-abc"
        assert event.correlation_id == "corr-abc"
        assert event.payload == {"n": 1}

    def test_new_event_explicit_correlation_wins(self) -> None:
        with correlation_context("corr-outer"):
            event = new_event("wairu", "x", correlation_id="corr-explicit")
        assert event.correlation_id == "corr-explicit"

    def test_new_event_without_context(self) -> None:
        event = new_event("semantiscan", "retrieval.done")
        assert event.correlation_id is None
        assert event.payload == {}


# =============================================================================
# CORRELATION HELPERS
# =============================================================================


class TestCorrelation:
    def test_set_get_reset(self) -> None:
        assert get_correlation_id() is None
        token = set_correlation_id("corr-1")
        assert get_correlation_id() == "corr-1"
        reset_correlation_id(token)
        assert get_correlation_id() is None

    def test_context_manager_generates_id(self) -> None:
        with correlation_context() as cid:
            assert cid.startswith("corr-")
            assert get_correlation_id() == cid
        assert get_correlation_id() is None

    def test_context_manager_nesting_restores_outer(self) -> None:
        with correlation_context("outer"):
            with correlation_context("inner"):
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"
        assert get_correlation_id() is None

    def test_context_manager_restores_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with correlation_context("boom"):
                raise RuntimeError("boom")
        assert get_correlation_id() is None

    def test_new_correlation_ids_unique(self) -> None:
        assert new_correlation_id() != new_correlation_id()

    async def test_asyncio_task_isolation(self) -> None:
        """Each task sets its own id without affecting siblings or the parent."""
        results: dict[str, str | None] = {}

        async def worker(name: str, cid: str) -> None:
            set_correlation_id(cid)
            await asyncio.sleep(0.01)
            results[name] = get_correlation_id()

        set_correlation_id("parent")
        await asyncio.gather(worker("a", "corr-a"), worker("b", "corr-b"))
        assert results == {"a": "corr-a", "b": "corr-b"}
        # Tasks run in copied contexts; parent's value is untouched.
        assert get_correlation_id() == "parent"

    async def test_asyncio_task_inherits_parent_context(self) -> None:
        """Tasks created inside a correlation scope see that correlation id."""
        with correlation_context("corr-parent"):
            observed = await asyncio.create_task(_read_correlation())
        assert observed == "corr-parent"


async def _read_correlation() -> str | None:
    await asyncio.sleep(0)
    return get_correlation_id()


# =============================================================================
# SINK REGISTRY
# =============================================================================


class TestSinkRegistry:
    def test_emit_without_sinks_is_noop(self) -> None:
        assert not has_sinks()
        emit(new_event("llmcore", "x"))  # must not raise

    def test_register_and_emit(self) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)
        assert has_sinks()

        event = new_event("llmcore", "llm.call", {"model": "m"})
        emit(event)
        assert received == [event]

    def test_duplicate_registration_delivers_once(self) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)
        register_sink(received.append)
        emit(new_event("llmcore", "x"))
        assert len(received) == 1

    def test_unregister_stops_delivery(self) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)
        unregister_sink(received.append)
        assert not has_sinks()
        emit(new_event("llmcore", "x"))
        assert received == []

    def test_unregister_unknown_sink_is_noop(self) -> None:
        unregister_sink(lambda e: None)  # must not raise

    def test_faulty_sink_does_not_break_emit_or_other_sinks(self) -> None:
        received: list[UnifiedEvent] = []

        def bad_sink(event: UnifiedEvent) -> None:
            raise ValueError("sink exploded")

        register_sink(bad_sink)
        register_sink(received.append)

        event = new_event("llmcore", "x")
        emit(event)  # must not raise
        assert received == [event]

    def test_sink_can_unregister_itself_during_emit(self) -> None:
        """Mutating the registry from inside a sink must not break delivery."""
        received: list[UnifiedEvent] = []

        def one_shot(event: UnifiedEvent) -> None:
            received.append(event)
            unregister_sink(one_shot)

        register_sink(one_shot)
        emit(new_event("llmcore", "x"))
        emit(new_event("llmcore", "y"))
        assert len(received) == 1
