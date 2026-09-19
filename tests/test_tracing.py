# tests/test_tracing.py
"""Tests for llmcore.tracing helpers.

Focuses on the graceful-degradation helpers that must never raise, in
particular the exception-recording path that previously referenced an
unimported ``trace`` module (F821 / silent NameError).
"""

from __future__ import annotations

import pytest

from llmcore.tracing import add_span_attributes, create_span, record_span_exception


class FakeSpan:
    """Minimal stand-in for an OpenTelemetry span."""

    def __init__(self) -> None:
        self.recorded_exceptions: list[BaseException] = []
        self.statuses: list[object] = []
        self.attributes: dict[str, str] = {}

    def record_exception(self, exception: BaseException) -> None:
        self.recorded_exceptions.append(exception)

    def set_status(self, status: object) -> None:
        self.statuses.append(status)

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value


class ExplodingSpan:
    """Span whose methods always raise, to exercise the swallow path."""

    def record_exception(self, exception: BaseException) -> None:
        raise RuntimeError("boom")

    def set_status(self, status: object) -> None:  # pragma: no cover - never reached
        raise RuntimeError("boom")


class TestRecordSpanException:
    def test_records_exception_and_sets_error_status(self) -> None:
        pytest.importorskip("opentelemetry")
        from opentelemetry import trace

        span = FakeSpan()
        error = ValueError("bad input")

        record_span_exception(span, error)

        assert span.recorded_exceptions == [error]
        assert len(span.statuses) == 1
        status = span.statuses[0]
        assert isinstance(status, trace.Status)
        assert status.status_code is trace.StatusCode.ERROR
        assert "bad input" in (status.description or "")

    def test_none_span_is_noop(self) -> None:
        # Must not raise.
        record_span_exception(None, ValueError("ignored"))

    def test_span_errors_are_swallowed(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("DEBUG", logger="llmcore.tracing"):
            record_span_exception(ExplodingSpan(), ValueError("ignored"))

        assert any("Failed to record span exception" in rec.message for rec in caplog.records)


class TestSpanHelpers:
    def test_create_span_without_tracer_returns_noop_context(self) -> None:
        with create_span(None, "unit.test") as span:
            assert span is None

    def test_add_span_attributes_skips_none_values(self) -> None:
        span = FakeSpan()
        add_span_attributes(span, {"a": 1, "b": None})
        assert span.attributes == {"a": "1"}

    def test_add_span_attributes_none_span_is_noop(self) -> None:
        add_span_attributes(None, {"a": 1})
