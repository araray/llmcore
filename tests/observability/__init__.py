# tests/observability/__init__.py
"""
Tests for the LLMCore observability module.

This package contains tests for:
- events.py: Structured event logging system
- metrics.py: Performance metrics collection
- analytics.py: Cost analytics and forecasting
- cost_tracker.py: API usage cost tracking (existing)

Test Files:
- test_events.py: Event logging, categories, buffer, replay
- test_metrics.py: Counters, gauges, histograms, registry
- test_analytics.py: Cost analysis, trends, forecasts, budget

Running Tests:
    # All observability tests
    pytest tests/observability/ -v

    # Individual modules
    pytest tests/observability/test_events.py -v
    pytest tests/observability/test_metrics.py -v
    pytest tests/observability/test_analytics.py -v

    # With coverage
    pytest tests/observability/ --cov=llmcore.observability --cov-report=html
"""
