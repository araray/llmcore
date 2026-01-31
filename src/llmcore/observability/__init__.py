# src/llmcore/observability/__init__.py
"""
Observability Module for LLMCore.

This module provides comprehensive observability features for tracking LLM
and embedding API usage, performance metrics, cost analytics, and structured
event logging.

Components:
    Cost Tracking (cost_tracker.py):
        - CostTracker: Track and estimate API costs
        - UsageRecord: Model for usage events
        - UsageSummary: Aggregated usage statistics
        - PRICING_DATA: Current pricing information

    Performance Metrics (metrics.py):
        - MetricsRegistry: Central registry for all metrics
        - Counter: Monotonically increasing metric
        - Gauge: Point-in-time value
        - Histogram: Distribution with percentiles
        - Timer: Timing operations

    Cost Analytics (analytics.py):
        - CostAnalyzer: Advanced cost analysis
        - Period comparisons (week-over-week, month-over-month)
        - Trend analysis and forecasting
        - Budget tracking and alerts

    Event Logging (events.py):
        - ObservabilityLogger: Structured event logging
        - EventCategory: Event categorization
        - ExecutionReplayer: Debug execution replay

Usage:
    >>> from llmcore.observability import (
    ...     CostTracker,
    ...     MetricsRegistry,
    ...     ObservabilityLogger,
    ...     CostAnalyzer,
    ... )
    >>>
    >>> # Track costs
    >>> tracker = CostTracker()
    >>> record = tracker.record(
    ...     provider="openai",
    ...     model="gpt-4o",
    ...     operation="chat",
    ...     input_tokens=1000,
    ...     output_tokens=500
    ... )
    >>>
    >>> # Collect metrics
    >>> registry = MetricsRegistry.get_default()
    >>> latency = registry.histogram("llm_latency_ms", "Request latency")
    >>> latency.observe(150.5)
    >>>
    >>> # Log events
    >>> logger = ObservabilityLogger()
    >>> logger.log_event(
    ...     category="cognitive",
    ...     event_type="phase_completed",
    ...     data={"phase": "THINK", "duration_ms": 1500}
    ... )
    >>>
    >>> # Analyze costs
    >>> analyzer = CostAnalyzer(tracker)
    >>> comparison = analyzer.compare_periods("week")

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 9
    - llmcore_spec_v2.md Section 13 (Observability System)
    - RAG_ECOSYSTEM_REDESIGN_SPEC.md Section 4.2

Version: 0.28.1 (Phase 9)
"""

# =============================================================================
# COST TRACKING
# =============================================================================

# =============================================================================
# COST ANALYTICS
# =============================================================================
from .analytics import (
    # Reports/Summaries
    AnalyticsSummary,
    # Enums
    AnomalyType,
    # Results
    BudgetAnalysis,
    BudgetStatus,
    # Analyzer
    CostAnalyzer,
    CostAnomaly,
    CostForecast,
    Period,
    PeriodComparison,
    TrendAnalysis,
    TrendDirection,
    UsagePattern,
    # Factory
    create_cost_analyzer,
)
from .cost_tracker import (
    PRICING_DATA,
    CostTracker,
    CostTrackingConfig,
    UsageRecord,
    UsageSummary,
    create_cost_tracker,
    get_price_per_million_tokens,
)

# =============================================================================
# EVENT LOGGING
# =============================================================================
from .events import (
    # Data models
    Event,
    # Core classes
    EventBuffer,
    EventBufferConfig,
    # Enums
    EventCategory,
    EventFileWriter,
    EventRotationConfig,
    ExecutionReplayer,
    # Replay
    ExecutionTrace,
    ObservabilityConfig,
    ObservabilityLogger,
    RotationStrategy,
    Severity,
    # Factories
    create_observability_logger,
    load_events_from_file,
)

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
from .metrics import (
    # Metric types
    Counter,
    Gauge,
    Histogram,
    # Additional types
    HistogramBucket,
    # Collectors
    LLMMetricsCollector,
    MetricLabels,
    # Models
    MetricSnapshot,
    # Registry
    MetricsRegistry,
    MetricsSummary,
    # Enums
    MetricType,
    MetricUnit,
    RateCounter,
    SystemMetricsCollector,
    Timer,
    # Functions
    create_metrics_registry,
    get_metrics_registry,
    get_metrics_summary,
    record_llm_call,
    timer,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "0.28.1"


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Cost Tracking
    "CostTracker",
    "CostTrackingConfig",
    "UsageRecord",
    "UsageSummary",
    "PRICING_DATA",
    "create_cost_tracker",
    "get_price_per_million_tokens",
    # Performance Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "HistogramBucket",
    "RateCounter",
    "Timer",
    "timer",
    "MetricType",
    "MetricUnit",
    "MetricLabels",
    "MetricSnapshot",
    "MetricsSummary",
    "LLMMetricsCollector",
    "SystemMetricsCollector",
    "create_metrics_registry",
    "get_metrics_registry",
    "get_metrics_summary",
    "record_llm_call",
    # Cost Analytics
    "CostAnalyzer",
    "Period",
    "TrendDirection",
    "BudgetStatus",
    "AnomalyType",
    "PeriodComparison",
    "TrendAnalysis",
    "CostForecast",
    "BudgetAnalysis",
    "CostAnomaly",
    "UsagePattern",
    "AnalyticsSummary",
    "create_cost_analyzer",
    # Event Logging
    "EventCategory",
    "Severity",
    "RotationStrategy",
    "Event",
    "EventRotationConfig",
    "EventBufferConfig",
    "ObservabilityConfig",
    "EventBuffer",
    "EventFileWriter",
    "ObservabilityLogger",
    "ExecutionTrace",
    "ExecutionReplayer",
    "create_observability_logger",
    "load_events_from_file",
]
