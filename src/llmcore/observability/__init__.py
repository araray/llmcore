# src/llmcore/observability/__init__.py
"""
Observability Module for LLMCore.

This module provides observability features for tracking LLM and embedding
API usage, including:

- Cost tracking and estimation
- Token usage analytics
- Rate limit monitoring
- Performance metrics

Components:
- CostTracker: Track and estimate API costs
- UsageRecord: Model for usage events
- PRICING_DATA: Current pricing information

Usage:
    from llmcore.observability import CostTracker, UsageRecord

    tracker = CostTracker()
    record = tracker.record(
        provider="openai",
        model="gpt-4o",
        operation="chat",
        input_tokens=1000,
        output_tokens=500
    )
    print(f"Estimated cost: ${record.estimated_cost_usd:.4f}")

    # Get daily summary
    summary = tracker.get_daily_summary()

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 1, Task 1.4
- RAG_ECOSYSTEM_REDESIGN_SPEC.md Section 4.2
"""

from .cost_tracker import (
    CostTracker,
    CostTrackingConfig,
    UsageRecord,
    UsageSummary,
    PRICING_DATA,
    create_cost_tracker,
    get_price_per_million_tokens,
)

__all__ = [
    "CostTracker",
    "CostTrackingConfig",
    "UsageRecord",
    "UsageSummary",
    "PRICING_DATA",
    "create_cost_tracker",
    "get_price_per_million_tokens",
]
