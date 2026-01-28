# tests/observability/test_cost_tracker.py
"""
Tests for the cost tracking implementation.

Tests cover:
- Cost calculation from pricing data
- Usage record creation and persistence
- Summary aggregation (daily, weekly, monthly)
- Provider and model breakdowns
- Session tracking
- Export functionality

Reference: UNIFIED_IMPLEMENTATION_PLAN.md Phase 1, Task 1.4
"""

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from llmcore.observability.cost_tracker import (
    PRICING_DATA,
    CostTracker,
    CostTrackingConfig,
    UsageRecord,
    UsageSummary,
    create_cost_tracker,
    get_price_per_million_tokens,
)


# =============================================================================
# PRICING DATA TESTS
# =============================================================================


class TestPricingData:
    """Tests for pricing data and lookup."""

    def test_openai_pricing_exists(self) -> None:
        """Verify OpenAI pricing data is present."""
        assert "openai" in PRICING_DATA
        assert "gpt-4o" in PRICING_DATA["openai"]
        assert "text-embedding-3-small" in PRICING_DATA["openai"]

    def test_anthropic_pricing_exists(self) -> None:
        """Verify Anthropic pricing data is present."""
        assert "anthropic" in PRICING_DATA
        assert "claude-3-opus" in PRICING_DATA["anthropic"]
        assert "claude-3.5-sonnet" in PRICING_DATA["anthropic"]

    def test_get_price_known_model(self) -> None:
        """Test price lookup for known model."""
        input_price = get_price_per_million_tokens("openai", "gpt-4o", "input")
        output_price = get_price_per_million_tokens("openai", "gpt-4o", "output")

        assert input_price == 2.50
        assert output_price == 10.00

    def test_get_price_embedding_model(self) -> None:
        """Test price lookup for embedding models (output should be 0)."""
        input_price = get_price_per_million_tokens(
            "openai", "text-embedding-3-small", "input"
        )
        output_price = get_price_per_million_tokens(
            "openai", "text-embedding-3-small", "output"
        )

        assert input_price == 0.02
        assert output_price == 0.0

    def test_get_price_ollama_free(self) -> None:
        """Test that Ollama models are free."""
        input_price = get_price_per_million_tokens("ollama", "llama3:8b", "input")
        output_price = get_price_per_million_tokens("ollama", "llama3:8b", "output")

        assert input_price == 0.0
        assert output_price == 0.0

    def test_get_price_unknown_model(self) -> None:
        """Test price lookup for unknown model (should return 0)."""
        price = get_price_per_million_tokens("openai", "unknown-model-xyz", "input")
        assert price == 0.0

    def test_get_price_unknown_provider(self) -> None:
        """Test price lookup for unknown provider (should return 0)."""
        price = get_price_per_million_tokens("unknown-provider", "model", "input")
        assert price == 0.0

    def test_get_price_model_card_override(self) -> None:
        """Test that model card pricing takes precedence."""
        model_card_pricing = {"input": 99.0, "output": 199.0}

        price = get_price_per_million_tokens(
            "openai", "gpt-4o", "input", model_card_pricing
        )

        assert price == 99.0  # Model card price, not default

    def test_get_price_prefix_match(self) -> None:
        """Test price lookup with model prefix matching."""
        # "gpt-4o-2024-11-20" should match "gpt-4o"
        price = get_price_per_million_tokens("openai", "gpt-4o-2024-11-20", "input")
        assert price == 2.50  # Same as gpt-4o


# =============================================================================
# USAGE RECORD TESTS
# =============================================================================


class TestUsageRecord:
    """Tests for UsageRecord model."""

    def test_basic_creation(self) -> None:
        """Test basic record creation."""
        record = UsageRecord(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        assert record.provider == "openai"
        assert record.model == "gpt-4o"
        assert record.total_tokens == 1500
        assert record.id is not None
        assert record.timestamp is not None

    def test_total_tokens_computed(self) -> None:
        """Test that total_tokens is computed if not provided."""
        record = UsageRecord(
            provider="test",
            model="test",
            input_tokens=100,
            output_tokens=50,
        )

        assert record.total_tokens == 150

    def test_optional_fields(self) -> None:
        """Test optional fields."""
        record = UsageRecord(
            provider="test",
            model="test",
            input_tokens=100,
            output_tokens=50,
            session_id="session_123",
            user_id="user_456",
            latency_ms=1500,
            metadata={"key": "value"},
        )

        assert record.session_id == "session_123"
        assert record.user_id == "user_456"
        assert record.latency_ms == 1500
        assert record.metadata == {"key": "value"}


# =============================================================================
# COST TRACKER TESTS
# =============================================================================


class TestCostTracker:
    """Tests for CostTracker."""

    def test_basic_recording(self, tmp_path: Path) -> None:
        """Test basic usage recording."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        # Cost calculation: (1000/1M * 2.50) + (500/1M * 10.00) = 0.0025 + 0.005 = 0.0075
        expected_cost = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)

        assert record.estimated_cost_usd == pytest.approx(expected_cost, abs=1e-8)
        assert record.provider == "openai"
        assert record.model == "gpt-4o"

        tracker.close()

    def test_embedding_cost(self, tmp_path: Path) -> None:
        """Test cost calculation for embeddings."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        record = tracker.record(
            provider="openai",
            model="text-embedding-3-small",
            operation="embedding",
            input_tokens=1000,
            output_tokens=0,  # Embeddings have no output tokens
        )

        # Cost: 1000/1M * 0.02 = 0.00002
        expected_cost = 1000 / 1_000_000 * 0.02

        assert record.estimated_cost_usd == pytest.approx(expected_cost, abs=1e-10)

        tracker.close()

    def test_ollama_free(self, tmp_path: Path) -> None:
        """Test that Ollama models have zero cost."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        record = tracker.record(
            provider="ollama",
            model="llama3:8b",
            operation="chat",
            input_tokens=10000,
            output_tokens=5000,
        )

        assert record.estimated_cost_usd == 0.0

        tracker.close()

    def test_disabled_tracker(self) -> None:
        """Test that disabled tracker returns records but doesn't persist."""
        tracker = CostTracker(enabled=False)

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        # Record is returned but cost is not calculated
        assert record.provider == "openai"
        assert not tracker.enabled

    def test_daily_summary(self, tmp_path: Path) -> None:
        """Test daily summary aggregation."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        # Record some usage
        tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )
        tracker.record(
            provider="anthropic",
            model="claude-3-sonnet",
            operation="chat",
            input_tokens=2000,
            output_tokens=1000,
        )

        summary = tracker.get_daily_summary()

        assert summary.total_calls == 2
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert summary.total_cost_usd > 0
        assert "openai" in summary.by_provider
        assert "anthropic" in summary.by_provider

        tracker.close()

    def test_session_summary(self, tmp_path: Path) -> None:
        """Test session summary (in-memory tracking)."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        summary = tracker.get_session_summary()

        assert summary.total_calls == 1
        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 500

        tracker.close()

    def test_summary_by_provider(self, tmp_path: Path) -> None:
        """Test breakdown by provider."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        # Multiple OpenAI calls
        for _ in range(3):
            tracker.record(
                provider="openai",
                model="gpt-4o",
                operation="chat",
                input_tokens=1000,
                output_tokens=500,
            )

        # One Anthropic call
        tracker.record(
            provider="anthropic",
            model="claude-3-sonnet",
            operation="chat",
            input_tokens=2000,
            output_tokens=1000,
        )

        by_provider = tracker.get_summary_by_provider(days=1)

        assert "openai" in by_provider
        assert "anthropic" in by_provider
        assert by_provider["openai"]["call_count"] == 3
        assert by_provider["anthropic"]["call_count"] == 1

        tracker.close()

    def test_summary_by_model(self, tmp_path: Path) -> None:
        """Test breakdown by model."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        tracker.record(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=500,
        )
        tracker.record(
            provider="openai", model="gpt-4o-mini",
            input_tokens=1000, output_tokens=500,
        )

        by_model = tracker.get_summary_by_model(days=1)

        assert "openai/gpt-4o" in by_model
        assert "openai/gpt-4o-mini" in by_model

        tracker.close()

    def test_with_latency(self, tmp_path: Path) -> None:
        """Test recording with latency."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500,
        )

        assert record.latency_ms == 1500

        summary = tracker.get_daily_summary()
        assert summary.avg_latency_ms is not None

        tracker.close()

    def test_export_to_json(self, tmp_path: Path) -> None:
        """Test JSON export."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        export_path = tmp_path / "export.json"
        count = tracker.export_to_json(str(export_path), days=1)

        assert count == 1
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["provider"] == "openai"
        assert data[0]["model"] == "gpt-4o"

        tracker.close()

    def test_cleanup_old_records(self, tmp_path: Path) -> None:
        """Test cleanup of old records."""
        tracker = CostTracker(
            db_path=str(tmp_path / "costs.db"),
            retention_days=1,
        )

        # Record something
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        # Cleanup (with retention_days=1, recent record shouldn't be deleted)
        deleted = tracker.cleanup_old_records()
        assert deleted == 0

        tracker.close()

    def test_factory_function(self, tmp_path: Path) -> None:
        """Test create_cost_tracker factory function."""
        config = {
            "enabled": True,
            "db_path": str(tmp_path / "factory_costs.db"),
            "retention_days": 30,
            "log_to_console": False,
        }

        tracker = create_cost_tracker(config)

        assert tracker.enabled
        record = tracker.record(
            provider="test",
            model="test",
            input_tokens=100,
            output_tokens=50,
        )

        assert record.provider == "test"

        tracker.close()


# =============================================================================
# COST CALCULATION ACCURACY TESTS
# =============================================================================


class TestCostCalculationAccuracy:
    """Tests for accurate cost calculation."""

    def test_gpt4o_cost_calculation(self, tmp_path: Path) -> None:
        """Test accurate cost calculation for GPT-4o."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        # 1M input tokens + 1M output tokens
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        # Expected: $2.50 + $10.00 = $12.50
        assert record.estimated_cost_usd == pytest.approx(12.50, abs=0.01)

        tracker.close()

    def test_claude_cost_calculation(self, tmp_path: Path) -> None:
        """Test accurate cost calculation for Claude models."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        # 1M input tokens + 1M output tokens
        record = tracker.record(
            provider="anthropic",
            model="claude-3-opus",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        # Expected: $15.00 + $75.00 = $90.00
        assert record.estimated_cost_usd == pytest.approx(90.00, abs=0.01)

        tracker.close()

    def test_typical_conversation_cost(self, tmp_path: Path) -> None:
        """Test cost for a typical conversation."""
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))

        # Typical conversation: ~500 input, ~1000 output
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=1000,
        )

        # Expected: (500/1M * 2.50) + (1000/1M * 10.00) = 0.00125 + 0.01 = 0.01125
        expected = (500 / 1_000_000 * 2.50) + (1000 / 1_000_000 * 10.00)
        assert record.estimated_cost_usd == pytest.approx(expected, abs=1e-8)

        tracker.close()


# =============================================================================
# USAGE SUMMARY TESTS
# =============================================================================


class TestUsageSummary:
    """Tests for UsageSummary model."""

    def test_summary_creation(self) -> None:
        """Test basic summary creation."""
        now = datetime.now(timezone.utc)
        summary = UsageSummary(
            period_start=now - timedelta(days=1),
            period_end=now,
            total_calls=10,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_tokens=15000,
            total_cost_usd=1.50,
        )

        assert summary.total_calls == 10
        assert summary.total_tokens == 15000
        assert summary.total_cost_usd == 1.50

    def test_summary_with_breakdowns(self) -> None:
        """Test summary with provider/model breakdowns."""
        now = datetime.now(timezone.utc)
        summary = UsageSummary(
            period_start=now - timedelta(days=1),
            period_end=now,
            by_provider={"openai": {"count": 5, "cost": 1.0}},
            by_model={"gpt-4o": {"count": 5, "cost": 1.0}},
            by_operation={"chat": {"count": 5, "cost": 1.0}},
        )

        assert "openai" in summary.by_provider
        assert summary.by_provider["openai"]["count"] == 5
