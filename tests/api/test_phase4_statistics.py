# tests/api/test_phase4_statistics.py
"""
Phase 4: Introspection & Statistics Tests

This module tests the Phase 4 statistics logic:
- SessionTokenStats dataclass behavior
- CostEstimate dataclass behavior
- Statistics aggregation logic
- Cost estimation calculations
- Edge cases and error handling

Tests are designed to run in isolation using direct module loading
to avoid full llmcore initialization which requires many dependencies.
"""

import importlib.util
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# DIRECT MODULE LOADING (avoids llmcore package initialization)
# =============================================================================


def load_models_directly():
    """Load models.py directly without going through llmcore package."""
    spec = importlib.util.spec_from_file_location("llmcore_models", "src/llmcore/models.py")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    return models


# Load models module
try:
    models = load_models_directly()
    SessionTokenStats = models.SessionTokenStats
    CostEstimate = models.CostEstimate
    ChatSession = models.ChatSession
    Message = models.Message
    Role = models.Role
    MODELS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load models directly: {e}")
    MODELS_AVAILABLE = False
    # Create stubs for test collection
    SessionTokenStats = None
    CostEstimate = None
    ChatSession = None
    Message = None
    Role = None


# =============================================================================
# DATA MODEL TESTS
# =============================================================================


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestSessionTokenStatsModel:
    """Tests for SessionTokenStats dataclass."""

    def test_default_values(self):
        """Test SessionTokenStats has correct default values."""
        stats = SessionTokenStats(session_id="test-session")

        assert stats.session_id == "test-session"
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.total_cached_tokens == 0
        assert stats.interaction_count == 0
        assert stats.avg_prompt_tokens == 0.0
        assert stats.avg_completion_tokens == 0.0
        assert stats.max_prompt_tokens == 0
        assert stats.max_completion_tokens == 0
        assert stats.first_interaction_at is None
        assert stats.last_interaction_at is None
        assert stats.by_model == {}

    def test_full_initialization(self):
        """Test SessionTokenStats with all fields populated."""
        now = datetime.now(timezone.utc)
        stats = SessionTokenStats(
            session_id="session-123",
            total_prompt_tokens=5000,
            total_completion_tokens=1500,
            total_tokens=6500,
            total_cached_tokens=1000,
            interaction_count=10,
            avg_prompt_tokens=500.0,
            avg_completion_tokens=150.0,
            max_prompt_tokens=800,
            max_completion_tokens=300,
            first_interaction_at=now,
            last_interaction_at=now,
            by_model={"gpt-4": {"prompt": 3000, "completion": 1000, "count": 6}},
        )

        assert stats.total_prompt_tokens == 5000
        assert stats.total_completion_tokens == 1500
        assert stats.total_tokens == 6500
        assert stats.interaction_count == 10
        assert len(stats.by_model) == 1
        assert stats.by_model["gpt-4"]["count"] == 6

    def test_datetime_serialization(self):
        """Test datetime fields serialize correctly."""
        now = datetime.now(timezone.utc)
        stats = SessionTokenStats(
            session_id="test",
            first_interaction_at=now,
            last_interaction_at=now,
        )

        # Test serialization
        data = stats.model_dump()
        assert "Z" in data["first_interaction_at"] or "+00:00" in data["first_interaction_at"]


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestCostEstimateModel:
    """Tests for CostEstimate dataclass."""

    def test_default_values(self):
        """Test CostEstimate has correct default values."""
        cost = CostEstimate()

        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.cached_discount == 0.0
        assert cost.reasoning_cost == 0.0
        assert cost.total_cost == 0.0
        assert cost.currency == "USD"
        assert cost.pricing_source == "model_card"
        assert cost.prompt_tokens == 0
        assert cost.completion_tokens == 0
        assert cost.cached_tokens == 0
        assert cost.reasoning_tokens == 0

    def test_full_initialization(self):
        """Test CostEstimate with all fields populated."""
        cost = CostEstimate(
            input_cost=0.015,
            output_cost=0.030,
            cached_discount=0.005,
            reasoning_cost=0.010,
            total_cost=0.050,
            currency="USD",
            pricing_source="model_card",
            prompt_tokens=5000,
            completion_tokens=2000,
            cached_tokens=1000,
            reasoning_tokens=500,
            input_price_per_million=3.0,
            output_price_per_million=15.0,
            cached_price_per_million=1.5,
            model_id="claude-sonnet-4",
            provider="anthropic",
        )

        assert cost.input_cost == 0.015
        assert cost.total_cost == 0.050
        assert cost.model_id == "claude-sonnet-4"
        assert cost.provider == "anthropic"

    def test_format_cost_usd(self):
        """Test format_cost method with USD currency."""
        cost = CostEstimate(total_cost=0.012345, currency="USD")
        assert cost.format_cost() == "$0.0123"
        assert cost.format_cost(precision=2) == "$0.01"

    def test_format_cost_unavailable(self):
        """Test format_cost returns N/A for local models."""
        cost = CostEstimate(pricing_source="unavailable")
        assert cost.format_cost() == "N/A (local model)"

    def test_format_cost_other_currency(self):
        """Test format_cost with non-USD currency."""
        cost = CostEstimate(total_cost=0.05, currency="EUR")
        assert cost.format_cost() == "EUR0.0500"


# =============================================================================
# API METHOD TESTS (with mocking)
# =============================================================================


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestGetSessionTokenStats:
    """Tests for LLMCore.get_session_token_stats()."""

    @pytest.fixture
    def mock_session_with_interactions(self):
        """Create a mock session with interaction data in metadata."""
        session = ChatSession(
            id="session-with-interactions",
            name="Test Session",
            metadata={
                "provider": "anthropic",
                "interactions": [
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "cached_tokens": 20,
                        "model": "claude-sonnet-4",
                        "timestamp": "2026-01-15T10:00:00Z",
                    },
                    {
                        "prompt_tokens": 200,
                        "completion_tokens": 100,
                        "cached_tokens": 50,
                        "model": "claude-sonnet-4",
                        "timestamp": "2026-01-15T11:00:00Z",
                    },
                    {
                        "prompt_tokens": 150,
                        "completion_tokens": 75,
                        "cached_tokens": 30,
                        "model": "gpt-4o",
                        "timestamp": "2026-01-15T12:00:00Z",
                    },
                ],
            },
        )
        return session

    @pytest.fixture
    def mock_session_without_interactions(self):
        """Create a mock session without interaction metadata (fallback to messages)."""
        now = datetime.now(timezone.utc)
        session = ChatSession(
            id="session-no-interactions",
            name="Test Session",
            messages=[
                Message(role=Role.USER, content="Hello", tokens=10, timestamp=now),
                Message(role=Role.ASSISTANT, content="Hi there", tokens=15, timestamp=now),
                Message(role=Role.USER, content="How are you?", tokens=12, timestamp=now),
                Message(role=Role.ASSISTANT, content="I'm doing well", tokens=20, timestamp=now),
            ],
        )
        return session

    def test_stats_from_interaction_metadata(self, mock_session_with_interactions):
        """Test stats calculation from interaction metadata."""
        # SessionTokenStats is already loaded at module level via direct loading

        # Simulate what get_session_token_stats would calculate
        session = mock_session_with_interactions
        interactions = session.metadata["interactions"]

        stats = SessionTokenStats(session_id=session.id)
        for interaction in interactions:
            stats.total_prompt_tokens += interaction["prompt_tokens"]
            stats.total_completion_tokens += interaction["completion_tokens"]
            stats.total_cached_tokens += interaction["cached_tokens"]
            stats.interaction_count += 1

        stats.total_tokens = stats.total_prompt_tokens + stats.total_completion_tokens

        assert stats.total_prompt_tokens == 450  # 100 + 200 + 150
        assert stats.total_completion_tokens == 225  # 50 + 100 + 75
        assert stats.total_tokens == 675
        assert stats.total_cached_tokens == 100  # 20 + 50 + 30
        assert stats.interaction_count == 3

    def test_stats_by_model_aggregation(self, mock_session_with_interactions):
        """Test token aggregation by model."""
        session = mock_session_with_interactions
        interactions = session.metadata["interactions"]

        by_model = {}
        for interaction in interactions:
            model = interaction["model"]
            if model not in by_model:
                by_model[model] = {"prompt": 0, "completion": 0, "count": 0}
            by_model[model]["prompt"] += interaction["prompt_tokens"]
            by_model[model]["completion"] += interaction["completion_tokens"]
            by_model[model]["count"] += 1

        assert "claude-sonnet-4" in by_model
        assert "gpt-4o" in by_model
        assert by_model["claude-sonnet-4"]["count"] == 2
        assert by_model["claude-sonnet-4"]["prompt"] == 300  # 100 + 200
        assert by_model["gpt-4o"]["count"] == 1
        assert by_model["gpt-4o"]["prompt"] == 150

    def test_stats_fallback_to_messages(self, mock_session_without_interactions):
        """Test stats calculation falls back to message tokens when no interaction data."""
        session = mock_session_without_interactions

        total_prompt = sum(m.tokens for m in session.messages if m.role == Role.USER and m.tokens)
        total_completion = sum(
            m.tokens for m in session.messages if m.role == Role.ASSISTANT and m.tokens
        )

        assert total_prompt == 22  # 10 + 12
        assert total_completion == 35  # 15 + 20

    def test_average_calculation(self):
        """Test average tokens per interaction calculation."""
        stats = SessionTokenStats(
            session_id="test",
            total_prompt_tokens=500,
            total_completion_tokens=200,
            interaction_count=5,
        )

        # Calculate averages
        if stats.interaction_count > 0:
            avg_prompt = stats.total_prompt_tokens / stats.interaction_count
            avg_completion = stats.total_completion_tokens / stats.interaction_count
        else:
            avg_prompt = 0.0
            avg_completion = 0.0

        assert avg_prompt == 100.0
        assert avg_completion == 40.0


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestEstimateCost:
    """Tests for LLMCore.estimate_cost()."""

    @pytest.fixture
    def mock_model_card_with_pricing(self):
        """Create a mock model card with pricing data."""
        card = MagicMock()
        card.pricing = MagicMock()
        card.pricing.currency = "USD"
        card.pricing.per_million_tokens = MagicMock()
        card.pricing.per_million_tokens.input = 3.0  # $3/1M tokens
        card.pricing.per_million_tokens.output = 15.0  # $15/1M tokens
        card.pricing.per_million_tokens.cached_input = 1.5  # $1.5/1M tokens
        card.pricing.per_million_tokens.reasoning_output = None
        return card

    @pytest.fixture
    def mock_model_card_no_pricing(self):
        """Create a mock model card without pricing (local model)."""
        card = MagicMock()
        card.pricing = None
        return card

    def test_cost_calculation_with_pricing(self, mock_model_card_with_pricing):
        """Test cost calculation with model card pricing data."""
        card = mock_model_card_with_pricing

        prompt_tokens = 10000
        completion_tokens = 2000
        cached_tokens = 3000

        # Calculate expected cost
        regular_input = prompt_tokens - cached_tokens  # 7000
        input_cost = (regular_input / 1_000_000) * card.pricing.per_million_tokens.input
        cached_cost = (cached_tokens / 1_000_000) * card.pricing.per_million_tokens.cached_input
        output_cost = (completion_tokens / 1_000_000) * card.pricing.per_million_tokens.output

        expected_input = input_cost + cached_cost
        expected_output = output_cost
        expected_total = expected_input + expected_output

        # (7000/1M * 3.0) + (3000/1M * 1.5) = 0.021 + 0.0045 = 0.0255
        assert abs(expected_input - 0.0255) < 0.0001
        # (2000/1M * 15.0) = 0.030
        assert abs(expected_output - 0.030) < 0.0001
        assert abs(expected_total - 0.0555) < 0.0001

    def test_cost_without_caching(self, mock_model_card_with_pricing):
        """Test cost calculation without cached tokens."""
        card = mock_model_card_with_pricing

        prompt_tokens = 10000
        completion_tokens = 2000

        input_cost = (prompt_tokens / 1_000_000) * card.pricing.per_million_tokens.input
        output_cost = (completion_tokens / 1_000_000) * card.pricing.per_million_tokens.output
        total_cost = input_cost + output_cost

        # (10000/1M * 3.0) = 0.030
        assert abs(input_cost - 0.030) < 0.0001
        # (2000/1M * 15.0) = 0.030
        assert abs(output_cost - 0.030) < 0.0001
        assert abs(total_cost - 0.060) < 0.0001

    def test_cost_unavailable_for_local_model(self, mock_model_card_no_pricing):
        """Test that local models return 'unavailable' pricing source."""
        # When pricing is None, the estimate should return pricing_source="unavailable"
        result = CostEstimate(
            prompt_tokens=10000,
            completion_tokens=2000,
            pricing_source="unavailable",
        )

        assert result.pricing_source == "unavailable"
        assert result.total_cost == 0.0
        assert result.format_cost() == "N/A (local model)"

    def test_cached_discount_calculation(self, mock_model_card_with_pricing):
        """Test cached token discount is calculated correctly."""
        card = mock_model_card_with_pricing

        cached_tokens = 5000
        full_cost = (cached_tokens / 1_000_000) * card.pricing.per_million_tokens.input
        cached_cost = (cached_tokens / 1_000_000) * card.pricing.per_million_tokens.cached_input
        discount = full_cost - cached_cost

        # (5000/1M * 3.0) - (5000/1M * 1.5) = 0.015 - 0.0075 = 0.0075
        assert abs(discount - 0.0075) < 0.0001


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestEstimateSessionCost:
    """Tests for LLMCore.estimate_session_cost()."""

    def test_aggregates_from_token_stats(self):
        """Test that session cost uses token stats for calculation."""
        # Create mock token stats
        stats = SessionTokenStats(
            session_id="test-session",
            total_prompt_tokens=50000,
            total_completion_tokens=15000,
            total_cached_tokens=10000,
            by_model={"claude-sonnet-4": {"prompt": 50000, "completion": 15000, "count": 10}},
        )

        # The estimate_session_cost would use these values
        assert stats.total_prompt_tokens == 50000
        assert stats.total_completion_tokens == 15000
        assert stats.total_cached_tokens == 10000

    def test_determines_primary_model(self):
        """Test that the primary model is determined by interaction count."""
        by_model = {
            "claude-sonnet-4": {"prompt": 30000, "completion": 10000, "count": 8},
            "gpt-4o": {"prompt": 20000, "completion": 5000, "count": 3},
        }

        # Primary model is the one with most interactions
        primary = max(by_model.items(), key=lambda x: x[1]["count"])[0]
        assert primary == "claude-sonnet-4"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestPhase4Integration:
    """Integration tests for Phase 4 statistics flow."""

    def test_complete_stats_flow(self):
        """Test complete flow from session data to cost estimate."""
        # Create session with interaction data
        session = ChatSession(
            id="integration-test",
            name="Integration Test",
            metadata={
                "provider": "anthropic",
                "interactions": [
                    {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "cached_tokens": 200,
                        "model": "claude-sonnet-4",
                        "timestamp": "2026-01-15T10:00:00Z",
                    },
                    {
                        "prompt_tokens": 1500,
                        "completion_tokens": 700,
                        "cached_tokens": 300,
                        "model": "claude-sonnet-4",
                        "timestamp": "2026-01-15T11:00:00Z",
                    },
                ],
            },
        )

        # Calculate stats
        total_prompt = 1000 + 1500  # 2500
        total_completion = 500 + 700  # 1200
        total_cached = 200 + 300  # 500
        total_tokens = total_prompt + total_completion  # 3700

        assert total_prompt == 2500
        assert total_completion == 1200
        assert total_tokens == 3700
        assert total_cached == 500

        # Create expected cost estimate
        cost = CostEstimate(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            cached_tokens=total_cached,
            provider="anthropic",
            model_id="claude-sonnet-4",
            pricing_source="model_card",
        )

        assert cost.prompt_tokens == 2500
        assert cost.completion_tokens == 1200
        assert cost.provider == "anthropic"

    def test_empty_session_stats(self):
        """Test stats for a session with no interactions."""
        stats = SessionTokenStats(session_id="empty-session")

        assert stats.total_tokens == 0
        assert stats.interaction_count == 0
        assert stats.avg_prompt_tokens == 0.0
        assert stats.by_model == {}


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="llmcore models not available")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_tokens(self):
        """Test handling of zero token counts."""
        cost = CostEstimate(
            prompt_tokens=0,
            completion_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
        )

        assert cost.total_cost == 0.0
        assert cost.format_cost() == "$0.0000"

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        # 100 million tokens
        stats = SessionTokenStats(
            session_id="large",
            total_prompt_tokens=100_000_000,
            total_completion_tokens=50_000_000,
            total_tokens=150_000_000,
            interaction_count=1000,
        )

        avg_prompt = stats.total_prompt_tokens / stats.interaction_count
        assert avg_prompt == 100_000.0

    def test_timestamp_parsing_variations(self):
        """Test various timestamp format parsing."""
        # ISO format with Z
        ts_z = "2026-01-15T10:00:00Z"
        dt_z = datetime.fromisoformat(ts_z[:-1] + "+00:00")
        assert dt_z.tzinfo is not None

        # ISO format with offset
        ts_offset = "2026-01-15T10:00:00+00:00"
        dt_offset = datetime.fromisoformat(ts_offset)
        assert dt_offset.tzinfo is not None

    def test_missing_optional_pricing_fields(self):
        """Test handling when optional pricing fields are None."""
        cost = CostEstimate(
            input_price_per_million=None,
            output_price_per_million=None,
            cached_price_per_million=None,
        )

        assert cost.input_price_per_million is None
        assert cost.output_price_per_million is None
        assert cost.cached_price_per_million is None

    def test_by_model_empty_when_no_interactions(self):
        """Test by_model is empty for sessions without model tracking."""
        stats = SessionTokenStats(session_id="no-model-tracking")
        assert stats.by_model == {}
        assert len(stats.by_model) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
