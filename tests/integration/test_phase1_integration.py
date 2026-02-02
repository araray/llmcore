# tests/integration/test_phase1_integration.py
"""
Integration tests for UNIFIED_IMPLEMENTATION_PLAN.md Phase 1 deliverables.

Tests the embedding cache and cost tracker working together as they would
in actual usage scenarios.
"""

import json
from pathlib import Path

import pytest

from llmcore.embedding.cache import (
    EmbeddingCacheConfig,
    create_embedding_cache,
)
from llmcore.observability.cost_tracker import (
    CostTrackingConfig,
    create_cost_tracker,
    get_price_per_million_tokens,
)


class TestCacheWithCostTracking:
    """Test cache and cost tracker integration."""

    def test_cache_reduces_tracked_costs(self, tmp_path: Path) -> None:
        """
        Verify that cache hits reduce the API calls that would be tracked.

        Scenario: An embedding manager would:
        1. Check cache first
        2. If miss, call API and track cost
        3. Store in cache for future hits
        """
        # Setup cache (use dict config for factory function)
        cache_config = {
            "enabled": True,
            "memory_size": 100,
            "disk_enabled": True,
            "disk_path": str(tmp_path / "cache.db"),
        }
        cache = create_embedding_cache(cache_config)

        # Setup cost tracker (use dict config for factory function)
        cost_config = {
            "enabled": True,
            "db_path": str(tmp_path / "costs.db"),
        }
        tracker = create_cost_tracker(cost_config)

        # Simulate embedding requests with caching
        texts = ["Hello world", "Test text", "Hello world"]  # Note: duplicate
        model = "text-embedding-3-small"
        provider = "openai"
        api_calls = 0

        for text in texts:
            # Check cache - API uses (text, model, provider)
            embedding = cache.get(text, model, provider)

            if embedding is None:
                # Simulate API call
                api_calls += 1
                embedding = [0.1] * 256  # Fake embedding

                # Track cost
                tracker.record(
                    provider=provider,
                    model=model,
                    operation="embedding",
                    input_tokens=len(text.split()),
                )

                # Store in cache - API uses (text, model, provider, embedding)
                cache.set(text, model, provider, embedding)

        # Verify: 2 unique texts = 2 API calls (not 3)
        assert api_calls == 2

        # Verify cache stats
        stats = cache.stats
        assert stats["memory"]["hits"] == 1  # "Hello world" hit
        assert stats["memory"]["misses"] == 2  # Two initial misses

    def test_cost_tracking_accuracy_with_realistic_usage(
        self, tmp_path: Path
    ) -> None:
        """Test cost tracking with realistic multi-model usage."""
        tracker = create_cost_tracker({
            "enabled": True,
            "db_path": str(tmp_path / "costs.db"),
        })

        # Simulate a typical session with mixed models
        # 1. GPT-4o for complex reasoning
        tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=2000,
            output_tokens=1000,
        )

        # 2. Claude for code generation (use a known model)
        tracker.record(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            operation="chat",
            input_tokens=1500,
            output_tokens=500,
        )

        # 3. Embeddings for RAG
        tracker.record(
            provider="openai",
            model="text-embedding-3-small",
            operation="embedding",
            input_tokens=5000,
        )

        # 4. Local model (free)
        tracker.record(
            provider="ollama",
            model="llama3.2:latest",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        # Get session summary (no arguments - uses current session)
        summary = tracker.get_session_summary()

        assert summary.total_calls == 4
        assert summary.total_input_tokens == 2000 + 1500 + 5000 + 1000
        assert summary.total_output_tokens == 1000 + 500 + 0 + 500

        # Ollama should be free
        by_provider = tracker.get_summary_by_provider(days=1)
        assert "ollama" in by_provider
        assert by_provider["ollama"]["total_cost_usd"] == 0.0

    def test_cache_persistence_across_instances(self, tmp_path: Path) -> None:
        """Test that cache persists across cache instance restarts."""
        db_path = str(tmp_path / "persistent_cache.db")

        # First instance: populate cache
        config1 = {
            "enabled": True,
            "memory_size": 100,
            "disk_enabled": True,
            "disk_path": db_path,
        }
        cache1 = create_embedding_cache(config1)

        text = "Test text"
        model = "test-model"
        provider = "test-provider"
        embedding = [0.5] * 128

        cache1.set(text, model, provider, embedding)

        # Second instance: verify data is there
        config2 = {
            "enabled": True,
            "memory_size": 100,
            "disk_enabled": True,
            "disk_path": db_path,
        }
        cache2 = create_embedding_cache(config2)

        # Memory cache is empty, but disk should have it
        stats = cache2.stats
        assert stats["memory"]["size"] == 0  # Memory cleared on restart

        # Get should find it on disk
        result = cache2.get(text, model, provider)
        assert result is not None
        assert result == embedding

    def test_cost_export_for_reporting(self, tmp_path: Path) -> None:
        """Test exporting cost data for external reporting."""
        tracker = create_cost_tracker({
            "enabled": True,
            "db_path": str(tmp_path / "costs.db"),
        })

        # Record some usage
        for i in range(5):
            tracker.record(
                provider="openai",
                model="gpt-4o-mini",
                operation="chat",
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
            )

        # Export to JSON
        export_path = tmp_path / "cost_report.json"
        tracker.export_to_json(str(export_path), days=1)

        # Verify export exists
        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)

        # The export is a list of records
        assert isinstance(data, list)
        assert len(data) == 5


class TestConfigurationIntegration:
    """Test configuration loading and defaults."""

    def test_cache_disabled_by_config(self, tmp_path: Path) -> None:
        """Verify cache respects enabled=false config."""
        cache = create_embedding_cache({"enabled": False})

        text = "test text"
        model = "model"
        provider = "provider"
        embedding = [1.0]

        cache.set(text, model, provider, embedding)

        # Should not store anything
        result = cache.get(text, model, provider)
        assert result is None

    def test_tracker_disabled_by_config(self, tmp_path: Path) -> None:
        """Verify tracker respects enabled=false config."""
        tracker = create_cost_tracker({
            "enabled": False,
            "db_path": str(tmp_path / "costs.db"),
        })

        # Record should be no-op
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
        )

        # Should return a zero-cost record
        assert record.estimated_cost_usd == 0.0

    def test_default_paths(self) -> None:
        """Verify default paths are reasonable."""
        cost_config = CostTrackingConfig()
        assert "llmcore" in cost_config.db_path or "cost" in cost_config.db_path

        cache_config = EmbeddingCacheConfig()
        assert cache_config.disk_path is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_handles_empty_embedding(self, tmp_path: Path) -> None:
        """Test cache behavior with empty embedding vectors."""
        cache = create_embedding_cache({
            "enabled": True,
            "disk_enabled": True,
            "disk_path": str(tmp_path / "cache.db"),
        })

        text = "text"
        model = "model"
        provider = "provider"

        cache.set(text, model, provider, [])

        result = cache.get(text, model, provider)
        assert result == []

    def test_tracker_handles_zero_tokens(self, tmp_path: Path) -> None:
        """Test tracker with zero token counts."""
        tracker = create_cost_tracker({
            "enabled": True,
            "db_path": str(tmp_path / "costs.db"),
        })

        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
        )

        assert record.estimated_cost_usd == 0.0
        assert record.total_tokens == 0

    def test_pricing_lookup_unknown_model(self) -> None:
        """Test pricing lookup for unknown models."""
        price = get_price_per_million_tokens(
            "unknown_provider",
            "unknown_model",
            token_type="input",
        )

        # Should return 0.0 for unknown models
        assert price == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
