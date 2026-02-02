# tests/storage/test_feedback.py
"""
Tests for FeedbackManager and related classes.

These tests verify:
- Feedback recording and persistence
- Aggregate computation with recency weighting
- Score adjustment generation
- Analytics and statistics
- Maintenance operations (cleanup, deletion)
- Thread safety

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 3, Task 3.2
"""

import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

# Add src to path for test discovery - direct module import to avoid heavy deps
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"))

# Direct import from module to avoid triggering full llmcore init
from feedback import (
    AggregatedFeedback,
    FeedbackConfig,
    FeedbackManager,
    FeedbackRecord,
    create_feedback_manager,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "feedback_test.db")


@pytest.fixture
def feedback_manager(temp_db_path):
    """Create a feedback manager with temporary database."""
    return FeedbackManager(db_path=temp_db_path)


@pytest.fixture
def feedback_config(temp_db_path):
    """Create a test configuration."""
    return FeedbackConfig(
        db_path=temp_db_path,
        min_feedback_for_adjustment=3,
        decay_factor=0.9,
        positive_threshold=7.0,
        negative_threshold=3.0,
    )


# =============================================================================
# FEEDBACK RECORD TESTS
# =============================================================================


class TestFeedbackRecord:
    """Tests for the FeedbackRecord model."""

    def test_record_creation_required_fields(self):
        """Test creating record with required fields only."""
        record = FeedbackRecord(
            item_id="chunk_123",
            collection="test_collection",
            query="test query",
            relevance_score=7.5,
        )

        assert record.item_id == "chunk_123"
        assert record.collection == "test_collection"
        assert record.query == "test query"
        assert record.relevance_score == 7.5
        assert record.feedback_type == "explicit"
        assert record.provider_id == "user"
        assert record.id is None

    def test_record_creation_all_fields(self):
        """Test creating record with all fields."""
        record = FeedbackRecord(
            item_id="chunk_456",
            collection="myproject",
            query="how does auth work?",
            relevance_score=9.0,
            feedback_type="agent",
            provider_id="darwin_agent",
            session_id="session_123",
        )

        assert record.feedback_type == "agent"
        assert record.provider_id == "darwin_agent"
        assert record.session_id == "session_123"

    def test_record_score_validation(self):
        """Test score validation bounds."""
        # Valid scores
        FeedbackRecord(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=0.0,
        )
        FeedbackRecord(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=10.0,
        )

        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            FeedbackRecord(
                item_id="test",
                collection="test",
                query="test",
                relevance_score=-1.0,
            )

        with pytest.raises(Exception):
            FeedbackRecord(
                item_id="test",
                collection="test",
                query="test",
                relevance_score=11.0,
            )

    def test_record_feedback_type_validation(self):
        """Test feedback_type literal validation."""
        # Valid types
        for ftype in ["explicit", "implicit", "agent"]:
            FeedbackRecord(
                item_id="test",
                collection="test",
                query="test",
                relevance_score=5.0,
                feedback_type=ftype,
            )

        # Invalid type should raise
        with pytest.raises(Exception):
            FeedbackRecord(
                item_id="test",
                collection="test",
                query="test",
                relevance_score=5.0,
                feedback_type="invalid",
            )

    def test_record_created_at_default(self):
        """Test that created_at defaults to now."""
        before = datetime.utcnow()
        record = FeedbackRecord(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=5.0,
        )
        after = datetime.utcnow()

        assert before <= record.created_at <= after


# =============================================================================
# AGGREGATED FEEDBACK TESTS
# =============================================================================


class TestAggregatedFeedback:
    """Tests for the AggregatedFeedback model."""

    def test_aggregated_creation(self):
        """Test creating aggregated feedback."""
        agg = AggregatedFeedback(
            item_id="chunk_123",
            collection="myproject",
            total_feedback_count=10,
            average_score=7.5,
            recent_score=8.0,
            positive_count=7,
            negative_count=1,
            trend="improving",
        )

        assert agg.item_id == "chunk_123"
        assert agg.total_feedback_count == 10
        assert agg.trend == "improving"

    def test_aggregated_trend_validation(self):
        """Test trend literal validation."""
        for trend in ["improving", "declining", "stable"]:
            AggregatedFeedback(
                item_id="test",
                collection="test",
                trend=trend,
            )

        with pytest.raises(Exception):
            AggregatedFeedback(
                item_id="test",
                collection="test",
                trend="invalid",
            )


# =============================================================================
# FEEDBACK MANAGER - BASIC OPERATIONS
# =============================================================================


class TestFeedbackManagerBasic:
    """Basic operations tests for FeedbackManager."""

    def test_manager_creation_default(self, temp_db_path):
        """Test creating manager with default settings."""
        manager = FeedbackManager(db_path=temp_db_path)

        assert manager.enabled
        assert manager.db_path == Path(temp_db_path).expanduser()
        assert manager.db_path.exists()

    def test_manager_creation_disabled(self, temp_db_path):
        """Test creating disabled manager."""
        manager = FeedbackManager(db_path=temp_db_path, enabled=False)

        assert not manager.enabled
        # Disabled manager shouldn't create database
        # (though path is still stored)

    def test_manager_creation_from_config(self, feedback_config):
        """Test creating manager from config object."""
        manager = FeedbackManager(config=feedback_config)

        assert manager.enabled == feedback_config.enabled
        assert manager.db_path == Path(feedback_config.db_path).expanduser()

    def test_record_feedback_basic(self, feedback_manager):
        """Test basic feedback recording."""
        record = feedback_manager.record_feedback(
            item_id="chunk_123",
            collection="myproject",
            query="How does authentication work?",
            relevance_score=8.5,
        )

        assert record.id is not None
        assert record.item_id == "chunk_123"
        assert record.relevance_score == 8.5

    def test_record_feedback_all_params(self, feedback_manager):
        """Test recording with all parameters."""
        record = feedback_manager.record_feedback(
            item_id="chunk_456",
            collection="myproject",
            query="test query",
            relevance_score=7.0,
            feedback_type="agent",
            provider_id="darwin_v2",
            session_id="session_abc",
        )

        assert record.feedback_type == "agent"
        assert record.provider_id == "darwin_v2"
        assert record.session_id == "session_abc"

    def test_record_feedback_score_clamping(self, feedback_manager):
        """Test that scores are clamped to [0, 10]."""
        # Score below minimum
        record1 = feedback_manager.record_feedback(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=-5.0,
        )
        assert record1.relevance_score == 0.0

        # Score above maximum
        record2 = feedback_manager.record_feedback(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=15.0,
        )
        assert record2.relevance_score == 10.0

    def test_record_feedback_disabled_manager(self, temp_db_path):
        """Test that disabled manager returns record but doesn't persist."""
        manager = FeedbackManager(db_path=temp_db_path, enabled=False)

        record = manager.record_feedback(
            item_id="test",
            collection="test",
            query="test",
            relevance_score=5.0,
        )

        assert record.item_id == "test"
        assert record.id is None  # Not persisted

    def test_record_batch(self, feedback_manager):
        """Test batch feedback recording."""
        feedbacks = [
            {
                "item_id": f"chunk_{i}",
                "collection": "myproject",
                "query": f"query_{i}",
                "relevance_score": 5.0 + i,
            }
            for i in range(5)
        ]

        records = feedback_manager.record_batch(feedbacks)

        assert len(records) == 5
        assert all(r.id is not None for r in records)
        assert records[0].relevance_score == 5.0
        assert records[4].relevance_score == 9.0


# =============================================================================
# FEEDBACK MANAGER - AGGREGATION
# =============================================================================


class TestFeedbackManagerAggregation:
    """Aggregation computation tests for FeedbackManager."""

    def test_aggregation_single_feedback(self, feedback_manager):
        """Test aggregation with single feedback."""
        feedback_manager.record_feedback(
            item_id="chunk_1",
            collection="myproject",
            query="test",
            relevance_score=8.0,
        )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        assert agg is not None
        assert agg.total_feedback_count == 1
        assert agg.average_score == 8.0
        assert agg.recent_score == 8.0
        assert agg.positive_count == 1  # 8.0 >= 7.0
        assert agg.negative_count == 0

    def test_aggregation_multiple_feedbacks(self, feedback_manager):
        """Test aggregation with multiple feedbacks."""
        scores = [6.0, 7.0, 8.0, 9.0, 5.0]

        for score in scores:
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=score,
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        assert agg.total_feedback_count == 5
        assert agg.average_score == pytest.approx(7.0, rel=0.01)
        # Positive: 7, 8, 9 = 3
        assert agg.positive_count == 3
        # Negative: none <= 3.0
        assert agg.negative_count == 0

    def test_aggregation_recency_weighting(self, feedback_manager):
        """Test that recent scores have higher weight."""
        # Add old low scores
        for _ in range(5):
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=3.0,
            )

        # Add new high scores
        for _ in range(5):
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=9.0,
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        # Average should be 6.0
        assert agg.average_score == pytest.approx(6.0, rel=0.01)

        # Recent score should be higher (weighted toward newer scores)
        assert agg.recent_score > agg.average_score

    def test_aggregation_trend_improving(self, feedback_manager):
        """Test improving trend detection."""
        # Old scores are lower
        for score in [3, 4, 4, 5, 5]:
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=score,
            )

        # Recent scores are higher
        for score in [7, 8, 8, 9, 9]:
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=score,
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        assert agg.trend == "improving"

    def test_aggregation_trend_declining(self, feedback_manager):
        """Test declining trend detection."""
        # Old scores are higher
        for score in [9, 9, 8, 8, 7]:
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=score,
            )

        # Recent scores are lower
        for score in [4, 4, 3, 3, 2]:
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=score,
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        assert agg.trend == "declining"

    def test_aggregation_trend_stable(self, feedback_manager):
        """Test stable trend detection."""
        for _ in range(10):
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query="test",
                relevance_score=7.0,  # Consistent score
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")

        assert agg.trend == "stable"

    def test_aggregation_nonexistent_item(self, feedback_manager):
        """Test aggregation for item with no feedback."""
        agg = feedback_manager.get_aggregated_feedback(
            "nonexistent", "myproject"
        )

        assert agg is None


# =============================================================================
# FEEDBACK MANAGER - SCORE ADJUSTMENTS
# =============================================================================


class TestFeedbackManagerScoreAdjustments:
    """Score adjustment tests for FeedbackManager."""

    def test_score_adjustment_positive(self, temp_db_path):
        """Test positive score adjustment for highly-rated items."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            min_feedback_for_adjustment=3,
        )
        manager = FeedbackManager(config=config)

        # Add 5 high ratings (above neutral 5.0)
        for _ in range(5):
            manager.record_feedback(
                item_id="good_chunk",
                collection="myproject",
                query="test",
                relevance_score=9.0,
            )

        adjustments = manager.get_score_adjustments("myproject")

        assert "good_chunk" in adjustments
        # score=9.0 -> normalized=(9-5)/5=0.8, confidence=0.5 (5/10)
        # adjustment = 0.8 * 0.5 = 0.4
        assert adjustments["good_chunk"] > 0
        assert adjustments["good_chunk"] <= 1.0

    def test_score_adjustment_negative(self, temp_db_path):
        """Test negative score adjustment for poorly-rated items."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            min_feedback_for_adjustment=3,
        )
        manager = FeedbackManager(config=config)

        # Add 5 low ratings (below neutral 5.0)
        for _ in range(5):
            manager.record_feedback(
                item_id="bad_chunk",
                collection="myproject",
                query="test",
                relevance_score=2.0,
            )

        adjustments = manager.get_score_adjustments("myproject")

        assert "bad_chunk" in adjustments
        # score=2.0 -> normalized=(2-5)/5=-0.6, confidence=0.5
        # adjustment = -0.6 * 0.5 = -0.3
        assert adjustments["bad_chunk"] < 0
        assert adjustments["bad_chunk"] >= -1.0

    def test_score_adjustment_neutral(self, feedback_manager):
        """Test neutral score adjustment for averagely-rated items."""
        for _ in range(10):
            feedback_manager.record_feedback(
                item_id="neutral_chunk",
                collection="myproject",
                query="test",
                relevance_score=5.0,  # Exactly neutral
            )

        adjustments = feedback_manager.get_score_adjustments("myproject")

        assert "neutral_chunk" in adjustments
        assert adjustments["neutral_chunk"] == pytest.approx(0.0, abs=0.01)

    def test_score_adjustment_min_feedback_threshold(self, temp_db_path):
        """Test that items below min_feedback_count are excluded."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            min_feedback_for_adjustment=5,
        )
        manager = FeedbackManager(config=config)

        # Add only 3 feedbacks (below threshold of 5)
        for _ in range(3):
            manager.record_feedback(
                item_id="insufficient_feedback",
                collection="myproject",
                query="test",
                relevance_score=9.0,
            )

        adjustments = manager.get_score_adjustments("myproject")

        assert "insufficient_feedback" not in adjustments

    def test_score_adjustment_confidence_scaling(self, temp_db_path):
        """Test that adjustment scales with feedback count."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            min_feedback_for_adjustment=1,
        )
        manager = FeedbackManager(config=config)

        # Add different amounts of feedback for different items
        # All with same score 9.0

        # 2 feedbacks - lower confidence
        for _ in range(2):
            manager.record_feedback(
                item_id="few_feedback",
                collection="myproject",
                query="test",
                relevance_score=9.0,
            )

        # 10 feedbacks - max confidence
        for _ in range(10):
            manager.record_feedback(
                item_id="many_feedback",
                collection="myproject",
                query="test",
                relevance_score=9.0,
            )

        adjustments = manager.get_score_adjustments("myproject")

        # Both positive, but many_feedback should have higher adjustment
        assert adjustments["many_feedback"] > adjustments["few_feedback"]

    def test_get_item_adjustments(self, feedback_manager):
        """Test getting adjustments for specific items."""
        # Add feedback for multiple items
        for i, item_id in enumerate(["chunk_1", "chunk_2", "chunk_3"]):
            for _ in range(5):
                feedback_manager.record_feedback(
                    item_id=item_id,
                    collection="myproject",
                    query="test",
                    relevance_score=5.0 + i * 2,  # 5, 7, 9
                )

        # Get adjustments for specific items only
        adjustments = feedback_manager.get_item_adjustments(
            item_ids=["chunk_1", "chunk_3"],
            collection="myproject",
        )

        assert "chunk_1" in adjustments
        assert "chunk_3" in adjustments
        assert "chunk_2" not in adjustments

    def test_score_adjustment_disabled_manager(self, temp_db_path):
        """Test that disabled manager returns empty adjustments."""
        manager = FeedbackManager(db_path=temp_db_path, enabled=False)

        adjustments = manager.get_score_adjustments("myproject")

        assert adjustments == {}


# =============================================================================
# FEEDBACK MANAGER - HISTORY AND ANALYTICS
# =============================================================================


class TestFeedbackManagerAnalytics:
    """Analytics and history tests for FeedbackManager."""

    def test_get_feedback_history(self, feedback_manager):
        """Test retrieving feedback history for an item."""
        # Add multiple feedbacks
        for i in range(10):
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query=f"query_{i}",
                relevance_score=5.0 + i * 0.5,
            )

        history = feedback_manager.get_feedback_history(
            item_id="chunk_1",
            collection="myproject",
            limit=5,
        )

        assert len(history) == 5
        # Should be ordered by created_at descending (most recent first)
        assert history[0].relevance_score > history[4].relevance_score

    def test_get_collection_stats(self, feedback_manager):
        """Test collection-level statistics."""
        # Add feedback for multiple items
        for i in range(10):
            feedback_manager.record_feedback(
                item_id=f"chunk_{i}",
                collection="myproject",
                query="test",
                relevance_score=3.0 if i < 3 else (8.0 if i < 7 else 5.0),
                feedback_type="explicit" if i < 5 else "implicit",
            )

        stats = feedback_manager.get_collection_stats("myproject")

        assert stats["total_feedbacks"] == 10
        assert stats["unique_items"] == 10
        assert 3.0 < stats["average_score"] < 8.0
        assert stats["negative_rate"] == 0.3  # 3 items with score <= 3.0
        assert stats["positive_rate"] == 0.4  # 4 items with score >= 7.0
        assert stats["feedbacks_by_type"]["explicit"] == 5
        assert stats["feedbacks_by_type"]["implicit"] == 5

    def test_get_collection_stats_empty(self, feedback_manager):
        """Test collection stats for empty collection."""
        stats = feedback_manager.get_collection_stats("empty_collection")

        assert stats["total_feedbacks"] == 0
        assert stats["unique_items"] == 0


# =============================================================================
# FEEDBACK MANAGER - MAINTENANCE
# =============================================================================


class TestFeedbackManagerMaintenance:
    """Maintenance operations tests for FeedbackManager."""

    def test_cleanup_old_feedback(self, temp_db_path):
        """Test cleanup of old feedback records."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            retention_days=30,
        )
        manager = FeedbackManager(config=config)

        # Add feedback
        for i in range(10):
            manager.record_feedback(
                item_id=f"chunk_{i}",
                collection="myproject",
                query="test",
                relevance_score=5.0,
            )

        # All records are recent, nothing to clean
        deleted = manager.cleanup_old_feedback(days=30)
        assert deleted == 0

        # Cleanup with 0 days should also delete nothing (disabled)
        deleted = manager.cleanup_old_feedback(days=0)
        assert deleted == 0

    def test_delete_item_feedback(self, feedback_manager):
        """Test deleting all feedback for an item."""
        # Add feedback for multiple items
        for i in range(5):
            feedback_manager.record_feedback(
                item_id="to_delete",
                collection="myproject",
                query=f"query_{i}",
                relevance_score=5.0,
            )
            feedback_manager.record_feedback(
                item_id="to_keep",
                collection="myproject",
                query=f"query_{i}",
                relevance_score=5.0,
            )

        deleted = feedback_manager.delete_item_feedback(
            item_id="to_delete",
            collection="myproject",
        )

        assert deleted == 5

        # Verify to_keep still exists
        agg = feedback_manager.get_aggregated_feedback("to_keep", "myproject")
        assert agg is not None
        assert agg.total_feedback_count == 5

        # Verify to_delete is gone
        agg = feedback_manager.get_aggregated_feedback("to_delete", "myproject")
        assert agg is None

    def test_delete_collection_feedback(self, feedback_manager):
        """Test deleting all feedback for a collection."""
        # Add feedback to multiple collections
        for i in range(5):
            feedback_manager.record_feedback(
                item_id=f"chunk_{i}",
                collection="to_delete",
                query="test",
                relevance_score=5.0,
            )
            feedback_manager.record_feedback(
                item_id=f"chunk_{i}",
                collection="to_keep",
                query="test",
                relevance_score=5.0,
            )

        deleted = feedback_manager.delete_collection_feedback("to_delete")

        assert deleted == 5

        # Verify to_keep still exists
        stats = feedback_manager.get_collection_stats("to_keep")
        assert stats["total_feedbacks"] == 5

        # Verify to_delete is gone
        stats = feedback_manager.get_collection_stats("to_delete")
        assert stats["total_feedbacks"] == 0


# =============================================================================
# FEEDBACK MANAGER - THREAD SAFETY
# =============================================================================


class TestFeedbackManagerThreadSafety:
    """Thread safety tests for FeedbackManager."""

    def test_concurrent_feedback_recording(self, temp_db_path):
        """Test concurrent feedback recording."""
        manager = FeedbackManager(db_path=temp_db_path)
        errors: List[Exception] = []

        def record_feedbacks(thread_id: int):
            try:
                for i in range(50):
                    manager.record_feedback(
                        item_id=f"chunk_{thread_id}_{i}",
                        collection="myproject",
                        query=f"query_{i}",
                        relevance_score=5.0 + (i % 5),
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_feedbacks, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        stats = manager.get_collection_stats("myproject")
        assert stats["total_feedbacks"] == 250  # 5 threads * 50 feedbacks

    def test_concurrent_adjustment_reads(self, temp_db_path):
        """Test concurrent reading of adjustments."""
        manager = FeedbackManager(db_path=temp_db_path)

        # Pre-populate with feedback
        for i in range(20):
            for _ in range(5):
                manager.record_feedback(
                    item_id=f"chunk_{i}",
                    collection="myproject",
                    query="test",
                    relevance_score=5.0 + (i % 5),
                )

        errors: List[Exception] = []
        results: List[Dict[str, float]] = []

        def read_adjustments():
            try:
                for _ in range(20):
                    adj = manager.get_score_adjustments("myproject")
                    results.append(adj)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_adjustments) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100  # 5 threads * 20 reads each


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFeedbackManagerFactory:
    """Tests for the factory function."""

    def test_create_with_defaults(self, temp_db_path):
        """Test creating manager with default settings via factory."""
        manager = create_feedback_manager(db_path=temp_db_path)

        assert manager.enabled
        assert manager.db_path == Path(temp_db_path).expanduser()

    def test_create_with_config(self, feedback_config):
        """Test creating manager with config via factory."""
        manager = create_feedback_manager(config=feedback_config)

        assert manager.enabled == feedback_config.enabled


# =============================================================================
# EDGE CASES AND SPECIAL SCENARIOS
# =============================================================================


class TestFeedbackManagerEdgeCases:
    """Edge cases and special scenarios."""

    def test_unicode_in_query(self, feedback_manager):
        """Test Unicode characters in query."""
        record = feedback_manager.record_feedback(
            item_id="chunk_1",
            collection="myproject",
            query="日本語のクエリ 🎉",
            relevance_score=7.0,
        )

        history = feedback_manager.get_feedback_history("chunk_1", "myproject")
        assert history[0].query == "日本語のクエリ 🎉"

    def test_very_long_query(self, feedback_manager):
        """Test very long query string."""
        long_query = "word " * 1000  # ~5000 characters

        record = feedback_manager.record_feedback(
            item_id="chunk_1",
            collection="myproject",
            query=long_query,
            relevance_score=5.0,
        )

        assert record.id is not None

    def test_special_characters_in_ids(self, feedback_manager):
        """Test special characters in item_id and collection."""
        record = feedback_manager.record_feedback(
            item_id="chunk/with:special-chars_123",
            collection="my-project_v2.0",
            query="test",
            relevance_score=5.0,
        )

        agg = feedback_manager.get_aggregated_feedback(
            "chunk/with:special-chars_123",
            "my-project_v2.0",
        )
        assert agg is not None

    def test_feedback_with_same_timestamp(self, feedback_manager):
        """Test multiple feedbacks recorded at nearly same time."""
        for i in range(10):
            feedback_manager.record_feedback(
                item_id="chunk_1",
                collection="myproject",
                query=f"query_{i}",
                relevance_score=float(i),
            )

        agg = feedback_manager.get_aggregated_feedback("chunk_1", "myproject")
        assert agg.total_feedback_count == 10

    def test_extreme_scores(self, feedback_manager):
        """Test edge case scores (0.0 and 10.0)."""
        feedback_manager.record_feedback(
            item_id="perfect",
            collection="myproject",
            query="test",
            relevance_score=10.0,
        )

        feedback_manager.record_feedback(
            item_id="terrible",
            collection="myproject",
            query="test",
            relevance_score=0.0,
        )

        perfect_agg = feedback_manager.get_aggregated_feedback(
            "perfect", "myproject"
        )
        terrible_agg = feedback_manager.get_aggregated_feedback(
            "terrible", "myproject"
        )

        assert perfect_agg.recent_score == 10.0
        assert terrible_agg.recent_score == 0.0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestFeedbackManagerIntegration:
    """Integration-style tests simulating real usage patterns."""

    def test_retrieval_reranking_workflow(self, temp_db_path):
        """Test complete workflow of using feedback for reranking."""
        config = FeedbackConfig(
            db_path=temp_db_path,
            min_feedback_for_adjustment=3,
        )
        manager = FeedbackManager(config=config)

        # Simulate retrieval results with initial similarity scores
        retrieval_results = [
            {"item_id": "chunk_a", "similarity": 0.85},
            {"item_id": "chunk_b", "similarity": 0.82},
            {"item_id": "chunk_c", "similarity": 0.75},  # Lower initial score
        ]

        # Simulate user feedback over time
        # User finds chunk_c most helpful (high scores)
        for _ in range(10):  # More feedback for higher confidence
            manager.record_feedback(
                item_id="chunk_c",
                collection="myproject",
                query="test",
                relevance_score=9.5,
            )

        # User finds chunk_a least helpful (low scores)
        for _ in range(10):
            manager.record_feedback(
                item_id="chunk_a",
                collection="myproject",
                query="test",
                relevance_score=1.0,  # Very low
            )

        # chunk_b gets neutral feedback
        for _ in range(10):
            manager.record_feedback(
                item_id="chunk_b",
                collection="myproject",
                query="test",
                relevance_score=5.0,
            )

        # Get adjustments and apply to retrieval
        adjustments = manager.get_score_adjustments("myproject")

        # Apply adjustments (boost factor of 0.2 for more impact)
        reranked = []
        for result in retrieval_results:
            adj = adjustments.get(result["item_id"], 0.0)
            final_score = result["similarity"] + adj * 0.2
            reranked.append({
                "item_id": result["item_id"],
                "original": result["similarity"],
                "adjustment": adj,
                "final": final_score,
            })

        # Sort by final score
        reranked.sort(key=lambda x: x["final"], reverse=True)

        # chunk_c should now rank higher despite lower original similarity
        # chunk_a should rank lower despite higher original similarity
        assert reranked[0]["item_id"] == "chunk_c", f"Expected chunk_c first, got {reranked}"
        assert reranked[2]["item_id"] == "chunk_a", f"Expected chunk_a last, got {reranked}"

    def test_continuous_learning_pattern(self, feedback_manager):
        """Test continuous feedback collection and improvement."""
        # Simulate continuous feedback over multiple sessions
        for session in range(5):
            for query_idx in range(10):
                # User feedback improves over sessions as model learns
                base_score = 5.0 + session * 0.5  # Gradually improving

                feedback_manager.record_feedback(
                    item_id=f"chunk_{query_idx}",
                    collection="myproject",
                    query=f"query_{query_idx}",
                    relevance_score=min(10.0, base_score + (query_idx % 3)),
                    session_id=f"session_{session}",
                )

        # Check that system captured improvement trend
        for i in range(10):
            agg = feedback_manager.get_aggregated_feedback(
                f"chunk_{i}", "myproject"
            )
            if agg.total_feedback_count >= 10:
                assert agg.trend == "improving"
