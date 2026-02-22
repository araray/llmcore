# src/llmcore/storage/feedback.py
"""
Feedback System for Retrieval Quality Improvement.

This module provides a feedback tracking system that:
1. Records user/agent feedback on retrieved items
2. Aggregates feedback with recency weighting
3. Generates score adjustments for retrieval reranking

The feedback system enables continuous improvement of RAG quality by
learning from explicit user ratings and implicit signals (clicks, usage time).

Key Concepts:
- FeedbackRecord: Individual feedback submission
- AggregatedFeedback: Computed statistics for an item
- Score Adjustment: Reranking boost/penalty based on feedback history

Feedback Types:
- explicit: User directly rates relevance (1-10 scale)
- implicit: Derived from user behavior (clicks, dwell time)
- agent: AI agent self-assessment of retrieval quality

Score Adjustment Algorithm:
The score adjustment is computed as:
    adjustment = (recent_score - 5) / 5 * confidence

Where:
- recent_score: Exponentially weighted recent average (last 10 feedbacks)
- confidence: min(1.0, feedback_count / 10)

This produces adjustments in [-1.0, +1.0] range, which can be added to
similarity scores during retrieval reranking.

Usage:
    manager = FeedbackManager()

    # Record feedback
    manager.record_feedback(
        item_id="chunk_abc123",
        collection="myproject",
        query="How does authentication work?",
        relevance_score=8.5,
        feedback_type="explicit"
    )

    # Get score adjustments for reranking
    adjustments = manager.get_score_adjustments("myproject")
    for item_id, boost in adjustments.items():
        # Apply to retrieval scores
        final_score = similarity_score + (boost * 0.1)

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 3, Task 3.2
- Storage_System_Spec_v2r0.md Section 5 (Feedback Integration)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class FeedbackConfig(BaseModel):
    """Configuration for the feedback system.

    Attributes:
        enabled: Whether feedback tracking is enabled.
        db_path: Path to the SQLite database file.
        min_feedback_for_adjustment: Minimum feedbacks before generating adjustments.
        decay_factor: Exponential decay factor for recency weighting.
        positive_threshold: Score threshold for counting as positive feedback.
        negative_threshold: Score threshold for counting as negative feedback.
        max_feedbacks_per_item: Maximum feedbacks to store per item.
        retention_days: Days to retain feedback records (0 = forever).
    """

    enabled: bool = Field(default=True, description="Enable feedback tracking")
    db_path: str = Field(
        default="~/.local/share/llmcore/feedback.db",
        description="Path to feedback SQLite database",
    )
    min_feedback_for_adjustment: int = Field(
        default=3, ge=1, description="Minimum feedback count for score adjustments"
    )
    decay_factor: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Exponential decay for recency"
    )
    positive_threshold: float = Field(
        default=7.0, ge=0.0, le=10.0, description="Score threshold for positive"
    )
    negative_threshold: float = Field(
        default=3.0, ge=0.0, le=10.0, description="Score threshold for negative"
    )
    max_feedbacks_per_item: int = Field(
        default=100, ge=1, description="Max feedbacks stored per item"
    )
    retention_days: int = Field(default=0, ge=0, description="Days to retain feedback (0=forever)")


# =============================================================================
# DATA MODELS
# =============================================================================


class FeedbackRecord(BaseModel):
    """A single feedback record.

    Represents one piece of feedback from a user or agent about
    the relevance of a retrieved item to a query.

    Attributes:
        id: Database ID (None until persisted).
        item_id: Identifier of the retrieved item (chunk ID).
        collection: Name of the collection containing the item.
        query: The query that led to this retrieval.
        relevance_score: Relevance rating from 0.0 (irrelevant) to 10.0 (perfect).
        feedback_type: Source of feedback (explicit, implicit, agent).
        provider_id: Identifier of the feedback provider (user ID or agent ID).
        session_id: Optional session identifier for grouping.
        created_at: When the feedback was recorded.
    """

    id: int | None = Field(default=None, description="Database record ID")
    item_id: str = Field(..., description="Retrieved item identifier")
    collection: str = Field(..., description="Collection name")
    query: str = Field(..., description="Query that triggered retrieval")
    relevance_score: float = Field(..., ge=0.0, le=10.0, description="Relevance score 0-10")
    feedback_type: Literal["explicit", "implicit", "agent"] = Field(
        default="explicit", description="Type of feedback"
    )
    provider_id: str = Field(default="user", description="Feedback provider ID")
    session_id: str | None = Field(default=None, description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class AggregatedFeedback(BaseModel):
    """Aggregated feedback statistics for an item.

    Computed from all feedback records for an item in a collection.

    Attributes:
        item_id: The item identifier.
        collection: The collection name.
        total_feedback_count: Total number of feedback records.
        average_score: Simple average of all scores.
        recent_score: Exponentially weighted recent average.
        positive_count: Number of feedbacks >= positive_threshold.
        negative_count: Number of feedbacks <= negative_threshold.
        trend: Direction of score changes (improving, declining, stable).
        first_feedback_at: Timestamp of first feedback.
        last_feedback_at: Timestamp of most recent feedback.
        updated_at: When aggregates were last computed.
    """

    item_id: str = Field(..., description="Item identifier")
    collection: str = Field(..., description="Collection name")
    total_feedback_count: int = Field(default=0, description="Total feedback count")
    average_score: float = Field(default=0.0, description="Average score")
    recent_score: float = Field(default=0.0, description="Recency-weighted score")
    positive_count: int = Field(default=0, description="Positive feedback count")
    negative_count: int = Field(default=0, description="Negative feedback count")
    trend: Literal["improving", "declining", "stable"] = Field(
        default="stable", description="Score trend"
    )
    first_feedback_at: datetime | None = Field(default=None, description="First feedback timestamp")
    last_feedback_at: datetime | None = Field(default=None, description="Last feedback timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


# =============================================================================
# FEEDBACK MANAGER
# =============================================================================


class FeedbackManager:
    """Manages feedback storage and aggregation.

    This class provides:
    - Recording individual feedback records
    - Automatic aggregation with recency weighting
    - Score adjustment generation for reranking
    - Feedback analytics and statistics

    The manager uses SQLite for persistence, making it suitable for
    single-process applications. For distributed systems, consider
    using PostgreSQL with the pgvector_enhanced storage.

    Example:
        manager = FeedbackManager()

        # Record feedback on retrieved items
        manager.record_feedback(
            item_id="chunk_123",
            collection="myproject",
            query="How does auth work?",
            relevance_score=8.5,
            feedback_type="explicit"
        )

        # Get adjustments for reranking
        boosts = manager.get_score_adjustments("myproject")
        # Apply: final_score = similarity + boosts.get(item_id, 0) * 0.1

    Attributes:
        enabled: Whether feedback tracking is active.
        db_path: Path to the SQLite database.
    """

    SCHEMA = """
    -- Individual feedback records
    CREATE TABLE IF NOT EXISTS feedback_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id TEXT NOT NULL,
        collection TEXT NOT NULL,
        query TEXT NOT NULL,
        relevance_score REAL NOT NULL,
        feedback_type TEXT NOT NULL,
        provider_id TEXT NOT NULL,
        session_id TEXT,
        created_at TEXT NOT NULL
    );

    -- Index for item/collection lookups
    CREATE INDEX IF NOT EXISTS idx_feedback_item_coll
        ON feedback_records(item_id, collection);

    -- Index for time-based queries
    CREATE INDEX IF NOT EXISTS idx_feedback_created
        ON feedback_records(created_at);

    -- Index for collection analytics
    CREATE INDEX IF NOT EXISTS idx_feedback_collection
        ON feedback_records(collection);

    -- Precomputed aggregates (updated on each new feedback)
    CREATE TABLE IF NOT EXISTS aggregated_feedback (
        item_id TEXT NOT NULL,
        collection TEXT NOT NULL,
        total_feedback_count INTEGER DEFAULT 0,
        average_score REAL DEFAULT 0.0,
        recent_score REAL DEFAULT 0.0,
        positive_count INTEGER DEFAULT 0,
        negative_count INTEGER DEFAULT 0,
        trend TEXT DEFAULT 'stable',
        first_feedback_at TEXT,
        last_feedback_at TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (item_id, collection)
    );
    """

    def __init__(
        self,
        db_path: str = "~/.local/share/llmcore/feedback.db",
        enabled: bool = True,
        config: FeedbackConfig | None = None,
    ) -> None:
        """Initialize the feedback manager.

        Args:
            db_path: Path to the SQLite database file.
            enabled: Whether feedback tracking is enabled.
            config: Optional configuration object (overrides other params).
        """
        # Apply config if provided
        if config is not None:
            db_path = config.db_path
            enabled = config.enabled
            self._config = config
        else:
            self._config = FeedbackConfig(db_path=db_path, enabled=enabled)

        self.enabled = enabled
        self.db_path = Path(db_path).expanduser()
        self._lock = threading.RLock()

        if enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            logger.debug(f"FeedbackManager initialized: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    # -------------------------------------------------------------------------
    # Feedback Recording
    # -------------------------------------------------------------------------

    def record_feedback(
        self,
        item_id: str,
        collection: str,
        query: str,
        relevance_score: float,
        feedback_type: Literal["explicit", "implicit", "agent"] = "explicit",
        provider_id: str = "user",
        session_id: str | None = None,
    ) -> FeedbackRecord:
        """Record feedback for a retrieved item.

        Args:
            item_id: Identifier of the retrieved item.
            collection: Collection containing the item.
            query: The query that led to this retrieval.
            relevance_score: Relevance rating from 0.0 to 10.0.
            feedback_type: Source of feedback.
            provider_id: Identifier of feedback provider.
            session_id: Optional session identifier.

        Returns:
            The recorded FeedbackRecord with assigned ID.

        Raises:
            ValueError: If relevance_score is out of range.
        """
        # Clamp score to valid range
        clamped_score = max(0.0, min(10.0, relevance_score))

        record = FeedbackRecord(
            item_id=item_id,
            collection=collection,
            query=query,
            relevance_score=clamped_score,
            feedback_type=feedback_type,
            provider_id=provider_id,
            session_id=session_id,
        )

        if not self.enabled:
            return record

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO feedback_records
                       (item_id, collection, query, relevance_score, feedback_type,
                        provider_id, session_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.item_id,
                        record.collection,
                        record.query,
                        record.relevance_score,
                        record.feedback_type,
                        record.provider_id,
                        record.session_id,
                        record.created_at.isoformat(),
                    ),
                )
                record.id = cursor.lastrowid

            # Update aggregates
            self._update_aggregates(item_id, collection)

        logger.debug(
            f"Recorded feedback: item={item_id}, collection={collection}, "
            f"score={clamped_score}, type={feedback_type}"
        )

        return record

    def record_batch(
        self,
        feedbacks: list[dict[str, Any]],
    ) -> list[FeedbackRecord]:
        """Record multiple feedback records efficiently.

        Args:
            feedbacks: List of feedback dictionaries with keys:
                item_id, collection, query, relevance_score,
                and optionally feedback_type, provider_id, session_id.

        Returns:
            List of recorded FeedbackRecord objects.
        """
        if not self.enabled:
            return []

        records = []
        affected_items: dict[str, set] = {}  # collection -> set of item_ids

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for fb in feedbacks:
                    record = FeedbackRecord(
                        item_id=fb["item_id"],
                        collection=fb["collection"],
                        query=fb["query"],
                        relevance_score=max(0.0, min(10.0, fb["relevance_score"])),
                        feedback_type=fb.get("feedback_type", "explicit"),
                        provider_id=fb.get("provider_id", "user"),
                        session_id=fb.get("session_id"),
                    )

                    cursor = conn.execute(
                        """INSERT INTO feedback_records
                           (item_id, collection, query, relevance_score, feedback_type,
                            provider_id, session_id, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            record.item_id,
                            record.collection,
                            record.query,
                            record.relevance_score,
                            record.feedback_type,
                            record.provider_id,
                            record.session_id,
                            record.created_at.isoformat(),
                        ),
                    )
                    record.id = cursor.lastrowid
                    records.append(record)

                    # Track for aggregate update
                    if record.collection not in affected_items:
                        affected_items[record.collection] = set()
                    affected_items[record.collection].add(record.item_id)

            # Update aggregates for all affected items
            for collection, item_ids in affected_items.items():
                for item_id in item_ids:
                    self._update_aggregates(item_id, collection)

        logger.debug(f"Recorded {len(records)} feedback records in batch")
        return records

    # -------------------------------------------------------------------------
    # Aggregate Computation
    # -------------------------------------------------------------------------

    def _update_aggregates(self, item_id: str, collection: str) -> None:
        """Update aggregated feedback for an item.

        Computes:
        - Average score across all feedbacks
        - Recent score with exponential decay weighting
        - Positive/negative counts
        - Trend direction

        Args:
            item_id: The item identifier.
            collection: The collection name.
        """
        decay = self._config.decay_factor
        positive_thresh = self._config.positive_threshold
        negative_thresh = self._config.negative_threshold

        with sqlite3.connect(self.db_path) as conn:
            # Get all feedbacks for this item, most recent first
            rows = conn.execute(
                """SELECT relevance_score, created_at FROM feedback_records
                   WHERE item_id = ? AND collection = ?
                   ORDER BY created_at DESC""",
                (item_id, collection),
            ).fetchall()

            if not rows:
                return

            scores = [row[0] for row in rows]
            timestamps = [row[1] for row in rows]
            total_count = len(scores)
            average_score = sum(scores) / total_count

            # Compute recent score with exponential decay
            # More recent feedbacks have higher weight
            recent_score = 0.0
            weight_sum = 0.0
            for i, score in enumerate(scores[:10]):  # Use up to 10 most recent
                weight = decay**i
                recent_score += score * weight
                weight_sum += weight
            recent_score = recent_score / weight_sum if weight_sum > 0 else average_score

            # Count positive and negative
            positive_count = sum(1 for s in scores if s >= positive_thresh)
            negative_count = sum(1 for s in scores if s <= negative_thresh)

            # Determine trend (compare recent 5 vs previous 5)
            trend = "stable"
            if total_count >= 10:
                recent_avg = sum(scores[:5]) / 5
                older_avg = sum(scores[5:10]) / 5
                if recent_avg > older_avg + 0.5:
                    trend = "improving"
                elif recent_avg < older_avg - 0.5:
                    trend = "declining"

            # Get first and last timestamps
            first_feedback_at = timestamps[-1] if timestamps else None
            last_feedback_at = timestamps[0] if timestamps else None

            # Upsert aggregates
            conn.execute(
                """INSERT INTO aggregated_feedback
                   (item_id, collection, total_feedback_count, average_score,
                    recent_score, positive_count, negative_count, trend,
                    first_feedback_at, last_feedback_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(item_id, collection) DO UPDATE SET
                       total_feedback_count = excluded.total_feedback_count,
                       average_score = excluded.average_score,
                       recent_score = excluded.recent_score,
                       positive_count = excluded.positive_count,
                       negative_count = excluded.negative_count,
                       trend = excluded.trend,
                       first_feedback_at = excluded.first_feedback_at,
                       last_feedback_at = excluded.last_feedback_at,
                       updated_at = excluded.updated_at""",
                (
                    item_id,
                    collection,
                    total_count,
                    average_score,
                    recent_score,
                    positive_count,
                    negative_count,
                    trend,
                    first_feedback_at,
                    last_feedback_at,
                    datetime.utcnow().isoformat(),
                ),
            )

    # -------------------------------------------------------------------------
    # Score Adjustments for Reranking
    # -------------------------------------------------------------------------

    def get_score_adjustments(
        self,
        collection: str,
        min_feedback_count: int | None = None,
    ) -> dict[str, float]:
        """Get score adjustments for retrieval reranking.

        Returns a dictionary mapping item_id to adjustment value in [-1.0, +1.0].
        Items with good feedback get positive adjustments, bad feedback
        gets negative adjustments.

        The adjustment formula:
            adjustment = (recent_score - 5) / 5 * confidence
            confidence = min(1.0, feedback_count / 10)

        Args:
            collection: Collection to get adjustments for.
            min_feedback_count: Minimum feedback count to include item
                               (defaults to config value).

        Returns:
            Dictionary mapping item_id to adjustment factor.
        """
        if not self.enabled:
            return {}

        if min_feedback_count is None:
            min_feedback_count = self._config.min_feedback_for_adjustment

        adjustments: dict[str, float] = {}

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT item_id, recent_score, total_feedback_count
                   FROM aggregated_feedback
                   WHERE collection = ? AND total_feedback_count >= ?""",
                (collection, min_feedback_count),
            ).fetchall()

            for item_id, recent_score, count in rows:
                # Normalize score from [0,10] to [-1,+1]
                normalized = (recent_score - 5) / 5
                # Apply confidence factor based on feedback count
                confidence = min(1.0, count / 10)
                adjustments[item_id] = normalized * confidence

        return adjustments

    def get_item_adjustments(
        self,
        item_ids: list[str],
        collection: str,
    ) -> dict[str, float]:
        """Get score adjustments for specific items.

        More efficient than get_score_adjustments when you only need
        adjustments for items in a specific result set.

        Args:
            item_ids: List of item IDs to get adjustments for.
            collection: Collection containing the items.

        Returns:
            Dictionary mapping item_id to adjustment factor.
        """
        if not self.enabled or not item_ids:
            return {}

        adjustments: dict[str, float] = {}
        min_count = self._config.min_feedback_for_adjustment

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(item_ids))
            rows = conn.execute(
                f"""SELECT item_id, recent_score, total_feedback_count
                    FROM aggregated_feedback
                    WHERE collection = ? AND item_id IN ({placeholders})
                    AND total_feedback_count >= ?""",
                [collection] + list(item_ids) + [min_count],
            ).fetchall()

            for item_id, recent_score, count in rows:
                normalized = (recent_score - 5) / 5
                confidence = min(1.0, count / 10)
                adjustments[item_id] = normalized * confidence

        return adjustments

    # -------------------------------------------------------------------------
    # Analytics & Queries
    # -------------------------------------------------------------------------

    def get_aggregated_feedback(
        self,
        item_id: str,
        collection: str,
    ) -> AggregatedFeedback | None:
        """Get aggregated feedback statistics for an item.

        Args:
            item_id: The item identifier.
            collection: The collection name.

        Returns:
            AggregatedFeedback object, or None if no feedback exists.
        """
        if not self.enabled:
            return None

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT item_id, collection, total_feedback_count, average_score,
                          recent_score, positive_count, negative_count, trend,
                          first_feedback_at, last_feedback_at, updated_at
                   FROM aggregated_feedback
                   WHERE item_id = ? AND collection = ?""",
                (item_id, collection),
            ).fetchone()

            if row is None:
                return None

            return AggregatedFeedback(
                item_id=row[0],
                collection=row[1],
                total_feedback_count=row[2],
                average_score=row[3],
                recent_score=row[4],
                positive_count=row[5],
                negative_count=row[6],
                trend=row[7],
                first_feedback_at=(datetime.fromisoformat(row[8]) if row[8] else None),
                last_feedback_at=(datetime.fromisoformat(row[9]) if row[9] else None),
                updated_at=datetime.fromisoformat(row[10]),
            )

    def get_feedback_history(
        self,
        item_id: str,
        collection: str,
        limit: int = 50,
    ) -> list[FeedbackRecord]:
        """Get feedback history for an item.

        Args:
            item_id: The item identifier.
            collection: The collection name.
            limit: Maximum records to return.

        Returns:
            List of FeedbackRecord objects, most recent first.
        """
        if not self.enabled:
            return []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, item_id, collection, query, relevance_score,
                          feedback_type, provider_id, session_id, created_at
                   FROM feedback_records
                   WHERE item_id = ? AND collection = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (item_id, collection, limit),
            ).fetchall()

            return [
                FeedbackRecord(
                    id=row[0],
                    item_id=row[1],
                    collection=row[2],
                    query=row[3],
                    relevance_score=row[4],
                    feedback_type=row[5],
                    provider_id=row[6],
                    session_id=row[7],
                    created_at=datetime.fromisoformat(row[8]),
                )
                for row in rows
            ]

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get feedback statistics for a collection.

        Args:
            collection: The collection name.

        Returns:
            Dictionary with statistics including:
            - total_feedbacks: Total feedback records
            - unique_items: Number of unique items with feedback
            - average_score: Overall average score
            - positive_rate: Percentage of positive feedbacks
            - negative_rate: Percentage of negative feedbacks
            - feedbacks_by_type: Breakdown by feedback type
        """
        if not self.enabled:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            # Total feedbacks
            total = conn.execute(
                "SELECT COUNT(*) FROM feedback_records WHERE collection = ?",
                (collection,),
            ).fetchone()[0]

            if total == 0:
                return {
                    "total_feedbacks": 0,
                    "unique_items": 0,
                    "average_score": 0.0,
                    "positive_rate": 0.0,
                    "negative_rate": 0.0,
                    "feedbacks_by_type": {},
                }

            # Unique items
            unique_items = conn.execute(
                "SELECT COUNT(DISTINCT item_id) FROM feedback_records WHERE collection = ?",
                (collection,),
            ).fetchone()[0]

            # Average score
            avg_score = conn.execute(
                "SELECT AVG(relevance_score) FROM feedback_records WHERE collection = ?",
                (collection,),
            ).fetchone()[0]

            # Positive/negative rates
            positive_thresh = self._config.positive_threshold
            negative_thresh = self._config.negative_threshold

            positive_count = conn.execute(
                "SELECT COUNT(*) FROM feedback_records WHERE collection = ? AND relevance_score >= ?",
                (collection, positive_thresh),
            ).fetchone()[0]

            negative_count = conn.execute(
                "SELECT COUNT(*) FROM feedback_records WHERE collection = ? AND relevance_score <= ?",
                (collection, negative_thresh),
            ).fetchone()[0]

            # By type
            by_type = {}
            for row in conn.execute(
                "SELECT feedback_type, COUNT(*) FROM feedback_records WHERE collection = ? GROUP BY feedback_type",
                (collection,),
            ):
                by_type[row[0]] = row[1]

            return {
                "total_feedbacks": total,
                "unique_items": unique_items,
                "average_score": avg_score,
                "positive_rate": positive_count / total,
                "negative_rate": negative_count / total,
                "feedbacks_by_type": by_type,
            }

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_old_feedback(self, days: int | None = None) -> int:
        """Remove feedback records older than specified days.

        Args:
            days: Number of days to retain (defaults to config value).
                  Set to 0 to disable cleanup.

        Returns:
            Number of records deleted.
        """
        if not self.enabled:
            return 0

        if days is None:
            days = self._config.retention_days

        if days == 0:
            return 0

        cutoff = datetime.utcnow() - timedelta(days=days)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM feedback_records WHERE created_at < ?",
                    (cutoff.isoformat(),),
                )
                deleted = cursor.rowcount

                # Note: Aggregates are not cleaned up, they remain as historical data
                # This is intentional to preserve accumulated learning

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old feedback records")

        return deleted

    def delete_item_feedback(self, item_id: str, collection: str) -> int:
        """Delete all feedback for a specific item.

        Useful when an item is deleted from the collection.

        Args:
            item_id: The item identifier.
            collection: The collection name.

        Returns:
            Number of records deleted.
        """
        if not self.enabled:
            return 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM feedback_records WHERE item_id = ? AND collection = ?",
                    (item_id, collection),
                )
                deleted = cursor.rowcount

                conn.execute(
                    "DELETE FROM aggregated_feedback WHERE item_id = ? AND collection = ?",
                    (item_id, collection),
                )

        return deleted

    def delete_collection_feedback(self, collection: str) -> int:
        """Delete all feedback for a collection.

        Useful when a collection is deleted.

        Args:
            collection: The collection name.

        Returns:
            Number of records deleted.
        """
        if not self.enabled:
            return 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM feedback_records WHERE collection = ?",
                    (collection,),
                )
                deleted = cursor.rowcount

                conn.execute(
                    "DELETE FROM aggregated_feedback WHERE collection = ?",
                    (collection,),
                )

        logger.info(f"Deleted {deleted} feedback records for collection {collection}")
        return deleted


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_feedback_manager(
    config: FeedbackConfig | None = None,
    **kwargs: Any,
) -> FeedbackManager:
    """Factory function to create a feedback manager.

    Args:
        config: Optional configuration object.
        **kwargs: Additional arguments passed to FeedbackManager.

    Returns:
        Configured FeedbackManager instance.
    """
    if config is not None:
        return FeedbackManager(config=config)
    return FeedbackManager(**kwargs)
