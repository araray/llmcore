# src/llmcore/observability/cost_tracker.py
"""
Cost Tracking for LLM and Embedding API Usage.

This module tracks:
- Token usage (input/output) for all API calls
- Estimated costs based on current pricing data
- API call counts and latency
- Per-session and aggregate statistics

Storage:
- SQLite database for persistence
- In-memory aggregation for current session
- Configurable retention period

The pricing data is sourced from model cards when available, with fallback
to the PRICING_DATA constant for known models.

Key Features:
- Thread-safe operations
- Efficient batch recording
- Flexible aggregation queries (daily, weekly, monthly, by model/provider)
- Export to CSV/JSON
- Budget alerting (future)

Usage:
    tracker = CostTracker(
        db_path="~/.llmcore/costs.db",
        enabled=True
    )

    # Record a single API call
    record = tracker.record(
        provider="openai",
        model="gpt-4o",
        operation="chat",
        input_tokens=1000,
        output_tokens=500,
        session_id="session_123",
    )
    print(f"Cost: ${record.estimated_cost_usd:.6f}")

    # Get today's summary
    summary = tracker.get_daily_summary()
    print(f"Today: {summary.total_calls} calls, ${summary.total_cost_usd:.2f}")

    # Get summary by provider
    by_provider = tracker.get_summary_by_provider(days=30)

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 1, Task 1.4
- RAG_ECOSYSTEM_REDESIGN_SPEC.md Section 4.2
- llmcore model_cards for pricing data
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PRICING DATA
# =============================================================================

# Prices per 1M tokens (as of January 2026)
# These are fallback values; actual pricing should come from model cards when available
# Format: {provider: {model: {"input": price, "output": price}}}
PRICING_DATA: dict[str, dict[str, dict[str, float]]] = {
    # OpenAI
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    },
    # Anthropic
    "anthropic": {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
        "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
        # Common aliases
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 1.00, "output": 5.00},
    },
    # Google (Gemini)
    "google": {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-pro": {"input": 0.50, "output": 1.50},
        "text-embedding-004": {"input": 0.0, "output": 0.0},  # Free tier
        "embedding-001": {"input": 0.0, "output": 0.0},
    },
    # DeepSeek
    "deepseek": {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    },
    # Mistral
    "mistral": {
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-small-latest": {"input": 0.20, "output": 0.60},
        "codestral-latest": {"input": 0.30, "output": 0.90},
        "open-mistral-nemo": {"input": 0.15, "output": 0.15},
        "mistral-embed": {"input": 0.10, "output": 0.0},
    },
    # xAI (Grok)
    "xai": {
        "grok-beta": {"input": 5.00, "output": 15.00},
        "grok-vision-beta": {"input": 5.00, "output": 15.00},
    },
    # Cohere (embeddings)
    "cohere": {
        "embed-english-v3.0": {"input": 0.10, "output": 0.0},
        "embed-multilingual-v3.0": {"input": 0.10, "output": 0.0},
        "embed-english-light-v3.0": {"input": 0.10, "output": 0.0},
        "command-r-plus": {"input": 2.50, "output": 10.00},
        "command-r": {"input": 0.15, "output": 0.60},
    },
    # Voyage AI (embeddings)
    "voyage": {
        "voyage-3": {"input": 0.06, "output": 0.0},
        "voyage-3-lite": {"input": 0.02, "output": 0.0},
        "voyage-code-3": {"input": 0.18, "output": 0.0},
        "voyage-finance-2": {"input": 0.12, "output": 0.0},
        "voyage-law-2": {"input": 0.12, "output": 0.0},
    },
    # Local (free)
    "ollama": {
        "*": {"input": 0.0, "output": 0.0},
    },
    "local": {
        "*": {"input": 0.0, "output": 0.0},
    },
}


def get_price_per_million_tokens(
    provider: str,
    model: str,
    token_type: Literal["input", "output"] = "input",
    model_card_pricing: dict[str, float] | None = None,
) -> float:
    """Get price per 1M tokens for a model.

    Args:
        provider: Provider name (e.g., "openai", "anthropic").
        model: Model identifier.
        token_type: "input" or "output".
        model_card_pricing: Optional pricing from model card.

    Returns:
        Price per 1M tokens in USD.
    """
    # Try model card pricing first
    if model_card_pricing:
        if token_type == "input" and "input" in model_card_pricing:
            return model_card_pricing["input"]
        if token_type == "output" and "output" in model_card_pricing:
            return model_card_pricing["output"]

    # Look up in pricing data
    provider_lower = provider.lower()
    model_lower = model.lower()

    if provider_lower not in PRICING_DATA:
        logger.warning(f"Unknown provider '{provider}', assuming free")
        return 0.0

    provider_pricing = PRICING_DATA[provider_lower]

    # Try exact match first
    if model_lower in provider_pricing:
        return provider_pricing[model_lower].get(token_type, 0.0)

    # Try wildcard (for local models)
    if "*" in provider_pricing:
        return provider_pricing["*"].get(token_type, 0.0)

    # Try prefix match (e.g., "gpt-4o-2024" matches "gpt-4o")
    for known_model, pricing in provider_pricing.items():
        if model_lower.startswith(known_model):
            return pricing.get(token_type, 0.0)

    logger.warning(f"Unknown model '{model}' for provider '{provider}', assuming free")
    return 0.0


# =============================================================================
# DATA MODELS
# =============================================================================


class OperationType(str, Enum):
    """Types of API operations."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    OTHER = "other"


class UsageRecord(BaseModel):
    """Record of a single API usage event.

    Attributes:
        id: Unique identifier for this record.
        timestamp: When the API call was made.
        provider: Provider name (openai, anthropic, etc.).
        model: Model identifier.
        operation: Type of operation (chat, embedding, etc.).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens (input + output).
        estimated_cost_usd: Estimated cost in USD.
        latency_ms: Latency in milliseconds (optional).
        session_id: Session identifier (optional).
        user_id: User identifier (optional).
        metadata: Additional metadata (optional).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    operation: str = Field(default="chat")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int | None = Field(default=None, ge=0)
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Compute total tokens if not set."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class UsageSummary(BaseModel):
    """Summary of usage for a period.

    Attributes:
        period_start: Start of the summary period.
        period_end: End of the summary period.
        total_calls: Total number of API calls.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        total_tokens: Total tokens.
        total_cost_usd: Total estimated cost in USD.
        by_provider: Breakdown by provider.
        by_model: Breakdown by model.
        by_operation: Breakdown by operation type.
    """

    period_start: datetime
    period_end: datetime
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float | None = None
    by_provider: dict[str, dict[str, Any]] = Field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = Field(default_factory=dict)
    by_operation: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CostTrackingConfig(BaseModel):
    """Configuration for cost tracking.

    Attributes:
        enabled: Whether cost tracking is enabled.
        db_path: Path to SQLite database.
        retention_days: Number of days to retain records (0 = forever).
        log_to_console: Log usage records to console.
        track_latency: Track request latency.
    """

    enabled: bool = Field(default=True, description="Enable cost tracking")
    db_path: str = Field(
        default="~/.llmcore/costs.db",
        description="Path to SQLite cost tracking database",
    )
    retention_days: int = Field(default=90, ge=0, description="Days to retain records (0=forever)")
    log_to_console: bool = Field(default=False, description="Log usage to console")
    track_latency: bool = Field(default=True, description="Track request latency")


# =============================================================================
# COST TRACKER
# =============================================================================


class CostTracker:
    """Track and analyze API usage costs.

    This class provides comprehensive cost tracking for LLM and embedding API calls.
    It stores records in SQLite and provides aggregation queries for analysis.

    Features:
    - Thread-safe recording
    - SQLite persistence
    - Flexible aggregation (by day, week, month, provider, model)
    - In-memory current session tracking
    - Export to CSV/JSON

    Example:
        tracker = CostTracker(db_path="~/.llmcore/costs.db")

        # Record usage
        record = tracker.record(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=1000,
            output_tokens=500,
        )

        # Get daily summary
        summary = tracker.get_daily_summary()
        print(f"Today's cost: ${summary.total_cost_usd:.2f}")
    """

    def __init__(
        self,
        config: CostTrackingConfig | None = None,
        db_path: str = "~/.llmcore/costs.db",
        enabled: bool = True,
        retention_days: int = 90,
    ) -> None:
        """Initialize cost tracker.

        Args:
            config: CostTrackingConfig instance. If provided, other args are ignored.
            db_path: Path to SQLite database (used if config is None).
            enabled: Enable tracking (used if config is None).
            retention_days: Days to retain records (used if config is None).
        """
        if config is not None:
            self._config = config
        else:
            self._config = CostTrackingConfig(
                enabled=enabled,
                db_path=db_path,
                retention_days=retention_days,
            )

        self._enabled = self._config.enabled
        self._db_path = Path(self._config.db_path).expanduser()
        self._lock = threading.RLock()
        self._local = threading.local()

        # Current session tracking (in-memory)
        self._session_records: list[UsageRecord] = []
        self._session_start = datetime.now(UTC)

        if self._enabled:
            self._init_db()
            logger.info(f"CostTracker initialized: db={self._db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    latency_ms INTEGER,
                    session_id TEXT,
                    user_id TEXT,
                    metadata TEXT
                )
            """)

            # Create indices
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON usage_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_provider_model
                ON usage_records(provider, model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_session
                ON usage_records(session_id)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def record(
        self,
        provider: str,
        model: str,
        operation: str = "chat",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_card_pricing: dict[str, float] | None = None,
    ) -> UsageRecord:
        """Record a single API usage event.

        Args:
            provider: Provider name (e.g., "openai").
            model: Model identifier (e.g., "gpt-4o").
            operation: Operation type (e.g., "chat", "embedding").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            latency_ms: Request latency in milliseconds.
            session_id: Session identifier for grouping.
            user_id: User identifier.
            metadata: Additional metadata.
            model_card_pricing: Optional pricing from model card.

        Returns:
            UsageRecord with computed cost.
        """
        if not self._enabled:
            # Return a record but don't persist
            return UsageRecord(
                provider=provider,
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Calculate cost
        input_price = get_price_per_million_tokens(provider, model, "input", model_card_pricing)
        output_price = get_price_per_million_tokens(provider, model, "output", model_card_pricing)

        estimated_cost = (input_tokens / 1_000_000) * input_price + (
            output_tokens / 1_000_000
        ) * output_price

        record = UsageRecord(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=round(estimated_cost, 8),
            latency_ms=latency_ms,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        # Store in database
        with self._lock:
            self._session_records.append(record)

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO usage_records
                    (id, timestamp, provider, model, operation, input_tokens, output_tokens,
                     total_tokens, estimated_cost_usd, latency_ms, session_id, user_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        int(record.timestamp.timestamp()),
                        record.provider,
                        record.model,
                        record.operation,
                        record.input_tokens,
                        record.output_tokens,
                        record.total_tokens,
                        record.estimated_cost_usd,
                        record.latency_ms,
                        record.session_id,
                        record.user_id,
                        json.dumps(record.metadata) if record.metadata else None,
                    ),
                )
                conn.commit()

        if self._config.log_to_console:
            logger.info(
                f"API Usage: {provider}/{model} {operation} "
                f"tokens={record.total_tokens} cost=${record.estimated_cost_usd:.6f}"
            )

        return record

    def get_daily_summary(
        self,
        target_date: date | None = None,
    ) -> UsageSummary:
        """Get usage summary for a specific day.

        Args:
            target_date: Date to summarize (default: today).

        Returns:
            UsageSummary for the day.
        """
        if target_date is None:
            target_date = date.today()

        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        return self._get_summary(start, end)

    def get_weekly_summary(
        self,
        weeks_ago: int = 0,
    ) -> UsageSummary:
        """Get usage summary for a week.

        Args:
            weeks_ago: Number of weeks in the past (0 = current week).

        Returns:
            UsageSummary for the week.
        """
        today = date.today()
        # Start of week (Monday)
        start_of_week = today - timedelta(days=today.weekday() + 7 * weeks_ago)
        end_of_week = start_of_week + timedelta(days=6)

        start = datetime.combine(start_of_week, datetime.min.time())
        end = datetime.combine(end_of_week, datetime.max.time())

        return self._get_summary(start, end)

    def get_monthly_summary(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> UsageSummary:
        """Get usage summary for a month.

        Args:
            year: Year (default: current year).
            month: Month 1-12 (default: current month).

        Returns:
            UsageSummary for the month.
        """
        today = date.today()
        year = year or today.year
        month = month or today.month

        start = datetime(year, month, 1)
        # Last day of month
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        return self._get_summary(start, end)

    def get_summary_by_provider(
        self,
        days: int = 30,
    ) -> dict[str, dict[str, Any]]:
        """Get usage breakdown by provider.

        Args:
            days: Number of days to include.

        Returns:
            Dictionary mapping provider to usage stats.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    provider,
                    COUNT(*) as call_count,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(estimated_cost_usd) as total_cost
                FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY provider
                ORDER BY total_cost DESC
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )

            result = {}
            for row in cursor:
                result[row["provider"]] = {
                    "call_count": row["call_count"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "total_tokens": row["total_tokens"],
                    "total_cost_usd": round(row["total_cost"], 6),
                }

            return result

    def get_summary_by_model(
        self,
        days: int = 30,
        provider: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get usage breakdown by model.

        Args:
            days: Number of days to include.
            provider: Filter by provider (optional).

        Returns:
            Dictionary mapping model to usage stats.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)

        query = """
            SELECT
                provider,
                model,
                COUNT(*) as call_count,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost
            FROM usage_records
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: list[Any] = [int(start.timestamp()), int(end.timestamp())]

        if provider:
            query += " AND provider = ?"
            params.append(provider)

        query += " GROUP BY provider, model ORDER BY total_cost DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)

            result = {}
            for row in cursor:
                key = f"{row['provider']}/{row['model']}"
                result[key] = {
                    "provider": row["provider"],
                    "model": row["model"],
                    "call_count": row["call_count"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "total_tokens": row["total_tokens"],
                    "total_cost_usd": round(row["total_cost"], 6),
                }

            return result

    def _get_summary(
        self,
        start: datetime,
        end: datetime,
    ) -> UsageSummary:
        """Get usage summary for a time range.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            UsageSummary for the period.
        """
        with self._get_connection() as conn:
            # Get totals
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as call_count,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                    AVG(latency_ms) as avg_latency
                FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )
            row = cursor.fetchone()

            summary = UsageSummary(
                period_start=start,
                period_end=end,
                total_calls=row["call_count"] or 0,
                total_input_tokens=row["input_tokens"] or 0,
                total_output_tokens=row["output_tokens"] or 0,
                total_tokens=row["total_tokens"] or 0,
                total_cost_usd=round(row["total_cost"] or 0, 6),
                avg_latency_ms=round(row["avg_latency"], 2) if row["avg_latency"] else None,
            )

            # Get by provider
            cursor = conn.execute(
                """
                SELECT provider, COUNT(*) as count, SUM(estimated_cost_usd) as cost
                FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY provider
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )
            for row in cursor:
                summary.by_provider[row["provider"]] = {
                    "count": row["count"],
                    "cost": round(row["cost"], 6),
                }

            # Get by model
            cursor = conn.execute(
                """
                SELECT model, COUNT(*) as count, SUM(estimated_cost_usd) as cost
                FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY model
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )
            for row in cursor:
                summary.by_model[row["model"]] = {
                    "count": row["count"],
                    "cost": round(row["cost"], 6),
                }

            # Get by operation
            cursor = conn.execute(
                """
                SELECT operation, COUNT(*) as count, SUM(estimated_cost_usd) as cost
                FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY operation
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )
            for row in cursor:
                summary.by_operation[row["operation"]] = {
                    "count": row["count"],
                    "cost": round(row["cost"], 6),
                }

            return summary

    def get_session_summary(self) -> UsageSummary:
        """Get summary for the current session (since tracker initialization).

        Returns:
            UsageSummary for current session.
        """
        summary = UsageSummary(
            period_start=self._session_start,
            period_end=datetime.now(UTC),
        )

        for record in self._session_records:
            summary.total_calls += 1
            summary.total_input_tokens += record.input_tokens
            summary.total_output_tokens += record.output_tokens
            summary.total_tokens += record.total_tokens
            summary.total_cost_usd += record.estimated_cost_usd

            # By provider
            if record.provider not in summary.by_provider:
                summary.by_provider[record.provider] = {"count": 0, "cost": 0.0}
            summary.by_provider[record.provider]["count"] += 1
            summary.by_provider[record.provider]["cost"] += record.estimated_cost_usd

            # By model
            if record.model not in summary.by_model:
                summary.by_model[record.model] = {"count": 0, "cost": 0.0}
            summary.by_model[record.model]["count"] += 1
            summary.by_model[record.model]["cost"] += record.estimated_cost_usd

        summary.total_cost_usd = round(summary.total_cost_usd, 6)
        return summary

    def cleanup_old_records(self) -> int:
        """Remove records older than retention period.

        Returns:
            Number of records deleted.
        """
        if self._config.retention_days <= 0:
            return 0

        cutoff = datetime.now(UTC) - timedelta(days=self._config.retention_days)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM usage_records WHERE timestamp < ?",
                    (int(cutoff.timestamp()),),
                )
                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old cost tracking records")

                return deleted

    def export_to_json(
        self,
        filepath: str,
        days: int = 30,
    ) -> int:
        """Export records to JSON file.

        Args:
            filepath: Output file path.
            days: Number of days to export.

        Returns:
            Number of records exported.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM usage_records
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                """,
                (int(start.timestamp()), int(end.timestamp())),
            )

            records = []
            for row in cursor:
                records.append(
                    {
                        "id": row["id"],
                        "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                        "provider": row["provider"],
                        "model": row["model"],
                        "operation": row["operation"],
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "total_tokens": row["total_tokens"],
                        "estimated_cost_usd": row["estimated_cost_usd"],
                        "latency_ms": row["latency_ms"],
                        "session_id": row["session_id"],
                        "user_id": row["user_id"],
                    }
                )

        output_path = Path(filepath).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        logger.info(f"Exported {len(records)} records to {output_path}")
        return len(records)

    @property
    def enabled(self) -> bool:
        """Return whether cost tracking is enabled."""
        return self._enabled

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_cost_tracker(
    config: dict[str, Any] | None = None,
) -> CostTracker:
    """Create a CostTracker from a configuration dictionary.

    Args:
        config: Configuration dictionary (typically from confy config).
                Expected keys match CostTrackingConfig attributes.

    Returns:
        Configured CostTracker instance.

    Example:
        config = llmcore_config.get("observability.cost_tracking", {})
        tracker = create_cost_tracker(config)
    """
    if config is None:
        config = {}

    tracking_config = CostTrackingConfig(**config)
    return CostTracker(config=tracking_config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PRICING_DATA",
    "CostTracker",
    "CostTrackingConfig",
    "OperationType",
    "UsageRecord",
    "UsageSummary",
    "create_cost_tracker",
    "get_price_per_million_tokens",
]
