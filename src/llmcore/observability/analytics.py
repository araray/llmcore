# src/llmcore/observability/analytics.py
"""
Advanced Cost Analytics for LLMCore.

This module provides sophisticated cost analysis capabilities including:

- Period comparisons (week-over-week, month-over-month)
- Trend analysis and forecasting
- Budget tracking and alerts
- Cost anomaly detection
- Usage pattern analysis

Architecture:
    CostAnalyzer wraps CostTracker to provide higher-level analytics.
    It queries the cost database and performs statistical analysis
    to derive insights about usage patterns and cost trends.

Usage:
    >>> from llmcore.observability.analytics import CostAnalyzer
    >>> from llmcore.observability.cost_tracker import CostTracker
    >>>
    >>> tracker = CostTracker()
    >>> analyzer = CostAnalyzer(tracker)
    >>>
    >>> # Get week-over-week comparison
    >>> comparison = analyzer.compare_periods("week")
    >>> print(f"Cost change: {comparison.change_percent:.1f}%")
    >>>
    >>> # Get trend forecast
    >>> forecast = analyzer.forecast_cost(days=30)
    >>> print(f"Projected 30-day cost: ${forecast.projected_cost:.2f}")
    >>>
    >>> # Check budget status
    >>> budget = analyzer.check_budget(monthly_budget=100.0)
    >>> print(f"Budget used: {budget.percent_used:.1f}%")

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 9
    - RAG_ECOSYSTEM_REDESIGN_SPEC.md Section 4.2 (Cost Tracking)
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from statistics import linear_regression, mean, stdev
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .cost_tracker import CostTracker, UsageSummary


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Period(str, Enum):
    """Time periods for analysis."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TrendDirection(str, Enum):
    """Direction of a trend."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class BudgetStatus(str, Enum):
    """Budget utilization status."""

    OK = "ok"
    WARNING = "warning"  # 80%+ used
    CRITICAL = "critical"  # 95%+ used
    EXCEEDED = "exceeded"  # 100%+ used


class AnomalyType(str, Enum):
    """Types of cost anomalies."""

    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    UNUSUAL_PROVIDER = "unusual_provider"  # Unusual provider usage
    UNUSUAL_MODEL = "unusual_model"  # Unusual model usage


# =============================================================================
# DATA MODELS
# =============================================================================


class PeriodComparison(BaseModel):
    """Comparison between two time periods."""

    current_period: str
    previous_period: str
    current_cost: float
    previous_cost: float
    change_amount: float
    change_percent: float
    current_calls: int
    previous_calls: int
    calls_change_percent: float

    # Breakdowns
    by_provider: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)


class TrendAnalysis(BaseModel):
    """Analysis of cost trends."""

    period: str
    data_points: int
    direction: TrendDirection
    slope: float  # Cost change per day
    r_squared: float  # Goodness of fit (0-1)
    daily_costs: list[tuple[str, float]] = Field(default_factory=list)

    # Interpretation
    interpretation: str = ""


class CostForecast(BaseModel):
    """Cost forecast for future period."""

    forecast_days: int
    method: str = "linear_regression"
    projected_cost: float
    confidence_level: float  # 0-1
    lower_bound: float  # Conservative estimate
    upper_bound: float  # High estimate

    # Daily projections
    daily_forecast: list[tuple[str, float]] = Field(default_factory=list)

    # Assumptions
    assumptions: list[str] = Field(default_factory=list)


class BudgetAnalysis(BaseModel):
    """Budget utilization analysis."""

    budget_amount: float
    budget_period: str  # "daily", "weekly", "monthly"
    period_start: datetime
    period_end: datetime

    amount_used: float
    amount_remaining: float
    percent_used: float
    status: BudgetStatus

    # Projection
    days_elapsed: int
    days_remaining: int
    projected_end_of_period: float
    will_exceed: bool

    # Recommendations
    recommendations: list[str] = Field(default_factory=list)


class CostAnomaly(BaseModel):
    """A detected cost anomaly."""

    anomaly_type: AnomalyType
    timestamp: datetime
    severity: str  # "low", "medium", "high"
    description: str

    # Context
    expected_value: float
    actual_value: float
    deviation_percent: float

    # Affected dimensions
    provider: str | None = None
    model: str | None = None


class UsagePattern(BaseModel):
    """Analysis of usage patterns."""

    analysis_period_days: int

    # Time-based patterns
    peak_hours: list[int] = Field(default_factory=list)  # 0-23
    peak_days: list[str] = Field(default_factory=list)  # Monday, Tuesday, etc.

    # Volume patterns
    avg_daily_calls: float = 0.0
    avg_daily_tokens: float = 0.0
    avg_daily_cost: float = 0.0

    # Provider/model patterns
    most_used_provider: str | None = None
    most_used_model: str | None = None
    most_expensive_provider: str | None = None
    most_expensive_model: str | None = None

    # Efficiency metrics
    avg_tokens_per_call: float = 0.0
    avg_cost_per_call: float = 0.0
    avg_cost_per_token: float = 0.0


class AnalyticsSummary(BaseModel):
    """Comprehensive analytics summary."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    analysis_period_days: int

    # Cost overview
    total_cost: float
    total_calls: int
    total_tokens: int

    # Comparisons
    week_over_week: PeriodComparison | None = None
    month_over_month: PeriodComparison | None = None

    # Trends
    trend: TrendAnalysis | None = None
    forecast: CostForecast | None = None

    # Patterns and anomalies
    usage_pattern: UsagePattern | None = None
    anomalies: list[CostAnomaly] = Field(default_factory=list)

    # Budget (if configured)
    budget: BudgetAnalysis | None = None


# =============================================================================
# COST ANALYZER
# =============================================================================


class CostAnalyzer:
    """
    Advanced cost analytics engine.

    Provides sophisticated analysis capabilities on top of CostTracker data.

    Args:
        tracker: CostTracker instance to analyze.
        anomaly_threshold: Standard deviations for anomaly detection (default: 2).
    """

    def __init__(
        self,
        tracker: CostTracker,
        anomaly_threshold: float = 2.0,
    ):
        self.tracker = tracker
        self.anomaly_threshold = anomaly_threshold

    # =========================================================================
    # PERIOD COMPARISONS
    # =========================================================================

    def compare_periods(
        self,
        period: str | Period = "week",
    ) -> PeriodComparison:
        """
        Compare current period with previous period.

        Args:
            period: Period to compare ("day", "week", "month").

        Returns:
            PeriodComparison with cost and usage changes.
        """
        if isinstance(period, str):
            period = Period(period)

        today = date.today()

        if period == Period.DAY:
            current_start = today
            previous_start = today - timedelta(days=1)
            period_days = 1
        elif period == Period.WEEK:
            # Start of current week (Monday)
            current_start = today - timedelta(days=today.weekday())
            previous_start = current_start - timedelta(days=7)
            period_days = 7
        elif period == Period.MONTH:
            current_start = today.replace(day=1)
            # Previous month
            if today.month == 1:
                previous_start = date(today.year - 1, 12, 1)
            else:
                previous_start = date(today.year, today.month - 1, 1)
            period_days = 30  # Approximate
        else:
            raise ValueError(f"Unsupported period: {period}")

        # Get summaries for both periods
        current_summary = self._get_period_summary(
            current_start, current_start + timedelta(days=period_days - 1)
        )
        previous_summary = self._get_period_summary(
            previous_start, previous_start + timedelta(days=period_days - 1)
        )

        # Calculate changes
        current_cost = current_summary.total_cost_usd
        previous_cost = previous_summary.total_cost_usd
        change_amount = current_cost - previous_cost
        change_percent = (change_amount / previous_cost * 100) if previous_cost > 0 else 0.0

        current_calls = current_summary.total_calls
        previous_calls = previous_summary.total_calls
        calls_change = (
            ((current_calls - previous_calls) / previous_calls * 100) if previous_calls > 0 else 0.0
        )

        return PeriodComparison(
            current_period=f"{current_start} to {current_start + timedelta(days=period_days - 1)}",
            previous_period=f"{previous_start} to {previous_start + timedelta(days=period_days - 1)}",
            current_cost=round(current_cost, 4),
            previous_cost=round(previous_cost, 4),
            change_amount=round(change_amount, 4),
            change_percent=round(change_percent, 2),
            current_calls=current_calls,
            previous_calls=previous_calls,
            calls_change_percent=round(calls_change, 2),
            by_provider=self._compare_by_dimension(current_summary, previous_summary, "provider"),
            by_model=self._compare_by_dimension(current_summary, previous_summary, "model"),
        )

    def _get_period_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> UsageSummary:
        """Get summary for a date range."""
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())
        return self.tracker._get_summary(start, end)

    def _compare_by_dimension(
        self,
        current: UsageSummary,
        previous: UsageSummary,
        dimension: str,
    ) -> dict[str, dict[str, float]]:
        """Compare costs by a specific dimension (provider, model)."""
        result = {}

        current_data = current.by_provider if dimension == "provider" else current.by_model
        previous_data = previous.by_provider if dimension == "provider" else previous.by_model

        all_keys = set(current_data.keys()) | set(previous_data.keys())

        for key in all_keys:
            current_cost = current_data.get(key, {}).get("cost", 0.0)
            previous_cost = previous_data.get(key, {}).get("cost", 0.0)
            change = current_cost - previous_cost
            change_pct = (change / previous_cost * 100) if previous_cost > 0 else 0.0

            result[key] = {
                "current": round(current_cost, 4),
                "previous": round(previous_cost, 4),
                "change": round(change, 4),
                "change_percent": round(change_pct, 2),
            }

        return result

    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================

    def analyze_trend(
        self,
        days: int = 30,
    ) -> TrendAnalysis:
        """
        Analyze cost trends over a period.

        Args:
            days: Number of days to analyze.

        Returns:
            TrendAnalysis with direction, slope, and interpretation.
        """
        # Get daily costs
        daily_costs = self._get_daily_costs(days)

        if len(daily_costs) < 2:
            return TrendAnalysis(
                period=f"Last {days} days",
                data_points=len(daily_costs),
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                daily_costs=daily_costs,
                interpretation="Insufficient data for trend analysis.",
            )

        # Perform linear regression
        x_values = list(range(len(daily_costs)))
        y_values = [cost for _, cost in daily_costs]

        try:
            slope, intercept = linear_regression(x_values, y_values)

            # Calculate R-squared
            y_mean = mean(y_values)
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            slope = 0.0
            r_squared = 0.0

        # Determine direction
        if abs(slope) < 0.01:  # Less than 1 cent per day
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Generate interpretation
        if direction == TrendDirection.STABLE:
            interpretation = "Costs are stable with minimal day-to-day variation."
        elif direction == TrendDirection.INCREASING:
            daily_increase = slope
            monthly_increase = slope * 30
            interpretation = (
                f"Costs are increasing by approximately ${daily_increase:.4f}/day "
                f"(${monthly_increase:.2f}/month projected increase)."
            )
        else:
            daily_decrease = abs(slope)
            monthly_decrease = abs(slope) * 30
            interpretation = (
                f"Costs are decreasing by approximately ${daily_decrease:.4f}/day "
                f"(${monthly_decrease:.2f}/month projected decrease)."
            )

        return TrendAnalysis(
            period=f"Last {days} days",
            data_points=len(daily_costs),
            direction=direction,
            slope=round(slope, 6),
            r_squared=round(r_squared, 4),
            daily_costs=daily_costs,
            interpretation=interpretation,
        )

    def _get_daily_costs(self, days: int) -> list[tuple[str, float]]:
        """Get cost data aggregated by day."""
        daily_costs = []
        today = date.today()

        for i in range(days - 1, -1, -1):
            day = today - timedelta(days=i)
            summary = self.tracker.get_daily_summary(day)
            daily_costs.append((day.isoformat(), summary.total_cost_usd))

        return daily_costs

    # =========================================================================
    # FORECASTING
    # =========================================================================

    def forecast_cost(
        self,
        days: int = 30,
        history_days: int = 30,
        confidence_level: float = 0.9,
    ) -> CostForecast:
        """
        Forecast future costs based on historical data.

        Args:
            days: Number of days to forecast.
            history_days: Days of history to use for prediction.
            confidence_level: Confidence level for bounds (0-1).

        Returns:
            CostForecast with projected costs and bounds.
        """
        # Get historical data
        daily_costs = self._get_daily_costs(history_days)

        if len(daily_costs) < 7:
            return CostForecast(
                forecast_days=days,
                method="insufficient_data",
                projected_cost=0.0,
                confidence_level=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                assumptions=["Insufficient historical data for forecasting."],
            )

        # Perform linear regression
        x_values = list(range(len(daily_costs)))
        y_values = [cost for _, cost in daily_costs]

        try:
            slope, intercept = linear_regression(x_values, y_values)
        except Exception:
            # Fallback to simple average
            avg_daily = mean(y_values)
            return CostForecast(
                forecast_days=days,
                method="average",
                projected_cost=round(avg_daily * days, 4),
                confidence_level=0.5,
                lower_bound=round(avg_daily * days * 0.7, 4),
                upper_bound=round(avg_daily * days * 1.3, 4),
                assumptions=["Using simple average due to regression failure."],
            )

        # Calculate daily forecast
        daily_forecast = []
        today = date.today()
        projected_total = 0.0

        for i in range(days):
            future_x = len(daily_costs) + i
            projected_day_cost = max(0, slope * future_x + intercept)
            projected_total += projected_day_cost

            forecast_date = today + timedelta(days=i + 1)
            daily_forecast.append((forecast_date.isoformat(), round(projected_day_cost, 4)))

        # Calculate bounds based on historical variance
        try:
            historical_stdev = stdev(y_values) if len(y_values) > 1 else 0.0
        except Exception:
            historical_stdev = 0.0

        # Z-score for confidence level (approximately)
        z_scores = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.645)

        margin = z * historical_stdev * (days**0.5)  # Error grows with sqrt of days

        lower_bound = max(0, projected_total - margin)
        upper_bound = projected_total + margin

        return CostForecast(
            forecast_days=days,
            method="linear_regression",
            projected_cost=round(projected_total, 4),
            confidence_level=confidence_level,
            lower_bound=round(lower_bound, 4),
            upper_bound=round(upper_bound, 4),
            daily_forecast=daily_forecast,
            assumptions=[
                f"Based on {history_days} days of historical data.",
                "Assumes current usage patterns continue.",
                f"Trend slope: ${slope:.4f}/day.",
            ],
        )

    # =========================================================================
    # BUDGET ANALYSIS
    # =========================================================================

    def check_budget(
        self,
        budget_amount: float,
        budget_period: str = "monthly",
    ) -> BudgetAnalysis:
        """
        Check budget utilization.

        Args:
            budget_amount: Budget amount in USD.
            budget_period: "daily", "weekly", or "monthly".

        Returns:
            BudgetAnalysis with utilization and projections.
        """
        today = date.today()
        now = datetime.now(UTC)

        # Determine period boundaries
        if budget_period == "daily":
            period_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=UTC)
            period_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=UTC)
            days_total = 1
        elif budget_period == "weekly":
            # Week starts on Monday
            week_start = today - timedelta(days=today.weekday())
            period_start = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=UTC)
            period_end = period_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            days_total = 7
        else:  # monthly
            period_start = datetime.combine(today.replace(day=1), datetime.min.time()).replace(
                tzinfo=UTC
            )
            # Last day of month
            if today.month == 12:
                next_month = date(today.year + 1, 1, 1)
            else:
                next_month = date(today.year, today.month + 1, 1)
            period_end = datetime.combine(
                next_month - timedelta(days=1), datetime.max.time()
            ).replace(tzinfo=UTC)
            days_total = (period_end.date() - period_start.date()).days + 1

        # Get current usage
        summary = self.tracker._get_summary(period_start, now)
        amount_used = summary.total_cost_usd
        amount_remaining = max(0, budget_amount - amount_used)
        percent_used = (amount_used / budget_amount * 100) if budget_amount > 0 else 0.0

        # Determine status
        if percent_used >= 100:
            status = BudgetStatus.EXCEEDED
        elif percent_used >= 95:
            status = BudgetStatus.CRITICAL
        elif percent_used >= 80:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.OK

        # Calculate projections
        days_elapsed = (now.date() - period_start.date()).days + 1
        days_remaining = days_total - days_elapsed

        if days_elapsed > 0:
            daily_rate = amount_used / days_elapsed
            projected_end = daily_rate * days_total
        else:
            daily_rate = 0.0
            projected_end = amount_used

        will_exceed = projected_end > budget_amount

        # Generate recommendations
        recommendations = []
        if status == BudgetStatus.EXCEEDED:
            recommendations.append("Budget exceeded. Consider increasing budget or reducing usage.")
        elif status == BudgetStatus.CRITICAL:
            recommendations.append("Budget nearly exhausted. Monitor usage closely.")
        elif will_exceed:
            recommendations.append(
                f"At current rate, projected to use ${projected_end:.2f} "
                f"(${projected_end - budget_amount:.2f} over budget)."
            )

        if daily_rate > budget_amount / days_total:
            target_daily = budget_amount / days_total
            reduction_needed = ((daily_rate - target_daily) / daily_rate) * 100
            recommendations.append(
                f"Daily usage (${daily_rate:.2f}) exceeds budget rate "
                f"(${target_daily:.2f}). Consider {reduction_needed:.0f}% reduction."
            )

        return BudgetAnalysis(
            budget_amount=budget_amount,
            budget_period=budget_period,
            period_start=period_start,
            period_end=period_end,
            amount_used=round(amount_used, 4),
            amount_remaining=round(amount_remaining, 4),
            percent_used=round(percent_used, 2),
            status=status,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            projected_end_of_period=round(projected_end, 4),
            will_exceed=will_exceed,
            recommendations=recommendations,
        )

    # =========================================================================
    # ANOMALY DETECTION
    # =========================================================================

    def detect_anomalies(
        self,
        days: int = 30,
    ) -> list[CostAnomaly]:
        """
        Detect cost anomalies in recent data.

        Args:
            days: Number of days to analyze.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        # Get daily costs
        daily_costs = self._get_daily_costs(days)

        if len(daily_costs) < 7:
            return anomalies

        # Calculate statistics
        costs = [cost for _, cost in daily_costs]
        avg_cost = mean(costs)
        cost_stdev = stdev(costs) if len(costs) > 1 else 0.0

        if cost_stdev == 0:
            return anomalies

        # Check each day for anomalies
        for date_str, cost in daily_costs[-7:]:  # Last week only
            z_score = (cost - avg_cost) / cost_stdev

            if abs(z_score) > self.anomaly_threshold:
                anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP
                severity = "high" if abs(z_score) > 3 else "medium"

                anomalies.append(
                    CostAnomaly(
                        anomaly_type=anomaly_type,
                        timestamp=datetime.fromisoformat(date_str).replace(tzinfo=UTC),
                        severity=severity,
                        description=(
                            f"{'Unusually high' if z_score > 0 else 'Unusually low'} "
                            f"cost on {date_str}: ${cost:.4f} "
                            f"(expected ~${avg_cost:.4f})"
                        ),
                        expected_value=round(avg_cost, 4),
                        actual_value=round(cost, 4),
                        deviation_percent=round((cost - avg_cost) / avg_cost * 100, 2),
                    )
                )

        return anomalies

    # =========================================================================
    # USAGE PATTERNS
    # =========================================================================

    def analyze_usage_patterns(
        self,
        days: int = 30,
    ) -> UsagePattern:
        """
        Analyze usage patterns over a period.

        Args:
            days: Number of days to analyze.

        Returns:
            UsagePattern with identified patterns.
        """
        # Get summaries for analysis
        summary = self.tracker.get_summary_by_provider(days=days)
        model_summary = self.tracker.get_summary_by_model(days=days)

        # Calculate daily averages
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        total_summary = self.tracker._get_summary(start, end)

        avg_daily_calls = total_summary.total_calls / days if days > 0 else 0.0
        avg_daily_tokens = total_summary.total_tokens / days if days > 0 else 0.0
        avg_daily_cost = total_summary.total_cost_usd / days if days > 0 else 0.0

        # Find most used and expensive
        most_used_provider = None
        most_used_model = None
        most_expensive_provider = None
        most_expensive_model = None

        max_calls = 0
        max_cost = 0.0

        for provider, data in summary.items():
            if data.get("call_count", 0) > max_calls:
                max_calls = data["call_count"]
                most_used_provider = provider
            if data.get("total_cost_usd", 0) > max_cost:
                max_cost = data["total_cost_usd"]
                most_expensive_provider = provider

        max_calls = 0
        max_cost = 0.0

        for key, data in model_summary.items():
            if data.get("call_count", 0) > max_calls:
                max_calls = data["call_count"]
                most_used_model = data.get("model", key)
            if data.get("total_cost_usd", 0) > max_cost:
                max_cost = data["total_cost_usd"]
                most_expensive_model = data.get("model", key)

        # Efficiency metrics
        avg_tokens_per_call = (
            total_summary.total_tokens / total_summary.total_calls
            if total_summary.total_calls > 0
            else 0.0
        )
        avg_cost_per_call = (
            total_summary.total_cost_usd / total_summary.total_calls
            if total_summary.total_calls > 0
            else 0.0
        )
        avg_cost_per_token = (
            total_summary.total_cost_usd / total_summary.total_tokens
            if total_summary.total_tokens > 0
            else 0.0
        )

        return UsagePattern(
            analysis_period_days=days,
            avg_daily_calls=round(avg_daily_calls, 2),
            avg_daily_tokens=round(avg_daily_tokens, 2),
            avg_daily_cost=round(avg_daily_cost, 4),
            most_used_provider=most_used_provider,
            most_used_model=most_used_model,
            most_expensive_provider=most_expensive_provider,
            most_expensive_model=most_expensive_model,
            avg_tokens_per_call=round(avg_tokens_per_call, 2),
            avg_cost_per_call=round(avg_cost_per_call, 6),
            avg_cost_per_token=round(avg_cost_per_token, 8),
        )

    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================

    def get_analytics_summary(
        self,
        days: int = 30,
        budget_amount: float | None = None,
        budget_period: str = "monthly",
    ) -> AnalyticsSummary:
        """
        Get comprehensive analytics summary.

        Args:
            days: Number of days for analysis.
            budget_amount: Optional budget to check against.
            budget_period: Budget period type.

        Returns:
            Complete AnalyticsSummary.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        total_summary = self.tracker._get_summary(start, end)

        summary = AnalyticsSummary(
            analysis_period_days=days,
            total_cost=round(total_summary.total_cost_usd, 4),
            total_calls=total_summary.total_calls,
            total_tokens=total_summary.total_tokens,
        )

        # Add comparisons
        try:
            summary.week_over_week = self.compare_periods(Period.WEEK)
        except Exception as e:
            logger.warning(f"Week-over-week comparison failed: {e}")

        try:
            summary.month_over_month = self.compare_periods(Period.MONTH)
        except Exception as e:
            logger.warning(f"Month-over-month comparison failed: {e}")

        # Add trend analysis
        try:
            summary.trend = self.analyze_trend(days)
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")

        # Add forecast
        try:
            summary.forecast = self.forecast_cost(days=30, history_days=days)
        except Exception as e:
            logger.warning(f"Forecasting failed: {e}")

        # Add usage patterns
        try:
            summary.usage_pattern = self.analyze_usage_patterns(days)
        except Exception as e:
            logger.warning(f"Usage pattern analysis failed: {e}")

        # Add anomalies
        try:
            summary.anomalies = self.detect_anomalies(days)
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        # Add budget analysis if configured
        if budget_amount is not None:
            try:
                summary.budget = self.check_budget(budget_amount, budget_period)
            except Exception as e:
                logger.warning(f"Budget analysis failed: {e}")

        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_cost_analyzer(
    tracker: CostTracker,
    anomaly_threshold: float = 2.0,
) -> CostAnalyzer:
    """
    Create a CostAnalyzer instance.

    Args:
        tracker: CostTracker to analyze.
        anomaly_threshold: Standard deviations for anomaly detection.

    Returns:
        Configured CostAnalyzer.
    """
    return CostAnalyzer(tracker, anomaly_threshold)
