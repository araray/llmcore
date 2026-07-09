# src/llmcore/agents/cognitive/phases/usage.py
"""Provider usage extraction + cost attribution for the cognitive phases (2.7).

Every phase LLM call captures a typed :class:`PhaseUsage` record (prompt/
completion/total tokens plus model-card cost) so iteration totals, agent
state totals, the circuit breaker's ``COST_LIMIT`` and session token stats
stop reporting zero for Darwin runs.

``PhaseUsage`` itself is defined in ``..models`` (the phase output models
reference it, and this package's ``__init__`` imports the phase modules —
defining it here would be a circular import); it is re-exported here so
``phases.usage`` remains the one import surface for usage handling.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..models import PhaseUsage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _pricing_for(provider: str, model: str) -> Any | None:
    """Memoized model-card pricing lookup (None when unknown)."""
    try:
        from ....model_cards import get_model_card_registry

        card = get_model_card_registry().get(provider, model)
    except Exception:
        logger.debug("Model card lookup failed for %s/%s", provider, model, exc_info=True)
        return None
    if card is None or card.pricing is None:
        return None
    return card.pricing


def _coerce_tokens(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def extract_usage(
    response: Any,
    provider_name: str | None,
    model: str | None,
) -> PhaseUsage | None:
    """Extract a :class:`PhaseUsage` from a provider chat response.

    Reads the OpenAI-style ``usage`` block (``prompt_tokens``/
    ``completion_tokens``/``total_tokens``); a missing total is computed
    from the parts. Cost comes from the model-card registry's pricing
    (memoized per provider+model); unknown models keep ``cost=None``.

    Args:
        response: Raw provider response (only dict shapes carry usage).
        provider_name: Provider name for pricing lookup (e.g. "openai").
        model: Model id for pricing lookup (e.g. "gpt-4o").

    Returns:
        A PhaseUsage, or None when the response carries no usage data.
    """
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = _coerce_tokens(usage.get("prompt_tokens"))
    completion_tokens = _coerce_tokens(usage.get("completion_tokens"))
    total_tokens = _coerce_tokens(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    provider_str = str(provider_name) if provider_name else None
    model_str = str(model) if model else None

    cost: float | None = None
    if provider_str and model_str:
        pricing = _pricing_for(provider_str, model_str)
        if pricing is not None:
            try:
                cost = pricing.get_cost(
                    input_tokens=prompt_tokens, output_tokens=completion_tokens
                )
            except Exception:
                logger.debug(
                    "Cost computation failed for %s/%s", provider_str, model_str,
                    exc_info=True,
                )

    return PhaseUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        provider=provider_str,
        model=model_str,
    )


__all__ = ["PhaseUsage", "extract_usage"]
