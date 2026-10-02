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


def _cache_tokens(usage: dict[str, Any]) -> tuple[int, int, bool]:
    """Read cache counters, and say whether the prompt count excludes them.

    Returns ``(cached, written, prompt_excludes_cached)``.

    The two provider families disagree about what a prompt-token count
    means, and the disagreement is silent:

    * **Anthropic-style** reports ``input_tokens`` as *fresh only*, with
      ``cache_read_input_tokens`` and ``cache_creation_input_tokens``
      alongside. Observed on real traffic, ``input_tokens`` can be ~6 while
      cache reads are ~922,000 — so pricing the prompt count alone misses
      almost the entire bill.
    * **OpenAI-style** reports ``prompt_tokens`` as the whole prompt, with
      the cached part broken out under ``prompt_tokens_details``. Pricing
      that count at the fresh-input rate overcharges every cached token.

    Getting this backwards is not a rounding error in either direction: one
    way understates a cached agent turn by orders of magnitude, the other
    overstates it by up to ~15x. The provider's own key names are the only
    reliable signal of which contract applies, so they are what we read.
    """
    cached = _coerce_tokens(usage.get("cache_read_input_tokens"))
    written = _coerce_tokens(usage.get("cache_creation_input_tokens"))
    if cached or written or "input_tokens" in usage:
        # Anthropic-style: the prompt count is fresh tokens only.
        return cached, written, True

    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _coerce_tokens(details.get("cached_tokens"))
    # Some OpenAI-compatible gateways flatten it.
    if not cached:
        cached = _coerce_tokens(usage.get("cached_tokens"))
    written = written or _coerce_tokens(usage.get("cache_creation_tokens"))
    return cached, written, False


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

    Prompt caching is accounted for, which matters more than it sounds:
    the circuit breaker's ``COST_LIMIT`` acts on this number, and on
    cache-heavy agent traffic cache reads are the overwhelming majority of
    input tokens. See :func:`_cache_tokens` for the two incompatible
    provider conventions and what each one gets wrong when ignored.

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

    cached_tokens, cache_write_tokens, prompt_excludes_cached = _cache_tokens(usage)
    if prompt_excludes_cached and cached_tokens:
        # Normalise to this codebase's contract: prompt_tokens is the whole
        # prompt and cached_tokens is a subset of it. Without this the
        # cached bulk never reaches pricing at all.
        prompt_tokens += cached_tokens
        if total_tokens:
            total_tokens += cached_tokens
    cached_tokens = min(cached_tokens, prompt_tokens)

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
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cache_write_tokens=cache_write_tokens,
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
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=cost,
        provider=provider_str,
        model=model_str,
    )


__all__ = ["PhaseUsage", "extract_usage"]
