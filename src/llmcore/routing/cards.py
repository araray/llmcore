# src/llmcore/routing/cards.py
"""Model-card lookups that routing decisions depend on.

Two strategies and one failover rule need facts about a model rather than
observations of it:

* ``lowest_cost`` needs per-token pricing.
* ``ContextLengthError`` failover needs context windows, so an overflowing
  prompt can be sent to a bigger model instead of simply failing.
* A pre-call window check can skip a target that provably cannot hold the
  prompt, which is cheaper than finding out from the vendor.

llmcore already ships these facts in its bundled model cards, so this module
is a thin, defensive adapter over
:mod:`llmcore.model_cards` — not a second source of truth.

Everything here returns ``None`` when a fact is unknown, and every caller must
treat unknown as "no opinion" rather than as zero. A missing card is normal:
users point llmcore at models that ship no card at all (a fresh Ollama pull, a
private vLLM checkpoint), and those must stay routable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Target

logger = logging.getLogger(__name__)

__all__ = [
    "SELF_HOSTED_PROVIDERS",
    "canonical_card_provider",
    "context_window",
    "estimate_cost_usd",
    "lookup_card",
]

#: Providers whose inference llmcore cannot be billed per token for, because
#: the endpoint is the user's own. Their cards carry no ``pricing`` block, and
#: treating that absence as *unknown* would make ``lowest_cost`` rank a local
#: model behind a paid API rather than ahead of it -- exactly backwards.
#:
#: This says "no per-token vendor charge", not "free to run": GPU rental and
#: electricity are real, but they are not per-token costs, so llmcore has no
#: honest number to put against them.
SELF_HOSTED_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm"})

#: Provider type -> the name its model cards are filed under.
#:
#: This is **not** the same mapping as ``_PROVIDER_INSTANCE_ALIASES`` in the
#: provider manager, and conflating the two is a bug this module shipped with:
#: that map folds vendor aliases onto a canonical *provider type*
#: (``google`` -> ``gemini``), while cards are filed by *vendor directory*,
#: where the same pair runs the other way (``gemini`` -> ``google``). Pointing
#: a Gemini target at a ``gemini`` card namespace that does not exist silently
#: disabled pricing and the context-window check for one of the most used
#: providers in the library, with no error anywhere -- the lookup just returned
#: None, and None means "unknown", which every caller handles quietly.
#:
#: ``tests/routing/test_cards.py`` now asserts that every provider type with
#: cards resolves to a namespace that exists, so the next provider added cannot
#: reintroduce it.
_CARD_PROVIDER_ALIASES: dict[str, str] = {
    # Provider type is `gemini`; the cards live in `google/`.
    "gemini": "google",
    "moonshot": "kimi",
    "glm": "zai",
    "zhipu": "zai",
    "zhipuai": "zai",
    "bigmodel": "zai",
    "jev": "typesafe",
    "fal_ai": "fal",
    "eleven_labs": "elevenlabs",
    "friendliai": "friendli",
    "friendli_ai": "friendli",
}


def canonical_card_provider(provider: str) -> str:
    """Map a provider or vendor alias to the name its cards are filed under."""
    key = provider.strip().lower()
    return _CARD_PROVIDER_ALIASES.get(key, key)


def lookup_card(target: Target, *, provider_type: str | None = None) -> Any | None:
    """Return the :class:`~llmcore.model_cards.ModelCard` for ``target``.

    Args:
        target: The target whose model to look up.
        provider_type: The provider *type*, when ``target.provider`` is a
            configured instance name (``my-openai``) rather than a type. A
            card is filed under the type, so routing passes the resolved type
            where it knows it.

    Returns:
        The card, or ``None`` when the model has none — which is not an error.
    """
    if target.model is None:
        return None
    provider = canonical_card_provider(provider_type or target.provider)
    try:
        from ..model_cards import get_model_card_registry

        return get_model_card_registry().get(provider, target.model)
    except Exception:  # pragma: no cover - a card store problem must not break routing
        logger.debug("model-card lookup failed for %s", target.key, exc_info=True)
        return None


def context_window(target: Target, *, provider_type: str | None = None) -> int | None:
    """Return ``target``'s context window in tokens, or ``None`` if unknown."""
    card = lookup_card(target, provider_type=provider_type)
    if card is None:
        return None
    try:
        window = card.get_context_length()
    except Exception:  # pragma: no cover - defensive
        return None
    return int(window) if window else None


def estimate_cost_usd(
    target: Target,
    *,
    input_tokens: int,
    output_tokens: int = 0,
    provider_type: str | None = None,
) -> float | None:
    """Estimate what one call to ``target`` would cost, in USD.

    Returns ``0.0`` for a self-hosted provider (see
    :data:`SELF_HOSTED_PROVIDERS`), and ``None`` when the model has a card but
    no pricing — unknown, which callers must not read as free.

    Non-USD pricing is returned as-is rather than converted: inventing an
    exchange rate would be worse than comparing two prices in the same
    currency, which is the overwhelmingly common case. A pool mixing
    currencies gets a debug log and a best-effort comparison.
    """
    provider = canonical_card_provider(provider_type or target.provider)
    if provider in SELF_HOSTED_PROVIDERS:
        return 0.0

    card = lookup_card(target, provider_type=provider_type)
    pricing = getattr(card, "pricing", None) if card is not None else None
    per_million = getattr(pricing, "per_million_tokens", None)
    if per_million is None:
        return None

    input_price = getattr(per_million, "input", None)
    output_price = getattr(per_million, "output", None)
    if input_price is None and output_price is None:
        return None

    currency = (getattr(pricing, "currency", "USD") or "USD").upper()
    if currency != "USD":
        logger.debug(
            "%s is priced in %s; comparing it against USD targets without conversion",
            target.key,
            currency,
        )

    cost = (input_tokens / 1_000_000) * float(input_price or 0.0)
    if output_tokens:
        cost += (output_tokens / 1_000_000) * float(output_price or input_price or 0.0)
    return cost
