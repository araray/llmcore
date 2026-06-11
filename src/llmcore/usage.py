"""Per-call token-usage reporting for :meth:`llmcore.LLMCore.chat_with_usage`.

This module defines :class:`ChatUsage`, the small, immutable value object
returned alongside the assistant text by :meth:`LLMCore.chat_with_usage`.
It exposes the token counts that ``llmcore`` already computes internally
during a non-streaming ``chat()`` call (see
:class:`llmcore.models.ContextPreparationDetails`) so that *callers* can
meter usage without enabling session persistence.

Design notes:
    * **Additive / opt-in.** ``ChatUsage`` is a brand-new public type; the
      existing ``chat() -> str`` contract is untouched. Nothing in
      ``llmcore`` depends on this module, so the framework stays a clean,
      general-purpose library (no inversion of dependencies, no coupling
      to any downstream consumer).
    * **Dual naming.** Convergence's metering bridge (and other callers)
      prefer ``tokens_in`` / ``tokens_out``; ``llmcore``'s own
      introspection model uses ``prompt_tokens`` / ``completion_tokens``.
      ``ChatUsage`` carries the latter as fields and exposes the former as
      read-only aliases, so it is a drop-in for either convention.
    * **Graceful degradation.** When usage is unavailable (e.g. the
      per-call introspection details were never cached), every count is
      ``None``; downstream meters treat that as a no-op rather than
      recording a zero-token event.

References:
    Convergence v0.9 Spec — Part 3 §3.2 (the llmcore token-usage
    dependency; "Option A — usage-returning call"), §3.2.3 (the
    ``Usage(tokens_in, tokens_out)`` shape the bridge consumes).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ContextPreparationDetails

__all__ = ["ChatUsage"]


def _as_int_or_none(value: object) -> int | None:
    """Coerce a token count to ``int`` or ``None``.

    ``llmcore`` reports token counts as ``int`` (including a legitimate
    ``0`` for an empty completion). Anything that is ``None`` or not
    int-coercible degrades to ``None`` so a single malformed field maps to
    "no usage reported" rather than raising.

    Args:
        value: A candidate token count (``int``, ``None``, or other).

    Returns:
        The value as an ``int``, or ``None`` if it is ``None`` /
        non-coercible.
    """
    if value is None:
        return None
    if isinstance(value, int):  # also normalises bool -> 0/1
        return int(value)
    if isinstance(value, (str, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class ChatUsage:
    """Token usage for a single non-streaming completion.

    All counts are post-fallback / as-served: they describe the model that
    actually produced the response, not any requested-but-unavailable one.

    Attributes:
        prompt_tokens: Tokens in the prepared prompt/context, or ``None``
            when usage is unavailable.
        completion_tokens: Tokens in the assistant response, or ``None``.
        total_tokens: ``prompt_tokens + completion_tokens`` as computed by
            ``llmcore``, or ``None``.
        provider: The provider name that served the call (e.g.
            ``"openai"``), or ``None``.
        model: The model identifier that served the call, or ``None``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    provider: str | None = None
    model: str | None = None

    @property
    def tokens_in(self) -> int | None:
        """Alias for :attr:`prompt_tokens` (caller-side naming)."""
        return self.prompt_tokens

    @property
    def tokens_out(self) -> int | None:
        """Alias for :attr:`completion_tokens` (caller-side naming)."""
        return self.completion_tokens

    @property
    def is_available(self) -> bool:
        """``True`` when at least one of the token counts is present.

        Meters can branch on this to decide whether to record an event;
        when ``False`` the call should be treated as un-metered.
        """
        return self.prompt_tokens is not None or self.completion_tokens is not None

    @classmethod
    def from_context_details(
        cls, details: ContextPreparationDetails | None
    ) -> "ChatUsage":
        """Build a :class:`ChatUsage` from per-call introspection details.

        Args:
            details: The :class:`~llmcore.models.ContextPreparationDetails`
                cached by a non-streaming ``chat()`` call, or ``None`` when
                no such record exists.

        Returns:
            A populated :class:`ChatUsage` when ``details`` is present, or
            an all-``None`` instance (graceful no-op) when it is ``None``.
        """
        if details is None:
            return cls()
        provider = getattr(details, "provider", None)
        model = getattr(details, "model", None)
        return cls(
            prompt_tokens=_as_int_or_none(getattr(details, "prompt_tokens", None)),
            completion_tokens=_as_int_or_none(
                getattr(details, "completion_tokens", None)
            ),
            total_tokens=_as_int_or_none(getattr(details, "total_tokens", None)),
            provider=provider or None,
            model=model or None,
        )
