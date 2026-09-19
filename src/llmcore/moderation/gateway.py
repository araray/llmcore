# src/llmcore/moderation/gateway.py
"""
Moderation gateway protocol, the no-op gateway, and the ``moderate`` helper.

A *gateway* answers one question — "what does the moderation backend say
about this text?" — and returns a :class:`~llmcore.moderation.models.ModerationResult`.
It never decides anything; that is the job of
:class:`~llmcore.moderation.policy.ModerationPolicy`.

:func:`moderate` is the one-call seam most callers want: it runs
``gateway.check(...)``, routes gateway exceptions through the policy's
fail-safe (an enabled-but-broken gateway must **block**, never silently
allow), and returns the resulting
:class:`~llmcore.moderation.models.ModerationDecision`.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .models import ModerationResult
from .policy import ModerationPolicy

__all__ = [
    "ModerationGateway",
    "NoopGateway",
    "moderate",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class ModerationGateway(Protocol):
    """Protocol every moderation backend implements.

    Implementations should raise
    :class:`~llmcore.exceptions.ModerationError` (or any exception — the
    :func:`moderate` helper treats all of them fail-safe) when the backend
    cannot produce a verdict.
    """

    async def check(self, text: str, *, context: str = "") -> ModerationResult:
        """Return the backend's raw verdict for *text*.

        Args:
            text: The content to classify.
            context: Caller-supplied surface tag (e.g. ``"input"``,
                ``"output"``, ``"tool"``) for logging/observability. Never
                affects classification.

        Returns:
            The backend's :class:`ModerationResult`.
        """
        ...


class NoopGateway:
    """Gateway that never flags anything.

    Useful for tests and as an explicit stand-in when moderation is wired
    but intentionally inert. It is *not* what "disabled" means in config —
    a disabled ``[moderation]`` section yields no gateway at all (see
    :func:`~llmcore.moderation.factory.build_moderation_gateway`).
    """

    async def check(self, text: str, *, context: str = "") -> ModerationResult:
        return ModerationResult(flagged=False, provider="noop")


async def moderate(
    gateway: ModerationGateway,
    policy: ModerationPolicy,
    text: str,
    *,
    context: str = "",
):
    """Check *text* through *gateway* and evaluate it under *policy*.

    This is the fail-safe seam: any exception the gateway raises is routed
    through :meth:`ModerationPolicy.decision_for_error`, which blocks when
    ``policy.fail_safe`` is true (the default). Callers therefore never
    need their own try/except to stay safe.

    Args:
        gateway: The moderation backend to consult.
        policy: The policy that turns the raw verdict into a decision.
        text: The content to moderate.
        context: Surface tag (``"input"``, ``"output"``, ``"tool"``, ...).

    Returns:
        The policy's :class:`~llmcore.moderation.models.ModerationDecision`.
    """
    try:
        result = await gateway.check(text, context=context)
    except Exception as exc:
        logger.warning("Moderation gateway error (context=%r): %s", context, exc)
        return policy.decision_for_error(exc, context=context)
    return policy.evaluate(result, context=context)
