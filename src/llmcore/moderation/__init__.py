# src/llmcore/moderation/__init__.py
"""
Two-stage moderation gateway for the wairu ecosystem (plan SF-1 / DDS-06).

Stage 1 — a :class:`ModerationGateway` (e.g. OpenAI's moderation endpoint)
returns raw per-category scores for a piece of text. Stage 2 — a
:class:`ModerationPolicy` applies per-category thresholds and produces a
:class:`ModerationDecision` (allow / warn / block).

Key safety property: **fail-safe**. When moderation is enabled and the
gateway errors (timeout, auth, transport), the policy blocks — it never
silently allows. Disabled remains the default; consumers opt in via the
``[moderation]`` config section and
:func:`~llmcore.moderation.factory.build_moderation_gateway`.

Usage:
    >>> from llmcore.moderation import ModerationPolicy, NoopGateway, moderate
    >>> decision = await moderate(NoopGateway(), ModerationPolicy(), "hello")
    >>> decision.allowed
    True
"""

from __future__ import annotations

from .factory import build_moderation_gateway, build_moderation_policy
from .gateway import ModerationGateway, NoopGateway, moderate
from .models import ModerationAction, ModerationDecision, ModerationResult
from .openai_gateway import DEFAULT_MODERATION_MODEL, OpenAIModerationGateway
from .policy import ModerationPolicy

__all__ = [
    "DEFAULT_MODERATION_MODEL",
    "ModerationAction",
    "ModerationDecision",
    "ModerationGateway",
    "ModerationPolicy",
    "ModerationResult",
    "NoopGateway",
    "OpenAIModerationGateway",
    "build_moderation_gateway",
    "build_moderation_policy",
    "moderate",
]
