# src/llmcore/moderation/factory.py
"""
Config-driven construction of moderation gateways and policies (SF-1).

The ``[moderation]`` config section (see ``default_config.toml``)::

    [moderation]
    enabled = false            # master switch — OFF by default
    provider = "openai"        # or "noop" for inert dry-run wiring
    model = "omni-moderation-latest"
    fail_safe = true           # gateway error while enabled => BLOCK
    default_action = "block"   # or "warn"
    default_threshold = 0.9    # optional; unlisted categories -> provider flag
    timeout = 30.0
    max_retries = 2

    [moderation.thresholds]    # optional per-category score thresholds
    violence = 0.8
    self_harm = 0.5

:func:`build_moderation_gateway` follows the ecosystem's degrade-to-None
factory convention **only for the disabled case**: ``enabled = false`` (the
default) returns ``None`` so callers skip the feature entirely. When
moderation is *enabled* but cannot be constructed (missing ``openai``
package, no API key, unknown provider) the factory raises instead —
silently returning ``None`` there would disable a requested safety net,
violating the SF-1 fail-safe guarantee.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..exceptions import ConfigError
from .gateway import ModerationGateway, NoopGateway
from .models import ModerationAction
from .policy import ModerationPolicy

__all__ = ["build_moderation_gateway", "build_moderation_policy"]

logger = logging.getLogger(__name__)

_VALID_DEFAULT_ACTIONS = (ModerationAction.BLOCK, ModerationAction.WARN)


def _extract_section(config: Any) -> dict[str, Any]:
    """Normalize *config* to the ``[moderation]`` section as a plain dict.

    Accepts a full config mapping (plain dict or confy ``Config`` — both are
    Mappings) containing a ``moderation`` sub-mapping, the section itself,
    an object exposing ``get("moderation")``, or ``None``.
    """
    if config is None:
        return {}
    if isinstance(config, Mapping):
        section = config.get("moderation")
        if isinstance(section, Mapping):
            return dict(section)
        return dict(config)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            section = getter("moderation", {})
        except Exception:
            section = {}
        if isinstance(section, Mapping):
            return dict(section)
    return {}


def build_moderation_policy(config: Any = None) -> ModerationPolicy:
    """Build a :class:`ModerationPolicy` from the ``[moderation]`` section.

    Invalid values degrade in the *conservative* direction with a warning
    (unknown ``default_action`` becomes ``block``; non-numeric thresholds
    are dropped, falling back to the provider's own flags).

    Args:
        config: Full config mapping, the ``[moderation]`` section itself,
            or ``None`` for an all-defaults policy.

    Returns:
        The configured policy (always constructible; safe defaults).
    """
    section = _extract_section(config)

    thresholds: dict[str, float] = {}
    raw_thresholds = section.get("thresholds", {})
    if isinstance(raw_thresholds, Mapping):
        for name, value in raw_thresholds.items():
            try:
                thresholds[str(name)] = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-numeric moderation threshold %r=%r", name, value)
    elif raw_thresholds:
        logger.warning("Ignoring malformed [moderation.thresholds]: %r", raw_thresholds)

    default_threshold: float | None = None
    raw_default = section.get("default_threshold")
    if raw_default is not None:
        try:
            default_threshold = float(raw_default)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric moderation default_threshold %r", raw_default)

    raw_action = str(section.get("default_action", "block")).lower()
    try:
        default_action = ModerationAction(raw_action)
    except ValueError:
        default_action = ModerationAction.BLOCK
        logger.warning("Unknown moderation default_action %r; using 'block'", raw_action)
    if default_action not in _VALID_DEFAULT_ACTIONS:
        logger.warning(
            "moderation default_action %r is not enforceable; using 'block' "
            "(use enabled=false to turn moderation off)",
            raw_action,
        )
        default_action = ModerationAction.BLOCK

    return ModerationPolicy(
        thresholds=thresholds,
        default_threshold=default_threshold,
        default_action=default_action,
        fail_safe=bool(section.get("fail_safe", True)),
    )


def build_moderation_gateway(config: Any = None) -> ModerationGateway | None:
    """Build the configured moderation gateway, or ``None`` when disabled.

    Degrade-to-None applies **only** to the disabled case (``enabled`` is
    false or absent — moderation is OFF by default). When enabled, any
    construction failure propagates so a requested safety net can never be
    silently skipped.

    Args:
        config: Full config mapping, the ``[moderation]`` section itself,
            or ``None`` (treated as disabled).

    Returns:
        A gateway instance, or ``None`` when moderation is disabled.

    Raises:
        ConfigError: Unknown ``provider`` value, or (from the gateway)
            missing API key / client construction failure while enabled.
        ImportError: The provider's SDK is not installed while enabled.
    """
    section = _extract_section(config)
    if not bool(section.get("enabled", False)):
        return None

    provider = str(section.get("provider", "openai")).lower()
    if provider == "openai":
        from .openai_gateway import OpenAIModerationGateway

        return OpenAIModerationGateway(section)
    if provider == "noop":
        logger.warning("Moderation enabled with the 'noop' provider: nothing will be flagged.")
        return NoopGateway()
    raise ConfigError(f"Unknown moderation provider '{provider}' (expected 'openai' or 'noop').")
