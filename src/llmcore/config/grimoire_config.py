# src/llmcore/config/grimoire_config.py
"""Grimoire control-plane configuration ([grimoire] section).

Grimoire is a HARD dependency of llmcore (0.52.0+): the bundled pack under
``llmcore/grimoire_pack/`` is ALWAYS the base layer, so agents work with zero
config. This section configures the optional overlay layers and the fail-loud
semantics:

- zero config          → bundled pack only (always valid; validated in CI)
- admin/user overlays  → loaded STRICT; any parse failure, duplicate id, or
                         override that breaks a required template id is a
                         HARD startup error naming the layer — never a
                         silent fallback.

Example::

    [grimoire]
    user_repo_paths = ["~/.config/llmcore/grimoire"]
    prompt_map = { thinking_prompt = "my/company/think" }

    [grimoire.metrics]
    enabled = true
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from confy.config import Config as ConfyConfig

logger = logging.getLogger(__name__)


class GrimoireMetricsConfig(BaseModel):
    """[grimoire.metrics] — optional persistent prompt-usage telemetry."""

    enabled: bool = Field(default=False, description="Persist prompt usage to JSONL")
    path: str = Field(
        default="",
        description="JSONL path (default: ~/.local/state/llmcore/prompt_usage.jsonl)",
    )


class GrimoireConfig(BaseModel):
    """[grimoire] — the control-plane composition llmcore auto-wires."""

    admin_repo_path: str = Field(
        default="",
        description="Optional read-only admin overlay layer (above bundled)",
    )
    user_repo_paths: list[str] = Field(
        default_factory=list,
        description="Ordered user overlay roots, lowest → highest precedence "
        "(each above bundled/admin); the LAST one is the writable layer",
    )
    extra_pack_paths: list[str] = Field(
        default_factory=list,
        description="Additional read-only packs layered ABOVE bundled and "
        "BELOW admin/user (hosts like wairu register their bundled packs here)",
    )
    prompt_map: dict[str, str] = Field(
        default_factory=dict,
        description="llmcore template id → spell id overrides (merged over "
        "the built-in DEFAULT_TEMPLATE_MAP)",
    )
    strict: bool = Field(
        default=True,
        description="Strict conjure (missing required variables raise)",
    )
    validate_on_startup: bool = Field(
        default=True,
        description="When overlays are configured, render every required "
        "template once at init and hard-fail on any error",
    )
    metrics: GrimoireMetricsConfig = Field(default_factory=GrimoireMetricsConfig)


def load_grimoire_config(
    config: "ConfyConfig | None" = None,
    overrides: dict[str, Any] | None = None,
) -> GrimoireConfig:
    """Load the [grimoire] section from a confy Config (missing → defaults).

    Args:
        config: Unified confy Config (from ``LLMCore.config``); the
            ``"grimoire"`` section is read when present.
        overrides: Optional runtime overrides merged last.

    Returns:
        A validated :class:`GrimoireConfig` (defaults on any absence).
    """
    section: dict[str, Any] = {}
    if config is not None:
        try:
            raw = config.get("grimoire", {})
            if isinstance(raw, dict):
                section = dict(raw)
            elif raw:
                section = dict(raw)  # confy mapping-likes
        except Exception as exc:
            logger.debug("No [grimoire] config section (%s); using defaults", exc)
    if overrides:
        section.update(overrides)
    try:
        return GrimoireConfig(**section)
    except Exception as exc:
        # A malformed [grimoire] section is a configuration error the operator
        # must see — this is the control plane, not an optional feature.
        raise ValueError(f"Invalid [grimoire] configuration: {exc}") from exc


__all__ = ["GrimoireConfig", "GrimoireMetricsConfig", "load_grimoire_config"]
