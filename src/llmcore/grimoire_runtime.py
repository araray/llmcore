# src/llmcore/grimoire_runtime.py
"""Grimoire control-plane runtime wiring (0.52.0).

Grimoire is a hard dependency of llmcore: ALL agent prompts come from grimoire
spells and the builtin tool contracts are grimoire runes. This module owns:

- :func:`bundled_pack_path` — locate the packaged ``llmcore-builtin`` pack
  (always the BASE layer; agents work with zero configuration).
- :data:`REQUIRED_SPELLS` — the single manifest of every template id llmcore
  consumes, with representative variables. Drives BOTH the adapter's template
  map coverage and startup/CI validation.
- :func:`build_grimoire` — compose the layered Grimoire per
  ``[grimoire]`` config: bundled < extra packs < admin < user (highest wins).
- :func:`build_prompt_registry` — the ``GrimoirePromptRegistryAdapter`` over
  that instance (with the optional usage store).
- :func:`validate_grimoire_startup` — fail-loud validation: when overlays are
  configured, every required template must render; any failure raises
  :class:`~llmcore.exceptions.ConfigError` naming the template, the resolved
  spell, the winning layer, and the cause. NEVER a silent fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmcore.exceptions import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from llmcore.agents.prompts.grimoire_adapter import GrimoirePromptRegistryAdapter
    from llmcore.config.grimoire_config import GrimoireConfig

logger = logging.getLogger(__name__)

#: Layer name of the packaged base pack.
BUNDLED_LAYER = "llmcore-builtin"


def bundled_pack_path() -> Path:
    """Path to the packaged ``llmcore-builtin`` grimoire pack.

    Uses ``importlib.resources`` so the pack resolves in installed
    environments as well as source checkouts. Grimoire needs a real
    directory; setuptools wheels install unzipped, so ``as_file`` is only a
    formality for exotic loaders.
    """
    pack = resources.files("llmcore") / "grimoire_pack"
    try:
        path = Path(str(pack))
    except Exception:  # pragma: no cover - exotic loaders
        with resources.as_file(pack) as concrete:
            path = Path(concrete)
    if not path.is_dir():  # pragma: no cover - packaging error
        raise ConfigError(
            f"Bundled grimoire pack missing at {path} — broken llmcore install"
        )
    return path


@dataclass(frozen=True)
class RequiredSpell:
    """One required template id with representative render variables."""

    template_id: str
    sample_vars: dict[str, str] = field(default_factory=dict)
    #: Whether consumers use the role-structured render (render_messages).
    expects_system: bool = True


#: Every template id llmcore consumes. THE single source of truth: the
#: adapter's DEFAULT_TEMPLATE_MAP must cover each id, the bundled pack must
#: render each one (CI gate), and startup validation re-renders them against
#: the composed layers when overlays are configured.
REQUIRED_SPELLS: tuple[RequiredSpell, ...] = (
    RequiredSpell(
        "planning_prompt",
        {"goal": "Sample goal", "context": "", "constraints": "", "existing_plan_section": ""},
    ),
    RequiredSpell(
        "thinking_prompt",
        {
            "goal": "Sample goal",
            "current_step": "Do the thing",
            "history": "No previous actions.",
            "context": "",
            "tools": "- finish(answer): finish",
            "remaining_steps": "unlimited",
        },
    ),
    RequiredSpell(
        "validation_prompt",
        {
            "goal": "Sample goal",
            "proposed_action": "calculator({'expression': '1+1'})",
            "reasoning": "check",
            "risk_tolerance": "medium",
        },
    ),
    RequiredSpell(
        "reflection_prompt",
        {
            "goal": "Sample goal",
            "plan": "1. Step one",
            "current_step_display": "1. Step one",
            "last_action": "calculator({'expression': '1+1'})",
            "observation": "2",
            "iteration": "1",
            "action_success": "true",
            "matches_expectation": "",
        },
    ),
    RequiredSpell(
        "finalize_prompt",
        {
            "goal": "Sample goal",
            "history": "[]",
            "context": "",
            "reason": "synthesis_fallback",
        },
    ),
    RequiredSpell("activity_system", {}),
    RequiredSpell(
        "activity_execute",
        {
            "goal": "Sample goal",
            "current_step": "Do the thing",
            "activities_section": "",
            "history_section": "",
            "context_section": "",
        },
        expects_system=False,
    ),
    RequiredSpell("goal_classifier", {"goal": "Sample goal"}, expects_system=False),
    RequiredSpell("fast_path", {"goal": "hello"}),
    RequiredSpell(
        "goal_decomposition",
        {"goal_description": "Sample goal", "context_json": "None"},
    ),
    RequiredSpell(
        "darwin_arbiter_generation",
        {"task": "t", "context": "c", "additional_instructions": ""},
        expects_system=False,
    ),
    RequiredSpell(
        "darwin_arbiter_evaluation",
        {"task": "t", "code": "print(1)", "criteria_list": "- correctness"},
    ),
    RequiredSpell(
        "darwin_arbiter_selection",
        {"task": "t", "candidates_summary": "candidate_1: 9.0"},
    ),
    RequiredSpell(
        "darwin_tdd_spec_generation",
        {"requirements": "add()", "language": "python", "framework": "pytest", "min_tests": "3"},
    ),
    RequiredSpell(
        "darwin_tdd_test_generation",
        {
            "name": "test_add",
            "description": "d",
            "test_type": "unit",
            "inputs": "{}",
            "expected_output": "2",
            "expected_behavior": "",
            "expected_exception": "",
            "language": "python",
            "framework": "pytest",
        },
    ),
    RequiredSpell(
        "darwin_tdd_implementation",
        {
            "requirements": "add()",
            "language": "python",
            "test_file": "def test(): ...",
            "previous_implementation": "",
            "test_failures": "",
        },
    ),
)


def build_grimoire(cfg: "GrimoireConfig") -> Any:
    """Compose the layered Grimoire instance for this process.

    Layer order (lowest → highest precedence)::

        llmcore-builtin < extra packs < admin < user...

    The bundled layer always loads. Overlay layers load STRICT: a parse
    failure or duplicate id inside a layer raises immediately (wrapped as
    :class:`ConfigError` naming the layer).

    Returns:
        A ``grimoire.Grimoire`` facade (layered when overlays exist, single
        pack otherwise — identical read API either way).
    """
    from grimoire import Grimoire

    roots: list[tuple[str, str | Path, bool]] = [
        (BUNDLED_LAYER, bundled_pack_path(), False)
    ]
    for i, extra in enumerate(cfg.extra_pack_paths):
        path = Path(extra).expanduser()
        if not path.is_dir():
            raise ConfigError(f"[grimoire] extra_pack_paths[{i}] does not exist: {path}")
        roots.append((f"pack{i}" if i else "pack", path, False))
    if cfg.admin_repo_path:
        admin = Path(cfg.admin_repo_path).expanduser()
        if not admin.is_dir():
            raise ConfigError(f"[grimoire] admin_repo_path does not exist: {admin}")
        roots.append(("admin", admin, False))
    for i, user in enumerate(cfg.user_repo_paths):
        writable = i == len(cfg.user_repo_paths) - 1  # last user layer is writable
        name = "user" if len(cfg.user_repo_paths) == 1 else f"user{i}"
        roots.append((name, Path(user).expanduser(), writable))

    if len(roots) == 1:
        # Zero-config fast path: a single bundled pack needs no layer machinery.
        return Grimoire(bundled_pack_path(), strict=cfg.strict)

    try:
        return Grimoire.layered(roots, strict=cfg.strict, load_strict=True)
    except Exception as exc:
        raise ConfigError(f"Grimoire control-plane layers failed to load: {exc}") from exc


def build_prompt_registry(
    grimoire: Any, cfg: "GrimoireConfig"
) -> "GrimoirePromptRegistryAdapter":
    """Build the prompt-registry adapter over a composed Grimoire instance."""
    from llmcore.agents.prompts.grimoire_adapter import GrimoirePromptRegistryAdapter

    usage_store = None
    if cfg.metrics.enabled:
        from llmcore.agents.prompts.usage_store import JsonlPromptUsageStore

        usage_store = JsonlPromptUsageStore(cfg.metrics.path or None)

    return GrimoirePromptRegistryAdapter(
        grimoire,
        template_map=dict(cfg.prompt_map),
        strict=cfg.strict,
        usage_store=usage_store,
    )


def _resolve_layer_hint(grimoire: Any, spell_id: str) -> str:
    """Best-effort 'which layer wins' hint for error messages."""
    try:
        return str(grimoire.resolve_layer(spell_id, "spell"))
    except Exception:
        return "unknown"


def validate_grimoire_startup(
    grimoire: Any,
    registry: "GrimoirePromptRegistryAdapter",
    cfg: "GrimoireConfig",
    *,
    overlays_present: bool,
) -> None:
    """Fail-loud startup validation of the composed control plane.

    Semantics (locked design decision): with NO overlays configured, the
    bundled pack is trusted (its completeness is a CI/pytest gate — zero-config
    startup stays fast). With overlays present, (a) each overlay layer is
    structurally validated in isolation, and (b) EVERY required template id is
    rendered once against the composed view; any failure raises
    :class:`ConfigError` naming template, spell, winning layer, and cause.
    """
    if not (overlays_present and cfg.validate_on_startup):
        return

    # (a) Per-layer structural validation for every non-bundled layer.
    layer_names = list(grimoire.layers or [])
    for layer_name in layer_names:
        if layer_name == BUNDLED_LAYER:
            continue
        try:
            result = grimoire.validate(layer=layer_name)
        except Exception as exc:
            raise ConfigError(
                f"Grimoire layer {layer_name!r} failed validation: {exc}"
            ) from exc
        errors = [
            d
            for d in getattr(result, "diagnostics", [])
            if str(getattr(d, "severity", "")).lower().endswith("error")
        ]
        if errors:
            raise ConfigError(
                f"Grimoire layer {layer_name!r} has {len(errors)} validation "
                f"error(s); first: {errors[0]}"
            )

    # (b) Render every required template once against the composed layers.
    for required in REQUIRED_SPELLS:
        spell_id = registry.resolve_template_id(required.template_id)
        try:
            registry.render_messages(required.template_id, dict(required.sample_vars))
        except Exception as exc:
            layer = _resolve_layer_hint(grimoire, spell_id)
            raise ConfigError(
                f"Grimoire startup validation failed for template "
                f"{required.template_id!r} (spell {spell_id!r}, resolved from "
                f"layer {layer!r}): {exc}. Fix or remove the overriding spell "
                f"— llmcore never falls back silently."
            ) from exc

    logger.info(
        "Grimoire startup validation OK: %d required templates render across %s",
        len(REQUIRED_SPELLS),
        layer_names or ["bundled"],
    )


__all__ = [
    "BUNDLED_LAYER",
    "REQUIRED_SPELLS",
    "RequiredSpell",
    "build_grimoire",
    "build_prompt_registry",
    "bundled_pack_path",
    "validate_grimoire_startup",
]
