# src/llmcore/agents/grimoire_tool_catalog.py
"""Grimoire-backed tool catalog (0.52.0 — catalog+policy, host execution).

Locked control-plane design: **grimoire runes are the single source of truth
for tool CONTRACTS and POLICY** (schema, risk level, approval requirement,
OWASP tags, execution target), while **handlers/execution stay host-side**
(llmcore's ``ToolManager`` and its secure ``_IMPLEMENTATION_REGISTRY``).

The catalog is the bridge:

1. Runes come from the composed (layered) Grimoire instance — bundled pack
   contracts, host packs, user overrides, and in-memory runtime runes all
   resolve through the same precedence rules.
2. Hosts ``bind()`` callables to ``(rune_id, command)`` — a rune with no
   bound handler is *visible* in :meth:`contracts` but is NEVER registered
   for execution (execution safety).
3. :meth:`apply_to` feeds a ``ToolManager``: the ``Tool`` schema is built
   from the rune command, policy metadata travels via ``register_runtime_tool
   (metadata=...)`` and surfaces in ``get_tool_inventory``.

Per-command runtime hints live in rune-level ``mappings`` with dotted keys::

    mappings:
      llmcore.tool_name.finish: finish
      llmcore.implementation_key.finish: llmcore.tools.flow.finish

Resolution order for a hint ``H`` of command ``C``:
``mappings["llmcore.H.C"]`` → ``mappings["llmcore.H"]`` → default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from llmcore.agents.tools import ToolManager


@dataclass(frozen=True)
class CatalogEntry:
    """One bound (rune command → handler) catalog record."""

    rune_id: str
    command: str
    tool_name: str
    implementation_key: str
    implementation: Callable


def _mapping_hint(rune: Any, hint: str, command: str, default: str) -> str:
    """Resolve a per-command runtime hint from rune-level ``mappings``."""
    mappings = getattr(rune, "mappings", None) or {}
    return str(
        mappings.get(f"llmcore.{hint}.{command}")
        or mappings.get(f"llmcore.{hint}")
        or default
    )


class GrimoireToolCatalog:
    """Feed llmcore's ToolManager from grimoire rune contracts."""

    def __init__(self, grimoire: Any) -> None:
        self._grimoire = grimoire
        self._bindings: dict[tuple[str, str], CatalogEntry] = {}

    # ── Binding ─────────────────────────────────────────────────────────────

    def bind(
        self,
        rune_id: str,
        command: str,
        implementation: Callable,
        *,
        implementation_key: str | None = None,
    ) -> CatalogEntry:
        """Bind a host callable to a rune command.

        Args:
            rune_id: The rune id (must exist in the composed Grimoire).
            command: The command name within the rune.
            implementation: The callable executed for this tool.
            implementation_key: Override for the secure-registry key; defaults
                to the rune's ``llmcore.implementation_key.<command>`` mapping,
                then ``"<rune_id>::<command>"``.

        Returns:
            The recorded :class:`CatalogEntry`.

        Raises:
            KeyError: If the rune or command does not exist (catalog+policy
                requires the CONTRACT to exist before a handler can bind).
        """
        rune = self._grimoire.get_rune(rune_id)
        cmd = rune.get_command(command)
        if cmd is None:
            raise KeyError(f"Rune {rune_id!r} has no command {command!r}")

        key = implementation_key or _mapping_hint(
            rune, "implementation_key", command, f"{rune_id}::{command}"
        )
        tool_name = _mapping_hint(rune, "tool_name", command, command)
        entry = CatalogEntry(
            rune_id=rune_id,
            command=command,
            tool_name=tool_name,
            implementation_key=key,
            implementation=implementation,
        )
        self._bindings[(rune_id, command)] = entry
        return entry

    def bind_many(
        self, bindings: dict[tuple[str, str], Callable]
    ) -> list[CatalogEntry]:
        """Bind several ``(rune_id, command) -> callable`` pairs."""
        return [
            self.bind(rune_id, command, implementation)
            for (rune_id, command), implementation in bindings.items()
        ]

    # ── Introspection ───────────────────────────────────────────────────────

    def contracts(
        self, *, tags: list[str] | None = None, rune_ids: list[str] | None = None
    ) -> list[Any]:
        """Rune contracts in scope — INCLUDING unbound ones (policy view)."""
        if rune_ids is not None:
            runes = []
            for rid in rune_ids:
                try:
                    runes.append(self._grimoire.get_rune(rid))
                except Exception:
                    logger.warning("contracts: rune %r not found, skipping", rid)
            return runes
        return list(self._grimoire.repo.list_runes(tags=tags))

    def bound(self) -> list[CatalogEntry]:
        """All recorded bindings."""
        return list(self._bindings.values())

    # ── ToolManager application ─────────────────────────────────────────────

    def apply_to(
        self,
        tool_manager: "ToolManager",
        *,
        tags: list[str] | None = None,
        rune_ids: list[str] | None = None,
        require_bound: bool = True,
    ) -> list[str]:
        """Register every BOUND rune command in scope on a ToolManager.

        The ``Tool`` schema comes from the rune command
        (``command_parameters_schema``); policy metadata (risk/approval/owasp/
        permissions/execution_target/tags + rune provenance) travels via
        ``register_runtime_tool(metadata=...)`` and surfaces in
        ``get_tool_inventory``.

        Args:
            tool_manager: The manager to feed.
            tags / rune_ids: Contract scope (default: every rune).
            require_bound: When ``True`` (default), an in-scope rune command
                with no bound handler raises ``ConfigError`` — used by
                ``load_default_tools`` to enforce pack completeness. When
                ``False``, unbound commands are skipped with a warning.

        Returns:
            The registered tool names.
        """
        from grimoire.runes.schema import command_parameters_schema

        from llmcore.exceptions import ConfigError
        from llmcore.models import Tool

        registered: list[str] = []
        for rune in self.contracts(tags=tags, rune_ids=rune_ids):
            for cmd in rune.commands:
                entry = self._bindings.get((rune.id, cmd.name))
                if entry is None:
                    if require_bound:
                        raise ConfigError(
                            f"Rune command {rune.id}::{cmd.name} has no bound "
                            "implementation — bind() a handler or narrow the scope"
                        )
                    logger.warning(
                        "Skipping unbound rune command %s::%s", rune.id, cmd.name
                    )
                    continue

                schema = command_parameters_schema(cmd)
                tool = Tool(
                    name=entry.tool_name,
                    description=str(cmd.summary or rune.description or entry.tool_name),
                    parameters=schema,
                )
                risk = cmd.risk_level or rune.risk_level
                metadata: dict[str, Any] = {
                    "source": "grimoire",
                    "rune_id": rune.id,
                    "rune_version": str(getattr(rune, "version", "")),
                    "content_hash": str(getattr(rune, "content_hash", "") or ""),
                    "risk_level": str(getattr(risk, "value", risk) or "low"),
                    "requires_approval": bool(
                        cmd.requires_approval or rune.requires_approval
                    ),
                }
                if rune.owasp_categories or cmd.owasp_categories:
                    metadata["owasp"] = [
                        str(c) for c in (cmd.owasp_categories or rune.owasp_categories)
                    ]
                if rune.permissions:
                    metadata["permissions"] = [
                        str(getattr(p, "value", p)) for p in rune.permissions
                    ]
                if cmd.execution_target:
                    metadata["execution_target"] = str(cmd.execution_target)
                if rune.tags:
                    metadata["tags"] = [str(t) for t in rune.tags]

                tool_manager.register_runtime_tool(
                    tool,
                    entry.implementation_key,
                    entry.implementation,
                    metadata=metadata,
                )
                registered.append(entry.tool_name)

        logger.info("GrimoireToolCatalog registered %d tool(s)", len(registered))
        return registered


def bind_builtin_tools(catalog: GrimoireToolCatalog) -> None:
    """Bind the 5 llmcore builtin callables to their pack rune contracts."""
    from llmcore.agents import tools as builtin

    catalog.bind_many(
        {
            ("llmcore/flow", "finish"): builtin.finish,
            ("llmcore/flow", "human_approval"): builtin.human_approval,
            ("llmcore/search", "semantic_search"): builtin.semantic_search,
            ("llmcore/search", "episodic_search"): builtin.episodic_search,
            ("llmcore/calculation", "calculator"): builtin.calculator,
        }
    )


__all__ = ["CatalogEntry", "GrimoireToolCatalog", "bind_builtin_tools"]
