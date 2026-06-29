"""Optional Grimoire-backed prompt registry adapter.

The cognitive phases depend on a small prompt-registry surface:
``render()``, ``get_template()``, and ``record_use()``. This adapter lets a
Grimoire spell repository satisfy that surface without making llmcore import
or depend on grimoire unless the adapter is explicitly instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import PromptMetrics
from .registry import TemplateNotFoundError

DEFAULT_TEMPLATE_MAP: dict[str, str] = {
    "planning_prompt": "planning_prompt",
    "thinking_prompt": "thinking_prompt",
    "validation_prompt": "validation_prompt",
    "reflection_prompt": "reflection_prompt",
}


@dataclass(frozen=True)
class GrimoirePromptVersion:
    """Minimal prompt-version facade returned by the Grimoire adapter."""

    id: str
    template_id: str
    grimoire_id: str
    version_number: int = 1
    content_hash: str | None = None


@dataclass(frozen=True)
class GrimoirePromptTemplate:
    """Minimal prompt-template facade compatible with cognitive phase metrics."""

    id: str
    name: str
    grimoire_id: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    active_version: GrimoirePromptVersion | None = None

    @property
    def active_version_id(self) -> str | None:
        """Return the active version ID when one exists."""
        return self.active_version.id if self.active_version else None


class GrimoirePromptRegistryAdapter:
    """Expose Grimoire spells through llmcore's prompt-registry surface."""

    def __init__(
        self,
        grimoire: Any | None = None,
        *,
        repo_path: str | Path | None = None,
        template_map: dict[str, str] | None = None,
        include_role_headers: bool = False,
        strict: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            grimoire: Existing Grimoire facade. When omitted, ``repo_path`` is
                used to instantiate ``grimoire.Grimoire`` lazily.
            repo_path: Grimoire repository root for lazy construction.
            template_map: Mapping from llmcore template IDs to Grimoire spell IDs.
                Unmapped IDs resolve by identity.
            include_role_headers: Whether rendered multi-block prompts include
                ``[ROLE]`` headers. Defaults to plain content joins, matching
                ``PromptRegistry.render()`` expectations.
            strict: Passed through to Grimoire conjure calls.
        """
        if grimoire is None:
            try:
                from grimoire import Grimoire
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise ImportError(
                    "GrimoirePromptRegistryAdapter requires the optional grimoire package"
                ) from exc
            grimoire = Grimoire(repo_path)

        self._grimoire = grimoire
        self._template_map = {**DEFAULT_TEMPLATE_MAP, **(template_map or {})}
        self._include_role_headers = include_role_headers
        self._strict = strict
        self._metrics: dict[str, PromptMetrics] = {}

    def resolve_template_id(self, template_id: str) -> str:
        """Resolve an llmcore template ID to a Grimoire spell ID."""
        return self._template_map.get(template_id, template_id)

    def get_template(self, template_id: str) -> GrimoirePromptTemplate:
        """Return lightweight metadata for a mapped Grimoire spell."""
        spell_id = self.resolve_template_id(template_id)
        try:
            spell = self._grimoire.get_spell(spell_id)
        except Exception as exc:
            raise TemplateNotFoundError(
                f"Template {template_id!r} mapped to missing Grimoire spell {spell_id!r}"
            ) from exc

        version_id = self._version_id(template_id, spell)
        return GrimoirePromptTemplate(
            id=template_id,
            name=str(getattr(spell, "name", spell_id)),
            grimoire_id=spell_id,
            description=getattr(spell, "description", None),
            tags=[str(tag) for tag in getattr(spell, "tags", [])],
            active_version=GrimoirePromptVersion(
                id=version_id,
                template_id=template_id,
                grimoire_id=spell_id,
                version_number=self._version_number(getattr(spell, "version", None)),
                content_hash=getattr(spell, "content_hash", None),
            ),
        )

    def render(
        self,
        template_id: str,
        variables: dict[str, Any] | None = None,
        version_id: str | None = None,
        validate: bool = True,
    ) -> str:
        """Render a mapped Grimoire spell as a plain prompt string."""
        del version_id  # Grimoire spell versioning is selected by repository state.
        spell_id = self.resolve_template_id(template_id)
        try:
            result = self._grimoire.conjure_spell(
                spell_id,
                variables=variables or {},
                strict=validate and self._strict,
            )
        except AttributeError:
            result = self._grimoire.conjure(
                spell_id,
                variables=variables or {},
                strict=validate and self._strict,
            )
        except Exception as exc:
            raise TemplateNotFoundError(
                f"Failed to render template {template_id!r} via Grimoire spell {spell_id!r}: {exc}"
            ) from exc

        if isinstance(result, list):
            raise ValueError(f"Grimoire artifact {spell_id!r} rendered as a ritual, not a prompt")
        return self._conjured_to_text(result)

    def get_metrics(self, version_id: str) -> PromptMetrics:
        """Return in-memory usage metrics for a Grimoire-backed prompt version."""
        if version_id not in self._metrics:
            self._metrics[version_id] = PromptMetrics(version_id=version_id)
        return self._metrics[version_id]

    def record_use(
        self,
        version_id: str,
        success: bool,
        iterations: int | None = None,
        tokens: int | None = None,
        latency_ms: float | None = None,
        quality_score: float | None = None,
    ) -> None:
        """Record in-memory usage metrics for cognitive phase compatibility."""
        self.get_metrics(version_id).record_use(
            success=success,
            iterations=iterations,
            tokens=tokens,
            latency_ms=latency_ms,
            quality_score=quality_score,
        )

    def _conjured_to_text(self, result: Any) -> str:
        blocks = getattr(result, "blocks", None)
        if not blocks:
            to_text = getattr(result, "to_text", None)
            if callable(to_text):
                return str(to_text())
            return str(result)

        parts: list[str] = []
        for block in blocks:
            content = str(getattr(block, "content", ""))
            if not content:
                continue
            if self._include_role_headers:
                role = getattr(block, "role", "")
                role_value = str(getattr(role, "value", role) or "").upper()
                parts.append(f"[{role_value}]\n{content}" if role_value else content)
            else:
                parts.append(content)
        return "\n\n".join(parts)

    @staticmethod
    def _version_number(version: Any) -> int:
        try:
            return max(1, int(str(version).split(".", maxsplit=1)[0]))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _version_id(template_id: str, spell: Any) -> str:
        spell_id = str(getattr(spell, "id", template_id))
        version = str(getattr(spell, "version", "1.0.0"))
        content_hash = str(getattr(spell, "content_hash", ""))
        return f"grimoire:{template_id}:{spell_id}:{version}:{content_hash}"


__all__ = [
    "DEFAULT_TEMPLATE_MAP",
    "GrimoirePromptRegistryAdapter",
    "GrimoirePromptTemplate",
    "GrimoirePromptVersion",
]
