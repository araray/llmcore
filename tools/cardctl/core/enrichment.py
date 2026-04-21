# tools/cardctl/core/enrichment.py
"""Enrichment overlay system for model card generation.

Enrichment files (TOML) contain manually-curated data that provider APIs
don't expose: pricing, architecture details, display name overrides, tags,
provider-specific extension fields, and exclusion patterns.

Enrichments survive card regeneration — the generator never modifies them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from .common import ENRICHMENTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ModelEnrichment:
    """Resolved enrichment data for a single model."""

    pricing: dict[str, float] | None = None  # {"input": X, "output": Y}
    architecture: dict[str, str] | None = None  # {"family": ..., "type": ..., ...}
    overrides: dict[str, Any] = field(default_factory=dict)  # arbitrary field overrides
    extra_aliases: list[str] = field(default_factory=list)
    provider_extension: dict[str, Any] = field(default_factory=dict)
    extra_tags: list[str] = field(default_factory=list)


class EnrichmentStore:
    """Loads and queries enrichment overlays for a provider.

    Usage::

        store = EnrichmentStore.load("openai")
        enrichment = store.get("gpt-4o")
        # enrichment.pricing == {"input": 2.5, "output": 10.0}
    """

    def __init__(
        self,
        pricing: dict[str, dict[str, float]],
        architecture: dict[str, dict[str, str]],
        overrides: dict[str, dict[str, Any]],
        aliases: dict[str, list[str]],
        provider_extension: dict[str, dict[str, Any]],
        exclude_patterns: list[str],
    ):
        self._pricing = pricing
        self._architecture = architecture
        self._overrides = overrides
        self._aliases = aliases
        self._provider_extension = provider_extension
        self.exclude_patterns = exclude_patterns

    @classmethod
    def load(cls, provider: str, enrichments_dir: Path | None = None) -> EnrichmentStore:
        """Load enrichment TOML for *provider*.

        Returns an empty store if no file exists or TOML support is unavailable.
        """
        directory = enrichments_dir or ENRICHMENTS_DIR
        path = directory / f"{provider}.toml"

        if tomllib is None:
            logger.warning("tomllib/tomli not available; enrichments disabled.")
            return cls._empty()

        if not path.exists():
            logger.debug("No enrichment file for %s at %s", provider, path)
            return cls._empty()

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(
            pricing=data.get("pricing", {}),
            architecture=data.get("architecture", {}),
            overrides=data.get("overrides", {}),
            aliases=cls._parse_aliases(data.get("aliases", {})),
            provider_extension=data.get("provider_extension", {}),
            exclude_patterns=data.get("exclude", {}).get("patterns", []),
        )

    @classmethod
    def _empty(cls) -> EnrichmentStore:
        return cls(
            pricing={},
            architecture={},
            overrides={},
            aliases={},
            provider_extension={},
            exclude_patterns=[],
        )

    @staticmethod
    def _parse_aliases(raw: dict[str, Any]) -> dict[str, list[str]]:
        """Normalize alias entries to list[str]."""
        result: dict[str, list[str]] = {}
        for key, val in raw.items():
            if isinstance(val, list):
                result[key] = val
            elif isinstance(val, str):
                result[key] = [val]
        return result

    def is_excluded(self, model_id: str) -> bool:
        """Check if a model ID matches any exclusion pattern."""
        for pattern in self.exclude_patterns:
            if model_id.startswith(pattern) or model_id == pattern:
                return True
        return False

    def get(self, model_id: str) -> ModelEnrichment:
        """Resolve enrichments for a model, using exact then prefix matching."""
        return ModelEnrichment(
            pricing=self._match(self._pricing, model_id),
            architecture=self._match(self._architecture, model_id),
            overrides=self._match(self._overrides, model_id) or {},
            extra_aliases=self._aliases.get(model_id, []),
            provider_extension=self._match(self._provider_extension, model_id) or {},
            extra_tags=_list_wrap(
                (self._match(self._overrides, model_id) or {}).pop("tags", [])
            ),
        )

    @staticmethod
    def _match(table: dict[str, Any], model_id: str) -> Any | None:
        """Exact match first, then longest-prefix match."""
        # Exact
        if model_id in table:
            return table[model_id]
        # Longest prefix
        best: tuple[int, Any] = (-1, None)
        for key, val in table.items():
            if model_id.startswith(key) and len(key) > best[0]:
                best = (len(key), val)
        return best[1]


def _list_wrap(val: Any) -> list[str]:
    """Wrap a value as a list of strings."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val]
    return []
