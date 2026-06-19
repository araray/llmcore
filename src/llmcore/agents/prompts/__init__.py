# src/llmcore/agents/prompts/__init__.py
"""
Prompt Library for Darwin Layer 2.

Centralized prompt management with versioning and A/B testing support.
"""

from .composer import (
    CircularInclusionError,
    MissingSnippetError,
    MissingVariableError,
    PromptComposer,
)
from .grimoire_adapter import (
    GrimoirePromptRegistryAdapter,
    GrimoirePromptTemplate,
    GrimoirePromptVersion,
)
from .models import (
    PromptCategory,
    PromptMetrics,
    PromptSnippet,
    PromptTemplate,
    PromptVariable,
    PromptVersion,
    VersionStatus,
)
from .registry import (
    PromptRegistry,
    RegistryError,
    SnippetNotFoundError,
    TemplateExistsError,
    TemplateNotFoundError,
)
from .template_loader import TemplateLoader, load_default_templates

__all__ = [
    "CircularInclusionError",
    "GrimoirePromptRegistryAdapter",
    "GrimoirePromptTemplate",
    "GrimoirePromptVersion",
    "MissingSnippetError",
    "MissingVariableError",
    "PromptCategory",
    "PromptComposer",
    "PromptMetrics",
    "PromptRegistry",
    "PromptSnippet",
    "PromptTemplate",
    "PromptVariable",
    "PromptVersion",
    "RegistryError",
    "SnippetNotFoundError",
    "TemplateExistsError",
    "TemplateLoader",
    "TemplateNotFoundError",
    "VersionStatus",
    "load_default_templates",
]
