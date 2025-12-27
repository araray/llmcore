# src/llmcore/agents/prompts/__init__.py
"""
Prompt Library for Darwin Layer 2.

Centralized prompt management with versioning and A/B testing support.
"""

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
    TemplateNotFoundError,
    SnippetNotFoundError,
    TemplateExistsError,
)

from .composer import (
    PromptComposer,
    MissingVariableError,
    MissingSnippetError,
    CircularInclusionError,
)

from .template_loader import TemplateLoader, load_default_templates

__all__ = [
    # Models
    "PromptCategory",
    "PromptMetrics",
    "PromptSnippet",
    "PromptTemplate",
    "PromptVariable",
    "PromptVersion",
    "VersionStatus",
    # Registry
    "PromptRegistry",
    "RegistryError",
    "TemplateNotFoundError",
    "SnippetNotFoundError",
    "TemplateExistsError",
    # Composer
    "PromptComposer",
    "MissingVariableError",
    "MissingSnippetError",
    "CircularInclusionError",
    # Loader
    "TemplateLoader",
    "load_default_templates",
]
