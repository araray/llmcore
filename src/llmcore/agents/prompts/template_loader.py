# src/llmcore/agents/prompts/template_loader.py
"""
Template Loader for Darwin Layer 2 Prompt Library.

This module provides utilities for loading prompt templates from
TOML files and registering the built-in default snippets.

The TemplateLoader handles:
- Loading templates from TOML configuration files
- Registering generic reusable snippets
- Registering templates with a PromptRegistry

NOTE (0.52.0): the four built-in cognitive templates were deleted — agent
prompts render from the grimoire control plane (bundled ``llmcore/cognitive/*``
spells). ``load_default_templates`` now loads snippets only.

References:
    - Technical Spec: Section 5.2 (Prompt Library Architecture)
    - Dossier: Step 2.3 (Template Loading)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import PromptRegistry

from .models import (
    PromptCategory,
    PromptSnippet,
    PromptTemplate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPLATE LOADER CLASS
# =============================================================================


class TemplateLoader:
    """
    Loads and registers prompt templates from various sources.

    The TemplateLoader can load templates from:
    - TOML configuration files
    - Built-in default definitions
    - Custom template dictionaries

    Example:
        >>> from llmcore.agents.prompts import TemplateLoader, PromptRegistry
        >>>
        >>> registry = PromptRegistry()
        >>> loader = TemplateLoader(registry)
        >>>
        >>> # Load from TOML file
        >>> loader.load_from_toml(Path("prompts/planning.toml"))
        >>>
        >>> # Load defaults
        >>> loader.load_defaults()
    """

    def __init__(self, registry: "PromptRegistry"):
        """
        Initialize the template loader.

        Args:
            registry: The prompt registry to load templates into
        """
        self.registry = registry

    def load_defaults(self) -> int:
        """
        Load the built-in default snippets (templates deleted in 0.52.0).

        Returns:
            Number of snippets loaded
        """
        return load_default_templates(self.registry)

    def load_from_toml(self, filepath: Path) -> int:
        """
        Load templates from a TOML file.

        Args:
            filepath: Path to the TOML file

        Returns:
            Number of templates loaded
        """
        try:
            import toml
        except ImportError:
            # Python 3.11+ can use tomllib for reading
            import tomllib

            with open(filepath, "rb") as f:
                data = tomllib.load(f)
        else:
            with open(filepath) as f:
                data = toml.load(f)

        count = 0

        # Load snippets first
        for key, snippet_data in data.get("snippets", {}).items():
            snippet = PromptSnippet(key=key, **snippet_data)
            self.registry.register_snippet(snippet)
            count += 1

        # Load templates
        for template_id, template_data in data.get("templates", {}).items():
            template = PromptTemplate(id=template_id, **template_data)
            self.registry.register_template(template)
            count += 1

        logger.info(f"Loaded {count} items from {filepath}")
        return count


# =============================================================================
# DEFAULT TEMPLATE DEFINITIONS
# =============================================================================


def load_default_templates(registry: "PromptRegistry") -> int:
    """
    Load built-in default snippets into a registry.

    .. deprecated:: 0.52.0
        The four cognitive default templates (``planning_prompt``,
        ``thinking_prompt``, ``reflection_prompt``, ``validation_prompt``)
        were DELETED — their variable sets had drifted from what the phases
        pass (the documented variable-mismatch source), and the cognitive
        cycle now renders exclusively from the grimoire control plane
        (bundled ``llmcore/cognitive/*`` spells). Only the generic reusable
        snippets remain, for hosts composing their own templates into an
        in-memory ``PromptRegistry``.

    Args:
        registry: The prompt registry to populate

    Returns:
        Number of snippets loaded
    """
    count = 0

    # =========================================================================
    # SNIPPETS
    # =========================================================================

    snippets = [
        PromptSnippet(
            key="agent_identity",
            content="You are an autonomous AI agent with advanced reasoning capabilities.",
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="react_framework",
            content=(
                "Use the ReAct (Reasoning + Acting) framework:\n"
                "1. Think: Analyze the situation and reason about next steps\n"
                "2. Act: Choose and execute an appropriate action\n"
                "3. Observe: Examine the results of your action\n"
                "4. Reflect: Learn from the outcome and adjust strategy"
            ),
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="tool_usage_instructions",
            content=(
                "When using tools:\n"
                "- Choose the most appropriate tool for the task\n"
                "- Provide all required parameters\n"
                "- Verify the tool output before proceeding\n"
                "- Handle errors gracefully"
            ),
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="step_format",
            content=(
                "Format each step as:\n"
                "THOUGHT: [Your reasoning about what to do next]\n"
                "ACTION: [The tool to use]\n"
                "ACTION_INPUT: [The input for the tool in JSON format]"
            ),
            category=PromptCategory.SNIPPET,
        ),
    ]

    for snippet in snippets:
        try:
            registry.register_snippet(snippet)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register snippet {snippet.key}: {e}")

    logger.info(f"Loaded {count} default snippets")
    return count


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TemplateLoader",
    "load_default_templates",
]
