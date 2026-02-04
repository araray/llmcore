# src/llmcore/agents/prompts/composer.py
"""
Prompt Composer for template rendering and composition.

This module provides the PromptComposer class which handles:
- Variable substitution ({{variable_name}})
- Snippet inclusion ({{@snippet_key}})
- Nested composition
- Validation of required variables

The composer uses a simple templating syntax that is LLM-friendly and easy to read.

Template Syntax:
    {{variable_name}}       - Variable placeholder
    {{@snippet_key}}        - Snippet inclusion
    {{variable_name|default}} - Variable with default value

Example:
    >>> from llmcore.agents.prompts import PromptComposer, PromptSnippet
    >>>
    >>> composer = PromptComposer()
    >>>
    >>> # Register snippets
    >>> composer.register_snippet(PromptSnippet(
    ...     key="greeting",
    ...     content="Hello! I'm here to help."
    ... ))
    >>>
    >>> # Render template
    >>> template = "{{@greeting}}\\n\\nTask: {{task}}"
    >>> result = composer.render(template, {"task": "Analyze data"})
    >>> print(result)
    Hello! I'm here to help.

    Task: Analyze data

References:
    - Technical Spec: Section 5.2.2 (Template Composition)
    - Dossier: Step 2.2 (Prompt Registry & Composer)
"""

import logging
import re
from typing import Dict, List, Optional, Set

from .models import PromptSnippet

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ComposerError(Exception):
    """Base exception for composer errors."""

    pass


class MissingVariableError(ComposerError):
    """Raised when a required variable is not provided."""

    pass


class MissingSnippetError(ComposerError):
    """Raised when a referenced snippet is not found."""

    pass


class CircularInclusionError(ComposerError):
    """Raised when snippet inclusion creates a circular dependency."""

    pass


# =============================================================================
# PROMPT COMPOSER
# =============================================================================


class PromptComposer:
    """
    Renders prompt templates with variable substitution and snippet inclusion.

    The composer handles two types of placeholders:
    1. Variables: {{variable_name}} or {{variable_name|default_value}}
    2. Snippets: {{@snippet_key}}

    Variables are substituted with provided values, while snippets are
    recursively expanded from registered snippet content.

    Example:
        >>> composer = PromptComposer()
        >>>
        >>> # Register snippets
        >>> composer.register_snippet(PromptSnippet(
        ...     key="instructions",
        ...     content="Follow these guidelines..."
        ... ))
        >>>
        >>> # Render with variables and snippets
        >>> template = "Goal: {{goal}}\\n{{@instructions}}"
        >>> result = composer.render(template, {"goal": "Process data"})

    Attributes:
        _snippets: Dictionary of registered snippets
    """

    # Regex patterns for parsing
    VARIABLE_PATTERN = re.compile(r"\{\{([a-z][a-z0-9_]*?)(?:\|([^}]*))?\}\}")
    SNIPPET_PATTERN = re.compile(r"\{\{@([a-z][a-z0-9_]*)\}\}")

    def __init__(self):
        """Initialize the composer with empty snippet registry."""
        self._snippets: dict[str, PromptSnippet] = {}

    def register_snippet(self, snippet: PromptSnippet) -> None:
        """
        Register a snippet for inclusion in templates.

        Args:
            snippet: The snippet to register
        """
        self._snippets[snippet.key] = snippet
        logger.debug(f"Registered snippet: {snippet.key}")

    def register_snippets(self, snippets: list[PromptSnippet]) -> None:
        """
        Register multiple snippets at once.

        Args:
            snippets: List of snippets to register
        """
        for snippet in snippets:
            self.register_snippet(snippet)

    def unregister_snippet(self, key: str) -> None:
        """
        Remove a snippet from the registry.

        Args:
            key: The snippet key to remove
        """
        if key in self._snippets:
            del self._snippets[key]
            logger.debug(f"Unregistered snippet: {key}")

    def get_snippet(self, key: str) -> PromptSnippet | None:
        """
        Get a registered snippet by key.

        Args:
            key: The snippet key

        Returns:
            The snippet if found, None otherwise
        """
        return self._snippets.get(key)

    def list_snippets(self) -> list[str]:
        """List all registered snippet keys."""
        return list(self._snippets.keys())

    def extract_variables(self, template: str) -> set[str]:
        """
        Extract all variable names from a template.

        Args:
            template: The template content

        Returns:
            Set of variable names found in the template
        """
        variables = set()
        for match in self.VARIABLE_PATTERN.finditer(template):
            var_name = match.group(1)
            variables.add(var_name)
        return variables

    def extract_snippets(self, template: str) -> set[str]:
        """
        Extract all snippet references from a template.

        Args:
            template: The template content

        Returns:
            Set of snippet keys referenced in the template
        """
        snippets = set()
        for match in self.SNIPPET_PATTERN.finditer(template):
            snippet_key = match.group(1)
            snippets.add(snippet_key)
        return snippets

    def validate_variables(
        self, template: str, variables: dict[str, str], required_vars: set[str] | None = None
    ) -> None:
        """
        Validate that all required variables are provided.

        Args:
            template: The template content
            variables: Provided variable values
            required_vars: Optional set of required variable names.
                          If None, all variables without defaults are required.

        Raises:
            MissingVariableError: If required variables are missing
        """
        # Extract variables with their defaults
        template_vars = {}
        for match in self.VARIABLE_PATTERN.finditer(template):
            var_name = match.group(1)
            default_value = match.group(2)  # May be None
            template_vars[var_name] = default_value

        # Determine which variables are actually required
        if required_vars is None:
            # Variables without defaults are required
            required_vars = {name for name, default in template_vars.items() if default is None}

        # Check for missing required variables
        missing = required_vars - set(variables.keys())
        if missing:
            raise MissingVariableError(f"Missing required variables: {', '.join(sorted(missing))}")

    def _expand_snippets(self, template: str, visited: set[str] | None = None) -> str:
        """
        Recursively expand all snippet inclusions in a template.

        Args:
            template: The template content
            visited: Set of already visited snippet keys (for circular detection)

        Returns:
            Template with all snippets expanded

        Raises:
            MissingSnippetError: If a referenced snippet is not found
            CircularInclusionError: If circular snippet inclusion is detected
        """
        if visited is None:
            visited = set()

        result = template

        # Find all snippet references
        for match in self.SNIPPET_PATTERN.finditer(template):
            snippet_key = match.group(1)
            placeholder = match.group(0)  # Full {{@snippet_key}}

            # Check for circular inclusion
            if snippet_key in visited:
                raise CircularInclusionError(f"Circular snippet inclusion detected: {snippet_key}")

            # Get snippet content
            snippet = self._snippets.get(snippet_key)
            if not snippet:
                raise MissingSnippetError(f"Snippet not found: {snippet_key}")

            # Recursively expand nested snippets
            visited_copy = visited.copy()
            visited_copy.add(snippet_key)
            expanded_content = self._expand_snippets(snippet.content, visited_copy)

            # Replace placeholder with expanded content
            result = result.replace(placeholder, expanded_content)

        return result

    def _substitute_variables(self, template: str, variables: dict[str, str]) -> str:
        """
        Substitute all variable placeholders with provided values.

        Args:
            template: The template content (with snippets already expanded)
            variables: Dictionary of variable values

        Returns:
            Template with all variables substituted
        """
        result = template

        for match in self.VARIABLE_PATTERN.finditer(template):
            var_name = match.group(1)
            default_value = match.group(2)
            placeholder = match.group(0)  # Full {{var}} or {{var|default}}

            # Get value (provided or default)
            value = variables.get(var_name)
            if value is None:
                value = default_value if default_value is not None else ""

            # Replace placeholder with value
            result = result.replace(placeholder, value)

        return result

    def render(
        self,
        template: str,
        variables: dict[str, str] | None = None,
        required_vars: set[str] | None = None,
        validate: bool = True,
    ) -> str:
        """
        Render a template with variable substitution and snippet expansion.

        Rendering happens in two phases:
        1. Snippet expansion: {{@snippet_key}} → snippet content (recursive)
        2. Variable substitution: {{variable}} → value

        Args:
            template: The template content to render
            variables: Dictionary of variable values (default: {})
            required_vars: Optional set of required variable names
            validate: Whether to validate required variables (default: True)

        Returns:
            Fully rendered template string

        Raises:
            MissingVariableError: If required variables are missing
            MissingSnippetError: If referenced snippet is not found
            CircularInclusionError: If circular snippet inclusion is detected

        Example:
            >>> composer = PromptComposer()
            >>> composer.register_snippet(PromptSnippet(
            ...     key="intro",
            ...     content="Welcome to the system."
            ... ))
            >>> template = "{{@intro}}\\n\\nGoal: {{goal}}"
            >>> result = composer.render(template, {"goal": "Analyze data"})
        """
        variables = variables or {}

        # Phase 1: Expand snippets
        try:
            expanded = self._expand_snippets(template)
        except (MissingSnippetError, CircularInclusionError) as e:
            logger.error(f"Snippet expansion failed: {e}")
            raise

        # Phase 2: Validate variables (if requested)
        if validate:
            try:
                self.validate_variables(expanded, variables, required_vars)
            except MissingVariableError as e:
                logger.error(f"Variable validation failed: {e}")
                raise

        # Phase 3: Substitute variables
        result = self._substitute_variables(expanded, variables)

        logger.debug(f"Rendered template: {len(template)} → {len(result)} characters")
        return result

    def clear_snippets(self) -> None:
        """Clear all registered snippets."""
        self._snippets.clear()
        logger.debug("Cleared all snippets")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CircularInclusionError",
    "ComposerError",
    "MissingSnippetError",
    "MissingVariableError",
    "PromptComposer",
]
