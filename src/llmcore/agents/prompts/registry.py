# src/llmcore/agents/prompts/registry.py
"""
Prompt Registry for centralized template management.

This module provides the PromptRegistry class which serves as the central
management system for all prompt templates, versions, and snippets. It provides:

- Template lifecycle management (create, update, activate, archive)
- Version management (create new versions, set active version)
- Snippet management (register, unregister)
- Rendering interface (combines registry + composer)
- Persistence (save/load from TOML files)

The registry is the primary interface for working with prompts in the agent system.

Example:
    >>> from llmcore.agents.prompts import PromptRegistry
    >>>
    >>> # Initialize with defaults
    >>> registry = PromptRegistry.with_defaults()
    >>>
    >>> # Render a template
    >>> prompt = registry.render(
    ...     template_id="planning_prompt",
    ...     variables={"goal": "Calculate prime numbers"}
    ... )
    >>>
    >>> # Create a new template
    >>> registry.create_template(
    ...     template_id="custom_prompt",
    ...     name="Custom Prompt",
    ...     category=PromptCategory.CUSTOM,
    ...     initial_content="Task: {{task}}\\n{{@instructions}}"
    ... )

References:
    - Technical Spec: Section 5.2.3 (Prompt Registry)
    - Dossier: Step 2.2 (Prompt Registry & Composer)
"""

import logging
from pathlib import Path

import toml

from .composer import PromptComposer
from .models import (
    PromptCategory,
    PromptMetrics,
    PromptSnippet,
    PromptTemplate,
    PromptVariable,
    PromptVersion,
    VersionStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class TemplateNotFoundError(RegistryError):
    """Raised when a template is not found."""

    pass


class SnippetNotFoundError(RegistryError):
    """Raised when a snippet is not found."""

    pass


class TemplateExistsError(RegistryError):
    """Raised when trying to create a template that already exists."""

    pass


# =============================================================================
# PROMPT REGISTRY
# =============================================================================


class PromptRegistry:
    """
    Central registry for managing prompt templates, versions, and snippets.

    The registry provides a high-level interface for:
    - Creating and managing templates
    - Creating and activating versions
    - Registering and managing snippets
    - Rendering templates with variables
    - Tracking metrics for versions
    - Persisting to/loading from TOML files

    Example:
        >>> registry = PromptRegistry()
        >>>
        >>> # Create a template
        >>> registry.create_template(
        ...     template_id="greeting",
        ...     name="Greeting Prompt",
        ...     category=PromptCategory.SYSTEM,
        ...     initial_content="Hello, {{name}}!"
        ... )
        >>>
        >>> # Render it
        >>> result = registry.render("greeting", {"name": "Alice"})
        >>> print(result)  # "Hello, Alice!"

    Attributes:
        _templates: Dictionary of templates by ID
        _snippets: Dictionary of snippets by key
        _metrics: Dictionary of metrics by version ID
        _composer: PromptComposer instance for rendering
    """

    def __init__(self):
        """Initialize an empty prompt registry."""
        self._templates: dict[str, PromptTemplate] = {}
        self._snippets: dict[str, PromptSnippet] = {}
        self._metrics: dict[str, PromptMetrics] = {}
        self._composer = PromptComposer()

    # =========================================================================
    # TEMPLATE MANAGEMENT
    # =========================================================================

    def register_template(self, template: PromptTemplate) -> None:
        """
        Register a template in the registry.

        Args:
            template: The template to register

        Raises:
            TemplateExistsError: If template ID already exists
        """
        if template.id in self._templates:
            raise TemplateExistsError(f"Template already exists: {template.id}")

        self._templates[template.id] = template
        logger.info(f"Registered template: {template.id}")

    def create_template(
        self,
        template_id: str,
        name: str,
        category: PromptCategory,
        description: str | None = None,
        tags: list[str] | None = None,
        initial_content: str | None = None,
        variables: list[PromptVariable] | None = None,
    ) -> PromptTemplate:
        """
        Create and register a new template with an initial version.

        Args:
            template_id: Unique identifier (snake_case)
            name: Human-readable name
            category: Prompt category
            description: Optional description
            tags: Optional tags
            initial_content: Optional content for version 1
            variables: Optional variable definitions

        Returns:
            The created template

        Raises:
            TemplateExistsError: If template ID already exists
        """
        # Create template
        template = PromptTemplate(
            id=template_id, name=name, category=category, description=description, tags=tags or []
        )

        # Create initial version if content provided
        if initial_content:
            version = PromptVersion(
                template_id=template_id,
                version_number=1,
                content=initial_content,
                variables=variables or [],
                status=VersionStatus.ACTIVE,
            )
            version.activate()
            template.add_version(version)
            template.set_active_version(version.id)

            # Initialize metrics
            metrics = PromptMetrics(version_id=version.id)
            self._metrics[version.id] = metrics

        # Register template
        self.register_template(template)
        return template

    def get_template(self, template_id: str) -> PromptTemplate:
        """
        Get a template by ID.

        Args:
            template_id: The template ID

        Returns:
            The template

        Raises:
            TemplateNotFoundError: If template not found
        """
        if template_id not in self._templates:
            raise TemplateNotFoundError(f"Template not found: {template_id}")
        return self._templates[template_id]

    def list_templates(
        self, category: PromptCategory | None = None, tags: list[str] | None = None
    ) -> list[PromptTemplate]:
        """
        List templates, optionally filtered by category and/or tags.

        Args:
            category: Optional category filter
            tags: Optional tags filter (template must have all tags)

        Returns:
            List of matching templates
        """
        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        if tags:
            tag_set = set(tags)
            templates = [t for t in templates if tag_set.issubset(set(t.tags))]

        return templates

    def delete_template(self, template_id: str) -> None:
        """
        Delete a template and all its versions.

        Args:
            template_id: The template ID to delete

        Raises:
            TemplateNotFoundError: If template not found
        """
        template = self.get_template(template_id)

        # Delete metrics for all versions
        for version in template.versions:
            if version.id in self._metrics:
                del self._metrics[version.id]

        # Delete template
        del self._templates[template_id]
        logger.info(f"Deleted template: {template_id}")

    # =========================================================================
    # VERSION MANAGEMENT
    # =========================================================================

    def create_version(
        self,
        template_id: str,
        content: str,
        variables: list[PromptVariable] | None = None,
        change_description: str | None = None,
        auto_activate: bool = False,
    ) -> PromptVersion:
        """
        Create a new version for a template.

        Args:
            template_id: The template ID
            content: The prompt content
            variables: Variable definitions
            change_description: What changed in this version
            auto_activate: Whether to activate this version immediately

        Returns:
            The created version

        Raises:
            TemplateNotFoundError: If template not found
        """
        template = self.get_template(template_id)

        # Determine version number
        if template.versions:
            version_number = max(v.version_number for v in template.versions) + 1
        else:
            version_number = 1

        # Create version
        version = PromptVersion(
            template_id=template_id,
            version_number=version_number,
            content=content,
            variables=variables or [],
            status=VersionStatus.DRAFT,
            change_description=change_description,
        )

        # Add to template
        template.add_version(version)

        # Initialize metrics
        metrics = PromptMetrics(version_id=version.id)
        self._metrics[version.id] = metrics

        # Activate if requested
        if auto_activate:
            template.set_active_version(version.id)

        logger.info(f"Created version {version_number} for template {template_id}")
        return version

    def activate_version(self, template_id: str, version_id: str) -> None:
        """
        Set a version as the active version for a template.

        Args:
            template_id: The template ID
            version_id: The version ID to activate

        Raises:
            TemplateNotFoundError: If template not found
            ValueError: If version not found in template
        """
        template = self.get_template(template_id)
        template.set_active_version(version_id)
        logger.info(f"Activated version {version_id} for template {template_id}")

    def get_version(self, template_id: str, version_id: str) -> PromptVersion:
        """
        Get a specific version from a template.

        Args:
            template_id: The template ID
            version_id: The version ID

        Returns:
            The version

        Raises:
            TemplateNotFoundError: If template not found
            ValueError: If version not found
        """
        template = self.get_template(template_id)
        version = template.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found in template {template_id}")
        return version

    # =========================================================================
    # SNIPPET MANAGEMENT
    # =========================================================================

    def register_snippet(self, snippet: PromptSnippet) -> None:
        """
        Register a snippet for use in templates.

        Args:
            snippet: The snippet to register
        """
        self._snippets[snippet.key] = snippet
        self._composer.register_snippet(snippet)
        logger.info(f"Registered snippet: {snippet.key}")

    def get_snippet(self, key: str) -> PromptSnippet:
        """
        Get a snippet by key.

        Args:
            key: The snippet key

        Returns:
            The snippet

        Raises:
            SnippetNotFoundError: If snippet not found
        """
        if key not in self._snippets:
            raise SnippetNotFoundError(f"Snippet not found: {key}")
        return self._snippets[key]

    def list_snippets(self, category: PromptCategory | None = None) -> list[PromptSnippet]:
        """
        List snippets, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of matching snippets
        """
        snippets = list(self._snippets.values())

        if category:
            snippets = [s for s in snippets if s.category == category]

        return snippets

    def delete_snippet(self, key: str) -> None:
        """
        Delete a snippet.

        Args:
            key: The snippet key

        Raises:
            SnippetNotFoundError: If snippet not found
        """
        if key not in self._snippets:
            raise SnippetNotFoundError(f"Snippet not found: {key}")

        del self._snippets[key]
        self._composer.unregister_snippet(key)
        logger.info(f"Deleted snippet: {key}")

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(
        self,
        template_id: str,
        variables: dict[str, str] | None = None,
        version_id: str | None = None,
        validate: bool = True,
    ) -> str:
        """
        Render a template with variable substitution.

        Uses the active version unless a specific version is requested.

        Args:
            template_id: The template ID to render
            variables: Dictionary of variable values
            version_id: Optional specific version ID (uses active if None)
            validate: Whether to validate required variables

        Returns:
            Rendered prompt string

        Raises:
            TemplateNotFoundError: If template not found
            ValueError: If template has no active version
            MissingVariableError: If required variables are missing
        """
        template = self.get_template(template_id)

        # Get version to render
        if version_id:
            version = template.get_version(version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found in template {template_id}")
        else:
            version = template.active_version
            if not version:
                raise ValueError(f"Template {template_id} has no active version")

        # Get required variables
        required_vars = version.required_variable_names if validate else None

        # Render using composer
        result = self._composer.render(
            template=version.content,
            variables=variables,
            required_vars=required_vars,
            validate=validate,
        )

        logger.debug(f"Rendered template {template_id} (version {version.version_number})")
        return result

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self, version_id: str) -> PromptMetrics:
        """
        Get metrics for a specific version.

        Args:
            version_id: The version ID

        Returns:
            Metrics for the version (creates empty metrics if not exists)
        """
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
        """
        Record usage metrics for a version.

        Args:
            version_id: The version ID
            success: Whether the use was successful
            iterations: Number of cognitive loop iterations
            tokens: Total tokens consumed
            latency_ms: End-to-end latency in milliseconds
            quality_score: Quality rating (0.0 to 1.0)
        """
        metrics = self.get_metrics(version_id)
        metrics.record_use(success, iterations, tokens, latency_ms, quality_score)
        logger.debug(f"Recorded use for version {version_id}: success={success}")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save_to_toml(self, filepath: Path) -> None:
        """
        Save the registry to a TOML file.

        Args:
            filepath: Path to save the TOML file
        """
        data = {"templates": {}, "snippets": {}}

        # Serialize templates
        for template_id, template in self._templates.items():
            data["templates"][template_id] = template.model_dump()

        # Serialize snippets
        for key, snippet in self._snippets.items():
            data["snippets"][key] = snippet.model_dump()

        # Write TOML
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            toml.dump(data, f)

        logger.info(f"Saved registry to {filepath}")

    @classmethod
    def load_from_toml(cls, filepath: Path) -> "PromptRegistry":
        """
        Load a registry from a TOML file.

        Args:
            filepath: Path to the TOML file

        Returns:
            Loaded registry
        """
        with open(filepath) as f:
            data = toml.load(f)

        registry = cls()

        # Load snippets first (templates may reference them)
        for key, snippet_data in data.get("snippets", {}).items():
            snippet = PromptSnippet(**snippet_data)
            registry.register_snippet(snippet)

        # Load templates
        for template_id, template_data in data.get("templates", {}).items():
            template = PromptTemplate(**template_data)
            registry._templates[template_id] = template

            # Initialize metrics for all versions
            for version in template.versions:
                registry._metrics[version.id] = PromptMetrics(version_id=version.id)

        logger.info(f"Loaded registry from {filepath}")
        return registry

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def with_defaults(cls) -> "PromptRegistry":
        """
        Create a registry with built-in default templates and snippets.

        Returns:
            Registry with default prompts loaded
        """
        from .template_loader import load_default_templates

        registry = cls()
        load_default_templates(registry)

        logger.info("Created registry with default templates")
        return registry


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PromptRegistry",
    "RegistryError",
    "SnippetNotFoundError",
    "TemplateExistsError",
    "TemplateNotFoundError",
]
