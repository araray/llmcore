# tests/agents/prompts/test_prompt_library.py
"""
Unit tests for the Prompt Library System.

Tests cover:
- Core models (PromptTemplate, PromptVersion, PromptSnippet, PromptMetrics)
- PromptComposer (variable substitution, snippet inclusion)
- PromptRegistry (template management, rendering)
- Template loading from TOML

References:
    - Dossier: Step 2.1-2.3 (Prompt Library Implementation)
"""

import pytest

from llmcore.agents.prompts import (
    CircularInclusionError,
    MissingSnippetError,
    MissingVariableError,
    # Enums
    PromptCategory,
    # Composer
    PromptComposer,
    PromptMetrics,
    # Registry
    PromptRegistry,
    # Models
    PromptSnippet,
    PromptTemplate,
    PromptVariable,
    PromptVersion,
    TemplateNotFoundError,
    VersionStatus,
)

# =============================================================================
# MODEL TESTS
# =============================================================================


class TestPromptSnippet:
    """Tests for PromptSnippet model."""

    def test_create_snippet(self):
        """Test creating a basic snippet."""
        snippet = PromptSnippet(
            key="test_snippet", content="This is a test snippet.", category=PromptCategory.SNIPPET
        )

        assert snippet.key == "test_snippet"
        assert snippet.content == "This is a test snippet."
        assert snippet.category == PromptCategory.SNIPPET
        assert len(snippet.content_hash) == 16  # SHA-256 truncated

    def test_snippet_validation(self):
        """Test snippet key validation."""
        # Valid snake_case key
        snippet = PromptSnippet(key="valid_key", content="test")
        assert snippet.key == "valid_key"

        # Invalid key should raise error
        with pytest.raises(ValueError):
            PromptSnippet(key="Invalid-Key", content="test")

    def test_snippet_content_hash(self):
        """Test content hash generation."""
        snippet1 = PromptSnippet(key="test1", content="Same content")
        snippet2 = PromptSnippet(key="test2", content="Same content")
        snippet3 = PromptSnippet(key="test3", content="Different content")

        # Same content = same hash
        assert snippet1.content_hash == snippet2.content_hash
        # Different content = different hash
        assert snippet1.content_hash != snippet3.content_hash


class TestPromptVariable:
    """Tests for PromptVariable model."""

    def test_create_required_variable(self):
        """Test creating a required variable."""
        var = PromptVariable(name="goal", description="The agent's goal", required=True)

        assert var.name == "goal"
        assert var.required is True
        assert var.default_value is None

    def test_create_optional_variable(self):
        """Test creating an optional variable with default."""
        var = PromptVariable(
            name="timeout", description="Timeout in seconds", required=False, default_value="30"
        )

        assert var.required is False
        assert var.default_value == "30"

    def test_required_variable_cannot_have_default(self):
        """Test that required variables cannot have default values."""
        with pytest.raises(ValueError, match="Required variables cannot have a default_value"):
            PromptVariable(name="bad_var", required=True, default_value="something")


class TestPromptMetrics:
    """Tests for PromptMetrics model."""

    def test_create_metrics(self):
        """Test creating metrics instance."""
        metrics = PromptMetrics(version_id="v1")

        assert metrics.version_id == "v1"
        assert metrics.total_uses == 0
        assert metrics.success_rate is None

    def test_record_use_success(self):
        """Test recording a successful use."""
        metrics = PromptMetrics(version_id="v1")

        metrics.record_use(success=True, iterations=5, tokens=1000, latency_ms=250.0)

        assert metrics.total_uses == 1
        assert metrics.successful_uses == 1
        assert metrics.failed_uses == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_iterations_to_success == 5.0
        assert metrics.avg_tokens_used == 1000.0
        assert metrics.avg_latency_ms == 250.0

    def test_record_multiple_uses(self):
        """Test recording multiple uses and running averages."""
        metrics = PromptMetrics(version_id="v1")

        # First use
        metrics.record_use(success=True, tokens=1000)
        assert metrics.avg_tokens_used == 1000.0

        # Second use
        metrics.record_use(success=True, tokens=1200)
        assert metrics.avg_tokens_used == 1100.0  # Average of 1000 and 1200

        # Third use (failure)
        metrics.record_use(success=False, tokens=800)
        assert metrics.total_uses == 3
        assert metrics.successful_uses == 2
        assert metrics.failed_uses == 1
        assert metrics.success_rate == 2 / 3

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PromptMetrics(version_id="v1")

        metrics.record_use(success=True)
        metrics.record_use(success=True)
        metrics.record_use(success=False)
        metrics.record_use(success=True)

        assert metrics.success_rate == 0.75  # 3/4


class TestPromptVersion:
    """Tests for PromptVersion model."""

    def test_create_version(self):
        """Test creating a version."""
        version = PromptVersion(
            template_id="test_template",
            version_number=1,
            content="Test: {{variable}}",
            variables=[PromptVariable(name="variable", required=True)],
        )

        assert version.template_id == "test_template"
        assert version.version_number == 1
        assert version.status == VersionStatus.DRAFT
        assert len(version.variables) == 1

    def test_version_activation(self):
        """Test version activation."""
        version = PromptVersion(template_id="test", version_number=1, content="test")

        assert version.status == VersionStatus.DRAFT
        assert version.activated_at is None

        version.activate()

        assert version.status == VersionStatus.ACTIVE
        assert version.activated_at is not None

    def test_variable_names_property(self):
        """Test variable_names property."""
        version = PromptVersion(
            template_id="test",
            version_number=1,
            content="test",
            variables=[
                PromptVariable(name="var1", required=True),
                PromptVariable(name="var2", required=False),
            ],
        )

        assert version.variable_names == {"var1", "var2"}
        assert version.required_variable_names == {"var1"}


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_create_template(self):
        """Test creating a template."""
        template = PromptTemplate(
            id="test_template", name="Test Template", category=PromptCategory.PLANNING
        )

        assert template.id == "test_template"
        assert template.name == "Test Template"
        assert template.category == PromptCategory.PLANNING
        assert len(template.versions) == 0

    def test_add_version(self):
        """Test adding versions to a template."""
        template = PromptTemplate(id="test", name="Test", category=PromptCategory.CUSTOM)

        version1 = PromptVersion(template_id="test", version_number=1, content="v1")

        template.add_version(version1)

        assert len(template.versions) == 1
        assert template.latest_version == version1

    def test_set_active_version(self):
        """Test setting active version."""
        template = PromptTemplate(id="test", name="Test", category=PromptCategory.CUSTOM)

        version1 = PromptVersion(template_id="test", version_number=1, content="v1")
        version2 = PromptVersion(template_id="test", version_number=2, content="v2")

        template.add_version(version1)
        template.add_version(version2)

        # First activate version1
        template.set_active_version(version1.id)
        assert template.active_version == version1
        assert version1.status == VersionStatus.ACTIVE

        # Now activate version2 - version1 should become ARCHIVED
        template.set_active_version(version2.id)

        assert template.active_version == version2
        assert version2.status == VersionStatus.ACTIVE
        assert version1.status == VersionStatus.ARCHIVED


# =============================================================================
# COMPOSER TESTS
# =============================================================================


class TestPromptComposer:
    """Tests for PromptComposer."""

    def test_simple_variable_substitution(self):
        """Test simple variable substitution."""
        composer = PromptComposer()

        template = "Hello {{name}}, welcome to {{place}}!"
        variables = {"name": "Alice", "place": "Wonderland"}

        result = composer.render(template, variables)

        assert result == "Hello Alice, welcome to Wonderland!"

    def test_variable_with_default(self):
        """Test variable with default value."""
        composer = PromptComposer()

        template = "Hello {{name|World}}!"

        # Without providing variable
        result1 = composer.render(template, {}, validate=False)
        assert result1 == "Hello World!"

        # With providing variable
        result2 = composer.render(template, {"name": "Alice"}, validate=False)
        assert result2 == "Hello Alice!"

    def test_missing_required_variable(self):
        """Test error when required variable is missing."""
        composer = PromptComposer()

        template = "Goal: {{goal}}"

        with pytest.raises(MissingVariableError, match="Missing required variables: goal"):
            composer.render(template, {}, required_vars={"goal"})

    def test_snippet_inclusion(self):
        """Test snippet inclusion."""
        composer = PromptComposer()

        # Register snippet
        snippet = PromptSnippet(key="greeting", content="Hello! I'm here to help.")
        composer.register_snippet(snippet)

        # Render template with snippet
        template = "{{@greeting}}\n\nTask: {{task}}"
        result = composer.render(template, {"task": "Analyze data"})

        assert result == "Hello! I'm here to help.\n\nTask: Analyze data"

    def test_missing_snippet(self):
        """Test error when snippet is missing."""
        composer = PromptComposer()

        template = "{{@missing_snippet}}"

        with pytest.raises(MissingSnippetError, match="Snippet not found: missing_snippet"):
            composer.render(template, {})

    def test_nested_snippet_expansion(self):
        """Test nested snippet expansion."""
        composer = PromptComposer()

        # Register snippets
        snippet1 = PromptSnippet(key="inner", content="Inner content")
        snippet2 = PromptSnippet(key="outer", content="Outer: {{@inner}}")

        composer.register_snippet(snippet1)
        composer.register_snippet(snippet2)

        # Render
        template = "{{@outer}}"
        result = composer.render(template, {})

        assert result == "Outer: Inner content"

    def test_circular_snippet_detection(self):
        """Test detection of circular snippet inclusion."""
        composer = PromptComposer()

        # Create circular reference
        snippet1 = PromptSnippet(key="a", content="A: {{@b}}")
        snippet2 = PromptSnippet(key="b", content="B: {{@a}}")

        composer.register_snippet(snippet1)
        composer.register_snippet(snippet2)

        template = "{{@a}}"

        with pytest.raises(CircularInclusionError):
            composer.render(template, {})

    def test_extract_variables(self):
        """Test extracting variable names from template."""
        composer = PromptComposer()

        template = "Goal: {{goal}}\nContext: {{context}}\nStatus: {{status|pending}}"

        variables = composer.extract_variables(template)

        assert variables == {"goal", "context", "status"}

    def test_extract_snippets(self):
        """Test extracting snippet references from template."""
        composer = PromptComposer()

        template = "{{@intro}}\n\nTask: {{task}}\n\n{{@outro}}"

        snippets = composer.extract_snippets(template)

        assert snippets == {"intro", "outro"}


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = PromptRegistry()

        assert len(registry.list_templates()) == 0
        assert len(registry.list_snippets()) == 0

    def test_create_template(self):
        """Test creating a template through registry."""
        registry = PromptRegistry()

        template = registry.create_template(
            template_id="test_prompt",
            name="Test Prompt",
            category=PromptCategory.CUSTOM,
            initial_content="Task: {{task}}",
            variables=[PromptVariable(name="task", required=True)],
        )

        assert template.id == "test_prompt"
        assert len(template.versions) == 1
        assert template.active_version is not None

    def test_get_template(self):
        """Test retrieving a template."""
        registry = PromptRegistry()

        registry.create_template(template_id="test", name="Test", category=PromptCategory.CUSTOM)

        template = registry.get_template("test")
        assert template.id == "test"

    def test_get_nonexistent_template(self):
        """Test error when getting nonexistent template."""
        registry = PromptRegistry()

        with pytest.raises(TemplateNotFoundError):
            registry.get_template("nonexistent")

    def test_register_snippet(self):
        """Test registering a snippet."""
        registry = PromptRegistry()

        snippet = PromptSnippet(key="test", content="Test content")
        registry.register_snippet(snippet)

        retrieved = registry.get_snippet("test")
        assert retrieved.key == "test"
        assert retrieved.content == "Test content"

    def test_render_template(self):
        """Test rendering a template through registry."""
        registry = PromptRegistry()

        # Create template
        registry.create_template(
            template_id="greeting",
            name="Greeting",
            category=PromptCategory.CUSTOM,
            initial_content="Hello {{name}}!",
            variables=[PromptVariable(name="name", required=True)],
        )

        # Render
        result = registry.render("greeting", {"name": "Alice"})

        assert result == "Hello Alice!"

    def test_render_with_snippet(self):
        """Test rendering template with snippet inclusion."""
        registry = PromptRegistry()

        # Register snippet
        snippet = PromptSnippet(key="intro", content="Welcome!")
        registry.register_snippet(snippet)

        # Create template
        registry.create_template(
            template_id="test",
            name="Test",
            category=PromptCategory.CUSTOM,
            initial_content="{{@intro}}\n\nTask: {{task}}",
        )

        # Render
        result = registry.render("test", {"task": "Process data"})

        assert result == "Welcome!\n\nTask: Process data"

    def test_create_new_version(self):
        """Test creating a new version for existing template."""
        registry = PromptRegistry()

        # Create initial template
        template = registry.create_template(
            template_id="test",
            name="Test",
            category=PromptCategory.CUSTOM,
            initial_content="Version 1",
        )

        # Create new version
        v2 = registry.create_version(
            template_id="test", content="Version 2", change_description="Updated content"
        )

        assert v2.version_number == 2
        assert len(template.versions) == 2

    def test_activate_version(self):
        """Test activating a specific version."""
        registry = PromptRegistry()

        # Create template with initial version
        registry.create_template(
            template_id="test", name="Test", category=PromptCategory.CUSTOM, initial_content="v1"
        )

        # Create second version
        v2 = registry.create_version(template_id="test", content="v2")

        # Activate v2
        registry.activate_version("test", v2.id)

        template = registry.get_template("test")
        assert template.active_version == v2

    def test_list_templates_by_category(self):
        """Test listing templates filtered by category."""
        registry = PromptRegistry()

        registry.create_template("t1", "T1", PromptCategory.PLANNING)
        registry.create_template("t2", "T2", PromptCategory.REASONING)
        registry.create_template("t3", "T3", PromptCategory.PLANNING)

        planning_templates = registry.list_templates(category=PromptCategory.PLANNING)

        assert len(planning_templates) == 2
        assert all(t.category == PromptCategory.PLANNING for t in planning_templates)

    def test_record_metrics(self):
        """Test recording usage metrics."""
        registry = PromptRegistry()

        # Create template
        template = registry.create_template(
            template_id="test", name="Test", category=PromptCategory.CUSTOM, initial_content="test"
        )

        version_id = template.active_version.id

        # Record uses
        registry.record_use(version_id, success=True, tokens=1000)
        registry.record_use(version_id, success=True, tokens=1200)
        registry.record_use(version_id, success=False, tokens=800)

        # Get metrics
        metrics = registry.get_metrics(version_id)

        assert metrics.total_uses == 3
        assert metrics.successful_uses == 2
        assert metrics.failed_uses == 1
        assert metrics.success_rate == 2 / 3


# =============================================================================
# TEMPLATE LOADER TESTS
# =============================================================================


class TestTemplateLoader:
    """Tests for TemplateLoader."""

    def test_load_default_templates(self):
        """Test loading built-in templates."""
        registry = PromptRegistry.with_defaults()

        # Should have loaded planning, thinking, reflection, validation templates
        templates = registry.list_templates()

        # At minimum, should have some templates loaded
        assert len(templates) > 0

        # Check for expected template IDs
        template_ids = [t.id for t in templates]

        # These are defined in our TOML files
        assert "planning_prompt" in template_ids
        assert "thinking_prompt" in template_ids
        assert "reflection_prompt" in template_ids
        assert "validation_prompt" in template_ids

    def test_loaded_templates_have_variables(self):
        """Test that loaded templates have proper variable definitions."""
        registry = PromptRegistry.with_defaults()

        template = registry.get_template("planning_prompt")
        version = template.active_version

        assert version is not None
        assert len(version.variables) > 0

        # Should have 'goal' variable
        var_names = [v.name for v in version.variables]
        assert "goal" in var_names

    def test_loaded_snippets_available(self):
        """Test that snippets are loaded and available."""
        registry = PromptRegistry.with_defaults()

        # Should have loaded common snippets
        snippets = registry.list_snippets()

        # At minimum, should have some snippets
        assert len(snippets) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPromptLibraryIntegration:
    """Integration tests for the complete prompt library system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: create, version, render."""
        registry = PromptRegistry()

        # 1. Register reusable snippet
        snippet = PromptSnippet(
            key="safety_reminder", content="Remember: Always execute code in sandbox."
        )
        registry.register_snippet(snippet)

        # 2. Create template with snippet
        template = registry.create_template(
            template_id="code_execution",
            name="Code Execution Prompt",
            category=PromptCategory.TOOL_USE,
            initial_content="{{@safety_reminder}}\n\nExecute: {{code}}",
            variables=[PromptVariable(name="code", required=True)],
        )

        # 3. Render template
        result = registry.render("code_execution", {"code": "print('hello')"})

        assert "Remember: Always execute code in sandbox." in result
        assert "print('hello')" in result

        # 4. Create new version
        v2 = registry.create_version(
            template_id="code_execution",
            content="{{@safety_reminder}}\n\nCode:\n```\n{{code}}\n```",
            change_description="Added code block formatting",
        )

        # 5. Activate new version
        registry.activate_version("code_execution", v2.id)

        # 6. Render with new version
        result2 = registry.render("code_execution", {"code": "print('world')"})

        assert "```" in result2
        assert "print('world')" in result2

        # 7. Record metrics
        version_id = v2.id
        registry.record_use(version_id, success=True, tokens=500)

        metrics = registry.get_metrics(version_id)
        assert metrics.total_uses == 1
        assert metrics.successful_uses == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
