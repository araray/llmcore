# tests/agents/sandbox/images/test_selector.py
"""
Tests for the ImageSelector class.

Tests cover:
    - Task-based image selection
    - Runtime-based image selection
    - Capability-based image selection
    - Custom configuration
    - Access mode determination
"""

import pytest

from llmcore.agents.sandbox.images import (
    AccessMode,
    ImageCapability,
    ImageRegistry,
    ImageSelector,
    SelectionConfig,
    SelectionResult,
    TASK_CAPABILITY_MAP,
    TASK_IMAGE_MAP,
    RUNTIME_IMAGE_MAP,
)


# ==============================================================================
# Basic Selection Tests
# ==============================================================================


class TestBasicSelection:
    """Tests for basic image selection."""

    def test_select_for_research_task(self, default_selector: ImageSelector):
        """Test selecting image for research task."""
        result = default_selector.select_for_task("research")
        assert result.image == "llmcore-sandbox-research:1.0.0"
        assert result.reason == "task_mapping:research"
        assert result.access_mode == AccessMode.FULL

    def test_select_for_websearch_task(self, default_selector: ImageSelector):
        """Test selecting image for websearch task."""
        result = default_selector.select_for_task("websearch")
        assert result.image == "llmcore-sandbox-websearch:1.0.0"
        assert result.reason == "task_mapping:websearch"
        assert result.access_mode == AccessMode.FULL

    def test_select_for_python_dev(self, default_selector: ImageSelector):
        """Test selecting image for Python development."""
        result = default_selector.select_for_task("python_dev")
        assert result.image == "llmcore-sandbox-python:1.0.0"
        assert "python" in result.reason.lower()

    def test_select_for_shell_task(self, default_selector: ImageSelector):
        """Test selecting image for shell task."""
        result = default_selector.select_for_task("shell")
        assert result.image == "llmcore-sandbox-shell:1.0.0"

    def test_select_for_unknown_task(self, default_selector: ImageSelector):
        """Test selecting image for unknown task falls back to default."""
        result = default_selector.select_for_task("unknown_task")
        assert result.reason == "default"


# ==============================================================================
# Runtime Selection Tests
# ==============================================================================


class TestRuntimeSelection:
    """Tests for runtime-based image selection."""

    def test_select_for_python_runtime(self, default_selector: ImageSelector):
        """Test selecting image for Python runtime."""
        result = default_selector.select_for_runtime("python")
        assert result.image == "llmcore-sandbox-python:1.0.0"

    def test_select_for_python3_runtime(self, default_selector: ImageSelector):
        """Test selecting image for python3 runtime."""
        result = default_selector.select_for_runtime("python3")
        assert result.image == "llmcore-sandbox-python:1.0.0"

    def test_select_for_nodejs_runtime(self, default_selector: ImageSelector):
        """Test selecting image for Node.js runtime."""
        result = default_selector.select_for_runtime("nodejs")
        assert result.image == "llmcore-sandbox-nodejs:1.0.0"

    def test_select_for_node_runtime(self, default_selector: ImageSelector):
        """Test selecting image for node runtime."""
        result = default_selector.select_for_runtime("node")
        assert result.image == "llmcore-sandbox-nodejs:1.0.0"

    def test_select_for_bash_runtime(self, default_selector: ImageSelector):
        """Test selecting image for bash runtime."""
        result = default_selector.select_for_runtime("bash")
        assert result.image == "llmcore-sandbox-shell:1.0.0"

    def test_select_for_unknown_runtime(self, default_selector: ImageSelector):
        """Test selecting image for unknown runtime falls back."""
        result = default_selector.select_for_runtime("unknown_runtime")
        assert result.reason == "default"


# ==============================================================================
# Capability Selection Tests
# ==============================================================================


class TestCapabilitySelection:
    """Tests for capability-based image selection."""

    def test_select_for_python_capability(self, default_selector: ImageSelector):
        """Test selecting image by Python capability."""
        result = default_selector.select_for_capabilities(
            {ImageCapability.PYTHON}
        )
        assert result.manifest is not None
        assert ImageCapability.PYTHON in result.manifest.capabilities

    def test_select_for_multiple_capabilities(self, default_selector: ImageSelector):
        """Test selecting image by multiple capabilities."""
        result = default_selector.select_for_capabilities(
            {ImageCapability.PYTHON, ImageCapability.NETWORK}
        )
        # Should match research or websearch
        assert result.manifest is not None
        assert ImageCapability.PYTHON in result.manifest.capabilities
        assert ImageCapability.NETWORK in result.manifest.capabilities

    def test_select_for_research_capabilities(self, default_selector: ImageSelector):
        """Test selecting image by research capabilities."""
        result = default_selector.select_for_capabilities(
            {ImageCapability.RESEARCH}
        )
        assert result.manifest is not None
        assert ImageCapability.RESEARCH in result.manifest.capabilities


# ==============================================================================
# Task with Runtime Override Tests
# ==============================================================================


class TestTaskWithRuntime:
    """Tests for task selection with runtime override."""

    def test_task_with_runtime_prefers_task(self, default_selector: ImageSelector):
        """Test that task mapping takes precedence."""
        result = default_selector.select_for_task(
            task_type="research",
            runtime="nodejs"
        )
        # Research task mapping should win
        assert result.image == "llmcore-sandbox-research:1.0.0"

    def test_unknown_task_with_runtime(self, default_selector: ImageSelector):
        """Test unknown task falls back to runtime."""
        result = default_selector.select_for_task(
            task_type="unknown_task",
            runtime="python"
        )
        # Should use Python runtime mapping
        assert result.image == "llmcore-sandbox-python:1.0.0"


# ==============================================================================
# Custom Configuration Tests
# ==============================================================================


class TestCustomConfiguration:
    """Tests for custom selector configuration."""

    def test_custom_task_override(self, custom_selector: ImageSelector):
        """Test custom task-to-image mapping."""
        result = custom_selector.select_for_task("custom_task")
        assert result.image == "custom-image:1.0.0"

    def test_custom_runtime_override(self, custom_selector: ImageSelector):
        """Test custom runtime-to-image mapping."""
        result = custom_selector.select_for_runtime("custom_runtime")
        assert result.image == "custom-runtime-image:1.0.0"

    def test_custom_default_image(self):
        """Test custom default image."""
        config = SelectionConfig(
            default_image="my-custom-default:2.0.0"
        )
        registry = ImageRegistry()
        selector = ImageSelector(registry, config)

        result = selector.select_for_task("unknown_task")
        assert result.image == "my-custom-default:2.0.0"

    def test_add_task_mapping(self, default_selector: ImageSelector):
        """Test adding task mapping at runtime."""
        default_selector.add_task_mapping("new_task", "new-image:1.0.0")

        result = default_selector.select_for_task("new_task")
        assert result.image == "new-image:1.0.0"

    def test_add_runtime_mapping(self, default_selector: ImageSelector):
        """Test adding runtime mapping at runtime."""
        default_selector.add_runtime_mapping("go", "golang-image:1.0.0")

        result = default_selector.select_for_runtime("go")
        assert result.image == "golang-image:1.0.0"


# ==============================================================================
# Access Mode Tests
# ==============================================================================


class TestAccessMode:
    """Tests for access mode determination."""

    def test_restricted_by_default(self, default_selector: ImageSelector):
        """Test that default access mode is restricted."""
        result = default_selector.select_for_task("python_dev")
        assert result.access_mode == AccessMode.RESTRICTED

    def test_full_access_for_research(self, default_selector: ImageSelector):
        """Test research images get full access."""
        result = default_selector.select_for_task("research")
        assert result.access_mode == AccessMode.FULL

    def test_full_access_for_websearch(self, default_selector: ImageSelector):
        """Test websearch images get full access."""
        result = default_selector.select_for_task("websearch")
        assert result.access_mode == AccessMode.FULL

    def test_restricted_for_shell(self, default_selector: ImageSelector):
        """Test shell images get restricted access."""
        result = default_selector.select_for_task("shell")
        assert result.access_mode == AccessMode.RESTRICTED


# ==============================================================================
# Selection Result Tests
# ==============================================================================


class TestSelectionResult:
    """Tests for SelectionResult object."""

    def test_result_has_manifest(self, default_selector: ImageSelector):
        """Test that result includes manifest when available."""
        result = default_selector.select_for_task("python_dev")
        assert result.manifest is not None
        assert result.manifest.name == "llmcore-sandbox-python"

    def test_result_has_reason(self, default_selector: ImageSelector):
        """Test that result includes selection reason."""
        result = default_selector.select_for_task("research")
        assert "research" in result.reason.lower()

    def test_result_tracks_alternatives(self, default_selector: ImageSelector):
        """Test that result may track alternatives."""
        result = default_selector.select_for_task("unknown")
        # Alternatives list exists even if empty
        assert isinstance(result.alternatives, list)


# ==============================================================================
# Helper Method Tests
# ==============================================================================


class TestHelperMethods:
    """Tests for selector helper methods."""

    def test_get_task_types(self, default_selector: ImageSelector):
        """Test getting list of known task types."""
        task_types = default_selector.get_task_types()
        assert "research" in task_types
        assert "websearch" in task_types
        assert "python_dev" in task_types

    def test_get_runtimes(self, default_selector: ImageSelector):
        """Test getting list of known runtimes."""
        runtimes = default_selector.get_runtimes()
        assert "python" in runtimes
        assert "nodejs" in runtimes
        assert "bash" in runtimes

    def test_get_image_for_task(self, default_selector: ImageSelector):
        """Test getting image name for task."""
        image = default_selector.get_image_for_task("research")
        assert image == "llmcore-sandbox-research:1.0.0"

    def test_get_image_for_runtime(self, default_selector: ImageSelector):
        """Test getting image name for runtime."""
        image = default_selector.get_image_for_runtime("python")
        assert image == "llmcore-sandbox-python:1.0.0"

    def test_get_image_for_unknown_task(self, default_selector: ImageSelector):
        """Test getting image for unknown task returns None."""
        image = default_selector.get_image_for_task("unknown")
        assert image is None


# ==============================================================================
# Mapping Constants Tests
# ==============================================================================


class TestMappingConstants:
    """Tests for mapping constant dictionaries."""

    def test_task_capability_map_has_research(self):
        """Test task capability map includes research."""
        assert "research" in TASK_CAPABILITY_MAP
        caps = TASK_CAPABILITY_MAP["research"]
        assert ImageCapability.RESEARCH in caps

    def test_task_capability_map_has_websearch(self):
        """Test task capability map includes websearch."""
        assert "websearch" in TASK_CAPABILITY_MAP
        caps = TASK_CAPABILITY_MAP["websearch"]
        assert ImageCapability.WEBSEARCH in caps

    def test_task_image_map_has_expected_tasks(self):
        """Test task image map has expected tasks."""
        expected = ["research", "websearch", "python_dev", "nodejs_dev", "shell"]
        for task in expected:
            assert task in TASK_IMAGE_MAP

    def test_runtime_image_map_has_expected_runtimes(self):
        """Test runtime image map has expected runtimes."""
        expected = ["python", "python3", "nodejs", "node", "bash", "shell"]
        for runtime in expected:
            assert runtime in RUNTIME_IMAGE_MAP


# ==============================================================================
# Case Insensitivity Tests
# ==============================================================================


class TestCaseInsensitivity:
    """Tests for case-insensitive task and runtime handling."""

    def test_task_case_insensitive(self, default_selector: ImageSelector):
        """Test task names are case insensitive."""
        result1 = default_selector.select_for_task("RESEARCH")
        result2 = default_selector.select_for_task("Research")
        result3 = default_selector.select_for_task("research")

        assert result1.image == result2.image == result3.image

    def test_runtime_case_insensitive(self, default_selector: ImageSelector):
        """Test runtime names are case insensitive."""
        result1 = default_selector.select_for_runtime("PYTHON")
        result2 = default_selector.select_for_runtime("Python")
        result3 = default_selector.select_for_runtime("python")

        assert result1.image == result2.image == result3.image

    def test_whitespace_handling(self, default_selector: ImageSelector):
        """Test whitespace is stripped from input."""
        result = default_selector.select_for_task("  research  ")
        assert result.image == "llmcore-sandbox-research:1.0.0"
