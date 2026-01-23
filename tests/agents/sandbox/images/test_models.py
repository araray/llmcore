# tests/agents/sandbox/images/test_models.py
"""
Tests for the images data models.

Tests cover:
    - ImageTier enum
    - ImageCapability enum
    - AccessMode enum
    - ResourceLimits dataclass
    - ImageManifest dataclass
    - ImageMetadata dataclass
"""

from datetime import datetime
from typing import Any, Dict

import pytest

from llmcore.agents.sandbox.images import (
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageMetadata,
    ImageTier,
    ResourceLimits,
    BUILTIN_MANIFESTS,
    BASE_IMAGE_MANIFEST,
    PYTHON_IMAGE_MANIFEST,
    NODEJS_IMAGE_MANIFEST,
    SHELL_IMAGE_MANIFEST,
    RESEARCH_IMAGE_MANIFEST,
    WEBSEARCH_IMAGE_MANIFEST,
)


# ==============================================================================
# ImageTier Tests
# ==============================================================================


class TestImageTier:
    """Tests for ImageTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert ImageTier.BASE.value == "base"
        assert ImageTier.SPECIALIZED.value == "specialized"
        assert ImageTier.TASK.value == "task"

    def test_tier_from_string(self):
        """Test creating tier from string."""
        assert ImageTier("base") == ImageTier.BASE
        assert ImageTier("specialized") == ImageTier.SPECIALIZED
        assert ImageTier("task") == ImageTier.TASK

    def test_tier_invalid_value(self):
        """Test invalid tier value raises error."""
        with pytest.raises(ValueError):
            ImageTier("invalid")

    def test_tier_is_string_enum(self):
        """Test that tier is a string enum."""
        assert isinstance(ImageTier.BASE, str)
        assert ImageTier.BASE == "base"


# ==============================================================================
# ImageCapability Tests
# ==============================================================================


class TestImageCapability:
    """Tests for ImageCapability enum."""

    def test_core_capabilities(self):
        """Test core capability values."""
        assert ImageCapability.SHELL.value == "shell"
        assert ImageCapability.NETWORK.value == "network"
        assert ImageCapability.FILESYSTEM.value == "filesystem"

    def test_language_capabilities(self):
        """Test language runtime capabilities."""
        assert ImageCapability.PYTHON.value == "python"
        assert ImageCapability.NODEJS.value == "nodejs"
        assert ImageCapability.RUST.value == "rust"

    def test_tool_capabilities(self):
        """Test tool capabilities."""
        assert ImageCapability.GIT.value == "git"
        assert ImageCapability.CURL.value == "curl"
        assert ImageCapability.JQ.value == "jq"

    def test_task_capabilities(self):
        """Test task-specific capabilities."""
        assert ImageCapability.RESEARCH.value == "research"
        assert ImageCapability.WEBSEARCH.value == "websearch"

    def test_capability_from_string(self):
        """Test creating capability from string."""
        assert ImageCapability("python") == ImageCapability.PYTHON
        assert ImageCapability("shell") == ImageCapability.SHELL

    def test_capability_invalid_value(self):
        """Test invalid capability raises error."""
        with pytest.raises(ValueError):
            ImageCapability("invalid_capability")

    def test_all_capabilities_count(self, all_capabilities):
        """Test total number of capabilities."""
        # Should have at least 15 capabilities
        assert len(all_capabilities) >= 15


# ==============================================================================
# AccessMode Tests
# ==============================================================================


class TestAccessMode:
    """Tests for AccessMode enum."""

    def test_access_mode_values(self):
        """Test access mode values."""
        assert AccessMode.RESTRICTED.value == "restricted"
        assert AccessMode.FULL.value == "full"

    def test_access_mode_from_string(self):
        """Test creating access mode from string."""
        assert AccessMode("restricted") == AccessMode.RESTRICTED
        assert AccessMode("full") == AccessMode.FULL


# ==============================================================================
# ResourceLimits Tests
# ==============================================================================


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_values(self, default_resource_limits: ResourceLimits):
        """Test default resource limit values."""
        assert default_resource_limits.memory_limit == "512m"
        assert default_resource_limits.cpu_limit == 1.0
        assert default_resource_limits.timeout_seconds == 600
        assert default_resource_limits.pids_limit == 100

    def test_custom_values(self, high_resource_limits: ResourceLimits):
        """Test custom resource limit values."""
        assert high_resource_limits.memory_limit == "4g"
        assert high_resource_limits.cpu_limit == 4.0
        assert high_resource_limits.timeout_seconds == 1800
        assert high_resource_limits.pids_limit == 500

    def test_to_dict(self, default_resource_limits: ResourceLimits):
        """Test serialization to dictionary."""
        data = default_resource_limits.to_dict()
        assert data["memory_limit"] == "512m"
        assert data["cpu_limit"] == 1.0
        assert data["timeout_seconds"] == 600
        assert data["pids_limit"] == 100

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "memory_limit": "2g",
            "cpu_limit": 2.0,
            "timeout_seconds": 300,
            "pids_limit": 200,
        }
        limits = ResourceLimits.from_dict(data)
        assert limits.memory_limit == "2g"
        assert limits.cpu_limit == 2.0
        assert limits.timeout_seconds == 300
        assert limits.pids_limit == 200

    def test_from_dict_with_defaults(self):
        """Test deserialization fills in defaults."""
        limits = ResourceLimits.from_dict({})
        assert limits.memory_limit == "512m"
        assert limits.cpu_limit == 1.0


# ==============================================================================
# ImageManifest Tests
# ==============================================================================


class TestImageManifest:
    """Tests for ImageManifest dataclass."""

    def test_basic_properties(self, sample_manifest: ImageManifest):
        """Test basic manifest properties."""
        assert sample_manifest.name == "llmcore-sandbox-test"
        assert sample_manifest.version == "1.0.0"
        assert sample_manifest.tier == ImageTier.SPECIALIZED

    def test_full_name(self, sample_manifest: ImageManifest):
        """Test full_name property."""
        assert sample_manifest.full_name == "llmcore-sandbox-test:1.0.0"

    def test_tier_checks(
        self,
        base_manifest: ImageManifest,
        python_manifest: ImageManifest,
        research_manifest: ImageManifest,
    ):
        """Test tier check methods."""
        assert base_manifest.is_base is True
        assert base_manifest.is_specialized is False
        assert base_manifest.is_task is False

        assert python_manifest.is_base is False
        assert python_manifest.is_specialized is True
        assert python_manifest.is_task is False

        assert research_manifest.is_base is False
        assert research_manifest.is_specialized is False
        assert research_manifest.is_task is True

    def test_has_capability(self, python_manifest: ImageManifest):
        """Test has_capability method."""
        assert python_manifest.has_capability(ImageCapability.PYTHON) is True
        assert python_manifest.has_capability(ImageCapability.SHELL) is True
        assert python_manifest.has_capability(ImageCapability.NODEJS) is False
        assert python_manifest.has_capability(ImageCapability.BROWSER) is False

    def test_has_all_capabilities(self, python_manifest: ImageManifest):
        """Test has_all_capabilities method."""
        required = {ImageCapability.PYTHON, ImageCapability.SHELL}
        assert python_manifest.has_all_capabilities(required) is True

        required_missing = {ImageCapability.PYTHON, ImageCapability.BROWSER}
        assert python_manifest.has_all_capabilities(required_missing) is False

    def test_has_tool(self, python_manifest: ImageManifest):
        """Test has_tool method."""
        assert python_manifest.has_tool("python3") is True
        assert python_manifest.has_tool("pip") is True
        assert python_manifest.has_tool("npm") is False

    def test_to_dict(self, sample_manifest: ImageManifest):
        """Test serialization to dictionary."""
        data = sample_manifest.to_dict()
        assert data["name"] == "llmcore-sandbox-test"
        assert data["version"] == "1.0.0"
        assert data["tier"] == "specialized"
        assert "python" in data["capabilities"]
        assert "python3" in data["tools"]

    def test_from_dict(self, sample_manifest_dict: Dict[str, Any]):
        """Test deserialization from dictionary."""
        manifest = ImageManifest.from_dict(sample_manifest_dict)
        assert manifest.name == "llmcore-sandbox-test"
        assert manifest.version == "1.0.0"
        assert manifest.tier == ImageTier.SPECIALIZED
        assert ImageCapability.PYTHON in manifest.capabilities

    def test_from_dict_with_unknown_capabilities(self):
        """Test that unknown capabilities are ignored."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "tier": "base",
            "capabilities": ["shell", "unknown_capability"],
        }
        manifest = ImageManifest.from_dict(data)
        assert ImageCapability.SHELL in manifest.capabilities
        assert len(manifest.capabilities) == 1

    def test_from_dict_with_invalid_tier(self):
        """Test that invalid tier defaults to BASE."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "tier": "invalid_tier",
        }
        manifest = ImageManifest.from_dict(data)
        assert manifest.tier == ImageTier.BASE

    def test_roundtrip_serialization(self, sample_manifest: ImageManifest):
        """Test that to_dict/from_dict roundtrips correctly."""
        data = sample_manifest.to_dict()
        restored = ImageManifest.from_dict(data)
        assert restored.name == sample_manifest.name
        assert restored.version == sample_manifest.version
        assert restored.tier == sample_manifest.tier
        assert restored.capabilities == sample_manifest.capabilities


# ==============================================================================
# ImageMetadata Tests
# ==============================================================================


class TestImageMetadata:
    """Tests for ImageMetadata dataclass."""

    def test_properties_from_manifest(self, sample_metadata: ImageMetadata):
        """Test properties delegated to manifest."""
        assert sample_metadata.name == "llmcore-sandbox-test"
        assert sample_metadata.version == "1.0.0"
        assert sample_metadata.full_name == "llmcore-sandbox-test:1.0.0"
        assert sample_metadata.tier == ImageTier.SPECIALIZED

    def test_docker_metadata(self, sample_metadata: ImageMetadata):
        """Test Docker-specific metadata."""
        assert sample_metadata.docker_id == "sha256:abc123"
        assert sample_metadata.size_bytes == 100_000_000
        assert sample_metadata.available_locally is True

    def test_capabilities_property(self, sample_metadata: ImageMetadata):
        """Test capabilities property."""
        caps = sample_metadata.capabilities
        assert ImageCapability.PYTHON in caps
        assert ImageCapability.SHELL in caps

    def test_to_dict(self, sample_metadata: ImageMetadata):
        """Test serialization."""
        data = sample_metadata.to_dict()
        assert data["docker_id"] == "sha256:abc123"
        assert data["size_bytes"] == 100_000_000
        assert data["available_locally"] is True
        assert "manifest" in data


# ==============================================================================
# Builtin Manifests Tests
# ==============================================================================


class TestBuiltinManifests:
    """Tests for builtin image manifests."""

    def test_builtin_manifests_exist(self):
        """Test that all expected builtin manifests exist."""
        expected = [
            "llmcore-sandbox-base",
            "llmcore-sandbox-python",
            "llmcore-sandbox-nodejs",
            "llmcore-sandbox-shell",
            "llmcore-sandbox-research",
            "llmcore-sandbox-websearch",
        ]
        for name in expected:
            assert name in BUILTIN_MANIFESTS, f"Missing builtin: {name}"

    def test_base_manifest(self):
        """Test base image manifest."""
        assert BASE_IMAGE_MANIFEST.name == "llmcore-sandbox-base"
        assert BASE_IMAGE_MANIFEST.tier == ImageTier.BASE
        assert ImageCapability.SHELL in BASE_IMAGE_MANIFEST.capabilities
        assert BASE_IMAGE_MANIFEST.default_access_mode == AccessMode.RESTRICTED

    def test_python_manifest(self):
        """Test Python image manifest."""
        assert PYTHON_IMAGE_MANIFEST.name == "llmcore-sandbox-python"
        assert PYTHON_IMAGE_MANIFEST.tier == ImageTier.SPECIALIZED
        assert ImageCapability.PYTHON in PYTHON_IMAGE_MANIFEST.capabilities
        assert "python3" in PYTHON_IMAGE_MANIFEST.tools

    def test_nodejs_manifest(self):
        """Test Node.js image manifest."""
        assert NODEJS_IMAGE_MANIFEST.name == "llmcore-sandbox-nodejs"
        assert NODEJS_IMAGE_MANIFEST.tier == ImageTier.SPECIALIZED
        assert ImageCapability.NODEJS in NODEJS_IMAGE_MANIFEST.capabilities
        assert "node" in NODEJS_IMAGE_MANIFEST.tools

    def test_shell_manifest(self):
        """Test shell image manifest."""
        assert SHELL_IMAGE_MANIFEST.name == "llmcore-sandbox-shell"
        assert SHELL_IMAGE_MANIFEST.tier == ImageTier.SPECIALIZED
        assert ImageCapability.SHELL in SHELL_IMAGE_MANIFEST.capabilities
        assert "bash" in SHELL_IMAGE_MANIFEST.tools

    def test_research_manifest(self):
        """Test research image manifest."""
        assert RESEARCH_IMAGE_MANIFEST.name == "llmcore-sandbox-research"
        assert RESEARCH_IMAGE_MANIFEST.tier == ImageTier.TASK
        assert ImageCapability.RESEARCH in RESEARCH_IMAGE_MANIFEST.capabilities
        # Research needs network
        assert RESEARCH_IMAGE_MANIFEST.default_access_mode == AccessMode.FULL

    def test_websearch_manifest(self):
        """Test websearch image manifest."""
        assert WEBSEARCH_IMAGE_MANIFEST.name == "llmcore-sandbox-websearch"
        assert WEBSEARCH_IMAGE_MANIFEST.tier == ImageTier.TASK
        assert ImageCapability.WEBSEARCH in WEBSEARCH_IMAGE_MANIFEST.capabilities
        assert ImageCapability.BROWSER in WEBSEARCH_IMAGE_MANIFEST.capabilities
        # Websearch needs network
        assert WEBSEARCH_IMAGE_MANIFEST.default_access_mode == AccessMode.FULL

    def test_builtin_manifest_hierarchy(self):
        """Test that specialized images reference base."""
        assert PYTHON_IMAGE_MANIFEST.base_image == "llmcore-sandbox-base:1.0.0"
        assert NODEJS_IMAGE_MANIFEST.base_image == "llmcore-sandbox-base:1.0.0"
        assert SHELL_IMAGE_MANIFEST.base_image == "llmcore-sandbox-base:1.0.0"

    def test_task_images_reference_specialized(self):
        """Test that task images reference specialized images."""
        assert RESEARCH_IMAGE_MANIFEST.base_image == "llmcore-sandbox-python:1.0.0"
        assert WEBSEARCH_IMAGE_MANIFEST.base_image == "llmcore-sandbox-python:1.0.0"
