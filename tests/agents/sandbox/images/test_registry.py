# tests/agents/sandbox/images/test_registry.py
"""
Tests for the ImageRegistry class.

Tests cover:
    - Registry initialization
    - Image registration and lookup
    - Filtering and listing
    - Docker discovery
    - Error handling
"""

from unittest.mock import MagicMock, patch

import pytest

from llmcore.agents.sandbox.images import (
    ImageCapability,
    ImageManifest,
    ImageMetadata,
    ImageNotFoundError,
    ImageRegistry,
    ImageTier,
)

# ==============================================================================
# Registry Initialization Tests
# ==============================================================================


class TestRegistryInitialization:
    """Tests for ImageRegistry initialization."""

    def test_empty_registry(self, empty_registry: ImageRegistry):
        """Test creating empty registry."""
        assert empty_registry.count == 0
        assert empty_registry.last_refresh is None

    def test_builtin_registry(self, builtin_registry: ImageRegistry):
        """Test registry with builtins loaded."""
        # Should have all builtin images
        assert builtin_registry.count >= 6
        assert builtin_registry.has_image("llmcore-sandbox-base:1.0.0")
        assert builtin_registry.has_image("llmcore-sandbox-python:1.0.0")

    def test_custom_patterns(self):
        """Test registry with custom image patterns."""
        registry = ImageRegistry(
            image_patterns=["custom-image-*"],
            include_builtins=False,
        )
        # Patterns should be stored
        assert "custom-image-*" in registry._image_patterns


# ==============================================================================
# Image Registration Tests
# ==============================================================================


class TestImageRegistration:
    """Tests for image registration."""

    def test_register_image(self, empty_registry: ImageRegistry, sample_metadata: ImageMetadata):
        """Test registering an image."""
        empty_registry.register(sample_metadata)
        assert empty_registry.count == 1
        assert empty_registry.has_image(sample_metadata.full_name)

    def test_register_overwrites(self, empty_registry: ImageRegistry, sample_metadata: ImageMetadata):
        """Test that registering same image overwrites."""
        empty_registry.register(sample_metadata)

        # Modify and re-register
        sample_metadata.size_bytes = 999
        empty_registry.register(sample_metadata)

        assert empty_registry.count == 1
        retrieved = empty_registry.get(sample_metadata.full_name)
        assert retrieved.size_bytes == 999

    def test_unregister_image(self, populated_registry: ImageRegistry, sample_metadata: ImageMetadata):
        """Test unregistering an image."""
        full_name = sample_metadata.full_name
        assert populated_registry.has_image(full_name)

        result = populated_registry.unregister(full_name)
        assert result is True
        assert not populated_registry.has_image(full_name)

    def test_unregister_nonexistent(self, empty_registry: ImageRegistry):
        """Test unregistering nonexistent image returns False."""
        result = empty_registry.unregister("nonexistent:1.0.0")
        assert result is False

    def test_clear_registry(self, populated_registry: ImageRegistry):
        """Test clearing the registry."""
        initial_count = populated_registry.count
        assert initial_count > 0

        populated_registry.clear()
        # Builtins should be reloaded
        assert populated_registry.count >= 6


# ==============================================================================
# Image Lookup Tests
# ==============================================================================


class TestImageLookup:
    """Tests for image lookup operations."""

    def test_get_existing_image(self, builtin_registry: ImageRegistry):
        """Test getting an existing image."""
        metadata = builtin_registry.get("llmcore-sandbox-python:1.0.0")
        assert metadata.name == "llmcore-sandbox-python"

    def test_get_without_version(self, builtin_registry: ImageRegistry):
        """Test getting image without specifying version."""
        metadata = builtin_registry.get("llmcore-sandbox-python")
        assert metadata.name == "llmcore-sandbox-python"

    def test_get_nonexistent_image(self, builtin_registry: ImageRegistry):
        """Test getting nonexistent image raises error."""
        with pytest.raises(ImageNotFoundError) as exc_info:
            builtin_registry.get("nonexistent-image:1.0.0")
        assert "nonexistent" in str(exc_info.value).lower()
        assert exc_info.value.image == "nonexistent-image:1.0.0"

    def test_get_manifest(self, builtin_registry: ImageRegistry):
        """Test get_manifest convenience method."""
        manifest = builtin_registry.get_manifest("llmcore-sandbox-python")
        assert isinstance(manifest, ImageManifest)
        assert manifest.name == "llmcore-sandbox-python"

    def test_has_image_true(self, builtin_registry: ImageRegistry):
        """Test has_image returns True for existing image."""
        assert builtin_registry.has_image("llmcore-sandbox-base:1.0.0") is True

    def test_has_image_false(self, builtin_registry: ImageRegistry):
        """Test has_image returns False for nonexistent image."""
        assert builtin_registry.has_image("nonexistent:1.0.0") is False

    def test_is_available_locally(self, builtin_registry: ImageRegistry):
        """Test is_available_locally method."""
        # Builtins are marked as not available locally by default
        assert builtin_registry.is_available_locally("llmcore-sandbox-python:1.0.0") is False


# ==============================================================================
# Image Listing Tests
# ==============================================================================


class TestImageListing:
    """Tests for image listing operations."""

    def test_list_all_images(self, builtin_registry: ImageRegistry):
        """Test listing all images."""
        images = builtin_registry.list_images()
        assert len(images) >= 6

    def test_list_by_tier(self, builtin_registry: ImageRegistry):
        """Test listing images by tier."""
        base_images = builtin_registry.list_images(tier=ImageTier.BASE)
        assert len(base_images) >= 1
        assert all(img.tier == ImageTier.BASE for img in base_images)

        task_images = builtin_registry.list_images(tier=ImageTier.TASK)
        assert len(task_images) >= 2
        assert all(img.tier == ImageTier.TASK for img in task_images)

    def test_list_by_capability(self, builtin_registry: ImageRegistry):
        """Test listing images by required capability."""
        python_images = builtin_registry.list_images(
            capabilities={ImageCapability.PYTHON}
        )
        assert len(python_images) >= 2  # python, research, websearch
        for img in python_images:
            assert ImageCapability.PYTHON in img.capabilities

    def test_list_by_multiple_capabilities(self, builtin_registry: ImageRegistry):
        """Test listing by multiple capabilities."""
        images = builtin_registry.list_images(
            capabilities={ImageCapability.PYTHON, ImageCapability.NETWORK}
        )
        for img in images:
            assert ImageCapability.PYTHON in img.capabilities
            assert ImageCapability.NETWORK in img.capabilities

    def test_list_available_only(self, builtin_registry: ImageRegistry):
        """Test listing only locally available images."""
        # Builtins are not locally available by default
        images = builtin_registry.list_images(available_only=True)
        assert len(images) == 0

    def test_list_by_tier_grouped(self, builtin_registry: ImageRegistry):
        """Test listing images grouped by tier."""
        grouped = builtin_registry.list_by_tier()

        assert ImageTier.BASE in grouped
        assert ImageTier.SPECIALIZED in grouped
        assert ImageTier.TASK in grouped

        assert len(grouped[ImageTier.BASE]) >= 1
        assert len(grouped[ImageTier.SPECIALIZED]) >= 3
        assert len(grouped[ImageTier.TASK]) >= 2

    def test_list_sorted_by_tier_and_name(self, builtin_registry: ImageRegistry):
        """Test that list_images returns sorted results."""
        images = builtin_registry.list_images()

        # Should be sorted: BASE first, then SPECIALIZED, then TASK
        tier_order = [img.tier for img in images]
        base_indices = [i for i, t in enumerate(tier_order) if t == ImageTier.BASE]
        specialized_indices = [i for i, t in enumerate(tier_order) if t == ImageTier.SPECIALIZED]
        task_indices = [i for i, t in enumerate(tier_order) if t == ImageTier.TASK]

        if base_indices and specialized_indices:
            assert max(base_indices) < min(specialized_indices)
        if specialized_indices and task_indices:
            assert max(specialized_indices) < min(task_indices)


# ==============================================================================
# Find by Capability Tests
# ==============================================================================


class TestFindByCapability:
    """Tests for find_by_capability method."""

    def test_find_python_capability(self, builtin_registry: ImageRegistry):
        """Test finding image with Python capability."""
        metadata = builtin_registry.find_by_capability(ImageCapability.PYTHON)
        assert metadata is not None
        assert ImageCapability.PYTHON in metadata.capabilities

    def test_find_research_capability(self, builtin_registry: ImageRegistry):
        """Test finding image with research capability."""
        metadata = builtin_registry.find_by_capability(ImageCapability.RESEARCH)
        assert metadata is not None
        assert ImageCapability.RESEARCH in metadata.capabilities

    def test_find_nonexistent_capability(self, builtin_registry: ImageRegistry):
        """Test finding image with nonexistent capability returns None."""
        # Create a capability check that shouldn't match anything
        metadata = builtin_registry.find_by_capability(ImageCapability.RUST)
        # Rust is not in our builtin images
        assert metadata is None


# ==============================================================================
# Docker Integration Tests
# ==============================================================================


class TestDockerIntegration:
    """Tests for Docker integration."""

    @pytest.mark.asyncio
    async def test_refresh_with_mock_docker(
        self,
        empty_registry: ImageRegistry,
        mock_docker_client: MagicMock,
    ):
        """Test refreshing registry with mock Docker client."""
        # Inject mock client
        empty_registry._docker_client = mock_docker_client

        count = await empty_registry.refresh()
        assert count > 0
        assert empty_registry.last_refresh is not None

    @pytest.mark.asyncio
    async def test_refresh_without_docker(self, empty_registry: ImageRegistry):
        """Test refresh when Docker is unavailable."""
        with patch("docker.from_env") as mock:
            mock.side_effect = Exception("Docker not available")
            count = await empty_registry.refresh()
            # Should return 0 (no images discovered)
            assert count == 0

    def test_pattern_matching(self, empty_registry: ImageRegistry):
        """Test image pattern matching."""
        # Test llmcore pattern
        assert empty_registry._matches_pattern("llmcore-sandbox-python:1.0.0") is True
        assert empty_registry._matches_pattern("llmcore-sandbox-base") is True

        # Test non-matching
        assert empty_registry._matches_pattern("ubuntu:24.04") is False
        assert empty_registry._matches_pattern("python:3.11-slim") is False


# ==============================================================================
# Serialization Tests
# ==============================================================================


class TestRegistrySerialization:
    """Tests for registry serialization."""

    def test_to_dict(self, builtin_registry: ImageRegistry):
        """Test registry to_dict method."""
        data = builtin_registry.to_dict()

        assert "images" in data
        assert "image_patterns" in data
        assert "last_refresh" in data

        assert len(data["images"]) == builtin_registry.count
