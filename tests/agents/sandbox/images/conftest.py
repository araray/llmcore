# tests/agents/sandbox/images/conftest.py
"""
Pytest fixtures and configuration for images module tests.

This module provides fixtures for:
    - Sample manifests and metadata
    - Mock Docker client
    - Temporary file handling
    - Registry setup
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest

from llmcore.agents.sandbox.images import (
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageMetadata,
    ImageRegistry,
    ImageSelector,
    ImageTier,
    ResourceLimits,
    SelectionConfig,
)

# ==============================================================================
# Manifest Fixtures
# ==============================================================================


@pytest.fixture
def sample_manifest_dict() -> Dict[str, Any]:
    """Create a sample manifest dictionary."""
    return {
        "name": "llmcore-sandbox-test",
        "version": "1.0.0",
        "tier": "specialized",
        "base_image": "llmcore-sandbox-base:1.0.0",
        "capabilities": ["python", "shell", "git"],
        "tools": ["python3", "pip", "git", "bash"],
        "default_access_mode": "restricted",
        "resource_limits": {
            "memory_limit": "1g",
            "cpu_limit": 2.0,
            "timeout_seconds": 600,
            "pids_limit": 100,
        },
        "environment": {"PYTHONPATH": "/workspace"},
        "working_directory": "/workspace",
        "description": "Test image for unit tests",
        "build_date": "2026-01-23T00:00:00",
        "vcs_ref": "abc1234",
    }


@pytest.fixture
def sample_manifest(sample_manifest_dict: Dict[str, Any]) -> ImageManifest:
    """Create a sample ImageManifest."""
    return ImageManifest.from_dict(sample_manifest_dict)


@pytest.fixture
def base_manifest() -> ImageManifest:
    """Create a base tier ImageManifest."""
    return ImageManifest(
        name="llmcore-sandbox-base",
        version="1.0.0",
        tier=ImageTier.BASE,
        base_image="ubuntu:24.04",
        capabilities={ImageCapability.SHELL, ImageCapability.FILESYSTEM},
        tools=["bash", "sh", "cat", "ls"],
        default_access_mode=AccessMode.RESTRICTED,
    )


@pytest.fixture
def python_manifest() -> ImageManifest:
    """Create a Python specialized ImageManifest."""
    return ImageManifest(
        name="llmcore-sandbox-python",
        version="1.0.0",
        tier=ImageTier.SPECIALIZED,
        base_image="llmcore-sandbox-base:1.0.0",
        capabilities={
            ImageCapability.SHELL,
            ImageCapability.FILESYSTEM,
            ImageCapability.PYTHON,
            ImageCapability.GIT,
        },
        tools=["python3", "pip", "git", "bash"],
        default_access_mode=AccessMode.RESTRICTED,
    )


@pytest.fixture
def research_manifest() -> ImageManifest:
    """Create a research task ImageManifest."""
    return ImageManifest(
        name="llmcore-sandbox-research",
        version="1.0.0",
        tier=ImageTier.TASK,
        base_image="llmcore-sandbox-python:1.0.0",
        capabilities={
            ImageCapability.SHELL,
            ImageCapability.FILESYSTEM,
            ImageCapability.PYTHON,
            ImageCapability.NETWORK,
            ImageCapability.RESEARCH,
        },
        tools=["python3", "pip", "pandoc", "curl"],
        default_access_mode=AccessMode.FULL,
    )


# ==============================================================================
# Metadata Fixtures
# ==============================================================================


@pytest.fixture
def sample_metadata(sample_manifest: ImageManifest) -> ImageMetadata:
    """Create a sample ImageMetadata."""
    return ImageMetadata(
        manifest=sample_manifest,
        docker_id="sha256:abc123",
        size_bytes=100_000_000,
        created=datetime(2026, 1, 23),
        available_locally=True,
        labels={"llmcore.sandbox.tier": "specialized"},
    )


@pytest.fixture
def base_metadata(base_manifest: ImageManifest) -> ImageMetadata:
    """Create base image metadata."""
    return ImageMetadata(
        manifest=base_manifest,
        docker_id="sha256:base123",
        size_bytes=50_000_000,
        created=datetime(2026, 1, 1),
        available_locally=True,
    )


@pytest.fixture
def python_metadata(python_manifest: ImageManifest) -> ImageMetadata:
    """Create Python image metadata."""
    return ImageMetadata(
        manifest=python_manifest,
        docker_id="sha256:python123",
        size_bytes=200_000_000,
        created=datetime(2026, 1, 10),
        available_locally=True,
    )


@pytest.fixture
def research_metadata(research_manifest: ImageManifest) -> ImageMetadata:
    """Create research image metadata."""
    return ImageMetadata(
        manifest=research_manifest,
        docker_id="sha256:research123",
        size_bytes=500_000_000,
        created=datetime(2026, 1, 20),
        available_locally=True,
    )


# ==============================================================================
# Resource Limits Fixtures
# ==============================================================================


@pytest.fixture
def default_resource_limits() -> ResourceLimits:
    """Create default resource limits."""
    return ResourceLimits()


@pytest.fixture
def high_resource_limits() -> ResourceLimits:
    """Create high resource limits."""
    return ResourceLimits(
        memory_limit="4g",
        cpu_limit=4.0,
        timeout_seconds=1800,
        pids_limit=500,
    )


# ==============================================================================
# Registry Fixtures
# ==============================================================================


@pytest.fixture
def empty_registry() -> ImageRegistry:
    """Create an empty registry without builtins."""
    return ImageRegistry(include_builtins=False)


@pytest.fixture
def builtin_registry() -> ImageRegistry:
    """Create a registry with builtin manifests."""
    return ImageRegistry(include_builtins=True)


@pytest.fixture
def populated_registry(
    builtin_registry: ImageRegistry,
    sample_metadata: ImageMetadata,
) -> ImageRegistry:
    """Create a registry with additional test images."""
    builtin_registry.register(sample_metadata)
    return builtin_registry


# ==============================================================================
# Selector Fixtures
# ==============================================================================


@pytest.fixture
def default_selector(builtin_registry: ImageRegistry) -> ImageSelector:
    """Create a default image selector."""
    return ImageSelector(builtin_registry)


@pytest.fixture
def custom_selector_config() -> SelectionConfig:
    """Create a custom selection config."""
    return SelectionConfig(
        default_image="llmcore-sandbox-python:1.0.0",
        prefer_local=True,
        prefer_task_images=True,
        task_overrides={"custom_task": "custom-image:1.0.0"},
        runtime_overrides={"custom_runtime": "custom-runtime-image:1.0.0"},
    )


@pytest.fixture
def custom_selector(
    builtin_registry: ImageRegistry,
    custom_selector_config: SelectionConfig,
) -> ImageSelector:
    """Create a custom configured selector."""
    return ImageSelector(builtin_registry, custom_selector_config)


# ==============================================================================
# File Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory(prefix="llmcore_images_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def manifest_file(temp_dir: Path, sample_manifest_dict: Dict[str, Any]) -> Path:
    """Create a temporary manifest JSON file."""
    manifest_path = temp_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sample_manifest_dict, f, indent=2)
    return manifest_path


@pytest.fixture
def invalid_manifest_file(temp_dir: Path) -> Path:
    """Create a file with invalid JSON."""
    invalid_path = temp_dir / "invalid.json"
    with open(invalid_path, "w") as f:
        f.write("{not valid json")
    return invalid_path


# ==============================================================================
# Mock Docker Client
# ==============================================================================


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    mock_client = MagicMock()

    # Mock version
    mock_client.version.return_value = {"Version": "24.0.0"}
    mock_client.ping.return_value = True

    # Create mock images
    mock_images = []

    for name, tier in [
        ("llmcore-sandbox-base", "base"),
        ("llmcore-sandbox-python", "specialized"),
        ("llmcore-sandbox-nodejs", "specialized"),
        ("llmcore-sandbox-research", "task"),
    ]:
        mock_image = MagicMock()
        mock_image.tags = [f"{name}:1.0.0"]
        mock_image.short_id = f"sha256:{name[:8]}"
        mock_image.attrs = {
            "Size": 100_000_000,
            "Created": "2026-01-23T00:00:00Z",
        }
        mock_image.labels = {
            "llmcore.sandbox.tier": tier,
            "llmcore.sandbox.capabilities": "shell,filesystem",
        }
        mock_images.append(mock_image)

    mock_client.images.list.return_value = mock_images

    # Mock images.get
    def get_image(name):
        for img in mock_images:
            if name in img.tags or name.split(":")[0] in [t.split(":")[0] for t in img.tags]:
                return img
        raise Exception(f"Image not found: {name}")

    mock_client.images.get.side_effect = get_image

    return mock_client


@pytest.fixture
def mock_docker_unavailable():
    """Create a mock where Docker is unavailable."""
    with patch("docker.from_env") as mock:
        mock.side_effect = Exception("Docker not available")
        yield mock


# ==============================================================================
# Test Data
# ==============================================================================


@pytest.fixture
def all_capabilities() -> set:
    """Get all defined capabilities."""
    return set(ImageCapability)


@pytest.fixture
def all_tiers() -> list:
    """Get all defined tiers."""
    return list(ImageTier)
