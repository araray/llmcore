# src/llmcore/agents/sandbox/images/manifest.py
"""
Manifest parsing utilities for container images.

This module provides functions for loading, parsing, and validating
image manifests from JSON files or Docker container inspection.

Manifest files are stored at `/etc/llmcore/capabilities.json` inside
each container image.

Example:
    >>> manifest = load_manifest_from_file("manifest.json")
    >>> print(manifest.capabilities)
    {<ImageCapability.PYTHON: 'python'>, <ImageCapability.SHELL: 'shell'>}

    >>> manifest = load_manifest_from_docker("llmcore-sandbox-python:1.0.0")
    >>> print(manifest.name)
    llmcore-sandbox-python
"""

import json
import logging
from pathlib import Path
from typing import Any

from .models import (
    BUILTIN_MANIFESTS,
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageTier,
)

logger = logging.getLogger(__name__)

# Path where manifests are stored inside containers
MANIFEST_PATH_IN_CONTAINER = "/etc/llmcore/capabilities.json"


class ManifestError(Exception):
    """Error loading or parsing a manifest."""

    pass


class ManifestNotFoundError(ManifestError):
    """Manifest file not found."""

    pass


class ManifestParseError(ManifestError):
    """Error parsing manifest content."""

    pass


class ManifestValidationError(ManifestError):
    """Manifest validation failed."""

    pass


def load_manifest_from_file(path: str | Path) -> ImageManifest:
    """
    Load an image manifest from a JSON file.

    Args:
        path: Path to the manifest JSON file

    Returns:
        Parsed ImageManifest

    Raises:
        ManifestNotFoundError: If file doesn't exist
        ManifestParseError: If JSON is invalid
        ManifestValidationError: If required fields are missing
    """
    path = Path(path)

    if not path.exists():
        raise ManifestNotFoundError(f"Manifest file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ManifestParseError(f"Invalid JSON in manifest: {e}")

    return parse_manifest(data)


def load_manifest_from_string(content: str) -> ImageManifest:
    """
    Load an image manifest from a JSON string.

    Args:
        content: JSON string containing manifest data

    Returns:
        Parsed ImageManifest

    Raises:
        ManifestParseError: If JSON is invalid
        ManifestValidationError: If required fields are missing
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ManifestParseError(f"Invalid JSON: {e}")

    return parse_manifest(data)


def load_manifest_from_docker(
    image: str,
    docker_client: Any | None = None,
) -> ImageManifest:
    """
    Load an image manifest from a Docker image.

    This extracts the manifest file from the container image.

    Args:
        image: Docker image name (e.g., "llmcore-sandbox-python:1.0.0")
        docker_client: Optional Docker client (uses docker.from_env() if not provided)

    Returns:
        Parsed ImageManifest, or builtin manifest if extraction fails

    Raises:
        ManifestError: If image cannot be inspected
    """
    # Check if we have a builtin manifest first
    image_name = image.split(":")[0]
    if image_name in BUILTIN_MANIFESTS:
        logger.debug(f"Using builtin manifest for {image_name}")
        return BUILTIN_MANIFESTS[image_name]

    if docker_client is None:
        try:
            import docker

            docker_client = docker.from_env()
        except ImportError:
            raise ManifestError("docker-py not installed")
        except Exception as e:
            raise ManifestError(f"Cannot connect to Docker: {e}")

    try:
        # Run a temporary container to extract the manifest
        result = docker_client.containers.run(
            image,
            f"cat {MANIFEST_PATH_IN_CONTAINER}",
            remove=True,
            detach=False,
        )

        if isinstance(result, bytes):
            result = result.decode("utf-8")

        return load_manifest_from_string(result)

    except Exception as e:
        logger.warning(f"Cannot extract manifest from {image}: {e}")
        # Try to create a minimal manifest from Docker inspection
        return _create_manifest_from_inspection(image, docker_client)


def _create_manifest_from_inspection(
    image: str,
    docker_client: Any,
) -> ImageManifest:
    """
    Create a minimal manifest from Docker image inspection.

    This is used as a fallback when no manifest file is present.

    Args:
        image: Docker image name
        docker_client: Docker client

    Returns:
        Minimal ImageManifest based on inspection
    """
    try:
        image_obj = docker_client.images.get(image)
        labels = image_obj.labels or {}
        config = image_obj.attrs.get("Config", {})

        # Extract name and version from image string
        parts = image.split(":")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"

        # Determine tier from labels or name
        tier_str = labels.get("llmcore.sandbox.tier", "base")
        try:
            tier = ImageTier(tier_str)
        except ValueError:
            tier = ImageTier.BASE

        # Parse capabilities from labels
        capabilities = set()
        cap_str = labels.get("llmcore.sandbox.capabilities", "")
        for cap in cap_str.split(","):
            cap = cap.strip()
            if cap:
                try:
                    capabilities.add(ImageCapability(cap))
                except ValueError:
                    pass

        # Always add shell capability
        capabilities.add(ImageCapability.SHELL)
        capabilities.add(ImageCapability.FILESYSTEM)

        # Parse access mode from labels
        access_str = labels.get("llmcore.sandbox.access_mode", "restricted")
        try:
            access_mode = AccessMode(access_str)
        except ValueError:
            access_mode = AccessMode.RESTRICTED

        return ImageManifest(
            name=name,
            version=version,
            tier=tier,
            capabilities=capabilities,
            tools=[],
            default_access_mode=access_mode,
            working_directory=config.get("WorkingDir", "/workspace"),
            entrypoint=config.get("Entrypoint"),
            description=labels.get("org.opencontainers.image.description", ""),
        )

    except Exception as e:
        logger.warning(f"Cannot inspect image {image}: {e}")
        # Return absolute minimal manifest
        parts = image.split(":")
        return ImageManifest(
            name=parts[0],
            version=parts[1] if len(parts) > 1 else "latest",
            tier=ImageTier.BASE,
            capabilities={ImageCapability.SHELL, ImageCapability.FILESYSTEM},
        )


def parse_manifest(data: dict[str, Any]) -> ImageManifest:
    """
    Parse manifest data dictionary into ImageManifest.

    Args:
        data: Dictionary containing manifest data

    Returns:
        Validated ImageManifest

    Raises:
        ManifestValidationError: If required fields are missing or invalid
    """
    # Validate required fields
    if "name" not in data:
        raise ManifestValidationError("Manifest missing required field: name")
    if "version" not in data:
        raise ManifestValidationError("Manifest missing required field: version")

    return ImageManifest.from_dict(data)


def validate_manifest(manifest: ImageManifest) -> None:
    """
    Validate an ImageManifest for completeness and consistency.

    Args:
        manifest: Manifest to validate

    Raises:
        ManifestValidationError: If validation fails
    """
    errors = []

    # Check name format
    if not manifest.name:
        errors.append("Image name cannot be empty")
    elif not manifest.name.startswith("llmcore-sandbox-"):
        logger.warning(f"Image name '{manifest.name}' doesn't follow naming convention")

    # Check version format (semantic versioning)
    if not manifest.version:
        errors.append("Image version cannot be empty")
    else:
        parts = manifest.version.split(".")
        if len(parts) < 2:
            errors.append(f"Invalid version format: {manifest.version}")

    # Check tier-specific requirements
    if manifest.tier == ImageTier.BASE:
        # Base images shouldn't have a parent
        if manifest.base_image and "llmcore-sandbox" in manifest.base_image:
            errors.append("Base image shouldn't extend another llmcore image")

    elif manifest.tier in (ImageTier.SPECIALIZED, ImageTier.TASK):
        # Specialized and task images should have a parent
        if not manifest.base_image:
            logger.warning(f"Image {manifest.name} has no base_image specified")

    # Check capabilities
    if ImageCapability.SHELL not in manifest.capabilities:
        logger.warning(f"Image {manifest.name} missing shell capability")

    if errors:
        raise ManifestValidationError(f"Manifest validation failed: {'; '.join(errors)}")


def save_manifest_to_file(manifest: ImageManifest, path: str | Path) -> None:
    """
    Save an ImageManifest to a JSON file.

    Args:
        manifest: Manifest to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.debug(f"Saved manifest to {path}")


def get_builtin_manifest(name: str) -> ImageManifest | None:
    """
    Get a builtin manifest by image name.

    Args:
        name: Image name (with or without version tag)

    Returns:
        ImageManifest if found, None otherwise
    """
    # Strip version tag if present
    base_name = name.split(":")[0]
    return BUILTIN_MANIFESTS.get(base_name)


def list_builtin_manifests() -> dict[str, ImageManifest]:
    """
    Get all builtin image manifests.

    Returns:
        Dictionary mapping image names to manifests
    """
    return BUILTIN_MANIFESTS.copy()
