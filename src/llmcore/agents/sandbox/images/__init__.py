# src/llmcore/agents/sandbox/images/__init__.py
"""
Container Image System for LLMCore Sandbox.

This module provides a comprehensive system for managing Docker container
images used in sandbox execution. It includes:

    - **Models**: Data structures for image manifests and metadata
    - **Registry**: Discovery and management of available images
    - **Selector**: Automatic task-based image selection
    - **Manifest**: Parsing and validation utilities

Architecture:
    The image system is organized into three tiers:

    1. **BASE** (Tier 1): Minimal foundation with security hardening
    2. **SPECIALIZED** (Tier 2): Language-specific runtime environments
    3. **TASK** (Tier 3): Purpose-built images for specific workflows

Image Naming Convention:
    llmcore-sandbox-{tier}-{name}:{version}

    Examples:
        - llmcore-sandbox-base:1.0.0
        - llmcore-sandbox-python:1.0.0
        - llmcore-sandbox-research:1.0.0

Basic Usage:
    >>> from llmcore.agents.sandbox.images import ImageRegistry, ImageSelector
    >>>
    >>> # Create registry and discover images
    >>> registry = ImageRegistry()
    >>> await registry.refresh()
    >>>
    >>> # Select image for task
    >>> selector = ImageSelector(registry)
    >>> result = selector.select_for_task("research")
    >>> print(result.image)  # llmcore-sandbox-research:1.0.0
    >>>
    >>> # Or select by runtime
    >>> result = selector.select_for_runtime("python")
    >>> print(result.image)  # llmcore-sandbox-python:1.0.0

Integration with DockerSandboxProvider:
    >>> from llmcore.agents.sandbox import DockerSandboxProvider
    >>> from llmcore.agents.sandbox.images import ImageSelector, ImageRegistry
    >>>
    >>> registry = ImageRegistry()
    >>> selector = ImageSelector(registry)
    >>>
    >>> # Select image for task
    >>> result = selector.select_for_task("python_dev")
    >>>
    >>> # Create provider with selected image
    >>> provider = DockerSandboxProvider(
    ...     image=result.image,
    ...     image_whitelist=["llmcore-sandbox-*"],
    ... )

See Also:
    - LLMCORE_CONTAINER_IMAGES_MASTER_PLAN.md: Full specification
    - LLMCORE_AGENTIC_SYSTEM_MASTER_PLAN_G3.md: Section 23-25
"""

# Models - Core data structures
# Manifest utilities
from .manifest import (
    # Constants
    MANIFEST_PATH_IN_CONTAINER,
    # Exceptions
    ManifestError,
    ManifestNotFoundError,
    ManifestParseError,
    ManifestValidationError,
    # Functions
    get_builtin_manifest,
    list_builtin_manifests,
    load_manifest_from_docker,
    load_manifest_from_file,
    load_manifest_from_string,
    parse_manifest,
    save_manifest_to_file,
    validate_manifest,
)
from .models import (
    # Builtin manifests
    BASE_IMAGE_MANIFEST,
    BUILTIN_MANIFESTS,
    NODEJS_IMAGE_MANIFEST,
    PYTHON_IMAGE_MANIFEST,
    RESEARCH_IMAGE_MANIFEST,
    SHELL_IMAGE_MANIFEST,
    WEBSEARCH_IMAGE_MANIFEST,
    # Enums
    AccessMode,
    ImageCapability,
    # Data classes
    ImageManifest,
    ImageMetadata,
    ImageTier,
    ResourceLimits,
)

# Registry
from .registry import (
    # Exceptions
    ImageNotFoundError,
    # Classes
    ImageRegistry,
    ImageRegistryError,
)

# Selector
from .selector import (
    # Mappings
    RUNTIME_IMAGE_MAP,
    TASK_CAPABILITY_MAP,
    TASK_IMAGE_MAP,
    # Classes
    ImageSelector,
    SelectionConfig,
    SelectionResult,
)

__all__ = [
    # Enums
    "AccessMode",
    "ImageCapability",
    "ImageTier",
    # Data classes
    "ImageManifest",
    "ImageMetadata",
    "ResourceLimits",
    # Builtin manifests
    "BASE_IMAGE_MANIFEST",
    "BUILTIN_MANIFESTS",
    "NODEJS_IMAGE_MANIFEST",
    "PYTHON_IMAGE_MANIFEST",
    "RESEARCH_IMAGE_MANIFEST",
    "SHELL_IMAGE_MANIFEST",
    "WEBSEARCH_IMAGE_MANIFEST",
    # Manifest exceptions
    "ManifestError",
    "ManifestNotFoundError",
    "ManifestParseError",
    "ManifestValidationError",
    # Manifest functions
    "get_builtin_manifest",
    "list_builtin_manifests",
    "load_manifest_from_docker",
    "load_manifest_from_file",
    "load_manifest_from_string",
    "parse_manifest",
    "save_manifest_to_file",
    "validate_manifest",
    "MANIFEST_PATH_IN_CONTAINER",
    # Registry
    "ImageNotFoundError",
    "ImageRegistryError",
    "ImageRegistry",
    # Selector
    "ImageSelector",
    "SelectionConfig",
    "SelectionResult",
    "RUNTIME_IMAGE_MAP",
    "TASK_CAPABILITY_MAP",
    "TASK_IMAGE_MAP",
]
