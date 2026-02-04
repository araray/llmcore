# src/llmcore/agents/sandbox/images/models.py
"""
Data models for the Container Image System.

This module defines the core data structures for managing container images
including manifests, capabilities, and metadata.

The image system is organized into three tiers:
    - TIER 1 (BASE): Foundation image with minimal tools
    - TIER 2 (SPECIALIZED): Language-specific images (python, nodejs, etc.)
    - TIER 3 (TASK): Task-oriented images (research, websearch, etc.)

Example:
    >>> manifest = ImageManifest(
    ...     name="llmcore-sandbox-python",
    ...     version="1.0.0",
    ...     tier=ImageTier.SPECIALIZED,
    ...     capabilities=[ImageCapability.PYTHON, ImageCapability.SHELL],
    ... )
    >>> print(manifest.full_name)
    llmcore-sandbox-python:1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ImageTier(str, Enum):
    """
    Container image tier classification.

    Tiers determine the inheritance hierarchy and feature set:
        BASE: Minimal foundation with security hardening
        SPECIALIZED: Language-specific runtime environments
        TASK: Purpose-built images for specific workflows
    """

    BASE = "base"
    SPECIALIZED = "specialized"
    TASK = "task"


class ImageCapability(str, Enum):
    """
    Capabilities available in container images.

    These define what tools and runtimes are available
    for automatic image selection.
    """

    # Core capabilities
    SHELL = "shell"
    NETWORK = "network"
    FILESYSTEM = "filesystem"

    # Language runtimes
    PYTHON = "python"
    NODEJS = "nodejs"
    RUST = "rust"

    # Tools
    GIT = "git"
    CURL = "curl"
    JQ = "jq"
    PANDOC = "pandoc"
    PDF_TOOLS = "pdf_tools"
    BROWSER = "browser"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"

    # Task-specific
    RESEARCH = "research"
    WEBSEARCH = "websearch"
    DOCGEN = "docgen"
    TESTING = "testing"
    CODE_ANALYSIS = "code_analysis"


class AccessMode(str, Enum):
    """
    Security access mode for container images.

    RESTRICTED: Network disabled, limited capabilities
    FULL: Full access including network
    """

    RESTRICTED = "restricted"
    FULL = "full"


@dataclass
class ResourceLimits:
    """
    Resource limits for container execution.

    Attributes:
        memory_limit: Memory limit (e.g., "512m", "1g")
        cpu_limit: CPU limit as float (e.g., 1.0, 2.0)
        timeout_seconds: Maximum execution time
        pids_limit: Maximum number of processes
    """

    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout_seconds: int = 600
    pids_limit: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "timeout_seconds": self.timeout_seconds,
            "pids_limit": self.pids_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceLimits":
        """Create from dictionary."""
        return cls(
            memory_limit=data.get("memory_limit", "512m"),
            cpu_limit=data.get("cpu_limit", 1.0),
            timeout_seconds=data.get("timeout_seconds", 600),
            pids_limit=data.get("pids_limit", 100),
        )


@dataclass
class ImageManifest:
    """
    Container image manifest defining its properties and capabilities.

    The manifest is the source of truth for what an image provides.
    It's stored in each image at /etc/llmcore/capabilities.json.

    Attributes:
        name: Image name without version (e.g., "llmcore-sandbox-python")
        version: Semantic version (e.g., "1.0.0")
        tier: Image tier classification
        base_image: Parent image this was built from
        capabilities: Set of capabilities provided
        tools: List of available command-line tools
        default_access_mode: Default security access mode
        resource_limits: Default resource limits
        environment: Default environment variables
        working_directory: Default working directory
        entrypoint: Container entrypoint
        description: Human-readable description
        build_date: When the image was built
        vcs_ref: Git commit reference
    """

    name: str
    version: str
    tier: ImageTier
    base_image: str | None = None
    capabilities: set[ImageCapability] = field(default_factory=set)
    tools: list[str] = field(default_factory=list)
    default_access_mode: AccessMode = AccessMode.RESTRICTED
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    environment: dict[str, str] = field(default_factory=dict)
    working_directory: str = "/workspace"
    entrypoint: str | None = None
    description: str = ""
    build_date: datetime | None = None
    vcs_ref: str | None = None

    @property
    def full_name(self) -> str:
        """Get the full image name with version tag."""
        return f"{self.name}:{self.version}"

    @property
    def is_base(self) -> bool:
        """Check if this is a base tier image."""
        return self.tier == ImageTier.BASE

    @property
    def is_specialized(self) -> bool:
        """Check if this is a specialized tier image."""
        return self.tier == ImageTier.SPECIALIZED

    @property
    def is_task(self) -> bool:
        """Check if this is a task tier image."""
        return self.tier == ImageTier.TASK

    def has_capability(self, capability: ImageCapability) -> bool:
        """Check if image has a specific capability."""
        return capability in self.capabilities

    def has_all_capabilities(self, required: set[ImageCapability]) -> bool:
        """Check if image has all required capabilities."""
        return required.issubset(self.capabilities)

    def has_tool(self, tool: str) -> bool:
        """Check if image has a specific tool."""
        return tool in self.tools

    def to_dict(self) -> dict[str, Any]:
        """
        Convert manifest to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the manifest
        """
        return {
            "name": self.name,
            "version": self.version,
            "tier": self.tier.value,
            "base_image": self.base_image,
            "capabilities": [c.value for c in self.capabilities],
            "tools": self.tools,
            "default_access_mode": self.default_access_mode.value,
            "resource_limits": self.resource_limits.to_dict(),
            "environment": self.environment,
            "working_directory": self.working_directory,
            "entrypoint": self.entrypoint,
            "description": self.description,
            "build_date": self.build_date.isoformat() if self.build_date else None,
            "vcs_ref": self.vcs_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageManifest":
        """
        Create manifest from dictionary.

        Args:
            data: Dictionary with manifest data

        Returns:
            ImageManifest instance
        """
        # Parse capabilities
        capabilities = set()
        for cap in data.get("capabilities", []):
            try:
                capabilities.add(ImageCapability(cap))
            except ValueError:
                pass  # Ignore unknown capabilities

        # Parse build date
        build_date = None
        if data.get("build_date"):
            try:
                build_date = datetime.fromisoformat(data["build_date"])
            except (ValueError, TypeError):
                pass

        # Parse access mode
        try:
            access_mode = AccessMode(data.get("default_access_mode", "restricted"))
        except ValueError:
            access_mode = AccessMode.RESTRICTED

        # Parse tier
        try:
            tier = ImageTier(data.get("tier", "base"))
        except ValueError:
            tier = ImageTier.BASE

        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            tier=tier,
            base_image=data.get("base_image"),
            capabilities=capabilities,
            tools=data.get("tools", []),
            default_access_mode=access_mode,
            resource_limits=ResourceLimits.from_dict(data.get("resource_limits", {})),
            environment=data.get("environment", {}),
            working_directory=data.get("working_directory", "/workspace"),
            entrypoint=data.get("entrypoint"),
            description=data.get("description", ""),
            build_date=build_date,
            vcs_ref=data.get("vcs_ref"),
        )


@dataclass
class ImageMetadata:
    """
    Runtime metadata about a container image.

    This is computed at runtime when discovering images,
    combining the manifest with Docker inspection data.

    Attributes:
        manifest: The image manifest
        docker_id: Docker image ID
        size_bytes: Image size in bytes
        created: When the image was created
        available_locally: Whether the image is pulled locally
        labels: Docker labels from the image
    """

    manifest: ImageManifest
    docker_id: str | None = None
    size_bytes: int = 0
    created: datetime | None = None
    available_locally: bool = False
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get the image name."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Get the image version."""
        return self.manifest.version

    @property
    def full_name(self) -> str:
        """Get the full image name with version."""
        return self.manifest.full_name

    @property
    def tier(self) -> ImageTier:
        """Get the image tier."""
        return self.manifest.tier

    @property
    def capabilities(self) -> set[ImageCapability]:
        """Get the image capabilities."""
        return self.manifest.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "manifest": self.manifest.to_dict(),
            "docker_id": self.docker_id,
            "size_bytes": self.size_bytes,
            "created": self.created.isoformat() if self.created else None,
            "available_locally": self.available_locally,
            "labels": self.labels,
        }


# ============================================================================
# Predefined Image Manifests
# ============================================================================

# Base image manifest
BASE_IMAGE_MANIFEST = ImageManifest(
    name="llmcore-sandbox-base",
    version="1.0.0",
    tier=ImageTier.BASE,
    base_image="ubuntu:24.04",
    capabilities={
        ImageCapability.SHELL,
        ImageCapability.FILESYSTEM,
    },
    tools=["bash", "sh", "cat", "ls", "cp", "mv", "rm", "mkdir", "chmod", "chown"],
    default_access_mode=AccessMode.RESTRICTED,
    resource_limits=ResourceLimits(memory_limit="512m", cpu_limit=1.0),
    working_directory="/workspace",
    description="Minimal base image with security hardening",
)

# Python specialized image
PYTHON_IMAGE_MANIFEST = ImageManifest(
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
    tools=[
        "python3",
        "python",
        "pip",
        "pip3",
        "git",
        "bash",
        "curl",
    ],
    default_access_mode=AccessMode.RESTRICTED,
    resource_limits=ResourceLimits(memory_limit="1g", cpu_limit=2.0),
    working_directory="/workspace",
    description="Python 3.12 development environment",
)

# Node.js specialized image
NODEJS_IMAGE_MANIFEST = ImageManifest(
    name="llmcore-sandbox-nodejs",
    version="1.0.0",
    tier=ImageTier.SPECIALIZED,
    base_image="llmcore-sandbox-base:1.0.0",
    capabilities={
        ImageCapability.SHELL,
        ImageCapability.FILESYSTEM,
        ImageCapability.NODEJS,
        ImageCapability.GIT,
    },
    tools=[
        "node",
        "npm",
        "npx",
        "git",
        "bash",
        "curl",
    ],
    default_access_mode=AccessMode.RESTRICTED,
    resource_limits=ResourceLimits(memory_limit="1g", cpu_limit=2.0),
    working_directory="/workspace",
    description="Node.js 22 LTS development environment",
)

# Shell utilities image
SHELL_IMAGE_MANIFEST = ImageManifest(
    name="llmcore-sandbox-shell",
    version="1.0.0",
    tier=ImageTier.SPECIALIZED,
    base_image="llmcore-sandbox-base:1.0.0",
    capabilities={
        ImageCapability.SHELL,
        ImageCapability.FILESYSTEM,
        ImageCapability.GIT,
        ImageCapability.CURL,
        ImageCapability.JQ,
    },
    tools=[
        "bash",
        "zsh",
        "sh",
        "git",
        "curl",
        "wget",
        "jq",
        "yq",
        "sed",
        "awk",
        "grep",
        "find",
        "xargs",
        "tar",
        "gzip",
        "unzip",
    ],
    default_access_mode=AccessMode.RESTRICTED,
    resource_limits=ResourceLimits(memory_limit="512m", cpu_limit=1.0),
    working_directory="/workspace",
    description="Shell utilities and text processing tools",
)

# Research task image
RESEARCH_IMAGE_MANIFEST = ImageManifest(
    name="llmcore-sandbox-research",
    version="1.0.0",
    tier=ImageTier.TASK,
    base_image="llmcore-sandbox-python:1.0.0",
    capabilities={
        ImageCapability.SHELL,
        ImageCapability.FILESYSTEM,
        ImageCapability.PYTHON,
        ImageCapability.NETWORK,
        ImageCapability.PANDOC,
        ImageCapability.PDF_TOOLS,
        ImageCapability.RESEARCH,
    },
    tools=[
        "python3",
        "pip",
        "git",
        "pandoc",
        "pdftotext",
        "curl",
        "wget",
    ],
    default_access_mode=AccessMode.FULL,  # Research needs network
    resource_limits=ResourceLimits(memory_limit="2g", cpu_limit=2.0, timeout_seconds=1200),
    working_directory="/workspace",
    description="Research and document processing environment",
)

# Web search task image
WEBSEARCH_IMAGE_MANIFEST = ImageManifest(
    name="llmcore-sandbox-websearch",
    version="1.0.0",
    tier=ImageTier.TASK,
    base_image="llmcore-sandbox-python:1.0.0",
    capabilities={
        ImageCapability.SHELL,
        ImageCapability.FILESYSTEM,
        ImageCapability.PYTHON,
        ImageCapability.NETWORK,
        ImageCapability.BROWSER,
        ImageCapability.WEBSEARCH,
    },
    tools=[
        "python3",
        "pip",
        "chromium-browser",
        "chromedriver",
        "curl",
    ],
    default_access_mode=AccessMode.FULL,  # Web search needs network
    resource_limits=ResourceLimits(memory_limit="2g", cpu_limit=2.0, timeout_seconds=600),
    working_directory="/workspace",
    description="Web search and browser automation environment",
)


# Mapping of image names to their manifests
BUILTIN_MANIFESTS: dict[str, ImageManifest] = {
    "llmcore-sandbox-base": BASE_IMAGE_MANIFEST,
    "llmcore-sandbox-python": PYTHON_IMAGE_MANIFEST,
    "llmcore-sandbox-nodejs": NODEJS_IMAGE_MANIFEST,
    "llmcore-sandbox-shell": SHELL_IMAGE_MANIFEST,
    "llmcore-sandbox-research": RESEARCH_IMAGE_MANIFEST,
    "llmcore-sandbox-websearch": WEBSEARCH_IMAGE_MANIFEST,
}
