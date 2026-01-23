# src/llmcore/agents/sandbox/images/registry.py
"""
Image Registry for managing available container images.

The ImageRegistry provides discovery, listing, and lookup of container
images available for sandbox execution. It maintains a cache of image
metadata and supports both builtin and Docker-discovered images.

Features:
    - Automatic discovery of local Docker images
    - Builtin manifest support for llmcore images
    - Image metadata caching
    - Filtering by tier and capabilities

Example:
    >>> registry = ImageRegistry()
    >>> await registry.refresh()
    >>> images = registry.list_images(tier=ImageTier.SPECIALIZED)
    >>> for img in images:
    ...     print(f"{img.full_name}: {img.capabilities}")
"""

import asyncio
import fnmatch
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .models import (
    BUILTIN_MANIFESTS,
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageMetadata,
    ImageTier,
)
from .manifest import (
    ManifestError,
    get_builtin_manifest,
    load_manifest_from_docker,
)

logger = logging.getLogger(__name__)


class ImageRegistryError(Exception):
    """Error in image registry operations."""

    pass


class ImageNotFoundError(ImageRegistryError):
    """Requested image not found in registry."""

    def __init__(self, image: str, available: Optional[List[str]] = None):
        self.image = image
        self.available = available or []
        msg = f"Image not found: {image}"
        if available:
            msg += f". Available: {', '.join(available[:5])}"
            if len(available) > 5:
                msg += f" (+{len(available) - 5} more)"
        super().__init__(msg)


class ImageRegistry:
    """
    Registry for managing available container images.

    The registry maintains a cache of ImageMetadata for all known images.
    It can discover images from Docker and uses builtin manifests for
    official llmcore images.

    Attributes:
        _images: Cache of image metadata keyed by full name
        _docker_client: Docker client for image discovery
        _image_patterns: Glob patterns to filter discovered images
        _auto_refresh: Whether to auto-refresh on operations
        _last_refresh: Timestamp of last refresh

    Example:
        >>> registry = ImageRegistry(image_patterns=["llmcore-sandbox-*"])
        >>> await registry.refresh()
        >>> image = registry.get("llmcore-sandbox-python:1.0.0")
        >>> print(image.capabilities)
    """

    def __init__(
        self,
        image_patterns: Optional[List[str]] = None,
        auto_refresh: bool = False,
        include_builtins: bool = True,
    ):
        """
        Initialize the image registry.

        Args:
            image_patterns: Glob patterns for image discovery (default: llmcore-sandbox-*)
            auto_refresh: Whether to refresh automatically on operations
            include_builtins: Whether to include builtin manifests
        """
        self._images: Dict[str, ImageMetadata] = {}
        self._docker_client: Optional[Any] = None
        self._image_patterns = image_patterns or ["llmcore-sandbox-*"]
        self._auto_refresh = auto_refresh
        self._include_builtins = include_builtins
        self._last_refresh: Optional[datetime] = None

        # Initialize builtin images
        if include_builtins:
            self._load_builtins()

    def _load_builtins(self) -> None:
        """Load builtin image manifests into the registry."""
        for name, manifest in BUILTIN_MANIFESTS.items():
            metadata = ImageMetadata(
                manifest=manifest,
                available_locally=False,  # Will be updated on refresh
            )
            self._images[manifest.full_name] = metadata
            logger.debug(f"Loaded builtin manifest: {manifest.full_name}")

    def _get_docker_client(self) -> Any:
        """
        Get or create Docker client.

        Returns:
            Docker client instance

        Raises:
            ImageRegistryError: If Docker is not available
        """
        if self._docker_client is None:
            try:
                import docker

                self._docker_client = docker.from_env()
            except ImportError:
                raise ImageRegistryError(
                    "docker-py not installed. Install with: pip install docker"
                )
            except Exception as e:
                raise ImageRegistryError(f"Cannot connect to Docker: {e}")

        return self._docker_client

    async def refresh(self, force: bool = False) -> int:
        """
        Refresh the registry by discovering Docker images.

        Args:
            force: Force refresh even if recently refreshed

        Returns:
            Number of images discovered
        """
        # Run Docker operations in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._refresh_sync, force)

    def _refresh_sync(self, force: bool = False) -> int:
        """Synchronous refresh implementation."""
        try:
            client = self._get_docker_client()
        except ImageRegistryError as e:
            logger.warning(f"Cannot refresh registry: {e}")
            return len(self._images)

        discovered = 0

        try:
            # Get all local images
            docker_images = client.images.list()

            for docker_image in docker_images:
                # Check tags against patterns
                for tag in docker_image.tags or []:
                    if self._matches_pattern(tag):
                        try:
                            metadata = self._process_docker_image(docker_image, tag)
                            if metadata:
                                self._images[metadata.full_name] = metadata
                                discovered += 1
                        except Exception as e:
                            logger.warning(f"Error processing image {tag}: {e}")

        except Exception as e:
            logger.error(f"Error refreshing registry: {e}")

        self._last_refresh = datetime.now()
        logger.info(f"Registry refresh complete: {discovered} images discovered")

        return len(self._images)

    def _matches_pattern(self, image_name: str) -> bool:
        """Check if image name matches any configured pattern."""
        for pattern in self._image_patterns:
            if fnmatch.fnmatch(image_name, pattern):
                return True
            # Also check without version tag
            base_name = image_name.split(":")[0]
            if fnmatch.fnmatch(base_name, pattern):
                return True
        return False

    def _process_docker_image(
        self,
        docker_image: Any,
        tag: str,
    ) -> Optional[ImageMetadata]:
        """
        Process a Docker image into ImageMetadata.

        Args:
            docker_image: Docker image object
            tag: Image tag being processed

        Returns:
            ImageMetadata or None if processing fails
        """
        # Extract name and version
        parts = tag.split(":")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"

        # Check for builtin manifest
        builtin = get_builtin_manifest(name)
        if builtin:
            manifest = builtin
            logger.debug(f"Using builtin manifest for {tag}")
        else:
            # Try to load manifest from image
            try:
                manifest = load_manifest_from_docker(tag, self._docker_client)
            except ManifestError as e:
                logger.debug(f"Cannot load manifest from {tag}: {e}")
                # Create minimal manifest
                manifest = ImageManifest(
                    name=name,
                    version=version,
                    tier=ImageTier.BASE,
                    capabilities={ImageCapability.SHELL, ImageCapability.FILESYSTEM},
                )

        # Get Docker image attributes
        attrs = docker_image.attrs or {}
        created = None
        if attrs.get("Created"):
            try:
                created = datetime.fromisoformat(attrs["Created"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return ImageMetadata(
            manifest=manifest,
            docker_id=docker_image.short_id,
            size_bytes=attrs.get("Size", 0),
            created=created,
            available_locally=True,
            labels=docker_image.labels or {},
        )

    def get(self, image: str) -> ImageMetadata:
        """
        Get image metadata by name.

        Args:
            image: Image name with optional version (e.g., "llmcore-sandbox-python:1.0.0")

        Returns:
            ImageMetadata for the image

        Raises:
            ImageNotFoundError: If image not in registry
        """
        # Normalize name
        if ":" not in image:
            # Try with default version
            image = f"{image}:1.0.0"

        if image in self._images:
            return self._images[image]

        # Try without version specificity
        base_name = image.split(":")[0]
        for full_name, metadata in self._images.items():
            if full_name.startswith(base_name + ":"):
                return metadata

        available = list(self._images.keys())
        raise ImageNotFoundError(image, available)

    def get_manifest(self, image: str) -> ImageManifest:
        """
        Get image manifest by name.

        Args:
            image: Image name

        Returns:
            ImageManifest for the image

        Raises:
            ImageNotFoundError: If image not in registry
        """
        return self.get(image).manifest

    def list_images(
        self,
        tier: Optional[ImageTier] = None,
        capabilities: Optional[Set[ImageCapability]] = None,
        available_only: bool = False,
    ) -> List[ImageMetadata]:
        """
        List images with optional filtering.

        Args:
            tier: Filter by image tier
            capabilities: Filter by required capabilities
            available_only: Only return locally available images

        Returns:
            List of matching ImageMetadata
        """
        result = []

        for metadata in self._images.values():
            # Filter by tier
            if tier is not None and metadata.tier != tier:
                continue

            # Filter by capabilities
            if capabilities and not metadata.manifest.has_all_capabilities(capabilities):
                continue

            # Filter by availability
            if available_only and not metadata.available_locally:
                continue

            result.append(metadata)

        # Sort by tier (base first), then name
        tier_order = {ImageTier.BASE: 0, ImageTier.SPECIALIZED: 1, ImageTier.TASK: 2}
        result.sort(key=lambda m: (tier_order.get(m.tier, 3), m.name))

        return result

    def list_by_tier(self) -> Dict[ImageTier, List[ImageMetadata]]:
        """
        List images grouped by tier.

        Returns:
            Dictionary mapping tiers to image lists
        """
        result: Dict[ImageTier, List[ImageMetadata]] = {
            ImageTier.BASE: [],
            ImageTier.SPECIALIZED: [],
            ImageTier.TASK: [],
        }

        for metadata in self._images.values():
            result[metadata.tier].append(metadata)

        return result

    def find_by_capability(
        self,
        capability: ImageCapability,
        prefer_local: bool = True,
    ) -> Optional[ImageMetadata]:
        """
        Find an image with a specific capability.

        Args:
            capability: Required capability
            prefer_local: Prefer locally available images

        Returns:
            ImageMetadata or None if not found
        """
        candidates = []

        for metadata in self._images.values():
            if metadata.manifest.has_capability(capability):
                candidates.append(metadata)

        if not candidates:
            return None

        # Sort by preference
        def sort_key(m: ImageMetadata) -> tuple:
            # Prefer: local, task tier, specialized tier, base tier
            tier_order = {ImageTier.TASK: 0, ImageTier.SPECIALIZED: 1, ImageTier.BASE: 2}
            return (
                0 if m.available_locally else 1,
                tier_order.get(m.tier, 3),
            )

        if prefer_local:
            candidates.sort(key=sort_key)

        return candidates[0]

    def has_image(self, image: str) -> bool:
        """
        Check if an image is in the registry.

        Args:
            image: Image name

        Returns:
            True if image exists
        """
        try:
            self.get(image)
            return True
        except ImageNotFoundError:
            return False

    def is_available_locally(self, image: str) -> bool:
        """
        Check if an image is available locally.

        Args:
            image: Image name

        Returns:
            True if image is pulled locally
        """
        try:
            metadata = self.get(image)
            return metadata.available_locally
        except ImageNotFoundError:
            return False

    def register(self, metadata: ImageMetadata) -> None:
        """
        Register an image in the registry.

        Args:
            metadata: Image metadata to register
        """
        self._images[metadata.full_name] = metadata
        logger.debug(f"Registered image: {metadata.full_name}")

    def unregister(self, image: str) -> bool:
        """
        Remove an image from the registry.

        Args:
            image: Image name to remove

        Returns:
            True if removed, False if not found
        """
        # Normalize name
        if ":" not in image:
            image = f"{image}:1.0.0"

        if image in self._images:
            del self._images[image]
            logger.debug(f"Unregistered image: {image}")
            return True
        return False

    def clear(self) -> None:
        """Clear all images from the registry."""
        self._images.clear()
        if self._include_builtins:
            self._load_builtins()

    @property
    def count(self) -> int:
        """Get the number of images in the registry."""
        return len(self._images)

    @property
    def last_refresh(self) -> Optional[datetime]:
        """Get the timestamp of the last refresh."""
        return self._last_refresh

    def to_dict(self) -> Dict[str, Any]:
        """
        Export registry state as dictionary.

        Returns:
            Dictionary with registry state
        """
        return {
            "images": {name: meta.to_dict() for name, meta in self._images.items()},
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "image_patterns": self._image_patterns,
        }
