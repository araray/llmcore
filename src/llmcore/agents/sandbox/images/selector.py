# src/llmcore/agents/sandbox/images/selector.py
"""
Image Selector for automatic task-based image selection.

The ImageSelector chooses the most appropriate container image based on
task requirements, capabilities needed, and configuration preferences.

Selection Strategy:
    1. Check for explicit task-to-image mappings
    2. Infer capabilities from task type and context
    3. Select best matching image from registry
    4. Fall back to default image if no match

Example:
    >>> selector = ImageSelector(registry)
    >>> image = selector.select_for_task(task_type="research")
    >>> print(image)  # llmcore-sandbox-research:1.0.0

    >>> image = selector.select_for_runtime(runtime="python")
    >>> print(image)  # llmcore-sandbox-python:1.0.0
"""

import logging
from dataclasses import dataclass, field

from .models import (
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageMetadata,
    ImageTier,
)
from .registry import ImageNotFoundError, ImageRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Task Type Mappings
# ============================================================================

# Mapping of task types to required capabilities
TASK_CAPABILITY_MAP: dict[str, set[ImageCapability]] = {
    # Research tasks
    "research": {ImageCapability.PYTHON, ImageCapability.RESEARCH, ImageCapability.NETWORK},
    "literature_review": {ImageCapability.PYTHON, ImageCapability.RESEARCH},
    "document_analysis": {ImageCapability.PYTHON, ImageCapability.PDF_TOOLS},
    # Web tasks
    "websearch": {ImageCapability.PYTHON, ImageCapability.WEBSEARCH, ImageCapability.BROWSER},
    "web_scraping": {ImageCapability.PYTHON, ImageCapability.NETWORK, ImageCapability.BROWSER},
    "api_testing": {ImageCapability.PYTHON, ImageCapability.NETWORK, ImageCapability.CURL},
    # Development tasks
    "python_dev": {ImageCapability.PYTHON, ImageCapability.GIT},
    "nodejs_dev": {ImageCapability.NODEJS, ImageCapability.GIT},
    "frontend": {ImageCapability.NODEJS},
    "backend": {ImageCapability.PYTHON},
    "fullstack": {ImageCapability.PYTHON, ImageCapability.NODEJS},
    # Data tasks
    "data_analysis": {ImageCapability.PYTHON, ImageCapability.DATA_ANALYSIS},
    "visualization": {ImageCapability.PYTHON, ImageCapability.VISUALIZATION},
    "data_processing": {ImageCapability.PYTHON},
    # Document tasks
    "docgen": {ImageCapability.PYTHON, ImageCapability.PANDOC},
    "pdf_generation": {ImageCapability.PYTHON, ImageCapability.PDF_TOOLS},
    "markdown": {ImageCapability.SHELL, ImageCapability.PANDOC},
    # Testing tasks
    "testing": {ImageCapability.PYTHON, ImageCapability.TESTING},
    "integration_testing": {
        ImageCapability.PYTHON,
        ImageCapability.TESTING,
        ImageCapability.NETWORK,
    },
    "code_review": {ImageCapability.PYTHON, ImageCapability.CODE_ANALYSIS},
    # Shell tasks
    "shell": {ImageCapability.SHELL},
    "scripting": {ImageCapability.SHELL, ImageCapability.JQ},
    "automation": {ImageCapability.SHELL, ImageCapability.CURL},
    "file_processing": {ImageCapability.SHELL},
}

# Mapping of task types to preferred images
TASK_IMAGE_MAP: dict[str, str] = {
    "research": "llmcore-sandbox-research:1.0.0",
    "literature_review": "llmcore-sandbox-research:1.0.0",
    "document_analysis": "llmcore-sandbox-research:1.0.0",
    "websearch": "llmcore-sandbox-websearch:1.0.0",
    "web_scraping": "llmcore-sandbox-websearch:1.0.0",
    "python_dev": "llmcore-sandbox-python:1.0.0",
    "nodejs_dev": "llmcore-sandbox-nodejs:1.0.0",
    "frontend": "llmcore-sandbox-nodejs:1.0.0",
    "backend": "llmcore-sandbox-python:1.0.0",
    "fullstack": "llmcore-sandbox-python:1.0.0",
    "data_analysis": "llmcore-sandbox-python:1.0.0",
    "visualization": "llmcore-sandbox-python:1.0.0",
    "shell": "llmcore-sandbox-shell:1.0.0",
    "scripting": "llmcore-sandbox-shell:1.0.0",
    "automation": "llmcore-sandbox-shell:1.0.0",
}

# Mapping of runtime to images
RUNTIME_IMAGE_MAP: dict[str, str] = {
    "python": "llmcore-sandbox-python:1.0.0",
    "python3": "llmcore-sandbox-python:1.0.0",
    "python3.12": "llmcore-sandbox-python:1.0.0",
    "node": "llmcore-sandbox-nodejs:1.0.0",
    "nodejs": "llmcore-sandbox-nodejs:1.0.0",
    "npm": "llmcore-sandbox-nodejs:1.0.0",
    "bash": "llmcore-sandbox-shell:1.0.0",
    "sh": "llmcore-sandbox-shell:1.0.0",
    "shell": "llmcore-sandbox-shell:1.0.0",
    "zsh": "llmcore-sandbox-shell:1.0.0",
}


@dataclass
class SelectionConfig:
    """
    Configuration for image selection.

    Attributes:
        default_image: Fallback image when no match found
        prefer_local: Prefer locally available images
        prefer_task_images: Prefer task-tier images over specialized
        task_overrides: Custom task-to-image mappings
        runtime_overrides: Custom runtime-to-image mappings
        restricted_patterns: Image patterns that are restricted (network disabled)
        full_access_patterns: Image patterns with full access (network enabled)
    """

    default_image: str = "llmcore-sandbox-python:1.0.0"
    prefer_local: bool = True
    prefer_task_images: bool = True
    task_overrides: dict[str, str] = field(default_factory=dict)
    runtime_overrides: dict[str, str] = field(default_factory=dict)
    restricted_patterns: list[str] = field(
        default_factory=lambda: [
            "*-slim",
            "llmcore-sandbox-base:*",
            "llmcore-sandbox-shell:*",
        ]
    )
    full_access_patterns: list[str] = field(
        default_factory=lambda: [
            "*-full",
            "llmcore-sandbox-websearch:*",
            "llmcore-sandbox-research:*",
        ]
    )


@dataclass
class SelectionResult:
    """
    Result of image selection.

    Attributes:
        image: Selected image name
        manifest: Image manifest
        reason: Why this image was selected
        alternatives: Other candidate images
        access_mode: Determined access mode
    """

    image: str
    manifest: ImageManifest | None = None
    reason: str = "default"
    alternatives: list[str] = field(default_factory=list)
    access_mode: AccessMode = AccessMode.RESTRICTED


class ImageSelector:
    """
    Selector for automatic task-based image selection.

    The selector uses a combination of task mappings, capability requirements,
    and registry lookup to choose the best image for a given task.

    Example:
        >>> registry = ImageRegistry()
        >>> selector = ImageSelector(registry)
        >>> result = selector.select_for_task("research")
        >>> print(result.image)  # llmcore-sandbox-research:1.0.0
    """

    def __init__(
        self,
        registry: ImageRegistry | None = None,
        config: SelectionConfig | None = None,
    ):
        """
        Initialize the image selector.

        Args:
            registry: Image registry for lookups
            config: Selection configuration
        """
        self._registry = registry or ImageRegistry()
        self._config = config or SelectionConfig()

        # Merge custom mappings
        self._task_images = {**TASK_IMAGE_MAP, **self._config.task_overrides}
        self._runtime_images = {**RUNTIME_IMAGE_MAP, **self._config.runtime_overrides}

    @property
    def registry(self) -> ImageRegistry:
        """Get the image registry."""
        return self._registry

    @property
    def config(self) -> SelectionConfig:
        """Get the selection configuration."""
        return self._config

    def select_for_task(
        self,
        task_type: str,
        runtime: str | None = None,
        capabilities: set[ImageCapability] | None = None,
        prefer_local: bool | None = None,
    ) -> SelectionResult:
        """
        Select an image for a specific task type.

        Args:
            task_type: Type of task (e.g., "research", "websearch", "python_dev")
            runtime: Preferred runtime (e.g., "python", "nodejs")
            capabilities: Additional required capabilities
            prefer_local: Override prefer_local config

        Returns:
            SelectionResult with selected image
        """
        task_type = task_type.lower().strip()
        prefer_local = prefer_local if prefer_local is not None else self._config.prefer_local
        alternatives = []

        logger.debug(f"Selecting image for task: {task_type}, runtime: {runtime}")

        # 1. Check explicit task mapping (trust user/config mappings directly)
        if task_type in self._task_images:
            image = self._task_images[task_type]
            return self._build_result(image, f"task_mapping:{task_type}", alternatives)

        # 2. Check runtime mapping if provided (trust user/config mappings directly)
        if runtime:
            runtime = runtime.lower().strip()
            if runtime in self._runtime_images:
                image = self._runtime_images[runtime]
                return self._build_result(image, f"runtime_mapping:{runtime}", alternatives)

        # 3. Infer capabilities and find matching image
        required_caps = capabilities or set()

        # Add capabilities from task type
        if task_type in TASK_CAPABILITY_MAP:
            required_caps = required_caps.union(TASK_CAPABILITY_MAP[task_type])

        if required_caps:
            image = self._select_by_capabilities(required_caps, prefer_local)
            if image:
                return self._build_result(
                    image.full_name,
                    f"capability_match:{','.join(c.value for c in required_caps)}",
                    alternatives,
                )

        # 4. Fallback to default
        logger.debug(f"Using default image for task: {task_type}")
        return self._build_result(
            self._config.default_image,
            "default",
            alternatives,
        )

    def select_for_runtime(
        self,
        runtime: str,
        prefer_local: bool | None = None,
    ) -> SelectionResult:
        """
        Select an image for a specific runtime.

        Args:
            runtime: Runtime name (e.g., "python", "nodejs", "bash")
            prefer_local: Override prefer_local config

        Returns:
            SelectionResult with selected image
        """
        runtime = runtime.lower().strip()
        prefer_local = prefer_local if prefer_local is not None else self._config.prefer_local
        alternatives = []

        # Check runtime mapping (trust user/config mappings directly)
        if runtime in self._runtime_images:
            image = self._runtime_images[runtime]
            return self._build_result(image, f"runtime_mapping:{runtime}", alternatives)

        # Try to find by capability
        cap_map = {
            "python": ImageCapability.PYTHON,
            "python3": ImageCapability.PYTHON,
            "nodejs": ImageCapability.NODEJS,
            "node": ImageCapability.NODEJS,
            "bash": ImageCapability.SHELL,
            "shell": ImageCapability.SHELL,
        }

        if runtime in cap_map:
            metadata = self._registry.find_by_capability(cap_map[runtime], prefer_local)
            if metadata:
                return self._build_result(
                    metadata.full_name,
                    f"capability_search:{runtime}",
                    alternatives,
                )

        # Fallback
        return self._build_result(self._config.default_image, "default", alternatives)

    def select_for_capabilities(
        self,
        capabilities: set[ImageCapability],
        prefer_local: bool | None = None,
    ) -> SelectionResult:
        """
        Select an image that has all required capabilities.

        Args:
            capabilities: Set of required capabilities
            prefer_local: Override prefer_local config

        Returns:
            SelectionResult with selected image
        """
        prefer_local = prefer_local if prefer_local is not None else self._config.prefer_local

        image = self._select_by_capabilities(capabilities, prefer_local)
        if image:
            return self._build_result(
                image.full_name,
                f"capability_match:{','.join(c.value for c in capabilities)}",
                [],
            )

        return self._build_result(self._config.default_image, "default", [])

    def _select_by_capabilities(
        self,
        capabilities: set[ImageCapability],
        prefer_local: bool,
    ) -> ImageMetadata | None:
        """
        Select the best image matching required capabilities.

        Args:
            capabilities: Required capabilities
            prefer_local: Prefer locally available images

        Returns:
            Best matching ImageMetadata or None
        """
        candidates = self._registry.list_images(capabilities=capabilities)

        if not candidates:
            logger.debug(f"No images found with capabilities: {capabilities}")
            return None

        # Score and sort candidates
        def score_image(metadata: ImageMetadata) -> tuple:
            """Score an image for selection priority."""
            # Factors (higher is better, but we sort ascending so negate)
            local_score = 0 if metadata.available_locally else 1
            tier_score = {
                ImageTier.TASK: 0,
                ImageTier.SPECIALIZED: 1,
                ImageTier.BASE: 2,
            }.get(metadata.tier, 3)

            # Extra capabilities (fewer extras = better fit)
            extra_caps = len(metadata.capabilities - capabilities)

            if not prefer_local:
                local_score = 0

            if not self._config.prefer_task_images:
                tier_score = 0

            return (local_score, tier_score, extra_caps)

        candidates.sort(key=score_image)
        return candidates[0]

    def _build_result(
        self,
        image: str,
        reason: str,
        alternatives: list[str],
    ) -> SelectionResult:
        """Build a SelectionResult from image name."""
        manifest = None
        access_mode = AccessMode.RESTRICTED

        try:
            metadata = self._registry.get(image)
            manifest = metadata.manifest
            access_mode = manifest.default_access_mode
        except ImageNotFoundError:
            # Try to determine access mode from patterns
            access_mode = self._determine_access_mode(image)

        return SelectionResult(
            image=image,
            manifest=manifest,
            reason=reason,
            alternatives=alternatives,
            access_mode=access_mode,
        )

    def _determine_access_mode(self, image: str) -> AccessMode:
        """Determine access mode from image name patterns."""
        import fnmatch

        # Check full access patterns first
        for pattern in self._config.full_access_patterns:
            if fnmatch.fnmatch(image, pattern):
                return AccessMode.FULL

        # Check restricted patterns
        for pattern in self._config.restricted_patterns:
            if fnmatch.fnmatch(image, pattern):
                return AccessMode.RESTRICTED

        # Default to restricted
        return AccessMode.RESTRICTED

    def get_task_types(self) -> list[str]:
        """Get list of known task types."""
        return sorted(set(self._task_images.keys()) | set(TASK_CAPABILITY_MAP.keys()))

    def get_runtimes(self) -> list[str]:
        """Get list of known runtimes."""
        return sorted(self._runtime_images.keys())

    def get_image_for_task(self, task_type: str) -> str | None:
        """
        Get the preferred image for a task type.

        Args:
            task_type: Task type name

        Returns:
            Image name or None if no specific mapping
        """
        return self._task_images.get(task_type.lower().strip())

    def get_image_for_runtime(self, runtime: str) -> str | None:
        """
        Get the preferred image for a runtime.

        Args:
            runtime: Runtime name

        Returns:
            Image name or None if no specific mapping
        """
        return self._runtime_images.get(runtime.lower().strip())

    def add_task_mapping(self, task_type: str, image: str) -> None:
        """
        Add or update a task-to-image mapping.

        Args:
            task_type: Task type name
            image: Image name
        """
        self._task_images[task_type.lower().strip()] = image

    def add_runtime_mapping(self, runtime: str, image: str) -> None:
        """
        Add or update a runtime-to-image mapping.

        Args:
            runtime: Runtime name
            image: Image name
        """
        self._runtime_images[runtime.lower().strip()] = image
