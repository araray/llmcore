# src/llmcore/agents/prompts/models.py
"""
Core data models for the Prompt Library System.

This module defines the foundational models for managing versioned prompt
templates, reusable snippets, and performance metrics. The design treats
prompts as first-class artifacts that can be versioned, composed, and
optimized independently of code changes.

Design Principles:
    - Immutability: PromptVersions are immutable once created
    - Composability: Templates can include snippets via {{@snippet_key}}
    - Measurability: Every version tracks performance metrics
    - Traceability: Content hashes enable drift detection

References:
    - Technical Spec: Section 5.2 (Prompt Library Architecture)
    - Dossier: Step 2.1 (Prompt Library Core Models)
"""

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# ENUMERATIONS
# =============================================================================


class PromptCategory(str, Enum):
    """Categories for organizing prompts and snippets."""

    SYSTEM = "system"  # System-level prompts
    PLANNING = "planning"  # Strategic planning prompts
    REASONING = "reasoning"  # Think/reasoning prompts
    REFLECTION = "reflection"  # Self-reflection prompts
    TOOL_USE = "tool_use"  # Tool usage instructions
    PERSONA = "persona"  # Persona-specific prompts
    SNIPPET = "snippet"  # Reusable fragments
    CUSTOM = "custom"  # User-defined


class VersionStatus(str, Enum):
    """Status of a prompt version."""

    DRAFT = "draft"  # Under development
    ACTIVE = "active"  # Currently in use
    ARCHIVED = "archived"  # No longer in use
    DEPRECATED = "deprecated"  # Replaced by newer version


# =============================================================================
# SNIPPET MODEL
# =============================================================================


class PromptSnippet(BaseModel):
    """
    A reusable prompt fragment that can be included in templates.

    Snippets are small, focused pieces of prompt text that can be
    composed into larger templates using the {{@key}} syntax.

    Example:
        >>> snippet = PromptSnippet(
        ...     key="tool_usage_instructions",
        ...     content="Always use the appropriate tool for the task...",
        ...     category=PromptCategory.TOOL_USE
        ... )
        >>> # Use in template: {{@tool_usage_instructions}}

    Attributes:
        key: Unique identifier (snake_case)
        content: The snippet content to be included
        category: Category for organization
        description: Human-readable purpose description
        tags: Tags for filtering and discovery
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """

    key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique identifier for the snippet (snake_case)",
    )
    content: str = Field(..., min_length=1, description="The snippet content to be included")
    category: PromptCategory = Field(
        default=PromptCategory.SNIPPET, description="Category for organization"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of the snippet's purpose"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for filtering and discovery")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to include content_hash in serialization."""
        data = super().model_dump(**kwargs)
        data["content_hash"] = self.content_hash
        return data


# =============================================================================
# VARIABLE MODEL
# =============================================================================


class PromptVariable(BaseModel):
    """
    Definition of a variable placeholder in a prompt template.

    Variables are placeholders in the form {{variable_name}} that are
    substituted at render time with actual values.

    Example:
        >>> var = PromptVariable(
        ...     name="goal",
        ...     description="The agent's primary objective",
        ...     required=True,
        ...     example="Calculate the sum of prime numbers less than 100"
        ... )

    Attributes:
        name: Variable name (matches {{name}} in template)
        description: What this variable represents
        required: Whether this variable must be provided
        default_value: Default value if not provided (only if not required)
        example: Example value for documentation
    """

    name: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Variable name (matches {{name}} in template)",
    )
    description: Optional[str] = Field(default=None, description="What this variable represents")
    required: bool = Field(default=True, description="Whether this variable must be provided")
    default_value: Optional[str] = Field(
        default=None, description="Default value if not provided (only if not required)"
    )
    example: Optional[str] = Field(default=None, description="Example value for documentation")

    @model_validator(mode="after")
    def validate_default(self) -> "PromptVariable":
        """Ensure default_value is only set if not required."""
        if self.required and self.default_value is not None:
            raise ValueError("Required variables cannot have a default_value")
        return self


# =============================================================================
# METRICS MODEL
# =============================================================================


class PromptMetrics(BaseModel):
    """
    Performance metrics for a prompt version.

    Tracks usage statistics and outcomes to enable A/B testing
    and continuous prompt optimization.

    Example:
        >>> metrics = PromptMetrics(version_id="v1")
        >>> metrics.total_uses = 100
        >>> metrics.successful_uses = 85
        >>> print(f"Success rate: {metrics.success_rate:.1%}")
        Success rate: 85.0%

    Attributes:
        version_id: Associated version ID
        total_uses: Total number of times this version was used
        successful_uses: Number of successful completions
        failed_uses: Number of failed completions
        avg_iterations_to_success: Average cognitive loop iterations when successful
        avg_tokens_used: Average token consumption per use
        avg_latency_ms: Average end-to-end latency in milliseconds
        success_rate: Ratio of successful uses (0.0 to 1.0)
        avg_quality_score: Average quality rating (0.0 to 1.0)
        last_used_at: Timestamp of last use
        created_at: Metrics creation timestamp
        updated_at: Last metrics update timestamp
    """

    version_id: str = Field(..., description="Associated version ID")

    # Usage counts
    total_uses: int = Field(default=0, ge=0)
    successful_uses: int = Field(default=0, ge=0)
    failed_uses: int = Field(default=0, ge=0)

    # Performance metrics
    avg_iterations_to_success: Optional[float] = Field(
        default=None, description="Average cognitive loop iterations when successful"
    )
    avg_tokens_used: Optional[float] = Field(
        default=None, description="Average token consumption per use"
    )
    avg_latency_ms: Optional[float] = Field(
        default=None, description="Average end-to-end latency in milliseconds"
    )

    # Quality metrics (0.0 to 1.0)
    success_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Ratio of successful uses"
    )
    avg_quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Average quality rating"
    )

    # Timestamps
    last_used_at: Optional[datetime] = Field(default=None, description="Timestamp of last use")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_counts(self) -> "PromptMetrics":
        """Ensure successful + failed = total."""
        if self.successful_uses + self.failed_uses != self.total_uses:
            # Auto-correct total_uses
            self.total_uses = self.successful_uses + self.failed_uses
        return self

    def calculate_success_rate(self) -> float:
        """Calculate and update success rate."""
        if self.total_uses == 0:
            self.success_rate = 0.0
        else:
            self.success_rate = self.successful_uses / self.total_uses
        return self.success_rate

    def record_use(
        self,
        success: bool,
        iterations: Optional[int] = None,
        tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Record a single use of this prompt version.

        Args:
            success: Whether the use was successful
            iterations: Number of cognitive loop iterations
            tokens: Total tokens consumed
            latency_ms: End-to-end latency in milliseconds
            quality_score: Quality rating (0.0 to 1.0)
        """
        self.total_uses += 1
        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1

        # Update running averages
        if iterations is not None:
            if self.avg_iterations_to_success is None:
                self.avg_iterations_to_success = float(iterations)
            else:
                # Running average
                self.avg_iterations_to_success = (
                    self.avg_iterations_to_success * (self.total_uses - 1) + iterations
                ) / self.total_uses

        if tokens is not None:
            if self.avg_tokens_used is None:
                self.avg_tokens_used = float(tokens)
            else:
                self.avg_tokens_used = (
                    self.avg_tokens_used * (self.total_uses - 1) + tokens
                ) / self.total_uses

        if latency_ms is not None:
            if self.avg_latency_ms is None:
                self.avg_latency_ms = latency_ms
            else:
                self.avg_latency_ms = (
                    self.avg_latency_ms * (self.total_uses - 1) + latency_ms
                ) / self.total_uses

        if quality_score is not None:
            if self.avg_quality_score is None:
                self.avg_quality_score = quality_score
            else:
                self.avg_quality_score = (
                    self.avg_quality_score * (self.total_uses - 1) + quality_score
                ) / self.total_uses

        # Update derived metrics
        self.calculate_success_rate()
        self.last_used_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


# =============================================================================
# VERSION MODEL
# =============================================================================


class PromptVersion(BaseModel):
    """
    An immutable version of a prompt template.

    Each version represents a snapshot of the prompt at a point in time.
    Versions are immutable - modifications create new versions.

    Example:
        >>> version = PromptVersion(
        ...     template_id="planning_prompt",
        ...     version_number=2,
        ...     content="Plan: {{goal}}\\n\\nContext: {{context}}",
        ...     variables=[
        ...         PromptVariable(name="goal", required=True),
        ...         PromptVariable(name="context", required=False, default_value="")
        ...     ]
        ... )

    Attributes:
        id: Unique version ID
        template_id: Parent template ID
        version_number: Sequential version number
        content: The actual prompt template content
        variables: Variable definitions for this template
        status: Current status (draft/active/archived/deprecated)
        change_description: What changed in this version
        created_by: Who created this version
        created_at: Creation timestamp
        activated_at: When this version became active
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = Field(..., description="Parent template ID")
    version_number: int = Field(..., ge=1, description="Sequential version number")
    content: str = Field(..., min_length=1, description="The actual prompt template content")
    variables: List[PromptVariable] = Field(
        default_factory=list, description="Variable definitions for this template"
    )
    status: VersionStatus = Field(default=VersionStatus.DRAFT, description="Current status")
    change_description: Optional[str] = Field(
        default=None, description="What changed in this version"
    )
    created_by: Optional[str] = Field(default=None, description="Who created this version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = Field(
        default=None, description="When this version became active"
    )

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def variable_names(self) -> Set[str]:
        """Set of all variable names defined in this version."""
        return {var.name for var in self.variables}

    @property
    def required_variable_names(self) -> Set[str]:
        """Set of required variable names."""
        return {var.name for var in self.variables if var.required}

    def activate(self) -> None:
        """Mark this version as active."""
        self.status = VersionStatus.ACTIVE
        if self.activated_at is None:
            self.activated_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive this version."""
        self.status = VersionStatus.ARCHIVED

    def deprecate(self) -> None:
        """Mark this version as deprecated."""
        self.status = VersionStatus.DEPRECATED

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to include computed properties."""
        data = super().model_dump(**kwargs)
        data["content_hash"] = self.content_hash
        data["variable_names"] = list(self.variable_names)
        data["required_variable_names"] = list(self.required_variable_names)
        return data


# =============================================================================
# TEMPLATE MODEL
# =============================================================================


class PromptTemplate(BaseModel):
    """
    A named collection of prompt versions with metadata.

    Templates are the primary organizing unit for prompts. Each template
    can have multiple versions, with one active version at a time.

    Example:
        >>> template = PromptTemplate(
        ...     id="planning_prompt",
        ...     name="Strategic Planning Prompt",
        ...     category=PromptCategory.PLANNING,
        ...     description="Generates high-level task plans"
        ... )
        >>> # Add versions later via PromptRegistry

    Attributes:
        id: Unique template identifier (snake_case)
        name: Human-readable name
        category: Prompt category
        description: Template purpose and usage
        tags: Tags for discovery
        active_version_id: Currently active version ID
        versions: All versions of this template
        created_at: Template creation timestamp
        updated_at: Last modification timestamp
    """

    id: str = Field(
        ..., pattern=r"^[a-z][a-z0-9_]*$", description="Unique template identifier (snake_case)"
    )
    name: str = Field(..., min_length=1, description="Human-readable name")
    category: PromptCategory = Field(..., description="Prompt category")
    description: Optional[str] = Field(default=None, description="Template purpose and usage")
    tags: List[str] = Field(default_factory=list, description="Tags for discovery")
    active_version_id: Optional[str] = Field(
        default=None, description="Currently active version ID"
    )
    versions: List[PromptVersion] = Field(
        default_factory=list, description="All versions of this template"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def version_count(self) -> int:
        """Total number of versions."""
        return len(self.versions)

    @property
    def active_version(self) -> Optional[PromptVersion]:
        """Get the currently active version."""
        if not self.active_version_id:
            return None
        for version in self.versions:
            if version.id == self.active_version_id:
                return version
        return None

    @property
    def latest_version(self) -> Optional[PromptVersion]:
        """Get the latest version (highest version_number)."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version_number)

    def add_version(self, version: PromptVersion) -> None:
        """
        Add a new version to this template.

        Args:
            version: The version to add

        Raises:
            ValueError: If version.template_id doesn't match this template's id
        """
        if version.template_id != self.id:
            raise ValueError(
                f"Version template_id '{version.template_id}' doesn't match template id '{self.id}'"
            )

        # Ensure version_number is correct
        if not self.versions:
            version.version_number = 1
        else:
            expected_number = max(v.version_number for v in self.versions) + 1
            if version.version_number != expected_number:
                version.version_number = expected_number

        self.versions.append(version)
        self.updated_at = datetime.utcnow()

    def set_active_version(self, version_id: str) -> None:
        """
        Set the active version for this template.

        Args:
            version_id: The version ID to activate

        Raises:
            ValueError: If version_id doesn't exist in this template
        """
        # Verify version exists
        version = None
        for v in self.versions:
            if v.id == version_id:
                version = v
                break

        if not version:
            raise ValueError(f"Version {version_id} not found in template {self.id}")

        # Deactivate old version
        if self.active_version:
            self.active_version.status = VersionStatus.ARCHIVED

        # Activate new version
        version.activate()
        self.active_version_id = version_id
        self.updated_at = datetime.utcnow()

    def get_version(self, version_id: str) -> Optional[PromptVersion]:
        """Get a specific version by ID."""
        for version in self.versions:
            if version.id == version_id:
                return version
        return None

    def get_version_by_number(self, version_number: int) -> Optional[PromptVersion]:
        """Get a specific version by version number."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to include computed properties."""
        data = super().model_dump(**kwargs)
        data["version_count"] = self.version_count
        return data


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PromptCategory",
    "VersionStatus",
    # Models
    "PromptSnippet",
    "PromptVariable",
    "PromptMetrics",
    "PromptVersion",
    "PromptTemplate",
]
