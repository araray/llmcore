# src/llmcore/agents/activities/schema.py
"""
Activity System Schema.

Defines the core data models for the Activity-based execution system, which provides
a model-agnostic abstraction for tool execution using structured text protocols.

The Activity System enables:
- Universal model support (any instruction-following LLM can use activities)
- Structured request/response format using XML
- Risk-level based HITL integration
- Extensible activity registry

Research Foundation:
- ReAct: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- CoALA: Sumers et al., "Cognitive Architectures for Language Agents" (2023)

References:
    - Master Plan: Section 9 (Activity Orchestration System)
    - Technical Spec: Section 5.4 (Activity System)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    # Fallback if pydantic not available
    from dataclasses import dataclass as BaseModel  # type: ignore

    ConfigDict = dict  # type: ignore
    Field = field  # type: ignore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ActivityCategory(str, Enum):
    """Category of activity for organization and filtering."""

    FILE_OPERATIONS = "file_operations"  # File read/write/search/delete
    CODE_EXECUTION = "code_execution"  # Python/bash/shell execution
    WEB = "web"  # HTTP requests, web search
    DATA = "data"  # JSON/CSV/data processing
    MEMORY = "memory"  # Memory store/search
    CONTROL = "control"  # Flow control (final_answer, ask_human)
    SYSTEM = "system"  # System operations
    CUSTOM = "custom"  # User-defined activities


class RiskLevel(str, Enum):
    """Risk level for HITL integration."""

    NONE = "none"  # Always auto-execute (final_answer, think_aloud)
    LOW = "low"  # Auto-execute by default (read_file, list_directory)
    MEDIUM = "medium"  # Prompt if configured (write_file, http_request)
    HIGH = "high"  # Always prompt (execute_shell, delete_file)
    CRITICAL = "critical"  # Prompt + confirm (execute_sudo, drop_database)


class ExecutionTarget(str, Enum):
    """Where the activity should be executed."""

    LOCAL = "local"  # Local machine (host)
    DOCKER = "docker"  # Docker container
    VM = "vm"  # Virtual machine sandbox
    REMOTE = "remote"  # Remote host via SSH
    DRY_RUN = "dry_run"  # Simulate only, no actual execution


class ActivityStatus(str, Enum):
    """Status of activity execution."""

    PENDING = "pending"  # Not yet started
    RUNNING = "running"  # Currently executing
    SUCCESS = "success"  # Completed successfully
    FAILED = "failed"  # Execution failed
    TIMEOUT = "timeout"  # Execution timed out
    REJECTED = "rejected"  # Rejected by HITL or validation
    SKIPPED = "skipped"  # Skipped (e.g., already completed)


class ParameterType(str, Enum):
    """Parameter data types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


# =============================================================================
# PARAMETER SCHEMA
# =============================================================================


class ParameterSchema(BaseModel):
    """Schema for an activity parameter."""

    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(ParameterType.STRING, description="Parameter type")
    description: str = Field("", description="Parameter description")
    required: bool = Field(False, description="Whether parameter is required")
    default: Any | None = Field(None, description="Default value")
    enum: list[str] | None = Field(None, description="Allowed values if constrained")
    min_value: float | None = Field(None, description="Minimum value for numbers")
    max_value: float | None = Field(None, description="Maximum value for numbers")
    pattern: str | None = Field(None, description="Regex pattern for strings")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# ACTIVITY DEFINITION
# =============================================================================


class ActivityDefinition(BaseModel):
    """
    Definition of an executable activity.

    An activity is a named, parameterized action that can be requested by an LLM
    and executed by the activity system. Activities are the model-agnostic
    alternative to native tool calling.

    Example:
        >>> activity = ActivityDefinition(
        ...     name="file_read",
        ...     category=ActivityCategory.FILE_OPERATIONS,
        ...     description="Read the contents of a file",
        ...     parameters=[
        ...         ParameterSchema(name="path", type=ParameterType.STRING, required=True),
        ...         ParameterSchema(name="encoding", type=ParameterType.STRING, default="utf-8"),
        ...     ],
        ...     risk_level=RiskLevel.LOW,
        ... )
    """

    name: str = Field(..., description="Unique activity name (snake_case)")
    category: ActivityCategory = Field(..., description="Activity category")
    description: str = Field(..., description="Human-readable description")
    parameters: list[ParameterSchema] = Field(
        default_factory=list, description="List of parameter schemas"
    )
    risk_level: RiskLevel = Field(RiskLevel.LOW, description="Risk level for HITL")
    requires_sandbox: bool = Field(False, description="Whether activity requires sandbox execution")
    timeout_seconds: int = Field(60, description="Default execution timeout")
    supported_targets: list[ExecutionTarget] = Field(
        default_factory=lambda: [ExecutionTarget.LOCAL, ExecutionTarget.DOCKER],
        description="Supported execution targets",
    )
    version: str = Field("1.0.0", description="Activity version")
    deprecated: bool = Field(False, description="Whether activity is deprecated")
    deprecation_message: str | None = Field(None, description="Deprecation message")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    examples: list[str] = Field(default_factory=list, description="Usage examples")

    model_config = ConfigDict(extra="allow")

    def get_required_parameters(self) -> list[ParameterSchema]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]

    def get_parameter(self, name: str) -> ParameterSchema | None:
        """Get parameter schema by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def to_prompt_format(self) -> str:
        """Format activity for inclusion in LLM prompts."""
        lines = [f"### {self.name}", f"Category: {self.category.value}"]
        lines.append(f"Description: {self.description}")
        lines.append(f"Risk Level: {self.risk_level.value}")

        if self.parameters:
            lines.append("Parameters:")
            for param in self.parameters:
                req_marker = " (required)" if param.required else ""
                default = f" [default: {param.default}]" if param.default else ""
                lines.append(f"  - {param.name}: {param.description}{req_marker}{default}")

        return "\n".join(lines)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for validation."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type.value}
            if param.description:
                prop["description"] = param.description
            if param.enum:
                prop["enum"] = param.enum
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value
            if param.pattern:
                prop["pattern"] = param.pattern

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


# =============================================================================
# ACTIVITY REQUEST
# =============================================================================


class ActivityRequest(BaseModel):
    """
    A request to execute an activity.

    Parsed from LLM output (typically XML format) and validated against
    the activity definition before execution.

    Example:
        >>> request = ActivityRequest(
        ...     activity="file_read",
        ...     parameters={"path": "/var/log/app.log"},
        ...     reason="Need to check application logs for errors",
        ... )
    """

    activity: str = Field(..., description="Activity name to execute")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Activity parameters")
    target: ExecutionTarget = Field(ExecutionTarget.DOCKER, description="Execution target")
    reason: str | None = Field(None, description="Reasoning for this activity")
    request_id: str | None = Field(None, description="Unique request identifier")
    timeout_seconds: int | None = Field(None, description="Override default timeout")
    priority: int = Field(0, description="Execution priority (higher = sooner)")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# ACTIVITY RESULT
# =============================================================================


class ActivityResult(BaseModel):
    """
    Result of activity execution.

    Contains the output, status, and metadata from executing an activity.
    Used to build observations for the LLM.
    """

    activity: str = Field(..., description="Activity that was executed")
    status: ActivityStatus = Field(..., description="Execution status")
    output: str = Field("", description="Activity output (stdout)")
    error: str | None = Field(None, description="Error message if failed")
    return_code: int | None = Field(None, description="Process return code")
    duration_ms: int = Field(0, description="Execution duration in milliseconds")
    target: ExecutionTarget = Field(
        ExecutionTarget.LOCAL, description="Where activity was executed"
    )
    request_id: str | None = Field(None, description="Request identifier")
    structured_output: dict[str, Any] | None = Field(
        None, description="Parsed structured output if available"
    )
    artifacts: list[str] = Field(default_factory=list, description="Paths to created artifacts")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

    model_config = ConfigDict(extra="allow")

    @property
    def success(self) -> bool:
        """Check if activity completed successfully."""
        return self.status == ActivityStatus.SUCCESS

    def to_observation(self, max_length: int = 2000) -> str:
        """
        Format result as an observation for the LLM.

        Args:
            max_length: Maximum observation length

        Returns:
            Formatted observation string
        """
        lines = [f"Activity: {self.activity}", f"Status: {self.status.value}"]

        if self.success:
            output = self.output
            if len(output) > max_length:
                output = (
                    output[:max_length] + f"\n... [truncated {len(self.output) - max_length} chars]"
                )
            lines.append(f"Output:\n{output}")
        else:
            if self.error:
                lines.append(f"Error: {self.error}")
            if self.return_code is not None:
                lines.append(f"Return code: {self.return_code}")

        if self.artifacts:
            lines.append(f"Artifacts: {', '.join(self.artifacts)}")

        lines.append(f"Duration: {self.duration_ms}ms")

        return "\n".join(lines)


# =============================================================================
# ACTIVITY EXECUTION
# =============================================================================


@dataclass
class ActivityExecution:
    """
    Record of a single activity execution.

    Pairs a request with its result for tracking and replay.
    """

    request: ActivityRequest
    result: ActivityResult | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    hitl_approval: bool | None = None  # True=approved, False=rejected, None=not needed

    @property
    def pending(self) -> bool:
        """Check if execution is still pending."""
        return self.result is None

    @property
    def success(self) -> bool:
        """Check if execution completed successfully."""
        return self.result is not None and self.result.success


# =============================================================================
# ACTIVITY LOOP RESULT
# =============================================================================


class ActivityLoopResult(BaseModel):
    """
    Result of processing LLM output through the activity loop.

    Indicates whether to continue the cognitive cycle and includes
    any observations from executed activities.
    """

    should_continue: bool = Field(True, description="Whether to continue iteration")
    is_final_answer: bool = Field(False, description="Whether this is a final answer")
    observation: str = Field("", description="Combined observation from activities")
    executions: list[ActivityExecution] = Field(
        default_factory=list, description="Activity executions"
    )
    remaining_text: str = Field("", description="Text not parsed as activities")
    parse_errors: list[str] = Field(default_factory=list, description="Errors during parsing")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def all_succeeded(self) -> bool:
        """Check if all activities succeeded."""
        return all(e.success for e in self.executions)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ActivityCategory",
    "RiskLevel",
    "ExecutionTarget",
    "ActivityStatus",
    "ParameterType",
    # Parameter
    "ParameterSchema",
    # Core models
    "ActivityDefinition",
    "ActivityRequest",
    "ActivityResult",
    "ActivityExecution",
    "ActivityLoopResult",
]
