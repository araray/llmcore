# src/llmcore/agents/activities/registry.py
"""
Activity Registry.

Manages registration, discovery, and lookup of activities. The registry supports:
- Built-in activities (file operations, code execution, etc.)
- User-defined activities
- Plugin-based activity extensions

Registry Architecture:
    ┌────────────────────────────────────────────────────────┐
    │                      ACTIVITY REGISTRY                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │   BUILTIN    │  │    USER      │  │   PLUGIN     │  │
    │  │ file_read    │  │ my_deploy    │  │ aws_s3_sync  │  │
    │  │ python_exec  │  │ custom_lint  │  │ slack_notify │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    │                            │                           │
    │                   ┌────────┴────────┐                  │
    │                   │  Unified Index  │                  │
    │                   └─────────────────┘                  │
    └────────────────────────────────────────────────────────┘

References:
    - Master Plan: Section 10 (Activity Registry & Discovery)
    - Technical Spec: Section 5.4.3 (Registry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from .schema import (
    ActivityCategory,
    ActivityDefinition,
    ActivityRequest,
    ParameterSchema,
    ParameterType,
    RiskLevel,
)

if TYPE_CHECKING:
    from .executor import ActivityExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# ACTIVITY HANDLER TYPE
# =============================================================================

# Activity handler signature: (request, context) -> output
ActivityHandler = Callable[[ActivityRequest, "ExecutionContext"], str]


@dataclass
class ExecutionContext:
    """Context passed to activity handlers during execution."""

    working_dir: str = "/tmp"
    sandbox_available: bool = False
    sandbox_id: Optional[str] = None
    timeout_seconds: int = 60
    environment: Dict[str, str] = field(default_factory=dict)
    session_id: Optional[str] = None
    working_memory: Dict[str, Any] = field(default_factory=dict)  # In-context memory for activities


# =============================================================================
# REGISTERED ACTIVITY
# =============================================================================


@dataclass
class RegisteredActivity:
    """
    An activity registered in the registry.

    Pairs the activity definition with its implementation handler.
    """

    definition: ActivityDefinition
    handler: Optional[ActivityHandler] = None
    source: str = "builtin"  # builtin, user, plugin
    enabled: bool = True

    @property
    def name(self) -> str:
        """Get activity name."""
        return self.definition.name

    @property
    def category(self) -> ActivityCategory:
        """Get activity category."""
        return self.definition.category


# =============================================================================
# ACTIVITY REGISTRY
# =============================================================================


class ActivityRegistry:
    """
    Registry for activity definitions and handlers.

    Manages the catalog of available activities and provides lookup,
    filtering, and formatting capabilities.

    Example:
        >>> registry = ActivityRegistry()
        >>> registry.register_builtins()
        >>>
        >>> # Lookup activity
        >>> activity = registry.get("file_read")
        >>> print(activity.definition.description)
        >>>
        >>> # Filter by category
        >>> file_ops = registry.filter_by_category(ActivityCategory.FILE_OPERATIONS)
        >>>
        >>> # Format for LLM prompt
        >>> prompt = registry.format_for_prompt()
    """

    def __init__(self, auto_register_builtins: bool = True):
        """
        Initialize the activity registry.

        Args:
            auto_register_builtins: If True, automatically register built-in activities
        """
        self._activities: Dict[str, RegisteredActivity] = {}
        self._categories: Dict[ActivityCategory, Set[str]] = {
            cat: set() for cat in ActivityCategory
        }
        self._tags: Dict[str, Set[str]] = {}

        if auto_register_builtins:
            self.register_builtins()

    def register(
        self,
        definition: ActivityDefinition,
        handler: Optional[ActivityHandler] = None,
        source: str = "user",
        enabled: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Register an activity.

        Args:
            definition: Activity definition
            handler: Optional handler function
            source: Source of the activity (builtin, user, plugin)
            enabled: Whether the activity is enabled
            overwrite: If True, overwrite existing activity with same name

        Raises:
            ValueError: If activity already exists and overwrite is False
        """
        name = definition.name

        if name in self._activities and not overwrite:
            raise ValueError(
                f"Activity '{name}' already registered. Use overwrite=True to replace."
            )

        registered = RegisteredActivity(
            definition=definition,
            handler=handler,
            source=source,
            enabled=enabled,
        )

        self._activities[name] = registered
        self._categories[definition.category].add(name)

        # Index by tags
        for tag in definition.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(name)

        logger.debug(f"Registered activity: {name} ({source})")

    def unregister(self, name: str) -> bool:
        """
        Unregister an activity.

        Args:
            name: Activity name to unregister

        Returns:
            True if activity was removed, False if not found
        """
        if name not in self._activities:
            return False

        activity = self._activities.pop(name)
        self._categories[activity.category].discard(name)

        for tag in activity.definition.tags:
            if tag in self._tags:
                self._tags[tag].discard(name)

        logger.debug(f"Unregistered activity: {name}")
        return True

    def get(self, name: str) -> Optional[RegisteredActivity]:
        """
        Get a registered activity by name.

        Args:
            name: Activity name

        Returns:
            RegisteredActivity if found, None otherwise
        """
        return self._activities.get(name)

    def get_definition(self, name: str) -> Optional[ActivityDefinition]:
        """
        Get activity definition by name.

        Args:
            name: Activity name

        Returns:
            ActivityDefinition if found, None otherwise
        """
        activity = self.get(name)
        return activity.definition if activity else None

    def exists(self, name: str) -> bool:
        """Check if activity is registered."""
        return name in self._activities

    def is_enabled(self, name: str) -> bool:
        """Check if activity is enabled."""
        activity = self.get(name)
        return activity.enabled if activity else False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """
        Enable or disable an activity.

        Args:
            name: Activity name
            enabled: Whether to enable or disable

        Returns:
            True if activity was found and updated
        """
        activity = self.get(name)
        if activity:
            activity.enabled = enabled
            return True
        return False

    def filter_by_category(
        self, category: ActivityCategory, enabled_only: bool = True
    ) -> List[RegisteredActivity]:
        """
        Filter activities by category.

        Args:
            category: Category to filter by
            enabled_only: If True, only return enabled activities

        Returns:
            List of matching activities
        """
        names = self._categories.get(category, set())
        activities = [self._activities[n] for n in names]

        if enabled_only:
            activities = [a for a in activities if a.enabled]

        return activities

    def filter_by_tag(self, tag: str, enabled_only: bool = True) -> List[RegisteredActivity]:
        """
        Filter activities by tag.

        Args:
            tag: Tag to filter by
            enabled_only: If True, only return enabled activities

        Returns:
            List of matching activities
        """
        names = self._tags.get(tag, set())
        activities = [self._activities[n] for n in names]

        if enabled_only:
            activities = [a for a in activities if a.enabled]

        return activities

    def filter_by_risk_level(
        self, max_risk: RiskLevel, enabled_only: bool = True
    ) -> List[RegisteredActivity]:
        """
        Filter activities by maximum risk level.

        Args:
            max_risk: Maximum allowed risk level
            enabled_only: If True, only return enabled activities

        Returns:
            List of activities at or below the risk level
        """
        risk_order = [
            RiskLevel.NONE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        max_index = risk_order.index(max_risk)

        activities = []
        for activity in self._activities.values():
            if enabled_only and not activity.enabled:
                continue
            risk_index = risk_order.index(activity.definition.risk_level)
            if risk_index <= max_index:
                activities.append(activity)

        return activities

    def list_all(self, enabled_only: bool = True) -> List[RegisteredActivity]:
        """
        List all registered activities.

        Args:
            enabled_only: If True, only return enabled activities

        Returns:
            List of all activities
        """
        activities = list(self._activities.values())
        if enabled_only:
            activities = [a for a in activities if a.enabled]
        return activities

    def list_names(self, enabled_only: bool = True) -> List[str]:
        """
        List all activity names.

        Args:
            enabled_only: If True, only return enabled activities

        Returns:
            List of activity names
        """
        if enabled_only:
            return [n for n, a in self._activities.items() if a.enabled]
        return list(self._activities.keys())

    def format_for_prompt(
        self,
        categories: Optional[List[ActivityCategory]] = None,
        max_risk: Optional[RiskLevel] = None,
        include_examples: bool = False,
        enabled_only: bool = True,
    ) -> str:
        """
        Format activities for inclusion in LLM prompts.

        Args:
            categories: Limit to specific categories (None for all)
            max_risk: Maximum risk level to include
            include_examples: Include usage examples
            enabled_only: Only include enabled activities

        Returns:
            Formatted string for prompt
        """
        lines = ["# Available Activities\n"]
        lines.append("To execute an activity, use the following XML format:\n")
        lines.append("```xml")
        lines.append("<activity_request>")
        lines.append("  <activity>ACTIVITY_NAME</activity>")
        lines.append("  <parameters>")
        lines.append("    <param_name>value</param_name>")
        lines.append("  </parameters>")
        lines.append("  <reason>Why you're doing this</reason>")
        lines.append("</activity_request>")
        lines.append("```\n")

        # Group by category
        by_category: Dict[ActivityCategory, List[RegisteredActivity]] = {}

        for activity in self.list_all(enabled_only=enabled_only):
            if categories and activity.category not in categories:
                continue

            if max_risk:
                risk_order = [
                    RiskLevel.NONE,
                    RiskLevel.LOW,
                    RiskLevel.MEDIUM,
                    RiskLevel.HIGH,
                    RiskLevel.CRITICAL,
                ]
                if risk_order.index(activity.definition.risk_level) > risk_order.index(max_risk):
                    continue

            if activity.category not in by_category:
                by_category[activity.category] = []
            by_category[activity.category].append(activity)

        # Format each category
        for category in ActivityCategory:
            if category not in by_category:
                continue

            activities = by_category[category]
            lines.append(f"\n## {category.value.replace('_', ' ').title()}\n")

            for activity in sorted(activities, key=lambda a: a.name):
                defn = activity.definition
                lines.append(f"### {defn.name}")
                lines.append(f"- **Description**: {defn.description}")
                lines.append(f"- **Risk Level**: {defn.risk_level.value}")

                if defn.parameters:
                    lines.append("- **Parameters**:")
                    for param in defn.parameters:
                        req = " (required)" if param.required else ""
                        default = (
                            f" [default: {param.default}]" if param.default is not None else ""
                        )
                        lines.append(
                            f"  - `{param.name}` ({param.type.value}): {param.description}{req}{default}"
                        )

                if include_examples and defn.examples:
                    lines.append("- **Examples**:")
                    for example in defn.examples[:2]:  # Limit examples
                        lines.append(f"  ```\n  {example}\n  ```")

                lines.append("")

        return "\n".join(lines)

    def register_builtins(self) -> None:
        """Register all built-in activities."""
        for defn in get_builtin_activities():
            self.register(defn, source="builtin", overwrite=True)
        logger.info(f"Registered {len(self._activities)} built-in activities")

    def __len__(self) -> int:
        """Get number of registered activities."""
        return len(self._activities)

    def __contains__(self, name: str) -> bool:
        """Check if activity is registered."""
        return name in self._activities


# =============================================================================
# BUILT-IN ACTIVITIES
# =============================================================================


def get_builtin_activities() -> List[ActivityDefinition]:
    """
    Get all built-in activity definitions.

    Returns:
        List of built-in ActivityDefinition objects
    """
    return [
        # File Operations
        ActivityDefinition(
            name="file_read",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Read the contents of a file",
            parameters=[
                ParameterSchema(
                    name="path",
                    type=ParameterType.STRING,
                    description="Path to the file to read",
                    required=True,
                ),
                ParameterSchema(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
                ParameterSchema(
                    name="max_bytes",
                    type=ParameterType.INTEGER,
                    description="Maximum bytes to read",
                    required=False,
                    default=None,
                ),
            ],
            risk_level=RiskLevel.LOW,
            tags=["file", "read", "safe"],
        ),
        ActivityDefinition(
            name="file_write",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Write content to a file (creates or overwrites)",
            parameters=[
                ParameterSchema(
                    name="path",
                    type=ParameterType.STRING,
                    description="Path to the file to write",
                    required=True,
                ),
                ParameterSchema(
                    name="content",
                    type=ParameterType.STRING,
                    description="Content to write to the file",
                    required=True,
                ),
                ParameterSchema(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
                ParameterSchema(
                    name="append",
                    type=ParameterType.BOOLEAN,
                    description="Append instead of overwrite",
                    required=False,
                    default=False,
                ),
            ],
            risk_level=RiskLevel.MEDIUM,
            requires_sandbox=True,
            tags=["file", "write"],
        ),
        ActivityDefinition(
            name="file_search",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Search for files matching a pattern",
            parameters=[
                ParameterSchema(
                    name="path",
                    type=ParameterType.STRING,
                    description="Directory path to search in",
                    required=True,
                ),
                ParameterSchema(
                    name="pattern",
                    type=ParameterType.STRING,
                    description="Glob pattern (e.g., *.py, **/*.txt)",
                    required=False,
                    default="*",
                ),
                ParameterSchema(
                    name="recursive",
                    type=ParameterType.BOOLEAN,
                    description="Search recursively",
                    required=False,
                    default=True,
                ),
                ParameterSchema(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maximum number of results",
                    required=False,
                    default=100,
                ),
            ],
            risk_level=RiskLevel.LOW,
            tags=["file", "search", "safe"],
        ),
        ActivityDefinition(
            name="file_delete",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Delete a file or directory",
            parameters=[
                ParameterSchema(
                    name="path",
                    type=ParameterType.STRING,
                    description="Path to delete",
                    required=True,
                ),
                ParameterSchema(
                    name="recursive",
                    type=ParameterType.BOOLEAN,
                    description="Delete directories recursively",
                    required=False,
                    default=False,
                ),
            ],
            risk_level=RiskLevel.HIGH,
            requires_sandbox=True,
            tags=["file", "delete", "dangerous"],
        ),
        # Code Execution
        ActivityDefinition(
            name="python_exec",
            category=ActivityCategory.CODE_EXECUTION,
            description="Execute Python code",
            parameters=[
                ParameterSchema(
                    name="code",
                    type=ParameterType.STRING,
                    description="Python code to execute",
                    required=True,
                ),
                ParameterSchema(
                    name="timeout",
                    type=ParameterType.INTEGER,
                    description="Execution timeout in seconds",
                    required=False,
                    default=30,
                ),
            ],
            risk_level=RiskLevel.HIGH,
            requires_sandbox=True,
            timeout_seconds=30,
            tags=["code", "python", "execute"],
        ),
        ActivityDefinition(
            name="bash_exec",
            category=ActivityCategory.CODE_EXECUTION,
            description="Execute bash/shell commands",
            parameters=[
                ParameterSchema(
                    name="command",
                    type=ParameterType.STRING,
                    description="Shell command to execute",
                    required=True,
                ),
                ParameterSchema(
                    name="timeout",
                    type=ParameterType.INTEGER,
                    description="Execution timeout in seconds",
                    required=False,
                    default=30,
                ),
                ParameterSchema(
                    name="working_dir",
                    type=ParameterType.STRING,
                    description="Working directory",
                    required=False,
                ),
            ],
            risk_level=RiskLevel.HIGH,
            requires_sandbox=True,
            timeout_seconds=30,
            tags=["code", "bash", "shell", "execute"],
        ),
        ActivityDefinition(
            name="code_lint",
            category=ActivityCategory.CODE_EXECUTION,
            description="Lint code for errors and style issues",
            parameters=[
                ParameterSchema(
                    name="code",
                    type=ParameterType.STRING,
                    description="Code to lint",
                    required=True,
                ),
                ParameterSchema(
                    name="language",
                    type=ParameterType.STRING,
                    description="Programming language",
                    required=True,
                    enum=["python", "javascript", "typescript"],
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["code", "lint", "safe"],
        ),
        # Web
        ActivityDefinition(
            name="web_fetch",
            category=ActivityCategory.WEB,
            description="Fetch content from a URL",
            parameters=[
                ParameterSchema(
                    name="url",
                    type=ParameterType.STRING,
                    description="URL to fetch",
                    required=True,
                ),
                ParameterSchema(
                    name="method",
                    type=ParameterType.STRING,
                    description="HTTP method",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "DELETE"],
                ),
                ParameterSchema(
                    name="headers",
                    type=ParameterType.OBJECT,
                    description="HTTP headers",
                    required=False,
                ),
            ],
            risk_level=RiskLevel.LOW,
            timeout_seconds=30,
            tags=["web", "http", "fetch"],
        ),
        ActivityDefinition(
            name="web_search",
            category=ActivityCategory.WEB,
            description="Search the web for information",
            parameters=[
                ParameterSchema(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ParameterSchema(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maximum number of results",
                    required=False,
                    default=10,
                ),
            ],
            risk_level=RiskLevel.LOW,
            tags=["web", "search"],
        ),
        # Data
        ActivityDefinition(
            name="json_query",
            category=ActivityCategory.DATA,
            description="Query JSON data using JSONPath or jq syntax",
            parameters=[
                ParameterSchema(
                    name="data",
                    type=ParameterType.STRING,
                    description="JSON data to query",
                    required=True,
                ),
                ParameterSchema(
                    name="query",
                    type=ParameterType.STRING,
                    description="JSONPath or jq query",
                    required=True,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["data", "json", "query", "safe"],
        ),
        ActivityDefinition(
            name="csv_analyze",
            category=ActivityCategory.DATA,
            description="Analyze CSV data",
            parameters=[
                ParameterSchema(
                    name="data",
                    type=ParameterType.STRING,
                    description="CSV data or file path",
                    required=True,
                ),
                ParameterSchema(
                    name="operation",
                    type=ParameterType.STRING,
                    description="Analysis operation",
                    required=True,
                    enum=["describe", "head", "tail", "columns", "count"],
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["data", "csv", "analyze", "safe"],
        ),
        # Memory
        ActivityDefinition(
            name="memory_store",
            category=ActivityCategory.MEMORY,
            description="Store information in working or long-term memory",
            parameters=[
                ParameterSchema(
                    name="key",
                    type=ParameterType.STRING,
                    description="Key to store under",
                    required=True,
                ),
                ParameterSchema(
                    name="value",
                    type=ParameterType.STRING,
                    description="Value to store (use JSON string for complex data)",
                    required=True,
                ),
                ParameterSchema(
                    name="memory_type",
                    type=ParameterType.STRING,
                    description="Type of memory: 'working' (default) or 'longterm'",
                    required=False,
                    default="working",
                    enum=["working", "longterm"],
                ),
                ParameterSchema(
                    name="metadata",
                    type=ParameterType.OBJECT,
                    description="Optional metadata dict to store with the value",
                    required=False,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["memory", "store", "safe"],
        ),
        ActivityDefinition(
            name="memory_search",
            category=ActivityCategory.MEMORY,
            description="Search working and/or long-term memory for relevant items",
            parameters=[
                ParameterSchema(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ParameterSchema(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maximum results to return",
                    required=False,
                    default=5,
                ),
                ParameterSchema(
                    name="search_working",
                    type=ParameterType.BOOLEAN,
                    description="Whether to search working memory (default: True)",
                    required=False,
                    default=True,
                ),
                ParameterSchema(
                    name="search_longterm",
                    type=ParameterType.BOOLEAN,
                    description="Whether to search long-term memory (default: True)",
                    required=False,
                    default=True,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["memory", "search", "safe"],
        ),
        # Control
        ActivityDefinition(
            name="final_answer",
            category=ActivityCategory.CONTROL,
            description="Provide the final answer to the user's goal",
            parameters=[
                ParameterSchema(
                    name="answer",
                    type=ParameterType.STRING,
                    description="The final answer",
                    required=True,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["control", "answer", "safe"],
        ),
        ActivityDefinition(
            name="ask_human",
            category=ActivityCategory.CONTROL,
            description="Ask the human for clarification or input",
            parameters=[
                ParameterSchema(
                    name="question",
                    type=ParameterType.STRING,
                    description="Question to ask",
                    required=True,
                ),
                ParameterSchema(
                    name="context",
                    type=ParameterType.STRING,
                    description="Context for the question",
                    required=False,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["control", "human", "input", "safe"],
        ),
        ActivityDefinition(
            name="think_aloud",
            category=ActivityCategory.CONTROL,
            description="Record reasoning or thinking process",
            parameters=[
                ParameterSchema(
                    name="thought",
                    type=ParameterType.STRING,
                    description="The thought or reasoning",
                    required=True,
                ),
            ],
            risk_level=RiskLevel.NONE,
            tags=["control", "thinking", "safe"],
        ),
    ]


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

# Default global registry instance
_default_registry: Optional[ActivityRegistry] = None


def get_default_registry() -> ActivityRegistry:
    """
    Get the default global activity registry.

    Returns:
        Default ActivityRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ActivityRegistry(auto_register_builtins=True)
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default global registry."""
    global _default_registry
    _default_registry = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "ActivityRegistry",
    "RegisteredActivity",
    "ExecutionContext",
    # Types
    "ActivityHandler",
    # Functions
    "get_builtin_activities",
    "get_default_registry",
    "reset_default_registry",
]
