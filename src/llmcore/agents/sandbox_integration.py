# src/llmcore/agents/sandbox_integration.py
"""
Sandbox Integration Module for LLMCore Agents.

This module provides the binding code that integrates the sandbox system
with llmcore's existing AgentManager and ToolManager. It handles:

1. Registration of sandbox tools with ToolManager
2. Sandbox lifecycle management during agent runs
3. Output tracking and artifact preservation
4. Configuration bridging from llmcore config to sandbox config

Usage:
    # In AgentManager initialization
    from llmcore.agents.sandbox_integration import (
        SandboxIntegration,
        register_sandbox_tools
    )

    # Register sandbox tools globally
    register_sandbox_tools(tool_manager)

    # Create integration for agent runs
    integration = SandboxIntegration(config)
    await integration.initialize()

    # During agent run
    async with integration.sandbox_context(task) as sandbox:
        # Tools now execute in sandbox
        result = await tool_manager.execute_tool(tool_call)

SECURITY INVARIANT:
    When sandbox is active, all execute_* tools run inside the sandbox.
    Code NEVER executes on the host system.
"""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from .sandbox import (
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    OutputTracker,
    SandboxConfig,
    SandboxError,
    SandboxProvider,
    SandboxRegistry,
    SandboxRegistryConfig,
    clear_active_sandbox,
    create_registry_config,
    load_sandbox_config,
    set_active_sandbox,
)

if TYPE_CHECKING:
    from ..models import AgentTask
    from .tools import ToolManager

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL REGISTRATION
# =============================================================================


def register_sandbox_tools(tool_manager: "ToolManager") -> None:
    """
    Register sandbox tools with the ToolManager's implementation registry.

    This adds sandbox execution capabilities to the agent's toolkit.
    Should be called once during llmcore initialization.

    Args:
        tool_manager: The ToolManager instance to register tools with

    Example:
        >>> from llmcore.agents import ToolManager
        >>> from llmcore.agents.sandbox_integration import register_sandbox_tools
        >>>
        >>> tool_manager = ToolManager(memory_manager, storage_manager)
        >>> register_sandbox_tools(tool_manager)

    Registered tools:
        - llmcore.tools.sandbox.execute_shell
        - llmcore.tools.sandbox.execute_python
        - llmcore.tools.sandbox.save_file
        - llmcore.tools.sandbox.load_file
        - llmcore.tools.sandbox.replace_in_file
        - llmcore.tools.sandbox.list_files
        - llmcore.tools.sandbox.file_exists
        - llmcore.tools.sandbox.delete_file
        - llmcore.tools.sandbox.create_directory
        - llmcore.tools.sandbox.get_state
        - llmcore.tools.sandbox.set_state
        - llmcore.tools.sandbox.get_sandbox_info
    """
    # Import the tools module's registry
    from . import tools as tool_module

    # Add sandbox tool implementations to the registry
    if hasattr(tool_module, "_IMPLEMENTATION_REGISTRY"):
        tool_module._IMPLEMENTATION_REGISTRY.update(SANDBOX_TOOL_IMPLEMENTATIONS)
        logger.info(f"Registered {len(SANDBOX_TOOL_IMPLEMENTATIONS)} sandbox tools")
    else:
        logger.warning("Could not find _IMPLEMENTATION_REGISTRY in tools module")

    # Add descriptions
    if hasattr(tool_module, "_IMPLEMENTATION_DESCRIPTIONS"):
        from .sandbox.tools import SANDBOX_TOOL_DESCRIPTIONS

        tool_module._IMPLEMENTATION_DESCRIPTIONS.update(SANDBOX_TOOL_DESCRIPTIONS)


def get_sandbox_tool_definitions() -> list:
    """
    Get tool definitions for sandbox tools in OpenAI function calling format.

    Returns:
        List of tool schema dictionaries for function calling

    Example:
        >>> tools = get_sandbox_tool_definitions()
        >>> # Use with LLM provider
        >>> response = await provider.chat(messages, tools=tools)
    """
    return list(SANDBOX_TOOL_SCHEMAS.values())


# =============================================================================
# SANDBOX INTEGRATION CLASS
# =============================================================================


class SandboxIntegration:
    """
    Manages sandbox lifecycle integration with AgentManager.

    This class provides:
    - Sandbox registry initialization from llmcore config
    - Output tracking for agent runs
    - Context manager for sandbox-scoped execution
    - Cleanup and error handling

    Example:
        >>> from llmcore.agents.sandbox_integration import SandboxIntegration
        >>>
        >>> # Initialize from config
        >>> integration = SandboxIntegration.from_llmcore_config(llmcore_config)
        >>> await integration.initialize()
        >>>
        >>> # Use in agent run
        >>> async with integration.sandbox_context(task) as ctx:
        ...     # All tool executions now happen in sandbox
        ...     result = await tool_manager.execute_tool(tool_call)
        ...     ctx.log_execution("execute_shell", "ls -la", result)
        >>>
        >>> # Cleanup when done
        >>> await integration.shutdown()

    Attributes:
        registry: The SandboxRegistry instance
        tracker: The OutputTracker instance
        config: The SandboxRegistryConfig
    """

    def __init__(self, config: SandboxRegistryConfig):
        """
        Initialize sandbox integration.

        Args:
            config: SandboxRegistryConfig for the registry
        """
        self._config = config
        self._registry: Optional[SandboxRegistry] = None
        self._tracker: Optional[OutputTracker] = None
        self._initialized = False

    @classmethod
    def from_llmcore_config(cls, llmcore_config: Any) -> "SandboxIntegration":
        """
        Create SandboxIntegration from llmcore's configuration object.

        Args:
            llmcore_config: The llmcore Config object containing sandbox settings

        Returns:
            Configured SandboxIntegration instance

        Example:
            >>> from llmcore import LLMCore
            >>> llm = await LLMCore.create()
            >>> integration = SandboxIntegration.from_llmcore_config(llm.config)
        """
        # Extract sandbox config section
        sandbox_dict = {}

        if hasattr(llmcore_config, "agents") and hasattr(llmcore_config.agents, "sandbox"):
            sandbox_section = llmcore_config.agents.sandbox
            if hasattr(sandbox_section, "as_dict"):
                sandbox_dict = sandbox_section.as_dict()
            elif isinstance(sandbox_section, dict):
                sandbox_dict = sandbox_section

        # Load and convert config
        sandbox_config = load_sandbox_config(overrides=sandbox_dict)
        registry_config = create_registry_config(sandbox_config)

        return cls(registry_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SandboxIntegration":
        """
        Create SandboxIntegration from a configuration dictionary.

        Args:
            config_dict: Dictionary with sandbox configuration

        Returns:
            Configured SandboxIntegration instance
        """
        sandbox_config = load_sandbox_config(overrides=config_dict)
        registry_config = create_registry_config(sandbox_config)
        return cls(registry_config)

    async def initialize(self) -> None:
        """
        Initialize the sandbox registry and output tracker.

        Must be called before using sandbox_context().
        """
        if self._initialized:
            return

        self._registry = SandboxRegistry(self._config)
        self._tracker = OutputTracker(base_path=self._config.outputs_path)
        self._initialized = True

        logger.info("Sandbox integration initialized")

    async def shutdown(self) -> None:
        """
        Shutdown the sandbox integration, cleaning up all sandboxes.
        """
        if self._registry:
            await self._registry.cleanup_all()

        self._initialized = False
        logger.info("Sandbox integration shut down")

    @asynccontextmanager
    async def sandbox_context(
        self, task: "AgentTask", sandbox_config: Optional[SandboxConfig] = None
    ) -> AsyncGenerator["SandboxContext", None]:
        """
        Context manager for sandbox-scoped agent execution.

        Creates a sandbox, sets it as active for tool execution,
        tracks outputs, and ensures cleanup on exit.

        Args:
            task: The AgentTask being executed
            sandbox_config: Optional custom sandbox configuration

        Yields:
            SandboxContext with sandbox, run_id, and helper methods

        Example:
            >>> async with integration.sandbox_context(task) as ctx:
            ...     print(f"Sandbox ID: {ctx.sandbox_id}")
            ...     result = await sandbox.execute_shell("ls")
            ...     ctx.log_execution("execute_shell", "ls", result)

        Raises:
            SandboxError: If sandbox creation or initialization fails
        """
        if not self._initialized:
            await self.initialize()

        # Create sandbox config if not provided
        if sandbox_config is None:
            sandbox_config = SandboxConfig(
                timeout_seconds=getattr(task, "timeout", 600) or 600,
                network_enabled=getattr(task, "network_enabled", False),
            )

        # Create sandbox
        sandbox = await self._registry.create_sandbox(sandbox_config)
        sandbox_id = sandbox.get_config().sandbox_id

        # Create tracking entry
        run_id = await self._tracker.create_run(
            sandbox_id=sandbox_id,
            sandbox_type=sandbox.get_info().get("provider", "unknown"),
            access_level=sandbox.get_access_level().value,
            task_description=getattr(task, "goal", str(task)),
            metadata={
                "task_id": getattr(task, "id", None),
                "tenant_id": getattr(task, "tenant_id", None),
            },
        )

        # Set as active for tool execution
        set_active_sandbox(sandbox, self._registry)

        # Create context object
        ctx = SandboxContext(
            sandbox=sandbox, registry=self._registry, tracker=self._tracker, run_id=run_id
        )

        success = False
        error_message = ""

        try:
            yield ctx
            success = True

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in sandbox context: {e}")
            raise

        finally:
            # Always cleanup
            clear_active_sandbox()

            # Finalize tracking
            await self._tracker.finalize_run(
                run_id,
                sandbox=sandbox,
                success=success,
                error_message=error_message,
                preserve_state=True,
            )

            # Cleanup sandbox
            await self._registry.cleanup_sandbox(sandbox_id)

    @property
    def registry(self) -> Optional[SandboxRegistry]:
        """Get the sandbox registry."""
        return self._registry

    @property
    def tracker(self) -> Optional[OutputTracker]:
        """Get the output tracker."""
        return self._tracker

    @property
    def is_initialized(self) -> bool:
        """Check if integration is initialized."""
        return self._initialized


class SandboxContext:
    """
    Context object provided during sandbox-scoped execution.

    Provides convenient access to sandbox, tracking, and helper methods.

    Attributes:
        sandbox: The active SandboxProvider
        sandbox_id: The sandbox's unique ID
        run_id: The output tracking run ID
    """

    def __init__(
        self,
        sandbox: SandboxProvider,
        registry: SandboxRegistry,
        tracker: OutputTracker,
        run_id: str,
    ):
        self.sandbox = sandbox
        self._registry = registry
        self._tracker = tracker
        self.run_id = run_id

    @property
    def sandbox_id(self) -> str:
        """Get the sandbox ID."""
        return self.sandbox.get_config().sandbox_id

    @property
    def access_level(self) -> str:
        """Get the sandbox access level."""
        return self.sandbox.get_access_level().value

    async def log_execution(self, tool_name: str, input_text: str, result: Any) -> None:
        """
        Log a tool execution to the output tracker.

        Args:
            tool_name: Name of the tool executed
            input_text: Input provided to the tool
            result: ExecutionResult or similar from the tool
        """
        await self._tracker.log_execution(self.run_id, tool_name, input_text, result)

    async def track_file(self, path: str, size_bytes: int = 0, description: str = "") -> None:
        """
        Track a file created in the sandbox.

        Args:
            path: Path to the file
            size_bytes: File size in bytes
            description: Description of the file
        """
        await self._tracker.track_file(self.run_id, path, size_bytes, description)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check if a tool is allowed in this sandbox.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is allowed
        """
        return self._registry.is_tool_allowed(tool_name, self.sandbox.get_access_level())

    async def execute_shell(self, command: str, timeout: int = None) -> Any:
        """
        Execute a shell command in the sandbox.

        Convenience method that also logs the execution.

        Args:
            command: Shell command to execute
            timeout: Optional timeout override

        Returns:
            ExecutionResult from the sandbox
        """
        result = await self.sandbox.execute_shell(command, timeout)
        await self.log_execution("execute_shell", command, result)
        return result

    async def execute_python(self, code: str, timeout: int = None) -> Any:
        """
        Execute Python code in the sandbox.

        Convenience method that also logs the execution.

        Args:
            code: Python code to execute
            timeout: Optional timeout override

        Returns:
            ExecutionResult from the sandbox
        """
        result = await self.sandbox.execute_python(code, timeout)
        await self.log_execution("execute_python", code[:200], result)
        return result


# =============================================================================
# AGENTMANAGER MIXIN
# =============================================================================


class SandboxAgentMixin:
    """
    Mixin class that adds sandbox capabilities to AgentManager.

    This mixin can be used to extend AgentManager with sandbox support
    without modifying the original class.

    Example:
        >>> class SandboxEnabledAgentManager(AgentManager, SandboxAgentMixin):
        ...     pass
        >>>
        >>> manager = SandboxEnabledAgentManager(provider_manager, ...)
        >>> await manager.initialize_sandbox(config)
        >>> result = await manager.run_agent_loop_sandboxed(task)
    """

    _sandbox_integration: Optional[SandboxIntegration] = None

    async def initialize_sandbox(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize sandbox support for the agent manager.

        Args:
            config: Optional sandbox configuration dictionary
        """
        if config:
            self._sandbox_integration = SandboxIntegration.from_dict(config)
        else:
            # Try to get from llmcore config
            if hasattr(self, "_provider_manager") and hasattr(self._provider_manager, "config"):
                self._sandbox_integration = SandboxIntegration.from_llmcore_config(
                    self._provider_manager.config
                )
            else:
                # Use defaults
                self._sandbox_integration = SandboxIntegration.from_dict({})

        await self._sandbox_integration.initialize()

        # Register sandbox tools with our tool manager
        if hasattr(self, "_tool_manager"):
            register_sandbox_tools(self._tool_manager)

    async def shutdown_sandbox(self) -> None:
        """Shutdown sandbox support."""
        if self._sandbox_integration:
            await self._sandbox_integration.shutdown()

    async def run_agent_loop_sandboxed(self, task: "AgentTask", **kwargs) -> str:
        """
        Run the agent loop with sandbox isolation.

        This wraps run_agent_loop with sandbox context management.

        Args:
            task: The AgentTask to execute
            **kwargs: Additional arguments for run_agent_loop

        Returns:
            Result string from the agent
        """
        if not self._sandbox_integration:
            raise SandboxError("Sandbox not initialized. Call initialize_sandbox() first.")

        async with self._sandbox_integration.sandbox_context(task) as ctx:
            # The sandbox is now active - all tool executions go through it
            return await self.run_agent_loop(task, **kwargs)
