# src/llmcore/agents/manager.py
"""
Agent Management for LLMCore.

This module provides the AgentManager, which orchestrates the agentic execution loop.
It coordinates between the LLM provider, memory systems, and tool execution to
achieve complex goals autonomously.

REFACTORED: This class now acts as a high-level orchestrator, delegating the
implementation of the cognitive cycle (Plan, Think, Act, etc.) and prompt
management to the `cognitive_cycle` and `prompt_utils` modules respectively.
This separation of concerns makes the agent's architecture clearer and more maintainable.

UPDATED: Added sandbox integration for isolated code execution. Agent tasks can now
execute code in Docker containers or VMs, ensuring host system security.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import LLMCoreError
from ..memory.manager import MemoryManager
from ..models import AgentTask
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from . import cognitive_cycle
from .tools import ToolManager

# Sandbox integration imports
from .sandbox_integration import (
    SandboxIntegration,
    register_sandbox_tools,
)
from .sandbox import SandboxError

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Orchestrates the autonomous agent execution loop with strategic planning and reflection.

    Manages the Plan -> (Think -> Act -> Observe -> Reflect) cycle by calling
    functions from the `cognitive_cycle` module. It initializes and holds the
    necessary dependencies (managers for providers, memory, storage, tools) and
    passes them to the cognitive functions as needed.

    UPDATED: Now supports optional sandbox integration for isolated code execution.
    When sandbox is enabled, all execute_* tools run inside Docker/VM containers.

    Attributes:
        _provider_manager: ProviderManager for LLM interactions
        _memory_manager: MemoryManager for context retrieval
        _storage_manager: StorageManager for episodic memory logging
        _tool_manager: ToolManager for tool registration and execution
        _sandbox_integration: Optional SandboxIntegration for isolated execution
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        memory_manager: MemoryManager,
        storage_manager: StorageManager
    ):
        """
        Initialize the AgentManager with required dependencies.

        Args:
            provider_manager: The ProviderManager for LLM interactions.
            memory_manager: The MemoryManager for context retrieval.
            storage_manager: The StorageManager for episodic memory logging.
        """
        self._provider_manager = provider_manager
        self._memory_manager = memory_manager
        self._storage_manager = storage_manager
        self._tool_manager = ToolManager(memory_manager, storage_manager)

        # NEW: Sandbox integration (optional, initialized separately)
        self._sandbox_integration: Optional[SandboxIntegration] = None
        self._sandbox_enabled: bool = False

        # Initialize tracing
        self._tracer = None
        try:
            from ..tracing import get_tracer
            self._tracer = get_tracer("llmcore.agents.manager")
        except Exception as e:
            logger.debug(f"Tracing not available for AgentManager: {e}")

        logger.info("AgentManager initialized as orchestrator.")

    # =========================================================================
    # SANDBOX INTEGRATION METHODS (NEW)
    # =========================================================================

    async def initialize_sandbox(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize sandbox support for agent execution.

        When sandbox is enabled, all execute_shell, execute_python, and file
        operation tools will run inside isolated Docker containers or VMs.
        This ensures agent-generated code never executes on the host system.

        Args:
            config: Optional sandbox configuration dictionary. If None, uses
                    defaults from llmcore config or built-in defaults.

        Example:
            >>> await agent_manager.initialize_sandbox({
            ...     "mode": "docker",
            ...     "docker": {
            ...         "image": "python:3.11-slim",
            ...         "memory_limit": "1g"
            ...     }
            ... })

        Raises:
            SandboxError: If sandbox initialization fails
        """
        try:
            if config:
                self._sandbox_integration = SandboxIntegration.from_dict(config)
            else:
                # Try to get config from provider_manager if available
                if hasattr(self._provider_manager, 'config'):
                    self._sandbox_integration = SandboxIntegration.from_llmcore_config(
                        self._provider_manager.config
                    )
                else:
                    # Use defaults
                    self._sandbox_integration = SandboxIntegration.from_dict({})

            await self._sandbox_integration.initialize()

            # Register sandbox tools with our tool manager
            register_sandbox_tools(self._tool_manager)

            self._sandbox_enabled = True
            logger.info("Sandbox integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")
            raise SandboxError(f"Sandbox initialization failed: {e}")

    async def shutdown_sandbox(self) -> None:
        """
        Shutdown sandbox support and cleanup all active sandboxes.

        Should be called when the AgentManager is being disposed of,
        or when sandbox support is no longer needed.
        """
        if self._sandbox_integration:
            await self._sandbox_integration.shutdown()
            self._sandbox_enabled = False
            logger.info("Sandbox integration shut down")

    @property
    def sandbox_enabled(self) -> bool:
        """Check if sandbox is enabled."""
        return self._sandbox_enabled

    @property
    def sandbox_integration(self) -> Optional[SandboxIntegration]:
        """Get the sandbox integration instance."""
        return self._sandbox_integration

    # =========================================================================
    # AGENT LOOP METHODS
    # =========================================================================

    async def run_agent_loop(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None,
        use_sandbox: Optional[bool] = None
    ) -> str:
        """
        Orchestrates the Plan -> (Think -> Act -> Observe -> Reflect) loop.

        UPDATED: Now supports optional sandbox isolation. When sandbox is enabled
        and use_sandbox is True (or None with sandbox initialized), all code
        execution happens inside isolated containers.

        Args:
            task: The AgentTask containing the goal and constraints.
            provider_name: Override the default LLM provider.
            model_name: Override the default model.
            max_iterations: Maximum cognitive cycles before stopping.
            session_id: Optional session ID for memory context.
            db_session: Database session for tool loading.
            enabled_toolkits: List of toolkit names to enable.
            use_sandbox: Whether to use sandbox for this run.
                         None = use sandbox if initialized.
                         True = require sandbox (error if not initialized).
                         False = skip sandbox even if initialized.

        Returns:
            A string containing the final result/answer from the agent.

        Raises:
            LLMCoreError: If the agent loop fails.
            SandboxError: If use_sandbox=True but sandbox not initialized.
        """
        # Determine if we should use sandbox
        should_use_sandbox = self._should_use_sandbox(use_sandbox)

        if should_use_sandbox:
            return await self._run_agent_loop_sandboxed(
                task=task,
                provider_name=provider_name,
                model_name=model_name,
                max_iterations=max_iterations,
                session_id=session_id,
                db_session=db_session,
                enabled_toolkits=enabled_toolkits
            )
        else:
            return await self._run_agent_loop_direct(
                task=task,
                provider_name=provider_name,
                model_name=model_name,
                max_iterations=max_iterations,
                session_id=session_id,
                db_session=db_session,
                enabled_toolkits=enabled_toolkits
            )

    def _should_use_sandbox(self, use_sandbox: Optional[bool]) -> bool:
        """
        Determine if sandbox should be used for this run.

        Args:
            use_sandbox: User preference (True/False/None)

        Returns:
            bool: Whether to use sandbox

        Raises:
            SandboxError: If use_sandbox=True but sandbox not initialized
        """
        if use_sandbox is True:
            if not self._sandbox_enabled:
                raise SandboxError(
                    "Sandbox requested but not initialized. "
                    "Call initialize_sandbox() first."
                )
            return True
        elif use_sandbox is False:
            return False
        else:
            # None = use sandbox if available
            return self._sandbox_enabled

    async def _run_agent_loop_sandboxed(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None
    ) -> str:
        """
        Run the agent loop with sandbox isolation.

        Creates a sandbox context for the task, ensuring all tool executions
        happen inside the isolated environment.
        """
        logger.info(f"Running agent loop with sandbox for task: {task.goal[:50]}...")

        async with self._sandbox_integration.sandbox_context(task) as ctx:
            # Log sandbox info
            logger.debug(f"Sandbox created: {ctx.sandbox_id} (access: {ctx.access_level})")

            # Run the cognitive loop with sandbox active
            return await self._run_cognitive_loop(
                task=task,
                provider_name=provider_name,
                model_name=model_name,
                max_iterations=max_iterations,
                session_id=session_id,
                db_session=db_session,
                enabled_toolkits=enabled_toolkits,
                sandbox_context=ctx
            )

    async def _run_agent_loop_direct(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None
    ) -> str:
        """
        Run the agent loop without sandbox (original behavior).

        WARNING: Tool executions happen on the host system. Only use this
        for trusted tasks or when code execution is not needed.
        """
        logger.info(f"Running agent loop (no sandbox) for task: {task.goal[:50]}...")

        return await self._run_cognitive_loop(
            task=task,
            provider_name=provider_name,
            model_name=model_name,
            max_iterations=max_iterations,
            session_id=session_id,
            db_session=db_session,
            enabled_toolkits=enabled_toolkits,
            sandbox_context=None
        )

    async def _run_cognitive_loop(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None,
        sandbox_context: Optional[Any] = None
    ) -> str:
        """
        Execute the core cognitive loop (Plan -> Think -> Act -> Observe -> Reflect).

        This is the internal implementation that handles the actual agent execution.
        It can operate with or without a sandbox context.

        Args:
            task: The AgentTask to execute
            provider_name: Optional LLM provider override
            model_name: Optional model override
            max_iterations: Maximum cognitive cycles
            session_id: Optional session ID for context
            db_session: Database session for tools
            enabled_toolkits: Toolkits to enable
            sandbox_context: Optional SandboxContext if running sandboxed

        Returns:
            The final result string from the agent
        """
        from ..tracing import create_span, add_span_attributes, record_span_exception

        with create_span(self._tracer, "agent.run_loop") as span:
            try:
                add_span_attributes(span, {
                    "agent.goal": task.goal[:100],
                    "agent.max_iterations": max_iterations,
                    "agent.sandbox_enabled": sandbox_context is not None
                })

                # Generate session ID if not provided
                if not session_id:
                    import uuid
                    session_id = f"agent-{uuid.uuid4().hex[:8]}"

                # Load tools for this run
                if db_session and enabled_toolkits:
                    await self._tool_manager.load_tools_for_run(
                        db_session,
                        enabled_toolkits
                    )

                # Initialize agent state
                from ..models import AgentState
                agent_state = AgentState(goal=task.goal)

                # Execute planning step
                await cognitive_cycle.plan_step(
                    agent_state=agent_state,
                    session_id=session_id,
                    provider_manager=self._provider_manager,
                    tracer=self._tracer,
                    provider_name=provider_name,
                    model_name=model_name
                )

                # Main cognitive loop
                for iteration in range(max_iterations):
                    logger.debug(f"Cognitive iteration {iteration + 1}/{max_iterations}")

                    # Think: Decide next action
                    await cognitive_cycle.think_step(
                        agent_state=agent_state,
                        session_id=session_id,
                        provider_manager=self._provider_manager,
                        memory_manager=self._memory_manager,
                        tool_manager=self._tool_manager,
                        tracer=self._tracer,
                        provider_name=provider_name,
                        model_name=model_name
                    )

                    # Check if agent decided to finish
                    if agent_state.is_finished:
                        logger.info("Agent completed task")
                        break

                    # Check for HITL pause
                    if agent_state.awaiting_human_approval:
                        logger.info("Agent paused for human approval")
                        # In a real implementation, this would pause and wait
                        # For now, we'll just note it in the result
                        return f"HITL_PAUSE: {agent_state.pending_approval_prompt}"

                    # Act: Execute the chosen tool
                    tool_result = await cognitive_cycle.act_step(
                        agent_state=agent_state,
                        tool_manager=self._tool_manager,
                        session_id=session_id,
                        tracer=self._tracer
                    )

                    # Log execution if sandboxed
                    if sandbox_context and agent_state.pending_tool_call:
                        await sandbox_context.log_execution(
                            agent_state.pending_tool_call.name,
                            str(agent_state.pending_tool_call.arguments)[:200],
                            tool_result
                        )

                    # Observe: Process tool result
                    await cognitive_cycle.observe_step(
                        agent_state=agent_state,
                        tool_result=tool_result,
                        tracer=self._tracer
                    )

                    # Reflect: Update understanding
                    await cognitive_cycle.reflect_step(
                        agent_state=agent_state,
                        session_id=session_id,
                        provider_manager=self._provider_manager,
                        storage_manager=self._storage_manager,
                        tracer=self._tracer,
                        provider_name=provider_name,
                        model_name=model_name
                    )

                # Return final result
                final_result = agent_state.final_answer or "Task completed without explicit answer."

                add_span_attributes(span, {
                    "agent.iterations": iteration + 1,
                    "agent.success": True
                })

                return final_result

            except Exception as e:
                record_span_exception(span, e)
                logger.error(f"Agent loop failed: {e}", exc_info=True)
                raise LLMCoreError(f"Agent execution failed: {str(e)}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self._tool_manager.get_tool_names()

    def get_tool_definitions(self) -> List[Any]:
        """Get tool definitions for LLM function calling."""
        return self._tool_manager.get_tool_definitions()

    async def cleanup(self) -> None:
        """
        Cleanup agent manager resources.

        Should be called when disposing of the AgentManager.
        """
        if self._sandbox_enabled:
            await self.shutdown_sandbox()
        logger.info("AgentManager cleanup completed")
