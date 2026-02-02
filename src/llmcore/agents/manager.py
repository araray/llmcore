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

DARWIN LAYER 2: Added EnhancedAgentManager that extends AgentManager with:
- 8-phase enhanced cognitive cycle
- Persona-based customization
- Memory integration
- Multiple execution modes (SINGLE, LEGACY, MULTI)
- Full backward compatibility maintained
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import LLMCoreError
from ..memory.manager import MemoryManager
from ..models import AgentTask
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from . import cognitive_cycle

# Observability integration imports (Phase 8)
from .observability_factory import ObservabilityComponents

# Sandbox integration imports
from .sandbox import SandboxError
from .sandbox_integration import SandboxIntegration, register_sandbox_tools
from .tools import ToolManager

logger = logging.getLogger(__name__)


# =============================================================================
# ORIGINAL AGENT MANAGER (PRESERVED)
# =============================================================================


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
        storage_manager: StorageManager,
        observability: Optional[ObservabilityComponents] = None,
    ):
        """
        Initialize the AgentManager with required dependencies.

        Args:
            provider_manager: The ProviderManager for LLM interactions.
            memory_manager: The MemoryManager for context retrieval.
            storage_manager: The StorageManager for episodic memory logging.
            observability: Optional observability components (Phase 8 integration).
                          If None, observability is disabled for this manager.
        """
        self._provider_manager = provider_manager
        self._memory_manager = memory_manager
        self._storage_manager = storage_manager
        self._tool_manager = ToolManager(memory_manager, storage_manager)

        # NEW: Sandbox integration (optional, initialized separately)
        self._sandbox_integration: Optional[SandboxIntegration] = None
        self._sandbox_enabled: bool = False

        # NEW (Phase 8): Observability integration
        self._observability = observability

        # Initialize tracing
        self._tracer = None
        try:
            from ..tracing import get_tracer

            self._tracer = get_tracer("llmcore.agents.manager")
        except Exception as e:
            logger.debug(f"Tracing not available for AgentManager: {e}")

        logger.info("AgentManager initialized as orchestrator.")

    # =========================================================================
    # SANDBOX INTEGRATION METHODS
    # =========================================================================

    async def initialize_sandbox(self, config: Optional[Dict[str, Any]] = None) -> None:
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
                if hasattr(self._provider_manager, "config"):
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
        use_sandbox: Optional[bool] = None,
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
                enabled_toolkits=enabled_toolkits,
            )
        else:
            return await self._run_agent_loop_direct(
                task=task,
                provider_name=provider_name,
                model_name=model_name,
                max_iterations=max_iterations,
                session_id=session_id,
                db_session=db_session,
                enabled_toolkits=enabled_toolkits,
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
                    "Sandbox requested but not initialized. Call initialize_sandbox() first."
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
        enabled_toolkits: Optional[List[str]] = None,
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
                sandbox_context=ctx,
            )

    async def _run_agent_loop_direct(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None,
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
            sandbox_context=None,
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
        sandbox_context: Optional[Any] = None,
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
        from ..tracing import add_span_attributes, create_span, record_span_exception

        # Track execution timing for observability
        start_time = time.monotonic()

        with create_span(self._tracer, "agent.run_loop") as span:
            try:
                add_span_attributes(
                    span,
                    {
                        "agent.goal": task.goal[:100],
                        "agent.max_iterations": max_iterations,
                        "agent.sandbox_enabled": sandbox_context is not None,
                    },
                )

                # Generate session ID if not provided
                if not session_id:
                    import uuid

                    session_id = f"agent-{uuid.uuid4().hex[:8]}"

                # === Phase 8: Log lifecycle start ===
                if self._observability and self._observability.logger:
                    try:
                        await self._observability.logger.log_lifecycle_start(
                            goal=task.goal,
                            goal_complexity=getattr(task, "complexity", None),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log lifecycle start: {e}")

                # Load tools for this run
                if db_session and enabled_toolkits:
                    await self._tool_manager.load_tools_for_run(db_session, enabled_toolkits)

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
                    model_name=model_name,
                )

                # === Phase 8: Log planning complete ===
                if self._observability and self._observability.logger:
                    try:
                        await self._observability.logger.log_cognitive_phase(
                            phase="plan",
                            input_summary=task.goal[:200],
                            output_summary=f"Generated {len(agent_state.plan)} plan steps",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log plan phase: {e}")

                # Track iteration for final result
                iteration = 0

                # Main cognitive loop
                for iteration in range(max_iterations):
                    logger.debug(f"Cognitive iteration {iteration + 1}/{max_iterations}")
                    iteration_start = time.monotonic()

                    # === Phase 8: Log iteration start ===
                    if self._observability and self._observability.logger:
                        try:
                            self._observability.logger.set_iteration(iteration + 1)
                            await self._observability.logger.log_iteration_start(
                                iteration=iteration + 1,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log iteration start: {e}")

                    # Think: Decide next action
                    think_start = time.monotonic()
                    await cognitive_cycle.think_step(
                        agent_state=agent_state,
                        session_id=session_id,
                        provider_manager=self._provider_manager,
                        memory_manager=self._memory_manager,
                        tool_manager=self._tool_manager,
                        tracer=self._tracer,
                        provider_name=provider_name,
                        model_name=model_name,
                    )
                    think_duration = (time.monotonic() - think_start) * 1000

                    # === Phase 8: Log think phase ===
                    if self._observability and self._observability.logger:
                        try:
                            await self._observability.logger.log_cognitive_phase(
                                phase="think",
                                input_summary=f"Goal: {task.goal[:100]}",
                                output_summary=f"Decided action: {agent_state.pending_tool_call.name if agent_state.pending_tool_call else 'finish'}",
                                duration_ms=think_duration,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log think phase: {e}")

                    # Check if agent decided to finish
                    if agent_state.is_finished:
                        logger.info("Agent completed task")
                        break

                    # Check for HITL pause
                    if agent_state.awaiting_human_approval:
                        logger.info("Agent paused for human approval")
                        # === Phase 8: Log HITL pause ===
                        if self._observability and self._observability.logger:
                            try:
                                from .observability import HITLEventType

                                await self._observability.logger.log_hitl(
                                    event_type=HITLEventType.APPROVAL_REQUESTED,
                                    request_id=f"req-{session_id}-{iteration}",
                                    action_type="pending_action",
                                    risk_level="medium",
                                    approval_status="pending",
                                )
                            except Exception as e:
                                logger.debug(f"Failed to log HITL pause: {e}")
                        return f"HITL_PAUSE: {agent_state.pending_approval_prompt}"

                    # Act: Execute the chosen tool
                    act_start = time.monotonic()
                    tool_result = await cognitive_cycle.act_step(
                        agent_state=agent_state,
                        tool_manager=self._tool_manager,
                        session_id=session_id,
                        tracer=self._tracer,
                    )
                    act_duration = (time.monotonic() - act_start) * 1000

                    # === Phase 8: Log activity ===
                    if (
                        self._observability
                        and self._observability.logger
                        and agent_state.pending_tool_call
                    ):
                        try:
                            success = not tool_result.content.startswith("ERROR:")
                            await self._observability.logger.log_activity(
                                activity_name=agent_state.pending_tool_call.name,
                                activity_input=agent_state.pending_tool_call.arguments,
                                activity_output=tool_result.content[:500],
                                success=success,
                                error_message=tool_result.content if not success else None,
                                duration_ms=act_duration,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log activity: {e}")

                    # Log execution if sandboxed
                    if sandbox_context and agent_state.pending_tool_call:
                        await sandbox_context.log_execution(
                            agent_state.pending_tool_call.name,
                            str(agent_state.pending_tool_call.arguments)[:200],
                            tool_result,
                        )

                    # Observe: Process tool result
                    await cognitive_cycle.observe_step(
                        agent_state=agent_state, tool_result=tool_result, tracer=self._tracer
                    )

                    # Reflect: Update understanding
                    reflect_start = time.monotonic()
                    await cognitive_cycle.reflect_step(
                        agent_state=agent_state,
                        session_id=session_id,
                        provider_manager=self._provider_manager,
                        storage_manager=self._storage_manager,
                        tracer=self._tracer,
                        provider_name=provider_name,
                        model_name=model_name,
                    )
                    reflect_duration = (time.monotonic() - reflect_start) * 1000

                    # === Phase 8: Log reflect phase ===
                    if self._observability and self._observability.logger:
                        try:
                            await self._observability.logger.log_cognitive_phase(
                                phase="reflect",
                                input_summary=f"Tool result: {tool_result.content[:100]}",
                                output_summary=agent_state.scratchpad[:200]
                                if agent_state.scratchpad
                                else "Reflected",
                                duration_ms=reflect_duration,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log reflect phase: {e}")

                    # === Phase 8: Log iteration end ===
                    iteration_duration = (time.monotonic() - iteration_start) * 1000
                    if self._observability and self._observability.logger:
                        try:
                            await self._observability.logger.log_iteration_end(
                                iteration=iteration + 1,
                                duration_ms=iteration_duration,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log iteration end: {e}")

                # Return final result
                final_result = agent_state.final_answer or "Task completed without explicit answer."

                # === Phase 8: Log lifecycle end (success) ===
                total_duration = (time.monotonic() - start_time) * 1000
                if self._observability and self._observability.logger:
                    try:
                        await self._observability.logger.log_lifecycle_end(
                            status="success",
                            total_iterations=iteration + 1,
                            duration_ms=total_duration,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log lifecycle end: {e}")

                add_span_attributes(
                    span, {"agent.iterations": iteration + 1, "agent.success": True}
                )

                return final_result

            except Exception as e:
                record_span_exception(span, e)
                logger.error(f"Agent loop failed: {e}", exc_info=True)

                # === Phase 8: Log lifecycle end (failure) ===
                total_duration = (time.monotonic() - start_time) * 1000
                if self._observability and self._observability.logger:
                    try:
                        await self._observability.logger.log_lifecycle_end(
                            status="failure",
                            exit_reason=str(e),
                            duration_ms=total_duration,
                        )
                        # Also log error event
                        await self._observability.logger.log_error(
                            error_type="execution_error",
                            error_message=str(e),
                            recoverable=False,
                        )
                    except Exception as log_e:
                        logger.debug(f"Failed to log lifecycle failure: {log_e}")

                raise LLMCoreError(f"Agent execution failed: {e!s}")

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

        # NEW (Phase 8): Close observability
        if self._observability:
            await self._observability.close()

        logger.info("AgentManager cleanup completed")

    @property
    def observability(self) -> Optional[ObservabilityComponents]:
        """Get the observability components (Phase 8)."""
        return self._observability

    @property
    def event_logger(self) -> Optional[Any]:
        """Get the event logger if observability is enabled (Phase 8)."""
        if self._observability and self._observability.enabled:
            return self._observability.logger
        return None


# =============================================================================
# DARWIN LAYER 2 - ENHANCED AGENT MANAGER (NEW)
# =============================================================================


class AgentMode(str, Enum):
    """Agent execution mode for EnhancedAgentManager."""

    SINGLE = "single"  # Single autonomous agent (Darwin Layer 2)
    LEGACY = "legacy"  # Legacy compatibility mode (uses original AgentManager)
    MULTI = "multi"  # Multi-agent (Darwin Layer 3)


class EnhancedAgentManager(AgentManager):
    """
    Enhanced agent manager with Darwin Layer 2 capabilities.

    Extends the original AgentManager with:
    - 8-phase enhanced cognitive cycle
    - Persona-based customization
    - Memory integration
    - Multiple execution modes
    - Full backward compatibility

    The EnhancedAgentManager maintains ALL original AgentManager functionality
    while adding new Darwin Layer 2 features. Existing code continues to work
    unchanged.

    Example - Original Method (Still Works):
        >>> manager = EnhancedAgentManager(...)
        >>> task = AgentTask(goal="Process files")
        >>> result = await manager.run_agent_loop(task=task)

    Example - New Enhanced Method:
        >>> manager = EnhancedAgentManager(...)
        >>> result = await manager.run(
        ...     goal="Analyze sales data",
        ...     mode=AgentMode.SINGLE,
        ...     persona="analyst"
        ... )

    Attributes:
        All AgentManager attributes (inherited)
        persona_manager: PersonaManager for persona management
        memory_integrator: CognitiveMemoryIntegrator for enhanced memory
        single_agent: SingleAgentMode for Layer 2 execution
        prompt_registry: Optional PromptRegistry
        default_mode: Default execution mode for run()
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        memory_manager: MemoryManager,
        storage_manager: StorageManager,
        prompt_registry: Optional[Any] = None,
        tracer: Optional[Any] = None,
        default_mode: AgentMode = AgentMode.SINGLE,
        observability: Optional[ObservabilityComponents] = None,
        agents_config: Optional[Any] = None,  # G3: AgentsConfig for capability/activity settings
    ):
        """
        Initialize the enhanced agent manager.

        Args:
            provider_manager: Provider manager for LLM calls
            memory_manager: Memory manager
            storage_manager: Storage manager
            prompt_registry: Optional prompt registry (Darwin Layer 2)
            tracer: Optional OpenTelemetry tracer
            default_mode: Default mode for run() method
            observability: Optional observability components (Phase 8)
        """
        # Initialize parent class (original AgentManager)
        super().__init__(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            observability=observability,
        )

        # Store additional components
        self.prompt_registry = prompt_registry
        self.default_mode = default_mode

        # Use parent's tracer if not provided
        if tracer is not None:
            self._tracer = tracer

        # G3: Store agents_config (load from defaults if not provided)
        self._agents_config = agents_config

        # Initialize Darwin Layer 2 components
        try:
            from .memory import CognitiveMemoryIntegrator
            from .persona import PersonaManager
            from .single_agent import SingleAgentMode

            self.persona_manager = PersonaManager()
            self.memory_integrator = CognitiveMemoryIntegrator(
                memory_manager=memory_manager, storage_manager=storage_manager
            )
            self.single_agent = SingleAgentMode(
                provider_manager=provider_manager,
                memory_manager=memory_manager,
                storage_manager=storage_manager,
                tool_manager=self._tool_manager,
                prompt_registry=prompt_registry,
                tracer=self._tracer,
                agents_config=self._agents_config,  # G3: Pass config for capability/activity settings
            )

            logger.info(f"EnhancedAgentManager initialized (default_mode={default_mode.value})")

        except ImportError as e:
            logger.warning(
                f"Darwin Layer 2 components not available: {e}. "
                "Using original AgentManager functionality only."
            )
            self.persona_manager = None
            self.memory_integrator = None
            self.single_agent = None

    # =========================================================================
    # DARWIN LAYER 2 - NEW ENHANCED METHODS
    # =========================================================================

    async def run(
        self,
        goal: str,
        mode: Optional[AgentMode] = None,
        persona: Optional[Any] = None,
        context: Optional[str] = None,
        max_iterations: int = 10,
        **kwargs,
    ) -> Any:
        """
        Run an agent with Darwin Layer 2 capabilities.

        This is the new enhanced interface that supports personas, advanced
        cognitive cycles, and multiple execution modes. Use this for new code.

        For backward compatibility, use the original run_agent_loop() method.

        Args:
            goal: The task goal/objective
            mode: Execution mode (SINGLE, LEGACY, MULTI)
            persona: Persona for SINGLE mode (string ID or AgentPersona object)
            context: Optional initial context
            max_iterations: Maximum iterations
            **kwargs: Additional mode-specific arguments

        Returns:
            AgentResult with execution details

        Example:
            >>> # Enhanced mode with persona
            >>> result = await manager.run(
            ...     goal="Analyze Q4 sales trends",
            ...     mode=AgentMode.SINGLE,
            ...     persona="analyst",
            ...     max_iterations=15
            ... )
            >>>
            >>> # Legacy mode (uses original cognitive cycle)
            >>> result = await manager.run(
            ...     goal="Simple calculation",
            ...     mode=AgentMode.LEGACY
            ... )
        """
        if self.single_agent is None:
            raise RuntimeError(
                "Darwin Layer 2 components not available. Use run_agent_loop() instead."
            )

        from .single_agent import AgentResult

        # Use default mode if not specified
        execution_mode = mode or self.default_mode

        logger.info(
            f"Running enhanced agent: mode={execution_mode.value}, goal='{goal[:50]}...', persona={persona}"
        )

        # Route to appropriate mode
        if execution_mode == AgentMode.SINGLE:
            return await self.single_agent.run(
                goal=goal, persona=persona, context=context, max_iterations=max_iterations, **kwargs
            )

        elif execution_mode == AgentMode.LEGACY:
            # Use original AgentManager cognitive loop
            from ..models import AgentState, AgentTask

            agent_state = AgentState(goal=goal)
            task = AgentTask(goal=goal, context=context or "", agent_state=agent_state)
            final_answer = await self.run_agent_loop(
                task=task, max_iterations=max_iterations, **kwargs
            )

            # Wrap in AgentResult for consistency
            return AgentResult(
                goal=goal,
                final_answer=final_answer,
                success=True,
                iteration_count=max_iterations,
                total_tokens=0,
                total_time_seconds=0.0,
                session_id=kwargs.get("session_id", "legacy"),
            )

        elif execution_mode == AgentMode.MULTI:
            raise NotImplementedError("Multi-agent mode requires Darwin Layer 3 implementation")

        else:
            raise ValueError(f"Unknown agent mode: {execution_mode}")

    # =========================================================================
    # PERSONA MANAGEMENT (Darwin Layer 2)
    # =========================================================================

    def create_persona(self, name: str, description: str, **kwargs) -> Any:
        """
        Create a custom persona (Darwin Layer 2).

        Args:
            name: Persona name
            description: Persona description
            **kwargs: Additional persona configuration

        Returns:
            Created AgentPersona
        """
        if self.persona_manager is None:
            raise RuntimeError("Persona system not available")

        return self.persona_manager.create_persona(
            persona_id=name.lower().replace(" ", "_"), name=name, description=description, **kwargs
        )

    def list_personas(self) -> List[Any]:
        """List all available personas (Darwin Layer 2)."""
        if self.persona_manager is None:
            return []
        return self.persona_manager.list_personas()

    def get_persona(self, persona_id: str) -> Optional[Any]:
        """Get a persona by ID (Darwin Layer 2)."""
        if self.persona_manager is None:
            return None
        return self.persona_manager.get_persona(persona_id)

    # =========================================================================
    # MEMORY MANAGEMENT (Darwin Layer 2)
    # =========================================================================

    async def consolidate_memory(self, session_id: str, agent_state: Any) -> None:
        """
        Consolidate memories from a completed session (Darwin Layer 2).

        Args:
            session_id: Session identifier
            agent_state: Final agent state
        """
        if self.memory_integrator is None:
            logger.warning("Memory integrator not available")
            return

        await self.memory_integrator.consolidate_session_memory(
            session_id=session_id, agent_state=agent_state
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def set_default_mode(self, mode: AgentMode) -> None:
        """Set the default execution mode for run()."""
        self.default_mode = mode
        logger.info(f"Default mode set to: {mode.value}")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available capabilities.

        Returns:
            Dictionary of capabilities
        """
        capabilities = {
            "modes": [mode.value for mode in AgentMode],
            "default_mode": self.default_mode.value,
            "sandbox_enabled": self.sandbox_enabled,
            "observability_enabled": self._observability.enabled if self._observability else False,
            "original_methods": ["run_agent_loop", "initialize_sandbox", "shutdown_sandbox"],
            "enhanced_methods": ["run", "create_persona", "consolidate_memory"],
        }

        if self.persona_manager:
            capabilities["personas"] = [p.name for p in self.list_personas()]
            capabilities["cognitive_phases"] = 8

        if self.prompt_registry:
            capabilities["prompt_library"] = True

        if self.memory_integrator:
            capabilities["memory_integration"] = True

        return capabilities


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AgentManager",  # Original (preserved)
    "AgentMode",  # New enum
    "EnhancedAgentManager",  # New Darwin Layer 2
]
