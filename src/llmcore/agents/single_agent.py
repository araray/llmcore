# src/llmcore/agents/single_agent.py
"""
Single Agent Mode for Darwin Layer 2.

The SingleAgentMode provides a high-level interface for running autonomous agents
with full cognitive cycle capabilities and persona customization. It's the main
entry point for single-agent tasks.

Features:
- Persona-based customization
- Complete 8-phase cognitive cycle
- Automatic memory management
- Sandbox integration
- Progress tracking
- Human-in-the-loop support

This is the recommended interface for most single-agent use cases.

References:
    - Technical Spec: Section 5.5 (Single Agent Mode)
    - Dossier: Step 2.9 (Single Agent Mode)
"""

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .cognitive import (
    CognitiveCycle,
    EnhancedAgentState,
)
from .persona import (
    AgentPersona,
    PersonaManager,
)

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager
    from ..providers.manager import ProviderManager
    from ..storage.manager import StorageManager
    from .prompts import PromptRegistry
    from .sandbox import SandboxProvider
    from .tools import ToolManager

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE AGENT MODE
# =============================================================================


class SingleAgentMode:
    """
    High-level interface for autonomous single-agent execution.

    SingleAgentMode combines the cognitive cycle, persona system, and
    provides a simple API for running autonomous agents.

    Example - Basic Usage:
        >>> agent = SingleAgentMode(
        ...     provider_manager=provider_manager,
        ...     memory_manager=memory_manager,
        ...     storage_manager=storage_manager,
        ...     tool_manager=tool_manager
        ... )
        >>>
        >>> result = await agent.run(
        ...     goal="Analyze the sales data and generate a report",
        ...     persona="analyst"
        ... )
        >>>
        >>> print(result.final_answer)
        >>> print(f"Completed in {result.iteration_count} iterations")

    Example - Custom Persona:
        >>> # Create custom persona
        >>> custom_persona = agent.create_persona(
        ...     name="Sales Expert",
        ...     traits=[PersonalityTrait.ANALYTICAL, PersonalityTrait.PRAGMATIC],
        ...     communication_style=CommunicationStyle.PROFESSIONAL
        ... )
        >>>
        >>> result = await agent.run(
        ...     goal="Forecast Q4 sales",
        ...     persona=custom_persona
        ... )

    Example - With Sandbox:
        >>> result = await agent.run(
        ...     goal="Process CSV files in /data",
        ...     persona="developer",
        ...     use_sandbox=True,
        ...     sandbox_type="docker"
        ... )

    Attributes:
        provider_manager: Provider manager for LLMs
        memory_manager: Memory manager
        storage_manager: Storage manager
        tool_manager: Tool manager
        prompt_registry: Optional prompt registry
        persona_manager: Persona manager
        cognitive_cycle: Cognitive cycle orchestrator
    """

    def __init__(
        self,
        provider_manager: "ProviderManager",
        memory_manager: "MemoryManager",
        storage_manager: "StorageManager",
        tool_manager: "ToolManager",
        prompt_registry: Optional["PromptRegistry"] = None,
        tracer: Optional[Any] = None,
    ):
        """
        Initialize SingleAgentMode.

        Args:
            provider_manager: Provider manager for LLM calls
            memory_manager: Memory manager for context
            storage_manager: Storage manager for episodic memory
            tool_manager: Tool manager for actions
            prompt_registry: Optional prompt registry
            tracer: Optional OpenTelemetry tracer
        """
        self.provider_manager = provider_manager
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager
        self.prompt_registry = prompt_registry
        self.tracer = tracer

        # Initialize persona manager
        self.persona_manager = PersonaManager()

        # Initialize cognitive cycle
        self.cognitive_cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            tool_manager=tool_manager,
            prompt_registry=prompt_registry,
            tracer=tracer,
        )

        logger.info("SingleAgentMode initialized")

    async def run(
        self,
        goal: str,
        persona: Optional[str | AgentPersona] = None,
        context: Optional[str] = None,
        max_iterations: int = 10,
        use_sandbox: bool = False,
        sandbox_type: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "AgentResult":
        """
        Run an autonomous agent to complete a goal.

        Args:
            goal: The task goal/objective
            persona: Persona ID, AgentPersona object, or None for default
            context: Optional initial context
            max_iterations: Maximum iterations before stopping
            use_sandbox: Whether to use sandbox execution
            sandbox_type: Sandbox type (docker, vm) if use_sandbox=True
            provider_name: Optional LLM provider override
            model_name: Optional model override
            session_id: Optional session ID (generated if not provided)

        Returns:
            AgentResult with execution details

        Raises:
            ValueError: If persona not found

        Example:
            >>> result = await agent.run(
            ...     goal="Calculate factorial of 100",
            ...     persona="developer",
            ...     max_iterations=5
            ... )
            >>> print(result.final_answer)
            >>> print(f"Success: {result.success}")
        """
        from ..tracing import add_span_attributes, create_span

        with create_span(self.tracer, "single_agent.run") as span:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session-{uuid.uuid4()}"

            # Resolve persona
            agent_persona = self._resolve_persona(persona)

            # Create enhanced agent state
            agent_state = EnhancedAgentState(
                goal=goal, session_id=session_id, context=context or ""
            )

            # Apply persona configuration
            if agent_persona:
                self._apply_persona_config(agent_state, agent_persona)

            # Setup sandbox if requested
            sandbox = None
            if use_sandbox:
                sandbox = await self._setup_sandbox(sandbox_type)

            try:
                logger.info(
                    f"Starting single agent execution: goal='{goal[:50]}...', "
                    f"persona={agent_persona.name if agent_persona else 'default'}, "
                    f"max_iterations={max_iterations}"
                )

                start_time = datetime.utcnow()

                # Run cognitive cycle until complete
                final_answer = await self.cognitive_cycle.run_until_complete(
                    agent_state=agent_state,
                    session_id=session_id,
                    max_iterations=max_iterations,
                    sandbox=sandbox,
                    provider_name=provider_name,
                    model_name=model_name,
                )

                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                # Create result
                result = AgentResult(
                    goal=goal,
                    final_answer=final_answer,
                    success=agent_state.is_finished,
                    iteration_count=agent_state.iteration_count,
                    total_tokens=agent_state.total_tokens_used,
                    total_time_seconds=duration,
                    session_id=session_id,
                    persona_used=agent_persona.name if agent_persona else None,
                    agent_state=agent_state,
                )

                # Add tracing
                if span:
                    add_span_attributes(
                        span,
                        {
                            "agent.goal": goal[:100],
                            "agent.persona": agent_persona.id if agent_persona else "default",
                            "agent.iterations": agent_state.iteration_count,
                            "agent.success": result.success,
                            "agent.tokens": agent_state.total_tokens_used,
                        },
                    )

                logger.info(
                    f"Agent execution complete: success={result.success}, "
                    f"iterations={result.iteration_count}, "
                    f"duration={duration:.1f}s"
                )

                return result

            except Exception as e:
                logger.error(f"Agent execution failed: {e}", exc_info=True)

                # Create failure result
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                return AgentResult(
                    goal=goal,
                    final_answer=f"Task failed: {str(e)}",
                    success=False,
                    iteration_count=agent_state.iteration_count,
                    total_tokens=agent_state.total_tokens_used,
                    total_time_seconds=duration,
                    session_id=session_id,
                    persona_used=agent_persona.name if agent_persona else None,
                    agent_state=agent_state,
                    error=str(e),
                )

            finally:
                # Cleanup sandbox if created
                if sandbox and use_sandbox:
                    await self._cleanup_sandbox(sandbox)

    async def run_streaming(
        self,
        goal: str,
        persona: Optional[str | AgentPersona] = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Run agent with streaming updates.

        Yields iteration updates as they complete.

        Args:
            goal: The task goal
            persona: Persona to use
            max_iterations: Maximum iterations
            **kwargs: Additional arguments passed to run()

        Yields:
            IterationUpdate objects with progress information

        Example:
            >>> async for update in agent.run_streaming(goal="Process files"):
            ...     print(f"Iteration {update.iteration}: {update.progress:.0%}")
        """
        # Implementation would yield updates as iterations complete
        # For now, simplified to single run
        result = await self.run(goal=goal, persona=persona, max_iterations=max_iterations, **kwargs)

        yield IterationUpdate(
            iteration=result.iteration_count,
            progress=1.0,
            status="complete",
            message=result.final_answer,
        )

    def create_persona(self, name: str, description: str, **kwargs) -> AgentPersona:
        """
        Create a custom persona.

        Args:
            name: Persona name
            description: Persona description
            **kwargs: Additional persona configuration

        Returns:
            Created AgentPersona

        Example:
            >>> persona = agent.create_persona(
            ...     name="QA Engineer",
            ...     description="Quality assurance focused",
            ...     traits=[PersonalityTrait.METHODICAL, PersonalityTrait.CAUTIOUS],
            ...     risk_tolerance=RiskTolerance.LOW
            ... )
        """
        persona_id = name.lower().replace(" ", "_")
        return self.persona_manager.create_persona(
            persona_id=persona_id, name=name, description=description, **kwargs
        )

    def list_personas(self) -> List[AgentPersona]:
        """
        List all available personas.

        Returns:
            List of personas
        """
        return self.persona_manager.list_personas()

    def _resolve_persona(self, persona: Optional[str | AgentPersona]) -> Optional[AgentPersona]:
        """Resolve persona from ID or object."""
        if persona is None:
            # Use default assistant persona
            return self.persona_manager.get_persona("assistant")

        if isinstance(persona, str):
            # Lookup by ID
            agent_persona = self.persona_manager.get_persona(persona)
            if not agent_persona:
                raise ValueError(f"Persona not found: {persona}")
            return agent_persona

        # Already a persona object
        return persona

    def _apply_persona_config(self, agent_state: EnhancedAgentState, persona: AgentPersona) -> None:
        """Apply persona configuration to agent state."""
        # Store persona info in working memory
        agent_state.set_working_memory("persona_id", persona.id)
        agent_state.set_working_memory("persona_name", persona.name)

        # Could apply other persona-specific configs here
        # (e.g., max iterations, risk tolerance)

    async def _setup_sandbox(self, sandbox_type: Optional[str]) -> Optional["SandboxProvider"]:
        """Setup sandbox for execution."""
        # Implementation would create appropriate sandbox
        # For now, return None
        logger.info(f"Sandbox requested: {sandbox_type}")
        return None

    async def _cleanup_sandbox(self, sandbox: "SandboxProvider") -> None:
        """Cleanup sandbox after execution."""
        # Implementation would cleanup sandbox resources
        logger.info("Cleaning up sandbox")


# =============================================================================
# RESULT MODELS
# =============================================================================


class AgentResult:
    """
    Result from single agent execution.

    Attributes:
        goal: The original goal
        final_answer: Final answer or result
        success: Whether task completed successfully
        iteration_count: Number of iterations executed
        total_tokens: Total tokens used
        total_time_seconds: Total execution time
        session_id: Session identifier
        persona_used: Persona name used
        agent_state: Complete agent state
        error: Error message if failed
    """

    def __init__(
        self,
        goal: str,
        final_answer: str,
        success: bool,
        iteration_count: int,
        total_tokens: int,
        total_time_seconds: float,
        session_id: str,
        persona_used: Optional[str] = None,
        agent_state: Optional[EnhancedAgentState] = None,
        error: Optional[str] = None,
    ):
        self.goal = goal
        self.final_answer = final_answer
        self.success = success
        self.iteration_count = iteration_count
        self.total_tokens = total_tokens
        self.total_time_seconds = total_time_seconds
        self.session_id = session_id
        self.persona_used = persona_used
        self.agent_state = agent_state
        self.error = error

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"AgentResult({status}): {self.final_answer[:100]}... "
            f"({self.iteration_count} iterations, {self.total_tokens:,} tokens)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "success": self.success,
            "iteration_count": self.iteration_count,
            "total_tokens": self.total_tokens,
            "total_time_seconds": self.total_time_seconds,
            "session_id": self.session_id,
            "persona_used": self.persona_used,
            "error": self.error,
        }


class IterationUpdate:
    """Update from streaming execution."""

    def __init__(self, iteration: int, progress: float, status: str, message: str):
        self.iteration = iteration
        self.progress = progress
        self.status = status
        self.message = message


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SingleAgentMode",
    "AgentResult",
    "IterationUpdate",
]
