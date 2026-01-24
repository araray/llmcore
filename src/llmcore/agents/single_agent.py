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
- Goal classification with fast-path routing (G3)
- Capability pre-checking (G3)
- Circuit breaker protection (G3)

This is the recommended interface for most single-agent use cases.

References:
    - Technical Spec: Section 5.5 (Single Agent Mode)
    - Dossier: Step 2.9 (Single Agent Mode)
    - G3_COMPLETE_IMPLEMENTATION_PLAN.md
"""

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .cognitive import (
    CognitiveCycle,
    EnhancedAgentState,
)
from .cognitive.goal_classifier import (
    GoalClassification,
    GoalClassifier,
    GoalComplexity,
)
from .learning.fast_path import FastPathExecutor
from .persona import (
    AgentPersona,
    PersonaManager,
)
from .routing.capability_checker import (
    Capability,
    CapabilityChecker,
    CompatibilityResult,
)

if TYPE_CHECKING:
    from ..config.agents_config import AgentsConfig
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
        agents_config: Optional["AgentsConfig"] = None,
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
            agents_config: Optional agents configuration (uses defaults if not provided)
        """
        self.provider_manager = provider_manager
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager
        self.prompt_registry = prompt_registry
        self.tracer = tracer

        # Load agents config (G3)
        # If not provided, try to load from environment/defaults
        if agents_config is None:
            import os

            from ..config.agents_config import load_agents_config

            # Check if config file path is set via environment
            config_path = os.environ.get("LLMCORE_CONFIG_PATH")
            if config_path:
                from pathlib import Path

                self._agents_config = load_agents_config(config_path=Path(config_path))
                logger.debug(f"Loaded agents config from {config_path}")
            else:
                # Will use defaults + environment variable overrides
                self._agents_config = load_agents_config()
                logger.debug("Loaded agents config with defaults + env overrides")
        else:
            self._agents_config = agents_config

        # Initialize persona manager
        self.persona_manager = PersonaManager()

        # Initialize goal classifier (G3)
        self.goal_classifier = GoalClassifier()

        # Initialize fast-path executor (G3)
        # Use the default provider for fast-path calls
        default_provider = provider_manager.get_provider()
        self.fast_path_executor = FastPathExecutor(llm_provider=default_provider)

        # Initialize capability checker (G3 Phase 4)
        self.capability_checker = CapabilityChecker()

        # Initialize cognitive cycle
        self.cognitive_cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            tool_manager=tool_manager,
            prompt_registry=prompt_registry,
            tracer=tracer,
        )

        logger.info("SingleAgentMode initialized with G3 components")

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
        skip_validation: bool = False,
        skip_goal_classification: bool = False,
    ) -> "AgentResult":
        """
        Run an autonomous agent to complete a goal.

        This method implements G3 goal classification and fast-path routing:
        1. Classify the goal complexity (trivial, simple, moderate, complex)
        2. If trivial and fast-path enabled, bypass cognitive cycle for <5s response
        3. Otherwise, run full cognitive cycle with complexity-appropriate max_iterations

        Args:
            goal: The task goal/objective
            persona: Persona ID, AgentPersona object, or None for default
            context: Optional initial context
            max_iterations: Maximum iterations before stopping (may be overridden by classification)
            use_sandbox: Whether to use sandbox execution
            sandbox_type: Sandbox type (docker, vm) if use_sandbox=True
            provider_name: Optional LLM provider override
            model_name: Optional model override
            session_id: Optional session ID (generated if not provided)
            skip_validation: If True, auto-approve all actions (bypass HITL)
            skip_goal_classification: If True, skip goal classification (use provided max_iterations)

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

            start_time = datetime.utcnow()

            # =================================================================
            # G3 Phase 1: Goal Classification
            # =================================================================
            classification: Optional[GoalClassification] = None
            effective_max_iterations = max_iterations

            if not skip_goal_classification and self._agents_config.goals.classifier_enabled:
                try:
                    classification = self.goal_classifier.classify(goal)

                    logger.info(
                        f"Goal classified: complexity={classification.complexity.value}, "
                        f"intent={classification.intent.value}, "
                        f"confidence={classification.confidence:.2f}"
                    )

                    # Check for fast-path eligible goals (G3)
                    if (
                        classification.complexity == GoalComplexity.TRIVIAL
                        and self._agents_config.fast_path.enabled
                    ):
                        return await self._execute_fast_path(
                            goal=goal,
                            classification=classification,
                            context=context,
                            session_id=session_id,
                            persona=persona,
                            start_time=start_time,
                            span=span,
                        )

                    # Adjust max_iterations based on complexity (G3)
                    effective_max_iterations = self._get_complexity_iterations(
                        classification.complexity, max_iterations
                    )

                    logger.debug(
                        f"Adjusted max_iterations: {max_iterations} -> {effective_max_iterations} "
                        f"(complexity: {classification.complexity.value})"
                    )

                except Exception as e:
                    logger.warning(f"Goal classification failed, continuing with defaults: {e}")
                    classification = None

            # =================================================================
            # Normal Execution Path (non-trivial goals)
            # =================================================================

            # G3: Track if we need proactive activity execution
            use_activity_execution = False

            # =================================================================
            # G3 Phase 4: Capability Pre-Check
            # =================================================================
            if self._agents_config.capability_check.enabled:
                # Determine if goal requires tools
                requires_tools = True  # Default: assume tools needed
                if classification:
                    requires_tools = classification.requires_tools

                # Get the target model
                actual_provider = provider_name or self.provider_manager.get_default_provider_name()
                actual_model = model_name or self.provider_manager.get_default_model()

                # Check compatibility
                compat_result = self.capability_checker.check_compatibility(
                    model=actual_model,
                    requires_tools=requires_tools,
                    requires_vision=False,  # Could be determined from goal analysis
                )

                if not compat_result.compatible:
                    if self._agents_config.capability_check.strict_mode:
                        # Fail fast with helpful message
                        error_msg = self._format_capability_error(compat_result)
                        logger.error(f"Model capability check failed: {error_msg}")

                        end_time = datetime.utcnow()
                        duration = (end_time - start_time).total_seconds()

                        return AgentResult(
                            goal=goal,
                            final_answer=error_msg,
                            success=False,
                            iteration_count=0,
                            total_tokens=0,
                            total_time_seconds=duration,
                            session_id=session_id,
                            error=error_msg,
                            classification=classification,
                        )
                    else:
                        # Warn and enable proactive activity fallback
                        logger.warning(
                            f"Model capability warning for {actual_model}: "
                            f"{[str(i) for i in compat_result.issues]}. "
                            f"Automatically enabling activity-based execution."
                        )
                        use_activity_execution = True
                else:
                    logger.info(f"Capability check passed for {actual_provider}/{actual_model}")

            # CRITICAL FIX: Ensure tools are loaded before running
            # The ToolManager may have been initialized with empty tool lists.
            # If no tools are loaded, load the default built-in tools.
            if not self.tool_manager.get_tool_definitions():
                logger.info("No tools loaded - loading default tools for agent run")
                self.tool_manager.load_default_tools()
                loaded_tools = self.tool_manager.get_tool_names()
                logger.info(f"Loaded {len(loaded_tools)} default tools: {loaded_tools}")

            # Resolve persona
            agent_persona = self._resolve_persona(persona)

            # Create enhanced agent state
            agent_state = EnhancedAgentState(
                goal=goal, session_id=session_id, context=context or ""
            )

            # Store classification in working memory for cognitive cycle
            if classification:
                agent_state.set_working_memory("goal_classification", classification)
                agent_state.set_working_memory("goal_complexity", classification.complexity.value)

            # G3: Store activity execution flag if capability check triggered fallback
            if use_activity_execution:
                agent_state.set_working_memory("use_activity_execution", True)
                logger.info("Proactive activity execution enabled via capability pre-check")

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
                    f"max_iterations={effective_max_iterations}, "
                    f"complexity={classification.complexity.value if classification else 'unknown'}"
                )

                # Run cognitive cycle until complete
                final_answer = await self.cognitive_cycle.run_until_complete(
                    agent_state=agent_state,
                    session_id=session_id,
                    max_iterations=effective_max_iterations,
                    sandbox=sandbox,
                    provider_name=provider_name,
                    model_name=model_name,
                    skip_validation=skip_validation,
                    agents_config=self._agents_config,
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
                    classification=classification,
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
                            "agent.complexity": classification.complexity.value
                            if classification
                            else "unknown",
                            "agent.fast_path": False,
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
                    classification=classification,
                )

            finally:
                # Cleanup sandbox if created
                if sandbox and use_sandbox:
                    await self._cleanup_sandbox(sandbox)

    async def _execute_fast_path(
        self,
        goal: str,
        classification: GoalClassification,
        context: Optional[str],
        session_id: str,
        persona: Optional[str | AgentPersona],
        start_time: datetime,
        span: Optional[Any],
    ) -> "AgentResult":
        """
        Execute fast-path for trivial goals.

        This bypasses the full cognitive cycle for <5s response times.

        Args:
            goal: The user's goal
            classification: Goal classification result
            context: Optional context
            session_id: Session ID
            persona: Persona to use
            start_time: Execution start time
            span: Tracing span

        Returns:
            AgentResult from fast-path execution
        """
        from ..tracing import add_span_attributes

        logger.info(f"Fast-path execution for trivial goal: '{goal[:50]}...'")

        try:
            fast_path_result = await self.fast_path_executor.execute(
                goal=goal,
                classification=classification,
                context=context,
            )

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Resolve persona for result
            agent_persona = self._resolve_persona(persona)

            result = AgentResult(
                goal=goal,
                final_answer=fast_path_result.response,
                success=fast_path_result.success,
                iteration_count=fast_path_result.iterations,
                total_tokens=0,  # Fast-path token tracking TBD
                total_time_seconds=duration,
                session_id=session_id,
                persona_used=agent_persona.name if agent_persona else None,
                agent_state=None,
                classification=classification,
                fast_path=True,
            )

            if span:
                add_span_attributes(
                    span,
                    {
                        "agent.goal": goal[:100],
                        "agent.fast_path": True,
                        "agent.fast_path_strategy": fast_path_result.strategy.value,
                        "agent.fast_path_duration_ms": fast_path_result.duration_ms,
                        "agent.complexity": classification.complexity.value,
                    },
                )

            logger.info(
                f"Fast-path complete: success={result.success}, "
                f"duration={duration:.2f}s, "
                f"strategy={fast_path_result.strategy.value}"
            )

            return result

        except Exception as e:
            logger.error(f"Fast-path execution failed: {e}", exc_info=True)

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return AgentResult(
                goal=goal,
                final_answer=f"Fast-path failed: {str(e)}",
                success=False,
                iteration_count=0,
                total_tokens=0,
                total_time_seconds=duration,
                session_id=session_id,
                error=str(e),
                classification=classification,
                fast_path=True,
            )

    def _get_complexity_iterations(self, complexity: GoalComplexity, default: int) -> int:
        """
        Get max iterations based on goal complexity.

        Args:
            complexity: Goal complexity level
            default: Default max iterations

        Returns:
            Appropriate max_iterations for the complexity level
        """
        goals_config = self._agents_config.goals

        complexity_map = {
            GoalComplexity.TRIVIAL: goals_config.trivial_max_iterations,
            GoalComplexity.SIMPLE: goals_config.simple_max_iterations,
            GoalComplexity.MODERATE: goals_config.moderate_max_iterations,
            GoalComplexity.COMPLEX: goals_config.complex_max_iterations,
        }

        return complexity_map.get(complexity, default)

    def _format_capability_error(self, result: CompatibilityResult) -> str:
        """
        Format capability check failure into user-friendly message.

        Args:
            result: CompatibilityResult from capability checker

        Returns:
            Formatted error message with suggestions
        """
        issues = "\n".join(f"  - {issue}" for issue in result.issues)
        msg = f"Model '{result.model}' does not support required capabilities:\n{issues}"

        if self._agents_config.capability_check.suggest_alternatives and result.suggestions:
            suggestions = ", ".join(result.suggestions[:3])
            msg += f"\n\nSuggested alternatives: {suggestions}"

        return msg

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
        classification: Goal classification result (G3)
        fast_path: Whether fast-path was used (G3)
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
        classification: Optional[GoalClassification] = None,
        fast_path: bool = False,
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
        self.classification = classification  # G3
        self.fast_path = fast_path  # G3

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        fast_path_indicator = " [fast-path]" if self.fast_path else ""
        return (
            f"AgentResult({status}{fast_path_indicator}): {self.final_answer[:100]}... "
            f"({self.iteration_count} iterations, {self.total_tokens:,} tokens)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "success": self.success,
            "iteration_count": self.iteration_count,
            "total_tokens": self.total_tokens,
            "total_time_seconds": self.total_time_seconds,
            "session_id": self.session_id,
            "persona_used": self.persona_used,
            "error": self.error,
            "fast_path": self.fast_path,
        }

        # Add classification if available
        if self.classification:
            result["classification"] = {
                "complexity": self.classification.complexity.value,
                "intent": self.classification.intent.value,
                "confidence": self.classification.confidence,
            }

        return result


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
