# src/llmcore/agents/cognitive/phases/perceive.py
"""
PERCEIVE Phase Implementation.

The PERCEIVE phase is the first phase of the enhanced cognitive cycle. It is
responsible for gathering and processing all inputs needed for decision-making:
- Retrieving relevant context from memory systems
- Capturing current environmental state
- Creating a snapshot of working memory
- Preparing inputs for subsequent phases

This phase does NOT involve LLM calls - it's purely data gathering and
preparation.

Context Retrieval Modes:
    1. **Legacy Mode**: Uses MemoryManager directly for backward compatibility.
    2. **Synthesis Mode**: Uses ContextSynthesizer for sophisticated multi-source
       context assembly with prioritization, scoring, and budget fitting.

Synthesis mode is recommended for autonomous operation where context quality
directly impacts agent performance.

References:
    - Technical Spec: Section 5.3.1 (PERCEIVE Phase)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
    - Dossier: Step 2.5 (Cognitive Phases - PERCEIVE)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..models import EnhancedAgentState, PerceiveInput, PerceiveOutput

if TYPE_CHECKING:
    from ....autonomous.goals import GoalManager
    from ....autonomous.skills import SkillLoader
    from ....context import ContextSynthesizer
    from ....memory.manager import MemoryManager
    from ...sandbox import SandboxProvider

logger = logging.getLogger(__name__)


# =============================================================================
# PERCEIVE PHASE FUNCTION
# =============================================================================


async def perceive_phase(
    agent_state: EnhancedAgentState,
    perceive_input: PerceiveInput,
    memory_manager: "MemoryManager",
    sandbox: Optional["SandboxProvider"] = None,
    context_synthesizer: Optional["ContextSynthesizer"] = None,
    tracer: Any | None = None,
) -> PerceiveOutput:
    """
    Execute the PERCEIVE phase of the cognitive cycle.

    This phase gathers all necessary inputs for decision-making:
    1. Retrieves relevant context from semantic and episodic memory
    2. Captures current sandbox/environmental state
    3. Creates snapshot of working memory
    4. Prepares data for subsequent phases

    Context Retrieval Strategy:
        - If ``context_synthesizer`` is provided, uses the sophisticated
          multi-source synthesis approach with prioritization and scoring.
        - Otherwise, falls back to direct ``memory_manager`` retrieval
          for backward compatibility.

    Args:
        agent_state: Current enhanced agent state.
        perceive_input: Input configuration for perception.
        memory_manager: Memory manager for context retrieval (legacy mode).
        sandbox: Optional active sandbox provider.
        context_synthesizer: Optional ContextSynthesizer for advanced
            multi-source context assembly. When provided, memory_manager
            is used only as a fallback.
        tracer: Optional OpenTelemetry tracer.

    Returns:
        PerceiveOutput with gathered context and state.

    Example:
        >>> # Legacy mode (backward compatible)
        >>> perceive_input = PerceiveInput(
        ...     goal="Calculate factorial of 10",
        ...     force_refresh=False
        ... )
        >>> output = await perceive_phase(
        ...     agent_state=state,
        ...     perceive_input=perceive_input,
        ...     memory_manager=memory_manager
        ... )
        >>>
        >>> # Synthesis mode (recommended for autonomous operation)
        >>> from llmcore.context import ContextSynthesizer
        >>> synthesizer = ContextSynthesizer(max_tokens=100_000)
        >>> # ... register sources ...
        >>> output = await perceive_phase(
        ...     agent_state=state,
        ...     perceive_input=perceive_input,
        ...     memory_manager=memory_manager,
        ...     context_synthesizer=synthesizer
        ... )
    """
    from ....tracing import add_span_attributes, create_span

    with create_span(tracer, "cognitive.perceive") as span:
        logger.debug("Starting PERCEIVE phase")

        # 1. Retrieve context - use synthesizer if available, else legacy
        if context_synthesizer is not None:
            retrieved_context = await _retrieve_context_with_synthesizer(
                goal=perceive_input.goal,
                context_query=perceive_input.context_query,
                synthesizer=context_synthesizer,
                force_refresh=perceive_input.force_refresh,
                agent_state=agent_state,
            )
            synthesis_mode = True
        else:
            retrieved_context = await _retrieve_context(
                goal=perceive_input.goal,
                context_query=perceive_input.context_query,
                memory_manager=memory_manager,
                force_refresh=perceive_input.force_refresh,
            )
            synthesis_mode = False

        # 2. Capture working memory snapshot
        working_memory_snapshot = dict(agent_state.working_memory)

        # 3. Capture environmental state
        environmental_state = await _capture_environmental_state(
            agent_state=agent_state, sandbox=sandbox
        )

        # 4. Create output
        output = PerceiveOutput(
            retrieved_context=retrieved_context,
            working_memory_snapshot=working_memory_snapshot,
            environmental_state=environmental_state,
            perceived_at=datetime.utcnow(),
        )

        # 5. Add tracing attributes
        if span:
            add_span_attributes(
                span,
                {
                    "perceive.context_items": len(retrieved_context),
                    "perceive.working_memory_size": len(working_memory_snapshot),
                    "perceive.has_sandbox": sandbox is not None,
                    "perceive.synthesis_mode": synthesis_mode,
                },
            )

        logger.debug(
            f"PERCEIVE phase complete: {len(retrieved_context)} context items, "
            f"{len(working_memory_snapshot)} working memory entries "
            f"(synthesis_mode={synthesis_mode})"
        )

        return output


# =============================================================================
# CONTEXT RETRIEVAL HELPERS
# =============================================================================


async def _retrieve_context_with_synthesizer(
    goal: str,
    context_query: str | None,
    synthesizer: "ContextSynthesizer",
    force_refresh: bool,
    agent_state: EnhancedAgentState,
) -> list[str]:
    """
    Retrieve context using the ContextSynthesizer.

    Uses the synthesizer to gather context from multiple sources in parallel,
    score and prioritize chunks, and fit them into the token budget.

    Args:
        goal: The current goal.
        context_query: Optional specific query.
        synthesizer: The ContextSynthesizer instance.
        force_refresh: Whether to bypass caches (currently unused, reserved).
        agent_state: Current agent state for additional context.

    Returns:
        List of context strings, one per synthesizer section.
    """
    # Build a lightweight task object for the synthesizer
    # The synthesizer sources expect a task with a `description` attribute
    task = _TaskProxy(description=context_query or goal, goal=goal)

    try:
        # Synthesize context from all registered sources
        synthesized = await synthesizer.synthesize(current_task=task)

        if not synthesized.content.strip():
            logger.debug("Synthesizer returned empty context")
            return []

        # Parse synthesized content into separate context strings
        # The synthesizer formats as "## SOURCE_NAME\n\ncontent\n\n---\n\n"
        context_strings = _parse_synthesized_content(synthesized.content)

        logger.debug(
            f"Synthesized context: {synthesized.total_tokens} tokens, "
            f"{len(synthesized.sources_included)} sources "
            f"({', '.join(synthesized.sources_included)}), "
            f"compression={synthesized.compression_applied}, "
            f"time={synthesized.synthesis_time_ms:.1f}ms"
        )

        return context_strings

    except Exception as e:
        logger.warning(f"Context synthesis failed, returning empty context: {e}")
        return []


async def _retrieve_context(
    goal: str, context_query: str | None, memory_manager: "MemoryManager", force_refresh: bool
) -> list[str]:
    """
    Retrieve relevant context from memory systems (legacy mode).

    Args:
        goal: The current goal.
        context_query: Optional specific query.
        memory_manager: Memory manager instance.
        force_refresh: Whether to force refresh.

    Returns:
        List of context strings.
    """
    query = context_query or goal

    try:
        # Retrieve from semantic memory (RAG)
        context_items = await memory_manager.retrieve_relevant_context(
            goal=query,  # Parameter is named 'goal', not 'query'
        )

        # Convert to strings
        context_strings = []
        for item in context_items:
            # Extract content based on item type
            if hasattr(item, "content"):
                context_strings.append(item.content)
            elif hasattr(item, "text"):
                context_strings.append(item.text)
            elif isinstance(item, str):
                context_strings.append(item)
            else:
                context_strings.append(str(item))

        logger.debug(f"Retrieved {len(context_strings)} context items from memory")
        return context_strings

    except Exception as e:
        logger.warning(f"Failed to retrieve context from memory: {e}")
        return []


def _parse_synthesized_content(content: str) -> list[str]:
    """
    Parse synthesized content into separate context strings.

    The ContextSynthesizer formats content as:
        ## SOURCE_NAME

        content...

        ---

        ## ANOTHER_SOURCE
        ...

    This function splits on the section dividers and returns each
    section as a separate string for compatibility with PerceiveOutput.

    Args:
        content: The assembled content from ContextSynthesizer.

    Returns:
        List of context strings, one per section.
    """
    if not content.strip():
        return []

    # Split on the section divider
    sections = content.split("\n\n---\n\n")

    # Clean up each section
    context_strings = []
    for section in sections:
        stripped = section.strip()
        if stripped:
            context_strings.append(stripped)

    return context_strings


async def _capture_environmental_state(
    agent_state: EnhancedAgentState, sandbox: Optional["SandboxProvider"]
) -> dict[str, Any]:
    """
    Capture current environmental state.

    Args:
        agent_state: Current agent state.
        sandbox: Optional active sandbox.

    Returns:
        Dictionary of environmental state.
    """
    env_state = {
        "has_sandbox": sandbox is not None,
        "iteration_count": agent_state.iteration_count,
        "total_tool_calls": agent_state.total_tool_calls,
        "plan_version": agent_state.plan_version,
    }

    # Add sandbox info if available
    if sandbox:
        try:
            sandbox_info = sandbox.get_info()
            env_state.update(
                {
                    "sandbox_provider": sandbox_info.get("provider"),
                    "sandbox_status": sandbox_info.get("status"),
                    "sandbox_access_level": sandbox_info.get("access_level"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get sandbox info: {e}")

    return env_state


# =============================================================================
# HELPER CLASSES
# =============================================================================


class _TaskProxy:
    """
    Lightweight proxy object representing the current task.

    Used to pass task information to context sources without
    requiring a full Goal object (which may involve circular imports
    or additional dependencies).

    Attributes:
        description: Task description for keyword matching and retrieval.
        goal: The original goal string.
    """

    __slots__ = ("description", "goal")

    def __init__(self, description: str, goal: str) -> None:
        self.description = description
        self.goal = goal

    def __str__(self) -> str:
        return self.description


# =============================================================================
# SYNTHESIZER FACTORY
# =============================================================================


def create_default_synthesizer(
    goal_manager: Optional["GoalManager"] = None,
    skill_loader: Optional["SkillLoader"] = None,
    retrieval_fn: Any | None = None,
    max_tokens: int = 100_000,
    compression_threshold: float = 0.75,
) -> "ContextSynthesizer":
    """
    Create a ContextSynthesizer with default source configuration.

    This factory function creates a synthesizer pre-configured with all
    five context source tiers at their canonical priorities:

        1. Goals (100): Current goals, progress, learned strategies.
        2. Recent (80): Recent conversation turns and tool executions.
        3. Semantic (60): RAG-retrieved chunks.
        4. Skills (50): On-demand SKILL.md files.
        5. Episodic (40): Past experiences and failure patterns.

    Sources are only registered if their dependencies are provided.

    Args:
        goal_manager: GoalManager for goal context (optional).
        skill_loader: SkillLoader for skill context (optional).
        retrieval_fn: Async retrieval function for semantic context (optional).
            Signature: ``async def fn(query: str, top_k: int) -> List[Dict]``
        max_tokens: Maximum tokens for the context budget.
        compression_threshold: Trigger compression above this utilization.

    Returns:
        Configured ContextSynthesizer ready for use.

    Example:
        >>> from llmcore.autonomous import GoalManager, SkillLoader
        >>> from llmcore.agents.cognitive.phases.perceive import (
        ...     create_default_synthesizer
        ... )
        >>>
        >>> goal_mgr = GoalManager(...)
        >>> skill_ldr = SkillLoader()
        >>> synthesizer = create_default_synthesizer(
        ...     goal_manager=goal_mgr,
        ...     skill_loader=skill_ldr,
        ...     max_tokens=100_000
        ... )
    """
    from ....context import ContextSynthesizer
    from ....context.sources import (
        EpisodicContextSource,
        GoalContextSource,
        RecentContextSource,
        SemanticContextSource,
        SkillContextSource,
    )

    synthesizer = ContextSynthesizer(
        max_tokens=max_tokens,
        compression_threshold=compression_threshold,
    )

    # Register sources in priority order (highest first)

    # 1. Goals (priority 100) - always relevant
    if goal_manager is not None:
        synthesizer.add_source(
            "goals",
            GoalContextSource(goal_manager),
            priority=100,
        )
        logger.debug("Registered GoalContextSource with priority 100")

    # 2. Recent (priority 80) - conversation continuity
    recent_source = RecentContextSource(max_turns=20)
    synthesizer.add_source("recent", recent_source, priority=80)
    logger.debug("Registered RecentContextSource with priority 80")

    # 3. Semantic (priority 60) - RAG retrieval
    if retrieval_fn is not None:
        synthesizer.add_source(
            "semantic",
            SemanticContextSource(retrieval_fn=retrieval_fn),
            priority=60,
        )
        logger.debug("Registered SemanticContextSource with priority 60")

    # 4. Skills (priority 50) - on-demand knowledge
    if skill_loader is not None:
        synthesizer.add_source(
            "skills",
            SkillContextSource(skill_loader, default_priority=50),
            priority=50,
        )
        logger.debug("Registered SkillContextSource with priority 50")

    # 5. Episodic (priority 40) - historical patterns
    episodic_source = EpisodicContextSource(max_episodes=100)
    synthesizer.add_source("episodic", episodic_source, priority=40)
    logger.debug("Registered EpisodicContextSource with priority 40")

    return synthesizer


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "perceive_phase",
    "create_default_synthesizer",
]
