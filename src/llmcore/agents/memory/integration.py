# src/llmcore/agents/memory/integration.py
"""
Memory Integration for Darwin Layer 2.

Enhanced memory integration that connects the cognitive cycle with
episodic and semantic memory systems. Provides:
- Automatic iteration recording
- Learning extraction and storage
- Context retrieval optimization
- Memory consolidation

References:
    - Technical Spec: Section 5.6 (Memory Integration)
    - Dossier: Step 2.10 (Memory Integration)
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...memory.manager import MemoryManager
    from ...models import Episode, EpisodeType
    from ...storage.manager import StorageManager
    from ..cognitive import CycleIteration, EnhancedAgentState

logger = logging.getLogger(__name__)


# =============================================================================
# MEMORY INTEGRATOR
# =============================================================================


class CognitiveMemoryIntegrator:
    """
    Integrates cognitive cycle with memory systems.

    The CognitiveMemoryIntegrator handles:
    - Recording iterations as episodes
    - Extracting learnings for semantic memory
    - Optimizing context retrieval
    - Consolidating memories

    Example:
        >>> integrator = CognitiveMemoryIntegrator(
        ...     memory_manager=memory_manager,
        ...     storage_manager=storage_manager
        ... )
        >>>
        >>> # Record iteration
        >>> await integrator.record_iteration(
        ...     iteration=iteration,
        ...     agent_state=state,
        ...     session_id="session-123"
        ... )
        >>>
        >>> # Retrieve relevant context
        >>> context = await integrator.retrieve_context(
        ...     goal="Calculate factorial",
        ...     current_step="Implement recursion",
        ...     session_id="session-123"
        ... )
    """

    def __init__(self, memory_manager: "MemoryManager", storage_manager: "StorageManager"):
        """
        Initialize the memory integrator.

        Args:
            memory_manager: Memory manager for semantic memory
            storage_manager: Storage manager for episodic memory
        """
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager

    async def record_iteration(
        self,
        iteration: "CycleIteration",
        agent_state: "EnhancedAgentState",
        session_id: str,
        extract_learnings: bool = True,
    ) -> None:
        """
        Record a cognitive iteration in memory.

        Stores the iteration as an episode and optionally extracts
        learnings for semantic memory.

        Args:
            iteration: The completed iteration
            agent_state: Current agent state
            session_id: Session identifier
            extract_learnings: Whether to extract learnings
        """
        from ...models import Episode, EpisodeType

        try:
            # Create episode from iteration
            episode = self._create_episode_from_iteration(
                iteration=iteration, agent_state=agent_state, session_id=session_id
            )

            # Store episode
            await self.storage_manager.store_episode(episode)

            logger.debug(
                f"Recorded iteration {iteration.iteration_number} "
                f"as episode for session {session_id}"
            )

            # Extract and store learnings if requested
            if extract_learnings and iteration.reflect_output:
                await self._extract_learnings(
                    iteration=iteration, agent_state=agent_state, session_id=session_id
                )

        except Exception as e:
            logger.error(f"Failed to record iteration: {e}")

    async def retrieve_context(
        self,
        goal: str,
        current_step: Optional[str] = None,
        session_id: Optional[str] = None,
        max_items: int = 5,
    ) -> List[str]:
        """
        Retrieve relevant context for the current cognitive state.

        Combines episodic memories (past iterations) with semantic
        memories (general knowledge) to provide optimal context.

        Args:
            goal: The current goal
            current_step: Current plan step
            session_id: Session ID for episodic context
            max_items: Maximum context items to return

        Returns:
            List of context strings
        """
        context_items = []

        # Build query
        query_parts = [goal]
        if current_step:
            query_parts.append(current_step)
        query = " ".join(query_parts)

        # Retrieve from semantic memory (general knowledge)
        try:
            semantic_items = await self.memory_manager.retrieve_relevant_context(
                query=query, limit=max_items // 2
            )
            context_items.extend([item.content for item in semantic_items])
        except Exception as e:
            logger.warning(f"Failed to retrieve semantic context: {e}")

        # Retrieve from episodic memory (past iterations)
        if session_id:
            try:
                episodic_items = await self._retrieve_episodic_context(
                    query=query, session_id=session_id, limit=max_items // 2
                )
                context_items.extend(episodic_items)
            except Exception as e:
                logger.warning(f"Failed to retrieve episodic context: {e}")

        return context_items[:max_items]

    async def consolidate_session_memory(
        self, session_id: str, agent_state: "EnhancedAgentState"
    ) -> None:
        """
        Consolidate memories from a completed session.

        Extracts key learnings from the entire session and stores
        them in semantic memory for future use.

        Args:
            session_id: Session identifier
            agent_state: Final agent state
        """
        try:
            # Get all insights from iterations
            all_insights = []
            for iteration in agent_state.iterations:
                if iteration.reflect_output and iteration.reflect_output.insights:
                    all_insights.extend(iteration.reflect_output.insights)

            if not all_insights:
                logger.debug("No insights to consolidate")
                return

            # Create consolidated memory
            consolidated = self._create_consolidated_memory(
                goal=agent_state.goal,
                insights=all_insights,
                success=agent_state.is_finished,
                iterations=agent_state.iteration_count,
            )

            # Store in semantic memory
            await self.memory_manager.store_memory(
                content=consolidated,
                metadata={
                    "type": "session_learning",
                    "session_id": session_id,
                    "goal": agent_state.goal,
                    "success": agent_state.is_finished,
                },
            )

            logger.info(f"Consolidated {len(all_insights)} insights from session {session_id}")

        except Exception as e:
            logger.error(f"Failed to consolidate session memory: {e}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _create_episode_from_iteration(
        self, iteration: "CycleIteration", agent_state: "EnhancedAgentState", session_id: str
    ) -> "Episode":
        """Create an Episode from a CycleIteration."""
        from ...models import Episode, EpisodeType

        # Build episode content for the data field
        content_parts = [f"Goal: {agent_state.goal}"]

        # Add action
        if iteration.think_output and iteration.think_output.proposed_action:
            action = iteration.think_output.proposed_action
            content_parts.append(f"Action: {action.name}({action.arguments})")

        # Add observation
        if iteration.observe_output:
            content_parts.append(f"Observation: {iteration.observe_output.observation}")

        # Add evaluation
        if iteration.reflect_output:
            content_parts.append(f"Evaluation: {iteration.reflect_output.evaluation}")

        # Create episode with correct field names:
        # - event_type (NOT episode_type) - this is the actual field name in Episode model
        # - data (NOT content) - Episodes use data: Dict[str, Any], not content: str
        return Episode(
            session_id=session_id,
            event_type=EpisodeType.TOOL_USE,  # Correct field name
            data={  # Episodes use 'data' dict, not 'content' string
                "content": "\n".join(content_parts),
                "iteration": iteration.iteration_number,
                "goal": agent_state.goal,
                "success": iteration.status.value == "completed",
            },
        )

    async def _extract_learnings(
        self, iteration: "CycleIteration", agent_state: "EnhancedAgentState", session_id: str
    ) -> None:
        """Extract learnings from iteration and store in semantic memory."""
        if not iteration.reflect_output or not iteration.reflect_output.insights:
            return

        for insight in iteration.reflect_output.insights:
            try:
                await self.memory_manager.store_memory(
                    content=insight,
                    metadata={
                        "type": "iteration_learning",
                        "session_id": session_id,
                        "iteration": iteration.iteration_number,
                        "goal": agent_state.goal,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to store insight: {e}")

    async def _retrieve_episodic_context(
        self, query: str, session_id: str, limit: int
    ) -> List[str]:
        """Retrieve episodic context from past iterations."""
        # Get recent episodes from this session
        try:
            episodes = await self.storage_manager.get_episodes(session_id=session_id, limit=limit)

            # Episode uses 'data' dict, not 'content' attribute
            # Extract content from the data dict, with fallback for empty/missing data
            return [ep.data.get("content", "") for ep in episodes if ep.data]
        except Exception as e:
            logger.warning(f"Failed to retrieve episodes: {e}")
            return []

    def _create_consolidated_memory(
        self, goal: str, insights: List[str], success: bool, iterations: int
    ) -> str:
        """Create consolidated memory from session insights."""
        status = "successfully" if success else "unsuccessfully"

        memory_parts = [
            f"Session Learning: {goal}",
            f"Result: Completed {status} in {iterations} iterations",
            "",
            "Key Insights:",
        ]

        # Deduplicate and format insights
        unique_insights = list(set(insights))
        for i, insight in enumerate(unique_insights[:10], 1):  # Top 10
            memory_parts.append(f"{i}. {insight}")

        return "\n".join(memory_parts)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["CognitiveMemoryIntegrator"]
