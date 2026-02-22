# src/llmcore/agents/cognitive/phases/update.py
"""
UPDATE Phase Implementation.

The UPDATE phase is the final phase of the cognitive cycle. It applies
updates to the agent state and memory based on the reflection output:
- Updates agent state (plan, progress, confidence)
- Records episodes in episodic memory
- Updates working memory
- Marks plan steps as complete
- Determines if iteration should continue

This phase ensures that learnings from each iteration are captured and
applied to improve future performance.

References:
    - Technical Spec: Section 5.3.8 (UPDATE Phase)
    - Dossier: Step 2.7 (Cognitive Phases - UPDATE)
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..models import EnhancedAgentState, UpdateInput, UpdateOutput

if TYPE_CHECKING:
    from ....storage.manager import StorageManager

logger = logging.getLogger(__name__)


# =============================================================================
# UPDATE PHASE FUNCTION
# =============================================================================


async def update_phase(
    agent_state: EnhancedAgentState,
    update_input: UpdateInput,
    storage_manager: Optional["StorageManager"] = None,
    session_id: str | None = None,
    tracer: Any | None = None,
) -> UpdateOutput:
    """
    Execute the UPDATE phase of the cognitive cycle.

    Updates state and memory by:
    1. Applying plan updates from reflection
    2. Marking completed steps
    3. Updating progress estimate
    4. Recording episodes in memory
    5. Updating working memory
    6. Determining continuation

    Args:
        agent_state: Current enhanced agent state
        update_input: Input with reflection output and state
        storage_manager: Optional storage manager for episodic memory
        session_id: Optional session ID for memory
        tracer: Optional OpenTelemetry tracer

    Returns:
        UpdateOutput with applied changes

    Example:
        >>> update_input = UpdateInput(
        ...     reflection=reflect_output,
        ...     current_state=agent_state
        ... )
        >>>
        >>> output = await update_phase(
        ...     agent_state=agent_state,
        ...     update_input=update_input,
        ...     storage_manager=storage_manager,
        ...     session_id="session-123"
        ... )
        >>>
        >>> print(f"Continue: {output.should_continue}")
    """
    from ....models import Episode, EpisodeType
    from ....tracing import add_span_attributes, create_span

    with create_span(tracer, "cognitive.update") as span:
        logger.debug("Starting UPDATE phase")

        reflection = update_input.reflection
        state_updates = {}
        memory_updates = []
        working_memory_updates = {}

        # 1. Update plan if needed
        if reflection.plan_needs_update and reflection.updated_plan:
            agent_state.update_plan(
                new_plan=reflection.updated_plan, reasoning="Plan updated based on reflection"
            )
            state_updates["plan_updated"] = True
            state_updates["new_plan_version"] = agent_state.plan_version

            logger.info(f"Plan updated to version {agent_state.plan_version}")

        # 2. Mark step as complete if needed
        if reflection.step_completed:
            if agent_state.current_plan_step_index < len(agent_state.plan_steps_status):
                agent_state.plan_steps_status[agent_state.current_plan_step_index] = "completed"

                # Move to next step
                if agent_state.current_plan_step_index < len(agent_state.plan) - 1:
                    agent_state.current_plan_step_index += 1
                    state_updates["current_step_index"] = agent_state.current_plan_step_index
                    logger.info(f"Advanced to step {agent_state.current_plan_step_index + 1}")
                else:
                    logger.info("All plan steps completed")

        # 3. Update progress
        agent_state.progress_estimate = reflection.progress_estimate
        state_updates["progress"] = reflection.progress_estimate

        # 4. Store insights in working memory
        if reflection.insights:
            existing_insights = agent_state.get_working_memory("insights", [])
            existing_insights.extend(reflection.insights)
            agent_state.set_working_memory("insights", existing_insights)
            working_memory_updates["insights_count"] = len(existing_insights)

        # 5. Store next focus in working memory
        if reflection.next_focus:
            agent_state.set_working_memory("next_focus", reflection.next_focus)
            working_memory_updates["next_focus"] = reflection.next_focus

        # 6. Record episode in episodic memory
        if storage_manager and session_id:
            try:
                # Create episode from current iteration
                if agent_state.current_iteration:
                    episode_content = _create_episode_content(
                        iteration=agent_state.current_iteration,
                        reflection=reflection,
                        agent_state=agent_state,
                    )

                    episode = Episode(
                        session_id=session_id,
                        episode_type=EpisodeType.TOOL_USE,
                        content=episode_content,
                        metadata={
                            "iteration": agent_state.iteration_count + 1,
                            "progress": reflection.progress_estimate,
                            "insights_count": len(reflection.insights),
                        },
                    )

                    # Store episode (method is add_episode, not store_episode)
                    await storage_manager.add_episode(episode)
                    memory_updates.append(
                        {
                            "type": "episode",
                            "episode_type": EpisodeType.TOOL_USE.value,
                            "iteration": agent_state.iteration_count + 1,
                        }
                    )

                    logger.debug("Episode recorded in episodic memory")

            except Exception as e:
                logger.warning(f"Failed to record episode: {e}")

        # 7. Determine if should continue
        should_continue = _should_continue(agent_state=agent_state, reflection=reflection)

        # 8. Create output
        output = UpdateOutput(
            state_updates=state_updates,
            memory_updates=memory_updates,
            working_memory_updates=working_memory_updates,
            should_continue=should_continue,
        )

        # 9. Add tracing
        if span:
            add_span_attributes(
                span,
                {
                    "update.plan_updated": reflection.plan_needs_update,
                    "update.step_completed": reflection.step_completed,
                    "update.progress": reflection.progress_estimate,
                    "update.should_continue": should_continue,
                    "update.memory_updates": len(memory_updates),
                },
            )

        logger.info(
            f"UPDATE phase complete: progress={reflection.progress_estimate:.1%}, "
            f"continue={should_continue}"
        )

        return output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _create_episode_content(
    iteration: Any,  # CycleIteration
    reflection: Any,  # ReflectOutput
    agent_state: EnhancedAgentState,
) -> str:
    """
    Create episode content from iteration and reflection.

    Args:
        iteration: The cycle iteration
        reflection: Reflection output
        agent_state: Current agent state

    Returns:
        Formatted episode content
    """
    content_parts = []

    # Goal context
    content_parts.append(f"Goal: {agent_state.goal}")

    # Action taken
    if iteration.think_output and iteration.think_output.proposed_action:
        action = iteration.think_output.proposed_action
        content_parts.append(f"Action: {action.name}({action.arguments})")

    # Result
    if iteration.observe_output:
        content_parts.append(f"Observation: {iteration.observe_output.observation}")

    # Evaluation
    content_parts.append(f"Evaluation: {reflection.evaluation}")

    # Insights
    if reflection.insights:
        content_parts.append("Insights:")
        for insight in reflection.insights:
            content_parts.append(f"  - {insight}")

    return "\n".join(content_parts)


def _should_continue(
    agent_state: EnhancedAgentState,
    reflection: Any,  # ReflectOutput
) -> bool:
    """
    Determine if the cognitive loop should continue.

    Args:
        agent_state: Current agent state
        reflection: Reflection output

    Returns:
        True if should continue, False otherwise
    """
    # Don't continue if finished
    if agent_state.is_finished:
        return False

    # Don't continue if waiting for human approval
    if agent_state.awaiting_human_approval:
        return False

    # Don't continue if progress is at 100%
    if reflection.progress_estimate >= 1.0:
        agent_state.is_finished = True
        return False

    # Don't continue if all steps are completed
    if all(status == "completed" for status in agent_state.plan_steps_status):
        agent_state.is_finished = True
        return False

    # Otherwise, continue
    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["update_phase"]
