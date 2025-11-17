# src/llmcore/agents/cognitive_cycle.py
"""
Core cognitive cycle implementation for LLMCore agents.

This module contains the standalone functions that execute the primary steps
of an agent's operation: Plan, Think, Act, Observe, and Reflect. It also
includes the logic for handling Human-in-the-Loop (HITL) workflows, such as
pausing for and resuming from human approval.

By separating this logic from the AgentManager, we make the cognitive architecture
of the agent more explicit and easier to modify or extend.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..memory.manager import MemoryManager
from ..models import (AgentState, AgentTask, Episode, EpisodeType, Message,
                      Role, ToolCall, ToolResult)
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from . import prompt_utils
from .tools import ToolManager

logger = logging.getLogger(__name__)

async def plan_step(
    agent_state: AgentState,
    session_id: str,
    provider_manager: ProviderManager,
    tracer: Optional[Any] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Execute the PLANNING step: decompose the goal into actionable sub-tasks.

    Args:
        agent_state: Current agent state with goal to be planned.
        session_id: Session ID for context retrieval.
        provider_manager: The manager for LLM provider interactions.
        tracer: The OpenTelemetry tracer instance.
        provider_name: Optional provider override.
        model_name: Optional model override.
    """
    from ..tracing import create_span, add_span_attributes, record_span_exception

    with create_span(tracer, "agent.plan_step") as span:
        try:
            logger.debug(f"Executing planning step for goal: {agent_state.goal}")

            planning_prompt = prompt_utils.load_planning_prompt_template()
            formatted_prompt = planning_prompt.format(goal=agent_state.goal)

            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are a strategic planning agent. Create a clear, actionable plan to achieve the given goal.",
                    session_id=session_id
                ),
                Message(
                    role=Role.USER,
                    content=formatted_prompt,
                    session_id=session_id
                )
            ]

            logger.debug("Calling LLM for strategic planning...")
            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False
            )

            response_content = response['choices'][0]['message']['content'] or ""
            plan_steps = prompt_utils.parse_plan_from_response(response_content)

            if plan_steps:
                agent_state.plan = plan_steps
                agent_state.current_plan_step_index = 0
                agent_state.plan_steps_status = ['pending'] * len(plan_steps)
                logger.info(f"Successfully generated plan with {len(plan_steps)} steps")
            else:
                agent_state.plan = [
                    "Analyze the goal and determine necessary actions",
                    "Execute the required actions step by step",
                    "Use the finish tool with the final result"
                ]
                agent_state.current_plan_step_index = 0
                agent_state.plan_steps_status = ['pending'] * len(agent_state.plan)
                logger.warning("Failed to parse plan from LLM response, using fallback plan")

            if span:
                add_span_attributes(span, {
                    "plan.provider": provider.get_name(),
                    "plan.model": target_model,
                    "plan.prompt_length": len(formatted_prompt),
                    "plan.response_length": len(response_content),
                    "plan.steps_generated": len(agent_state.plan),
                    "plan.parsing_success": len(plan_steps) > 0
                })

        except Exception as e:
            logger.error(f"Error in planning step: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)
            agent_state.plan = [
                "Proceed with the goal using available tools",
                "Use the finish tool when ready with results"
            ]
            agent_state.current_plan_step_index = 0
            agent_state.plan_steps_status = ['pending'] * len(agent_state.plan)

async def reflect_step(
    agent_state: AgentState,
    last_tool_call: ToolCall,
    last_observation: ToolResult,
    session_id: str,
    provider_manager: ProviderManager,
    tracer: Optional[Any] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Execute the REFLECTION step: critically evaluate progress and update plan if needed.

    Args:
        agent_state: Agent state to potentially update.
        last_tool_call: The action that was just taken.
        last_observation: The result of the last action.
        session_id: Session ID for context.
        provider_manager: The manager for LLM provider interactions.
        tracer: The OpenTelemetry tracer instance.
        provider_name: Optional provider override.
        model_name: Optional model override.
    """
    from ..tracing import create_span, add_span_attributes, record_span_exception

    with create_span(tracer, "agent.reflect_step") as span:
        try:
            logger.debug(f"Executing reflection step for tool: {last_tool_call.name}")

            reflection_prompt = prompt_utils.load_reflection_prompt_template()
            plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(agent_state.plan)])
            formatted_prompt = reflection_prompt.format(
                goal=agent_state.goal,
                plan=plan_text,
                last_action_name=last_tool_call.name,
                last_action_arguments=json.dumps(last_tool_call.arguments),
                last_observation=last_observation.content
            )

            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are a critical evaluation agent. Assess progress and determine if the plan needs updating.",
                    session_id=session_id
                ),
                Message(
                    role=Role.USER,
                    content=formatted_prompt,
                    session_id=session_id
                )
            ]

            logger.debug("Calling LLM for critical reflection...")
            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False
            )

            response_content = response['choices'][0]['message']['content'] or ""
            reflection_result = prompt_utils.parse_reflection_response(response_content)

            if reflection_result:
                evaluation = reflection_result.get('evaluation', '')
                plan_step_completed = reflection_result.get('plan_step_completed', False)
                updated_plan = reflection_result.get('updated_plan')

                agent_state.scratchpad = f"Reflection: {evaluation[:300]}..."

                if plan_step_completed and agent_state.current_plan_step_index < len(agent_state.plan_steps_status):
                    agent_state.plan_steps_status[agent_state.current_plan_step_index] = 'completed'
                    agent_state.current_plan_step_index = min(
                        agent_state.current_plan_step_index + 1,
                        len(agent_state.plan) - 1
                    )
                    logger.debug(f"Marked plan step {agent_state.current_plan_step_index} as completed")

                if updated_plan and isinstance(updated_plan, list):
                    agent_state.plan = updated_plan
                    agent_state.plan_steps_status = ['pending'] * len(updated_plan)
                    agent_state.current_plan_step_index = 0
                    logger.info(f"Plan updated based on reflection with {len(updated_plan)} new steps")

                if span:
                    add_span_attributes(span, {
                        "reflect.provider": provider.get_name(),
                        "reflect.model": target_model,
                        "reflect.evaluation_length": len(evaluation),
                        "reflect.step_completed": plan_step_completed,
                        "reflect.plan_updated": updated_plan is not None,
                        "reflect.parsing_success": True
                    })
            else:
                logger.warning("Failed to parse reflection response from LLM")
                if span:
                    add_span_attributes(span, {"reflect.parsing_success": False})

        except Exception as e:
            logger.error(f"Error in reflection step: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

async def think_step(
    agent_state: AgentState,
    session_id: str,
    memory_manager: MemoryManager,
    provider_manager: ProviderManager,
    tool_manager: ToolManager,
    tracer: Optional[Any] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[ToolCall]]:
    """
    Execute the THINK step: retrieve context and call LLM for reasoning.

    Args:
        agent_state: Current agent state with goal and history.
        session_id: Session ID for context retrieval.
        memory_manager: The manager for context retrieval.
        provider_manager: The manager for LLM provider interactions.
        tool_manager: The manager for available tools.
        tracer: The OpenTelemetry tracer instance.
        provider_name: Optional provider override.
        model_name: Optional model override.

    Returns:
        Tuple of (thought, tool_call) or (None, None) if parsing fails.
    """
    from ..tracing import create_span, add_span_attributes, record_span_exception

    with create_span(tracer, "agent.think_step") as span:
        try:
            context_items = await memory_manager.retrieve_relevant_context(agent_state.goal)
            prompt = prompt_utils.build_enhanced_agent_prompt(
                agent_state, context_items, tool_manager.get_tool_definitions()
            )

            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are an autonomous AI agent with strategic planning capabilities. Follow your plan while using the ReAct format: provide a Thought explaining your reasoning, then specify an Action (tool call) to take.",
                    session_id=session_id
                ),
                Message(
                    role=Role.USER,
                    content=prompt,
                    session_id=session_id
                )
            ]

            logger.debug("Calling LLM for enhanced agent reasoning with plan context...")
            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False,
                tools=tool_manager.get_tool_definitions()
            )

            response_content = response['choices'][0]['message']['content'] or ""
            result = prompt_utils.parse_agent_response(
                response_content, response, tool_manager.get_tool_names()
            )

            if span:
                add_span_attributes(span, {
                    "think.provider": provider.get_name(),
                    "think.model": target_model,
                    "think.context_items": len(context_items),
                    "think.prompt_length": len(prompt),
                    "think.response_length": len(response_content),
                    "think.parsing_success": result[0] is not None and result[1] is not None,
                    "think.plan_context_included": len(agent_state.plan) > 0
                })

            return result

        except Exception as e:
            logger.error(f"Error in enhanced think step: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)
            return None, None

async def act_step(
    tool_call: ToolCall,
    session_id: str,
    task: AgentTask,
    tool_manager: ToolManager,
    tracer: Optional[Any] = None,
    db_session: Optional[AsyncSession] = None
) -> ToolResult:
    """
    Execute the ACT step: run the requested tool, with HITL check.

    Args:
        tool_call: The tool call to execute.
        session_id: Session ID for tools that need context.
        task: The current AgentTask (needed for HITL workflow).
        tool_manager: The manager for tool execution.
        tracer: The OpenTelemetry tracer instance.
        db_session: Database session for updating task state.

    Returns:
        ToolResult containing the execution result.
    """
    from ..tracing import create_span, add_span_attributes, record_span_exception

    with create_span(tracer, "agent.act_step") as span:
        logger.debug(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")

        if span:
            add_span_attributes(span, {
                "act.tool_name": tool_call.name,
                "act.tool_id": tool_call.id,
                "act.arguments_count": len(tool_call.arguments),
                "act.session_id": session_id,
                "act.hitl_check": True
            })

        try:
            if tool_call.name == "human_approval":
                logger.info(f"Human approval requested for task {task.task_id}")
                approval_prompt = tool_call.arguments.get("prompt", "Approval requested for pending action")
                pending_action = tool_call.arguments.get("pending_action", {})

                await update_task_for_approval(task, approval_prompt, pending_action, db_session)

                if span:
                    add_span_attributes(span, {
                        "act.hitl_triggered": True,
                        "act.approval_prompt": approval_prompt[:100],
                        "act.pending_action": pending_action.get("name", "unknown")
                    })

                return ToolResult(tool_call_id=tool_call.id, content="PAUSED_FOR_APPROVAL")

            result = await tool_manager.execute_tool(tool_call, session_id)

            # Record tool execution metrics
            try:
                status = "success" if not result.content.startswith("ERROR:") else "error"
                record_tool_execution(
                    tenant_id=tenant_id,
                    tool_name=tool_call.name,
                    status=status
                )
            except Exception as e:
                logger.debug(f"Failed to record tool execution metrics: {e}")

            if span:
                add_span_attributes(span, {
                    "act.result_length": len(result.content),
                    "act.success": not result.content.startswith("ERROR:"),
                    "act.hitl_triggered": False
                })

            return result

        except Exception as e:
            logger.error(f"Error in act step: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)
            return ToolResult(tool_call_id=tool_call.id, content=f"ERROR: {str(e)}")

async def observe_step(
    agent_state: AgentState,
    thought: str,
    tool_call: ToolCall,
    observation: str,
    session_id: str,
    storage_manager: StorageManager,
    tracer: Optional[Any] = None
) -> None:
    """
    Execute the OBSERVE step: update state and log experience.

    Args:
        agent_state: Agent state to update.
        thought: The agent's reasoning.
        tool_call: The action taken.
        observation: The result of the action.
        session_id: Session ID for episodic logging.
        storage_manager: The manager for episodic memory logging.
        tracer: The OpenTelemetry tracer instance.
    """
    from ..tracing import create_span, add_span_attributes, record_span_exception

    with create_span(tracer, "agent.observe_step") as span:
        try:
            agent_state.history_of_thoughts.append(thought)
            agent_state.observations[tool_call.id] = {
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "result": observation,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            current_step = ""
            if agent_state.plan and agent_state.current_plan_step_index < len(agent_state.plan):
                current_step = f" (Step {agent_state.current_plan_step_index + 1}: {agent_state.plan[agent_state.current_plan_step_index]})"

            agent_state.scratchpad = f"Last thought: {thought}\nLast action: {tool_call.name}{current_step}\nLast observation: {observation[:200]}..."

            episode = Episode(
                session_id=session_id,
                event_type=EpisodeType.AGENT_REFLECTION,
                data={
                    "thought": thought,
                    "action": {"tool_name": tool_call.name, "tool_call_id": tool_call.id, "arguments": tool_call.arguments},
                    "observation": observation,
                    "goal": agent_state.goal,
                    "current_plan_step": agent_state.current_plan_step_index + 1 if agent_state.plan else 0,
                    "plan": agent_state.plan,
                    "iteration_summary": f"Agent reasoned, used {tool_call.name}, observed result, executing plan step {agent_state.current_plan_step_index + 1}"
                }
            )

            try:
                await storage_manager.add_episode(episode)
                logger.debug(f"Logged enhanced T-A-O cycle as episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"Failed to log episode: {e}")

            if span:
                add_span_attributes(span, {
                    "observe.thought_length": len(thought),
                    "observe.observation_length": len(observation),
                    "observe.total_thoughts": len(agent_state.history_of_thoughts),
                    "observe.total_observations": len(agent_state.observations),
                    "observe.episode_logged": True,
                    "observe.plan_step": agent_state.current_plan_step_index + 1
                })

        except Exception as e:
            logger.error(f"Error in observe step: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

# --- HITL Workflow Functions ---

def check_if_resuming_task(task: AgentTask) -> bool:
    """Check if this task is resuming from a HITL workflow."""
    return task.pending_action_data is not None

async def handle_task_resumption(
    task: AgentTask,
    session_id: str,
    tool_manager: ToolManager,
    storage_manager: StorageManager,
    tracer: Optional[Any] = None,
    db_session: Optional[AsyncSession] = None
) -> Optional[str]:
    """Handle the resumption of a task from HITL workflow."""
    try:
        pending_data = task.pending_action_data
        if not pending_data:
            return None

        if "rejection_reason" in pending_data:
            rejection_reason = pending_data["rejection_reason"]
            logger.info(f"Task {task.task_id} was rejected: {rejection_reason}")
            await clear_pending_action_data(task, db_session)
            task.agent_state.scratchpad += f"\n\nHuman rejected the pending action: {rejection_reason}"
            return None

        else:
            approved_action = pending_data.get("approved_action")
            if approved_action:
                logger.info(f"Task {task.task_id} was approved, executing pending action")
                tool_call = ToolCall(
                    id=approved_action.get("id", str(uuid.uuid4())),
                    name=approved_action["name"],
                    arguments=approved_action["arguments"]
                )
                tool_result = await tool_manager.execute_tool(tool_call, session_id)
                await observe_step(
                    task.agent_state, "Executing approved action", tool_call,
                    tool_result.content, session_id, storage_manager, tracer
                )
                await clear_pending_action_data(task, db_session)

                if tool_call.name == "finish":
                    final_answer = tool_result.content.replace("TASK_COMPLETE: ", "")
                    logger.info("Agent completed task after approval with final answer")
                    return final_answer
            return None

    except Exception as e:
        logger.error(f"Error handling task resumption: {e}", exc_info=True)
        return None

async def update_task_for_approval(
    task: AgentTask,
    approval_prompt: str,
    pending_action: Dict[str, Any],
    db_session: Optional[AsyncSession] = None
) -> None:
    """Update the task state for human approval workflow."""
    try:
        task.status = "PENDING_APPROVAL"
        task.approval_prompt = approval_prompt
        task.pending_action_data = {
            "approved_action": pending_action,
            "requested_at": datetime.now(timezone.utc).isoformat()
        }
        task.updated_at = datetime.now(timezone.utc)

        if db_session:
            update_query = text("""
                UPDATE agent_tasks
                SET status = :status, approval_prompt = :approval_prompt,
                    pending_action_data = :pending_action_data, updated_at = :updated_at
                WHERE task_id = :task_id
            """)
            await db_session.execute(update_query, {
                "task_id": task.task_id, "status": task.status,
                "approval_prompt": task.approval_prompt,
                "pending_action_data": json.dumps(task.pending_action_data),
                "updated_at": task.updated_at
            })
            await db_session.commit()
            logger.info(f"Task {task.task_id} updated to PENDING_APPROVAL state")
        else:
            logger.warning("No database session available to persist HITL state")

    except Exception as e:
        logger.error(f"Error updating task for approval: {e}", exc_info=True)
        raise

async def clear_pending_action_data(task: AgentTask, db_session: Optional[AsyncSession]) -> None:
    """Clear the pending action data from the task."""
    try:
        task.pending_action_data = None
        task.approval_prompt = None
        task.updated_at = datetime.now(timezone.utc)

        if db_session:
            update_query = text("""
                UPDATE agent_tasks
                SET pending_action_data = NULL, approval_prompt = NULL, updated_at = :updated_at
                WHERE task_id = :task_id
            """)
            await db_session.execute(update_query, {
                "task_id": task.task_id, "updated_at": task.updated_at
            })
            await db_session.commit()

    except Exception as e:
        logger.error(f"Error clearing pending action data: {e}", exc_info=True)
