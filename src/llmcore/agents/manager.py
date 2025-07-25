# src/llmcore/agents/manager.py
"""
Agent Management for LLMCore.

Orchestrates the Plan -> Think -> Act -> Observe -> Reflect execution loop for autonomous agent behavior.
Implements an enhanced cognitive architecture with explicit planning and reflection steps.

UPDATED: Added planning and reflection steps to transform the agent from reactive to strategic.
UPDATED: Enhanced cognitive cycle: Plan -> (Think -> Act -> Observe -> Reflect) with plan tracking.
UPDATED: Added manual OpenTelemetry spans around key agent logic steps for enhanced tracing.
UPDATED: Integrated dynamic tool loading to support tenant-specific toolsets.
UPDATED: Added Human-in-the-Loop (HITL) workflow support with pause/resume capabilities.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..exceptions import LLMCoreError, ProviderError
from ..memory.manager import MemoryManager
from ..models import (AgentState, AgentTask, Episode, EpisodeType, Message,
                      Role, ToolCall, ToolResult)
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from .tools import ToolManager

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Orchestrates the autonomous agent execution loop with strategic planning and reflection.

    Manages the Plan -> (Think -> Act -> Observe -> Reflect) cycle, coordinating between
    the LLM provider, memory systems, and tool execution to achieve complex goals autonomously.
    The enhanced cognitive architecture includes explicit planning before execution and
    critical reflection after each action to enable strategic problem-solving.

    UPDATED: Added distributed tracing spans for detailed agent behavior analysis.
    UPDATED: Integrated dynamic tool loading for tenant-specific tool management.
    UPDATED: Enhanced with planning and reflection capabilities for strategic reasoning.
    UPDATED: Added Human-in-the-Loop (HITL) workflow support for safe execution.
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

        # Initialize tracing
        self._tracer = None
        try:
            from ..tracing import get_tracer
            self._tracer = get_tracer("llmcore.agents.manager")
        except Exception as e:
            logger.debug(f"Tracing not available for AgentManager: {e}")

        logger.info("AgentManager initialized with planning, reflection, and HITL capabilities")

    async def run_agent_loop(
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
        Orchestrates the Plan -> (Think -> Act -> Observe -> Reflect) loop for autonomous task execution.

        UPDATED: Added HITL workflow support - can pause execution for human approval and resume.

        Args:
            task: The AgentTask containing the goal and initial state.
            provider_name: Optional override for the LLM provider.
            model_name: Optional override for the model name.
            max_iterations: Maximum number of loop iterations to prevent infinite loops.
            session_id: Optional session ID for episodic memory context.
            db_session: Tenant-scoped database session for dynamic tool loading.
            enabled_toolkits: List of toolkit names to enable for this run.

        Returns:
            The final result or answer from the agent.

        Raises:
            ProviderError: If LLM provider interactions fail.
            LLMCoreError: If the agent loop encounters unrecoverable errors.
        """
        agent_state = task.agent_state
        actual_session_id = session_id or task.task_id

        logger.info(f"Starting enhanced agent loop with planning and HITL for goal: '{agent_state.goal[:100]}...'")

        # Create main agent loop span
        span_attributes = {
            "agent.task_id": task.task_id,
            "agent.goal": agent_state.goal[:200],  # Truncate for span
            "agent.max_iterations": max_iterations,
            "agent.session_id": actual_session_id,
            "agent.provider": provider_name or "default",
            "agent.model": model_name or "default",
            "agent.enabled_toolkits": str(enabled_toolkits) if enabled_toolkits else "all",
            "agent.enhanced_mode": True,  # Flag for the new cognitive architecture
            "agent.hitl_enabled": True  # Flag for HITL support
        }

        from ..tracing import create_span
        with create_span(self._tracer, "agent.enhanced_execution_loop", **span_attributes) as main_span:
            try:
                from ..tracing import add_span_attributes

                # Check if this is a resuming task (from HITL workflow)
                is_resuming_task = self._check_if_resuming_task(task, db_session)
                if is_resuming_task:
                    logger.info(f"Resuming agent task {task.task_id} from HITL workflow")
                    if main_span:
                        add_span_attributes(main_span, {"agent.resuming_from_hitl": True})

                    # Handle the resumption logic
                    resume_result = await self._handle_task_resumption(task, actual_session_id, provider_name, model_name, db_session)
                    if resume_result:
                        return resume_result

                # Load tools for this agent run if database session is available
                if db_session:
                    try:
                        await self._tool_manager.load_tools_for_run(db_session, enabled_toolkits)
                        loaded_tools = self._tool_manager.get_tool_names()
                        logger.info(f"Loaded {len(loaded_tools)} tools for agent run: {loaded_tools}")

                        if main_span:
                            add_span_attributes(main_span, {
                                "agent.tools_loaded": len(loaded_tools),
                                "agent.available_tools": str(loaded_tools)[:200]  # Truncate for span
                            })
                    except Exception as e:
                        logger.error(f"Failed to load tools for agent run: {e}", exc_info=True)
                        error_msg = f"Failed to load tools for agent: {str(e)}"
                        if main_span:
                            add_span_attributes(main_span, {
                                "agent.error": error_msg,
                                "agent.status": "failed"
                            })
                        return f"Agent error: {error_msg}"
                else:
                    logger.warning("No database session provided for tool loading - agent will have no tools available")

                # STEP 1: PLANNING - Execute planning step before the main loop (skip if resuming)
                if not is_resuming_task:
                    await self._plan_step(agent_state, actual_session_id, provider_name, model_name)

                    # Initialize plan tracking
                    if agent_state.plan and not agent_state.plan_steps_status:
                        agent_state.plan_steps_status = ['pending'] * len(agent_state.plan)

                    logger.info(f"Agent generated plan with {len(agent_state.plan)} steps: {agent_state.plan}")

                # MAIN LOOP: Think -> Act -> Observe -> Reflect
                for iteration in range(max_iterations):
                    iteration_attributes = {
                        "agent.iteration": iteration + 1,
                        "agent.iteration_total": max_iterations,
                        "agent.current_plan_step": agent_state.current_plan_step_index + 1 if agent_state.plan else 0,
                        "agent.plan_steps_completed": sum(1 for status in agent_state.plan_steps_status if status == 'completed')
                    }

                    with create_span(self._tracer, "agent.iteration", **iteration_attributes) as iter_span:
                        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

                        # 1. THINK: Construct prompt and call LLM (enhanced with plan context)
                        thought, tool_call = await self._think_step(
                            agent_state,
                            actual_session_id,
                            provider_name,
                            model_name
                        )

                        if not thought or not tool_call:
                            error_msg = f"Failed to get valid thought and action from LLM at iteration {iteration + 1}"
                            logger.error(error_msg)
                            if main_span:
                                add_span_attributes(main_span, {
                                    "agent.error": error_msg,
                                    "agent.status": "failed"
                                })
                            return f"Agent error: {error_msg}"

                        # 2. ACT: Execute the tool (with HITL check)
                        tool_result = await self._act_step(tool_call, actual_session_id, task, db_session)

                        # Check if the task was paused for human approval
                        if tool_result.content == "PAUSED_FOR_APPROVAL":
                            logger.info(f"Agent task {task.task_id} paused for human approval")
                            if main_span:
                                add_span_attributes(main_span, {
                                    "agent.status": "paused_for_approval",
                                    "agent.iterations_used": iteration + 1
                                })
                            # Return a special message indicating the task is paused
                            return f"Agent task paused for human approval. Use the API to approve or reject the pending action."

                        # 3. OBSERVE: Process the result and update state
                        observation = tool_result.content
                        await self._observe_step(agent_state, thought, tool_call, observation, actual_session_id)

                        # 4. REFLECT: Critically evaluate progress and update plan if needed
                        await self._reflect_step(agent_state, tool_call, tool_result, actual_session_id, provider_name, model_name)

                        # Add iteration details to span
                        if iter_span:
                            add_span_attributes(iter_span, {
                                "agent.thought": thought[:200],  # Truncate for span
                                "agent.tool_called": tool_call.name,
                                "agent.tool_success": "TASK_COMPLETE" not in observation or "ERROR" not in observation,
                                "agent.observation_length": len(observation),
                                "agent.reflection_completed": True
                            })

                        # 5. CHECK FOR FINISH
                        if tool_call.name == "finish":
                            final_answer = observation.replace("TASK_COMPLETE: ", "")
                            logger.info(f"Agent completed task after {iteration + 1} iterations with {len(agent_state.plan)} planned steps")

                            # Record successful completion metrics
                            try:
                                from ..api_server.metrics import record_agent_execution
                                from ..api_server.middleware.observability import get_current_request_context
                                context = get_current_request_context()
                                tenant_id = context.get('tenant_id', 'unknown')

                                record_agent_execution(
                                    tenant_id=tenant_id,
                                    iterations=iteration + 1,
                                    status="completed"
                                )
                            except Exception as e:
                                logger.debug(f"Failed to record agent metrics: {e}")

                            if main_span:
                                add_span_attributes(main_span, {
                                    "agent.status": "completed",
                                    "agent.iterations_used": iteration + 1,
                                    "agent.final_answer_length": len(final_answer),
                                    "agent.plan_execution_completed": True
                                })

                            return final_answer

                        # Log progress
                        logger.debug(f"Iteration {iteration + 1} complete - Thought: {thought[:100]}... Action: {tool_call.name}")

                # Max iterations reached
                final_state_summary = self._summarize_agent_state(agent_state)
                logger.warning(f"Agent reached max iterations ({max_iterations}) without completion")

                # Record timeout metrics
                try:
                    from ..api_server.metrics import record_agent_execution
                    from ..api_server.middleware.observability import get_current_request_context
                    context = get_current_request_context()
                    tenant_id = context.get('tenant_id', 'unknown')

                    record_agent_execution(
                        tenant_id=tenant_id,
                        iterations=max_iterations,
                        status="timeout"
                    )
                except Exception as e:
                    logger.debug(f"Failed to record agent metrics: {e}")

                if main_span:
                    add_span_attributes(main_span, {
                        "agent.status": "timeout",
                        "agent.iterations_used": max_iterations
                    })

                return f"Agent reached maximum iterations ({max_iterations}) without completing the task. Current progress: {final_state_summary}"

            except Exception as e:
                error_msg = f"Agent loop failed: {str(e)}"
                logger.error(error_msg, exc_info=True)

                # Record error metrics
                try:
                    from ..api_server.metrics import record_agent_execution
                    from ..api_server.middleware.observability import get_current_request_context
                    context = get_current_request_context()
                    tenant_id = context.get('tenant_id', 'unknown')

                    record_agent_execution(
                        tenant_id=tenant_id,
                        iterations=0,  # Failed before completing any iterations
                        status="error"
                    )
                except Exception as metrics_error:
                    logger.debug(f"Failed to record agent metrics: {metrics_error}")

                if main_span:
                    from ..tracing import record_span_exception, add_span_attributes
                    record_span_exception(main_span, e)
                    add_span_attributes(main_span, {
                        "agent.status": "error",
                        "agent.error_message": str(e)
                    })

                return f"Agent error: {error_msg}"

    def _check_if_resuming_task(self, task: AgentTask, db_session: Optional[AsyncSession]) -> bool:
        """
        Check if this task is resuming from a HITL workflow.

        Args:
            task: The AgentTask to check
            db_session: Database session for querying task state

        Returns:
            True if the task is resuming from HITL, False otherwise
        """
        # A task is resuming if it has pending_action_data (either approval or rejection)
        return task.pending_action_data is not None

    async def _handle_task_resumption(
        self,
        task: AgentTask,
        session_id: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        db_session: Optional[AsyncSession] = None
    ) -> Optional[str]:
        """
        Handle the resumption of a task from HITL workflow.

        Args:
            task: The AgentTask being resumed
            session_id: Session ID for the task
            provider_name: Optional provider override
            model_name: Optional model override
            db_session: Database session for updates

        Returns:
            Final result if task completes during resumption, None to continue normal loop
        """
        try:
            pending_data = task.pending_action_data
            if not pending_data:
                return None

            # Check if this was an approval or rejection
            if "rejection_reason" in pending_data:
                # Task was rejected - provide rejection feedback as observation
                rejection_reason = pending_data["rejection_reason"]
                logger.info(f"Task {task.task_id} was rejected: {rejection_reason}")

                # Clear the pending data
                await self._clear_pending_action_data(task, db_session)

                # The agent will process this rejection in its next think step
                # Update the scratchpad with the rejection info
                task.agent_state.scratchpad += f"\n\nHuman rejected the pending action: {rejection_reason}"

                return None  # Continue with normal loop

            else:
                # Task was approved - execute the pending action
                approved_action = pending_data.get("approved_action")
                if approved_action:
                    logger.info(f"Task {task.task_id} was approved, executing pending action")

                    # Reconstruct the ToolCall from the stored data
                    tool_call = ToolCall(
                        id=approved_action.get("id", str(uuid.uuid4())),
                        name=approved_action["name"],
                        arguments=approved_action["arguments"]
                    )

                    # Execute the approved action
                    tool_result = await self._tool_manager.execute_tool(tool_call, session_id)

                    # Process the observation
                    await self._observe_step(
                        task.agent_state,
                        "Executing approved action",
                        tool_call,
                        tool_result.content,
                        session_id
                    )

                    # Clear the pending data
                    await self._clear_pending_action_data(task, db_session)

                    # Check if this was a finish action
                    if tool_call.name == "finish":
                        final_answer = tool_result.content.replace("TASK_COMPLETE: ", "")
                        logger.info(f"Agent completed task after approval with final answer")
                        return final_answer

                return None  # Continue with normal loop

        except Exception as e:
            logger.error(f"Error handling task resumption: {e}", exc_info=True)
            return None

    async def _clear_pending_action_data(self, task: AgentTask, db_session: Optional[AsyncSession]) -> None:
        """
        Clear the pending action data from the task.

        Args:
            task: The AgentTask to update
            db_session: Database session for updates
        """
        try:
            # Clear the fields in the task object
            task.pending_action_data = None
            task.approval_prompt = None
            task.updated_at = datetime.now(timezone.utc)

            # Update in database if session is available
            if db_session:
                update_query = text("""
                    UPDATE agent_tasks
                    SET pending_action_data = NULL,
                        approval_prompt = NULL,
                        updated_at = :updated_at
                    WHERE task_id = :task_id
                """)
                await db_session.execute(update_query, {
                    "task_id": task.task_id,
                    "updated_at": task.updated_at
                })
                await db_session.commit()

        except Exception as e:
            logger.error(f"Error clearing pending action data: {e}", exc_info=True)

    async def _plan_step(
        self,
        agent_state: AgentState,
        session_id: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Execute the PLANNING step: decompose the goal into actionable sub-tasks.

        Args:
            agent_state: Current agent state with goal to be planned.
            session_id: Session ID for context retrieval.
            provider_name: Optional provider override.
            model_name: Optional model override.
        """
        from ..tracing import create_span, add_span_attributes

        with create_span(self._tracer, "agent.plan_step") as span:
            try:
                logger.debug(f"Executing planning step for goal: {agent_state.goal}")

                # Load planning prompt template
                planning_prompt = self._load_planning_prompt_template()
                formatted_prompt = planning_prompt.format(goal=agent_state.goal)

                # Get LLM provider
                provider = self._provider_manager.get_provider(provider_name)
                target_model = model_name or provider.default_model

                # Create messages for the LLM
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

                # Call the LLM for planning
                logger.debug("Calling LLM for strategic planning...")
                response = await provider.chat_completion(
                    context=messages,
                    model=target_model,
                    stream=False
                )

                # Extract response content
                response_content = self._extract_response_content(response, provider)

                # Parse the plan from the response
                plan_steps = self._parse_plan_from_response(response_content)

                if plan_steps:
                    agent_state.plan = plan_steps
                    agent_state.current_plan_step_index = 0
                    agent_state.plan_steps_status = ['pending'] * len(plan_steps)
                    logger.info(f"Successfully generated plan with {len(plan_steps)} steps")
                else:
                    # Fallback: create a basic plan
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
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)

                # Fallback plan on error
                agent_state.plan = [
                    "Proceed with the goal using available tools",
                    "Use the finish tool when ready with results"
                ]
                agent_state.current_plan_step_index = 0
                agent_state.plan_steps_status = ['pending'] * len(agent_state.plan)

    async def _reflect_step(
        self,
        agent_state: AgentState,
        last_tool_call: ToolCall,
        last_observation: ToolResult,
        session_id: str,
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
            provider_name: Optional provider override.
            model_name: Optional model override.
        """
        from ..tracing import create_span, add_span_attributes

        with create_span(self._tracer, "agent.reflect_step") as span:
            try:
                logger.debug(f"Executing reflection step for tool: {last_tool_call.name}")

                # Load reflection prompt template
                reflection_prompt = self._load_reflection_prompt_template()

                # Format the prompt with context
                plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(agent_state.plan)])

                formatted_prompt = reflection_prompt.format(
                    goal=agent_state.goal,
                    plan=plan_text,
                    last_action_name=last_tool_call.name,
                    last_action_arguments=json.dumps(last_tool_call.arguments),
                    last_observation=last_observation.content
                )

                # Get LLM provider
                provider = self._provider_manager.get_provider(provider_name)
                target_model = model_name or provider.default_model

                # Create messages for the LLM
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

                # Call the LLM for reflection
                logger.debug("Calling LLM for critical reflection...")
                response = await provider.chat_completion(
                    context=messages,
                    model=target_model,
                    stream=False
                )

                # Extract and parse reflection response
                response_content = self._extract_response_content(response, provider)
                reflection_result = self._parse_reflection_response(response_content)

                # Update agent state based on reflection
                if reflection_result:
                    evaluation = reflection_result.get('evaluation', '')
                    plan_step_completed = reflection_result.get('plan_step_completed', False)
                    updated_plan = reflection_result.get('updated_plan')

                    # Update scratchpad with reflection
                    agent_state.scratchpad = f"Reflection: {evaluation[:300]}..."

                    # Update plan step status if completed
                    if plan_step_completed and agent_state.current_plan_step_index < len(agent_state.plan_steps_status):
                        agent_state.plan_steps_status[agent_state.current_plan_step_index] = 'completed'
                        agent_state.current_plan_step_index = min(
                            agent_state.current_plan_step_index + 1,
                            len(agent_state.plan) - 1
                        )
                        logger.debug(f"Marked plan step {agent_state.current_plan_step_index} as completed")

                    # Update plan if reflection suggests changes
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
                        add_span_attributes(span, {
                            "reflect.parsing_success": False
                        })

            except Exception as e:
                logger.error(f"Error in reflection step: {e}", exc_info=True)
                if span:
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)

    def _load_planning_prompt_template(self) -> str:
        """Load the planning prompt template from file or return default."""
        try:
            # Try to load from file (in a real implementation, this would be a proper file path)
            planning_template = """You are a strategic planning agent. Your task is to decompose a high-level goal into a numbered list of simple, actionable sub-tasks.

GOAL: {goal}

INSTRUCTIONS:
- Break down the goal into a logical sequence of steps
- Each step should correspond to a single tool use or action
- Steps should be specific and actionable
- The final step must always be to use the "finish" tool with your final answer
- Number each step clearly (1, 2, 3, etc.)
- Keep steps concise but descriptive
- Aim for 3-8 steps total

EXAMPLE FORMAT:
1. Use semantic_search to find information about X
2. Use calculator to compute Y based on the results
3. Use episodic_search to check if we've done similar analysis before
4. Use finish tool with the final analysis and recommendations

Please provide your numbered plan for achieving the goal:"""
            return planning_template
        except Exception as e:
            logger.debug(f"Could not load planning template: {e}")
            return "Create a step-by-step plan to achieve this goal: {goal}"

    def _load_reflection_prompt_template(self) -> str:
        """Load the reflection prompt template from file or return default."""
        try:
            reflection_template = """You are a critical evaluation agent. Your task is to reflect on the last action taken and assess progress toward the goal.

ORIGINAL GOAL: {goal}

CURRENT PLAN:
{plan}

LAST ACTION TAKEN:
Tool: {last_action_name}
Arguments: {last_action_arguments}

OBSERVATION FROM LAST ACTION:
{last_observation}

REFLECTION INSTRUCTIONS:
Critically evaluate the last action and its results. Consider:
1. Did the action successfully complete the current plan step?
2. Does the observation reveal new information that changes our understanding?
3. Should the plan be modified, reordered, or updated?
4. Are we making progress toward the goal?

Respond with a JSON object containing:
- "evaluation": Your critical assessment of the last action and observation
- "plan_step_completed": true/false - whether the current plan step was successfully completed
- "updated_plan": null if no changes needed, or a new list of plan steps if the plan should be modified

RESPONSE FORMAT (must be valid JSON):
{{
  "evaluation": "Your detailed analysis here",
  "plan_step_completed": true,
  "updated_plan": null
}}"""
            return reflection_template
        except Exception as e:
            logger.debug(f"Could not load reflection template: {e}")
            return "Reflect on the last action and determine if the plan step was completed. Goal: {goal}"

    def _parse_plan_from_response(self, response_content: str) -> List[str]:
        """Parse a numbered plan from the LLM response."""
        try:
            plan_steps = []
            lines = response_content.strip().split('\n')

            for line in lines:
                line = line.strip()
                # Look for numbered steps (1., 2., etc.)
                if re.match(r'^\d+\.', line):
                    # Remove the number and period, clean up the step
                    step = re.sub(r'^\d+\.\s*', '', line).strip()
                    if step:
                        plan_steps.append(step)

            # Ensure we have at least a basic plan
            if not plan_steps:
                # Try to extract any meaningful lines as steps
                for line in lines:
                    line = line.strip()
                    if len(line) > 10 and not line.startswith('GOAL:') and not line.startswith('INSTRUCTIONS:'):
                        plan_steps.append(line)
                        if len(plan_steps) >= 3:  # Limit fallback extraction
                            break

            return plan_steps[:8]  # Limit to 8 steps maximum

        except Exception as e:
            logger.error(f"Error parsing plan from response: {e}")
            return []

    def _parse_reflection_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON reflection response from the LLM."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # Fallback: try to parse the whole response as JSON
            return json.loads(response_content.strip())

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reflection JSON: {e}")

            # Fallback: extract basic information from text
            try:
                evaluation = "Reflection analysis completed"
                plan_step_completed = "complet" in response_content.lower() or "success" in response_content.lower()

                return {
                    "evaluation": evaluation,
                    "plan_step_completed": plan_step_completed,
                    "updated_plan": None
                }
            except Exception:
                return None
        except Exception as e:
            logger.error(f"Error parsing reflection response: {e}")
            return None

    async def _think_step(
        self,
        agent_state: AgentState,
        session_id: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[ToolCall]]:
        """
        Execute the THINK step: retrieve context and call LLM for reasoning.

        Enhanced with plan context to provide strategic guidance to the tactical thinking.

        Args:
            agent_state: Current agent state with goal and history.
            session_id: Session ID for context retrieval.
            provider_name: Optional provider override.
            model_name: Optional model override.

        Returns:
            Tuple of (thought, tool_call) or (None, None) if parsing fails.
        """
        from ..tracing import create_span, add_span_attributes

        with create_span(self._tracer, "agent.think_step") as span:
            try:
                # Get relevant context from memory systems
                context_items = await self._memory_manager.retrieve_relevant_context(agent_state.goal)

                # Build the enhanced prompt with plan context
                prompt = self._build_enhanced_agent_prompt(agent_state, context_items)

                # Get LLM provider
                provider = self._provider_manager.get_provider(provider_name)
                target_model = model_name or provider.default_model

                # Create messages for the LLM
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

                # Call the LLM
                logger.debug("Calling LLM for enhanced agent reasoning with plan context...")
                response = await provider.chat_completion(
                    context=messages,
                    model=target_model,
                    stream=False,
                    tools=self._tool_manager.get_tool_definitions()
                )

                # Extract response content
                response_content = self._extract_response_content(response, provider)

                # Parse the response to extract thought and tool call
                result = self._parse_agent_response(response_content, response)

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
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)
                return None, None

    async def _act_step(
        self,
        tool_call: ToolCall,
        session_id: str,
        task: AgentTask,
        db_session: Optional[AsyncSession] = None
    ) -> ToolResult:
        """
        Execute the ACT step: run the requested tool.

        UPDATED: Added HITL workflow support - checks for human_approval tool and pauses execution.

        Args:
            tool_call: The tool call to execute.
            session_id: Session ID for tools that need context.
            task: The current AgentTask (needed for HITL workflow).
            db_session: Database session for updating task state.

        Returns:
            ToolResult containing the execution result.
        """
        from ..tracing import create_span, add_span_attributes

        with create_span(self._tracer, "agent.act_step") as span:
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
                # HITL WORKFLOW: Check if this is a human_approval tool call
                if tool_call.name == "human_approval":
                    logger.info(f"Human approval requested for task {task.task_id}")

                    # Extract the approval prompt and pending action from arguments
                    approval_prompt = tool_call.arguments.get("prompt", "Approval requested for pending action")
                    pending_action = tool_call.arguments.get("pending_action", {})

                    # Update the task in the database to PENDING_APPROVAL state
                    await self._update_task_for_approval(task, approval_prompt, pending_action, db_session)

                    if span:
                        add_span_attributes(span, {
                            "act.hitl_triggered": True,
                            "act.approval_prompt": approval_prompt[:100],
                            "act.pending_action": pending_action.get("name", "unknown")
                        })

                    # Return special result to pause the agent loop
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        content="PAUSED_FOR_APPROVAL"
                    )

                # Normal tool execution
                result = await self._tool_manager.execute_tool(tool_call, session_id)

                # Record tool execution metrics
                try:
                    from ..api_server.metrics import record_tool_execution
                    from ..api_server.middleware.observability import get_current_request_context
                    context = get_current_request_context()
                    tenant_id = context.get('tenant_id', 'unknown')

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
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)

                # Return error result
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"ERROR: {str(e)}"
                )

    async def _update_task_for_approval(
        self,
        task: AgentTask,
        approval_prompt: str,
        pending_action: Dict[str, Any],
        db_session: Optional[AsyncSession] = None
    ) -> None:
        """
        Update the task state for human approval workflow.

        Args:
            task: The AgentTask to update
            approval_prompt: The prompt for the human operator
            pending_action: The action that needs approval
            db_session: Database session for updates
        """
        try:
            # Update the task object
            task.status = "PENDING_APPROVAL"
            task.approval_prompt = approval_prompt
            task.pending_action_data = {
                "approved_action": pending_action,
                "requested_at": datetime.now(timezone.utc).isoformat()
            }
            task.updated_at = datetime.now(timezone.utc)

            # Update in database if session is available
            if db_session:
                update_query = text("""
                    UPDATE agent_tasks
                    SET status = :status,
                        approval_prompt = :approval_prompt,
                        pending_action_data = :pending_action_data,
                        updated_at = :updated_at
                    WHERE task_id = :task_id
                """)
                await db_session.execute(update_query, {
                    "task_id": task.task_id,
                    "status": task.status,
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

    async def _observe_step(
        self,
        agent_state: AgentState,
        thought: str,
        tool_call: ToolCall,
        observation: str,
        session_id: str
    ) -> None:
        """
        Execute the OBSERVE step: update state and log experience.

        Args:
            agent_state: Agent state to update.
            thought: The agent's reasoning.
            tool_call: The action taken.
            observation: The result of the action.
            session_id: Session ID for episodic logging.
        """
        from ..tracing import create_span, add_span_attributes

        with create_span(self._tracer, "agent.observe_step") as span:
            try:
                # Update agent state
                agent_state.history_of_thoughts.append(thought)
                agent_state.observations[tool_call.id] = {
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": observation,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Update scratchpad with latest reasoning
                current_step = ""
                if agent_state.plan and agent_state.current_plan_step_index < len(agent_state.plan):
                    current_step = f" (Step {agent_state.current_plan_step_index + 1}: {agent_state.plan[agent_state.current_plan_step_index]})"

                agent_state.scratchpad = f"Last thought: {thought}\nLast action: {tool_call.name}{current_step}\nLast observation: {observation[:200]}..."

                # Log the T-A-O cycle as an episode in episodic memory
                episode = Episode(
                    session_id=session_id,
                    event_type=EpisodeType.AGENT_REFLECTION,
                    data={
                        "thought": thought,
                        "action": {
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "arguments": tool_call.arguments
                        },
                        "observation": observation,
                        "goal": agent_state.goal,
                        "current_plan_step": agent_state.current_plan_step_index + 1 if agent_state.plan else 0,
                        "plan": agent_state.plan,
                        "iteration_summary": f"Agent reasoned, used {tool_call.name}, observed result, executing plan step {agent_state.current_plan_step_index + 1}"
                    }
                )

                try:
                    await self._storage_manager.add_episode(episode)
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
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)

    def _build_enhanced_agent_prompt(self, agent_state: AgentState, context_items: List[Any]) -> str:
        """
        Build the comprehensive prompt for the agent's reasoning step with plan context.

        Enhanced version that includes the current plan and highlights the current step
        for strategic context during tactical thinking.

        Args:
            agent_state: Current agent state.
            context_items: Relevant context from memory systems.

        Returns:
            Formatted prompt string for the LLM.
        """
        # Format context items
        context_str = ""
        if context_items:
            context_str = "\n\nRELEVANT CONTEXT:\n"
            for i, item in enumerate(context_items[:5]):  # Limit to top 5 items
                content = getattr(item, 'content', str(item))
                context_str += f"{i+1}. {content[:300]}{'...' if len(content) > 300 else ''}\n"

        # Format current plan and progress
        plan_str = ""
        if agent_state.plan:
            plan_str = "\n\nYOUR STRATEGIC PLAN:\n"
            for i, step in enumerate(agent_state.plan):
                status_icon = "✓" if i < len(agent_state.plan_steps_status) and agent_state.plan_steps_status[i] == "completed" else "○"
                current_marker = " 👉 CURRENT STEP" if i == agent_state.current_plan_step_index else ""
                plan_str += f"{status_icon} {i+1}. {step}{current_marker}\n"

        # Format thought history
        thoughts_str = ""
        if agent_state.history_of_thoughts:
            recent_thoughts = agent_state.history_of_thoughts[-3:]  # Last 3 thoughts
            thoughts_str = "\n\nPREVIOUS THOUGHTS:\n" + "\n".join(f"- {thought}" for thought in recent_thoughts)

        # Format recent observations
        observations_str = ""
        if agent_state.observations:
            recent_obs = list(agent_state.observations.values())[-3:]  # Last 3 observations
            observations_str = "\n\nRECENT OBSERVATIONS:\n"
            for obs in recent_obs:
                tool_name = obs.get('tool_name', 'unknown')
                result = obs.get('result', '')[:200]
                observations_str += f"- {tool_name}: {result}{'...' if len(obs.get('result', '')) > 200 else ''}\n"

        # Format available tools
        tools = self._tool_manager.get_tool_definitions()
        tools_str = "\n\nAVAILABLE TOOLS:\n"
        if tools:
            for tool in tools:
                tools_str += f"- {tool.name}: {tool.description}\n"
        else:
            tools_str += "No tools available for this run.\n"

        # Build the main enhanced prompt
        prompt = f"""GOAL: {agent_state.goal}

You are an autonomous AI agent with strategic planning capabilities. You have created a plan and are now executing it step by step.

{plan_str}{context_str}{thoughts_str}{observations_str}{tools_str}

INSTRUCTIONS:
- Follow your strategic plan while using the ReAct methodology
- Focus on the CURRENT STEP marked with 👉 in your plan
- Provide your reasoning in a "Thought:" section that considers both the current step and overall goal
- Then call exactly ONE tool to take action toward completing the current step
- If you have completed all planned steps and have enough information to answer the goal, use the "finish" tool
- Be thorough but efficient in your approach
- Stay focused on your plan but be flexible if circumstances change
- For sensitive or irreversible actions, use the "human_approval" tool to request approval first

CURRENT SITUATION:
{agent_state.scratchpad if agent_state.scratchpad else "Starting execution of your strategic plan."}

Please provide your Thought about the current step and then make a tool call to progress toward your goal."""

        return prompt

    def _extract_response_content(self, response: Dict[str, Any], provider) -> str:
        """Extract the text content from LLM response."""
        try:
            return response['choices'][0]['message']['content'] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Could not extract content from {provider.get_name()} response")
            return ""

    def _parse_agent_response(self, content: str, full_response: Dict[str, Any]) -> Tuple[Optional[str], Optional[ToolCall]]:
        """
        Parse the agent's response to extract thought and tool call.

        Args:
            content: The text content from the LLM.
            full_response: The full response dict which may contain tool calls.

        Returns:
            Tuple of (thought, tool_call) or (None, None) if parsing fails.
        """
        try:
            # Extract thought from content
            thought = self._extract_thought(content)

            # Try to get tool call from the response structure first (function calling)
            tool_call = self._extract_tool_call_from_response(full_response)

            # If no structured tool call, try to parse from content
            if not tool_call:
                tool_call = self._extract_tool_call_from_content(content)

            if thought and tool_call:
                return thought, tool_call
            else:
                logger.warning(f"Failed to parse agent response - thought: {bool(thought)}, tool_call: {bool(tool_call)}")
                return None, None

        except Exception as e:
            logger.error(f"Error parsing agent response: {e}", exc_info=True)
            return None, None

    def _extract_thought(self, content: str) -> Optional[str]:
        """Extract the thought section from agent response."""
        # Look for "Thought:" section
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n\n|\n[A-Z]|$)', content, re.DOTALL | re.IGNORECASE)
        if thought_match:
            return thought_match.group(1).strip()

        # Fallback: if no explicit "Thought:" marker, use first paragraph
        lines = content.strip().split('\n')
        if lines:
            # Find first substantial line that looks like reasoning
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('{') and not line.lower().startswith('action:'):
                    return line

        return None

    def _extract_tool_call_from_response(self, response: Dict[str, Any]) -> Optional[ToolCall]:
        """Extract tool call from structured LLM response (function calling)."""
        try:
            # Check for tool calls in the standard OpenAI format
            message = response.get('choices', [{}])[0].get('message', {})
            tool_calls = message.get('tool_calls', [])

            if tool_calls:
                tool_call_data = tool_calls[0]  # Take first tool call
                function_data = tool_call_data.get('function', {})

                return ToolCall(
                    id=tool_call_data.get('id', str(uuid.uuid4())),
                    name=function_data.get('name', ''),
                    arguments=json.loads(function_data.get('arguments', '{}'))
                )
        except Exception as e:
            logger.debug(f"No structured tool call found in response: {e}")

        return None

    def _extract_tool_call_from_content(self, content: str) -> Optional[ToolCall]:
        """Extract tool call from text content when no structured calling is available."""
        try:
            # Look for JSON-like tool call pattern
            json_match = re.search(r'\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    tool_data = json.loads(json_match.group(0))
                    return ToolCall(
                        id=str(uuid.uuid4()),
                        name=tool_data.get('name', ''),
                        arguments=tool_data.get('arguments', {})
                    )
                except json.JSONDecodeError:
                    pass

            # Look for explicit action patterns
            action_patterns = [
                r'Action:\s*(\w+)\s*\((.*?)\)',
                r'Tool:\s*(\w+)\s*\((.*?)\)',
                r'Call:\s*(\w+)\s*\((.*?)\)',
                r'Use:\s*(\w+)\s*\((.*?)\)'
            ]

            for pattern in action_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    tool_name = match.group(1).strip()
                    args_str = match.group(2).strip()

                    # Parse arguments
                    arguments = {}
                    if args_str:
                        # Try to parse as simple key=value pairs
                        for arg_pair in args_str.split(','):
                            if '=' in arg_pair:
                                key, value = arg_pair.split('=', 1)
                                key = key.strip().strip('"\'')
                                value = value.strip().strip('"\'')
                                arguments[key] = value
                            else:
                                # Single argument - assume it's the main parameter
                                main_param = self._get_main_parameter_name(tool_name)
                                if main_param:
                                    arguments[main_param] = args_str.strip().strip('"\'')

                    return ToolCall(
                        id=str(uuid.uuid4()),
                        name=tool_name,
                        arguments=arguments
                    )

            # Look for tool names mentioned and infer simple calls
            available_tools = self._tool_manager.get_tool_names()
            for tool_name in available_tools:
                if tool_name.lower() in content.lower():
                    # Found a tool mention - try to create a basic call
                    if tool_name == "finish":
                        # For finish tool, extract any quoted text as the answer
                        answer_match = re.search(r'["\']([^"\']+)["\']', content)
                        answer = answer_match.group(1) if answer_match else "Task completed"
                        return ToolCall(
                            id=str(uuid.uuid4()),
                            name="finish",
                            arguments={"answer": answer}
                        )
                    elif tool_name == "human_approval":
                        # For human_approval tool, extract approval prompt and pending action
                        prompt_match = re.search(r'prompt["\']?\s*:\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                        prompt = prompt_match.group(1) if prompt_match else "Approval requested"

                        # Look for pending action info
                        pending_action = {}
                        action_match = re.search(r'pending_action["\']?\s*:\s*\{([^}]+)\}', content, re.IGNORECASE)
                        if action_match:
                            try:
                                pending_action = json.loads('{' + action_match.group(1) + '}')
                            except:
                                pending_action = {"name": "unknown_action", "arguments": {}}

                        return ToolCall(
                            id=str(uuid.uuid4()),
                            name="human_approval",
                            arguments={"prompt": prompt, "pending_action": pending_action}
                        )
                    elif tool_name in ["semantic_search", "episodic_search"]:
                        # Extract potential search query
                        query_patterns = [
                            r'search(?:\s+for)?\s+["\']([^"\']+)["\']',
                            r'find\s+["\']([^"\']+)["\']',
                            r'query:\s*["\']([^"\']+)["\']'
                        ]
                        for query_pattern in query_patterns:
                            query_match = re.search(query_pattern, content, re.IGNORECASE)
                            if query_match:
                                return ToolCall(
                                    id=str(uuid.uuid4()),
                                    name=tool_name,
                                    arguments={"query": query_match.group(1)}
                                )
                    elif tool_name == "calculator":
                        # Extract mathematical expression
                        calc_patterns = [
                            r'calculate\s+["\']([^"\']+)["\']',
                            r'compute\s+["\']([^"\']+)["\']',
                            r'expression:\s*["\']([^"\']+)["\']'
                        ]
                        for calc_pattern in calc_patterns:
                            calc_match = re.search(calc_pattern, content, re.IGNORECASE)
                            if calc_match:
                                return ToolCall(
                                    id=str(uuid.uuid4()),
                                    name="calculator",
                                    arguments={"expression": calc_match.group(1)}
                                )

        except Exception as e:
            logger.debug(f"Error extracting tool call from content: {e}")

        return None

    def _get_main_parameter_name(self, tool_name: str) -> Optional[str]:
        """Get the main parameter name for a tool."""
        tool_param_map = {
            "semantic_search": "query",
            "episodic_search": "query",
            "calculator": "expression",
            "finish": "answer",
            "human_approval": "prompt"
        }
        return tool_param_map.get(tool_name)

    def _summarize_agent_state(self, agent_state: AgentState) -> str:
        """Create a summary of the current agent state including plan progress."""
        thoughts_count = len(agent_state.history_of_thoughts)
        observations_count = len(agent_state.observations)

        plan_progress = ""
        if agent_state.plan:
            completed_steps = sum(1 for status in agent_state.plan_steps_status if status == 'completed')
            plan_progress = f" Plan progress: {completed_steps}/{len(agent_state.plan)} steps completed."

        last_thought = ""
        if agent_state.history_of_thoughts:
            last_thought = agent_state.history_of_thoughts[-1][:100] + "..."

        return f"Completed {thoughts_count} reasoning steps and {observations_count} actions.{plan_progress} Last thought: {last_thought}"

    def get_tool_manager(self) -> ToolManager:
        """Get the ToolManager instance for external access."""
        return self._tool_manager
