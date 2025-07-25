# src/llmcore/agents/manager.py
"""
Agent Management for LLMCore.

Orchestrates the Think -> Act -> Observe execution loop for autonomous agent behavior.
Implements the ReAct (Reason + Act) paradigm for complex problem-solving.

UPDATED: Added manual OpenTelemetry spans around key agent logic steps for enhanced tracing.
UPDATED: Integrated dynamic tool loading to support tenant-specific toolsets.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

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
    Orchestrates the autonomous agent execution loop.

    Manages the Think -> Act -> Observe cycle, coordinating between the LLM provider,
    memory systems, and tool execution to achieve complex goals autonomously.

    UPDATED: Added distributed tracing spans for detailed agent behavior analysis.
    UPDATED: Integrated dynamic tool loading for tenant-specific tool management.
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

        logger.info("AgentManager initialized")

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
        Orchestrates the Think -> Act -> Observe loop for autonomous task execution.

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

        logger.info(f"Starting agent loop for goal: '{agent_state.goal[:100]}...'")

        # Create main agent loop span
        span_attributes = {
            "agent.task_id": task.task_id,
            "agent.goal": agent_state.goal[:200],  # Truncate for span
            "agent.max_iterations": max_iterations,
            "agent.session_id": actual_session_id,
            "agent.provider": provider_name or "default",
            "agent.model": model_name or "default",
            "agent.enabled_toolkits": str(enabled_toolkits) if enabled_toolkits else "all"
        }

        from ..tracing import create_span
        with create_span(self._tracer, "agent.execution_loop", **span_attributes) as main_span:
            try:
                from ..tracing import add_span_attributes

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

                for iteration in range(max_iterations):
                    iteration_attributes = {
                        "agent.iteration": iteration + 1,
                        "agent.iteration_total": max_iterations
                    }

                    with create_span(self._tracer, "agent.iteration", **iteration_attributes) as iter_span:
                        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

                        # 1. THINK: Construct prompt and call LLM
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

                        # 2. ACT: Execute the tool
                        tool_result = await self._act_step(tool_call, actual_session_id)

                        # 3. OBSERVE: Process the result and update state
                        observation = tool_result.content
                        await self._observe_step(agent_state, thought, tool_call, observation, actual_session_id)

                        # Add iteration details to span
                        if iter_span:
                            add_span_attributes(iter_span, {
                                "agent.thought": thought[:200],  # Truncate for span
                                "agent.tool_called": tool_call.name,
                                "agent.tool_success": "TASK_COMPLETE" not in observation or "ERROR" not in observation,
                                "agent.observation_length": len(observation)
                            })

                        # 4. CHECK FOR FINISH
                        if tool_call.name == "finish":
                            final_answer = observation.replace("TASK_COMPLETE: ", "")
                            logger.info(f"Agent completed task after {iteration + 1} iterations")

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
                                    "agent.final_answer_length": len(final_answer)
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

    async def _think_step(
        self,
        agent_state: AgentState,
        session_id: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[ToolCall]]:
        """
        Execute the THINK step: retrieve context and call LLM for reasoning.

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

                # Build the prompt
                prompt = self._build_agent_prompt(agent_state, context_items)

                # Get LLM provider
                provider = self._provider_manager.get_provider(provider_name)
                target_model = model_name or provider.default_model

                # Create messages for the LLM
                messages = [
                    Message(
                        role=Role.SYSTEM,
                        content="You are an autonomous AI agent. Follow the ReAct format: provide a Thought explaining your reasoning, then specify an Action (tool call) to take.",
                        session_id=session_id
                    ),
                    Message(
                        role=Role.USER,
                        content=prompt,
                        session_id=session_id
                    )
                ]

                # Call the LLM
                logger.debug("Calling LLM for agent reasoning...")
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
                        "think.parsing_success": result[0] is not None and result[1] is not None
                    })

                return result

            except Exception as e:
                logger.error(f"Error in think step: {e}", exc_info=True)
                if span:
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)
                return None, None

    async def _act_step(self, tool_call: ToolCall, session_id: str) -> ToolResult:
        """
        Execute the ACT step: run the requested tool.

        Args:
            tool_call: The tool call to execute.
            session_id: Session ID for tools that need context.

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
                    "act.session_id": session_id
                })

            try:
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
                        "act.success": not result.content.startswith("ERROR:")
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
                agent_state.scratchpad = f"Last thought: {thought}\nLast action: {tool_call.name}\nLast observation: {observation[:200]}..."

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
                        "iteration_summary": f"Agent reasoned, used {tool_call.name}, observed result"
                    }
                )

                try:
                    await self._storage_manager.add_episode(episode)
                    logger.debug(f"Logged T-A-O cycle as episode {episode.episode_id}")
                except Exception as e:
                    logger.warning(f"Failed to log episode: {e}")

                if span:
                    add_span_attributes(span, {
                        "observe.thought_length": len(thought),
                        "observe.observation_length": len(observation),
                        "observe.total_thoughts": len(agent_state.history_of_thoughts),
                        "observe.total_observations": len(agent_state.observations),
                        "observe.episode_logged": True
                    })

            except Exception as e:
                logger.error(f"Error in observe step: {e}", exc_info=True)
                if span:
                    from ..tracing import record_span_exception
                    record_span_exception(span, e)

    def _build_agent_prompt(self, agent_state: AgentState, context_items: List[Any]) -> str:
        """
        Build the comprehensive prompt for the agent's reasoning step.

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

        # Build the main prompt
        prompt = f"""GOAL: {agent_state.goal}

You are an autonomous AI agent working to achieve the above goal. Use the ReAct methodology:

1. **Think** about what you need to do next
2. **Act** by calling one of the available tools
3. **Observe** the result and continue

{context_str}{thoughts_str}{observations_str}{tools_str}

INSTRUCTIONS:
- Provide your reasoning in a "Thought:" section
- Then call exactly ONE tool to take action
- If you have enough information to answer the goal, use the "finish" tool with your final answer
- Be thorough but efficient in your approach

CURRENT SITUATION:
{agent_state.scratchpad if agent_state.scratchpad else "Starting fresh on this goal."}

Please provide your Thought and then make a tool call."""

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
            "finish": "answer"
        }
        return tool_param_map.get(tool_name)

    def _summarize_agent_state(self, agent_state: AgentState) -> str:
        """Create a summary of the current agent state."""
        thoughts_count = len(agent_state.history_of_thoughts)
        observations_count = len(agent_state.observations)

        last_thought = ""
        if agent_state.history_of_thoughts:
            last_thought = agent_state.history_of_thoughts[-1][:100] + "..."

        return f"Completed {thoughts_count} reasoning steps and {observations_count} actions. Last thought: {last_thought}"

    def get_tool_manager(self) -> ToolManager:
        """Get the ToolManager instance for external access."""
        return self._tool_manager
