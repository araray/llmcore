# src/llmcore/agents/prompt_utils.py
"""
Prompt management utilities for the LLMCore agent system.

This module provides helper functions for loading prompt templates,
building the comprehensive prompt sent to the LLM during the agent's
reasoning step, and parsing the structured response (Thought, Action)
from the model's output.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..models import AgentState, Tool, ToolCall

logger = logging.getLogger(__name__)


def load_planning_prompt_template() -> str:
    """Load the planning prompt template from file or return default."""
    # In a real implementation, this might load from a file.
    # For simplicity and to match the original, it's defined here.
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


def load_reflection_prompt_template() -> str:
    """Load the reflection prompt template from file or return default."""
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


def parse_plan_from_response(response_content: str) -> List[str]:
    """Parse a numbered plan from the LLM response."""
    try:
        plan_steps = []
        lines = response_content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if re.match(r"^\d+\.", line):
                step = re.sub(r"^\d+\.\s*", "", line).strip()
                if step:
                    plan_steps.append(step)

        if not plan_steps:
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith(("GOAL:", "INSTRUCTIONS:")):
                    plan_steps.append(line)
                    if len(plan_steps) >= 3:
                        break
        return plan_steps[:8]
    except Exception as e:
        logger.error(f"Error parsing plan from response: {e}")
        return []


def parse_reflection_response(response_content: str) -> Optional[Dict[str, Any]]:
    """Parse the JSON reflection response from the LLM."""
    try:
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(response_content.strip())
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reflection JSON: {e}")
        try:
            return {
                "evaluation": "Reflection analysis completed",
                "plan_step_completed": "complet" in response_content.lower()
                or "success" in response_content.lower(),
                "updated_plan": None,
            }
        except Exception:
            return None
    except Exception as e:
        logger.error(f"Error parsing reflection response: {e}")
        return None


def build_enhanced_agent_prompt(
    agent_state: AgentState, context_items: List[Any], tool_definitions: List[Tool]
) -> str:
    """
    Build the comprehensive prompt for the agent's reasoning step with plan context.

    Args:
        agent_state: Current agent state.
        context_items: Relevant context from memory systems.
        tool_definitions: The definitions of tools available for the current run.

    Returns:
        Formatted prompt string for the LLM.
    """
    context_str = ""
    if context_items:
        context_str = "\n\nRELEVANT CONTEXT:\n"
        for i, item in enumerate(context_items[:5]):
            content = getattr(item, "content", str(item))
            context_str += f"{i + 1}. {content[:300]}{'...' if len(content) > 300 else ''}\n"

    plan_str = ""
    if agent_state.plan:
        plan_str = "\n\nYOUR STRATEGIC PLAN:\n"
        for i, step in enumerate(agent_state.plan):
            status_icon = (
                "✓"
                if i < len(agent_state.plan_steps_status)
                and agent_state.plan_steps_status[i] == "completed"
                else "○"
            )
            current_marker = " 👉 CURRENT STEP" if i == agent_state.current_plan_step_index else ""
            plan_str += f"{status_icon} {i + 1}. {step}{current_marker}\n"

    thoughts_str = ""
    if agent_state.history_of_thoughts:
        recent_thoughts = agent_state.history_of_thoughts[-3:]
        thoughts_str = "\n\nPREVIOUS THOUGHTS:\n" + "\n".join(
            f"- {thought}" for thought in recent_thoughts
        )

    observations_str = ""
    if agent_state.observations:
        recent_obs = list(agent_state.observations.values())[-3:]
        observations_str = "\n\nRECENT OBSERVATIONS:\n"
        for obs in recent_obs:
            tool_name = obs.get("tool_name", "unknown")
            result = obs.get("result", "")[:200]
            observations_str += (
                f"- {tool_name}: {result}{'...' if len(obs.get('result', '')) > 200 else ''}\n"
            )

    tools_str = "\n\nAVAILABLE TOOLS:\n"
    if tool_definitions:
        for tool in tool_definitions:
            tools_str += f"- {tool.name}: {tool.description}\n"
    else:
        tools_str += "No tools available for this run.\n"

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


def parse_agent_response(
    content: str, full_response: Dict[str, Any], available_tools: List[str]
) -> Tuple[Optional[str], Optional[ToolCall]]:
    """
    Parse the agent's response to extract thought and tool call.

    Args:
        content: The text content from the LLM.
        full_response: The full response dict which may contain tool calls.
        available_tools: A list of names of available tools for parsing fallback.

    Returns:
        Tuple of (thought, tool_call) or (None, None) if parsing fails.
    """
    try:
        thought = _extract_thought(content)
        tool_call = _extract_tool_call_from_response(full_response)

        if not tool_call:
            tool_call = _extract_tool_call_from_content(content, available_tools)

        if thought and tool_call:
            return thought, tool_call
        else:
            logger.warning(
                f"Failed to parse agent response - thought: {bool(thought)}, tool_call: {bool(tool_call)}"
            )
            return None, None
    except Exception as e:
        logger.error(f"Error parsing agent response: {e}", exc_info=True)
        return None, None


def _extract_thought(content: str) -> Optional[str]:
    """Extract the thought section from agent response."""
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\n\n|\n[A-Z]|$)", content, re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        return thought_match.group(1).strip()

    lines = content.strip().split("\n")
    if lines:
        for line in lines:
            line = line.strip()
            if (
                len(line) > 10
                and not line.startswith("{")
                and not line.lower().startswith("action:")
            ):
                return line
    return None


def _extract_tool_call_from_response(response: Dict[str, Any]) -> Optional[ToolCall]:
    """Extract tool call from structured LLM response (function calling)."""
    try:
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tool_call_data = tool_calls[0]
            function_data = tool_call_data.get("function", {})
            return ToolCall(
                id=tool_call_data.get("id", str(uuid.uuid4())),
                name=function_data.get("name", ""),
                arguments=json.loads(function_data.get("arguments", "{}")),
            )
    except Exception as e:
        logger.debug(f"No structured tool call found in response: {e}")
    return None


def _extract_tool_call_from_content(content: str, available_tools: List[str]) -> Optional[ToolCall]:
    """Extract tool call from text content when no structured calling is available."""
    try:
        json_match = re.search(r'\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*\}', content, re.DOTALL)
        if json_match:
            try:
                tool_data = json.loads(json_match.group(0))
                return ToolCall(
                    id=str(uuid.uuid4()),
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("arguments", {}),
                )
            except json.JSONDecodeError:
                pass

        action_patterns = [
            r"Action:\s*(\w+)\s*\((.*?)\)",
            r"Tool:\s*(\w+)\s*\((.*?)\)",
            r"Call:\s*(\w+)\s*\((.*?)\)",
            r"Use:\s*(\w+)\s*\((.*?)\)",
        ]
        for pattern in action_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                tool_name, args_str = match.groups()
                tool_name = tool_name.strip()
                args_str = args_str.strip()
                arguments = {}
                if args_str:
                    for arg_pair in args_str.split(","):
                        if "=" in arg_pair:
                            key, value = arg_pair.split("=", 1)
                            arguments[key.strip().strip("\"'")] = value.strip().strip("\"'")
                return ToolCall(id=str(uuid.uuid4()), name=tool_name, arguments=arguments)

        for tool_name in available_tools:
            if tool_name.lower() in content.lower():
                if tool_name == "finish":
                    answer_match = re.search(r'["\']([^"\']+)["\']', content)
                    answer = answer_match.group(1) if answer_match else "Task completed"
                    return ToolCall(
                        id=str(uuid.uuid4()), name="finish", arguments={"answer": answer}
                    )
    except Exception as e:
        logger.debug(f"Error extracting tool call from content: {e}")
    return None
