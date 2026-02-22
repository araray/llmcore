# src/llmcore/agents/activities/prompts.py
"""
Activity System Prompts.

Provides system prompts for models without native tool support to use
the activity-based execution system.

The activity system prompt instructs the model to output structured XML
activity requests that can be parsed and executed by the activity loop.

References:
    - G3_COMPLETE_IMPLEMENTATION_PLAN.md
    - LLMCORE_AGENTIC_SYSTEM_MASTER_PLAN_G3.md Section 9
"""


# =============================================================================
# ACTIVITY SYSTEM PROMPT
# =============================================================================

ACTIVITY_SYSTEM_PROMPT = """
You are an autonomous AI agent. Since native function calling is not available,
you must use the activity system to perform actions.

## How to Request Activities

When you need to perform an action, use XML activity request format:

```xml
<activity_request>
    <activity>activity_name</activity>
    <parameters>
        <param_name>value</param_name>
        <another_param>another_value</another_param>
    </parameters>
    <reasoning>Brief explanation of why you're doing this</reasoning>
</activity_request>
```

## Available Activities

- **file_read**: Read the contents of a file
  - Parameters: path (required)
  - Example: <activity_request><activity>file_read</activity><parameters><path>/home/user/config.yaml</path></parameters><reasoning>Need to check configuration</reasoning></activity_request>

- **file_write**: Write content to a file
  - Parameters: path (required), content (required)

- **file_search**: Search for files matching a pattern
  - Parameters: pattern (required), directory (optional)

- **python_exec**: Execute Python code
  - Parameters: code (required)
  - Example: <activity_request><activity>python_exec</activity><parameters><code>print(2 + 2)</code></parameters><reasoning>Calculate arithmetic</reasoning></activity_request>

- **bash_exec**: Execute a shell/bash command
  - Parameters: command (required)

- **web_search**: Search the web
  - Parameters: query (required)

- **final_answer**: Complete the task with a final answer
  - Parameters: r (required) - the result/answer
  - Use when task is complete
  - Example: <activity_request><activity>final_answer</activity><parameters><r>4</r></parameters><reasoning>Task complete</reasoning></activity_request>

## Important Guidelines

1. **Always include reasoning**: Explain why you're taking each action
2. **One activity at a time**: Request activities sequentially
3. **Check results**: Wait for activity results before proceeding
4. **Use final_answer**: When done, use the final_answer activity

## Example Flow

1. Think about what to do
2. Request an activity with proper XML format
3. Receive observation/result
4. Think about next step
5. Repeat until task is complete
6. Use final_answer activity

Remember: Format your activity requests exactly as shown above.
"""


# =============================================================================
# PROMPT GENERATION
# =============================================================================


def generate_activity_prompt(
    goal: str,
    current_step: str,
    history: str | None = None,
    context: str | None = None,
    available_activities: list[str] | None = None,
) -> str:
    """
    Generate a prompt for activity-based execution.

    Args:
        goal: The task goal
        current_step: Current step being executed
        history: Recent action history
        context: Relevant context
        available_activities: List of available activity names

    Returns:
        Formatted prompt for activity-based execution
    """
    prompt = f"""You are solving this task using the activity system:

GOAL: {goal}

CURRENT STEP: {current_step}
"""

    # Include available activities so model knows valid names
    if available_activities:
        activities_list = ", ".join(available_activities)
        prompt += f"\nAVAILABLE ACTIVITIES: {activities_list}\n"
        prompt += "IMPORTANT: You MUST use one of the activities listed above. Do not invent activity names.\n"

    if history:
        prompt += f"\n\nRECENT HISTORY:\n{history}"

    if context:
        prompt += f"\n\nRELEVANT CONTEXT:\n{context}"

    prompt += """

Use the activity system to accomplish your goal. Format your response as:

1. Think about what to do next
2. Request an activity using the XML format

Example:
I need to read the configuration file to understand the current settings.

<activity_request>
    <activity>file_read</activity>
    <parameters>
        <path>/path/to/config.yaml</path>
    </parameters>
    <reasoning>Reading configuration to understand current settings</reasoning>
</activity_request>

OR if the task is complete:

I have completed the task successfully.

<activity_request>
    <activity>final_answer</activity>
    <parameters>
        <result>The calculation result is 42.</result>
    </parameters>
    <reasoning>Task complete, providing final answer</reasoning>
</activity_request>

Respond now:
"""

    return prompt


def get_activity_system_prompt() -> str:
    """Get the activity system prompt."""
    return ACTIVITY_SYSTEM_PROMPT


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ACTIVITY_SYSTEM_PROMPT",
    "generate_activity_prompt",
    "get_activity_system_prompt",
]
