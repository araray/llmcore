# src/llmcore/agents/prompts/template_loader.py
"""
Template Loader for Darwin Layer 2 Prompt Library.

This module provides utilities for loading prompt templates from
TOML files and creating default templates programmatically.

The TemplateLoader handles:
- Loading templates from TOML configuration files
- Creating built-in default templates
- Registering templates with a PromptRegistry

References:
    - Technical Spec: Section 5.2 (Prompt Library Architecture)
    - Dossier: Step 2.3 (Template Loading)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import PromptRegistry

from .models import (
    PromptCategory,
    PromptSnippet,
    PromptTemplate,
    PromptVariable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPLATE LOADER CLASS
# =============================================================================


class TemplateLoader:
    """
    Loads and registers prompt templates from various sources.

    The TemplateLoader can load templates from:
    - TOML configuration files
    - Built-in default definitions
    - Custom template dictionaries

    Example:
        >>> from llmcore.agents.prompts import TemplateLoader, PromptRegistry
        >>>
        >>> registry = PromptRegistry()
        >>> loader = TemplateLoader(registry)
        >>>
        >>> # Load from TOML file
        >>> loader.load_from_toml(Path("prompts/planning.toml"))
        >>>
        >>> # Load defaults
        >>> loader.load_defaults()
    """

    def __init__(self, registry: "PromptRegistry"):
        """
        Initialize the template loader.

        Args:
            registry: The prompt registry to load templates into
        """
        self.registry = registry

    def load_defaults(self) -> int:
        """
        Load all built-in default templates.

        Returns:
            Number of templates loaded
        """
        return load_default_templates(self.registry)

    def load_from_toml(self, filepath: Path) -> int:
        """
        Load templates from a TOML file.

        Args:
            filepath: Path to the TOML file

        Returns:
            Number of templates loaded
        """
        try:
            import toml
        except ImportError:
            # Python 3.11+ can use tomllib for reading
            import tomllib

            with open(filepath, "rb") as f:
                data = tomllib.load(f)
        else:
            with open(filepath, "r") as f:
                data = toml.load(f)

        count = 0

        # Load snippets first
        for key, snippet_data in data.get("snippets", {}).items():
            snippet = PromptSnippet(key=key, **snippet_data)
            self.registry.register_snippet(snippet)
            count += 1

        # Load templates
        for template_id, template_data in data.get("templates", {}).items():
            template = PromptTemplate(id=template_id, **template_data)
            self.registry.register_template(template)
            count += 1

        logger.info(f"Loaded {count} items from {filepath}")
        return count


# =============================================================================
# DEFAULT TEMPLATE DEFINITIONS
# =============================================================================


def load_default_templates(registry: "PromptRegistry") -> int:
    """
    Load built-in default templates into a registry.

    Creates the core templates used by the cognitive cycle:
    - planning_prompt: Strategic task decomposition
    - thinking_prompt: ReAct-style reasoning
    - reflection_prompt: Self-evaluation and learning
    - validation_prompt: Plan validation

    Also registers common snippets for reuse.

    Args:
        registry: The prompt registry to populate

    Returns:
        Number of templates and snippets loaded
    """
    count = 0

    # =========================================================================
    # SNIPPETS
    # =========================================================================

    snippets = [
        PromptSnippet(
            key="agent_identity",
            content="You are an autonomous AI agent with advanced reasoning capabilities.",
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="react_framework",
            content=(
                "Use the ReAct (Reasoning + Acting) framework:\n"
                "1. Think: Analyze the situation and reason about next steps\n"
                "2. Act: Choose and execute an appropriate action\n"
                "3. Observe: Examine the results of your action\n"
                "4. Reflect: Learn from the outcome and adjust strategy"
            ),
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="tool_usage_instructions",
            content=(
                "When using tools:\n"
                "- Choose the most appropriate tool for the task\n"
                "- Provide all required parameters\n"
                "- Verify the tool output before proceeding\n"
                "- Handle errors gracefully"
            ),
            category=PromptCategory.SNIPPET,
        ),
        PromptSnippet(
            key="step_format",
            content=(
                "Format each step as:\n"
                "THOUGHT: [Your reasoning about what to do next]\n"
                "ACTION: [The tool to use]\n"
                "ACTION_INPUT: [The input for the tool in JSON format]"
            ),
            category=PromptCategory.SNIPPET,
        ),
    ]

    for snippet in snippets:
        try:
            registry.register_snippet(snippet)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register snippet {snippet.key}: {e}")

    # =========================================================================
    # PLANNING TEMPLATE
    # =========================================================================

    planning_template = registry.create_template(
        template_id="planning_prompt",
        name="Planning Prompt",
        category=PromptCategory.PLANNING,
        description="Strategic planning prompt for task decomposition",
        tags=["cognitive", "planning", "strategy"],
        initial_content="""You are a strategic planning agent. Your task is to decompose a high-level goal into a numbered list of simple, actionable sub-tasks.

GOAL: {{goal}}

{{#context}}
CONTEXT:
{{context}}
{{/context}}

AVAILABLE TOOLS:
{{tools}}

INSTRUCTIONS:
- Break down the goal into a logical sequence of steps
- Each step should correspond to a single tool use or action
- Steps should be specific and actionable
- The final step must always be to use the "finish" tool with your final answer
- Number each step clearly (1, 2, 3, etc.)
- Keep steps concise but descriptive
- Aim for 3-8 steps total

Please provide your numbered plan for achieving the goal:""",
        variables=[
            PromptVariable(
                name="goal", description="The high-level goal to achieve", required=True
            ),
            PromptVariable(name="context", description="Additional context", required=False),
            PromptVariable(name="tools", description="Available tools", required=True),
        ],
    )
    count += 1

    # =========================================================================
    # THINKING TEMPLATE
    # =========================================================================

    thinking_template = registry.create_template(
        template_id="thinking_prompt",
        name="Thinking Prompt",
        category=PromptCategory.REASONING,
        description="ReAct-style reasoning prompt for the cognitive cycle",
        tags=["cognitive", "reasoning", "react"],
        initial_content="""{{@agent_identity}}

{{@react_framework}}

GOAL: {{goal}}

CURRENT PLAN:
{{plan}}

CURRENT STEP: {{current_step}}

{{#previous_observations}}
PREVIOUS OBSERVATIONS:
{{previous_observations}}
{{/previous_observations}}

AVAILABLE TOOLS:
{{tools}}

{{@step_format}}

Now, reason through your next action:""",
        variables=[
            PromptVariable(name="goal", description="The overall goal", required=True),
            PromptVariable(name="plan", description="The current plan", required=True),
            PromptVariable(
                name="current_step", description="Current step being executed", required=True
            ),
            PromptVariable(
                name="previous_observations", description="Previous action results", required=False
            ),
            PromptVariable(name="tools", description="Available tools", required=True),
        ],
    )
    count += 1

    # =========================================================================
    # REFLECTION TEMPLATE
    # =========================================================================

    reflection_template = registry.create_template(
        template_id="reflection_prompt",
        name="Reflection Prompt",
        category=PromptCategory.REFLECTION,
        description="Self-evaluation and learning prompt",
        tags=["cognitive", "reflection", "learning"],
        initial_content="""You are a critical evaluation agent. Your task is to reflect on the last action taken and assess progress toward the goal.

ORIGINAL GOAL: {{goal}}

CURRENT PLAN:
{{plan}}

LAST ACTION TAKEN:
Tool: {{last_action_name}}
Arguments: {{last_action_arguments}}

OBSERVATION FROM LAST ACTION:
{{last_observation}}

REFLECTION INSTRUCTIONS:
Critically evaluate the last action and its results. Consider:
1. Did the action successfully complete the current plan step?
2. Does the observation reveal new information that changes our understanding?
3. Should the plan be modified, reordered, or updated?
4. Are we making progress toward the goal?
5. What key insights can we extract from this iteration?

Provide your reflection in this format:
EVALUATION: [Success/Partial/Failed] - Brief assessment
PROGRESS: [0-100]% - Estimated progress toward goal
INSIGHTS: [List key learnings from this iteration]
PLAN_UPDATE: [Yes/No] - Whether the plan needs updating
{{#plan_update}}
UPDATED_PLAN: [New plan if needed]
{{/plan_update}}""",
        variables=[
            PromptVariable(name="goal", description="Original goal", required=True),
            PromptVariable(name="plan", description="Current plan", required=True),
            PromptVariable(name="last_action_name", description="Last tool used", required=True),
            PromptVariable(
                name="last_action_arguments", description="Last tool arguments", required=True
            ),
            PromptVariable(
                name="last_observation", description="Result of last action", required=True
            ),
        ],
    )
    count += 1

    # =========================================================================
    # VALIDATION TEMPLATE
    # =========================================================================

    validation_template = registry.create_template(
        template_id="validation_prompt",
        name="Validation Prompt",
        category=PromptCategory.REASONING,
        description="Plan validation and safety check prompt",
        tags=["cognitive", "validation", "safety"],
        initial_content="""You are a validation agent. Your task is to verify that the proposed plan is safe, feasible, and aligned with the goal.

GOAL: {{goal}}

PROPOSED PLAN:
{{plan}}

AVAILABLE TOOLS:
{{tools}}

VALIDATION CRITERIA:
1. FEASIBILITY: Can each step be executed with available tools?
2. COMPLETENESS: Does the plan address all aspects of the goal?
3. SAFETY: Are there any potentially harmful actions?
4. EFFICIENCY: Is the plan reasonably efficient?
5. CLARITY: Are the steps clear and unambiguous?

Provide your validation in this format:
VALID: [Yes/No]
CONFIDENCE: [High/Medium/Low]
ISSUES: [List any issues found, or "None"]
SUGGESTIONS: [Improvements if any, or "None"]""",
        variables=[
            PromptVariable(name="goal", description="The goal to validate against", required=True),
            PromptVariable(name="plan", description="The plan to validate", required=True),
            PromptVariable(name="tools", description="Available tools", required=True),
        ],
    )
    count += 1

    logger.info(f"Loaded {count} default templates and snippets")
    return count


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TemplateLoader",
    "load_default_templates",
]
