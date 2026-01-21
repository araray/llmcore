# src/llmcore/agents/cognitive/phases/reflect.py
"""
REFLECT Phase Implementation.

The REFLECT phase is where the agent evaluates its progress, learns from
experiences, and decides whether to update its plan. It provides:
- Self-evaluation of action effectiveness
- Progress estimation toward goal
- Key learning extraction
- Plan update recommendations
- Next focus determination

This phase enables the agent to adapt and improve its strategy based on
observed outcomes.

References:
    - Technical Spec: Section 5.3.7 (REFLECT Phase)
    - Dossier: Step 2.7 (Cognitive Phases - REFLECT)
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..models import EnhancedAgentState, ReflectInput, ReflectOutput

if TYPE_CHECKING:
    from ....models import Message, Role
    from ....providers.manager import ProviderManager

logger = logging.getLogger(__name__)


# =============================================================================
# REFLECT PHASE FUNCTION
# =============================================================================


async def reflect_phase(
    agent_state: EnhancedAgentState,
    reflect_input: ReflectInput,
    provider_manager: "ProviderManager",
    prompt_registry: Optional[Any] = None,  # PromptRegistry
    tracer: Optional[Any] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ReflectOutput:
    """
    Execute the REFLECT phase of the cognitive cycle.

    Reflects on the iteration by:
    1. Loading reflection prompt template
    2. Calling LLM for self-evaluation
    3. Parsing evaluation and insights
    4. Determining progress estimate
    5. Deciding if plan needs update
    6. Recording metrics

    Args:
        agent_state: Current enhanced agent state
        reflect_input: Input with action and observation
        provider_manager: Provider manager for LLM calls
        prompt_registry: Optional prompt registry
        tracer: Optional OpenTelemetry tracer
        provider_name: Optional provider override
        model_name: Optional model override

    Returns:
        ReflectOutput with evaluation and recommendations

    Example:
        >>> reflect_input = ReflectInput(
        ...     goal="Calculate factorial",
        ...     plan=["Step 1", "Step 2"],
        ...     current_step_index=0,
        ...     last_action=tool_call,
        ...     observation="Calculation succeeded: result is 3628800",
        ...     iteration_number=1
        ... )
        >>>
        >>> output = await reflect_phase(
        ...     agent_state=state,
        ...     reflect_input=reflect_input,
        ...     provider_manager=provider_manager
        ... )
        >>>
        >>> print(f"Progress: {output.progress_estimate:.1%}")
        >>> print(f"Update plan: {output.plan_needs_update}")
    """
    from ....models import Message, Role
    from ....tracing import add_span_attributes, create_span, record_span_exception

    with create_span(tracer, "cognitive.reflect") as span:
        try:
            logger.debug(f"Starting REFLECT phase (iteration {reflect_input.iteration_number})")

            # 1. Generate reflection prompt
            reflection_prompt = _generate_reflection_prompt(
                reflect_input=reflect_input, prompt_registry=prompt_registry
            )

            # 2. Call LLM for reflection
            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are a reflective AI agent. Honestly evaluate your actions, "
                    "identify learnings, and recommend improvements.",
                ),
                Message(role=Role.USER, content=reflection_prompt),
            ]

            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False,
                temperature=0.7,  # Some creativity in reflection
            )

            # Extract response content
            response_content = provider.extract_response_content(response)

            # 3. Parse reflection response
            output = _parse_reflection_response(
                response_text=response_content, reflect_input=reflect_input
            )

            # 4. Update agent state progress
            agent_state.progress_estimate = output.progress_estimate

            # 5. Record metrics
            if prompt_registry and hasattr(prompt_registry, "record_use"):
                try:
                    template = prompt_registry.get_template("reflection_prompt")
                    if template.active_version:
                        # Extract token usage from response dict
                        usage = response.get("usage", {}) if isinstance(response, dict) else None
                        total_tokens = usage.get("total_tokens") if usage else None
                        prompt_registry.record_use(
                            version_id=template.active_version.id,
                            success=True,  # Reflection always "succeeds"
                            tokens=total_tokens,
                        )
                except Exception as e:
                    logger.warning(f"Failed to record prompt metrics: {e}")

            # 6. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "reflect.iteration": reflect_input.iteration_number,
                        "reflect.progress": output.progress_estimate,
                        "reflect.plan_update": output.plan_needs_update,
                        "reflect.step_completed": output.step_completed,
                        "reflect.insights_count": len(output.insights),
                        "reflect.provider": provider.get_name(),
                        "reflect.model": target_model,
                    },
                )

            logger.info(
                f"REFLECT phase complete: progress={output.progress_estimate:.1%}, "
                f"plan_update={output.plan_needs_update}, step_done={output.step_completed}"
            )

            return output

        except Exception as e:
            logger.error(f"REFLECT phase failed: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

            # Return minimal reflection on error
            return ReflectOutput(
                evaluation=f"Reflection error: {str(e)}",
                progress_estimate=agent_state.progress_estimate,  # Keep current
                insights=[],
                plan_needs_update=False,
                step_completed=False,
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_reflection_prompt(reflect_input: ReflectInput, prompt_registry: Optional[Any]) -> str:
    """
    Generate the reflection prompt using prompt library or fallback.

    Args:
        reflect_input: Reflection input
        prompt_registry: Optional prompt registry

    Returns:
        Formatted reflection prompt
    """
    # Format plan
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(reflect_input.plan))

    # Format action - guard against None last_action
    if reflect_input.last_action:
        action_str = f"{reflect_input.last_action.name}({reflect_input.last_action.arguments})"
    else:
        # No action was proposed (THINK phase may have failed)
        action_str = "(no action proposed)"
        logger.warning("REFLECT: No last_action available - THINK phase may have failed")

    # Try to use prompt library
    if prompt_registry:
        try:
            return prompt_registry.render(
                template_id="reflection_prompt",
                variables={
                    "goal": reflect_input.goal,
                    "plan": plan_str,
                    "last_action": action_str,
                    "observation": reflect_input.observation,
                    "iteration": str(reflect_input.iteration_number),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to use prompt registry: {e}, falling back")

    # Fallback prompt
    current_step = (
        reflect_input.plan[reflect_input.current_step_index]
        if reflect_input.current_step_index < len(reflect_input.plan)
        else "Unknown"
    )

    prompt = f"""Reflect on your recent action and its outcome.

ORIGINAL GOAL:
{reflect_input.goal}

CURRENT PLAN:
{plan_str}

CURRENT STEP: {reflect_input.current_step_index + 1}. {current_step}

LAST ACTION:
{action_str}

OBSERVATION:
{reflect_input.observation}

ITERATION: {reflect_input.iteration_number}

REFLECTION QUESTIONS:
1. Did the action produce the expected result?
2. Are we making progress toward the goal?
3. Should we continue with the current plan or adjust?
4. What have we learned that could help in future iterations?
5. Is the current step complete?

PROVIDE:
- EVALUATION: Assess the action's effectiveness (success/partial/failure)
- PROGRESS: Estimate overall progress toward goal (0-100%)
- INSIGHTS: Key learnings from this iteration
- PLAN_UPDATE: Whether plan needs modification (yes/no)
- STEP_COMPLETED: Is current step done (yes/no)
- NEXT_FOCUS: What to prioritize in the next iteration

Be honest and critical in your self-assessment.
"""

    return prompt


def _parse_reflection_response(response_text: str, reflect_input: ReflectInput) -> ReflectOutput:
    """
    Parse the LLM reflection response.

    Args:
        response_text: Raw LLM response
        reflect_input: Original reflection input

    Returns:
        Parsed ReflectOutput
    """
    # Extract evaluation
    evaluation_match = re.search(
        r"EVALUATION:\s*(.+?)(?=\n(?:PROGRESS|$))", response_text, re.DOTALL | re.IGNORECASE
    )

    evaluation = "Action completed."
    if evaluation_match:
        evaluation = evaluation_match.group(1).strip()

    # Extract progress percentage
    progress_match = re.search(r"PROGRESS:\s*(\d+)%?", response_text, re.IGNORECASE)

    progress_estimate = 0.5  # Default to 50%
    if progress_match:
        progress_pct = int(progress_match.group(1))
        progress_estimate = min(max(progress_pct / 100.0, 0.0), 1.0)

    # Extract insights
    insights = []
    insights_match = re.search(
        r"INSIGHTS:\s*(.+?)(?=\n(?:PLAN_UPDATE|$))", response_text, re.DOTALL | re.IGNORECASE
    )

    if insights_match:
        insights_text = insights_match.group(1).strip()
        insight_items = re.findall(r"[-•]\s*(.+)", insights_text)
        if insight_items:
            insights = [i.strip() for i in insight_items]
        elif insights_text:
            insights = [insights_text]

    # Check if plan needs update
    plan_update_match = re.search(r"PLAN_UPDATE:\s*(yes|no)", response_text, re.IGNORECASE)

    plan_needs_update = False
    updated_plan = None

    if plan_update_match and plan_update_match.group(1).lower() == "yes":
        plan_needs_update = True
        # Look for updated plan
        updated_plan_match = re.search(
            r"UPDATED PLAN:\s*\n((?:\d+\.\s*.+\n?)+)", response_text, re.MULTILINE | re.IGNORECASE
        )
        if updated_plan_match:
            plan_text = updated_plan_match.group(1)
            steps = re.findall(r"\d+\.\s*(.+)", plan_text)
            if steps:
                updated_plan = [s.strip() for s in steps]

    # Check if step is completed
    step_completed_match = re.search(r"STEP_COMPLETED:\s*(yes|no)", response_text, re.IGNORECASE)

    step_completed = False
    if step_completed_match and step_completed_match.group(1).lower() == "yes":
        step_completed = True

    # Extract next focus
    next_focus_match = re.search(r"NEXT_FOCUS:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE)

    next_focus = None
    if next_focus_match:
        next_focus = next_focus_match.group(1).strip()

    return ReflectOutput(
        evaluation=evaluation,
        progress_estimate=progress_estimate,
        insights=insights,
        plan_needs_update=plan_needs_update,
        updated_plan=updated_plan,
        step_completed=step_completed,
        next_focus=next_focus,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["reflect_phase"]
