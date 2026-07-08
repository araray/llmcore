# src/llmcore/agents/cognitive/phases/plan.py
"""
PLAN Phase Implementation.

The PLAN phase creates a strategic decomposition of the goal into actionable
steps. It uses the prompt library system to generate high-quality plans using
versioned, optimized prompts.

Key Features:
- Uses prompt templates from the prompt library
- Supports plan refinement (updating existing plans)
- Identifies risks and estimates iterations
- Structured output parsing

References:
    - Technical Spec: Section 5.3.2 (PLAN Phase)
    - Dossier: Step 2.5 (Cognitive Phases - PLAN)
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from ..models import EnhancedAgentState, PlanInput, PlanOutput
from ._prompting import messages_from_registry, record_template_use, require_prompt_registry

if TYPE_CHECKING:
    from ....models import Message
    from ....providers.manager import ProviderManager

logger = logging.getLogger(__name__)


# =============================================================================
# PLAN PHASE FUNCTION
# =============================================================================


async def plan_phase(
    agent_state: EnhancedAgentState,
    plan_input: PlanInput,
    provider_manager: "ProviderManager",
    prompt_registry: Any | None = None,  # PromptRegistry
    tracer: Any | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> PlanOutput:
    """
    Execute the PLAN phase of the cognitive cycle.

    Creates a strategic plan by:
    1. Loading the planning prompt template
    2. Rendering it with goal and context
    3. Calling the LLM to generate plan
    4. Parsing structured output
    5. Recording metrics

    Args:
        agent_state: Current enhanced agent state
        plan_input: Input configuration for planning
        provider_manager: Provider manager for LLM calls
        prompt_registry: Prompt registry (REQUIRED as of 0.52.0 — raises
            ValueError when None; a render failure aborts the phase)
        tracer: Optional OpenTelemetry tracer
        provider_name: Optional provider override
        model_name: Optional model override

    Returns:
        PlanOutput with generated plan and reasoning

    Example:
        >>> plan_input = PlanInput(
        ...     goal="Calculate factorial of 10",
        ...     context="User wants step-by-step breakdown"
        ... )
        >>>
        >>> output = await plan_phase(
        ...     agent_state=state,
        ...     plan_input=plan_input,
        ...     provider_manager=provider_manager,
        ...     prompt_registry=registry
        ... )
        >>>
        >>> print(f"Generated {len(output.plan_steps)} steps")
    """
    from ....tracing import add_span_attributes, create_span, record_span_exception

    with create_span(tracer, "cognitive.plan") as span:
        require_prompt_registry(prompt_registry, "PLAN")
        logger.debug("Starting PLAN phase")

        # 1. Render the PLAN messages (system + user) from the registry.
        #    Built BEFORE the LLM try/except: a broken template must abort
        #    the phase (fail-loud), never degrade it.
        messages = _generate_planning_messages(
            plan_input=plan_input, prompt_registry=prompt_registry
        )

        try:
            # 2. Call LLM
            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            if callable(getattr(type(provider_manager), "chat_completion_with_retry", None)):
                response = await provider_manager.chat_completion_with_retry(
                    provider,
                    context=messages,
                    model=target_model,
                    stream=False,
                    tracer=tracer,
                    operation="cognitive.plan",
                    temperature=0.7,  # Some creativity in planning
                )
            else:
                response = await provider.chat_completion(
                    context=messages,
                    model=target_model,
                    stream=False,
                    temperature=0.7,
                )

            # Extract response content from provider-specific response format
            response_content = provider.extract_response_content(response)
            usage = response.get("usage", {}) if isinstance(response, dict) else None
            total_tokens = usage.get("total_tokens") if usage else None

            # 3. Parse response
            output = _parse_plan_response(response_text=response_content, plan_input=plan_input)
            output.tokens_used = total_tokens

            # 4. Update agent state
            agent_state.plan = output.step_descriptions
            agent_state.metadata["plan_step_specs"] = [
                step.model_dump(exclude_none=True) for step in output.plan_steps
            ]
            agent_state.plan_steps_status = ["pending"] * len(output.plan_steps)
            agent_state.current_plan_step_index = 0
            agent_state.plan_created_at = output.created_at
            agent_state.plan_version += 1

            # 5. Record prompt usage metrics (best-effort)
            record_template_use(
                prompt_registry,
                "planning_prompt",
                success=len(output.plan_steps) > 0,
                tokens=total_tokens,
            )

            # 6. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "plan.steps_count": len(output.plan_steps),
                        "plan.has_risks": len(output.risks_identified) > 0,
                        "plan.provider": provider.get_name(),
                        "plan.model": target_model,
                    },
                )

            logger.info(
                f"PLAN phase complete: {len(output.plan_steps)} steps, "
                f"{len(output.risks_identified)} risks identified"
            )

            return output

        except Exception as e:
            logger.error(f"PLAN phase failed: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

            # Return minimal plan on error
            return PlanOutput(
                plan_steps=["Analyze the problem", "Execute solution"],
                reasoning=f"Fallback plan due to error: {e!s}",
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_planning_messages(
    plan_input: PlanInput, prompt_registry: Any
) -> list["Message"]:
    """
    Render the PLAN phase messages (system + user) from the prompt registry.

    Rendering errors propagate — a broken template aborts the phase instead
    of degrading it (0.52.0 control plane, no silent fallback).

    Args:
        plan_input: Planning input configuration
        prompt_registry: Prompt registry (grimoire adapter)

    Returns:
        Role-structured messages for the LLM call
    """
    existing_plan_section = ""
    if plan_input.existing_plan:
        existing_plan_section = (
            "\n\nEXISTING PLAN:\n"
            + "".join(f"{i}. {step}\n" for i, step in enumerate(plan_input.existing_plan, 1))
            + "\nRefine or update this plan as needed."
        )

    return messages_from_registry(
        prompt_registry,
        "planning_prompt",
        {
            "goal": plan_input.goal,
            "context": plan_input.context or "",
            "constraints": plan_input.constraints or "",
            "existing_plan_section": existing_plan_section,
        },
    )


def _parse_plan_response(response_text: str, plan_input: PlanInput) -> PlanOutput:
    """
    Parse the LLM response into structured PlanOutput.

    Args:
        response_text: Raw LLM response
        plan_input: Original input (for context)

    Returns:
        Parsed PlanOutput
    """
    plan_steps = []
    reasoning = ""
    risks = []

    # Extract PLAN section
    plan_match = re.search(
        r"PLAN:\s*\n((?:\d+\.\s*.+\n?)+)", response_text, re.MULTILINE | re.IGNORECASE
    )

    if plan_match:
        plan_text = plan_match.group(1)
        # Extract numbered steps
        step_matches = re.findall(r"\d+\.\s*(.+)", plan_text)
        plan_steps = [step.strip() for step in step_matches if step.strip()]

    # Extract REASONING section
    reasoning_match = re.search(
        r"REASONING:\s*\n(.+?)(?=\n(?:RISKS|$))", response_text, re.DOTALL | re.IGNORECASE
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract RISKS section
    risks_match = re.search(
        r"RISKS:\s*\n((?:-\s*.+\n?)+)", response_text, re.MULTILINE | re.IGNORECASE
    )

    if risks_match:
        risks_text = risks_match.group(1)
        risk_matches = re.findall(r"-\s*(.+)", risks_text)
        risks = [risk.strip() for risk in risk_matches if risk.strip()]

    # Fallback if parsing failed
    if not plan_steps:
        logger.warning("Failed to parse plan steps, extracting from full response")
        # Try to extract any numbered list
        step_matches = re.findall(r"^\d+\.\s*(.+)$", response_text, re.MULTILINE)
        if step_matches:
            plan_steps = [step.strip() for step in step_matches]
        else:
            # Last resort: create single step
            plan_steps = [plan_input.goal]

    # Estimate iterations (rough heuristic: 1-2 iterations per step)
    estimated_iterations = len(plan_steps) * 2 if plan_steps else None

    return PlanOutput(
        plan_steps=plan_steps,
        reasoning=reasoning or "Plan created to achieve the goal.",
        estimated_iterations=estimated_iterations,
        risks_identified=risks,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["plan_phase"]
