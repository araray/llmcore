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
from typing import TYPE_CHECKING, Any

from ..models import EnhancedAgentState, ReflectInput, ReflectOutput

if TYPE_CHECKING:
    from ....providers.manager import ProviderManager

logger = logging.getLogger(__name__)


# =============================================================================
# REFLECT PHASE FUNCTION
# =============================================================================


async def reflect_phase(
    agent_state: EnhancedAgentState,
    reflect_input: ReflectInput,
    provider_manager: "ProviderManager",
    prompt_registry: Any | None = None,  # PromptRegistry
    tracer: Any | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
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
                evaluation=f"Reflection error: {e!s}",
                progress_estimate=agent_state.progress_estimate,  # Keep current
                insights=[],
                plan_needs_update=False,
                step_completed=False,
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_reflection_prompt(reflect_input: ReflectInput, prompt_registry: Any | None) -> str:
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

    # Extract progress percentage - try multiple patterns
    progress_estimate = _extract_progress(response_text)

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


def _extract_progress(text: str) -> float:
    """
    Extract progress percentage from reflection text using multiple strategies.

    Tries multiple patterns in order of specificity:
    1. Labeled format: "PROGRESS: 75%"
    2. XML format: "<progress>75</progress>"
    3. Natural language: "75% complete", "progress is 75%"
    4. Fraction format: "3/4 complete", "step 3 of 4"
    5. Content estimation based on keywords

    Args:
        text: Reflection text to parse

    Returns:
        Progress value between 0.0 and 1.0
    """
    # Pattern 1: Labeled format (most reliable)
    labeled_patterns = [
        r"PROGRESS:\s*(\d+(?:\.\d+)?)\s*%",
        r"PROGRESS:\s*(\d+(?:\.\d+)?)\s*(?:percent)?",
        r"progress[:\s]+(\d+(?:\.\d+)?)\s*%",
        r"progress[:\s]+(\d+(?:\.\d+)?)\s*(?:percent)?",
    ]

    for pattern in labeled_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Normalize to 0-1 range
            if value > 1:
                value = value / 100.0
            logger.debug(f"Extracted progress {value:.2f} using labeled pattern")
            return max(0.0, min(1.0, value))

    # Pattern 2: XML format
    xml_match = re.search(r"<progress>\s*(\d+(?:\.\d+)?)\s*</progress>", text, re.IGNORECASE)
    if xml_match:
        value = float(xml_match.group(1))
        if value > 1:
            value = value / 100.0
        logger.debug(f"Extracted progress {value:.2f} using XML pattern")
        return max(0.0, min(1.0, value))

    # Pattern 3: Natural language percentages
    natural_patterns = [
        r"(\d+(?:\.\d+)?)\s*%\s*(?:complete|done|finished|through|progress)",
        r"(?:complete|done|finished|through|progress)\s*(?:is\s+)?(\d+(?:\.\d+)?)\s*%",
        r"(?:approximately|about|around|roughly)\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+the\s+way|there)",
    ]

    for pattern in natural_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value > 1:
                value = value / 100.0
            logger.debug(f"Extracted progress {value:.2f} using natural language pattern")
            return max(0.0, min(1.0, value))

    # Pattern 4: Fraction format (step X of Y)
    fraction_patterns = [
        r"step\s*(\d+)\s*(?:of|/)\s*(\d+)",
        r"(\d+)\s*(?:of|/)\s*(\d+)\s*(?:steps?|tasks?|items?)",
        r"completed\s*(\d+)\s*(?:of|/)\s*(\d+)",
    ]

    for pattern in fraction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                value = current / total
                logger.debug(
                    f"Extracted progress {value:.2f} using fraction pattern ({current}/{total})"
                )
                return max(0.0, min(1.0, value))

    # Pattern 5: Content-based estimation
    return _estimate_progress_from_content(text)


def _estimate_progress_from_content(text: str) -> float:
    """
    Estimate progress when explicit value not found.

    Analyzes text for completion indicators using context-aware pattern
    matching. This function avoids false positives by:
    1. Checking for negative context patterns first
    2. Using regex with word boundaries
    3. Prioritizing explicit negative indicators over implicit positive ones

    Args:
        text: Text to analyze

    Returns:
        Estimated progress between 0.0 and 1.0

    Examples:
        >>> _estimate_progress_from_content("Task completed successfully")
        0.85
        >>> _estimate_progress_from_content("STEP_COMPLETED: No")
        0.35  # Negative context detected, not 0.85
        >>> _estimate_progress_from_content("Task is not completed yet")
        0.35  # Negative context detected
    """
    text_lower = text.lower()

    # =================================================================
    # STRATEGY 1: Check for EXPLICIT NEGATIVE INDICATORS first
    # These patterns indicate the task/step is explicitly NOT done
    # =================================================================
    negative_context_patterns = [
        # STEP_COMPLETED: No / step_completed: false patterns
        r"step_completed[:\s]*(no|false)",
        r"step[_\s]completed[:\s]*(no|false)",
        # Task/step is not done patterns
        r"(?:task|step|action)\s+(?:is\s+)?not\s+(?:completed?|finished|done)",
        r"not\s+(?:yet\s+)?(?:completed?|finished|done)",
        r"hasn't\s+(?:been\s+)?(?:completed?|finished)",
        r"have\s+not\s+(?:completed?|finished)",
        # Incomplete patterns
        r"\bincomplete\b",
        r"still\s+(?:working|in\s+progress|pending|ongoing)",
        r"needs?\s+(?:more\s+)?(?:work|steps?|actions?)",
        r"(?:more|additional)\s+(?:work|steps?|actions?)\s+(?:needed|required)",
        # Continuation indicators
        r"continue\s+(?:with|to|working)",
        r"proceeding\s+(?:to|with)",
        r"moving\s+(?:on|forward)\s+to",
    ]

    for pattern in negative_context_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected negative context via pattern: '{pattern}'")
            # Return moderate progress - task is in progress but not done
            return 0.35

    # =================================================================
    # STRATEGY 2: Check for BLOCKED/ERROR indicators (very low progress)
    # =================================================================
    blocked_patterns = [
        r"\bblocked\b",
        r"\bstuck\b",
        r"cannot\s+proceed",
        r"(?:error|errors?)\s+(?:occurred|encountered|found)",
        r"\bfailed\b",
        r"no\s+progress",
        r"unable\s+to\s+(?:proceed|continue|complete)",
        r"cannot\s+(?:complete|finish|proceed)",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected blocked state via pattern: '{pattern}'")
            return 0.05

    # =================================================================
    # STRATEGY 3: Check for HIGH progress indicators FIRST
    # High progress should be detected before medium patterns like "working on"
    # to avoid false negatives when text contains both.
    # =================================================================
    high_progress_patterns = [
        # Explicit completion patterns
        r"\btask\s+(?:is\s+)?(?:finally\s+)?completed?\b",
        r"\bstep\s+(?:is\s+)?completed?\b",
        r"\bsuccessfully\s+completed?\b",
        r"\bcompleted\s+successfully\b",
        r"\bfinished\s+(?:successfully|the\s+task)\b",
        r"\bdone\s+(?:successfully|with\s+the\s+task)\b",
        r"\bachieved\s+(?:the\s+|our\s+)?goal\b",
        r"\bfinal\s+step\s+(?:completed?|done)\b",
        r"\bwrapping\s+up\b",
        r"\ball\s+(?:steps?|tasks?)\s+(?:completed?|done|finished)\b",
        r"\bfinally\s+completed\b",
        # STEP_COMPLETED: Yes / step_completed: true
        r"step_completed[:\s]*(yes|true)",
        r"step[_\s]completed[:\s]*(yes|true)",
    ]

    for pattern in high_progress_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected high progress via pattern: '{pattern}'")
            return 0.85

    # =================================================================
    # STRATEGY 4: Check for MEDIUM-HIGH progress indicators
    # =================================================================
    medium_high_patterns = [
        r"good\s+progress",
        r"significant\s+progress",
        r"well\s+underway",
        r"most\s+(?:of\s+the\s+)?work\s+(?:done|completed?)",
        r"majority\s+(?:done|completed?)",
        r"halfway\s+(?:through|done|there)",
        r"nearly\s+(?:complete|done|finished|there)",
        r"almost\s+(?:complete|done|finished|there)",
    ]

    for pattern in medium_high_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected medium-high progress via pattern: '{pattern}'")
            return 0.65

    # =================================================================
    # STRATEGY 5: Check for LOW progress indicators
    # Check LOW before MEDIUM because "initial step in progress" should
    # be detected as early-stage (low) not mid-stage (medium).
    # =================================================================
    low_progress_patterns = [
        r"just\s+(?:started|beginning|began)",
        r"(?:first|initial)\s+step",
        r"starting\s+(?:out|up)",
        r"early\s+(?:stages?|phase)",
        r"getting\s+started",
        r"at\s+the\s+(?:beginning|start)",
    ]

    for pattern in low_progress_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected low progress via pattern: '{pattern}'")
            return 0.15

    # =================================================================
    # STRATEGY 6: Check for MEDIUM progress indicators
    # =================================================================
    medium_progress_patterns = [
        r"making\s+progress",
        r"some\s+progress",
        r"partial(?:ly)?\s+(?:complete|done)?",
        r"(?:in|currently\s+in)\s+progress",
        r"working\s+on\s+(?:it|this|the)",
        r"continuing\s+(?:with|to)",
        r"underway",
    ]

    for pattern in medium_progress_patterns:
        if re.search(pattern, text_lower):
            logger.debug(f"Detected medium progress via pattern: '{pattern}'")
            return 0.45

    # =================================================================
    # STRATEGY 7: Simple positive keywords (lower confidence)
    # Only use these if no other patterns matched
    # Use word boundaries to avoid partial matches
    # =================================================================
    simple_high_keywords = [
        r"\bcompleted\b",
        r"\bfinished\b",
        r"\bdone\b",
        r"\bachieved\b",
        r"\bsuccess(?:ful|fully)?\b",
    ]

    # Before using simple keywords, check there's no nearby negation
    # Look for "not", "no", "n't", "never", "without" within 5 words before the keyword
    negation_prefixes = [
        r"not\s+",
        r"no\s+",
        r"n't\s+",
        r"never\s+",
        r"without\s+",
        r"hasn't\s+",
        r"haven't\s+",
        r"isn't\s+",
        r"wasn't\s+",
        r"weren't\s+",
    ]

    for keyword_pattern in simple_high_keywords:
        match = re.search(keyword_pattern, text_lower)
        if match:
            # Check for negation within ~30 chars before the match
            start_pos = max(0, match.start() - 30)
            prefix_text = text_lower[start_pos : match.start()]

            has_negation = any(re.search(neg, prefix_text) for neg in negation_prefixes)
            if has_negation:
                logger.debug(
                    f"Keyword '{match.group()}' found but negated by prefix: '{prefix_text}'"
                )
                continue  # Skip this keyword, it's negated

            logger.debug(f"Detected high progress via simple keyword: '{match.group()}'")
            return 0.85

    # Default: moderate progress (encourages continuation)
    logger.debug("No progress patterns matched, using moderate default")
    return 0.35


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["reflect_phase"]
