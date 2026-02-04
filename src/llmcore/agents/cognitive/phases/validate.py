# src/llmcore/agents/cognitive/phases/validate.py
"""
VALIDATE Phase Implementation.

The VALIDATE phase verifies that a proposed action is safe, appropriate, and
likely to be effective before execution. It acts as a safety gate and can
trigger Human-in-the-Loop (HITL) approval for risky operations.

Key Features:
- Uses validation prompt templates
- Multi-criteria validation (safety, appropriateness, effectiveness)
- Confidence-based HITL triggers
- Dangerous pattern detection
- Structured validation output

References:
    - Technical Spec: Section 5.3.4 (VALIDATE Phase)
    - Dossier: Step 2.6 (Cognitive Phases - VALIDATE)
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from ..models import (
    ConfidenceLevel,
    EnhancedAgentState,
    ValidateInput,
    ValidateOutput,
    ValidationResult,
)

if TYPE_CHECKING:
    from ....providers.manager import ProviderManager

logger = logging.getLogger(__name__)


# Dangerous patterns that always require human approval
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",  # Recursive force delete from root
    r"DROP\s+DATABASE",  # Database deletion
    r"DELETE\s+FROM.*WHERE\s+1=1",  # Delete all rows
    r"sudo\s+",  # Sudo commands (if not in whitelist)
    r"chmod\s+777",  # Overly permissive permissions
    r"eval\s*\(",  # Eval injection risk
    r"exec\s*\(",  # Exec injection risk
]


# =============================================================================
# VALIDATE PHASE FUNCTION
# =============================================================================


async def validate_phase(
    agent_state: EnhancedAgentState,
    validate_input: ValidateInput,
    provider_manager: "ProviderManager",
    prompt_registry: Any | None = None,  # PromptRegistry
    tracer: Any | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> ValidateOutput:
    """
    Execute the VALIDATE phase of the cognitive cycle.

    Validates a proposed action by:
    1. Checking for dangerous patterns
    2. Loading validation prompt template
    3. Calling LLM for multi-criteria validation
    4. Parsing validation decision
    5. Determining if HITL is needed
    6. Recording metrics

    Args:
        agent_state: Current enhanced agent state
        validate_input: Input with action to validate
        provider_manager: Provider manager for LLM calls
        prompt_registry: Optional prompt registry
        tracer: Optional OpenTelemetry tracer
        provider_name: Optional provider override
        model_name: Optional model override

    Returns:
        ValidateOutput with validation decision

    Example:
        >>> validate_input = ValidateInput(
        ...     goal="Process files",
        ...     proposed_action=ToolCall(name="execute_shell", arguments={"command": "ls"}),
        ...     reasoning="Need to see directory contents",
        ...     risk_tolerance="medium"
        ... )
        >>>
        >>> output = await validate_phase(
        ...     agent_state=state,
        ...     validate_input=validate_input,
        ...     provider_manager=provider_manager
        ... )
        >>>
        >>> if output.result == ValidationResult.APPROVED:
        ...     # Proceed with action
    """
    from ....models import Message, Role
    from ....tracing import add_span_attributes, create_span, record_span_exception

    with create_span(tracer, "cognitive.validate") as span:
        try:
            logger.debug(
                f"Starting VALIDATE phase for action: {validate_input.proposed_action.name}"
            )

            # 1. Check for dangerous patterns first
            dangerous_check = _check_dangerous_patterns(validate_input.proposed_action)
            if dangerous_check:
                logger.warning(f"Dangerous pattern detected: {dangerous_check}")
                return ValidateOutput(
                    result=ValidationResult.REQUIRES_HUMAN_APPROVAL,
                    confidence=ConfidenceLevel.HIGH,
                    concerns=[f"Dangerous pattern detected: {dangerous_check}"],
                    suggestions=["Request human approval before executing"],
                    requires_human_approval=True,
                    approval_prompt=_generate_approval_prompt(validate_input, dangerous_check),
                )

            # 2. Generate validation prompt
            validation_prompt = _generate_validation_prompt(
                validate_input=validate_input, prompt_registry=prompt_registry
            )

            # 3. Call LLM for validation
            provider = provider_manager.get_provider(provider_name)
            target_model = model_name or provider.default_model

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are a safety validation agent. Carefully evaluate proposed actions "
                    "for safety, appropriateness, and effectiveness.",
                ),
                Message(role=Role.USER, content=validation_prompt),
            ]

            response = await provider.chat_completion(
                context=messages,
                model=target_model,
                stream=False,
                temperature=0.3,  # Lower temperature for consistent validation
            )

            # Extract response content
            response_content = provider.extract_response_content(response)

            # 4. Parse validation response
            output = _parse_validation_response(
                response_text=response_content, validate_input=validate_input
            )

            # 5. Update agent state
            agent_state.pending_validation = validate_input
            agent_state.validation_history.append(output)

            if output.requires_human_approval:
                agent_state.awaiting_human_approval = True
                agent_state.pending_approval_prompt = output.approval_prompt

            # 6. Record metrics
            if prompt_registry and hasattr(prompt_registry, "record_use"):
                try:
                    template = prompt_registry.get_template("validation_prompt")
                    if template.active_version:
                        # Extract token usage from response dict
                        usage = response.get("usage", {}) if isinstance(response, dict) else None
                        total_tokens = usage.get("total_tokens") if usage else None
                        prompt_registry.record_use(
                            version_id=template.active_version.id,
                            success=output.result != ValidationResult.REJECTED,
                            tokens=total_tokens,
                        )
                except Exception as e:
                    logger.warning(f"Failed to record prompt metrics: {e}")

            # 7. Add tracing
            if span:
                add_span_attributes(
                    span,
                    {
                        "validate.result": output.result.value,
                        "validate.confidence": output.confidence.value,
                        "validate.requires_hitl": output.requires_human_approval,
                        "validate.concerns_count": len(output.concerns),
                        "validate.provider": provider.get_name(),
                        "validate.model": target_model,
                    },
                )

            logger.info(
                f"VALIDATE phase complete: {output.result.value}, "
                f"confidence={output.confidence.value}, "
                f"HITL={output.requires_human_approval}"
            )

            return output

        except Exception as e:
            logger.error(f"VALIDATE phase failed: {e}", exc_info=True)
            if span:
                record_span_exception(span, e)

            # On error, be conservative: require approval
            return ValidateOutput(
                result=ValidationResult.REQUIRES_HUMAN_APPROVAL,
                confidence=ConfidenceLevel.LOW,
                concerns=[f"Validation error: {e!s}"],
                suggestions=["Manual review recommended"],
                requires_human_approval=True,
                approval_prompt=f"Validation failed with error. Review action: {validate_input.proposed_action.name}",
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _check_dangerous_patterns(tool_call: Any) -> str | None:
    """
    Check if the tool call contains dangerous patterns.

    Args:
        tool_call: The tool call to check

    Returns:
        Description of dangerous pattern if found, None otherwise
    """
    # Convert arguments to string for pattern matching
    args_str = str(tool_call.arguments)

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, args_str, re.IGNORECASE):
            return pattern

    return None


def _generate_validation_prompt(
    validate_input: ValidateInput, prompt_registry: Any | None
) -> str:
    """
    Generate the validation prompt using prompt library or fallback.

    Args:
        validate_input: Validation input
        prompt_registry: Optional prompt registry

    Returns:
        Formatted validation prompt
    """
    # Format proposed action
    action_str = (
        f"{validate_input.proposed_action.name}({validate_input.proposed_action.arguments})"
    )

    # Try to use prompt library
    if prompt_registry:
        try:
            return prompt_registry.render(
                template_id="validation_prompt",
                variables={
                    "goal": validate_input.goal,
                    "proposed_action": action_str,
                    "reasoning": validate_input.reasoning,
                    "risk_tolerance": validate_input.risk_tolerance,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to use prompt registry: {e}, falling back")

    # Fallback prompt
    prompt = f"""Validate the following proposed action:

OBJECTIVE: {validate_input.goal}

PROPOSED ACTION: {action_str}

REASONING: {validate_input.reasoning}

RISK TOLERANCE: {validate_input.risk_tolerance}

VALIDATION CRITERIA:
1. SAFETY: Is this action safe to execute?
   - Will it harm systems or data?
   - Are there potential side effects?

2. APPROPRIATENESS: Is this action suitable for the goal?
   - Does it align with the objective?
   - Is it the right tool for the task?

3. EFFECTIVENESS: Is this action likely to succeed?
   - Do we have necessary context/data?
   - Are the parameters correct?

4. REVERSIBILITY: Can we undo this if needed?

Provide your assessment:
APPROVED: yes/no
CONFIDENCE: low/medium/high
CONCERNS: [list any issues]
SUGGESTIONS: [improvements if needed]

If confidence is LOW or concerns are CRITICAL, recommend human approval.
"""

    return prompt


def _parse_validation_response(response_text: str, validate_input: ValidateInput) -> ValidateOutput:
    """
    Parse the LLM validation response.

    Args:
        response_text: Raw LLM response
        validate_input: Original validation input

    Returns:
        Parsed ValidateOutput
    """
    # Extract approval decision
    approved_match = re.search(r"APPROVED:\s*(yes|no)", response_text, re.IGNORECASE)

    approved = False
    if approved_match:
        approved = approved_match.group(1).lower() == "yes"

    # Extract confidence
    confidence_match = re.search(r"CONFIDENCE:\s*(low|medium|high)", response_text, re.IGNORECASE)

    confidence = ConfidenceLevel.MEDIUM
    if confidence_match:
        conf_str = confidence_match.group(1).lower()
        confidence = ConfidenceLevel(conf_str)

    # Extract concerns
    concerns = []
    concerns_match = re.search(
        r"CONCERNS:\s*(.+?)(?=\n(?:SUGGESTIONS|$))", response_text, re.DOTALL | re.IGNORECASE
    )

    if concerns_match:
        concerns_text = concerns_match.group(1).strip()
        # Parse list items
        concern_items = re.findall(r"[-•]\s*(.+)", concerns_text)
        if concern_items:
            concerns = [c.strip() for c in concern_items]
        elif concerns_text and concerns_text.lower() != "none":
            concerns = [concerns_text]

    # Extract suggestions
    suggestions = []
    suggestions_match = re.search(r"SUGGESTIONS:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE)

    if suggestions_match:
        suggestions_text = suggestions_match.group(1).strip()
        suggestion_items = re.findall(r"[-•]\s*(.+)", suggestions_text)
        if suggestion_items:
            suggestions = [s.strip() for s in suggestion_items]
        elif suggestions_text and suggestions_text.lower() != "none":
            suggestions = [suggestions_text]

    # Determine validation result
    requires_hitl = False

    # Check if human approval is explicitly mentioned
    if re.search(r"human\s+approval", response_text, re.IGNORECASE):
        requires_hitl = True

    # Low confidence always requires approval
    if confidence == ConfidenceLevel.LOW:
        requires_hitl = True

    # Critical concerns require approval
    critical_keywords = ["critical", "dangerous", "destructive", "irreversible"]
    for concern in concerns:
        if any(keyword in concern.lower() for keyword in critical_keywords):
            requires_hitl = True
            break

    # Determine final result
    if not approved:
        result = ValidationResult.REJECTED
    elif requires_hitl:
        result = ValidationResult.REQUIRES_HUMAN_APPROVAL
    else:
        result = ValidationResult.APPROVED

    # Generate approval prompt if needed
    approval_prompt = None
    if requires_hitl:
        approval_prompt = _generate_approval_prompt(validate_input, ", ".join(concerns))

    return ValidateOutput(
        result=result,
        confidence=confidence,
        concerns=concerns,
        suggestions=suggestions,
        requires_human_approval=requires_hitl,
        approval_prompt=approval_prompt,
    )


def _generate_approval_prompt(validate_input: ValidateInput, reason: str) -> str:
    """
    Generate a prompt for human approval.

    Args:
        validate_input: Validation input
        reason: Reason for requiring approval

    Returns:
        Human-readable approval prompt
    """
    action_str = (
        f"{validate_input.proposed_action.name}({validate_input.proposed_action.arguments})"
    )

    return f"""Human approval required for the following action:

Goal: {validate_input.goal}

Proposed Action: {action_str}

Reasoning: {validate_input.reasoning}

Reason for Approval: {reason}

Do you approve this action? (yes/no)
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["validate_phase"]
