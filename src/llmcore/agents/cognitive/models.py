# src/llmcore/agents/cognitive/models.py
"""
Enhanced cognitive cycle models for Darwin Layer 2.

This module defines the data models for the 8-phase enhanced cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

The models extend llmcore's existing AgentState with additional capabilities
for multi-phase reasoning, validation, and learning.

Design Principles:
    - Composability: Each phase has clear inputs and outputs
    - Traceability: Complete history of all iterations
    - Extensibility: Easy to add new phases or modify existing ones
    - Type Safety: Pydantic models for validation

References:
    - Technical Spec: Section 5.3 (Enhanced Cognitive Cycle)
    - Dossier: Step 2.4 (Cognitive Cycle Models)
"""

import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

# Import existing models from llmcore
from ...models import AgentState, ToolCall, ToolResult

STATE_SNAPSHOT_VERSION = "llmcore.enhanced_agent_state.v1"
_CONTEXT_COMPRESSION_KEY = "_context_compression"


def _truncate_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if max_chars < 1 or len(text) <= max_chars:
        return text, False
    suffix = "...[truncated]"
    keep = max(0, max_chars - len(suffix))
    return f"{text[:keep]}{suffix}", True


def _json_safe(
    value: Any,
    *,
    max_string_chars: int = 2000,
    max_items: int = 50,
    _depth: int = 0,
    _max_depth: int = 5,
) -> Any:
    if _depth >= _max_depth:
        return _truncate_text(value, max_string_chars)[0]

    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str):
            return _truncate_text(value, max_string_chars)[0]
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, BaseModel):
        return _json_safe(
            value.model_dump(mode="json"),
            max_string_chars=max_string_chars,
            max_items=max_items,
            _depth=_depth + 1,
            _max_depth=_max_depth,
        )

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            if index >= max_items:
                safe["__truncated_items__"] = len(items) - max_items
                break
            safe[str(key)] = _json_safe(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )
        return safe

    if isinstance(value, list | tuple | set):
        values = list(value)
        safe_list = [
            _json_safe(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )
            for item in values[:max_items]
        ]
        if len(values) > max_items:
            safe_list.append({"__truncated_items__": len(values) - max_items})
        return safe_list

    return _truncate_text(value, max_string_chars)[0]


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CognitivePhase(str, Enum):
    """Phases of the enhanced cognitive cycle."""

    PERCEIVE = "perceive"  # Gather and process inputs
    PLAN = "plan"  # Strategic decomposition
    THINK = "think"  # Reasoning about next action
    VALIDATE = "validate"  # Verify action safety
    ACT = "act"  # Execute chosen action
    OBSERVE = "observe"  # Process action results
    REFLECT = "reflect"  # Learn and evaluate
    UPDATE = "update"  # Update state and memory


class IterationStatus(str, Enum):
    """Status of a cognitive iteration."""

    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    INTERRUPTED = "interrupted"  # Interrupted by HITL or limit


class ValidationResult(str, Enum):
    """Result of action validation."""

    APPROVED = "approved"  # Action is safe to execute
    REJECTED = "rejected"  # Action should not be executed
    REQUIRES_HUMAN_APPROVAL = "requires_human_approval"  # Needs HITL


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions."""

    LOW = "low"  # 0-40%
    MEDIUM = "medium"  # 40-70%
    HIGH = "high"  # 70-100%


# =============================================================================
# PHASE INPUT/OUTPUT MODELS
# =============================================================================


class PerceiveInput(BaseModel):
    """Input to the PERCEIVE phase."""

    goal: str = Field(..., description="Current task goal")
    context_query: str | None = Field(
        default=None, description="Optional specific query for context retrieval"
    )
    force_refresh: bool = Field(
        default=False, description="Force refresh of context even if cached"
    )


class PerceiveOutput(BaseModel):
    """Output from the PERCEIVE phase."""

    retrieved_context: list[str] = Field(
        default_factory=list, description="Context items retrieved from memory"
    )
    working_memory_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of working memory state"
    )
    environmental_state: dict[str, Any] = Field(
        default_factory=dict, description="Current environmental state (sandbox, tools, etc.)"
    )
    perceived_at: datetime = Field(
        default_factory=datetime.utcnow, description="When perception occurred"
    )


class PlanInput(BaseModel):
    """Input to the PLAN phase."""

    goal: str = Field(..., description="The goal to plan for")
    context: str | None = Field(default=None, description="Relevant context for planning")
    constraints: str | None = Field(default=None, description="Constraints or requirements")
    existing_plan: list[str] | None = Field(default=None, description="Existing plan to refine")


class PlanStepSpec(BaseModel):
    """A structured plan step with optional tool intent.

    Legacy prose-only plans are represented by setting only ``description``.
    ``tool_name`` and ``input`` let future planners pre-commit a tool call
    without forcing the rest of the cognitive cycle to change shape.
    """

    index: int = Field(default=0, ge=0, description="0-based step index within the plan")
    description: str = Field(default="", description="Human-readable step description")
    tool_name: str | None = Field(default=None, description="Optional intended tool name")
    input: dict[str, Any] | None = Field(
        default=None, description="Optional structured tool input arguments"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Prior step indices this step depends on"
    )
    estimated_cost: float | None = Field(default=None, description="Optional planner cost estimate")

    def __str__(self) -> str:
        """Return the prose description for backward-compatible string use."""
        return self.description

    def __contains__(self, item: str) -> bool:
        """Allow legacy tests/callers to use substring checks on a step."""
        return item in self.description


class PlanOutput(BaseModel):
    """Output from the PLAN phase."""

    plan_steps: list[PlanStepSpec] = Field(
        default_factory=list, description="Ordered list of structured actionable steps"
    )
    reasoning: str = Field(default="", description="Strategic reasoning behind the plan")
    estimated_iterations: int | None = Field(
        default=None, description="Estimated iterations to complete plan"
    )
    risks_identified: list[str] = Field(
        default_factory=list, description="Potential risks or challenges identified"
    )
    tokens_used: int | None = Field(default=None, description="Provider tokens used")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_plan_steps(cls, data: Any) -> Any:
        """Accept legacy ``list[str]`` plans and structured dictionaries."""
        if not isinstance(data, dict):
            return data
        raw_steps = data.get("plan_steps")
        if raw_steps is None:
            return data
        if not isinstance(raw_steps, list):
            return data

        normalized_steps: list[Any] = []
        for index, step in enumerate(raw_steps):
            if isinstance(step, PlanStepSpec):
                normalized_steps.append(step)
            elif isinstance(step, str):
                normalized_steps.append({"index": index, "description": step})
            elif isinstance(step, dict):
                step_data = dict(step)
                step_data.setdefault("index", index)
                if "description" not in step_data and "step" in step_data:
                    step_data["description"] = str(step_data["step"])
                normalized_steps.append(step_data)
            else:
                normalized_steps.append({"index": index, "description": str(step)})

        output_data = dict(data)
        output_data["plan_steps"] = normalized_steps
        return output_data

    @property
    def step_descriptions(self) -> list[str]:
        """Return plan steps as plain descriptions for legacy state storage."""
        return [step.description for step in self.plan_steps]


class ThinkInput(BaseModel):
    """Input to the THINK phase."""

    goal: str = Field(..., description="Current objective")
    current_step: str = Field(..., description="Current plan step")
    current_step_spec: PlanStepSpec | None = Field(
        default=None, description="Structured plan step when the planner provided one"
    )
    history: str = Field(default="", description="Recent actions and observations")
    context: str = Field(default="", description="Relevant context from memory")
    available_tools: list[dict[str, Any]] = Field(
        default_factory=list, description="Available tools for the agent"
    )


class ThinkOutput(BaseModel):
    """Output from the THINK phase."""

    thought: str = Field(..., description="Agent's reasoning")
    proposed_action: ToolCall | None = Field(default=None, description="Proposed tool call")
    is_final_answer: bool = Field(
        default=False, description="Whether this provides the final answer"
    )
    final_answer: str | None = Field(default=None, description="Final answer if task is complete")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Confidence in the decision"
    )
    reasoning_tokens: int | None = Field(default=None, description="Tokens used in reasoning")
    using_activity_fallback: bool = Field(
        default=False, description="Whether activity fallback was used instead of native tools"
    )


class ValidateInput(BaseModel):
    """Input to the VALIDATE phase."""

    goal: str = Field(..., description="Current objective")
    proposed_action: ToolCall = Field(..., description="Action to validate")
    reasoning: str = Field(..., description="Reasoning behind the action")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance level")


class ValidateOutput(BaseModel):
    """Output from the VALIDATE phase."""

    result: ValidationResult = Field(..., description="Validation result")
    confidence: ConfidenceLevel = Field(..., description="Confidence in validation")
    concerns: list[str] = Field(default_factory=list, description="Issues or concerns identified")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")
    requires_human_approval: bool = Field(
        default=False, description="Whether human approval is needed"
    )
    approval_prompt: str | None = Field(
        default=None, description="Prompt to show to human if approval needed"
    )
    tokens_used: int | None = Field(default=None, description="Provider tokens used")


class ActInput(BaseModel):
    """Input to the ACT phase."""

    tool_call: ToolCall = Field(..., description="Tool call to execute")
    validation_result: ValidationResult | None = Field(
        default=None, description="Result of validation phase"
    )


class ActOutput(BaseModel):
    """Output from the ACT phase."""

    tool_result: ToolResult = Field(..., description="Result of tool execution")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    success: bool = Field(..., description="Whether execution succeeded")


class ObserveInput(BaseModel):
    """Input to the OBSERVE phase."""

    action_taken: ToolCall = Field(..., description="Action that was taken")
    action_result: ToolResult = Field(..., description="Result of the action")
    expected_outcome: str | None = Field(default=None, description="What was expected to happen")


class ObserveOutput(BaseModel):
    """Output from the OBSERVE phase."""

    observation: str = Field(..., description="Processed observation of the result")
    matches_expectation: bool | None = Field(
        default=None, description="Whether result matched expectation"
    )
    insights: list[str] = Field(default_factory=list, description="Key insights from observation")
    follow_up_needed: bool = Field(default=False, description="Whether follow-up action is needed")


class ReflectInput(BaseModel):
    """Input to the REFLECT phase."""

    goal: str = Field(..., description="Original goal")
    plan: list[str] = Field(..., description="Current plan")
    current_step_index: int = Field(..., description="Current step index")
    last_action: ToolCall | None = Field(
        default=None, description="Last action taken (None if no action was proposed)"
    )
    observation: str = Field(..., description="Observation from OBSERVE phase")
    iteration_number: int = Field(..., description="Current iteration number")


class ReflectOutput(BaseModel):
    """Output from the REFLECT phase."""

    evaluation: str = Field(..., description="Assessment of action effectiveness")
    progress_estimate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated progress toward goal (0.0 to 1.0)"
    )
    insights: list[str] = Field(
        default_factory=list, description="Key learnings from this iteration"
    )
    plan_needs_update: bool = Field(default=False, description="Whether plan should be modified")
    updated_plan: list[str] | None = Field(
        default=None, description="Updated plan if modification needed"
    )
    step_completed: bool = Field(default=False, description="Whether current step is complete")
    next_focus: str | None = Field(default=None, description="What to prioritize next")
    tokens_used: int | None = Field(default=None, description="Provider tokens used")


class UpdateInput(BaseModel):
    """Input to the UPDATE phase."""

    reflection: ReflectOutput = Field(..., description="Output from REFLECT phase")
    current_state: "EnhancedAgentState" = Field(..., description="Current agent state")


class UpdateOutput(BaseModel):
    """Output from the UPDATE phase."""

    state_updates: dict[str, Any] = Field(
        default_factory=dict, description="Updates to apply to agent state"
    )
    memory_updates: list[dict[str, Any]] = Field(
        default_factory=list, description="Updates to store in episodic memory"
    )
    working_memory_updates: dict[str, Any] = Field(
        default_factory=dict, description="Updates to working memory"
    )
    should_continue: bool = Field(default=True, description="Whether to continue cognitive loop")


# =============================================================================
# ITERATION TRACKING
# =============================================================================


class CycleIteration(BaseModel):
    """
    Complete record of a single cognitive cycle iteration.

    Tracks all 8 phases and their inputs/outputs for full traceability.

    Example:
        >>> iteration = CycleIteration(iteration_number=1)
        >>> iteration.perceive_output = PerceiveOutput(...)
        >>> iteration.plan_output = PlanOutput(...)
        >>> # ... all phases
        >>> iteration.mark_completed()
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration_number: int = Field(..., ge=1, description="Iteration sequence number")
    status: IterationStatus = Field(
        default=IterationStatus.IN_PROGRESS, description="Current status"
    )

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)

    # Phase outputs (populated as phases complete)
    perceive_output: PerceiveOutput | None = None
    plan_output: PlanOutput | None = None
    think_output: ThinkOutput | None = None
    validate_output: ValidateOutput | None = None
    act_output: ActOutput | None = None
    observe_output: ObserveOutput | None = None
    reflect_output: ReflectOutput | None = None
    update_output: UpdateOutput | None = None

    # Metadata
    total_tokens_used: int = Field(default=0, description="Total tokens in this iteration")
    total_time_ms: float = Field(default=0.0, description="Total time in milliseconds")
    error: str | None = Field(default=None, description="Error if failed")

    def mark_completed(self, success: bool = True) -> None:
        """Mark iteration as completed."""
        self.completed_at = datetime.utcnow()
        self.status = IterationStatus.COMPLETED if success else IterationStatus.FAILED

        # Calculate total time
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.total_time_ms = delta.total_seconds() * 1000

    def mark_interrupted(self, reason: str) -> None:
        """Mark iteration as interrupted."""
        self.status = IterationStatus.INTERRUPTED
        self.completed_at = datetime.utcnow()
        self.error = f"Interrupted: {reason}"

    @property
    def duration_ms(self) -> float:
        """Get iteration duration in milliseconds."""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return 0.0

    @property
    def phases_completed(self) -> set[CognitivePhase]:
        """Get set of completed phases."""
        completed = set()
        if self.perceive_output:
            completed.add(CognitivePhase.PERCEIVE)
        if self.plan_output:
            completed.add(CognitivePhase.PLAN)
        if self.think_output:
            completed.add(CognitivePhase.THINK)
        if self.validate_output:
            completed.add(CognitivePhase.VALIDATE)
        if self.act_output:
            completed.add(CognitivePhase.ACT)
        if self.observe_output:
            completed.add(CognitivePhase.OBSERVE)
        if self.reflect_output:
            completed.add(CognitivePhase.REFLECT)
        if self.update_output:
            completed.add(CognitivePhase.UPDATE)
        return completed

    def update_token_totals_from_phases(self) -> int:
        """Set and return total provider tokens captured by phase outputs."""
        total = sum(
            token_count
            for token_count in (
                self.plan_output.tokens_used if self.plan_output else None,
                self.think_output.reasoning_tokens if self.think_output else None,
                self.validate_output.tokens_used if self.validate_output else None,
                self.reflect_output.tokens_used if self.reflect_output else None,
            )
            if token_count is not None
        )
        self.total_tokens_used = total
        return total

    def to_history_summary(
        self,
        *,
        max_observation_chars: int = 1000,
        max_tool_result_chars: int = 1000,
        max_argument_chars: int = 1000,
    ) -> dict[str, Any]:
        """Return a compact, JSON-safe summary suitable for future prompts."""
        action_payload = None
        if self.think_output and self.think_output.proposed_action:
            action = self.think_output.proposed_action
            action_payload = {
                "id": action.id,
                "name": action.name,
                "arguments": _json_safe(
                    action.arguments,
                    max_string_chars=max_argument_chars,
                    max_items=20,
                ),
            }

        observation_payload = None
        if self.observe_output:
            content, truncated = _truncate_text(
                self.observe_output.observation,
                max_observation_chars,
            )
            observation_payload = {
                "content": content,
                "truncated": truncated,
                "matches_expectation": self.observe_output.matches_expectation,
                "follow_up_needed": self.observe_output.follow_up_needed,
                "insights": _json_safe(
                    self.observe_output.insights,
                    max_string_chars=300,
                    max_items=10,
                ),
            }

        tool_result_payload = None
        if self.act_output and self.act_output.tool_result:
            result = self.act_output.tool_result
            content, truncated = _truncate_text(result.content, max_tool_result_chars)
            tool_result_payload = {
                "tool_call_id": result.tool_call_id,
                "content": content,
                "truncated": truncated,
                "is_error": result.is_error,
                "execution_success": self.act_output.success,
                "execution_time_ms": self.act_output.execution_time_ms,
            }

        reflect_payload = None
        if self.reflect_output:
            reflect_payload = {
                "progress_estimate": self.reflect_output.progress_estimate,
                "step_completed": self.reflect_output.step_completed,
                "plan_needs_update": self.reflect_output.plan_needs_update,
                "next_focus": self.reflect_output.next_focus,
            }

        return {
            "iteration_number": self.iteration_number,
            "status": self.status.value,
            "phases_completed": sorted(phase.value for phase in self.phases_completed),
            "action": action_payload,
            "observation": observation_payload,
            "tool_result": tool_result_payload,
            "reflection": reflect_payload,
            "phase_tokens": {
                "plan": self.plan_output.tokens_used
                if self.plan_output and self.plan_output.tokens_used is not None
                else None,
                "think": self.think_output.reasoning_tokens
                if self.think_output and self.think_output.reasoning_tokens is not None
                else None,
                "validate": self.validate_output.tokens_used
                if self.validate_output and self.validate_output.tokens_used is not None
                else None,
                "reflect": self.reflect_output.tokens_used
                if self.reflect_output and self.reflect_output.tokens_used is not None
                else None,
            },
            "total_tokens_used": self.total_tokens_used,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# =============================================================================
# ENHANCED AGENT STATE
# =============================================================================


class EnhancedAgentState(AgentState):
    """
    Enhanced agent state with support for 8-phase cognitive cycle.

    Extends llmcore's base AgentState with additional fields for:
    - Iteration history tracking
    - Working memory
    - Confidence tracking
    - Validation state
    - Progress estimation

    Example:
        >>> state = EnhancedAgentState(
        ...     goal="Calculate factorial of 10",
        ...     session_id="session-123"
        ... )
        >>>
        >>> # Add iteration
        >>> iteration = CycleIteration(iteration_number=1)
        >>> state.add_iteration(iteration)
        >>>
        >>> # Update progress
        >>> state.progress_estimate = 0.5
    """

    # Session tracking
    session_id: str = Field(default="", description="Session identifier for this agent execution")
    context: str = Field(default="", description="Initial context for the task")

    # Iteration tracking
    iterations: list[CycleIteration] = Field(
        default_factory=list, description="Complete history of all iterations"
    )
    current_iteration: CycleIteration | None = Field(
        default=None, description="Currently executing iteration"
    )

    # Working memory (ephemeral within task)
    working_memory: dict[str, Any] = Field(
        default_factory=dict, description="Ephemeral working memory for this task"
    )

    # Progress tracking
    progress_estimate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated progress toward goal (0.0 to 1.0)"
    )

    # Confidence tracking
    overall_confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Overall confidence in current approach"
    )

    # Validation state
    pending_validation: ValidateInput | None = Field(
        default=None, description="Action awaiting validation"
    )
    validation_history: list[ValidateOutput] = Field(
        default_factory=list, description="History of validation decisions"
    )

    # Enhanced plan tracking
    plan_created_at: datetime | None = Field(default=None, description="When the plan was created")
    plan_updated_at: datetime | None = Field(
        default=None, description="When the plan was last updated"
    )
    plan_version: int = Field(default=0, description="Plan version number")

    # Metrics
    total_tokens_used: int = Field(default=0, description="Total tokens used across all iterations")
    total_tool_calls: int = Field(default=0, description="Total number of tool calls made")

    # Metadata for extensibility
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata for extensibility and goal tracking"
    )
    # === P0 FIX: Added missing fields ===
    # Fix #1: Final answer storage when task completes
    final_answer: str | None = Field(default=None, description="Final answer when task is complete")

    # Fix #2: Tool call pending execution from THINK phase
    pending_tool_call: ToolCall | None = None

    # Fix #4: Flag indicating agent is waiting for human approval
    awaiting_human_approval: bool = False

    # Fix #5: Prompt to show user when human approval is required
    pending_approval_prompt: str | None = Field(
        default=None, description="Prompt to display when awaiting human approval"
    )

    # Fix #3: Private field for explicit is_finished setting
    _is_finished_override: bool = PrivateAttr(default=False)
    # === END P0 FIX ===

    @property
    def is_finished(self) -> bool:
        """
        Check if the agent has finished its task.

        Returns True if:
        - All plan steps are completed, OR
        - The goal has been marked as achieved in metadata

        Returns:
            True if the agent's task is complete, False otherwise.
        """
        # P0 Fix #3: Check explicit override first
        if getattr(self, "_is_finished_override", False):
            return True

        # Check if all plan steps are completed
        if self.plan_steps_status:
            all_completed = all(status == "completed" for status in self.plan_steps_status)
            if all_completed:
                return True

        # Check metadata for goal achieved flag
        if self.metadata.get("goal_achieved", False):
            return True

        return False

    @is_finished.setter
    def is_finished(self, value: bool) -> None:
        """Allow explicit setting of finished state (P0 Fix #3)."""
        object.__setattr__(self, "_is_finished_override", value)

    def to_resume_snapshot(
        self,
        *,
        max_iterations: int = 5,
        max_string_chars: int = 2000,
        max_observation_chars: int = 1000,
        max_items: int = 50,
    ) -> dict[str, Any]:
        """Return a bounded, JSON-safe snapshot for checkpoint/resume surfaces."""
        plan = [_truncate_text(step, max_string_chars)[0] for step in self.plan[:max_items]]
        statuses = [str(status) for status in self.plan_steps_status[:max_items]]
        current_index = self.current_plan_step_index
        current_step = plan[current_index] if 0 <= current_index < len(plan) else ""
        recent_iterations = self.iterations[-max(0, max_iterations) :]

        return {
            "schema_version": STATE_SNAPSHOT_VERSION,
            "session_id": self.session_id,
            "goal": _truncate_text(self.goal, max_string_chars)[0],
            "context": _truncate_text(self.context, max_string_chars)[0],
            "plan": plan,
            "plan_steps_status": statuses,
            "current_plan_step_index": current_index,
            "current_plan_step": current_step,
            "plan_version": self.plan_version,
            "progress_estimate": self.progress_estimate,
            "overall_confidence": self.overall_confidence.value,
            "is_finished": self.is_finished,
            "final_answer": _truncate_text(self.final_answer, max_string_chars)[0]
            if self.final_answer
            else None,
            "awaiting_human_approval": self.awaiting_human_approval,
            "pending_approval_prompt": _truncate_text(
                self.pending_approval_prompt,
                max_string_chars,
            )[0]
            if self.pending_approval_prompt
            else None,
            "pending_tool_call": _json_safe(
                self.pending_tool_call,
                max_string_chars=max_string_chars,
                max_items=max_items,
            ),
            "pending_validation": _json_safe(
                self.pending_validation,
                max_string_chars=max_string_chars,
                max_items=max_items,
            ),
            "working_memory": _json_safe(
                self.working_memory,
                max_string_chars=max_string_chars,
                max_items=max_items,
            ),
            "metadata": _json_safe(
                self.metadata,
                max_string_chars=max_string_chars,
                max_items=max_items,
            ),
            "history_of_thoughts": _json_safe(
                self.history_of_thoughts[-max_items:],
                max_string_chars=max_string_chars,
                max_items=max_items,
            ),
            "observations": _json_safe(
                self.observations,
                max_string_chars=max_observation_chars,
                max_items=max_items,
            ),
            "iterations": [
                iteration.to_history_summary(max_observation_chars=max_observation_chars)
                for iteration in recent_iterations
            ],
            "current_iteration": self.current_iteration.to_history_summary(
                max_observation_chars=max_observation_chars
            )
            if self.current_iteration
            else None,
            "metrics": {
                "iteration_count": self.iteration_count,
                "successful_iterations": self.successful_iterations,
                "failed_iterations": self.failed_iterations,
                "total_tokens_used": self.total_tokens_used,
                "total_tool_calls": self.total_tool_calls,
                "average_iteration_time_ms": self.average_iteration_time_ms,
            },
        }

    @classmethod
    def from_resume_snapshot(cls, snapshot: dict[str, Any]) -> "EnhancedAgentState":
        """Rehydrate core resumable state from ``to_resume_snapshot`` output."""
        state = cls(
            goal=str(snapshot.get("goal") or ""),
            session_id=str(snapshot.get("session_id") or ""),
            context=str(snapshot.get("context") or ""),
        )
        state.plan = [str(item) for item in snapshot.get("plan") or []]
        state.plan_steps_status = [str(item) for item in snapshot.get("plan_steps_status") or []]
        state.current_plan_step_index = int(snapshot.get("current_plan_step_index") or 0)
        state.plan_version = int(snapshot.get("plan_version") or 0)
        state.progress_estimate = float(snapshot.get("progress_estimate") or 0.0)
        try:
            state.overall_confidence = ConfidenceLevel(
                snapshot.get("overall_confidence") or ConfidenceLevel.MEDIUM.value
            )
        except ValueError:
            state.overall_confidence = ConfidenceLevel.MEDIUM

        state.final_answer = snapshot.get("final_answer")
        state.awaiting_human_approval = bool(snapshot.get("awaiting_human_approval", False))
        state.pending_approval_prompt = snapshot.get("pending_approval_prompt")
        pending_tool_call = snapshot.get("pending_tool_call")
        pending_tool_call_fallback = None
        if isinstance(pending_tool_call, dict):
            try:
                state.pending_tool_call = ToolCall.model_validate(pending_tool_call)
            except Exception:
                pending_tool_call_fallback = pending_tool_call

        pending_validation = snapshot.get("pending_validation")
        pending_validation_fallback = None
        if isinstance(pending_validation, dict):
            try:
                state.pending_validation = ValidateInput.model_validate(pending_validation)
            except Exception:
                pending_validation_fallback = pending_validation

        state.working_memory = dict(snapshot.get("working_memory") or {})
        state.metadata = dict(snapshot.get("metadata") or {})
        if pending_tool_call_fallback is not None:
            state.metadata["_resume_pending_tool_call"] = pending_tool_call_fallback
        if pending_validation_fallback is not None:
            state.metadata["_resume_pending_validation"] = pending_validation_fallback
        state.history_of_thoughts = [str(item) for item in snapshot.get("history_of_thoughts") or []]
        observations = snapshot.get("observations") or {}
        state.observations = observations if isinstance(observations, dict) else {}
        metrics = snapshot.get("metrics") or {}
        state.total_tokens_used = int(metrics.get("total_tokens_used") or 0)
        state.total_tool_calls = int(metrics.get("total_tool_calls") or 0)
        state.metadata["_resume_snapshot_schema_version"] = snapshot.get("schema_version")
        state.metadata["_resume_snapshot_iterations"] = snapshot.get("iterations") or []
        state.is_finished = bool(snapshot.get("is_finished", False))
        return state

    def mark_context_compressed(
        self,
        *,
        reason: str,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
    ) -> dict[str, Any]:
        """Record a context compression event and cooldown marker."""
        previous = self.working_memory.get(_CONTEXT_COMPRESSION_KEY)
        if not isinstance(previous, dict):
            previous = {}
        metadata = {
            "compression_count": int(previous.get("compression_count") or 0) + 1,
            "last_iteration_count": self.iteration_count,
            "last_reason": reason,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "last_compressed_at": datetime.utcnow().isoformat(),
        }
        self.working_memory[_CONTEXT_COMPRESSION_KEY] = metadata
        return metadata

    def should_compress_context(self, *, min_iterations_between: int = 2) -> bool:
        """Return False during the compression cooldown window."""
        marker = self.working_memory.get(_CONTEXT_COMPRESSION_KEY)
        if not isinstance(marker, dict):
            return True
        try:
            last_iteration = int(marker.get("last_iteration_count", -min_iterations_between))
        except (TypeError, ValueError):
            return True
        return (self.iteration_count - last_iteration) >= max(0, min_iterations_between)

    def add_iteration(self, iteration: CycleIteration) -> None:
        """
        Add a completed iteration to history.

        Args:
            iteration: The iteration to add
        """
        self.iterations.append(iteration)

        # Update metrics
        self.total_tokens_used += iteration.total_tokens_used
        if iteration.act_output:
            self.total_tool_calls += 1

    def start_iteration(self, iteration_number: int) -> CycleIteration:
        """
        Start a new iteration.

        Args:
            iteration_number: The iteration number

        Returns:
            The new iteration
        """
        iteration = CycleIteration(iteration_number=iteration_number)
        self.current_iteration = iteration
        return iteration

    def complete_iteration(self, success: bool = True) -> None:
        """
        Complete the current iteration and add to history.

        Args:
            success: Whether the iteration was successful
        """
        if self.current_iteration:
            self.current_iteration.mark_completed(success)
            self.add_iteration(self.current_iteration)
            self.current_iteration = None

    def update_plan(self, new_plan: list[str], reasoning: str = "") -> None:
        """
        Update the agent's plan.

        Args:
            new_plan: The new plan steps
            reasoning: Reasoning for the update
        """
        self.plan = new_plan
        self.plan_steps_status = ["pending"] * len(new_plan)
        self.current_plan_step_index = 0
        self.plan_version += 1
        self.plan_updated_at = datetime.utcnow()

    @property
    def iteration_count(self) -> int:
        """Get total number of iterations."""
        return len(self.iterations)

    @property
    def successful_iterations(self) -> int:
        """Get count of successful iterations."""
        return sum(1 for it in self.iterations if it.status == IterationStatus.COMPLETED)

    @property
    def failed_iterations(self) -> int:
        """Get count of failed iterations."""
        return sum(1 for it in self.iterations if it.status == IterationStatus.FAILED)

    @property
    def average_iteration_time_ms(self) -> float:
        """Get average iteration time in milliseconds."""
        if not self.iterations:
            return 0.0

        total_time = sum(it.duration_ms for it in self.iterations)
        return total_time / len(self.iterations)

    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """Get value from working memory."""
        return self.working_memory.get(key, default)

    def set_working_memory(self, key: str, value: Any) -> None:
        """Set value in working memory."""
        self.working_memory[key] = value

    def clear_working_memory(self) -> None:
        """Clear all working memory."""
        self.working_memory.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ActInput",
    "ActOutput",
    "CognitivePhase",
    "ConfidenceLevel",
    "CycleIteration",
    "EnhancedAgentState",
    "IterationStatus",
    "ObserveInput",
    "ObserveOutput",
    "PerceiveInput",
    "PerceiveOutput",
    "PlanInput",
    "PlanOutput",
    "PlanStepSpec",
    "ReflectInput",
    "ReflectOutput",
    "ThinkInput",
    "ThinkOutput",
    "UpdateInput",
    "UpdateOutput",
    "ValidateInput",
    "ValidateOutput",
    "ValidationResult",
]
