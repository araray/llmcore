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

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

# Import existing models from llmcore
from ...models import AgentState, ToolCall, ToolResult

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
    context_query: Optional[str] = Field(
        default=None, description="Optional specific query for context retrieval"
    )
    force_refresh: bool = Field(
        default=False, description="Force refresh of context even if cached"
    )


class PerceiveOutput(BaseModel):
    """Output from the PERCEIVE phase."""

    retrieved_context: List[str] = Field(
        default_factory=list, description="Context items retrieved from memory"
    )
    working_memory_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of working memory state"
    )
    environmental_state: Dict[str, Any] = Field(
        default_factory=dict, description="Current environmental state (sandbox, tools, etc.)"
    )
    perceived_at: datetime = Field(
        default_factory=datetime.utcnow, description="When perception occurred"
    )


class PlanInput(BaseModel):
    """Input to the PLAN phase."""

    goal: str = Field(..., description="The goal to plan for")
    context: Optional[str] = Field(default=None, description="Relevant context for planning")
    constraints: Optional[str] = Field(default=None, description="Constraints or requirements")
    existing_plan: Optional[List[str]] = Field(default=None, description="Existing plan to refine")


class PlanOutput(BaseModel):
    """Output from the PLAN phase."""

    plan_steps: List[str] = Field(
        default_factory=list, description="Ordered list of actionable steps"
    )
    reasoning: str = Field(default="", description="Strategic reasoning behind the plan")
    estimated_iterations: Optional[int] = Field(
        default=None, description="Estimated iterations to complete plan"
    )
    risks_identified: List[str] = Field(
        default_factory=list, description="Potential risks or challenges identified"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ThinkInput(BaseModel):
    """Input to the THINK phase."""

    goal: str = Field(..., description="Current objective")
    current_step: str = Field(..., description="Current plan step")
    history: str = Field(default="", description="Recent actions and observations")
    context: str = Field(default="", description="Relevant context from memory")
    available_tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Available tools for the agent"
    )


class ThinkOutput(BaseModel):
    """Output from the THINK phase."""

    thought: str = Field(..., description="Agent's reasoning")
    proposed_action: Optional[ToolCall] = Field(default=None, description="Proposed tool call")
    is_final_answer: bool = Field(
        default=False, description="Whether this provides the final answer"
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final answer if task is complete"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Confidence in the decision"
    )
    reasoning_tokens: Optional[int] = Field(default=None, description="Tokens used in reasoning")


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
    concerns: List[str] = Field(default_factory=list, description="Issues or concerns identified")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    requires_human_approval: bool = Field(
        default=False, description="Whether human approval is needed"
    )
    approval_prompt: Optional[str] = Field(
        default=None, description="Prompt to show to human if approval needed"
    )


class ActInput(BaseModel):
    """Input to the ACT phase."""

    tool_call: ToolCall = Field(..., description="Tool call to execute")
    validation_result: Optional[ValidationResult] = Field(
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
    expected_outcome: Optional[str] = Field(default=None, description="What was expected to happen")


class ObserveOutput(BaseModel):
    """Output from the OBSERVE phase."""

    observation: str = Field(..., description="Processed observation of the result")
    matches_expectation: Optional[bool] = Field(
        default=None, description="Whether result matched expectation"
    )
    insights: List[str] = Field(default_factory=list, description="Key insights from observation")
    follow_up_needed: bool = Field(default=False, description="Whether follow-up action is needed")


class ReflectInput(BaseModel):
    """Input to the REFLECT phase."""

    goal: str = Field(..., description="Original goal")
    plan: List[str] = Field(..., description="Current plan")
    current_step_index: int = Field(..., description="Current step index")
    last_action: ToolCall = Field(..., description="Last action taken")
    observation: str = Field(..., description="Observation from OBSERVE phase")
    iteration_number: int = Field(..., description="Current iteration number")


class ReflectOutput(BaseModel):
    """Output from the REFLECT phase."""

    evaluation: str = Field(..., description="Assessment of action effectiveness")
    progress_estimate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated progress toward goal (0.0 to 1.0)"
    )
    insights: List[str] = Field(
        default_factory=list, description="Key learnings from this iteration"
    )
    plan_needs_update: bool = Field(default=False, description="Whether plan should be modified")
    updated_plan: Optional[List[str]] = Field(
        default=None, description="Updated plan if modification needed"
    )
    step_completed: bool = Field(default=False, description="Whether current step is complete")
    next_focus: Optional[str] = Field(default=None, description="What to prioritize next")


class UpdateInput(BaseModel):
    """Input to the UPDATE phase."""

    reflection: ReflectOutput = Field(..., description="Output from REFLECT phase")
    current_state: "EnhancedAgentState" = Field(..., description="Current agent state")


class UpdateOutput(BaseModel):
    """Output from the UPDATE phase."""

    state_updates: Dict[str, Any] = Field(
        default_factory=dict, description="Updates to apply to agent state"
    )
    memory_updates: List[Dict[str, Any]] = Field(
        default_factory=list, description="Updates to store in episodic memory"
    )
    working_memory_updates: Dict[str, Any] = Field(
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
    completed_at: Optional[datetime] = Field(default=None)

    # Phase outputs (populated as phases complete)
    perceive_output: Optional[PerceiveOutput] = None
    plan_output: Optional[PlanOutput] = None
    think_output: Optional[ThinkOutput] = None
    validate_output: Optional[ValidateOutput] = None
    act_output: Optional[ActOutput] = None
    observe_output: Optional[ObserveOutput] = None
    reflect_output: Optional[ReflectOutput] = None
    update_output: Optional[UpdateOutput] = None

    # Metadata
    total_tokens_used: int = Field(default=0, description="Total tokens in this iteration")
    total_time_ms: float = Field(default=0.0, description="Total time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error if failed")

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
    def phases_completed(self) -> Set[CognitivePhase]:
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

    # Iteration tracking
    iterations: List[CycleIteration] = Field(
        default_factory=list, description="Complete history of all iterations"
    )
    current_iteration: Optional[CycleIteration] = Field(
        default=None, description="Currently executing iteration"
    )

    # Working memory (ephemeral within task)
    working_memory: Dict[str, Any] = Field(
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
    pending_validation: Optional[ValidateInput] = Field(
        default=None, description="Action awaiting validation"
    )
    validation_history: List[ValidateOutput] = Field(
        default_factory=list, description="History of validation decisions"
    )

    # Enhanced plan tracking
    plan_created_at: Optional[datetime] = Field(
        default=None, description="When the plan was created"
    )
    plan_updated_at: Optional[datetime] = Field(
        default=None, description="When the plan was last updated"
    )
    plan_version: int = Field(default=0, description="Plan version number")

    # Metrics
    total_tokens_used: int = Field(default=0, description="Total tokens used across all iterations")
    total_tool_calls: int = Field(default=0, description="Total number of tool calls made")

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

    def update_plan(self, new_plan: List[str], reasoning: str = "") -> None:
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
    # Enums
    "CognitivePhase",
    "IterationStatus",
    "ValidationResult",
    "ConfidenceLevel",
    # Phase I/O Models
    "PerceiveInput",
    "PerceiveOutput",
    "PlanInput",
    "PlanOutput",
    "ThinkInput",
    "ThinkOutput",
    "ValidateInput",
    "ValidateOutput",
    "ActInput",
    "ActOutput",
    "ObserveInput",
    "ObserveOutput",
    "ReflectInput",
    "ReflectOutput",
    "UpdateInput",
    "UpdateOutput",
    # Iteration Tracking
    "CycleIteration",
    # Enhanced State
    "EnhancedAgentState",
]
