# src/llmcore/agents/persona/models.py
"""
Persona System Models for Darwin Layer 2.

The persona system enables customization of agent behavior, communication style,
and decision-making patterns. Personas define:
- Personality traits (analytical, creative, cautious, etc.)
- Communication preferences (verbosity, formality, emoji usage)
- Decision-making patterns (risk tolerance, planning depth)
- Prompt modifications (system prompts, phase-specific overrides)

Design Philosophy:
- Composable: Mix and match traits
- Extensible: Easy to add custom traits
- Overrideable: Phase-specific customization
- Serializable: Save/load personas

References:
    - Technical Spec: Section 5.4 (Persona System)
    - Dossier: Step 2.8 (Persona System)
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# ENUMERATIONS
# =============================================================================


class PersonalityTrait(str, Enum):
    """Core personality traits that influence behavior."""

    ANALYTICAL = "analytical"  # Logical, data-driven, systematic
    CREATIVE = "creative"  # Innovative, exploratory, divergent
    PRAGMATIC = "pragmatic"  # Practical, results-oriented, efficient
    CAUTIOUS = "cautious"  # Risk-averse, thorough, conservative
    BOLD = "bold"  # Risk-taking, confident, decisive
    METHODICAL = "methodical"  # Structured, process-oriented, detailed
    ADAPTIVE = "adaptive"  # Flexible, responsive, dynamic
    COLLABORATIVE = "collaborative"  # Team-oriented, communicative


class CommunicationStyle(str, Enum):
    """Communication style preferences."""

    CONCISE = "concise"  # Brief, to-the-point
    DETAILED = "detailed"  # Comprehensive, thorough
    TECHNICAL = "technical"  # Formal, precise, jargon-heavy
    CONVERSATIONAL = "conversational"  # Informal, friendly, approachable
    PROFESSIONAL = "professional"  # Formal, polished, business-like


class RiskTolerance(str, Enum):
    """Risk tolerance levels for decision-making."""

    VERY_LOW = "very_low"  # Extremely conservative, requires approval
    LOW = "low"  # Conservative, careful validation
    MEDIUM = "medium"  # Balanced approach
    HIGH = "high"  # Willing to take calculated risks
    VERY_HIGH = "very_high"  # Aggressive, minimal validation


class PlanningDepth(str, Enum):
    """How detailed plans should be."""

    MINIMAL = "minimal"  # High-level only, 2-3 steps
    STANDARD = "standard"  # Balanced detail, 4-6 steps
    DETAILED = "detailed"  # Comprehensive, 7-10 steps
    EXHAUSTIVE = "exhaustive"  # Very thorough, 10+ steps


# =============================================================================
# PERSONA MODELS
# =============================================================================


class PersonaTrait(BaseModel):
    """A single personality trait with intensity."""

    trait: PersonalityTrait = Field(..., description="The trait type")
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Trait intensity (0.0 = none, 1.0 = normal, 2.0 = extreme)",
    )

    def __str__(self) -> str:
        if self.intensity < 0.5:
            prefix = "slightly"
        elif self.intensity < 1.5:
            prefix = "moderately"
        else:
            prefix = "very"
        return f"{prefix} {self.trait.value}"


class CommunicationPreferences(BaseModel):
    """Preferences for how the agent communicates."""

    style: CommunicationStyle = Field(
        default=CommunicationStyle.PROFESSIONAL, description="Overall communication style"
    )

    verbosity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Verbosity level (0.0 = minimal, 1.0 = maximum)"
    )

    use_emojis: bool = Field(default=False, description="Whether to use emojis in responses")

    formality: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Formality level (0.0 = casual, 1.0 = formal)"
    )

    explain_reasoning: bool = Field(
        default=True, description="Whether to explain reasoning behind decisions"
    )


class DecisionMakingPreferences(BaseModel):
    """Preferences for decision-making and actions."""

    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.MEDIUM, description="Risk tolerance level"
    )

    planning_depth: PlanningDepth = Field(
        default=PlanningDepth.STANDARD, description="How detailed plans should be"
    )

    require_validation: bool = Field(
        default=True, description="Whether to validate actions before execution"
    )

    max_iterations_per_task: int = Field(
        default=10, ge=1, le=50, description="Maximum iterations before requiring human input"
    )

    prefer_tools: list[str] = Field(
        default_factory=list, description="Preferred tools to use when available"
    )

    avoid_tools: list[str] = Field(
        default_factory=list, description="Tools to avoid unless necessary"
    )


class PromptModifications(BaseModel):
    """Custom prompt modifications for this persona."""

    system_prompt_prefix: str | None = Field(
        default=None, description="Prefix to add to system prompts"
    )

    system_prompt_suffix: str | None = Field(
        default=None, description="Suffix to add to system prompts"
    )

    phase_prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Phase-specific prompt overrides (phase_name -> prompt_template_id)",
    )

    custom_instructions: str | None = Field(
        default=None, description="Custom instructions to include in all prompts"
    )


class AgentPersona(BaseModel):
    """
    Complete agent persona definition.

    A persona defines the agent's personality, communication style,
    decision-making preferences, and behavior patterns.

    Example:
        >>> persona = AgentPersona(
        ...     name="Data Analyst",
        ...     description="Analytical agent focused on data-driven insights",
        ...     traits=[
        ...         PersonaTrait(trait=PersonalityTrait.ANALYTICAL, intensity=1.5),
        ...         PersonaTrait(trait=PersonalityTrait.METHODICAL, intensity=1.2)
        ...     ],
        ...     communication=CommunicationPreferences(
        ...         style=CommunicationStyle.TECHNICAL,
        ...         verbosity=0.7
        ...     ),
        ...     decision_making=DecisionMakingPreferences(
        ...         planning_depth=PlanningDepth.DETAILED
        ...     )
        ... )
    """

    id: str = Field(..., description="Unique persona identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Persona description")

    # Personality
    traits: list[PersonaTrait] = Field(default_factory=list, description="Personality traits")

    # Communication
    communication: CommunicationPreferences = Field(
        default_factory=CommunicationPreferences, description="Communication preferences"
    )

    # Decision-making
    decision_making: DecisionMakingPreferences = Field(
        default_factory=DecisionMakingPreferences, description="Decision-making preferences"
    )

    # Prompt modifications
    prompts: PromptModifications = Field(
        default_factory=PromptModifications, description="Custom prompt modifications"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When persona was created"
    )

    updated_at: datetime | None = Field(default=None, description="When persona was last updated")

    is_builtin: bool = Field(default=False, description="Whether this is a built-in persona")

    @property
    def trait_summary(self) -> str:
        """Get summary of personality traits."""
        if not self.traits:
            return "No specific traits"
        return ", ".join(str(trait) for trait in self.traits)

    def get_system_prompt_addition(self) -> str:
        """
        Get system prompt additions based on persona.

        Returns:
            Additional text to add to system prompts
        """
        parts = []

        # Add prefix
        if self.prompts.system_prompt_prefix:
            parts.append(self.prompts.system_prompt_prefix)

        # Add trait-based instructions
        if self.traits:
            trait_instructions = self._generate_trait_instructions()
            parts.append(trait_instructions)

        # Add communication style
        comm_instructions = self._generate_communication_instructions()
        if comm_instructions:
            parts.append(comm_instructions)

        # Add custom instructions
        if self.prompts.custom_instructions:
            parts.append(self.prompts.custom_instructions)

        # Add suffix
        if self.prompts.system_prompt_suffix:
            parts.append(self.prompts.system_prompt_suffix)

        return "\n\n".join(parts) if parts else ""

    def _generate_trait_instructions(self) -> str:
        """Generate instructions based on personality traits."""
        instructions = []

        for trait in self.traits:
            if trait.trait == PersonalityTrait.ANALYTICAL:
                instructions.append(
                    "Approach problems analytically. Use data and logic to support decisions."
                )
            elif trait.trait == PersonalityTrait.CREATIVE:
                instructions.append(
                    "Think creatively and explore innovative solutions. Don't be afraid to try novel approaches."
                )
            elif trait.trait == PersonalityTrait.PRAGMATIC:
                instructions.append(
                    "Focus on practical, results-oriented solutions. Prioritize efficiency and effectiveness."
                )
            elif trait.trait == PersonalityTrait.CAUTIOUS:
                instructions.append(
                    "Be careful and thorough. Consider potential risks before acting."
                )
            elif trait.trait == PersonalityTrait.BOLD:
                instructions.append(
                    "Be confident and decisive. Take calculated risks when appropriate."
                )
            elif trait.trait == PersonalityTrait.METHODICAL:
                instructions.append(
                    "Follow systematic, step-by-step processes. Be thorough and detail-oriented."
                )
            elif trait.trait == PersonalityTrait.ADAPTIVE:
                instructions.append(
                    "Be flexible and responsive to changing conditions. Adjust your approach as needed."
                )
            elif trait.trait == PersonalityTrait.COLLABORATIVE:
                instructions.append(
                    "Communicate clearly and work cooperatively. Seek input and explain your reasoning."
                )

        return " ".join(instructions)

    def _generate_communication_instructions(self) -> str:
        """Generate instructions based on communication preferences."""
        instructions = []

        # Verbosity
        if self.communication.verbosity < 0.3:
            instructions.append("Keep responses concise and to the point.")
        elif self.communication.verbosity > 0.7:
            instructions.append("Provide detailed, comprehensive responses.")

        # Formality
        if self.communication.formality < 0.3:
            instructions.append("Use a casual, friendly tone.")
        elif self.communication.formality > 0.7:
            instructions.append("Maintain a formal, professional tone.")

        # Emojis
        if self.communication.use_emojis:
            instructions.append("Use emojis to enhance communication when appropriate.")

        # Explain reasoning
        if self.communication.explain_reasoning:
            instructions.append("Always explain your reasoning and decision-making process.")

        return " ".join(instructions)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PersonalityTrait",
    "CommunicationStyle",
    "RiskTolerance",
    "PlanningDepth",
    # Models
    "PersonaTrait",
    "CommunicationPreferences",
    "DecisionMakingPreferences",
    "PromptModifications",
    "AgentPersona",
]
