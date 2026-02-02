# src/llmcore/agents/persona/manager.py
"""
Persona Manager for Darwin Layer 2.

The PersonaManager handles:
- Loading and storing personas
- Applying personas to cognitive phases
- Built-in persona definitions
- Custom persona creation
- Persona serialization/deserialization

References:
    - Technical Spec: Section 5.4 (Persona System)
    - Dossier: Step 2.8 (Persona Manager)
"""

import logging
from typing import Any, Dict, List, Optional

from .models import (
    AgentPersona,
    CommunicationPreferences,
    CommunicationStyle,
    DecisionMakingPreferences,
    PersonalityTrait,
    PersonaTrait,
    PlanningDepth,
    PromptModifications,
    RiskTolerance,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PERSONA MANAGER
# =============================================================================


class PersonaManager:
    """
    Manages agent personas and applies them to cognitive phases.

    The PersonaManager:
    - Loads built-in personas
    - Creates custom personas
    - Applies persona modifications to prompts
    - Stores and retrieves personas

    Example:
        >>> manager = PersonaManager()
        >>>
        >>> # Get built-in persona
        >>> analyst = manager.get_persona("analyst")
        >>>
        >>> # Create custom persona
        >>> custom = manager.create_persona(
        ...     name="Custom Agent",
        ...     traits=[PersonalityTrait.CREATIVE],
        ...     communication_style=CommunicationStyle.CONVERSATIONAL
        ... )
        >>>
        >>> # Apply persona to prompt
        >>> modified_prompt = manager.apply_persona_to_prompt(
        ...     base_prompt="Plan the task",
        ...     persona=analyst
        ... )
    """

    def __init__(self):
        """Initialize the persona manager."""
        self.personas: Dict[str, AgentPersona] = {}
        self._load_builtin_personas()

    def _load_builtin_personas(self) -> None:
        """Load built-in personas."""
        builtin = [
            self._create_assistant_persona(),
            self._create_analyst_persona(),
            self._create_developer_persona(),
            self._create_researcher_persona(),
            self._create_creative_persona(),
        ]

        for persona in builtin:
            self.personas[persona.id] = persona

        logger.info(f"Loaded {len(builtin)} built-in personas")

    # =========================================================================
    # BUILT-IN PERSONAS
    # =========================================================================

    def _create_assistant_persona(self) -> AgentPersona:
        """Create the default Assistant persona."""
        return AgentPersona(
            id="assistant",
            name="Assistant",
            description="Balanced, helpful assistant focused on completing tasks efficiently",
            traits=[
                PersonaTrait(trait=PersonalityTrait.PRAGMATIC, intensity=1.2),
                PersonaTrait(trait=PersonalityTrait.COLLABORATIVE, intensity=1.0),
                PersonaTrait(trait=PersonalityTrait.ADAPTIVE, intensity=1.0),
            ],
            communication=CommunicationPreferences(
                style=CommunicationStyle.PROFESSIONAL,
                verbosity=0.6,
                formality=0.6,
                explain_reasoning=True,
            ),
            decision_making=DecisionMakingPreferences(
                risk_tolerance=RiskTolerance.MEDIUM,
                planning_depth=PlanningDepth.STANDARD,
                require_validation=True,
                max_iterations_per_task=10,
            ),
            is_builtin=True,
        )

    def _create_analyst_persona(self) -> AgentPersona:
        """Create the Analyst persona."""
        return AgentPersona(
            id="analyst",
            name="Data Analyst",
            description="Analytical, data-driven agent focused on thorough analysis",
            traits=[
                PersonaTrait(trait=PersonalityTrait.ANALYTICAL, intensity=1.8),
                PersonaTrait(trait=PersonalityTrait.METHODICAL, intensity=1.5),
                PersonaTrait(trait=PersonalityTrait.CAUTIOUS, intensity=1.2),
            ],
            communication=CommunicationPreferences(
                style=CommunicationStyle.TECHNICAL,
                verbosity=0.8,
                formality=0.8,
                explain_reasoning=True,
            ),
            decision_making=DecisionMakingPreferences(
                risk_tolerance=RiskTolerance.LOW,
                planning_depth=PlanningDepth.DETAILED,
                require_validation=True,
                max_iterations_per_task=15,
            ),
            prompts=PromptModifications(
                custom_instructions=(
                    "Always support decisions with data and evidence. "
                    "Perform thorough analysis before taking action."
                )
            ),
            is_builtin=True,
        )

    def _create_developer_persona(self) -> AgentPersona:
        """Create the Developer persona."""
        return AgentPersona(
            id="developer",
            name="Software Developer",
            description="Technical, systematic agent focused on code quality and best practices",
            traits=[
                PersonaTrait(trait=PersonalityTrait.METHODICAL, intensity=1.6),
                PersonaTrait(trait=PersonalityTrait.PRAGMATIC, intensity=1.3),
                PersonaTrait(trait=PersonalityTrait.ANALYTICAL, intensity=1.2),
            ],
            communication=CommunicationPreferences(
                style=CommunicationStyle.TECHNICAL,
                verbosity=0.7,
                formality=0.7,
                explain_reasoning=True,
            ),
            decision_making=DecisionMakingPreferences(
                risk_tolerance=RiskTolerance.MEDIUM,
                planning_depth=PlanningDepth.DETAILED,
                require_validation=True,
                max_iterations_per_task=12,
                prefer_tools=["execute_python", "save_file", "execute_shell"],
            ),
            prompts=PromptModifications(
                custom_instructions=(
                    "Follow software engineering best practices. "
                    "Write clean, maintainable code with proper error handling. "
                    "Consider edge cases and test your solutions."
                )
            ),
            is_builtin=True,
        )

    def _create_researcher_persona(self) -> AgentPersona:
        """Create the Researcher persona."""
        return AgentPersona(
            id="researcher",
            name="Researcher",
            description="Thorough, investigative agent focused on gathering comprehensive information",
            traits=[
                PersonaTrait(trait=PersonalityTrait.ANALYTICAL, intensity=1.5),
                PersonaTrait(trait=PersonalityTrait.METHODICAL, intensity=1.4),
                PersonaTrait(trait=PersonalityTrait.ADAPTIVE, intensity=1.1),
            ],
            communication=CommunicationPreferences(
                style=CommunicationStyle.DETAILED,
                verbosity=0.9,
                formality=0.7,
                explain_reasoning=True,
            ),
            decision_making=DecisionMakingPreferences(
                risk_tolerance=RiskTolerance.LOW,
                planning_depth=PlanningDepth.EXHAUSTIVE,
                require_validation=True,
                max_iterations_per_task=20,
                prefer_tools=["web_search", "read_file"],
            ),
            prompts=PromptModifications(
                custom_instructions=(
                    "Be thorough in your research. Gather information from multiple sources. "
                    "Cross-reference facts and verify accuracy. "
                    "Provide comprehensive answers with proper citations."
                )
            ),
            is_builtin=True,
        )

    def _create_creative_persona(self) -> AgentPersona:
        """Create the Creative persona."""
        return AgentPersona(
            id="creative",
            name="Creative Thinker",
            description="Innovative, exploratory agent focused on novel solutions",
            traits=[
                PersonaTrait(trait=PersonalityTrait.CREATIVE, intensity=1.8),
                PersonaTrait(trait=PersonalityTrait.BOLD, intensity=1.4),
                PersonaTrait(trait=PersonalityTrait.ADAPTIVE, intensity=1.3),
            ],
            communication=CommunicationPreferences(
                style=CommunicationStyle.CONVERSATIONAL,
                verbosity=0.7,
                formality=0.4,
                use_emojis=True,
                explain_reasoning=True,
            ),
            decision_making=DecisionMakingPreferences(
                risk_tolerance=RiskTolerance.HIGH,
                planning_depth=PlanningDepth.STANDARD,
                require_validation=True,
                max_iterations_per_task=12,
            ),
            prompts=PromptModifications(
                custom_instructions=(
                    "Think outside the box. Explore innovative and creative solutions. "
                    "Don't be afraid to try unconventional approaches. "
                    "Be playful and imaginative in your problem-solving."
                )
            ),
            is_builtin=True,
        )

    # =========================================================================
    # PERSONA MANAGEMENT
    # =========================================================================

    def get_persona(self, persona_id: str) -> Optional[AgentPersona]:
        """
        Get a persona by ID.

        Args:
            persona_id: Persona identifier

        Returns:
            AgentPersona if found, None otherwise
        """
        return self.personas.get(persona_id)

    def list_personas(self, builtin_only: bool = False) -> List[AgentPersona]:
        """
        List all available personas.

        Args:
            builtin_only: If True, only return built-in personas

        Returns:
            List of personas
        """
        personas = list(self.personas.values())

        if builtin_only:
            personas = [p for p in personas if p.is_builtin]

        return personas

    def create_persona(
        self,
        persona_id: str,
        name: str,
        description: str,
        traits: Optional[List[PersonalityTrait]] = None,
        communication_style: Optional[CommunicationStyle] = None,
        risk_tolerance: Optional[RiskTolerance] = None,
        planning_depth: Optional[PlanningDepth] = None,
        custom_instructions: Optional[str] = None,
    ) -> AgentPersona:
        """
        Create a custom persona.

        Args:
            persona_id: Unique identifier
            name: Human-readable name
            description: Persona description
            traits: Personality traits
            communication_style: Communication style
            risk_tolerance: Risk tolerance level
            planning_depth: Planning depth preference
            custom_instructions: Custom prompt instructions

        Returns:
            Created AgentPersona
        """
        # Build persona traits
        persona_traits = []
        if traits:
            for trait in traits:
                persona_traits.append(PersonaTrait(trait=trait, intensity=1.0))

        # Build communication preferences
        comm_prefs = CommunicationPreferences()
        if communication_style:
            comm_prefs.style = communication_style

        # Build decision-making preferences
        decision_prefs = DecisionMakingPreferences()
        if risk_tolerance:
            decision_prefs.risk_tolerance = risk_tolerance
        if planning_depth:
            decision_prefs.planning_depth = planning_depth

        # Build prompt modifications
        prompt_mods = PromptModifications()
        if custom_instructions:
            prompt_mods.custom_instructions = custom_instructions

        # Create persona
        persona = AgentPersona(
            id=persona_id,
            name=name,
            description=description,
            traits=persona_traits,
            communication=comm_prefs,
            decision_making=decision_prefs,
            prompts=prompt_mods,
            is_builtin=False,
        )

        # Store persona
        self.personas[persona_id] = persona

        logger.info(f"Created custom persona: {name} ({persona_id})")
        return persona

    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete a custom persona.

        Args:
            persona_id: Persona identifier

        Returns:
            True if deleted, False if not found or is built-in
        """
        persona = self.personas.get(persona_id)

        if not persona:
            return False

        if persona.is_builtin:
            logger.warning(f"Cannot delete built-in persona: {persona_id}")
            return False

        del self.personas[persona_id]
        logger.info(f"Deleted persona: {persona_id}")
        return True

    # =========================================================================
    # PERSONA APPLICATION
    # =========================================================================

    def apply_persona_to_prompt(
        self, base_prompt: str, persona: AgentPersona, phase: Optional[str] = None
    ) -> str:
        """
        Apply persona modifications to a prompt.

        Args:
            base_prompt: Original prompt text
            persona: Persona to apply
            phase: Optional phase name for phase-specific overrides

        Returns:
            Modified prompt with persona instructions
        """
        # Check for phase-specific override
        if phase and phase in persona.prompts.phase_prompts:
            # Use custom template for this phase
            # (Would integrate with prompt registry here)
            pass

        # Get persona additions
        additions = persona.get_system_prompt_addition()

        if not additions:
            return base_prompt

        # Combine
        modified = f"{additions}\n\n{base_prompt}"

        return modified

    def get_persona_config(self, persona: AgentPersona) -> Dict[str, Any]:
        """
        Get configuration dictionary for a persona.

        Returns configuration that can be passed to cognitive cycle
        or individual phases.

        Args:
            persona: The persona

        Returns:
            Configuration dictionary
        """
        return {
            "risk_tolerance": persona.decision_making.risk_tolerance.value,
            "planning_depth": persona.decision_making.planning_depth.value,
            "require_validation": persona.decision_making.require_validation,
            "max_iterations": persona.decision_making.max_iterations_per_task,
            "verbosity": persona.communication.verbosity,
            "explain_reasoning": persona.communication.explain_reasoning,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["PersonaManager"]
