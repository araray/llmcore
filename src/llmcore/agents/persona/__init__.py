# src/llmcore/agents/persona/__init__.py
"""
Persona System for Darwin Layer 2.

Provides agent personality customization through personas.
"""

from .models import (
    AgentPersona,
    PersonalityTrait,
    CommunicationStyle,
    RiskTolerance,
    PlanningDepth,
    PersonaTrait,
    CommunicationPreferences,
    DecisionMakingPreferences,
    PromptModifications,
)

from .manager import PersonaManager

__all__ = [
    "AgentPersona",
    "PersonalityTrait",
    "CommunicationStyle",
    "RiskTolerance",
    "PlanningDepth",
    "PersonaTrait",
    "CommunicationPreferences",
    "DecisionMakingPreferences",
    "PromptModifications",
    "PersonaManager",
]
