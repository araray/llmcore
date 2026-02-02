# src/llmcore/agents/persona/__init__.py
"""
Persona System for Darwin Layer 2.

Provides agent personality customization through personas.
"""

from .manager import PersonaManager
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

__all__ = [
    "AgentPersona",
    "CommunicationPreferences",
    "CommunicationStyle",
    "DecisionMakingPreferences",
    "PersonaManager",
    "PersonaTrait",
    "PersonalityTrait",
    "PlanningDepth",
    "PromptModifications",
    "RiskTolerance",
]
