# src/llmcore/agents/cognitive/phases/__init__.py
"""
Cognitive Phase Implementations.

Individual phase implementations for the 8-phase cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

Context Synthesis:
    The PERCEIVE phase supports an optional ContextSynthesizer for
    sophisticated multi-source context assembly. Use ``create_default_synthesizer``
    to create a pre-configured synthesizer with all five context source tiers.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §10 (Agent System)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from .act import act_phase
from .observe import observe_phase
from .perceive import create_default_synthesizer, perceive_phase
from .plan import plan_phase
from .reflect import reflect_phase
from .think import think_phase
from .update import update_phase
from .validate import validate_phase

__all__ = [
    # Phase functions
    "act_phase",
    "observe_phase",
    "perceive_phase",
    "plan_phase",
    "reflect_phase",
    "think_phase",
    "update_phase",
    "validate_phase",
    # Utilities
    "create_default_synthesizer",
]
