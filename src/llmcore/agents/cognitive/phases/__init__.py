# src/llmcore/agents/cognitive/phases/__init__.py
"""
Cognitive Phase Implementations.

Individual phase implementations for the 8-phase cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

Context Synthesis:
    The PERCEIVE phase supports an optional ContextSynthesizer for
    sophisticated multi-source context assembly. Use ``create_default_synthesizer``
    to create a pre-configured synthesizer with all five context source tiers.

    The ``CognitiveCycle`` orchestrator can be initialized with a synthesizer
    to automatically use synthesis mode in all PERCEIVE phases::

        from llmcore.agents.cognitive.phases import (
            CognitiveCycle,
            create_default_synthesizer,
        )

        synthesizer = create_default_synthesizer(goal_manager=gm, skill_loader=sl)
        cycle = CognitiveCycle(
            provider_manager=pm,
            memory_manager=mm,
            storage_manager=sm,
            tool_manager=tm,
            context_synthesizer=synthesizer,
        )

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §10 (Agent System)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from .act import act_phase
from .cycle import CognitiveCycle, StreamingIterationResult
from .observe import observe_phase
from .perceive import create_default_synthesizer, perceive_phase
from .plan import plan_phase
from .reflect import reflect_phase
from .think import think_phase
from .update import update_phase
from .validate import validate_phase

__all__ = [
    # Orchestrator
    "CognitiveCycle",
    "StreamingIterationResult",
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
