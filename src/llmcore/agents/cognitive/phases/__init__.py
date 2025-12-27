# src/llmcore/agents/cognitive/phases/__init__.py
"""
Cognitive Phase Implementations.

This package contains the individual phase implementations for the
8-phase enhanced cognitive cycle:

1. PERCEIVE - Gather inputs and context from environment
2. PLAN - Create or update strategic action plan
3. THINK - Reason about next action (ReAct-style)
4. VALIDATE - Verify action safety before execution
5. ACT - Execute the chosen tool/action
6. OBSERVE - Process and interpret action results
7. REFLECT - Evaluate effectiveness and extract learnings
8. UPDATE - Apply state and memory updates

Each phase function has a consistent interface:
- Takes agent_state and phase-specific input
- Returns phase-specific output
- Supports optional tracer for observability
- Handles errors gracefully

Usage:
    >>> from llmcore.agents.cognitive.phases import (
    ...     perceive_phase,
    ...     plan_phase,
    ...     think_phase,
    ... )
    >>>
    >>> output = await perceive_phase(
    ...     agent_state=state,
    ...     perceive_input=PerceiveInput(goal="Test"),
    ...     memory_manager=memory_manager,
    ... )

References:
    - Technical Spec: Section 5.3.1-5.3.8 (Phase Implementations)
    - Dossiers: Steps 2.5-2.7
"""

# Phase 1: PERCEIVE
from .perceive import perceive_phase

# Phase 2: PLAN
from .plan import plan_phase

# Phase 3: THINK
from .think import think_phase

# Phase 4: VALIDATE
from .validate import validate_phase

# Phase 5: ACT
from .act import act_phase

# Phase 6: OBSERVE
from .observe import observe_phase

# Phase 7: REFLECT
from .reflect import reflect_phase

# Phase 8: UPDATE
from .update import update_phase


__all__ = [
    # Individual phase functions
    "perceive_phase",
    "plan_phase",
    "think_phase",
    "validate_phase",
    "act_phase",
    "observe_phase",
    "reflect_phase",
    "update_phase",
]
