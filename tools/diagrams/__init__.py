"""Documentation diagrams for llmcore, generated from code.

``python -m tools.diagrams`` writes ``docs/diagrams/*.svg``. See
``docs/Diagrams_usage.md`` for how to add or change one.
"""

from .diagrams import DIAGRAMS, build_all
from .dsl import Diagram

__all__ = ["DIAGRAMS", "Diagram", "build_all"]
