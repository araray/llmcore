# src/llmcore/runtimes/__init__.py
"""Remote GPU runtimes for llmcore (see ``docs/COLAB_RUNTIME_SPEC.md``).

A runtime provisions compute somewhere else, serves an open-weights model on it,
and attaches the resulting OpenAI-compatible endpoint as a provider instance —
so a remotely served model is reachable through the same ``llm.chat()`` as any
hosted API.

**This subsystem is off by default and never provisions implicitly.** Unlike
every other provider, a runtime bills per minute from the moment it is assigned,
whether or not anyone calls it, so `LLMCore.create()` cannot start one no matter
what the config says.
"""

from .manager import RuntimeError_, RuntimeManager, SpendNotConfirmedError
from .models import (
    ModelSpec,
    Plan,
    Quantization,
    RuntimeHandle,
    RuntimePhase,
    RuntimeStatus,
)
from .protocols import ComputeRuntime
from .state import DEFAULT_STATE_DIR, RuntimeStateStore

__all__ = [
    "DEFAULT_STATE_DIR",
    "ComputeRuntime",
    "ModelSpec",
    "Plan",
    "Quantization",
    "RuntimeError_",
    "RuntimeHandle",
    "RuntimeManager",
    "RuntimePhase",
    "RuntimeStateStore",
    "RuntimeStatus",
    "SpendNotConfirmedError",
]
