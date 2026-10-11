# src/llmcore/runtimes/__init__.py
"""Remote GPU runtimes for llmcore (see the Colab runtime design spec).

A runtime provisions compute somewhere else, serves an open-weights model on it,
and attaches the resulting OpenAI-compatible endpoint as a provider instance —
so a remotely served model is reachable through the same ``llm.chat()`` as any
hosted API.

**This subsystem is off by default and never provisions implicitly.** Unlike
every other provider, a runtime bills per minute from the moment it is assigned,
whether or not anyone calls it, so `LLMCore.create()` cannot start one no matter
what the config says.
"""

from .accounts import (
    AccountBalance,
    SpendingLimit,
    SshKey,
    UsageBucket,
    UsageReport,
    openrouter_balance,
)
from .diagnostics import Check, Severity, run_diagnostics
from .ledger import DailyTotal, SpendEntry, SpendLedger
from .manager import (
    DailyBudgetExceededError,
    RuntimeError_,
    RuntimeManager,
    SpendNotConfirmedError,
)
from .models import (
    CostUnit,
    ModelSpec,
    Ownership,
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
    "AccountBalance",
    "Check",
    "ComputeRuntime",
    "CostUnit",
    "DailyBudgetExceededError",
    "DailyTotal",
    "ModelSpec",
    "Ownership",
    "Plan",
    "Quantization",
    "RuntimeError_",
    "RuntimeHandle",
    "RuntimeManager",
    "RuntimePhase",
    "RuntimeStateStore",
    "RuntimeStatus",
    "Severity",
    "SpendEntry",
    "SpendLedger",
    "SpendNotConfirmedError",
    "SpendingLimit",
    "SshKey",
    "UsageBucket",
    "UsageReport",
    "openrouter_balance",
    "run_diagnostics",
]
