# src/llmcore/routing/__init__.py
"""Routing for llmcore (see the routing subsystem design spec).

Five layers that compose, each usable without the others:

* **Target** — which provider, model and parameters. Addressable as a spec
  string, and autoprovisioned when absent from config.
* **Pool** — a set of interchangeable targets, with failover and selection.
* **Lane** — a named destination a classifier picks.
* **Cascade** — answer cheaply, verify, escalate only if it fell short.
* **Transform** — what must not leave this machine.

Config is a warm-up rather than a cage: every knob resolves config → env →
per-request, and the request always wins.
"""

from .cards import context_window, estimate_cost_usd
from .lanes import Lane, parse_lanes
from .manager import RoutingManager, RoutingResult
from .models import (
    Balance,
    Candidate,
    Classification,
    FailureKind,
    Finding,
    Outcome,
    RoutingPlan,
    RoutingRequest,
    SelectionStrategy,
    Target,
    TargetHealth,
    TransformAction,
    TransformResult,
    Verdict,
    classify_failure,
)
from .pools import Pool, select
from .protocols import (
    AUTHORITIES,
    COST_HINTS,
    BalanceProbe,
    PromptTransform,
    RequestClassifier,
    ResponseVerifier,
    RoutingStateStore,
)
from .settings import ClassifierBias, RefusalPolicy, RoutingSettings, UnknownVerdictPolicy
from .state import InMemoryRoutingState

__all__ = [
    "AUTHORITIES",
    "COST_HINTS",
    "Balance",
    "BalanceProbe",
    "Candidate",
    "Classification",
    "ClassifierBias",
    "FailureKind",
    "Finding",
    "InMemoryRoutingState",
    "Lane",
    "Outcome",
    "Pool",
    "PromptTransform",
    "RefusalPolicy",
    "RequestClassifier",
    "ResponseVerifier",
    "RoutingManager",
    "RoutingPlan",
    "RoutingRequest",
    "RoutingResult",
    "RoutingSettings",
    "RoutingStateStore",
    "SelectionStrategy",
    "Target",
    "TargetHealth",
    "TransformAction",
    "TransformResult",
    "UnknownVerdictPolicy",
    "Verdict",
    "classify_failure",
    "context_window",
    "estimate_cost_usd",
    "parse_lanes",
    "select",
]
