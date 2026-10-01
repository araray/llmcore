# src/llmcore/routing/__init__.py
"""Routing for llmcore (see ``docs/ROUTING_SUBSYSTEM_SPEC.md``).

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
from .settings import ClassifierBias, RefusalPolicy, RoutingSettings, UnknownVerdictPolicy

__all__ = [
    "Balance",
    "Candidate",
    "Classification",
    "ClassifierBias",
    "FailureKind",
    "Finding",
    "Outcome",
    "RefusalPolicy",
    "RoutingPlan",
    "RoutingRequest",
    "RoutingSettings",
    "SelectionStrategy",
    "Target",
    "TargetHealth",
    "TransformAction",
    "TransformResult",
    "UnknownVerdictPolicy",
    "Verdict",
    "classify_failure",
]
