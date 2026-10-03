# src/llmcore/runtimes/models.py
"""Core types for the remote-GPU runtime subsystem (spec phase R1).

A runtime is unlike every other thing llmcore talks to. A provider is stateless
and bills per request; **a runtime bills per minute from the moment it is
assigned, whether or not anyone calls it.** Every type here is shaped by that:
spend ceilings are fields rather than options, state is written to disk so a
runtime can always be found and killed, and nothing in this module can start
anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "CostUnit",
    "ModelSpec",
    "Plan",
    "Quantization",
    "RuntimeHandle",
    "RuntimePhase",
    "RuntimeStatus",
]


class Quantization(StrEnum):
    """Weight quantization a recipe will load.

    The GGUF variants are separate members rather than one ``GGUF`` because
    sizing cannot work without them: Q4 and Q8 differ by a factor of two in
    weight bytes, which is routinely the difference between fitting a 24 GB
    card and not. A single ``GGUF`` member would force the sizer to guess.

    Attributes:
        NONE: Full precision as published (fp16/bf16).
        FP8: 8-bit floating point.
        INT8: 8-bit integer.
        INT4: 4-bit integer (bitsandbytes and similar).
        AWQ: Activation-aware weight quantization, 4-bit.
        GPTQ: GPTQ-quantized weights, typically 4-bit.
        GGUF: llama.cpp's container format, precision unspecified. Prefer a
            specific variant; this exists for repos that say no more.
        GGUF_Q4: GGUF at roughly 4 bits per weight (Q4_K_M, IQ4_XS, ...).
        GGUF_Q5: GGUF at roughly 5 bits per weight.
        GGUF_Q8: GGUF at roughly 8 bits per weight.
    """

    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF = "gguf"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q5 = "gguf_q5"
    GGUF_Q8 = "gguf_q8"


class CostUnit(StrEnum):
    """The currency a backend bills in.

    This exists because the subsystem's one safety net -- a spend ceiling that
    fires even while a runtime is *busy*, which the idle reaper cannot catch --
    has to compare like with like. Colab bills in compute units and publishes
    no exchange rate to dollars; gpu.ai and DeepInfra bill in dollars and
    publish live per-hour prices. Carrying the unit alongside every rate and
    every ceiling keeps llmcore from inventing a conversion between them.

    There is deliberately no ``convert()``. A fabricated exchange rate is the
    same class of bug as every other one this subsystem's cost model has had:
    a plausible number standing in for an absent implementation.

    Attributes:
        COMPUTE_UNIT: Colab's unit. Has no published dollar value.
        USD: US dollars, as billed by gpu.ai, DeepInfra and most rental APIs.
    """

    COMPUTE_UNIT = "compute-unit"
    USD = "usd"

    def amount(self, value: float) -> str:
        """Render *value* as an amount in this unit."""
        if self is CostUnit.USD:
            return f"${value:,.2f}"
        return f"{value:,.2f} compute units"

    def rate(self, value: float) -> str:
        """Render *value* as a per-hour burn rate in this unit."""
        if self is CostUnit.USD:
            return f"${value:,.2f}/hour"
        return f"{value:g} compute units/hour"


class RuntimePhase(StrEnum):
    """Lifecycle phase of a runtime.

    ``DEGRADED`` exists separately from ``FAILED`` because the distinction
    matters to the provider layer: a degraded runtime still **costs money** and
    still needs releasing, so it must not be quietly forgotten the way a
    never-started one can be.

    Attributes:
        PLANNED: Sized but not provisioned. Costs nothing.
        STARTING: Provisioning or bootstrapping. Already billing.
        READY: Serving and reachable.
        DEGRADED: Assigned and billing, but not serving.
        STOPPING: Being torn down.
        STOPPED: Released. No longer billing.
        DETACHED: llmcore has let go of it -- tunnel and keepalive stopped --
            but the compute was **not** released, so it is still billing. The
            state ``close()`` leaves a runtime in, because a process exiting is
            not a reason to destroy compute someone is paying for.
        FAILED: Bootstrap failed and the VM was released.
    """

    PLANNED = "planned"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DETACHED = "detached"
    FAILED = "failed"

    @property
    def is_billing(self) -> bool:
        """Whether a runtime in this phase is probably costing money.

        Used by the reaper and by teardown-on-exit. ``DEGRADED`` counts: a
        broken runtime is still an assigned one.
        """
        return self in {
            RuntimePhase.STARTING,
            RuntimePhase.READY,
            RuntimePhase.DEGRADED,
            RuntimePhase.STOPPING,
            # Detached means llmcore stopped watching, not that the VM stopped.
            # Counting it as not-billing would hide exactly the leak this
            # subsystem's safety rules exist to prevent.
            RuntimePhase.DETACHED,
        }

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is expected."""
        return self in {RuntimePhase.STOPPED, RuntimePhase.FAILED}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What the caller wants served.

    Attributes:
        repo_id: Hugging Face repository id.
        revision: Git revision, branch or tag. ``None`` means the default.
        context_length: Desired context window in tokens.
        quantization: Requested quantization, or ``None`` to let the sizer pick.
        trust_remote_code: Whether the recipe may execute repo-provided code.
            Defaults to ``False``: running arbitrary code from a model repo is a
            decision the caller should have to make explicitly.
        extra: Recipe-specific knobs, passed through verbatim.
    """

    repo_id: str
    revision: str | None = None
    context_length: int = 8192
    quantization: Quantization | None = None
    trust_remote_code: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError(
                f"repo_id must look like 'owner/name', got {self.repo_id!r}."
            )
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")


@dataclass(frozen=True, slots=True)
class Plan:
    """A sizing decision: what it would take to serve a :class:`ModelSpec`.

    A plan is the output of ``estimate()``, which is **read-only and free**. It
    is deliberately a separate type from :class:`RuntimeHandle` so that sizing
    can be inspected, logged and approved without anything being provisioned.

    Attributes:
        spec: What this plan serves.
        sku: GPU SKU chosen (e.g. ``"L4"``, ``"A100-40"``).
        recipe: Server recipe (``"vllm"``, ``"llamacpp"``, ...).
        quantization: Quantization the recipe will load.
        vram_required_gb: Estimated VRAM for weights plus KV cache plus
            headroom.
        vram_available_gb: What is actually **usable** on the chosen SKU --
            the device total after ``gpu_memory_utilization`` and the headroom
            reserve, which is the number the fit decision compared against.
            Reporting the sticker VRAM here would make a plan look like it had
            several GB more room than the sizer believed.
        context_length: Context the plan is sized for, which may be below the
            request when the model could not otherwise fit.
        fits: Whether the model fits the chosen SKU at all.
        notes: Human-readable reasoning, for display before approving spend.
        estimated_cost_per_hour: Indicative cost, when known.
        cost_unit: What ``estimated_cost_per_hour`` is denominated in. ``None``
            when no rate is known.
        gpu_count: Devices the plan assumes. ``1`` for Colab, which has no
            other option; rental APIs sell multi-GPU nodes, and on those
            ``vram_available_gb`` is the aggregate across all of them.
        region: Where the plan priced its compute, when the backend has
            regions and the price varies by them. A plan that cannot say which
            region it priced cannot promise the price it quoted.
        offering_id: Backend-side identifier for the exact catalogue row this
            plan priced, so ``up()`` can pin the launch to it. A catalogue row
            is a quote, not a booking: without pinning, a backend is free to
            place the instance on a pricier offering than the one approved.
    """

    spec: ModelSpec
    sku: str
    recipe: str = "vllm"
    quantization: Quantization = Quantization.NONE
    vram_required_gb: float = 0.0
    vram_available_gb: float = 0.0
    context_length: int = 8192
    fits: bool = True
    notes: tuple[str, ...] = ()
    estimated_cost_per_hour: float | None = None
    cost_unit: CostUnit | None = None
    gpu_count: int = 1
    region: str | None = None
    offering_id: str | None = None

    @property
    def shape(self) -> str:
        """The compute shape as one display string.

        Catalogues disagree about whether the device count belongs in the SKU
        name: DeepInfra sells ``"1xA100-80GB"``, gpu.ai sells ``a100_80gb``
        with a separate count, and Colab sells ``"L4"`` with no count at all.
        Prefixing unconditionally produced ``"1x1xA100-80GB"``, so the prefix
        is added only when the name does not already carry it.
        """
        if self.gpu_count > 1 and not self.sku.lower().startswith(f"{self.gpu_count}x"):
            return f"{self.gpu_count}x{self.sku}"
        return self.sku

    @property
    def burn_rate(self) -> str | None:
        """The hourly rate rendered in its own unit, or ``None`` if unknown.

        Unknown is returned as ``None`` rather than a zero or a bare number:
        every caller of this displays it before asking for spend approval, and
        "0" would read as free.
        """
        if self.estimated_cost_per_hour is None or self.cost_unit is None:
            return None
        return self.cost_unit.rate(self.estimated_cost_per_hour)

    @property
    def headroom_gb(self) -> float:
        """VRAM left over on the chosen SKU. Negative means it does not fit."""
        return self.vram_available_gb - self.vram_required_gb

    def with_notes(self, *notes: str) -> Plan:
        """Return a copy with *notes* appended."""
        return replace(self, notes=(*self.notes, *notes))


@dataclass(slots=True)
class RuntimeHandle:
    """A provisioned runtime, and everything needed to find and kill it.

    Mutable by design: phase and deadlines change over a runtime's life, and the
    handle is the thing persisted to disk so a leaked VM can always be
    recovered. Serialized by :meth:`to_dict` into inspectable JSON — a human
    with a text editor has to be able to see what is running and what it costs.

    Attributes:
        name: Caller-chosen name; also the provider instance name once attached.
        runtime: Backend that owns it (``"colab"``).
        external_id: Backend-side identifier (e.g. a Colab session id).
        base_url: OpenAI-compatible endpoint, usually a local tunnel port.
        served_model: Model id the server was told to serve.
        api_style: Wire protocol, which decides the provider type to attach.
        recipe: Server recipe in use.
        sku: GPU SKU assigned.
        phase: Current lifecycle phase.
        started_at: When provisioning began.
        idle_deadline: When the reaper stops it for inactivity.
        hard_deadline: When the reaper stops it regardless of activity.
        last_activity_at: Last observed request, for idle reaping.
        max_spend: Optional ceiling on what this runtime may consume, in
            :attr:`spend_unit`. The only guard against the expensive failure
            mode: a runtime *busy* in a loop, which the idle reaper never
            touches.
        spend_used: Consumption so far, when the backend reports it.
        spend_unit: What ``max_spend`` and ``spend_used`` are denominated in.
            Both are compared directly, so a backend must report them in the
            same unit it accepts the ceiling in.
        state_path: Where this handle is persisted.
        error: Why it failed or degraded.
        metadata: Backend-specific extras.
    """

    name: str
    runtime: str
    external_id: str
    base_url: str
    served_model: str
    api_style: str = "openai"
    recipe: str = "vllm"
    sku: str = ""
    phase: RuntimePhase = RuntimePhase.STARTING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    idle_deadline: datetime | None = None
    hard_deadline: datetime | None = None
    last_activity_at: datetime | None = None
    max_spend: float | None = None
    spend_used: float | None = None
    spend_unit: CostUnit = CostUnit.COMPUTE_UNIT
    state_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- spend ceilings ------------------------------------------------

    def touch(self, when: datetime | None = None) -> None:
        """Record activity and push the idle deadline out."""
        now = when or datetime.now(timezone.utc)
        self.last_activity_at = now
        if self.idle_deadline is not None and self.idle_minutes:
            self.idle_deadline = now + timedelta(minutes=self.idle_minutes)

    @property
    def idle_minutes(self) -> float | None:
        """Configured idle window, recovered from metadata."""
        value = self.metadata.get("idle_minutes")
        return float(value) if value else None

    def expired_reason(self, now: datetime | None = None) -> str | None:
        """Why this runtime should be reaped, or ``None`` to keep it.

        Checks the hard deadline before the idle one, and the spend ceiling
        before either: an idle reaper does not protect against a runtime that is
        *busy* in a loop, which is the expensive failure mode.
        """
        moment = now or datetime.now(timezone.utc)
        if (
            self.max_spend is not None
            and self.spend_used is not None
            and self.spend_used >= self.max_spend
        ):
            return (
                f"spend ceiling reached "
                f"({self.spend_unit.amount(self.spend_used)} of "
                f"{self.spend_unit.amount(self.max_spend)})"
            )
        if self.hard_deadline is not None and moment >= self.hard_deadline:
            return f"hard lifetime deadline passed ({self.hard_deadline.isoformat()})"
        if self.idle_deadline is not None and moment >= self.idle_deadline:
            return f"idle since {(self.last_activity_at or self.started_at).isoformat()}"
        return None

    # --- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of this handle."""

        def _dt(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        return {
            "name": self.name,
            "runtime": self.runtime,
            "external_id": self.external_id,
            "base_url": self.base_url,
            "served_model": self.served_model,
            "api_style": self.api_style,
            "recipe": self.recipe,
            "sku": self.sku,
            "phase": str(self.phase),
            "started_at": _dt(self.started_at),
            "idle_deadline": _dt(self.idle_deadline),
            "hard_deadline": _dt(self.hard_deadline),
            "last_activity_at": _dt(self.last_activity_at),
            "max_spend": self.max_spend,
            "spend_used": self.spend_used,
            "spend_unit": str(self.spend_unit),
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Return indented JSON, so the state file is human-readable."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, state_path: Path | None = None) -> RuntimeHandle:
        """Rebuild a handle from :meth:`to_dict` output.

        Unknown phases become :attr:`RuntimePhase.DEGRADED` rather than raising:
        a state file written by a newer llmcore still describes a VM that is
        burning money, and refusing to parse it would hide that.
        """

        def _dt(value: Any) -> datetime | None:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None

        try:
            phase = RuntimePhase(str(payload.get("phase", "degraded")))
        except ValueError:
            phase = RuntimePhase.DEGRADED

        try:
            unit = CostUnit(str(payload.get("spend_unit", CostUnit.COMPUTE_UNIT)))
        except ValueError:
            unit = CostUnit.COMPUTE_UNIT

        started = _dt(payload.get("started_at")) or datetime.now(timezone.utc)
        return cls(
            name=str(payload.get("name", "")),
            runtime=str(payload.get("runtime", "")),
            external_id=str(payload.get("external_id", "")),
            base_url=str(payload.get("base_url", "")),
            served_model=str(payload.get("served_model", "")),
            api_style=str(payload.get("api_style", "openai")),
            recipe=str(payload.get("recipe", "vllm")),
            sku=str(payload.get("sku", "")),
            phase=phase,
            started_at=started,
            idle_deadline=_dt(payload.get("idle_deadline")),
            hard_deadline=_dt(payload.get("hard_deadline")),
            last_activity_at=_dt(payload.get("last_activity_at")),
            # Pre-CostUnit state files named these for Colab's unit. A state
            # file describes something that may still be billing, so an older
            # one has to keep loading rather than lose its ceiling.
            max_spend=payload.get("max_spend", payload.get("max_compute_units")),
            spend_used=payload.get("spend_used", payload.get("compute_units_used")),
            spend_unit=unit,
            state_path=state_path,
            error=payload.get("error"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """A point-in-time view of one runtime, for display.

    Separate from :class:`RuntimeHandle` because status is what a *caller* is
    shown — it adds derived values like elapsed time and includes whether the
    endpoint currently answers, which the handle itself cannot know.

    Attributes:
        name: Runtime name.
        runtime: Backend name.
        phase: Current phase.
        served_model: Model being served.
        sku: GPU SKU.
        base_url: Endpoint.
        uptime_seconds: Seconds since provisioning began.
        reachable: Whether a liveness probe succeeded; ``None`` if not probed.
        attached: Whether a provider instance is currently registered for it.
        expires_in_seconds: Seconds until the nearest deadline, when set.
        spend_used: What this runtime has consumed so far, when the backend
            reports it. For anything billed by the hour this is the first
            question an operator asks, and it was the one thing a status could
            not answer.
        max_spend: The ceiling it is being compared against, when set.
        spend_unit: What both are denominated in.
        error: Failure detail.
    """

    name: str
    runtime: str
    phase: RuntimePhase
    served_model: str = ""
    sku: str = ""
    base_url: str = ""
    uptime_seconds: float = 0.0
    reachable: bool | None = None
    attached: bool = False
    expires_in_seconds: float | None = None
    spend_used: float | None = None
    max_spend: float | None = None
    spend_unit: CostUnit = CostUnit.COMPUTE_UNIT
    error: str | None = None

    @property
    def spend_so_far(self) -> str | None:
        """Accrued spend rendered in its own unit, or ``None`` if unreported.

        ``None`` rather than ``"$0.00"``: a backend that does not report
        consumption has not told us it is free.
        """
        if self.spend_used is None:
            return None
        return self.spend_unit.amount(self.spend_used)

    @classmethod
    def from_handle(
        cls,
        handle: RuntimeHandle,
        *,
        reachable: bool | None = None,
        attached: bool = False,
        now: datetime | None = None,
    ) -> RuntimeStatus:
        """Build a status view from *handle*."""
        moment = now or datetime.now(timezone.utc)
        deadlines = [d for d in (handle.idle_deadline, handle.hard_deadline) if d]
        expires = (min(deadlines) - moment).total_seconds() if deadlines else None
        return cls(
            name=handle.name,
            runtime=handle.runtime,
            phase=handle.phase,
            served_model=handle.served_model,
            sku=handle.sku,
            base_url=handle.base_url,
            uptime_seconds=max(0.0, (moment - handle.started_at).total_seconds()),
            reachable=reachable,
            attached=attached,
            expires_in_seconds=expires,
            spend_used=handle.spend_used,
            max_spend=handle.max_spend,
            spend_unit=handle.spend_unit,
            error=handle.error,
        )
