# src/llmcore/runtimes/manager.py
"""The runtime manager (spec phase R1).

This is where the spec's safety model is enforced rather than merely described.
Four of its five rules are implemented here:

1. **No implicit spend.** Construction and :meth:`estimate` cannot provision.
   :meth:`up` refuses unless the subsystem is enabled *and* spend is confirmed.
2. **No implicit persistence of spend.** State is written before provisioning
   begins, so a crash mid-bootstrap still leaves a killable record.
3. **Bounded by default.** Idle and hard deadlines come from config defaults,
   not from the caller remembering to pass them.
4. **Fail closed.** If attach fails after ``up`` succeeded, the runtime is torn
   down rather than left burning.

Rule 5 (no implicit secrets) belongs to the backends, which own credentials.

The other half of the job is provider attachment. A runtime's endpoint is
OpenAI-compatible, so no new provider class is needed: the manager registers a
``vllm``-type instance into the live :class:`~llmcore.providers.manager.ProviderManager`
under the runtime's name, and ``llm.chat(..., provider_name=name)`` just works.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from ..exceptions import ConfigError, LLMCoreError
from .models import ModelSpec, Plan, Quantization, RuntimeHandle, RuntimePhase, RuntimeStatus
from .protocols import ComputeRuntime
from .state import DEFAULT_STATE_DIR, RuntimeStateStore

logger = logging.getLogger(__name__)

__all__ = ["RuntimeError_", "RuntimeManager", "SpendNotConfirmedError"]

#: Provider type attached for each wire protocol a recipe may speak.
_API_STYLE_PROVIDERS: dict[str, str] = {
    "openai": "vllm",
    "vllm": "vllm",
}


class RuntimeError_(LLMCoreError):
    """A runtime operation failed."""


class SpendNotConfirmedError(RuntimeError_):
    """``up()`` was called without confirming that money will be spent.

    Deliberately its own type so a caller can catch exactly this and prompt,
    rather than string-matching a generic error.
    """


class _ConfigView:
    """Wraps a ``config.get`` callable so a backend can take ``.get(...)``.

    The manager is handed a bare accessor (it is constructed from
    ``self.config.get``), while a backend wants something config-shaped. One
    three-line adapter beats threading two different parameter styles through
    every backend.
    """

    __slots__ = ("get",)

    def __init__(self, get: Callable[[str, Any], Any]) -> None:
        self.get = get


class RuntimeManager:
    """Provisions, tracks and attaches remote GPU runtimes.

    Args:
        backends: Mapping of backend name to :class:`ComputeRuntime`.
        provider_manager: Live provider manager to attach instances into.
            Optional — sizing and status work without one.
        config_get: A ``config.get(key, default)`` accessor.
    """

    def __init__(
        self,
        backends: dict[str, ComputeRuntime] | None = None,
        *,
        provider_manager: Any = None,
        config_get: Callable[[str, Any], Any] | None = None,
    ) -> None:
        get = config_get or (lambda _k, d=None: d)
        self._get = get
        self._backends: dict[str, ComputeRuntime] = dict(backends or {})
        self._providers = provider_manager

        self._enabled = bool(get("runtimes.enabled", False))
        self._default_backend = str(get("runtimes.default_backend", "colab"))
        self._confirm_spend = bool(get("runtimes.defaults.confirm_spend", True))
        self._idle_minutes = float(get("runtimes.defaults.idle_minutes", 45) or 0)
        self._max_lifetime_minutes = float(
            get("runtimes.defaults.max_lifetime_minutes", 240) or 0
        )
        self._recipe = str(get("runtimes.defaults.recipe", "vllm"))
        self.state = RuntimeStateStore(get("runtimes.state_dir", DEFAULT_STATE_DIR))

        #: Runtimes this process started or adopted, by name.
        self._handles: dict[str, RuntimeHandle] = {}
        #: Names this manager registered as provider instances.
        self._attached: set[str] = set()

        # Register the Colab backend unless the caller supplied its own. Doing
        # this in the constructor is safe because constructing a backend
        # contacts nothing: the CLI is discovered on first use, so
        # LLMCore.create() still cannot reach a provisioning API.
        if "colab" not in self._backends:
            try:
                from .colab import ColabRuntime

                self._backends["colab"] = ColabRuntime(
                    config=_ConfigView(get), state_store=self.state
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("The Colab backend could not be registered", exc_info=True)

        logger.debug(
            "RuntimeManager initialized (enabled=%s, backends=%s, confirm_spend=%s).",
            self._enabled,
            ", ".join(sorted(self._backends)) or "none",
            self._confirm_spend,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the subsystem may provision at all."""
        return self._enabled

    @property
    def backends(self) -> list[str]:
        """Registered backend names."""
        return sorted(self._backends)

    def register_backend(self, backend: ComputeRuntime) -> None:
        """Add *backend* under its own ``name``."""
        if not isinstance(backend, ComputeRuntime):
            raise ConfigError(
                f"{type(backend).__name__} does not satisfy the ComputeRuntime protocol."
            )
        self._backends[backend.name] = backend

    def _backend_for(self, name: str | None = None) -> ComputeRuntime:
        """Resolve a backend, defaulting to the configured one."""
        key = (name or self._default_backend).lower()
        backend = self._backends.get(key)
        if backend is None:
            available = ", ".join(sorted(self._backends)) or "none"
            raise ConfigError(
                f"No runtime backend '{key}' is registered. Available: {available}."
            )
        return backend

    def _require_enabled(self, operation: str) -> None:
        """Refuse a spending operation when the subsystem is disabled.

        Raises:
            RuntimeError_: If ``runtimes.enabled`` is not true.
        """
        if not self._enabled:
            raise RuntimeError_(
                f"Cannot {operation}: the runtimes subsystem is disabled. Set "
                f"[runtimes] enabled = true. It is off by default because a runtime "
                f"bills per minute from the moment it is assigned."
            )

    # ------------------------------------------------------------------
    # Sizing — read-only and free
    # ------------------------------------------------------------------

    async def estimate(
        self,
        repo_id: str,
        *,
        backend: str | None = None,
        context_length: int = 8192,
        quantization: Quantization | str | None = None,
        revision: str | None = None,
        **extra: Any,
    ) -> Plan:
        """Size a model without provisioning anything.

        Free and read-only, and deliberately usable while the subsystem is
        disabled: deciding whether to spend should not require enabling spend.
        """
        quant = Quantization(quantization) if isinstance(quantization, str) else quantization
        spec = ModelSpec(
            repo_id=repo_id,
            revision=revision,
            context_length=context_length,
            quantization=quant,
            extra=dict(extra),
        )
        return await self._backend_for(backend).estimate(spec)

    # ------------------------------------------------------------------
    # Provisioning — this spends money
    # ------------------------------------------------------------------

    async def up(
        self,
        repo_id: str,
        *,
        name: str,
        backend: str | None = None,
        plan: Plan | None = None,
        confirm_spend: bool | None = None,
        idle_minutes: float | None = None,
        max_lifetime_minutes: float | None = None,
        max_compute_units: float | None = None,
        attach: bool = True,
        **estimate_kwargs: Any,
    ) -> RuntimeHandle:
        """Provision a runtime and serve a model on it. **This spends money.**

        Args:
            repo_id: Hugging Face repo to serve.
            name: Name for the runtime, and for the provider instance attached.
            backend: Backend to use; the configured default otherwise.
            plan: A plan from :meth:`estimate`; one is computed when omitted.
            confirm_spend: Pass ``True`` to acknowledge billing. Required while
                ``runtimes.defaults.confirm_spend`` is on, which it is by
                default — provisioning is not something to do by accident.
            idle_minutes: Idle window before reaping. ``0`` disables it.
            max_lifetime_minutes: Hard kill regardless of activity. ``0``
                disables it.
            max_compute_units: Optional ceiling on backend compute units. An
                idle reaper does not protect against a runtime busy in a loop.
            attach: Register a provider instance for the endpoint.

        Raises:
            SpendNotConfirmedError: If confirmation is required and absent.
            RuntimeError_: If the subsystem is disabled, the name is taken, or
                provisioning fails.
        """
        self._require_enabled("provision a runtime")

        if name in self._handles and self._handles[name].phase.is_billing:
            raise RuntimeError_(
                f"A runtime named '{name}' is already running "
                f"({self._handles[name].phase}). Call down('{name}') first, or use "
                f"another name."
            )

        confirmed = self._confirm_spend is False if confirm_spend is None else confirm_spend
        if not confirmed:
            raise SpendNotConfirmedError(
                f"Refusing to provision '{name}': this starts billable compute that "
                f"costs money per minute until it is stopped. Pass confirm_spend=True "
                f"to proceed, or set [runtimes.defaults] confirm_spend = false to stop "
                f"being asked."
            )

        runtime = self._backend_for(backend)
        if plan is None:
            plan = await self.estimate(repo_id, backend=backend, **estimate_kwargs)
        if not plan.fits:
            raise RuntimeError_(
                f"Refusing to provision '{name}': {repo_id} does not fit the chosen "
                f"SKU {plan.sku} ({plan.vram_required_gb:.1f} GB needed, "
                f"{plan.vram_available_gb:.1f} GB available). "
                f"{' '.join(plan.notes)}"
            )

        idle = self._idle_minutes if idle_minutes is None else float(idle_minutes)
        hard = (
            self._max_lifetime_minutes
            if max_lifetime_minutes is None
            else float(max_lifetime_minutes)
        )

        handle = await runtime.up(plan, name=name)
        self._apply_limits(handle, idle=idle, hard=hard, units=max_compute_units)
        # Persisted immediately: the dangerous window is a crash between
        # assignment and bookkeeping, where money burns and nothing knows.
        self.state.save(handle)
        self._handles[name] = handle

        if attach:
            try:
                self.attach(handle)
            except Exception as e:
                # Fail closed. A runtime we cannot reach is still billing, so
                # release it rather than leaving it for the reaper.
                logger.error("Attach failed for '%s'; tearing the runtime down.", name)
                handle.error = f"attach failed: {e}"
                handle.phase = RuntimePhase.FAILED
                self.state.save(handle)
                await self.down(name)
                raise RuntimeError_(
                    f"Provisioned '{name}' but could not attach it as a provider, so it "
                    f"was released rather than left running. Cause: {e}"
                ) from e

        return handle

    def _apply_limits(
        self,
        handle: RuntimeHandle,
        *,
        idle: float,
        hard: float,
        units: float | None,
    ) -> None:
        """Stamp spend ceilings onto *handle*."""
        now = handle.started_at or datetime.now(timezone.utc)
        if idle > 0:
            handle.idle_deadline = now + timedelta(minutes=idle)
            handle.metadata["idle_minutes"] = idle
        if hard > 0:
            handle.hard_deadline = now + timedelta(minutes=hard)
            handle.metadata["max_lifetime_minutes"] = hard
        if units is not None:
            handle.max_compute_units = float(units)
        handle.last_activity_at = now

    # ------------------------------------------------------------------
    # Provider attachment
    # ------------------------------------------------------------------

    def attach(self, handle: RuntimeHandle) -> str:
        """Register *handle*'s endpoint as a provider instance.

        Returns the instance name. No new provider class is needed because the
        endpoint is OpenAI-compatible; ``api_style`` decides which existing
        provider type is used, so a future recipe speaking a different protocol
        attaches a different type without touching this layer.
        """
        if self._providers is None:
            raise RuntimeError_(
                "Cannot attach a runtime: no ProviderManager is available to this "
                "RuntimeManager."
            )
        provider_type = _API_STYLE_PROVIDERS.get(handle.api_style)
        if provider_type is None:
            raise RuntimeError_(
                f"Runtime '{handle.name}' reports api_style={handle.api_style!r}, which "
                f"maps to no provider type. Known: "
                f"{', '.join(sorted(_API_STYLE_PROVIDERS))}."
            )

        self._providers.register_instance(
            handle.name,
            provider_type,
            {
                "base_url": handle.base_url,
                "default_model": handle.served_model,
                # vLLM servers accept any bearer token; the endpoint is a local
                # tunnel, so there is no secret to carry here.
                "api_key": handle.metadata.get("api_key", "runtime-local"),
            },
            # Marked ephemeral so the provider manager knows this instance was
            # created by a subsystem rather than configured, and tears it down
            # with the rest on close_all(). Without the flag the instance would
            # outlive its runtime and hand callers a dead endpoint.
            ephemeral=True,
            # A runtime may legitimately be re-attached after a reconnect, and
            # refusing that would leave the endpoint unreachable.
            replace=True,
        )
        self._attached.add(handle.name)
        logger.info(
            "Attached runtime '%s' as provider instance (type=%s, model=%s).",
            handle.name,
            provider_type,
            handle.served_model,
        )
        return handle.name

    async def detach(self, name: str) -> bool:
        """Unregister the provider instance for *name*, if attached."""
        if name not in self._attached or self._providers is None:
            return False
        try:
            await self._providers.unregister_instance(name)
        except Exception as e:  # detaching must not block teardown
            logger.warning("Error unregistering provider instance '%s': %s", name, e)
        self._attached.discard(name)
        return True

    @property
    def attached(self) -> list[str]:
        """Names currently registered as provider instances."""
        return sorted(self._attached)

    # ------------------------------------------------------------------
    # Inspection and teardown
    # ------------------------------------------------------------------

    async def status(
        self, name: str | None = None, *, include_disk: bool = True
    ) -> list[RuntimeStatus]:
        """Report on tracked runtimes.

        With *include_disk*, records written by other processes are included —
        which is the point: a runtime started by a previous session is still
        costing money, and status is how someone finds it.
        """
        handles: dict[str, RuntimeHandle] = {}
        if include_disk:
            for handle in self.state.load_all():
                handles[handle.name] = handle
        handles.update(self._handles)

        if name is not None:
            handles = {k: v for k, v in handles.items() if k == name}

        return [
            RuntimeStatus.from_handle(h, attached=h.name in self._attached)
            for h in sorted(handles.values(), key=lambda h: h.started_at, reverse=True)
        ]

    async def down(self, name: str, *, release: bool = True, forget: bool = True) -> bool:
        """Stop a runtime and detach its provider instance.

        Idempotent and best-effort by design: this is the function someone
        reaches for when things are already wrong, so it detaches, releases and
        clears state even if individual steps fail.
        """
        handle = self._handles.get(name) or self.state.load(name)
        await self.detach(name)

        released = False
        if handle is not None and release:
            try:
                backend = self._backend_for(handle.runtime)
            except ConfigError:
                logger.warning(
                    "Runtime '%s' was started by backend '%s', which is not registered "
                    "here, so it could not be released automatically. It may still be "
                    "running and billing — release it with that backend's own tooling.",
                    name,
                    handle.runtime,
                )
                backend = None
            if backend is not None:
                try:
                    await backend.down(name, release=True)
                    released = True
                except Exception as e:
                    logger.error(
                        "Failed to release runtime '%s': %s. It may still be billing.",
                        name,
                        e,
                    )

        if handle is not None:
            handle.phase = RuntimePhase.STOPPED if released else handle.phase
            if forget and released:
                self.state.delete(name)
            else:
                self.state.save(handle)
        self._handles.pop(name, None)
        return released

    async def down_all(self) -> list[str]:
        """Stop every runtime this manager knows about."""
        names = sorted({*self._handles, *(h.name for h in self.state.load_all())})
        stopped = []
        for name in names:
            if await self.down(name):
                stopped.append(name)
        return stopped

    async def reap(self, *, now: datetime | None = None) -> list[tuple[str, str]]:
        """Stop runtimes whose spend ceilings have been reached.

        Returns ``(name, reason)`` for each one stopped.
        """
        reaped: list[tuple[str, str]] = []
        for handle in list(self._handles.values()):
            reason = handle.expired_reason(now)
            if reason is None:
                continue
            logger.warning("Reaping runtime '%s': %s", handle.name, reason)
            await self.down(handle.name)
            reaped.append((handle.name, reason))
        return reaped

    async def adopt(
        self, external_id: str, *, name: str, backend: str | None = None, attach: bool = True
    ) -> RuntimeHandle:
        """Take ownership of compute llmcore did not start.

        The recovery path for a leaked runtime: something is billing and nothing
        tracks it, so adopting it is what makes it killable.
        """
        self._require_enabled("adopt a runtime")
        handle = await self._backend_for(backend).adopt(external_id, name=name)
        self._apply_limits(
            handle, idle=self._idle_minutes, hard=self._max_lifetime_minutes, units=None
        )
        self.state.save(handle)
        self._handles[name] = handle
        if attach:
            self.attach(handle)
        return handle

    async def close(self) -> None:
        """Detach provider instances without stopping any runtime.

        Deliberately **not** a teardown. A process exiting is not a reason to
        destroy compute someone is paying for and may still want; the state
        files remain so the next session can find it. Use :meth:`down_all` to
        actually stop spending.
        """
        for name in list(self._attached):
            await self.detach(name)
