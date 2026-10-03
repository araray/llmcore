# src/llmcore/runtimes/deepinfra.py
"""The DeepInfra rental backend for the runtimes subsystem.

DeepInfra sells compute twice over, and the difference decides which API this
backend drives:

* **Serverless inference** -- billed per token, no instance, nothing to reap.
  That half is the existing ``deepinfra`` *provider*, not a runtime.
* **Dedicated LLM deployments** (``/deploy/llm``) -- GPUs reserved for one
  model and billed **per hour of reservation**. That is a runtime.
* **Container rentals** (``/v1/containers``) -- raw GPU nodes with a container
  image and no server on them.

This backend drives the dedicated-deployment API, because that is the one whose
product is "a GPU serving your model at an OpenAI-compatible endpoint", which
is what :class:`~llmcore.runtimes.protocols.ComputeRuntime` describes. Container
rentals would need llmcore to install and expose a server itself, and at the
time of writing DeepInfra offers them on B200 only, with no capacity available.

Two DeepInfra-specific shapes to know about:

* A dedicated deployment's name must be **owned**: ``<display_name>/<model>``,
  rejected otherwise. The display name comes from ``/v1/me``.
* ``/deploy/list`` returns serverless *references* (``type: "legacy"``)
  alongside dedicated deployments. Those cost nothing per hour, so reporting
  them as untracked runtimes would raise a cost alarm about something free.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import Any

from ..exceptions import ConfigError
from .manager import RuntimeError_
from .models import (
    CostUnit,
    ModelSpec,
    Plan,
    RuntimeHandle,
    RuntimePhase,
    RuntimeStatus,
)
from .sizing import GpuSku, Sizer
from .state import RuntimeStateStore

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_DEEPINFRA_API_BASE", "DeepInfraRuntime"]

#: API root. Note this is the bare host: the rental endpoints live at
#: ``/deploy/...`` while the OpenAI-compatible inference routes live under
#: ``/v1/openai``, so neither is a prefix of the other.
DEFAULT_DEEPINFRA_API_BASE = "https://api.deepinfra.com"

#: Where a dedicated deployment is actually called once it is serving.
OPENAI_ROUTE = "/v1/openai"

#: ``gpu_config`` strings look like ``"2xA100-80GB"``. The trailing figure is
#: per-device VRAM, so a 2x row is 160 GB in aggregate.
_SHAPE = re.compile(r"^(?P<count>\d+)x(?P<gpu>.+?)-(?P<vram>\d+(?:\.\d+)?)GB$", re.I)

#: DeepInfra deployment states, mapped onto llmcore's phases.
_PHASES: Mapping[str, RuntimePhase] = {
    "initializing": RuntimePhase.STARTING,
    "pending": RuntimePhase.STARTING,
    "deploying": RuntimePhase.STARTING,
    "updating": RuntimePhase.STARTING,
    "running": RuntimePhase.READY,
    "stopping": RuntimePhase.STOPPING,
    "stopped": RuntimePhase.STOPPED,
    "deleted": RuntimePhase.STOPPED,
    "failed": RuntimePhase.FAILED,
}

#: Deployment types that reserve hardware by the hour. A ``legacy`` deployment
#: is a serverless, per-token model reference: real, but not a rental.
BILLED_TYPES = frozenset({"llm", "lora"})


class DeepInfraRuntime:
    """Rents dedicated LLM deployments from DeepInfra."""

    name = "deepinfra"

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | Any = None,
        state_store: RuntimeStateStore | None = None,
        sizer: Sizer | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        get = getattr(config, "get", None) if config is not None else None
        self._get = get or (lambda _k, d=None: d)
        self._state = state_store or RuntimeStateStore()
        self._base = str(
            base_url
            or self._get("runtimes.deepinfra.base_url", DEFAULT_DEEPINFRA_API_BASE)
            or DEFAULT_DEEPINFRA_API_BASE
        ).rstrip("/")
        self._api_key = api_key
        self._client = client
        self._sizer = sizer
        self._catalogue: dict[str, GpuSku] | None = None
        self._owner: str | None = None

    # ------------------------------------------------------------------
    # Config and transport
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """The credential, resolved lazily so construction contacts nothing."""
        if self._api_key:
            return self._api_key
        var = str(
            self._get("runtimes.deepinfra.api_key_env_var", "DEEPINFRA_API_KEY")
            or "DEEPINFRA_API_KEY"
        )
        return os.environ.get(var) or os.environ.get("DEEPINFRA_TOKEN")

    def _require_key(self, operation: str) -> str:
        key = self.api_key
        if not key:
            raise ConfigError(
                f"Cannot {operation}: no DeepInfra API key. Set DEEPINFRA_API_KEY, or "
                f"runtimes.deepinfra.api_key_env_var to the variable that holds it. "
                f"Unlike gpu.ai, DeepInfra's GPU catalogue is behind the key too, so "
                f"sizing needs one as well."
            )
        return key

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - httpx is a core dep
                raise ConfigError("The DeepInfra runtime backend needs httpx.") from exc
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Any = None,
        allow: tuple[int, ...] = (),
    ) -> Any:
        """Make one API call and return the decoded body.

        Args:
            allow: Status codes to return ``None`` for rather than raise. Used
                by teardown, where "already gone" is the goal, not an error.
        """
        response = await self._http().request(
            method,
            f"{self._base}{path}",
            json=json,
            params=params,
            headers={"Authorization": f"Bearer {self._require_key(f'{method} {path}')}"},
        )
        if response.status_code in allow:
            return None
        if response.status_code >= 400:
            raise RuntimeError_(
                f"DeepInfra {method} {path} failed with {response.status_code}: "
                f"{response.text[:400]}"
            )
        if not response.content:
            return None
        return response.json()

    async def owner(self) -> str:
        """The account's display name, which every deployment name must carry.

        DeepInfra rejects a deployment whose name is not prefixed with it --
        ``"Model name prefix 'Qwen' must match user display name 'araray'"`` --
        so this is a hard requirement rather than a convention.
        """
        if self._owner:
            return self._owner
        configured = self._get("runtimes.deepinfra.owner", None)
        if configured:
            self._owner = str(configured)
            return self._owner
        me = await self._call("GET", "/v1/me") or {}
        owner = str(me.get("team_display_name") or me.get("display_name") or "")
        if not owner:
            raise RuntimeError_(
                "DeepInfra did not report a display name for this account, and a "
                "dedicated deployment's name must be prefixed with it. Set "
                "runtimes.deepinfra.owner to the name shown on the dashboard."
            )
        self._owner = owner
        return owner

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------

    async def catalogue(self, *, refresh: bool = False) -> dict[str, GpuSku]:
        """Build the SKU catalogue from DeepInfra's live GPU availability.

        DeepInfra packs the device count into the SKU name itself
        (``"2xH100-80GB"``), so each row is one rung and ``gpu_count`` is
        parsed back out of it rather than being a separate dimension to
        combine. There are no regions, so no region is recorded -- a plan that
        claimed one would be inventing it.
        """
        if self._catalogue is not None and not refresh:
            return self._catalogue

        payload = await self._call("GET", "/deploy/llm/gpu_availability") or {}
        catalogue: dict[str, GpuSku] = {}
        unavailable: list[str] = []
        for row in payload.get("gpus") or []:
            config = str(row.get("gpu_config") or "")
            price = row.get("usd_per_hour")
            match = _SHAPE.match(config)
            if not config or price is None or match is None:
                logger.debug("DeepInfra: unparsable availability row %r", row)
                continue
            if not row.get("available"):
                unavailable.append(config)
                continue
            count = int(match.group("count"))
            catalogue[config] = GpuSku(
                name=config,
                # Aggregate across the group, as vLLM shards over it.
                vram_gb=float(match.group("vram")) * count,
                cost_per_hour=float(price),
                cost_unit=CostUnit.USD,
                gpu_count=count,
                notes=(
                    f"${float(price):.2f}/hour reserved"
                    + (" (recommended)" if row.get("recommended") else "")
                ),
            )

        if not catalogue:
            raise RuntimeError_(
                "DeepInfra reports no available GPU configurations for dedicated "
                "deployments"
                + (f" (out of capacity: {', '.join(unavailable)})" if unavailable else "")
                + ", so there is nothing to size against."
            )
        self._catalogue = catalogue
        return catalogue

    async def _build_sizer(self) -> Sizer:
        catalogue = await self.catalogue()
        configured = self._get("runtimes.deepinfra.sku_ladder", None)
        ladder = tuple(configured) if configured else tuple(
            sorted(catalogue, key=lambda n: catalogue[n].cost_per_hour or float("inf"))
        )
        return Sizer(
            ladder=ladder,
            skus=catalogue,
            headroom_fraction=float(self._get("runtimes.defaults.headroom_fraction", 0.15)),
            gpu_memory_utilization=float(
                self._get("runtimes.defaults.gpu_memory_utilization", 0.90)
            ),
            hf_token=self._hf_token(),
        )

    def _hf_token(self) -> str | None:
        var = str(self._get("runtimes.deepinfra.hf_token_env_var", "HF_TOKEN") or "HF_TOKEN")
        return os.environ.get(var) or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ------------------------------------------------------------------
    # estimate — free, read-only
    # ------------------------------------------------------------------

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Size *spec* against DeepInfra's live availability. Provisions nothing.

        Needs a key, unlike gpu.ai's public catalogue, but still spends nothing
        and starts nothing.
        """
        sizer = self._sizer or await self._build_sizer()
        plan = await sizer.estimate(spec)
        notes: list[str] = []
        if plan.burn_rate:
            notes.append(
                f"burn rate ~{plan.burn_rate} from the moment the deployment is "
                f"reserved, whether or not anything calls it"
            )
        notes.append(
            "a dedicated deployment is billed for the reservation, not per token: "
            "stopping it is the only thing that stops the cost"
        )
        return plan.with_notes(*notes)

    # ------------------------------------------------------------------
    # up — this spends money
    # ------------------------------------------------------------------

    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle:
        """Reserve a dedicated deployment for ``plan.spec``. **Spends money.**

        Fails closed: once the deployment exists it is reserving hardware, so
        any later failure deletes it before raising.
        """
        if not plan.fits:
            raise RuntimeError_(
                f"Refusing to reserve: the plan says {plan.spec.repo_id} does not fit "
                f"{plan.shape} ({plan.vram_required_gb:.1f} GB needed, "
                f"{plan.vram_available_gb:.1f} GB usable). "
                f"{' '.join(plan.notes[-2:])}"
            )
        self._require_key("reserve a DeepInfra deployment")

        model_name = await self._deployment_name(plan.spec.repo_id)
        gpu, count = self._split_shape(plan)
        body: dict[str, Any] = {
            "model_name": model_name,
            "gpu": gpu,
            "num_gpus": count,
            "hf": {"repo": plan.spec.repo_id},
            # Pinned to one instance: ``max_instances`` above one lets
            # DeepInfra multiply the hourly cost without llmcore asking, and a
            # ceiling cannot bound a replica count it does not know.
            "settings": {"min_instances": 1, "max_instances": 1},
        }
        if plan.spec.revision:
            body["hf"]["revision"] = plan.spec.revision
        token = self._hf_token()
        if token:
            body["hf"]["token"] = token

        created = await self._call("POST", "/deploy/llm", json=body) or {}
        deploy_id = str(created.get("deploy_id") or "")
        if not deploy_id:
            raise RuntimeError_(
                f"DeepInfra accepted the deployment for '{name}' but returned no "
                f"deploy_id, so it cannot be followed or stopped. Check "
                f"https://deepinfra.com/dash/deployments before retrying."
            )

        try:
            deployment = await self._await_running(deploy_id, name=name)
            handle = self._handle_for(deployment, name=name, plan=plan)
        except Exception as exc:
            logger.error(
                "DeepInfra: '%s' failed to come up; deleting %s so it stops billing.",
                name,
                deploy_id,
            )
            await self._delete(deploy_id)
            raise RuntimeError_(
                f"Reserving '{name}' on DeepInfra failed and the deployment was "
                f"deleted rather than left billing. Cause: {exc}"
            ) from exc

        self._state.save(handle)
        return handle

    async def _deployment_name(self, repo_id: str) -> str:
        """Build an owned, accepted deployment name for *repo_id*."""
        owner = await self.owner()
        proposed = f"{owner}/{repo_id.rsplit('/', 1)[-1]}"
        try:
            suggested = await self._call(
                "GET", "/deploy/llm/suggest_name", params={"model_name": proposed}
            )
        except Exception as exc:
            # Only a de-duplicating convenience; the owned name is already
            # valid, so a failure here must not block a launch.
            logger.debug("DeepInfra: suggest_name failed (%s); using %s", exc, proposed)
            return proposed
        return str((suggested or {}).get("model_name") or proposed)

    def _split_shape(self, plan: Plan) -> tuple[str, int]:
        """Recover ``(gpu, num_gpus)`` from a ``gpu_config`` SKU name.

        The API wants the device name without the count (``"A100-80GB"``) and
        the count separately, while the catalogue keys on the packed form.
        """
        match = _SHAPE.match(plan.sku)
        if match is None:
            return plan.sku, max(1, int(plan.gpu_count))
        return f"{match.group('gpu')}-{match.group('vram')}GB", int(match.group("count"))

    async def _await_running(self, deploy_id: str, *, name: str) -> dict[str, Any]:
        """Wait for the deployment to serve, or fail with its own reason."""
        deadline = asyncio.get_running_loop().time() + float(
            self._get("runtimes.deepinfra.ready_timeout_seconds", 1800) or 1800
        )
        while True:
            deployment = await self._deployment(deploy_id)
            status = str(deployment.get("status") or "")
            phase = _PHASES.get(status, RuntimePhase.STARTING)
            if phase is RuntimePhase.READY:
                return deployment
            if phase in {RuntimePhase.FAILED, RuntimePhase.STOPPED}:
                raise RuntimeError_(
                    f"DeepInfra deployment {deploy_id} reached {status!r}: "
                    f"{deployment.get('fail_reason') or 'no reason given'}"
                )
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError_(
                    f"DeepInfra deployment {deploy_id} for '{name}' never became "
                    f"ready (status {status!r})."
                )
            await asyncio.sleep(15.0)

    async def _deployment(self, deploy_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/deploy/{deploy_id}") or {}

    async def _delete(self, deploy_id: str) -> None:
        """Delete a deployment, tolerating one that is already gone."""
        try:
            await self._call("DELETE", f"/deploy/{deploy_id}", allow=(404, 409))
        except Exception as exc:  # pragma: no cover - last-resort logging
            logger.error(
                "DeepInfra: deleting %s failed (%s). It may still be reserving "
                "hardware -- check https://deepinfra.com/dash/deployments.",
                deploy_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Handles and spend
    # ------------------------------------------------------------------

    def _handle_for(
        self, deployment: dict[str, Any], *, name: str, plan: Plan | None = None
    ) -> RuntimeHandle:
        """Build a handle from a deployment payload."""
        config = deployment.get("config") or {}
        gpu = str(config.get("gpu") or "")
        count = int(config.get("num_gpus") or (plan.gpu_count if plan else 1) or 1)
        sku = f"{count}x{gpu}" if gpu else (plan.sku if plan else "")
        rate = None
        if plan is not None:
            rate = plan.estimated_cost_per_hour

        created = _parse_dt(deployment.get("created_at")) or datetime.now(timezone.utc)
        metadata: dict[str, Any] = {
            "gpu": gpu or None,
            "num_gpus": count,
            "deploy_type": deployment.get("type"),
            "deploy_status": deployment.get("status"),
            "price_per_hour": rate,
            # The endpoint is DeepInfra's gateway, so unlike a tunnelled
            # runtime this one needs the account's real credential.
            "api_key": self.api_key,
        }
        return RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id=str(deployment.get("deploy_id") or ""),
            base_url=f"{self._base}{OPENAI_ROUTE}",
            # The deployment's own name, not the HF repo: that is the model id
            # DeepInfra's gateway routes on.
            served_model=str(deployment.get("model_name") or ""),
            api_style="openai",
            recipe="vllm",
            sku=sku,
            phase=_PHASES.get(str(deployment.get("status") or ""), RuntimePhase.DEGRADED),
            started_at=created,
            spend_unit=CostUnit.USD,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    async def _rented_seconds(self) -> dict[str, float]:
        """Seconds of reservation per deploy id, from DeepInfra's own meter.

        Preferred over multiplying a rate by wall-clock elapsed time, because
        a deployment that was stopped and restarted has billed for less than
        it has existed, and the meter knows that while a subtraction does not.
        """
        window_days = float(self._get("runtimes.deepinfra.usage_window_days", 30) or 30)
        since = int(time.time() - window_days * 86400)
        try:
            usage = await self._call("GET", "/payment/usage/rent", params={"from": since})
        except Exception as exc:
            logger.debug("DeepInfra: could not read rental usage: %s", exc)
            return {}
        durations = (usage or {}).get("id_to_duration") or {}
        return {str(k): float(v) for k, v in durations.items()}

    def _accrued(self, handle: RuntimeHandle, seconds: float | None) -> float | None:
        """Dollars this deployment has cost, from metered seconds and its rate."""
        rate = handle.metadata.get("price_per_hour")
        if rate is None or seconds is None:
            return None
        return float(rate) * (float(seconds) / 3600.0)

    # ------------------------------------------------------------------
    # status / logs / down / adopt
    # ------------------------------------------------------------------

    async def status(self, name: str | None = None) -> list[RuntimeStatus]:
        """Report on runtimes, reconciled against DeepInfra's deployment list.

        Only deployments in :data:`BILLED_TYPES` are reconciled. A ``legacy``
        entry is a serverless, per-token model reference -- it appears in the
        same listing and is not a rental, so reporting one as an untracked
        runtime would raise a cost alarm about something that costs nothing per
        hour.
        """
        handles = [self._state.load(name)] if name else self._state.load_all()
        handles = [handle for handle in handles if handle is not None]
        handles = [handle for handle in handles if handle.runtime == self.name]

        live: dict[str, dict[str, Any]] = {}
        listed = False
        try:
            for row in await self.deployments():
                if str(row.get("type") or "") in BILLED_TYPES and row.get("deploy_id"):
                    live[str(row["deploy_id"])] = row
            listed = True
        except Exception as exc:
            logger.debug("DeepInfra: could not list deployments: %s", exc)

        metered = await self._rented_seconds() if handles or live else {}

        out: list[RuntimeStatus] = []
        for handle in handles:
            deployment = live.get(handle.external_id)
            if deployment is None and handle.phase.is_billing and listed:
                handle.phase = RuntimePhase.STOPPED
                handle.error = "no matching DeepInfra deployment; marked stale"
                self._state.save(handle)
            elif deployment is not None:
                handle.phase = _PHASES.get(
                    str(deployment.get("status") or ""), RuntimePhase.DEGRADED
                )
                reason = deployment.get("fail_reason")
                handle.error = str(reason) if reason else handle.error
            handle.spend_used = self._accrued(handle, metered.get(handle.external_id))
            expired = handle.expired_reason()
            if expired and handle.phase.is_billing:
                handle.error = expired
            out.append(RuntimeStatus.from_handle(handle))

        if name is None and listed:
            known = {handle.external_id for handle in handles}
            for deploy_id in sorted(set(live) - known):
                deployment = live[deploy_id]
                label = str(deployment.get("model_name") or deploy_id)
                out.append(
                    RuntimeStatus(
                        name=label,
                        runtime=self.name,
                        phase=RuntimePhase.DEGRADED,
                        sku=str((deployment.get("config") or {}).get("gpu") or ""),
                        spend_unit=CostUnit.USD,
                        error=(
                            f"orphan: DeepInfra deployment {deploy_id} "
                            f"({deployment.get('type')}) is "
                            f"{deployment.get('status')} and llmcore has no record "
                            f"of it. It reserves hardware by the hour. Adopt it to "
                            f"make it killable: llm.runtimes.adopt({deploy_id!r}, "
                            f"name={label!r}, backend='deepinfra')"
                        ),
                    )
                )
        return out

    async def deployments(self) -> list[dict[str, Any]]:
        """Every deployment on the account, of every type."""
        listing = await self._call("GET", "/deploy/list")
        return list(listing or [])

    async def logs(
        self, name: str, *, component: str = "server", tail: int = 100
    ) -> AsyncIterator[str]:
        """Stream the deployment's log lines.

        DeepInfra does expose logs, which gpu.ai does not, so this is a real
        implementation rather than an explanation of why there is none.
        """
        handle = self._state.load(name)
        if handle is None or not handle.external_id:
            yield f"No DeepInfra deployment is recorded under the name {name!r}."
            return
        try:
            payload = await self._call(
                "GET",
                "/v1/deployment_logs/query",
                params={"deploy_id": handle.external_id, "limit": int(tail)},
            )
        except Exception as exc:
            yield f"Could not fetch logs for {name!r}: {exc}"
            return
        rows = payload if isinstance(payload, list) else (payload or {}).get("data") or []
        if not rows:
            yield f"DeepInfra returned no log lines for {name!r}."
            return
        for row in rows:
            if isinstance(row, str):
                yield row
            else:
                yield str(row.get("message") or row.get("log") or row)

    async def down(self, name: str, *, release: bool = True) -> None:
        """Stop *name*. Idempotent, as the protocol requires.

        With *release*, the deployment is **deleted**: DeepInfra reserves the
        hardware for as long as the deployment exists, so scaling it to zero is
        not the same as not paying for it. Without *release*, it is stopped and
        recorded as DETACHED, since stopping without deleting is exactly the
        "llmcore has let go but it may still cost" case that phase exists for.
        """
        handle = self._state.load(name)
        if handle is None:
            logger.debug("DeepInfra: nothing known as '%s'; nothing to stop.", name)
            return
        if not handle.external_id:
            handle.phase = RuntimePhase.STOPPED
            self._state.save(handle)
            return

        handle.phase = RuntimePhase.STOPPING
        self._state.save(handle)
        if release:
            await self._delete(handle.external_id)
            handle.phase = RuntimePhase.STOPPED
            logger.info(
                "Deleted DeepInfra deployment %s ('%s').", handle.external_id, name
            )
        else:
            try:
                await self._call(
                    "POST", f"/deploy/{handle.external_id}/stop", allow=(404, 409)
                )
            except Exception as exc:
                logger.error("DeepInfra: stopping %s failed: %s", handle.external_id, exc)
            handle.phase = RuntimePhase.DETACHED
            handle.error = (
                "stopped without deleting: DeepInfra reserves the hardware while "
                "the deployment exists, so this may still be billing"
            )
        handle.spend_used = self._accrued(
            handle, (await self._rented_seconds()).get(handle.external_id)
        )
        self._state.save(handle)

    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle:
        """Take ownership of a deployment llmcore did not create."""
        deployment = await self._deployment(external_id)
        if not deployment.get("deploy_id"):
            raise RuntimeError_(
                f"DeepInfra has no deployment {external_id!r}, so there is nothing "
                f"to adopt."
            )
        kind = str(deployment.get("type") or "")
        handle = self._handle_for(deployment, name=name)
        if kind not in BILLED_TYPES:
            # Adopted anyway rather than refused: the caller asked for it to be
            # manageable, and refusing would leave them without a way to delete
            # it through llmcore. But it must not be reported as a cost.
            handle.error = (
                f"adopted a {kind!r} deployment: it is serverless and billed per "
                f"token, not per hour, so it has no rental cost to bound"
            )
        rate = self._get("runtimes.deepinfra.assumed_price_per_hour", None)
        if rate and kind in BILLED_TYPES:
            handle.metadata["price_per_hour"] = float(rate)
        elif kind in BILLED_TYPES:
            # No rate means no ceiling: say so rather than let a spend ceiling
            # silently never fire on an adopted rental.
            handle.error = (
                "adopted without a known hourly rate, so a spend ceiling cannot "
                "be enforced on it. Set runtimes.deepinfra.assumed_price_per_hour, "
                "or rely on the lifetime deadline."
            )
        self._state.save(handle)
        logger.info("Adopted DeepInfra deployment %s as '%s'.", external_id, name)
        return handle


def _parse_dt(value: Any) -> datetime | None:
    """Parse an API timestamp, tolerating ``Z`` and absence."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
