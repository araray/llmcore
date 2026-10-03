# src/llmcore/runtimes/gpuai.py
"""The gpu.ai rental backend for the runtimes subsystem.

Unlike Colab, gpu.ai is an API with a published catalogue, so this backend
reads its prices rather than carrying a table of them. Three things it does
that Colab cannot, and that the protocol had no way to express until the
``CostUnit`` work:

* **It bills in dollars**, per hour, per whole instance -- so every rate and
  ceiling here is :attr:`~llmcore.runtimes.models.CostUnit.USD`.
* **It prices per region**, so the same GPU has seven prices and a plan has to
  record which one it quoted.
* **It enforces a lifetime server-side** via ``auto_terminate_hours``. That is
  strictly stronger than llmcore's reaper, which cannot fire if this process
  dies, so this backend sets both.

The serving recipe is gpu.ai's own ``vllm`` template rather than a bootstrap
script: it exposes an OpenAI-compatible server on port 8000 behind an HTTPS
tunnel, which is exactly the endpoint the provider layer wants. The tunnel is
guarded by HTTP Basic auth, which is why the handle carries a header rather
than a bearer token.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import uuid
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

__all__ = ["DEFAULT_GPUAI_API_BASE", "GpuAiRuntime"]

#: Public developer API root. ``/pricing``, ``/gpu-types`` and ``/environments``
#: under it need no credentials, which is what lets ``estimate`` stay free.
DEFAULT_GPUAI_API_BASE = "https://api.gpu.ai/v1"

#: The template that serves an OpenAI-compatible endpoint.
DEFAULT_TEMPLATE = "vllm"

#: ``spot`` is soft-deprecated upstream and returns no capacity sitewide, so
#: asking for it would simply never place. Named here rather than defaulted
#: silently so a reader can see the choice was made.
TIER = "on_demand"

#: gpu.ai's instance lifecycle, mapped onto llmcore's phases. ``unreachable``
#: becomes DEGRADED rather than FAILED for the reason the phase exists: an
#: unreachable instance is still allocated and still billing.
_PHASES: Mapping[str, RuntimePhase] = {
    "allocating": RuntimePhase.STARTING,
    "starting": RuntimePhase.STARTING,
    "running": RuntimePhase.READY,
    "stopping": RuntimePhase.STOPPING,
    "stopped": RuntimePhase.STOPPED,
    "terminated": RuntimePhase.STOPPED,
    "unreachable": RuntimePhase.DEGRADED,
    "error": RuntimePhase.FAILED,
}


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


class GpuAiRuntime:
    """Rents GPU instances from gpu.ai and serves a model on them."""

    name = "gpuai"

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
            base_url or self._get("runtimes.gpuai.base_url", DEFAULT_GPUAI_API_BASE)
            or DEFAULT_GPUAI_API_BASE
        ).rstrip("/")
        self._api_key = api_key
        self._client = client
        self._sizer = sizer
        self._catalogue: dict[str, GpuSku] | None = None

    # ------------------------------------------------------------------
    # Config and transport
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """The credential, resolved lazily so construction contacts nothing."""
        if self._api_key:
            return self._api_key
        var = str(
            self._get("runtimes.gpuai.api_key_env_var", "GPUAI_API_KEY") or "GPUAI_API_KEY"
        )
        return os.environ.get(var)

    def _require_key(self, operation: str) -> str:
        key = self.api_key
        if not key:
            raise ConfigError(
                f"Cannot {operation}: no gpu.ai API key. Set GPUAI_API_KEY, or "
                f"runtimes.gpuai.api_key_env_var to the variable that holds it. "
                f"Sizing needs no key -- only provisioning does."
            )
        return key

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - httpx is a core dep
                raise ConfigError("The gpu.ai runtime backend needs httpx.") from exc
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _call(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = True,
        json: Any = None,
        params: Any = None,
        headers: dict[str, str] | None = None,
        allow: tuple[int, ...] = (),
    ) -> Any:
        """Make one API call and return the decoded body.

        Args:
            authenticated: Whether to send the key. The catalogue endpoints do
                not need one, and sizing must work without it.
            allow: Status codes to return ``None`` for instead of raising.
                Used by ``down``, where a 404 means the thing is already gone,
                which is the outcome being asked for.
        """
        request_headers = dict(headers or {})
        if authenticated:
            request_headers["Authorization"] = f"Bearer {self._require_key(f'{method} {path}')}"
        response = await self._http().request(
            method,
            f"{self._base}{path}",
            json=json,
            params=params,
            headers=request_headers,
        )
        if response.status_code in allow:
            return None
        if response.status_code >= 400:
            raise RuntimeError_(
                f"gpu.ai {method} {path} failed with {response.status_code}: "
                f"{response.text[:400]}"
            )
        if not response.content:
            return None
        return response.json()

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------

    async def catalogue(self, *, refresh: bool = False) -> dict[str, GpuSku]:
        """Build the SKU catalogue from gpu.ai's live pricing.

        One rung per ``(gpu_type, gpu_count)`` shape, keeping the **cheapest
        available offering** for it and recording that offering's region and
        id. Collapsing the regions this way is what makes the ladder usable:
        seven prices for the same GPU would otherwise be seven rungs the sizer
        walks and seven notes it writes, to reach the same answer.

        The rate is the GPU price **plus** the instance disk, because both are
        billed. Quoting only ``price_per_hour`` would understate the bill by
        the disk's share, and understating a cost in a subsystem that exists to
        bound cost is the specific failure this one keeps having.
        """
        if self._catalogue is not None and not refresh:
            return self._catalogue

        vram: dict[str, float] = {}
        try:
            types = await self._call("GET", "/gpu-types", authenticated=False)
            for row in (types or {}).get("data", []):
                if row.get("gpu_type") and row.get("vram_gb"):
                    vram[str(row["gpu_type"])] = float(row["vram_gb"])
        except Exception as exc:
            logger.debug("gpu.ai: could not read /gpu-types: %s", exc)

        pricing = await self._call("GET", "/pricing", authenticated=False)
        best: dict[str, GpuSku] = {}
        for row in (pricing or {}).get("data", []):
            gpu_type = str(row.get("gpu_type") or "")
            count = int(row.get("gpu_count") or 0)
            price = row.get("price_per_hour")
            if not gpu_type or count <= 0 or price is None:
                continue
            # ``available`` is a *count of units*, not a flag. Treating it as a
            # boolean happens to work, but reporting it as one would throw away
            # the only capacity signal the catalogue carries.
            if int(row.get("available") or 0) <= 0:
                continue
            if row.get("tier") and str(row["tier"]) != TIER:
                continue

            disk_gb = float(row.get("instance_disk_gb") or 0)
            disk_rate = float(row.get("disk_price_per_gb_hour") or 0)
            total = float(price) + disk_gb * disk_rate

            per_gpu = vram.get(gpu_type)
            if per_gpu is None:
                # No VRAM figure means this rung cannot be sized, and guessing
                # one would produce a plan that claims to fit.
                logger.debug("gpu.ai: no VRAM known for %s; skipping", gpu_type)
                continue

            key = f"{count}x{gpu_type}"
            existing = best.get(key)
            if existing is not None and (existing.cost_per_hour or 0) <= total:
                continue
            notes = [f"{row.get('region')} at ${float(price):.2f}/hour"]
            if disk_gb and disk_rate:
                notes.append(f"plus {disk_gb:.0f} GB disk at ${disk_gb * disk_rate:.3f}/hour")
            if row.get("capacity_class"):
                notes.append(str(row["capacity_class"]))
            best[key] = GpuSku(
                name=key,
                # Aggregate across the tensor-parallel group: vLLM shards both
                # weights and KV cache over it.
                vram_gb=per_gpu * count,
                cost_per_hour=total,
                cost_unit=CostUnit.USD,
                gpu_count=count,
                region=str(row.get("region") or "") or None,
                offering_id=str(row.get("offering_id") or "") or None,
                needs_fp16=gpu_type.startswith(("t4", "v100")),
                notes="; ".join(notes),
            )

        if not best:
            raise RuntimeError_(
                "gpu.ai returned no priced, available on-demand offerings with a "
                "known VRAM figure, so there is nothing to size against."
            )
        self._catalogue = best
        return best

    async def _build_sizer(self) -> Sizer:
        catalogue = await self.catalogue()
        configured = self._get("runtimes.gpuai.sku_ladder", None)
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
        var = str(self._get("runtimes.gpuai.hf_token_env_var", "HF_TOKEN") or "HF_TOKEN")
        return os.environ.get(var) or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ------------------------------------------------------------------
    # estimate — free, read-only
    # ------------------------------------------------------------------

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Size *spec* against gpu.ai's live catalogue. Provisions nothing.

        Needs no API key: ``/pricing`` and ``/gpu-types`` are public, which
        keeps the free half of the protocol genuinely free.
        """
        sizer = self._sizer or await self._build_sizer()
        plan = await sizer.estimate(spec)
        rung = (await self.catalogue()).get(plan.sku)
        # The sizer already wrote the rung's own notes; repeating them here
        # printed every price line twice.
        notes: list[str] = []
        if plan.offering_id:
            notes.append(
                f"pinned to offering {plan.offering_id}, so the launch cannot be "
                f"placed on a pricier row than this one"
            )
        if plan.burn_rate:
            notes.append(
                f"burn rate ~{plan.burn_rate} from allocation until terminated, "
                f"whether or not anything calls it"
            )
        template = self._template()
        if rung is not None and rung.vram_gb < 24 and template == DEFAULT_TEMPLATE:
            notes.append(
                "the vllm template requires at least 24 GB of VRAM, so this rung "
                "cannot run it"
            )
        return plan.with_notes(*notes) if notes else plan

    def _template(self) -> str:
        return str(self._get("runtimes.gpuai.template", DEFAULT_TEMPLATE) or DEFAULT_TEMPLATE)

    # ------------------------------------------------------------------
    # up — this spends money
    # ------------------------------------------------------------------

    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle:
        """Rent an instance and serve ``plan.spec`` on it. **Spends money.**

        Fails closed: every step after the instance exists is inside one
        handler that terminates it before raising. A half-started rental is the
        expensive kind of failure, because nothing stops billing on its own
        until ``auto_terminate_hours`` expires.

        The lifetime limit is sent to the platform as well as stamped on the
        handle. llmcore's reaper cannot fire if this process dies; gpu.ai's
        timer can, so the belt is worth having alongside the braces.
        """
        if not plan.fits:
            raise RuntimeError_(
                f"Refusing to rent: the plan says {plan.spec.repo_id} does not fit "
                f"{plan.shape} ({plan.vram_required_gb:.1f} GB needed, "
                f"{plan.vram_available_gb:.1f} GB usable). "
                f"{' '.join(plan.notes[-2:])}"
            )
        self._require_key("rent a gpu.ai instance")

        gpu_type, count = self._split_shape(plan)
        body: dict[str, Any] = {
            "gpu_type": gpu_type,
            "gpu_count": count,
            "tier": TIER,
            "name": name,
            "template_id": self._template(),
            "env": {"MODEL": plan.spec.repo_id},
        }
        if plan.region:
            body["region"] = plan.region
        if plan.offering_id:
            # A catalogue row is a quote, not a booking. Without the pin the
            # platform may place this on a pricier row than the one approved.
            body["offering_id"] = plan.offering_id
        if plan.estimated_cost_per_hour:
            # The plan's rate already includes the disk component, so it is at
            # or above the catalogue price this ceiling is compared against.
            # Sending the bare GPU price instead would make a rounding
            # difference look like a price rise and fail a valid launch.
            body["max_price_per_hour"] = round(float(plan.estimated_cost_per_hour), 4)
            body["viewed_price_per_hour"] = round(float(plan.estimated_cost_per_hour), 4)
        hours = self._auto_terminate_hours()
        if hours:
            body["auto_terminate_hours"] = hours
        token = self._hf_token()
        if token:
            body["env"]["HF_TOKEN"] = token
        keys = self._get("runtimes.gpuai.ssh_key_ids", None)
        if keys:
            body["ssh_key_ids"] = list(keys)

        accepted = await self._call(
            "POST",
            "/instances",
            json=body,
            # Retried creates must not rent a second GPU.
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        operation_id = str((accepted or {}).get("operation_id") or "")
        for warning in (accepted or {}).get("warnings") or []:
            logger.warning("gpu.ai create advisory for '%s': %s", name, warning)
        if not operation_id:
            raise RuntimeError_(
                f"gpu.ai accepted the create for '{name}' but returned no "
                f"operation_id, so the instance cannot be followed or stopped. "
                f"Check https://gpu.ai for a stray instance before retrying."
            )

        instance_id = ""
        try:
            instance_id = await self._await_operation(operation_id, name=name)
            instance = await self._instance(instance_id, credentials=True)
            instance = await self._await_running(instance_id, instance, name=name)
            handle = self._handle_for(instance, name=name, plan=plan)
        except Exception as exc:
            if instance_id:
                logger.error(
                    "gpu.ai: '%s' failed to come up; terminating %s so it stops billing.",
                    name,
                    instance_id,
                )
                await self._terminate(instance_id)
            raise RuntimeError_(
                f"Renting '{name}' from gpu.ai failed and the instance was "
                f"released rather than left billing. Cause: {exc}"
            ) from exc

        self._state.save(handle)
        return handle

    def _split_shape(self, plan: Plan) -> tuple[str, int]:
        """Recover ``(gpu_type, gpu_count)`` from a catalogue SKU name."""
        sku = plan.sku
        prefix = f"{plan.gpu_count}x"
        gpu_type = sku[len(prefix):] if sku.lower().startswith(prefix.lower()) else sku
        return gpu_type, max(1, int(plan.gpu_count))

    def _auto_terminate_hours(self) -> int | None:
        """The platform-side lifetime limit, in whole hours.

        Rounded **up**: rounding down would have the platform kill a runtime
        before llmcore's own hard deadline, making the shorter limit the real
        one while the handle advertised the longer.
        """
        minutes = float(self._get("runtimes.defaults.max_lifetime_minutes", 240) or 0)
        if minutes <= 0:
            return None
        return max(1, math.ceil(minutes / 60.0))

    async def _await_operation(self, operation_id: str, *, name: str) -> str:
        """Poll an async operation to a terminal state; return the resource id."""
        deadline = asyncio.get_running_loop().time() + float(
            self._get("runtimes.gpuai.create_timeout_seconds", 900) or 900
        )
        while True:
            operation = await self._call("GET", f"/operations/{operation_id}") or {}
            state = str(operation.get("state") or "")
            resource = str(operation.get("resource_id") or "")
            if state == "succeeded":
                if not resource:
                    raise RuntimeError_(
                        f"gpu.ai operation {operation_id} succeeded without naming an "
                        f"instance, so '{name}' cannot be tracked or stopped."
                    )
                return resource
            if state in {"failed", "cancelled"}:
                detail = (operation.get("error") or {}).get("message") or state
                # Returned, not raised bare: a failed create may still have
                # left an instance behind, and the caller terminates it.
                if resource:
                    await self._terminate(resource)
                raise RuntimeError_(f"gpu.ai could not create '{name}': {detail}")
            if asyncio.get_running_loop().time() > deadline:
                if resource:
                    await self._terminate(resource)
                raise RuntimeError_(
                    f"gpu.ai create for '{name}' did not finish in time (operation "
                    f"{operation_id} still {state!r}); any instance it made was "
                    f"terminated."
                )
            await asyncio.sleep(5.0)

    async def _await_running(
        self, instance_id: str, instance: dict[str, Any], *, name: str
    ) -> dict[str, Any]:
        """Wait for the served endpoint to exist, not merely the instance.

        An instance reports ``running`` before the template's server has
        finished loading weights, so this waits for the tunnel URL as well --
        attaching a provider to a URL that is not there yet would fail the
        attach and tear down a healthy rental.
        """
        deadline = asyncio.get_running_loop().time() + float(
            self._get("runtimes.gpuai.ready_timeout_seconds", 1200) or 1200
        )
        while True:
            status = str(instance.get("status") or "")
            url = ((instance.get("connection") or {}).get("app_url")) or ""
            if status == "running" and url:
                return instance
            if _PHASES.get(status) in {RuntimePhase.FAILED, RuntimePhase.STOPPED}:
                raise RuntimeError_(
                    f"gpu.ai instance for '{name}' reached {status!r}: "
                    f"{instance.get('status_reason') or 'no reason given'}"
                )
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError_(
                    f"gpu.ai instance {instance_id} for '{name}' never served an "
                    f"endpoint (status {status!r})."
                )
            await asyncio.sleep(10.0)
            instance = await self._instance(instance_id, credentials=True)

    async def _instance(self, instance_id: str, *, credentials: bool = False) -> dict[str, Any]:
        params = {"include": "credentials"} if credentials else None
        return await self._call("GET", f"/instances/{instance_id}", params=params) or {}

    async def _terminate(self, instance_id: str) -> None:
        """Terminate an instance, tolerating one that is already gone."""
        try:
            await self._call("DELETE", f"/instances/{instance_id}", allow=(404, 409))
        except Exception as exc:  # pragma: no cover - last-resort logging
            logger.error(
                "gpu.ai: terminating %s failed (%s). It may still be billing -- "
                "check https://gpu.ai.",
                instance_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------

    def _handle_for(
        self, instance: dict[str, Any], *, name: str, plan: Plan | None = None
    ) -> RuntimeHandle:
        """Build a handle from an instance payload."""
        connection = instance.get("connection") or {}
        app_url = str(connection.get("app_url") or "").rstrip("/")
        metadata: dict[str, Any] = {
            "gpu_type": instance.get("gpu_type"),
            "gpu_count": instance.get("gpu_count"),
            "region": instance.get("region"),
            "price_per_hour": instance.get("price_per_hour"),
            "instance_status": instance.get("status"),
            "ssh_command": connection.get("ssh_command"),
            "terminal_url": connection.get("terminal_url"),
            "auto_terminate_at": instance.get("auto_terminate_at"),
        }
        user = connection.get("app_user")
        password = connection.get("app_password")
        if user and password:
            # The tunnel authenticates with HTTP Basic, not a bearer token, so
            # the credential has to travel as a header. ``api_key`` is still set
            # because vLLM wants *some* bearer and the provider layer requires
            # one; the Basic header is what actually gets past the tunnel.
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            metadata["default_headers"] = {"Authorization": f"Basic {token}"}
        elif app_url:
            logger.warning(
                "gpu.ai instance %s served %s but returned no basic-auth "
                "credentials, so requests to it will probably be rejected. "
                "Fetch them with GET /v1/instances/%s?include=credentials.",
                instance.get("id"),
                app_url,
                instance.get("id"),
            )

        started = _parse_dt(instance.get("created_at")) or datetime.now(timezone.utc)
        return RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id=str(instance.get("id") or ""),
            # The template serves OpenAI-compatible routes under /v1.
            base_url=f"{app_url}/v1" if app_url else "",
            served_model=(plan.spec.repo_id if plan else str(instance.get("name") or name)),
            api_style="openai",
            recipe=self._template(),
            sku=str(
                f"{instance.get('gpu_count') or 1}x{instance.get('gpu_type') or ''}"
            ).strip("x"),
            phase=_PHASES.get(str(instance.get("status") or ""), RuntimePhase.DEGRADED),
            started_at=started,
            spend_unit=CostUnit.USD,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _accrued(self, handle: RuntimeHandle, now: datetime | None = None) -> float | None:
        """Dollars this runtime has cost so far, derived from its rate.

        gpu.ai does not publish a per-instance running total, so this is
        ``rate x elapsed``. It is measured from ``created_at`` rather than
        ``ready_at`` because allocation time is not free, and because a
        ceiling that over-estimates fires early -- which is the safe direction
        to be wrong in when the alternative is an unbounded bill.
        """
        rate = handle.metadata.get("price_per_hour")
        if rate is None:
            return None
        moment = now or datetime.now(timezone.utc)
        hours = max(0.0, (moment - handle.started_at).total_seconds()) / 3600.0
        return float(rate) * hours

    # ------------------------------------------------------------------
    # status / logs / down / adopt
    # ------------------------------------------------------------------

    async def status(self, name: str | None = None) -> list[RuntimeStatus]:
        """Report on runtimes, reconciled against what gpu.ai says exists.

        The reconciliation matters more here than on Colab. A forgotten Colab
        session expires by itself; a forgotten rental bills by the hour until
        someone terminates it, so an instance gpu.ai has that llmcore does not
        know about is reported as an orphan with the command to adopt it.
        """
        handles = [self._state.load(name)] if name else self._state.load_all()
        handles = [handle for handle in handles if handle is not None]
        handles = [handle for handle in handles if handle.runtime == self.name]

        live: dict[str, dict[str, Any]] = {}
        try:
            live = {
                str(row.get("id")): row for row in await self.instances() if row.get("id")
            }
        except Exception as exc:
            logger.debug("gpu.ai: could not list instances: %s", exc)

        out: list[RuntimeStatus] = []
        for handle in handles:
            instance = live.get(handle.external_id)
            if instance is None and handle.phase.is_billing:
                # Only trusted when the listing actually succeeded: treating a
                # failed list call as "everything is gone" would mark live,
                # billing rentals as stopped and stop watching them.
                if live:
                    handle.phase = RuntimePhase.STOPPED
                    handle.error = "no matching gpu.ai instance; marked stale"
                    self._state.save(handle)
            elif instance is not None:
                handle.phase = _PHASES.get(
                    str(instance.get("status") or ""), RuntimePhase.DEGRADED
                )
                if instance.get("price_per_hour") is not None:
                    handle.metadata["price_per_hour"] = instance["price_per_hour"]
                reason = instance.get("status_reason")
                handle.error = str(reason) if reason else handle.error
            handle.spend_used = self._accrued(handle)
            expired = handle.expired_reason()
            if expired and handle.phase.is_billing:
                handle.error = expired
            out.append(RuntimeStatus.from_handle(handle))

        if name is None and live:
            known = {handle.external_id for handle in handles}
            for instance_id in sorted(set(live) - known):
                instance = live[instance_id]
                label = str(instance.get("name") or instance_id)
                rate = instance.get("price_per_hour")
                out.append(
                    RuntimeStatus(
                        name=label,
                        runtime=self.name,
                        phase=RuntimePhase.DEGRADED,
                        sku=f"{instance.get('gpu_count') or 1}x{instance.get('gpu_type') or ''}",
                        spend_unit=CostUnit.USD,
                        error=(
                            f"orphan: gpu.ai instance {instance_id} is "
                            f"{instance.get('status')}"
                            + (f" at ${float(rate):.2f}/hour" if rate else "")
                            + f" and llmcore has no record of it. Adopt it to make it "
                            f"killable: llm.runtimes.adopt({instance_id!r}, "
                            f"name={label!r}, backend='gpuai')"
                        ),
                    )
                )
        return out

    async def instances(self) -> list[dict[str, Any]]:
        """Every instance on the account, following the cursor to the end.

        Paging all the way through is deliberate: a leaked rental on page two
        bills exactly as much as one on page one.
        """
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(50):
            params = {"cursor": cursor} if cursor else None
            page = await self._call("GET", "/instances", params=params) or {}
            out.extend(page.get("data") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        return out

    async def logs(
        self, name: str, *, component: str = "server", tail: int = 100
    ) -> AsyncIterator[str]:
        """Explain where this backend's logs live, since the API serves none.

        gpu.ai's public API exposes no log endpoint for an instance. Yielding
        nothing would read as "the server is silent", which is a different and
        more alarming claim than "this backend cannot fetch logs", so it says
        which it is and how to get them.
        """
        handle = self._state.load(name)
        yield (
            "gpu.ai's public API has no instance log endpoint, so llmcore cannot "
            "stream logs for this runtime."
        )
        if handle is not None:
            terminal = handle.metadata.get("terminal_url")
            ssh = handle.metadata.get("ssh_command")
            if terminal:
                yield f"  web console: {terminal}"
            if ssh:
                yield f"  shell:       {ssh}"
            if not terminal and not ssh:
                yield "  no console or SSH coordinates were recorded for it."

    async def down(self, name: str, *, release: bool = True) -> None:
        """Terminate *name*. Idempotent, as the protocol requires.

        ``release`` is accepted for protocol conformance but cannot be honoured
        as "keep it but stop billing": gpu.ai bills an allocated instance
        whether or not it is serving, and there is no paused state that is
        free. Asking to keep it is therefore recorded as DETACHED -- still
        billing, llmcore no longer watching -- rather than quietly terminated.
        """
        handle = self._state.load(name)
        if handle is None:
            logger.debug("gpu.ai: nothing known as '%s'; nothing to stop.", name)
            return
        if not release:
            handle.phase = RuntimePhase.DETACHED
            handle.error = (
                "detached without releasing: the instance is still allocated and "
                "still billing"
            )
            self._state.save(handle)
            logger.warning(
                "Runtime '%s' was detached, not released. gpu.ai instance %s is "
                "still billing.",
                name,
                handle.external_id,
            )
            return

        handle.phase = RuntimePhase.STOPPING
        self._state.save(handle)
        if handle.external_id:
            await self._terminate(handle.external_id)
        handle.phase = RuntimePhase.STOPPED
        handle.spend_used = self._accrued(handle)
        self._state.save(handle)
        logger.info("Terminated gpu.ai instance %s ('%s').", handle.external_id, name)

    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle:
        """Take ownership of an instance llmcore did not start.

        The point of this path is that something is billing and nothing tracks
        it, so it is written to state even when it is not servable -- a handle
        that exists is a handle that ``down`` can kill.
        """
        instance = await self._instance(external_id, credentials=True)
        if not instance.get("id"):
            raise RuntimeError_(
                f"gpu.ai has no instance {external_id!r}, so there is nothing to adopt."
            )
        handle = self._handle_for(instance, name=name)
        handle.served_model = str(
            (instance.get("env") or {}).get("MODEL") or handle.served_model
        )
        if not handle.base_url:
            handle.error = (
                "adopted without a served endpoint: it is billing and can be "
                "terminated, but nothing can be routed to it"
            )
        handle.spend_used = self._accrued(handle)
        self._state.save(handle)
        logger.info("Adopted gpu.ai instance %s as '%s'.", external_id, name)
        return handle
