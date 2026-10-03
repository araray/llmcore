# tests/runtimes/test_rental_backends.py
"""The gpu.ai and DeepInfra rental backends.

These two exist because the runtime protocol was written for Colab, which
bills in compute units, sells one GPU shape at a time, has no regions and
expires its own sessions. A rental API does none of that: it bills dollars by
the hour, forever, until something terminates it. So the tests here lean on the
places where that difference bites -- fail-closed teardown, orphan detection,
and never quoting a cost in the wrong unit or the wrong direction.

Driven through each backend's injectable client rather than the network: the
shapes asserted here were taken from the vendors' own OpenAPI documents and
verified against live read-only calls, and a test that needs credentials to run
is a test that does not run.
"""

from __future__ import annotations

import json as jsonlib
from datetime import datetime, timedelta, timezone

import pytest

from llmcore.runtimes import CostUnit, ModelSpec, RuntimePhase, RuntimeStateStore
from llmcore.runtimes.deepinfra import DeepInfraRuntime
from llmcore.runtimes.gpuai import GpuAiRuntime
from llmcore.runtimes.manager import RuntimeError_
from llmcore.runtimes.protocols import ComputeRuntime
from llmcore.runtimes.sizing import GpuSku, Sizer

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# A fake transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (jsonlib.dumps(payload) if payload is not None else "")

    @property
    def content(self) -> bytes:
        return self.text.encode()

    def json(self):
        return self._payload


class FakeHttp:
    """Matches ``(METHOD, path)`` against a route table.

    A route may be a single response or a list, which is consumed one call at a
    time -- that is how a polling sequence (``pending`` then ``succeeded``) is
    expressed without sleeping.
    """

    def __init__(self, routes: dict[tuple[str, str], object]):
        self.routes = {(m.upper(), p): r for (m, p), r in routes.items()}
        self.calls: list[dict] = []

    async def request(self, method, url, json=None, params=None, headers=None):
        path = url.split("/v1", 1)[-1] if "api.gpu.ai" in url else url.split(".com", 1)[-1]
        self.calls.append(
            {"method": method.upper(), "path": path, "json": json,
             "params": params, "headers": headers or {}}
        )
        route = self.routes.get((method.upper(), path))
        if route is None:
            return FakeResponse(404, text=f"no fake route for {method} {path}")
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route

    def sent(self, method: str, path: str) -> list[dict]:
        return [
            c for c in self.calls if c["method"] == method.upper() and c["path"] == path
        ]


def config(**overrides):
    """A ``config.get``-shaped accessor."""

    class View:
        def get(self, key, default=None):
            return overrides.get(key, default)

    return View()


# ---------------------------------------------------------------------------
# gpu.ai fixtures
# ---------------------------------------------------------------------------

GPU_TYPES = {
    "data": [
        {"gpu_type": "a100_80gb", "vram_gb": 80},
        {"gpu_type": "a40", "vram_gb": 48},
        # Deliberately priced below but with no VRAM figure.
        {"gpu_type": "mystery_gpu"},
    ],
    "next_cursor": None,
}


def pricing_row(**over):
    row = {
        "gpu_type": "a100_80gb", "gpu_count": 1, "region": "us-east",
        "tier": "on_demand", "price_per_hour": 1.00, "available": 4,
        "instance_disk_gb": 100, "disk_price_per_gb_hour": 0.0002,
        "offering_id": "offer-default",
    }
    row.update(over)
    return row


def gpuai_runtime(tmp_path, routes, **cfg):
    return GpuAiRuntime(
        config=config(**cfg),
        state_store=RuntimeStateStore(tmp_path),
        api_key="test-key",
        client=FakeHttp(routes),
    )


CATALOGUE_ROUTES = {
    ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
    ("GET", "/pricing"): FakeResponse(200, {"data": [pricing_row()]}),
}


# ---------------------------------------------------------------------------


class TestGpuAiCatalogue:
    async def test_the_cheapest_region_wins_and_is_recorded(self, tmp_path):
        """gpu.ai prices the same GPU in seven regions. Keeping all of them as
        rungs would make the sizer walk seven to reach one answer, so the
        cheapest is kept -- and which one it was has to survive, or the plan
        cannot promise the price it quoted."""
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(200, {"data": [
            pricing_row(region="us-east", price_per_hour=2.0, offering_id="dear"),
            pricing_row(region="eu-west", price_per_hour=1.0, offering_id="cheap"),
            pricing_row(region="us-west", price_per_hour=3.0, offering_id="dearer"),
        ]})
        catalogue = await gpuai_runtime(tmp_path, routes).catalogue()
        rung = catalogue["1xa100_80gb"]
        assert rung.region == "eu-west"
        assert rung.offering_id == "cheap"

    async def test_the_rate_includes_the_disk(self, tmp_path):
        """Both the GPU and the instance disk are billed. Quoting only the GPU
        would understate the bill in a subsystem whose job is bounding it."""
        catalogue = await gpuai_runtime(tmp_path, dict(CATALOGUE_ROUTES)).catalogue()
        rung = catalogue["1xa100_80gb"]
        assert rung.cost_per_hour == pytest.approx(1.00 + 100 * 0.0002)
        assert rung.cost_unit is CostUnit.USD
        assert "disk" in rung.notes

    async def test_rows_with_no_capacity_are_excluded(self, tmp_path):
        """``available`` is a count of units, not a flag, and zero means the
        row cannot be placed."""
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(200, {"data": [
            pricing_row(gpu_type="a40", available=0, price_per_hour=0.1),
            pricing_row(available=4),
        ]})
        catalogue = await gpuai_runtime(tmp_path, routes).catalogue()
        assert "1xa40" not in catalogue
        assert "1xa100_80gb" in catalogue

    async def test_spot_rows_are_excluded(self, tmp_path):
        """Spot capacity is unavailable sitewide upstream, so a spot rung would
        be a ladder rung that never places."""
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(200, {"data": [
            pricing_row(gpu_type="a40", tier="spot", price_per_hour=0.1),
            pricing_row(tier="on_demand"),
        ]})
        catalogue = await gpuai_runtime(tmp_path, routes).catalogue()
        assert list(catalogue) == ["1xa100_80gb"]

    async def test_a_gpu_with_no_vram_figure_is_skipped_not_guessed(self, tmp_path):
        """A guessed VRAM figure produces a plan that claims to fit."""
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(200, {"data": [
            pricing_row(gpu_type="mystery_gpu", price_per_hour=0.01),
            pricing_row(),
        ]})
        catalogue = await gpuai_runtime(tmp_path, routes).catalogue()
        assert "1xmystery_gpu" not in catalogue

    async def test_multi_gpu_vram_aggregates(self, tmp_path):
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(
            200, {"data": [pricing_row(gpu_count=4, price_per_hour=4.0)]}
        )
        catalogue = await gpuai_runtime(tmp_path, routes).catalogue()
        assert catalogue["4xa100_80gb"].vram_gb == 320.0
        assert catalogue["4xa100_80gb"].gpu_count == 4

    async def test_an_empty_catalogue_refuses_rather_than_sizes(self, tmp_path):
        routes = dict(CATALOGUE_ROUTES)
        routes[("GET", "/pricing")] = FakeResponse(200, {"data": []})
        with pytest.raises(RuntimeError_, match="nothing to size against"):
            await gpuai_runtime(tmp_path, routes).catalogue()

    async def test_the_catalogue_needs_no_api_key(self, tmp_path):
        """``estimate`` must stay free, and a key is not free to require: the
        whole point of the estimate/up split is deciding before spending."""
        runtime = GpuAiRuntime(
            config=config(), state_store=RuntimeStateStore(tmp_path),
            api_key=None, client=FakeHttp(dict(CATALOGUE_ROUTES)),
        )
        await runtime.catalogue()
        for call in runtime._client.calls:
            assert "Authorization" not in call["headers"]


INSTANCE_RUNNING = {
    "id": "inst-1", "status": "running", "gpu_type": "a100_80gb", "gpu_count": 1,
    "region": "eu-west", "tier": "on_demand", "price_per_hour": 1.0,
    "created_at": "2026-10-03T00:00:00Z",
    "connection": {
        "app_url": "https://gpu-abcd.apps.gpu.ai",
        "app_user": "u", "app_password": "p",
        "terminal_url": "https://gpu-abcd-term.apps.gpu.ai",
        "ssh_command": "ssh root@1.2.3.4",
    },
}


def up_routes(over: dict | None = None):
    routes = {
        **CATALOGUE_ROUTES,
        ("POST", "/instances"): FakeResponse(202, {"operation_id": "op-1"}),
        ("GET", "/operations/op-1"): FakeResponse(
            200, {"state": "succeeded", "resource_id": "inst-1"}
        ),
        ("GET", "/instances/inst-1"): FakeResponse(200, INSTANCE_RUNNING),
        ("DELETE", "/instances/inst-1"): FakeResponse(204),
        ("GET", "/instances"): FakeResponse(200, {"data": [], "next_cursor": None}),
    }
    routes.update(over or {})
    return routes


async def plan_for(runtime, repo="Qwen/Qwen2.5-7B-Instruct", ctx=4096):
    sizer = Sizer(
        ladder=tuple(await runtime.catalogue()),
        skus=await runtime.catalogue(),
        offline=True,
    )
    runtime._sizer = sizer
    return await runtime.estimate(ModelSpec(repo_id=repo, context_length=ctx))


class TestGpuAiUp:
    async def test_a_plan_that_does_not_fit_spends_nothing(self, tmp_path):
        runtime = gpuai_runtime(tmp_path, up_routes())
        plan = await plan_for(runtime)
        with pytest.raises(RuntimeError_, match="Refusing to rent"):
            await runtime.up(_not_fitting(plan), name="r")
        assert runtime._client.sent("POST", "/instances") == []

    async def test_it_pins_the_offering_and_caps_the_price(self, tmp_path):
        """A catalogue row is a quote, not a booking: without the pin the
        platform may place the launch on a pricier row than the approved one."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        plan = await plan_for(runtime)
        await runtime.up(plan, name="r")
        body = runtime._client.sent("POST", "/instances")[0]["json"]
        assert body["offering_id"] == plan.offering_id
        assert body["region"] == plan.region
        assert body["tier"] == "on_demand"
        assert body["max_price_per_hour"] == pytest.approx(plan.estimated_cost_per_hour)

    async def test_it_asks_the_platform_to_kill_the_instance_too(self, tmp_path):
        """llmcore's reaper cannot fire if this process dies. gpu.ai's timer
        can, so the platform limit is the one that survives a crash."""
        runtime = gpuai_runtime(
            tmp_path, up_routes(), **{"runtimes.defaults.max_lifetime_minutes": 90}
        )
        await runtime.up(await plan_for(runtime), name="r")
        body = runtime._client.sent("POST", "/instances")[0]["json"]
        # 90 minutes rounds *up* to 2 hours: rounding down would make the
        # platform kill it before llmcore's own deadline.
        assert body["auto_terminate_hours"] == 2

    async def test_the_create_is_idempotency_keyed(self, tmp_path):
        """A retried create must not rent a second GPU."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        await runtime.up(await plan_for(runtime), name="r")
        assert runtime._client.sent("POST", "/instances")[0]["headers"]["Idempotency-Key"]

    async def test_the_handle_carries_basic_auth_not_a_bearer(self, tmp_path):
        """The tunnel authenticates with HTTP Basic. Attaching a bearer token
        would produce a reachable endpoint that rejects every request."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        handle = await runtime.up(await plan_for(runtime), name="r")
        assert handle.base_url == "https://gpu-abcd.apps.gpu.ai/v1"
        assert handle.metadata["default_headers"]["Authorization"].startswith("Basic ")
        assert handle.spend_unit is CostUnit.USD

    async def test_a_failure_after_create_terminates_the_instance(self, tmp_path):
        """Fail closed. A half-started rental bills until something kills it."""
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("GET", "/instances/inst-1"): FakeResponse(
                200, {"id": "inst-1", "status": "error",
                      "status_reason": "provisioning stalled"}
            ),
        }))
        with pytest.raises(RuntimeError_, match="released rather than left billing"):
            await runtime.up(await plan_for(runtime), name="r")
        assert runtime._client.sent("DELETE", "/instances/inst-1")

    async def test_an_accepted_create_with_no_operation_id_is_loud(self, tmp_path):
        """Nothing can be followed or stopped, so the message has to say to go
        and look for a stray rather than imply nothing happened."""
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("POST", "/instances"): FakeResponse(202, {}),
        }))
        with pytest.raises(RuntimeError_, match="stray instance"):
            await runtime.up(await plan_for(runtime), name="r")

    async def test_a_failed_operation_terminates_what_it_made(self, tmp_path):
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("GET", "/operations/op-1"): FakeResponse(200, {
                "state": "failed", "resource_id": "inst-1",
                "error": {"message": "no capacity"},
            }),
        }))
        with pytest.raises(RuntimeError_, match="no capacity"):
            await runtime.up(await plan_for(runtime), name="r")
        assert runtime._client.sent("DELETE", "/instances/inst-1")


def _not_fitting(plan):
    from dataclasses import replace

    return replace(plan, fits=False)


class TestGpuAiTeardownAndStatus:
    async def test_down_is_idempotent_against_an_already_gone_instance(self, tmp_path):
        """Tearing down something already gone is how recovery from a
        half-failed start works, so a 404 is the outcome, not an error."""
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("DELETE", "/instances/inst-1"): FakeResponse(404, text="gone"),
        }))
        await runtime.up(await plan_for(runtime), name="r")
        await runtime.down("r")
        await runtime.down("r")
        assert runtime._state.load("r").phase is RuntimePhase.STOPPED

    async def test_down_on_an_unknown_name_does_nothing(self, tmp_path):
        runtime = gpuai_runtime(tmp_path, up_routes())
        await runtime.down("never-existed")
        assert runtime._client.calls == []

    async def test_declining_to_release_does_not_terminate_but_says_so(self, tmp_path):
        """gpu.ai has no free paused state, so "keep it" means "keep paying".
        Silently terminating would destroy compute someone asked to keep;
        silently keeping would hide an ongoing bill."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        await runtime.up(await plan_for(runtime), name="r")
        await runtime.down("r", release=False)
        handle = runtime._state.load("r")
        assert handle.phase is RuntimePhase.DETACHED
        assert "still billing" in handle.error
        assert runtime._client.sent("DELETE", "/instances/inst-1") == []

    async def test_an_untracked_instance_is_an_orphan_with_an_adopt_command(self, tmp_path):
        """A forgotten Colab session expires. A forgotten rental does not."""
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("GET", "/instances"): FakeResponse(200, {
                "data": [dict(INSTANCE_RUNNING, id="stray", name="someone-elses")],
                "next_cursor": None,
            }),
        }))
        statuses = await runtime.status()
        assert len(statuses) == 1
        assert "orphan" in statuses[0].error
        assert "$1.00/hour" in statuses[0].error
        assert "adopt('stray'" in statuses[0].error.replace('"', "'")

    async def test_a_failed_listing_does_not_mark_live_runtimes_stale(self, tmp_path):
        """Reading a failed list call as "everything is gone" would stop
        llmcore watching rentals that are still billing."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        await runtime.up(await plan_for(runtime), name="r")
        runtime._client.routes[("GET", "/instances")] = FakeResponse(500, text="boom")
        statuses = await runtime.status()
        assert statuses[0].phase is RuntimePhase.READY
        assert runtime._state.load("r").phase is RuntimePhase.READY

    async def test_status_reports_accrued_dollars(self, tmp_path):
        runtime = gpuai_runtime(tmp_path, up_routes())
        handle = await runtime.up(await plan_for(runtime), name="r")
        handle.started_at = datetime.now(timezone.utc) - timedelta(hours=3)
        runtime._state.save(handle)
        status = (await runtime.status("r"))[0]
        # $1.00/hour for three hours, rendered in dollars rather than units.
        assert status.spend_used == pytest.approx(3.0, abs=0.01)
        assert status.spend_so_far.startswith("$3.0")

    async def test_the_spend_ceiling_fires_in_dollars(self, tmp_path):
        """The only guard against a runtime busy in a loop. It has to compare
        dollars with dollars."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        handle = await runtime.up(await plan_for(runtime), name="r")
        handle.started_at = datetime.now(timezone.utc) - timedelta(hours=10)
        handle.max_spend = 5.0
        runtime._state.save(handle)
        status = (await runtime.status("r"))[0]
        assert "spend ceiling reached" in status.error
        assert "$5.00" in status.error

    async def test_logs_say_there_are_none_rather_than_appear_silent(self, tmp_path):
        """Yielding nothing reads as "the server is quiet", which is a
        different and more alarming claim than "this API has no log endpoint"."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        await runtime.up(await plan_for(runtime), name="r")
        lines = [line async for line in runtime.logs("r")]
        assert any("no instance log endpoint" in line for line in lines)
        assert any("gpu-abcd-term.apps.gpu.ai" in line for line in lines)

    async def test_adopt_records_an_instance_with_no_endpoint_anyway(self, tmp_path):
        """The point of adopt is that something is billing and nothing tracks
        it. A handle that exists is a handle ``down`` can kill."""
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("GET", "/instances/inst-1"): FakeResponse(200, {
                "id": "inst-1", "status": "unreachable", "gpu_type": "a100_80gb",
                "gpu_count": 1, "price_per_hour": 1.0,
                "created_at": "2026-10-03T00:00:00Z", "connection": None,
            }),
        }))
        handle = await runtime.adopt("inst-1", name="rescued")
        assert handle.phase is RuntimePhase.DEGRADED
        assert "nothing can be routed to it" in handle.error
        assert runtime._state.load("rescued") is not None

    async def test_adopting_something_that_does_not_exist_is_refused(self, tmp_path):
        runtime = gpuai_runtime(tmp_path, up_routes({
            ("GET", "/instances/nope"): FakeResponse(200, {}),
        }))
        with pytest.raises(RuntimeError_, match="nothing to adopt"):
            await runtime.adopt("nope", name="x")

    async def test_instances_follows_the_cursor(self, tmp_path):
        """A leaked rental on page two bills exactly as much as one on page one."""
        runtime = gpuai_runtime(tmp_path, up_routes())
        runtime._client.routes[("GET", "/instances")] = [
            FakeResponse(200, {"data": [{"id": "a"}], "next_cursor": "c2"}),
            FakeResponse(200, {"data": [{"id": "b"}], "next_cursor": None}),
        ]
        assert [row["id"] for row in await runtime.instances()] == ["a", "b"]


# ---------------------------------------------------------------------------
# DeepInfra
# ---------------------------------------------------------------------------

AVAILABILITY = {"gpus": [
    {"gpu_config": "1xA100-80GB", "usd_per_hour": 0.89, "available": True},
    {"gpu_config": "2xA100-80GB", "usd_per_hour": 1.78, "available": True},
    {"gpu_config": "1xH100-80GB", "usd_per_hour": 2.20, "available": True,
     "recommended": True},
    {"gpu_config": "8xB200-180GB", "usd_per_hour": 29.52, "available": False},
]}

DEPLOY_RUNNING = {
    "type": "llm", "deploy_id": "dep-1", "model_name": "araray/Qwen2.5-7B-Instruct",
    "status": "running", "created_at": "2026-10-03T00:00:00+00:00",
    "config": {"gpu": "A100-80GB", "num_gpus": 1},
}

#: What this account really returns: three serverless references and no
#: rentals. They cost nothing per hour.
LEGACY_DEPLOYS = [
    {"type": "legacy", "deploy_id": "DSw", "model_name": "meta-llama/Llama-3.1-8B",
     "status": "running", "task": "text-generation"},
    {"type": "legacy", "deploy_id": "DHw", "model_name": "deepseek-ai/DeepSeek-V3",
     "status": "running", "task": "text-generation"},
]


def di_routes(over: dict | None = None):
    routes = {
        ("GET", "/v1/me"): FakeResponse(200, {"display_name": "araray"}),
        ("GET", "/deploy/llm/gpu_availability"): FakeResponse(200, AVAILABILITY),
        ("GET", "/deploy/llm/suggest_name"): FakeResponse(
            200, {"model_name": "araray/Qwen2.5-7B-Instruct"}
        ),
        ("POST", "/deploy/llm"): FakeResponse(200, {"deploy_id": "dep-1"}),
        ("GET", "/deploy/dep-1"): FakeResponse(200, DEPLOY_RUNNING),
        ("DELETE", "/deploy/dep-1"): FakeResponse(200, {}),
        ("POST", "/deploy/dep-1/stop"): FakeResponse(200, {}),
        ("GET", "/deploy/list"): FakeResponse(200, []),
        ("GET", "/payment/usage/rent"): FakeResponse(200, {"id_to_duration": {}}),
    }
    routes.update(over or {})
    return routes


def deepinfra(tmp_path, routes, **cfg):
    return DeepInfraRuntime(
        config=config(**cfg),
        state_store=RuntimeStateStore(tmp_path),
        api_key="test-key",
        client=FakeHttp(routes),
    )


class TestDeepInfraCatalogue:
    async def test_the_packed_shape_is_parsed_and_vram_aggregated(self, tmp_path):
        """DeepInfra packs the count into the name. The trailing figure is
        *per device*, so a 2x row is 160 GB across the group."""
        catalogue = await deepinfra(tmp_path, di_routes()).catalogue()
        assert catalogue["2xA100-80GB"].vram_gb == 160.0
        assert catalogue["2xA100-80GB"].gpu_count == 2
        assert catalogue["1xA100-80GB"].cost_unit is CostUnit.USD

    async def test_unavailable_configurations_are_excluded(self, tmp_path):
        catalogue = await deepinfra(tmp_path, di_routes()).catalogue()
        assert "8xB200-180GB" not in catalogue

    async def test_no_capacity_at_all_names_what_is_out_of_stock(self, tmp_path):
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/llm/gpu_availability"): FakeResponse(200, {"gpus": [
                {"gpu_config": "8xB200-180GB", "usd_per_hour": 29.52,
                 "available": False},
            ]}),
        }))
        with pytest.raises(RuntimeError_, match="out of capacity: 8xB200-180GB"):
            await runtime.catalogue()

    async def test_the_api_shape_is_split_back_out(self, tmp_path):
        """The create endpoint wants the device name and count separately."""
        runtime = deepinfra(tmp_path, di_routes())
        plan = await plan_for(runtime)
        assert runtime._split_shape(plan) == ("A100-80GB", plan.gpu_count)

    async def test_no_region_is_claimed(self, tmp_path):
        """DeepInfra has no regions, and a plan that named one would be
        inventing it."""
        plan = await plan_for(deepinfra(tmp_path, di_routes()))
        assert plan.region is None


class TestDeepInfraUp:
    async def test_the_deployment_name_is_owned(self, tmp_path):
        """DeepInfra refuses a name not prefixed with the account's display
        name, so this is a requirement rather than a convention."""
        runtime = deepinfra(tmp_path, di_routes())
        await runtime.up(await plan_for(runtime), name="r")
        body = runtime._client.sent("POST", "/deploy/llm")[0]["json"]
        assert body["model_name"].startswith("araray/")
        assert body["hf"]["repo"] == "Qwen/Qwen2.5-7B-Instruct"

    async def test_replicas_are_pinned_to_one(self, tmp_path):
        """A ceiling cannot bound a replica count it does not know about, and
        more replicas multiply the hourly cost."""
        runtime = deepinfra(tmp_path, di_routes())
        await runtime.up(await plan_for(runtime), name="r")
        settings = runtime._client.sent("POST", "/deploy/llm")[0]["json"]["settings"]
        assert settings == {"min_instances": 1, "max_instances": 1}

    async def test_the_endpoint_is_the_openai_gateway_with_the_deploy_name(self, tmp_path):
        """A dedicated deployment is reached through DeepInfra's own gateway,
        so it needs the account credential rather than a throwaway bearer."""
        runtime = deepinfra(tmp_path, di_routes())
        handle = await runtime.up(await plan_for(runtime), name="r")
        assert handle.base_url.endswith("/v1/openai")
        assert handle.served_model == "araray/Qwen2.5-7B-Instruct"
        assert handle.metadata["api_key"] == "test-key"
        assert handle.spend_unit is CostUnit.USD

    async def test_a_failed_deployment_is_deleted_not_left_reserved(self, tmp_path):
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/dep-1"): FakeResponse(200, dict(
                DEPLOY_RUNNING, status="failed", fail_reason="weights not found"
            )),
        }))
        with pytest.raises(RuntimeError_, match="deleted rather than left billing"):
            await runtime.up(await plan_for(runtime), name="r")
        assert runtime._client.sent("DELETE", "/deploy/dep-1")

    async def test_a_name_suggestion_failure_does_not_block_a_launch(self, tmp_path):
        """It only de-duplicates; the owned name is already valid."""
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/llm/suggest_name"): FakeResponse(500, text="boom"),
        }))
        handle = await runtime.up(await plan_for(runtime), name="r")
        assert handle.external_id == "dep-1"


class TestDeepInfraStatus:
    async def test_serverless_deploys_are_not_reported_as_leaked_rentals(self, tmp_path):
        """``/deploy/list`` mixes serverless references in with rentals. They
        bill per token, not per hour, so calling them orphans would raise a
        cost alarm about something with no hourly cost at all."""
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/list"): FakeResponse(200, LEGACY_DEPLOYS),
        }))
        assert await runtime.status() == []

    async def test_a_dedicated_deploy_we_do_not_know_is_an_orphan(self, tmp_path):
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/list"): FakeResponse(
                200, [*LEGACY_DEPLOYS, dict(DEPLOY_RUNNING, deploy_id="stray")]
            ),
        }))
        statuses = await runtime.status()
        assert len(statuses) == 1
        assert "orphan" in statuses[0].error
        assert "reserves hardware by the hour" in statuses[0].error

    async def test_spend_comes_from_the_meter_not_the_wall_clock(self, tmp_path):
        """A deployment stopped and restarted has billed for less than it has
        existed. Subtracting timestamps cannot know that; the meter does."""
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/list"): FakeResponse(200, [DEPLOY_RUNNING]),
            ("GET", "/payment/usage/rent"): FakeResponse(
                200, {"id_to_duration": {"dep-1": 1800.0}}
            ),
        }))
        handle = await runtime.up(await plan_for(runtime), name="r")
        handle.started_at = datetime.now(timezone.utc) - timedelta(hours=100)
        runtime._state.save(handle)
        status = (await runtime.status("r"))[0]
        # Half an hour metered at $0.89/hour, not a hundred hours of elapsed time.
        assert status.spend_used == pytest.approx(0.445, abs=0.001)


class TestDeepInfraTeardown:
    async def test_releasing_deletes_because_reserving_is_what_bills(self, tmp_path):
        """Scaling a deployment to zero is not the same as not paying for it."""
        runtime = deepinfra(tmp_path, di_routes())
        await runtime.up(await plan_for(runtime), name="r")
        await runtime.down("r")
        assert runtime._client.sent("DELETE", "/deploy/dep-1")
        assert runtime._state.load("r").phase is RuntimePhase.STOPPED

    async def test_not_releasing_stops_but_warns_it_may_still_bill(self, tmp_path):
        runtime = deepinfra(tmp_path, di_routes())
        await runtime.up(await plan_for(runtime), name="r")
        await runtime.down("r", release=False)
        handle = runtime._state.load("r")
        assert handle.phase is RuntimePhase.DETACHED
        assert "may still be billing" in handle.error
        assert runtime._client.sent("POST", "/deploy/dep-1/stop")
        assert runtime._client.sent("DELETE", "/deploy/dep-1") == []

    async def test_down_on_an_unknown_name_does_nothing(self, tmp_path):
        runtime = deepinfra(tmp_path, di_routes())
        await runtime.down("never-existed")
        assert runtime._client.calls == []

    async def test_adopting_a_serverless_deploy_says_it_has_no_rental_cost(self, tmp_path):
        """Adopted rather than refused -- the caller wants to be able to delete
        it -- but it must not be reported as an hourly cost."""
        runtime = deepinfra(tmp_path, di_routes({
            ("GET", "/deploy/dep-1"): FakeResponse(200, dict(DEPLOY_RUNNING, type="legacy")),
        }))
        handle = await runtime.adopt("dep-1", name="rescued")
        assert "billed per token, not per hour" in handle.error

    async def test_adopting_a_rental_with_no_known_rate_says_the_ceiling_cannot_fire(
        self, tmp_path
    ):
        """Otherwise a spend ceiling would be set and silently never fire."""
        runtime = deepinfra(tmp_path, di_routes())
        handle = await runtime.adopt("dep-1", name="rescued")
        assert "spend ceiling cannot" in handle.error

    async def test_an_assumed_rate_makes_an_adopted_rental_boundable(self, tmp_path):
        runtime = deepinfra(
            tmp_path, di_routes(),
            **{"runtimes.deepinfra.assumed_price_per_hour": 0.89},
        )
        handle = await runtime.adopt("dep-1", name="rescued")
        assert handle.metadata["price_per_hour"] == 0.89


class TestBothSatisfyTheProtocol:
    @pytest.mark.filterwarnings("ignore")
    async def test_structural_conformance(self):
        """Discovered structurally, like the media adapters, so a missing verb
        is a registration failure rather than a crash at teardown time."""
        assert isinstance(GpuAiRuntime(), ComputeRuntime)
        assert isinstance(DeepInfraRuntime(), ComputeRuntime)
