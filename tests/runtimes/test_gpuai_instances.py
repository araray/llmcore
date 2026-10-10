"""gpu.ai for callers that drive the box themselves.

``GpuAiRuntime.up()`` serves a model on gpu.ai's vllm template. Some callers
want the machine instead -- a bare instance reached over SSH, running their own
code -- and they need to see what kind of machine each offer is before renting
one. These tests cover that surface: raw offers, a bare-instance rent, public
get/terminate, SSH keys and the ownership filter for strays.

Driven through the injectable client, never the network. Row shapes are taken
from gpu.ai's public OpenAPI contract and a public ``/pricing`` read.
"""

from __future__ import annotations

import pytest

from llmcore.runtimes import RuntimeStateStore
from llmcore.runtimes.gpuai import GpuAiRuntime, GpuOffer

from .test_rental_backends import FakeHttp, FakeResponse, config

pytestmark = pytest.mark.asyncio


GPU_TYPES = {"data": [{"gpu_type": "rtx_6000_ada", "vram_gb": 48}], "next_cursor": None}

ENV = {
    "image": "ghcr.io/gpuai-dev/base-cuda124:2026-07",
    "image_digest": "ghcr.io/gpuai-dev/base-cuda124:2026-07@sha256:abc",
    "driver_min": "550.54.14",
    "cuda": "12.4",
}

#: Two real shapes from 2026-10-09: a slow-boot VM and a faster container.
VM_ROW = {
    "gpu_type": "rtx_6000_ada", "gpu_count": 1, "cpu_cores": 12, "ram_gb": 72,
    "storage_gb": 350, "region": "us-central", "tier": "on_demand",
    "price_per_hour": 0.65, "available": 92, "instant_boot": False,
    "community": False, "capacity_class": "secure", "certified_image": True,
    "deployment_type": "vm", "instance_disk_gb": 350, "disk_configurable": False,
    "disk_price_per_gb_hour": 0, "offering_id": "vm-1", "environment": ENV,
}
CONTAINER_ROW = {
    "gpu_type": "rtx_6000_ada", "gpu_count": 1, "cpu_cores": 14, "ram_gb": 78,
    "storage_gb": 0, "region": "us-west", "tier": "on_demand",
    "price_per_hour": 0.74, "available": 4, "instant_boot": True,
    "community": True, "capacity_class": "community", "certified_image": True,
    "deployment_type": "container", "instance_disk_gb": 100, "disk_configurable": True,
    "disk_price_per_gb_hour": 0.00014, "offering_id": "ct-1", "environment": ENV,
}


def runtime_with(tmp_path, routes, **cfg) -> GpuAiRuntime:
    return GpuAiRuntime(
        config=config(**cfg),
        state_store=RuntimeStateStore(tmp_path),
        api_key="test-key",
        client=FakeHttp(routes),
    )


def market(*rows, cursor=None):
    return FakeResponse(200, {"data": list(rows), "next_cursor": cursor})


class TestOffers:
    async def test_every_row_keeps_what_kind_of_machine_it_is(self, tmp_path):
        """The catalogue keeps only the cheapest row per shape, so it would pick
        the slow-boot VM over the instant container without saying so."""
        runtime = runtime_with(tmp_path, {
            ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
            ("GET", "/pricing"): market(CONTAINER_ROW, VM_ROW),
        })
        offers = await runtime.offers()
        assert [o.offering_id for o in offers] == ["vm-1", "ct-1"], "cheapest first"
        vm, container = offers
        assert vm.deployment_type == "vm" and vm.instant_boot is False
        assert container.deployment_type == "container" and container.instant_boot is True
        assert container.capacity_class == "community" and container.community is True
        assert vm.environment.cuda == "12.4"
        assert vm.environment.driver_min == "550.54.14"
        assert vm.vram_gb == 48 and vm.total_vram_gb == 48
        assert vm.shape == "1xrtx_6000_ada"

    async def test_an_unreported_spec_is_unknown_not_zero(self, tmp_path):
        offer = GpuOffer.from_api(CONTAINER_ROW)
        assert offer.storage_gb is None
        assert offer.vram_gb is None

    async def test_only_disk_beyond_the_allowance_is_priced(self, tmp_path):
        """Per the contract, disk up to ``instance_disk_gb`` is included."""
        offer = GpuOffer.from_api(CONTAINER_ROW)
        assert offer.cost_per_hour() == pytest.approx(0.74)
        assert offer.cost_per_hour(100) == pytest.approx(0.74)
        assert offer.cost_per_hour(300) == pytest.approx(0.74 + 200 * 0.00014)

    async def test_filters_are_sent_and_enforced(self, tmp_path):
        runtime = runtime_with(tmp_path, {
            ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
            # A server that ignored the filter must not leak excluded rows.
            ("GET", "/pricing"): market(VM_ROW, dict(CONTAINER_ROW, gpu_type="l40s")),
        })
        offers = await runtime.offers(gpu_type="rtx_6000_ada", region="us-central")
        assert [o.offering_id for o in offers] == ["vm-1"]
        params = runtime._client.sent("GET", "/pricing")[0]["params"]
        assert params["gpu_type"] == "rtx_6000_ada"
        assert params["region"] == "us-central"
        assert params["tier"] == "on_demand"

    async def test_unavailable_rows_only_on_request(self, tmp_path):
        sold_out = dict(CONTAINER_ROW, available=0)
        routes = {
            ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
            ("GET", "/pricing"): market(VM_ROW, sold_out),
        }
        assert len(await runtime_with(tmp_path, routes).offers()) == 1
        everything = await runtime_with(tmp_path, routes).offers(include_unavailable=True)
        assert len(everything) == 2

    async def test_the_market_is_paged_to_the_end(self, tmp_path):
        """The cheapest offering may well be on page two."""
        runtime = runtime_with(tmp_path, {
            ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
            ("GET", "/pricing"): [market(CONTAINER_ROW, cursor="p2"), market(VM_ROW)],
        })
        assert {o.offering_id for o in await runtime.offers()} == {"vm-1", "ct-1"}
        assert runtime._client.sent("GET", "/pricing")[1]["params"]["cursor"] == "p2"

    async def test_the_catalogue_is_paged_to_the_end_too(self, tmp_path):
        runtime = runtime_with(tmp_path, {
            ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
            ("GET", "/pricing"): [market(CONTAINER_ROW, cursor="p2"), market(VM_ROW)],
        })
        rung = (await runtime.catalogue())["1xrtx_6000_ada"]
        assert rung.offering_id == "vm-1"

    async def test_offers_need_no_key(self, tmp_path):
        runtime = GpuAiRuntime(
            config=config(), state_store=RuntimeStateStore(tmp_path), api_key=None,
            client=FakeHttp({
                ("GET", "/gpu-types"): FakeResponse(200, GPU_TYPES),
                ("GET", "/pricing"): market(VM_ROW),
            }),
        )
        await runtime.offers()
        assert all("Authorization" not in c["headers"] for c in runtime._client.calls)

    async def test_to_dict_is_json_safe(self, tmp_path):
        import json

        payload = GpuOffer.from_api(VM_ROW, vram_gb=48).to_dict()
        assert json.loads(json.dumps(payload))["environment"]["cuda"] == "12.4"
        assert "raw" not in payload


# ---------------------------------------------------------------------------
# Bare instances
# ---------------------------------------------------------------------------

from llmcore.runtimes import RuntimePhase  # noqa: E402
from llmcore.runtimes.gpuai import InstancePort, InstanceRequest  # noqa: E402
from llmcore.runtimes.manager import RuntimeError_  # noqa: E402

BARE_RUNNING = {
    "id": "inst-9", "name": "bellavox-ws-r1", "status": "running",
    "gpu_type": "rtx_6000_ada", "gpu_count": 1, "region": "us-central",
    "tier": "on_demand", "price_per_hour": 0.65,
    "created_at": "2026-10-10T00:00:00Z", "ready_at": "2026-10-10T00:04:00Z",
    "auto_terminate_hours": 2, "auto_terminate_at": "2026-10-10T02:04:00Z",
    # No app_url: nothing is served, and nothing should be waited for.
    "connection": {
        "ssh_command": "ssh root@frp.gpu.ai -p 41022",
        "hostname": "frp.gpu.ai", "port": 41022,
        "terminal_url": "https://inst-9-term.apps.gpu.ai",
    },
}


def rent_routes(over: dict | None = None):
    routes = {
        ("POST", "/instances"): FakeResponse(202, {"operation_id": "op-9"}),
        ("GET", "/operations/op-9"): FakeResponse(
            200, {"state": "succeeded", "resource_id": "inst-9"}
        ),
        ("GET", "/instances/inst-9"): FakeResponse(200, BARE_RUNNING),
        ("DELETE", "/instances/inst-9"): FakeResponse(204),
        ("GET", "/instances"): FakeResponse(200, {"data": [], "next_cursor": None}),
    }
    routes.update(over or {})
    return routes


def bare_request(**over) -> InstanceRequest:
    fields = {"environment": "certified", "ssh_key_ids": ("key-1",), "auto_terminate_hours": 2}
    fields.update(over)
    return InstanceRequest.from_offer(GpuOffer.from_api(VM_ROW, vram_gb=48), **fields)


class TestRent:
    async def test_the_body_is_a_bare_launch_pinned_and_capped(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        await runtime.rent(
            bare_request(env={"BELLAVOX_ROOT": "/root/bv"}), name="bellavox-ws-r1"
        )
        body = runtime._client.sent("POST", "/instances")[0]["json"]
        assert "template_id" not in body, "a bare instance serves nothing"
        assert "MODEL" not in body.get("env", {})
        assert body["environment"] == "certified"
        assert body["offering_id"] == "vm-1" and body["region"] == "us-central"
        assert body["max_price_per_hour"] == 0.65
        assert body["viewed_price_per_hour"] == 0.65
        assert body["auto_terminate_hours"] == 2
        assert body["ssh_key_ids"] == ["key-1"]
        assert body["name"] == "bellavox-ws-r1"
        assert body["tier"] == "on_demand"
        assert body["env"] == {"BELLAVOX_ROOT": "/root/bv"}

    async def test_it_waits_for_running_not_for_an_app_url(self, tmp_path):
        """A bare box has no app URL. Waiting for one would time out and then
        terminate a perfectly good instance."""
        runtime = runtime_with(tmp_path, rent_routes())
        handle = await runtime.rent(bare_request(), name="r")
        assert handle.phase is RuntimePhase.READY
        assert handle.base_url == ""
        assert handle.api_style == "ssh" and handle.recipe == "instance"
        # And the plain read: a bare box has no tunnel password to fetch.
        reads = runtime._client.sent("GET", "/instances/inst-9")
        assert all(call["params"] is None for call in reads)

    async def test_the_handle_says_how_to_reach_it(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        handle = await runtime.rent(bare_request(), name="r")
        assert handle.external_id == "inst-9"
        assert handle.metadata["ssh_command"] == "ssh root@frp.gpu.ai -p 41022"
        assert handle.metadata["hostname"] == "frp.gpu.ai"
        assert handle.metadata["port"] == 41022
        assert handle.metadata["auto_terminate_at"] == "2026-10-10T02:04:00Z"
        assert handle.metadata["price_per_hour"] == 0.65
        assert handle.sku == "1xrtx_6000_ada"
        assert handle.secrets == {}

    async def test_an_idempotency_key_given_is_the_one_sent_and_recorded(self, tmp_path):
        """A caller that records the key before renting can replay the create
        after a crash and get the same instance back, not a second one."""
        runtime = runtime_with(tmp_path, rent_routes())
        handle = await runtime.rent(bare_request(), name="r", idempotency_key="rent-42")
        sent = runtime._client.sent("POST", "/instances")[0]["headers"]
        assert sent["Idempotency-Key"] == "rent-42"
        assert handle.metadata["idempotency_key"] == "rent-42"
        assert handle.metadata["operation_id"] == "op-9"

    async def test_env_values_are_not_recorded(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        handle = await runtime.rent(bare_request(env={"TOKEN_X": "s3cr3t"}), name="r")
        text = handle.state_path.read_text()
        assert "s3cr3t" not in text
        assert "TOKEN_X" in text

    async def test_the_lifetime_defaults_to_the_config_rounded_up(self, tmp_path):
        runtime = runtime_with(
            tmp_path, rent_routes(), **{"runtimes.defaults.max_lifetime_minutes": 150}
        )
        await runtime.rent(bare_request(auto_terminate_hours=None), name="r")
        assert runtime._client.sent("POST", "/instances")[0]["json"]["auto_terminate_hours"] == 3

    async def test_a_failure_after_create_terminates_the_instance(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes({
            ("GET", "/instances/inst-9"): FakeResponse(
                200, dict(BARE_RUNNING, status="error", status_reason="stalled")
            ),
        }))
        with pytest.raises(RuntimeError_, match="released rather than left billing"):
            await runtime.rent(bare_request(), name="r")
        assert runtime._client.sent("DELETE", "/instances/inst-9")
        assert runtime._state.load("r").phase is RuntimePhase.STOPPED

    @pytest.mark.parametrize(
        ("over", "problem"),
        [
            ({"image": "pytorch/pytorch:2.4.1"}, "not both"),
            ({"ports": (InstancePort(8775),)}, "ports requires image"),
            ({"environment": None, "image": "x/y:1", "ports": (InstancePort(22),)},
             "port 22 is reserved"),
            ({"environment": None, "image": "x/y:1",
              "ports": (InstancePort(80, "tcp", primary=True),)},
             "primary port must be http"),
            ({"env": {"GPUAI_TOKEN": "x"}}, "reserved by gpu.ai"),
            ({"auto_terminate_hours": 0}, "between 1 and 720"),
            ({"auto_terminate_hours": 721}, "between 1 and 720"),
            ({"gpu_count": 0}, "at least 1"),
        ],
    )
    async def test_what_gpuai_would_refuse_is_refused_before_spending(
        self, tmp_path, over, problem
    ):
        runtime = runtime_with(tmp_path, rent_routes())
        with pytest.raises(RuntimeError_, match=problem):
            await runtime.rent(bare_request(**over), name="r")
        assert runtime._client.calls == []

    async def test_a_name_that_records_a_live_instance_is_not_overwritten(self, tmp_path):
        """Overwriting the record of a billing instance loses the only local
        trace of it."""
        runtime = runtime_with(tmp_path, rent_routes())
        await runtime.rent(bare_request(), name="r")
        with pytest.raises(RuntimeError_, match="already records"):
            await runtime.rent(bare_request(), name="r")
        assert len(runtime._client.sent("POST", "/instances")) == 1

    async def test_no_key_no_create(self, tmp_path):
        from llmcore.exceptions import ConfigError

        runtime = GpuAiRuntime(
            config=config(), state_store=RuntimeStateStore(tmp_path), api_key=None,
            client=FakeHttp(rent_routes()),
        )
        with pytest.raises(ConfigError):
            await runtime.rent(bare_request(), name="r")
        assert runtime._client.calls == []

    async def test_from_offer_rounds_the_ceiling_up_to_the_cent(self):
        """gpu.ai folds extra disk into a cent-rounded price; a ceiling below
        that would refuse the very offer it came from."""
        request = InstanceRequest.from_offer(GpuOffer.from_api(CONTAINER_ROW), disk_gb=300)
        assert request.max_price_per_hour == pytest.approx(0.77)
        assert request.disk_gb == 300
        exact = InstanceRequest.from_offer(GpuOffer.from_api(VM_ROW))
        assert exact.max_price_per_hour == pytest.approx(0.65)


class TestGetAndTerminate:
    async def test_get_reads_without_credentials_by_default(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        instance = await runtime.get("inst-9")
        assert instance["id"] == "inst-9"
        assert runtime._client.sent("GET", "/instances/inst-9")[0]["params"] is None
        await runtime.get("inst-9", credentials=True)
        assert runtime._client.sent("GET", "/instances/inst-9")[1]["params"] == {
            "include": "credentials"
        }

    async def test_get_of_a_missing_instance_raises(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        with pytest.raises(RuntimeError_, match="404"):
            await runtime.get("nope")

    async def test_terminate_is_idempotent_and_marks_the_record(self, tmp_path):
        runtime = runtime_with(tmp_path, rent_routes())
        await runtime.rent(bare_request(), name="r")
        await runtime.terminate("inst-9")
        runtime._client.routes[("DELETE", "/instances/inst-9")] = FakeResponse(404)
        await runtime.terminate("inst-9")
        assert runtime._state.load("r").phase is RuntimePhase.STOPPED

    async def test_a_terminate_that_fails_raises(self, tmp_path):
        """The caller asked for billing to stop. If it did not, they must know."""
        runtime = runtime_with(tmp_path, rent_routes({
            ("DELETE", "/instances/inst-9"): FakeResponse(500, text="boom"),
        }))
        with pytest.raises(RuntimeError_, match="500"):
            await runtime.terminate("inst-9")
