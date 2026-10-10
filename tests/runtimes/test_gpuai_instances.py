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
