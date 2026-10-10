"""Account-level reads: SSH keys, balances, spending limits and usage.

These are what a control panel shows before anything is rented, so two rules
matter more than the shapes: a **private** key is never uploaded, and a failed
balance read is a value the caller can show, not an exception that takes the
panel down with it. Driven through injectable clients; nothing live.
"""

from __future__ import annotations

import pytest

from llmcore.runtimes import RuntimeStateStore
from llmcore.runtimes.accounts import SshKey, check_public_key
from llmcore.runtimes.deepinfra import DeepInfraRuntime
from llmcore.runtimes.gpuai import GpuAiRuntime
from llmcore.runtimes.manager import RuntimeError_

from .test_rental_backends import FakeHttp, FakeResponse, config

pytestmark = pytest.mark.asyncio

PUBLIC = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBellaVoxTestKeyMaterial bellavox@warlock"
PRIVATE = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n"
    "-----END OPENSSH PRIVATE KEY-----"
)


def gpuai(tmp_path, routes, **kw) -> GpuAiRuntime:
    return GpuAiRuntime(
        config=config(), state_store=RuntimeStateStore(tmp_path),
        api_key=kw.pop("api_key", "test-key"), client=FakeHttp(routes),
    )


def deepinfra(tmp_path, routes, **kw) -> DeepInfraRuntime:
    return DeepInfraRuntime(
        config=config(), state_store=RuntimeStateStore(tmp_path),
        api_key=kw.pop("api_key", "test-key"), client=FakeHttp(routes),
    )


class TestSshKeys:
    async def test_gpuai_lists_every_page(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/ssh-keys"): [
            FakeResponse(200, {"data": [{
                "id": "k1", "name": "laptop", "fingerprint": "SHA256:aa",
                "created_at": "2026-10-01T00:00:00Z",
            }], "next_cursor": "c2"}),
            FakeResponse(200, {"data": [{
                "id": "k2", "name": "bellavox", "fingerprint": "SHA256:bb",
                "created_at": "2026-10-02T00:00:00Z",
            }], "next_cursor": None}),
        ]})
        keys = await runtime.ssh_keys()
        assert [k.id for k in keys] == ["k1", "k2"]
        assert keys[0] == SshKey(
            provider="gpuai", id="k1", name="laptop", fingerprint="SHA256:aa",
            created_at=keys[0].created_at,
        )
        assert keys[0].created_at.year == 2026

    async def test_gpuai_registers_a_public_key(self, tmp_path):
        runtime = gpuai(tmp_path, {("POST", "/ssh-keys"): FakeResponse(201, {
            "id": "k3", "name": "bellavox", "fingerprint": "SHA256:cc",
            "created_at": "2026-10-10T00:00:00Z",
        })})
        key = await runtime.create_ssh_key("bellavox", PUBLIC + "\n")
        assert key.id == "k3"
        call = runtime._client.sent("POST", "/ssh-keys")[0]
        assert call["json"] == {"name": "bellavox", "public_key": PUBLIC}
        assert call["headers"]["Idempotency-Key"]

    async def test_a_private_key_is_never_sent(self, tmp_path):
        runtime = gpuai(tmp_path, {})
        with pytest.raises(ValueError, match="private key"):
            await runtime.create_ssh_key("oops", PRIVATE)
        assert runtime._client.calls == []

    @pytest.mark.parametrize("bad", ["", "not a key", "ssh-ed25519", f"{PUBLIC}\n{PUBLIC}"])
    async def test_anything_but_one_public_key_is_refused(self, bad):
        with pytest.raises(ValueError):
            check_public_key(bad)

    async def test_deepinfra_lists_keys_with_their_content(self, tmp_path):
        runtime = deepinfra(tmp_path, {("GET", "/v1/ssh_keys"): FakeResponse(200, [
            {"id": "d1", "name": "warlock", "key": PUBLIC, "created_at": 1760000000},
        ])})
        [key] = await runtime.ssh_keys()
        assert key.provider == "deepinfra"
        assert key.public_key == PUBLIC
        assert key.created_at is not None

    async def test_a_failed_key_listing_raises(self, tmp_path):
        """Unlike a balance, a key list feeds a launch: guessing it empty
        would launch a box with no way in."""
        runtime = gpuai(tmp_path, {("GET", "/ssh-keys"): FakeResponse(401, text="no")})
        with pytest.raises(RuntimeError_, match="401"):
            await runtime.ssh_keys()
