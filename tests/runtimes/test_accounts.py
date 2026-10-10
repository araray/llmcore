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


# ---------------------------------------------------------------------------
# Money reads
# ---------------------------------------------------------------------------

from llmcore.runtimes import AccountBalance, openrouter_balance  # noqa: E402


class TestGpuAiBalance:
    async def test_cents_become_dollars_and_runway_follows(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/billing/balance"): FakeResponse(200, {
            "balance_cents": 2500, "balance_dollars": 25.0, "burn_cents_per_hour": 65,
            "auto_pay_enabled": False, "auto_pay_threshold_cents": 0,
            "auto_pay_amount_cents": 0,
        })})
        balance = await runtime.balance()
        assert balance.ok
        assert balance.balance_usd == pytest.approx(25.0)
        assert balance.burn_usd_per_hour == pytest.approx(0.65)
        assert balance.runway_hours == pytest.approx(25.0 / 0.65)
        assert balance.auto_pay is False

    async def test_an_unknown_burn_rate_stays_unknown(self, tmp_path):
        """``null`` means gpu.ai could not tell. Reading it as zero is how a
        panel says nothing is running while something is."""
        runtime = gpuai(tmp_path, {("GET", "/billing/balance"): FakeResponse(200, {
            "balance_cents": 2500, "balance_dollars": 25.0, "burn_cents_per_hour": None,
            "auto_pay_enabled": False, "auto_pay_threshold_cents": 0,
            "auto_pay_amount_cents": 0,
        })})
        balance = await runtime.balance()
        assert balance.burn_usd_per_hour is None
        assert balance.runway_hours is None
        assert any("unknown, not zero" in note for note in balance.notes)

    async def test_no_runway_at_zero_burn_or_no_credit(self):
        assert AccountBalance("gpuai", balance_usd=10, burn_usd_per_hour=0).runway_hours is None
        assert AccountBalance("gpuai", balance_usd=-1, burn_usd_per_hour=1).runway_hours is None

    async def test_a_failed_read_is_a_value_not_an_exception(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/billing/balance"): FakeResponse(503, text="down")})
        balance = await runtime.balance()
        assert not balance.ok
        assert "503" in balance.error
        assert balance.balance_usd is None

    async def test_no_key_is_an_error_value_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GPUAI_API_KEY", raising=False)
        runtime = gpuai(tmp_path, {}, api_key=None)
        balance = await runtime.balance()
        assert "no gpu.ai API key" in balance.error
        assert runtime._client.calls == []

    async def test_the_key_never_appears_in_an_error(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/billing/balance"): FakeResponse(401, text="bad")})
        balance = await runtime.balance()
        assert "test-key" not in balance.error
        assert "test-key" not in str(balance.to_dict())


class TestGpuAiSpendingLimit:
    async def test_the_cap_and_its_use(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/billing/spending-limit"): FakeResponse(200, {
            "monthly_limit_cents": 5000, "monthly_limit_dollars": 50.0,
            "current_month_spend_cents": 1234, "current_month_spend_dollars": 12.34,
            "current_day_spend_cents": 65, "daily_limit_cents": 1000,
            "percent_used": 24.68, "enforcement": "stop_running",
            "auto_terminate_hours": 4, "billing_cycle_start": "2026-10-01T00:00:00Z",
        })})
        limit = await runtime.spending_limit()
        assert limit.ok and limit.configured is True
        assert limit.monthly_limit_usd == pytest.approx(50.0)
        assert limit.month_spend_usd == pytest.approx(12.34)
        assert limit.daily_limit_usd == pytest.approx(10.0)
        assert limit.enforcement == "stop_running"
        assert limit.billing_cycle_start.day == 1

    async def test_a_404_means_no_cap_is_set(self, tmp_path):
        runtime = gpuai(tmp_path, {
            ("GET", "/billing/spending-limit"): FakeResponse(404, text="not found"),
        })
        limit = await runtime.spending_limit()
        assert limit.ok and limit.configured is False

    async def test_a_failed_read_is_a_value(self, tmp_path):
        runtime = gpuai(tmp_path, {
            ("GET", "/billing/spending-limit"): FakeResponse(500, text="boom"),
        })
        limit = await runtime.spending_limit()
        assert not limit.ok and limit.configured is None


class TestGpuAiUsage:
    async def test_per_instance_cost_from_the_meter(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/usage"): [
            FakeResponse(200, {"data": [
                {"bucket_start": "2026-10-09T00:00:00Z", "cost_cents": 130,
                 "gpu_seconds": 7200, "instance_id": "inst-1"},
            ], "next_cursor": "c2"}),
            FakeResponse(200, {"data": [
                {"bucket_start": "2026-10-10T00:00:00Z", "cost_cents": 65,
                 "gpu_seconds": 3600, "instance_id": "inst-1"},
                {"bucket_start": "2026-10-10T00:00:00Z", "cost_cents": 74,
                 "gpu_seconds": 3600, "instance_id": "inst-2"},
            ], "next_cursor": None}),
        ]})
        report = await runtime.usage(start="2026-10-09T00:00:00Z")
        assert report.ok
        assert report.total_cost_usd == pytest.approx(2.69)
        assert report.by_instance() == {
            "inst-1": pytest.approx(1.95), "inst-2": pytest.approx(0.74)
        }
        first = runtime._client.sent("GET", "/usage")[0]["params"]
        assert first == {
            "bucket": "day", "group_by": "instance_id", "start": "2026-10-09T00:00:00Z",
        }
        assert runtime._client.sent("GET", "/usage")[1]["params"]["cursor"] == "c2"

    async def test_a_datetime_window_is_sent_as_rfc3339(self, tmp_path):
        from datetime import datetime, timezone

        runtime = gpuai(tmp_path, {("GET", "/usage"): FakeResponse(
            200, {"data": [], "next_cursor": None}
        )})
        report = await runtime.usage(start=datetime(2026, 10, 1, tzinfo=timezone.utc))
        assert runtime._client.sent("GET", "/usage")[0]["params"]["start"] == (
            "2026-10-01T00:00:00Z"
        )
        assert report.total_cost_usd == 0.0

    async def test_a_failed_read_has_no_total_rather_than_zero(self, tmp_path):
        runtime = gpuai(tmp_path, {("GET", "/usage"): FakeResponse(500, text="x")})
        report = await runtime.usage()
        assert not report.ok
        assert report.total_cost_usd is None


CHECKLIST = {
    "email": True, "billing_address": True, "payment_method": True,
    "suspended": False, "overdue_invoices": 0, "stripe_balance": -12.5,
    "recent": 3.2, "limit": 20.0, "suspend_reason": None, "topup": False,
    "billing_address_info": {"line1": "private"}, "payment_method_info": {"last4": "4242"},
    "scoped_credits": [
        {"name": "promo", "granted_cents": 500, "remaining_cents": 300,
         "granted_ts": 1, "expired": False},
        {"name": "old", "granted_cents": 500, "remaining_cents": 500,
         "granted_ts": 1, "expired": True},
    ],
}


class TestDeepInfraBalance:
    async def test_the_inverted_sign_is_turned_the_right_way_up(self, tmp_path):
        """DeepInfra reports funds ready to spend as a *negative* number."""
        runtime = deepinfra(tmp_path, {
            ("GET", "/payment/checklist"): FakeResponse(200, CHECKLIST),
        })
        balance = await runtime.balance()
        assert balance.ok
        assert balance.balance_usd == pytest.approx(12.5)
        assert balance.spent_usd == pytest.approx(3.2)
        assert balance.limit_usd == pytest.approx(20.0)
        assert balance.scoped_credit_usd == pytest.approx(3.0), "expired credit excluded"
        assert balance.suspended is False
        assert balance.burn_usd_per_hour is None, "DeepInfra reports none"

    async def test_billing_details_are_not_carried(self, tmp_path):
        runtime = deepinfra(tmp_path, {
            ("GET", "/payment/checklist"): FakeResponse(200, CHECKLIST),
        })
        flat = str((await runtime.balance()).to_dict())
        assert "4242" not in flat and "private" not in flat

    async def test_a_suspension_is_said_out_loud(self, tmp_path):
        runtime = deepinfra(tmp_path, {("GET", "/payment/checklist"): FakeResponse(
            200, dict(CHECKLIST, suspended=True, suspend_reason="overdue")
        )})
        balance = await runtime.balance()
        assert balance.suspended is True
        assert any("suspended: overdue" in note for note in balance.notes)

    async def test_a_failed_read_is_a_value(self, tmp_path):
        runtime = deepinfra(tmp_path, {
            ("GET", "/payment/checklist"): FakeResponse(401, text="no"),
        })
        balance = await runtime.balance()
        assert not balance.ok and "401" in balance.error

    async def test_spending_limit(self, tmp_path):
        runtime = deepinfra(tmp_path, {
            ("GET", "/payment/config"): FakeResponse(200, {"limit": 50.0}),
        })
        limit = await runtime.spending_limit()
        assert limit.configured is True and limit.limit_usd == 50.0

    async def test_a_negative_limit_means_none(self, tmp_path):
        runtime = deepinfra(tmp_path, {
            ("GET", "/payment/config"): FakeResponse(200, {"limit": -1}),
        })
        limit = await runtime.spending_limit()
        assert limit.configured is False and limit.limit_usd is None


OPENROUTER = "https://openrouter.ai/api/v1/credits"


class TestOpenRouterBalance:
    async def test_credit_left_is_bought_minus_used(self):
        client = FakeHttp({("GET", OPENROUTER): FakeResponse(200, {
            "data": {"total_credits": 20.0, "total_usage": 7.25},
        })})
        balance = await openrouter_balance(api_key="or-key", client=client)
        assert balance.ok
        assert balance.balance_usd == pytest.approx(12.75)
        assert balance.total_credits_usd == 20.0 and balance.spent_usd == 7.25
        assert client.calls[0]["headers"]["Authorization"] == "Bearer or-key"

    async def test_the_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        client = FakeHttp({("GET", OPENROUTER): FakeResponse(200, {
            "data": {"total_credits": 1, "total_usage": 0},
        })})
        await openrouter_balance(client=client)
        assert client.calls[0]["headers"]["Authorization"] == "Bearer env-key"

    async def test_no_key_is_an_error_value(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        balance = await openrouter_balance(client=FakeHttp({}))
        assert not balance.ok and "OPENROUTER_API_KEY" in balance.error

    async def test_an_http_failure_is_an_error_value(self):
        client = FakeHttp({("GET", OPENROUTER): FakeResponse(403, text="forbidden")})
        balance = await openrouter_balance(api_key="or-key", client=client)
        assert not balance.ok and "403" in balance.error
        assert "or-key" not in balance.error

    async def test_a_transport_failure_is_an_error_value(self):
        class Broken:
            async def request(self, *a, **k):
                raise ConnectionError("network unreachable")

        balance = await openrouter_balance(api_key="or-key", client=Broken())
        assert balance.error == "network unreachable"
