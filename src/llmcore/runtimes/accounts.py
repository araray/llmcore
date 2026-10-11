# src/llmcore/runtimes/accounts.py
"""Account-level reads: SSH keys, balances, spending limits and usage.

A runtime is one instance; an account is what pays for all of them and what
they are reached with. These are small typed views of what the vendors'
account endpoints return, shaped so a control panel can show them without
knowing each vendor's JSON.

**The money reads never raise.** :class:`AccountBalance`,
:class:`SpendingLimit` and :class:`UsageReport` carry an ``error`` instead: a
dashboard showing three vendors must not go blank because one of them timed
out, and "could not read the balance" is itself the thing to show. Every
figure is in US dollars and ``None`` means *unknown* -- never zero, because
reading an unknown balance or burn rate as zero is how a panel reports that
nothing is running while something is.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OPENROUTER_API_BASE",
    "AccountBalance",
    "SpendingLimit",
    "SshKey",
    "UsageBucket",
    "UsageReport",
    "check_public_key",
    "openrouter_balance",
]

#: OpenRouter's API root; ``/credits`` under it reports the account's credit.
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

#: OpenSSH public-key type prefixes accepted for upload.
_PUBLIC_KEY_TYPES = (
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-",
)


def _parse_ts(value: Any) -> datetime | None:
    """An API timestamp: ISO 8601 (``Z`` tolerated) or epoch seconds."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def check_public_key(public_key: str) -> str:
    """Return *public_key* stripped, or raise ``ValueError`` if it is not one.

    Refuses anything that looks like a **private** key outright: registering
    one would send it to a vendor, and the mistake (pointing at ``id_ed25519``
    instead of ``id_ed25519.pub``) is an easy one to make.
    """
    text = (public_key or "").strip()
    if "PRIVATE KEY" in text:
        raise ValueError(
            "That is a private key. Pass the public half (the .pub file); a "
            "private key must never be uploaded anywhere."
        )
    if not text.startswith(_PUBLIC_KEY_TYPES) or len(text.split()) < 2:
        raise ValueError(
            "Not an OpenSSH public key: expected '<type> <base64> [comment]' "
            "with a type such as ssh-ed25519 or ssh-rsa."
        )
    if "\n" in text:
        raise ValueError("Pass one public key, on one line.")
    return text


@dataclass(frozen=True, slots=True)
class SshKey:
    """An SSH public key registered with a vendor.

    Attributes:
        provider: ``"gpuai"`` or ``"deepinfra"``.
        id: The vendor's key id -- what a launch takes (gpu.ai ``ssh_key_ids``).
        name: The label it was registered under.
        fingerprint: As the vendor reports it (gpu.ai), else ``None``.
        public_key: The key itself, when the vendor returns it (DeepInfra).
        created_at: When it was registered, when reported.
    """

    provider: str
    id: str
    name: str
    fingerprint: str | None = None
    public_key: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, provider: str, row: Mapping[str, Any]) -> SshKey:
        return cls(
            provider=provider,
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            fingerprint=str(row["fingerprint"]) if row.get("fingerprint") else None,
            public_key=str(row.get("public_key") or row.get("key") or "") or None,
            created_at=_parse_ts(row.get("created_at")),
        )


# ---------------------------------------------------------------------------
# Money: balance, spending limit, usage
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def describe_failure(exc: BaseException) -> str:
    """One line for an error value. The exception's own text, never a key:
    the backends' errors carry status and response body, not headers."""
    text = str(exc).strip() or type(exc).__name__
    return text.splitlines()[0][:400]


def _dollars(cents: Any) -> float | None:
    if cents is None:
        return None
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class AccountBalance:
    """What an account can still spend, and how fast it is spending it.

    Attributes:
        provider: ``"gpuai"``, ``"deepinfra"`` or ``"openrouter"``.
        balance_usd: Credit available to spend. Negative means money is owed.
        burn_usd_per_hour: The committed hourly rate of everything live
            (gpu.ai only). ``None`` is unknown, **not** zero.
        spent_usd: Usage the vendor reports alongside the balance:
            OpenRouter's lifetime ``total_usage``, DeepInfra's usage since the
            last invoice.
        total_credits_usd: Credit ever bought (OpenRouter).
        limit_usd: An account spending cap reported with the balance
            (DeepInfra); ``None`` when there is none or it is not reported.
        scoped_credit_usd: Credit spendable only on certain models
            (DeepInfra), not counted in ``balance_usd``.
        suspended: Whether the vendor has suspended the account.
        auto_pay: Whether automatic top-up is on.
        fetched_at: When this was read.
        error: Why it could not be read; every figure is ``None`` then.
        notes: Caveats worth showing beside the figures.
    """

    provider: str
    balance_usd: float | None = None
    burn_usd_per_hour: float | None = None
    spent_usd: float | None = None
    total_credits_usd: float | None = None
    limit_usd: float | None = None
    scoped_credit_usd: float | None = None
    suspended: bool | None = None
    auto_pay: bool | None = None
    fetched_at: datetime = field(default_factory=_now)
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the read succeeded."""
        return self.error is None

    @property
    def runway_hours(self) -> float | None:
        """Hours the balance lasts at the current burn rate.

        ``None`` whenever that is undefined -- no balance, no credit left, or a
        burn rate that is zero or unknown -- rather than an infinity or a zero
        that would read as a fact.
        """
        if not self.balance_usd or self.balance_usd <= 0:
            return None
        if not self.burn_usd_per_hour or self.burn_usd_per_hour <= 0:
            return None
        return self.balance_usd / self.burn_usd_per_hour

    @classmethod
    def failed(cls, provider: str, exc: BaseException | str) -> AccountBalance:
        """An error value for a read that did not succeed."""
        text = exc if isinstance(exc, str) else describe_failure(exc)
        return cls(provider=provider, error=text)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping, with ``runway_hours`` included."""
        return {
            "provider": self.provider,
            "balance_usd": self.balance_usd,
            "burn_usd_per_hour": self.burn_usd_per_hour,
            "spent_usd": self.spent_usd,
            "total_credits_usd": self.total_credits_usd,
            "limit_usd": self.limit_usd,
            "scoped_credit_usd": self.scoped_credit_usd,
            "suspended": self.suspended,
            "auto_pay": self.auto_pay,
            "runway_hours": self.runway_hours,
            "fetched_at": self.fetched_at.isoformat(),
            "error": self.error,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SpendingLimit:
    """An account-wide spending cap and how much of it is used.

    Attributes:
        provider: ``"gpuai"`` or ``"deepinfra"``.
        configured: Whether a cap is set at all; ``None`` if unknown (error).
        monthly_limit_usd: gpu.ai's monthly cap.
        month_spend_usd: Spend this billing month (gpu.ai).
        daily_limit_usd: gpu.ai's daily cap, when set.
        day_spend_usd: Spend today (gpu.ai).
        percent_used: Of the monthly cap (gpu.ai).
        enforcement: What happens when the cap trips (gpu.ai):
            ``stop_running``, ``block_launch`` or ``notify_only``.
        auto_terminate_hours: Lifetime gpu.ai applies when the cap trips.
        billing_cycle_start: Start of gpu.ai's current billing month.
        limit_usd: DeepInfra's cap, whose period the API does not state.
        fetched_at: When this was read.
        error: Why it could not be read.
        notes: Caveats worth showing beside the figures.
    """

    provider: str
    configured: bool | None = None
    monthly_limit_usd: float | None = None
    month_spend_usd: float | None = None
    daily_limit_usd: float | None = None
    day_spend_usd: float | None = None
    percent_used: float | None = None
    enforcement: str | None = None
    auto_terminate_hours: int | None = None
    billing_cycle_start: datetime | None = None
    limit_usd: float | None = None
    fetched_at: datetime = field(default_factory=_now)
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the read succeeded."""
        return self.error is None

    @classmethod
    def failed(cls, provider: str, exc: BaseException | str) -> SpendingLimit:
        """An error value for a read that did not succeed."""
        text = exc if isinstance(exc, str) else describe_failure(exc)
        return cls(provider=provider, error=text)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping."""
        return {
            "provider": self.provider,
            "configured": self.configured,
            "monthly_limit_usd": self.monthly_limit_usd,
            "month_spend_usd": self.month_spend_usd,
            "daily_limit_usd": self.daily_limit_usd,
            "day_spend_usd": self.day_spend_usd,
            "percent_used": self.percent_used,
            "enforcement": self.enforcement,
            "auto_terminate_hours": self.auto_terminate_hours,
            "billing_cycle_start": (
                self.billing_cycle_start.isoformat() if self.billing_cycle_start else None
            ),
            "limit_usd": self.limit_usd,
            "fetched_at": self.fetched_at.isoformat(),
            "error": self.error,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class UsageBucket:
    """Metered usage for one time bucket (and one instance or GPU type).

    Attributes:
        start: Start of the bucket.
        cost_usd: What the vendor billed for it.
        gpu_seconds: GPU-seconds metered in it.
        instance_id: The instance, when grouped by instance.
        gpu_type: The GPU type, when grouped by GPU type.
    """

    start: datetime | None
    cost_usd: float
    gpu_seconds: int | None = None
    instance_id: str | None = None
    gpu_type: str | None = None


@dataclass(frozen=True, slots=True)
class UsageReport:
    """The vendor's own meter: what was billed, bucket by bucket.

    The figure to reconcile against an estimate. A local ``rate x elapsed``
    over-counts -- gpu.ai does not bill provisioning and credits back windows
    when an instance was unreachable -- and this is what was actually charged.

    Attributes:
        provider: ``"gpuai"``.
        buckets: One row per bucket (and group), in the vendor's order.
        bucket: The bucket size asked for, when one was.
        group_by: The grouping asked for, when one was.
        start: Window start asked for.
        end: Window end asked for.
        fetched_at: When this was read.
        error: Why it could not be read; ``buckets`` is empty then.
    """

    provider: str
    buckets: tuple[UsageBucket, ...] = ()
    bucket: str | None = None
    group_by: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    fetched_at: datetime = field(default_factory=_now)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the read succeeded."""
        return self.error is None

    @property
    def total_cost_usd(self) -> float | None:
        """Everything billed in the window; ``None`` when the read failed."""
        if self.error is not None:
            return None
        return sum(b.cost_usd for b in self.buckets)

    def by_instance(self) -> dict[str, float]:
        """Dollars per instance id, summed over buckets."""
        out: dict[str, float] = {}
        for b in self.buckets:
            if b.instance_id:
                out[b.instance_id] = out.get(b.instance_id, 0.0) + b.cost_usd
        return out

    @classmethod
    def failed(cls, provider: str, exc: BaseException | str, **asked: Any) -> UsageReport:
        """An error value for a read that did not succeed."""
        text = exc if isinstance(exc, str) else describe_failure(exc)
        return cls(provider=provider, error=text, **asked)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping."""

        def iso(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "group_by": self.group_by,
            "start": iso(self.start),
            "end": iso(self.end),
            "total_cost_usd": self.total_cost_usd,
            "buckets": [
                {
                    "start": iso(b.start),
                    "cost_usd": b.cost_usd,
                    "gpu_seconds": b.gpu_seconds,
                    "instance_id": b.instance_id,
                    "gpu_type": b.gpu_type,
                }
                for b in self.buckets
            ],
            "fetched_at": self.fetched_at.isoformat(),
            "error": self.error,
        }


async def openrouter_balance(
    *,
    api_key: str | None = None,
    api_key_env_var: str = "OPENROUTER_API_KEY",
    base_url: str = DEFAULT_OPENROUTER_API_BASE,
    client: Any = None,
) -> AccountBalance:
    """OpenRouter's remaining credit, from ``GET /credits``. Never raises.

    OpenRouter is a provider, not a rental, but it draws on prepaid credit the
    same way, so a control panel wants it beside the others. The balance is
    ``total_credits - total_usage``.

    Args:
        api_key: The key; read from *api_key_env_var* when omitted.
        api_key_env_var: Environment variable holding the key.
        base_url: API root.
        client: An ``httpx.AsyncClient``-shaped object, for tests; a
            short-lived client is made and closed otherwise.
    """
    provider = "openrouter"
    key = api_key or os.environ.get(api_key_env_var)
    if not key:
        return AccountBalance.failed(provider, f"no OpenRouter API key (set {api_key_env_var})")
    owned = client is None
    try:
        if owned:
            import httpx

            client = httpx.AsyncClient(timeout=30.0)
        response = await client.request(
            "GET",
            f"{base_url.rstrip('/')}/credits",
            headers={"Authorization": f"Bearer {key}"},
        )
        if response.status_code >= 400:
            return AccountBalance.failed(
                provider,
                f"OpenRouter GET /credits failed with {response.status_code}: "
                f"{response.text[:300]}",
            )
        data = (response.json() or {}).get("data") or {}
        bought = _number(data.get("total_credits"))
        used = _number(data.get("total_usage"))
        balance = bought - used if bought is not None and used is not None else None
        return AccountBalance(
            provider=provider,
            balance_usd=balance,
            spent_usd=used,
            total_credits_usd=bought,
        )
    except Exception as exc:
        logger.debug("OpenRouter balance read failed: %s", exc)
        return AccountBalance.failed(provider, exc)
    finally:
        if owned and client is not None:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - closing is best effort
                pass
