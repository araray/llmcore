# src/llmcore/runtimes/accounts.py
"""Account-level reads for the rental vendors: SSH keys.

A runtime is one instance; an account is what pays for all of them and what
they are reached with. These are small typed views of what the vendors'
account endpoints return, shaped so a control panel can show them without
knowing each vendor's JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = ["SshKey", "check_public_key"]

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
