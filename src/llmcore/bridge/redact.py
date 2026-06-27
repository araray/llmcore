"""Secret redaction for error messages and logs (spec §13.4).

The bridge never lets provider API keys or bearer tokens cross the wire or hit
logs. ``redact`` is applied to every ``LlmcoreError.message`` and to structured
log fields before emission. It is intentionally conservative (false positives
are acceptable; leaks are not).
"""

from __future__ import annotations

import os
import re

__all__ = ["redact"]

# ``Authorization: Bearer <token>`` and bare ``Bearer <token>``.
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")

# Common vendor key shapes: sk-..., pk-..., key-..., tok-..., gsk_..., r8_...,
# xai-..., AKIA... (16+ trailing chars to avoid eating ordinary words).
_KEYISH = re.compile(
    r"\b(?:sk|pk|key|tok|gsk|r8|xai|sk-proj|api)[-_][A-Za-z0-9]{12,}\b"
    r"|\bAKIA[0-9A-Z]{12,}\b"
)

# Long opaque hex/base64-ish runs (>= 32 chars) that look like secrets.
_OPAQUE = re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")

_REPLACEMENT = "[REDACTED]"

# Env vars whose *values* must be scrubbed if they appear verbatim in text.
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY", "_PASSWORD")


def _secret_env_values() -> list[str]:
    out: list[str] = []
    for name, value in os.environ.items():
        if value and len(value) >= 6 and name.upper().endswith(_SECRET_ENV_SUFFIXES):
            out.append(value)
    # Longest first so substrings of one secret don't leave fragments.
    return sorted(set(out), key=len, reverse=True)


def redact(text: str | None) -> str:
    """Return ``text`` with likely secrets replaced by ``[REDACTED]``.

    Args:
        text: Arbitrary string (an exception message, a log field, ...).

    Returns:
        The redacted string. ``None`` becomes an empty string.
    """
    if not text:
        return ""
    redacted = text
    for value in _secret_env_values():
        redacted = redacted.replace(value, _REPLACEMENT)
    redacted = _BEARER.sub("Bearer " + _REPLACEMENT, redacted)
    redacted = _KEYISH.sub(_REPLACEMENT, redacted)
    redacted = _OPAQUE.sub(_REPLACEMENT, redacted)
    return redacted
