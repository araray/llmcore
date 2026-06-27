"""Unit tests for secret redaction."""

from __future__ import annotations

from llmcore.bridge.redact import redact


def test_bearer_token_redacted():
    out = redact("Authorization: Bearer abcDEF123.ghiJKL456.mnoPQR789")
    assert "abcDEF123.ghiJKL456.mnoPQR789" not in out
    assert "[REDACTED]" in out


def test_sk_key_redacted():
    out = redact("the key is sk-ABCDEFGHIJKLMNOP1234 ok")
    assert "sk-ABCDEFGHIJKLMNOP1234" not in out
    assert "[REDACTED]" in out


def test_plain_text_untouched():
    assert redact("hello world, nothing secret here") == "hello world, nothing secret here"


def test_none_and_empty():
    assert redact(None) == ""
    assert redact("") == ""


def test_env_secret_value_redacted(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "supersecretvalue1234567")
    out = redact("connecting with token supersecretvalue1234567 now")
    assert "supersecretvalue1234567" not in out
    assert "[REDACTED]" in out
