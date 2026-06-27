"""Deterministic test doubles for the bridge (test-scope only).

These live under ``llmcore.bridge._testing`` and are imported by the bridge test
suite. They are NOT part of the public bridge surface and add no dependency to
``llmcore`` core:

* :class:`FakeFacade` — satisfies ``LLMCoreFacade`` with deterministic, offline
  behaviour (echo responses, computed usage/cost, scriptable error/cancel
  triggers). Used for the authoritative transport+adapter conformance suite.
* :class:`FakeProvider` — a ``BaseProvider`` subclass registered into
  ``PROVIDER_MAP`` for the *real-LLMCore* integration test (skip-guarded on
  confy availability).
"""

from __future__ import annotations

from .fake_facade import FakeFacade, fake_count_tokens
from .fake_provider import FakeAudioProvider, FakeProvider, register_fake_provider

__all__ = [
    "FakeFacade",
    "FakeProvider",
    "FakeAudioProvider",
    "fake_count_tokens",
    "register_fake_provider",
]
