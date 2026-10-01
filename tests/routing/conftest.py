# tests/routing/conftest.py
"""Shared doubles for routing tests.

Routing is the one subsystem whose job is to decide *which* provider gets
called, so these fakes are deliberately dumb about everything except being
counted and being made to fail. A test that needed a real provider would be
testing the provider.
"""

from __future__ import annotations

import tomllib
from typing import Any

import pytest

from llmcore.routing.models import Target


class FakeProvider:
    """A provider that records its calls and can be scripted to fail.

    ``behaviour`` is consumed one entry per call, the last entry repeating, so
    ``[ProviderError(...), "ok"]`` means "fail once, then work" -- which is the
    shape every retry test needs.
    """

    def __init__(self, name: str, behaviour: list[Any] | None = None) -> None:
        self._name = name
        self.behaviour = list(behaviour or ["ok"])
        self.calls = 0
        self.params: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return self._name

    async def respond(self, params: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        self.params.append(dict(params or {}))
        entry = self.behaviour[min(self.calls - 1, len(self.behaviour) - 1)]
        if isinstance(entry, BaseException):
            raise entry
        return entry


class FakeProviderManager:
    """Resolves a target by its provider name, and nothing else."""

    def __init__(self, providers: dict[str, FakeProvider]) -> None:
        self.providers = providers
        self.resolutions: list[str] = []

    def resolve_target(
        self, target: Any, *, autoprovision: bool = True, cache: bool = True
    ) -> FakeProvider:
        parsed = target if isinstance(target, Target) else Target.parse(target)
        self.resolutions.append(parsed.key)
        provider = self.providers.get(parsed.provider)
        if provider is None:
            raise RuntimeError(f"no provider named {parsed.provider!r}")
        return provider

    def get_provider(self, name: str | None = None) -> FakeProvider:
        return next(iter(self.providers.values()))


@pytest.fixture
def providers() -> dict[str, FakeProvider]:
    return {
        "alpha": FakeProvider("alpha"),
        "beta": FakeProvider("beta"),
        "gamma": FakeProvider("gamma"),
        "ollama": FakeProvider("ollama"),
    }


@pytest.fixture
def provider_manager(providers) -> FakeProviderManager:
    return FakeProviderManager(providers)


@pytest.fixture
def make_config():
    """Build a real confy ``Config`` from TOML text.

    A real config object rather than a stub dict, because the nested-table
    reads routing relies on (``routing.pools.main``) behave differently from a
    flat mapping -- a stub that answered dotted keys would have hidden a bug
    that a real config does not have.

    ``load_dotenv_file=False`` is not optional. confy loads a ``.env`` from the
    working directory by default and exports it into ``os.environ``, so without
    this a test run picks up whatever real provider credentials the developer
    has sitting in the repo -- which makes credential-discovery assertions
    depend on the machine, and makes it possible for a test to reach a real
    vendor and spend real money.
    """
    from confy.loader import Config

    def build(toml: str) -> Any:
        return Config(
            defaults=tomllib.loads(toml), prefix="LLMCORE", load_dotenv_file=False
        )

    return build


@pytest.fixture
def runner():
    """A runner that just asks the provider to respond."""

    async def run(provider: FakeProvider, target: Target, params: dict[str, Any]) -> Any:
        return await provider.respond(params)

    return run
