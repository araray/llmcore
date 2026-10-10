# tests/api/test_init_failure_cleanup.py
"""A failed ``LLMCore.create()`` must not leave resources open.

This is not tidiness. ``StorageManager`` is constructed *before* the providers,
and its aiosqlite backend starts a **non-daemon** worker thread. When provider
setup then failed, that thread outlived the exception — so a caller that caught
the ``ConfigError`` and carried on could never exit, because interpreter
shutdown blocks forever joining it.

Observed symptom: ``llmcore-runtimes estimate`` falls back to a runtimes-only
path when the library cannot fully start. It computed the correct answer and
then hung until killed, with stdout never flushed. Nothing in the error
message pointed at storage.
"""

from __future__ import annotations

import pytest

from llmcore import LLMCore
from llmcore.exceptions import ConfigError


class _Spy:
    """A faithful enough StorageManager double: it opens, then records closes.

    ``initialize_storages()`` is part of the real sequence (api.py line 622),
    and leaving it out made the double raise ``AttributeError`` *before* the
    failure under test, which is how this file's first draft passed for the
    wrong reason.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.closed = 0
        self.opened = 0

    async def initialize_storages(self) -> None:
        self.opened += 1

    async def close(self) -> None:
        self.closed += 1

    async def close_all(self) -> None:
        self.closed += 1


@pytest.fixture
def failing_providers(monkeypatch):
    """Make provider setup fail the way a bad default provider does."""
    storage = _Spy()
    monkeypatch.setattr("llmcore.api.StorageManager", lambda *a, **k: storage)

    def explode(*args, **kwargs):
        raise ConfigError("Default provider 'nope' is not configured")

    monkeypatch.setattr("llmcore.api.ProviderManager", explode)
    return storage


class TestInitializationFailureReleasesResources:
    @pytest.mark.asyncio
    async def test_storage_is_closed_when_providers_fail(self, failing_providers):
        with pytest.raises(ConfigError):
            await LLMCore.create()
        assert failing_providers.closed == 1, (
            "storage opened before the failure was left open; its aiosqlite "
            "worker thread is non-daemon, so the process cannot exit"
        )

    @pytest.mark.asyncio
    async def test_the_original_failure_is_still_what_propagates(self, failing_providers):
        """Cleanup must not replace the error worth reporting."""
        with pytest.raises(ConfigError, match="not configured"):
            await LLMCore.create()

    @pytest.mark.asyncio
    async def test_the_cause_is_chained(self, monkeypatch):
        """`raise ... from e`, so the real traceback survives the wrapping."""
        monkeypatch.setattr("llmcore.api.StorageManager", lambda *a, **k: _Spy())

        def explode(*args, **kwargs):
            raise RuntimeError("the actual problem")

        monkeypatch.setattr("llmcore.api.ProviderManager", explode)
        with pytest.raises(ConfigError) as excinfo:
            await LLMCore.create()
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_cleanup_errors_do_not_mask_the_failure(self, monkeypatch):
        """A manager that throws on close is not a reason to lose the reason
        initialization failed."""

        class Hostile:
            async def initialize_storages(self) -> None:
                return None

            async def close(self) -> None:
                raise OSError("cannot close")

        monkeypatch.setattr("llmcore.api.StorageManager", lambda *a, **k: Hostile())

        def explode(*args, **kwargs):
            raise ConfigError("the real reason")

        monkeypatch.setattr("llmcore.api.ProviderManager", explode)
        with pytest.raises(ConfigError, match="the real reason"):
            await LLMCore.create()

    @pytest.mark.asyncio
    async def test_a_manager_that_never_got_built_is_skipped(self, monkeypatch):
        """Initialization can fail before storage exists at all, and cleanup
        must not turn that into an AttributeError."""

        def explode(*args, **kwargs):
            raise ConfigError("failed very early")

        monkeypatch.setattr("llmcore.api.StorageManager", explode)
        with pytest.raises(ConfigError, match="failed very early"):
            await LLMCore.create()
