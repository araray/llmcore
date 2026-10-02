# tests/agents/test_fast_path_config_wiring.py
"""The `[agents.fast_path]` config section must actually take effect.

`FastPathExecutor` accepts a `config`, but `single_agent` constructed it
without one, so the executor fell back to its own module-level defaults and
everything in the user's section except `enabled` was inert -- a cache size
or TTL a user set did nothing at all.

Two classes named `FastPathConfig` exist: the user-facing pydantic section
and the executor's runtime class. They disagree about field names
(`cache_enabled` vs `use_cache`, `templates_enabled` vs `use_templates`) and
about which fields exist, which is why the translation between them is
written out field by field and pinned here. Anything automatic would drop
precisely the renamed fields and leave them defaulted -- reproducing the bug
while looking correct.
"""

from __future__ import annotations

import pytest

from llmcore.agents.learning.fast_path import (
    FastPathConfig as RuntimeConfig,
)
from llmcore.agents.learning.fast_path import (
    FastPathExecutor,
    ResponseCache,
)
from llmcore.config.agents_config import FastPathConfig as UserConfig


class TestTranslation:
    def test_cache_sizing_is_carried_over(self):
        runtime = RuntimeConfig.from_agents_config(
            UserConfig(cache_max_entries=250, cache_ttl_seconds=90)
        )
        assert runtime.cache_max_entries == 250
        assert runtime.cache_ttl_seconds == 90.0

    def test_renamed_cache_flag_is_mapped(self):
        # cache_enabled -> use_cache
        assert RuntimeConfig.from_agents_config(
            UserConfig(cache_enabled=False)).use_cache is False
        assert RuntimeConfig.from_agents_config(
            UserConfig(cache_enabled=True)).use_cache is True

    def test_renamed_templates_flag_is_mapped(self):
        # templates_enabled -> use_templates
        assert RuntimeConfig.from_agents_config(
            UserConfig(templates_enabled=False)).use_templates is False

    def test_shared_names_are_carried_over(self):
        runtime = RuntimeConfig.from_agents_config(
            UserConfig(max_response_time_ms=1234, temperature=0.1,
                       max_tokens=60, fallback_on_timeout=False)
        )
        assert runtime.max_response_time_ms == 1234
        assert runtime.temperature == 0.1
        assert runtime.max_tokens == 60
        assert runtime.fallback_on_timeout is False

    def test_a_duck_typed_section_falls_back_to_defaults(self):
        class Partial:
            cache_max_entries = 7

        runtime = RuntimeConfig.from_agents_config(Partial())
        assert runtime.cache_max_entries == 7
        assert runtime.use_cache is True          # default, not a crash
        assert runtime.max_response_time_ms == 5000

    def test_a_mock_section_degrades_to_defaults(self):
        # Callers build managers with mocks; an attribute can exist and
        # still not be a number, which used to raise from the constructor.
        from unittest.mock import MagicMock

        runtime = RuntimeConfig.from_agents_config(MagicMock())
        assert runtime.cache_ttl_seconds == 3600.0
        assert runtime.cache_max_entries == 100
        assert runtime.max_response_time_ms == 5000

    def test_none_section_yields_defaults(self):
        runtime = RuntimeConfig.from_agents_config(None)
        assert runtime.cache_max_entries == 100
        assert runtime.use_cache is True


class TestExecutorHonoursIt:
    def test_cache_is_built_with_the_configured_size_and_ttl(self):
        executor = FastPathExecutor(
            config=RuntimeConfig.from_agents_config(
                UserConfig(cache_max_entries=250, cache_ttl_seconds=90)
            )
        )
        assert isinstance(executor._cache, ResponseCache)
        assert executor._cache.max_entries == 250
        assert executor._cache.ttl_seconds == 90.0

    def test_disabling_the_cache_builds_none(self):
        executor = FastPathExecutor(
            config=RuntimeConfig.from_agents_config(UserConfig(cache_enabled=False))
        )
        assert executor._cache is None

    def test_omitting_config_still_works_with_defaults(self):
        # Regression guard: the executor is public API and may be built bare.
        executor = FastPathExecutor()
        assert executor._cache.max_entries == 100
        assert executor._cache.ttl_seconds == 3600.0


class TestTheTwoClassesStillDiffer:
    """A guard, not an aspiration: if they are ever unified, delete this."""

    def test_the_user_section_uses_cache_enabled(self):
        assert "cache_enabled" in UserConfig.model_fields
        assert "use_cache" not in UserConfig.model_fields

    def test_the_runtime_class_uses_use_cache(self):
        runtime = RuntimeConfig()
        assert hasattr(runtime, "use_cache")
        assert not hasattr(runtime, "cache_enabled")

    def test_sizing_fields_exist_on_both_now(self):
        # These were absent from the runtime class, which is why the
        # executor could not honour them even when handed a config.
        assert "cache_max_entries" in UserConfig.model_fields
        assert hasattr(RuntimeConfig(), "cache_max_entries")
