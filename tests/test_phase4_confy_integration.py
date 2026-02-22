"""
Phase 4 Integration Tests — Confy-Delegating Config Loaders

Tests the new primary API paths where load_agents_config() and
load_sandbox_config() receive a unified confy Config object, as well
as backward compatibility with deprecated parameters.
"""

import os

# ---------------------------------------------------------------------------
# Ensure confy stub is importable
# ---------------------------------------------------------------------------
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confy_pkg"))

from confy.loader import Config as ConfyConfig
from confy.loader import deep_merge

from llmcore.agents.sandbox.config import (
    SandboxSystemConfig,
    load_sandbox_config,
)
from llmcore.config.agents_config import (
    AgentsConfig,
    load_agents_config,
)

# ===========================================================================
# Fixtures — using actual field names from the Pydantic/dataclass models
# ===========================================================================


@pytest.fixture
def agents_confy_config() -> ConfyConfig:
    """Build a confy Config that contains an [agents] section
    matching AgentsConfig's real Pydantic fields."""
    return ConfyConfig(
        defaults={
            "agents": {
                "max_iterations": 25,
                "default_timeout": 300,
                "default_persona": "researcher",
                "memory_enabled": False,
                "goals": {
                    "classifier_enabled": False,
                },
                "fast_path": {
                    "enabled": True,
                },
            },
            "logging": {"level": "INFO"},
        }
    )


@pytest.fixture
def sandbox_confy_config() -> ConfyConfig:
    """Build a confy Config that contains [agents.sandbox]
    matching SandboxSystemConfig's real dataclass fields."""
    return ConfyConfig(
        defaults={
            "agents": {
                "sandbox": {
                    "mode": "docker",
                    "fallback_enabled": False,
                    "docker": {
                        "image": "python:3.11-slim",
                        "memory_limit": "2g",
                        "cpu_limit": 1.5,
                        "timeout_seconds": 120,
                    },
                },
            },
        }
    )


@pytest.fixture
def full_confy_config() -> ConfyConfig:
    """A confy Config containing both agents and sandbox sections."""
    return ConfyConfig(
        defaults={
            "agents": {
                "max_iterations": 50,
                "default_timeout": 900,
                "default_persona": "coder",
                "memory_enabled": True,
                "sandbox": {
                    "mode": "docker",
                    "fallback_enabled": True,
                    "docker": {
                        "image": "python:3.11-slim",
                        "memory_limit": "4g",
                        "timeout_seconds": 300,
                    },
                },
            },
        }
    )


# ===========================================================================
# Test: load_agents_config with confy Config (new primary API)
# ===========================================================================


class TestLoadAgentsConfigFromConfy:
    """Tests for load_agents_config(config=<confy Config>)."""

    def test_basic_extraction(self, agents_confy_config):
        """Config= kwarg extracts [agents] section correctly."""
        result = load_agents_config(config=agents_confy_config)

        assert isinstance(result, AgentsConfig)
        assert result.max_iterations == 25
        assert result.default_timeout == 300
        assert result.default_persona == "researcher"
        assert result.memory_enabled is False

    def test_nested_sub_config(self, agents_confy_config):
        """Nested sub-sections like [agents.goals] are extracted."""
        result = load_agents_config(config=agents_confy_config)

        assert result.goals.classifier_enabled is False

    def test_overrides_on_top_of_confy(self, agents_confy_config):
        """Runtime overrides merge on top of confy-extracted data."""
        overrides = {
            "max_iterations": 100,
            "goals": {"classifier_enabled": True},
        }
        result = load_agents_config(config=agents_confy_config, overrides=overrides)

        assert result.max_iterations == 100
        assert result.goals.classifier_enabled is True
        # Non-overridden values preserved
        assert result.default_persona == "researcher"

    def test_empty_confy_gives_defaults(self):
        """Empty confy Config -> fall back to AgentsConfig defaults."""
        cfg = ConfyConfig()
        result = load_agents_config(config=cfg)

        assert isinstance(result, AgentsConfig)
        # Default values from the Pydantic model
        assert result.max_iterations == 10
        assert result.default_timeout == 600
        assert result.default_persona == "assistant"
        assert result.memory_enabled is True

    def test_no_deprecation_warning(self, agents_confy_config):
        """Using config= does NOT trigger deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_agents_config(config=agents_confy_config)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0

    def test_extra_keys_are_ignored(self):
        """Keys outside AgentsConfig schema are silently ignored."""
        cfg = ConfyConfig(
            defaults={
                "agents": {
                    "max_iterations": 15,
                    "unknown_field_xyz": "should be ignored",
                },
            }
        )
        result = load_agents_config(config=cfg)

        assert isinstance(result, AgentsConfig)
        assert result.max_iterations == 15


# ===========================================================================
# Test: load_sandbox_config with confy Config (new primary API)
# ===========================================================================


class TestLoadSandboxConfigFromConfy:
    """Tests for load_sandbox_config(config=<confy Config>)."""

    def test_basic_extraction(self, sandbox_confy_config):
        """Config= kwarg extracts [agents.sandbox] correctly."""
        result = load_sandbox_config(config=sandbox_confy_config)

        assert isinstance(result, SandboxSystemConfig)
        assert result.mode == "docker"
        assert result.fallback_enabled is False

    def test_docker_sub_section(self, sandbox_confy_config):
        """Docker settings extracted from nested [agents.sandbox.docker]."""
        result = load_sandbox_config(config=sandbox_confy_config)

        assert result.docker.image == "python:3.11-slim"
        assert result.docker.memory_limit == "2g"
        assert result.docker.cpu_limit == 1.5
        assert result.docker.timeout_seconds == 120

    def test_overrides_on_top_of_confy(self, sandbox_confy_config):
        """Runtime overrides merge on top of confy-extracted data."""
        overrides = {"docker": {"memory_limit": "8g"}}
        result = load_sandbox_config(config=sandbox_confy_config, overrides=overrides)

        assert result.docker.memory_limit == "8g"
        # Non-overridden preserved
        assert result.docker.image == "python:3.11-slim"
        assert result.docker.cpu_limit == 1.5

    def test_empty_confy_gives_defaults(self):
        """Empty confy Config -> fall back to SandboxSystemConfig defaults."""
        cfg = ConfyConfig()
        result = load_sandbox_config(config=cfg)

        assert isinstance(result, SandboxSystemConfig)
        # Dataclass defaults
        assert result.mode == "docker"
        assert result.docker.image == "python:3.11-slim"

    def test_no_deprecation_warning(self, sandbox_confy_config):
        """Using config= does NOT trigger deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_sandbox_config(config=sandbox_confy_config)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# ===========================================================================
# Test: Full config with both agents and sandbox sections
# ===========================================================================


class TestFullConfigExtraction:
    """Test extracting both agents and sandbox from a single confy Config."""

    def test_agents_from_full_config(self, full_confy_config):
        """load_agents_config extracts [agents] from full config."""
        result = load_agents_config(config=full_confy_config)

        assert result.max_iterations == 50
        assert result.default_timeout == 900
        assert result.default_persona == "coder"

    def test_sandbox_from_full_config(self, full_confy_config):
        """load_sandbox_config extracts [agents.sandbox] from full config."""
        result = load_sandbox_config(config=full_confy_config)

        assert isinstance(result, SandboxSystemConfig)
        assert result.docker.memory_limit == "4g"
        assert result.docker.timeout_seconds == 300


# ===========================================================================
# Test: Backward Compatibility — Deprecated Parameters
# ===========================================================================


class TestBackwardCompat:
    """Verify old-style API still works with deprecation warnings."""

    def test_agents_config_path_triggers_deprecation(self, tmp_path):
        """config_path= triggers DeprecationWarning."""
        toml_content = b"""
[agents]
max_iterations = 42
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_bytes(toml_content)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_agents_config(config_path=config_file)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

        assert isinstance(result, AgentsConfig)

    def test_agents_config_dict_triggers_deprecation(self):
        """config_dict= triggers DeprecationWarning."""
        config_dict = {"agents": {"max_iterations": 77}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_agents_config(config_dict=config_dict)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

        assert result.max_iterations == 77

    def test_sandbox_config_path_triggers_deprecation(self, tmp_path):
        """config_path= triggers DeprecationWarning for sandbox."""
        toml_content = b"""
[agents.sandbox]
mode = "docker"

[agents.sandbox.docker]
image = "python:3.11-slim"
"""
        config_file = tmp_path / "sandbox_config.toml"
        config_file.write_bytes(toml_content)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_sandbox_config(config_path=config_file)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

        assert isinstance(result, SandboxSystemConfig)

    def test_no_args_no_deprecation_agents(self):
        """Calling load_agents_config() with no arguments -> no deprecation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_agents_config()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0

        assert isinstance(result, AgentsConfig)

    def test_no_args_no_deprecation_sandbox(self):
        """Calling load_sandbox_config() with no arguments -> no deprecation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_sandbox_config()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0

        assert isinstance(result, SandboxSystemConfig)


# ===========================================================================
# Test: Confy deep_merge delegation
# ===========================================================================


class TestDeepMergeDelegation:
    """Verify deep_merge from confy works correctly in config loaders."""

    def test_confy_deep_merge_basic(self):
        """confy.loader.deep_merge does recursive merge."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        updates = {"b": {"c": 99, "e": 4}, "f": 5}
        result = deep_merge(base, updates)

        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["b"]["e"] == 4
        assert result["f"] == 5

    def test_confy_deep_merge_no_mutation(self):
        """deep_merge doesn't mutate base or updates."""
        base = {"a": {"b": 1}}
        updates = {"a": {"c": 2}}
        result = deep_merge(base, updates)

        assert "c" not in base["a"]
        assert "b" not in updates["a"]
        assert result["a"]["b"] == 1
        assert result["a"]["c"] == 2


# ===========================================================================
# Test: Confy Config multi-file loading
# ===========================================================================


class TestConfyMultiFileLoading:
    """Test confy Config with file_paths (Phase 1 feature used by api.py)."""

    def test_single_file_path(self, tmp_path):
        """Load a single TOML file via file_paths."""
        toml_content = b"""
[agents]
max_iterations = 33
"""
        config_file = tmp_path / "config.toml"
        config_file.write_bytes(toml_content)

        cfg = ConfyConfig(file_paths=[str(config_file)])
        assert cfg.agents.max_iterations == 33

    def test_multi_file_merge(self, tmp_path):
        """Later files override earlier files."""
        base_toml = b"""
[agents]
max_iterations = 10
default_persona = "assistant"
"""
        override_toml = b"""
[agents]
max_iterations = 50
"""
        base_file = tmp_path / "base.toml"
        base_file.write_bytes(base_toml)
        override_file = tmp_path / "override.toml"
        override_file.write_bytes(override_toml)

        cfg = ConfyConfig(file_paths=[str(base_file), str(override_file)])
        assert cfg.agents.max_iterations == 50
        assert cfg.agents.default_persona == "assistant"  # Preserved from base

    def test_namespaced_file(self, tmp_path):
        """Namespaced file_paths entry extracts a specific section."""
        toml_content = b"""
[tool.semantiscan]
chunk_size = 1500

[other_section]
irrelevant = true
"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_bytes(toml_content)

        cfg = ConfyConfig(file_paths=[(str(pyproject), "semantiscan")])
        assert cfg.semantiscan.chunk_size == 1500

    def test_missing_file_skipped(self, tmp_path):
        """Missing files in file_paths are silently skipped."""
        toml_content = b"""
[agents]
max_iterations = 20
"""
        real_file = tmp_path / "real.toml"
        real_file.write_bytes(toml_content)

        cfg = ConfyConfig(
            file_paths=[
                "/nonexistent/missing.toml",
                str(real_file),
            ]
        )
        assert cfg.agents.max_iterations == 20

    def test_app_defaults(self):
        """app_defaults are merged under their namespace at lowest precedence."""
        cfg = ConfyConfig(
            defaults={"agents": {"max_iterations": 10}},
            app_defaults={"semantiscan": {"chunk_size": 1000, "overlap": 200}},
        )
        assert cfg.agents.max_iterations == 10
        assert cfg.semantiscan.chunk_size == 1000
        assert cfg.semantiscan.overlap == 200


# ===========================================================================
# Test: Environment variable routing
# ===========================================================================


class TestEnvVarRouting:
    """Test env var collection with primary and app prefixes."""

    def test_primary_prefix(self):
        """Primary prefix env vars are collected.
        Convention: _ = dot separator, __ = literal underscore in key."""
        with patch.dict(os.environ, {"LLMCORE_AGENTS_MAX__ITERATIONS": "99"}, clear=False):
            cfg = ConfyConfig(prefix="LLMCORE")
            assert cfg.agents.max_iterations == 99

    def test_app_prefix(self):
        """App prefix env vars are nested under the app name."""
        with patch.dict(
            os.environ,
            {"SEMANTISCAN_CHUNK__SIZE": "2000"},
            clear=False,
        ):
            cfg = ConfyConfig(app_prefixes={"semantiscan": "SEMANTISCAN"})
            assert cfg.semantiscan.chunk_size == 2000

    def test_overrides_dict_highest_precedence(self):
        """overrides_dict beats both files and env vars."""
        with patch.dict(os.environ, {"LLMCORE_AGENTS_MAX__ITERATIONS": "99"}, clear=False):
            cfg = ConfyConfig(
                prefix="LLMCORE",
                overrides_dict={"agents.max_iterations": "5"},
            )
            assert cfg.agents.max_iterations == 5


# ===========================================================================
# Test: Config.app() accessor
# ===========================================================================


class TestConfyAppAccessor:
    """Test confy Config.app() method used by downstream consumers."""

    def test_app_returns_config_object(self):
        """app() returns a Config for an existing section."""
        cfg = ConfyConfig(defaults={"semantiscan": {"chunk_size": 500}})
        app_cfg = cfg.app("semantiscan")

        assert isinstance(app_cfg, ConfyConfig)
        assert app_cfg.chunk_size == 500

    def test_app_returns_empty_config_for_missing(self):
        """app() returns empty Config for a section that doesn't exist."""
        cfg = ConfyConfig()
        app_cfg = cfg.app("nonexistent")

        assert isinstance(app_cfg, ConfyConfig)
        assert len(app_cfg) == 0

    def test_app_created_section_persists(self):
        """app() creates and persists the section in the config."""
        cfg = ConfyConfig()
        app_cfg = cfg.app("new_app")
        app_cfg["key"] = "value"

        assert cfg.new_app.key == "value"


# ===========================================================================
# Test: as_dict() serialization
# ===========================================================================


class TestConfyAsDict:
    """Test Config.as_dict() for serialization."""

    def test_nested_config_to_plain_dict(self):
        """as_dict() recursively converts Config->dict."""
        cfg = ConfyConfig(defaults={"a": {"b": {"c": 1}}})
        d = cfg.as_dict()

        assert isinstance(d, dict)
        assert not isinstance(d["a"], ConfyConfig)
        assert d["a"]["b"]["c"] == 1

    def test_round_trip(self):
        """as_dict() output can recreate the same config."""
        original = {"agents": {"max_iterations": 15, "goals": {"classifier_enabled": False}}}
        cfg = ConfyConfig(defaults=original)
        d = cfg.as_dict()

        cfg2 = ConfyConfig(defaults=d)
        assert cfg2.agents.max_iterations == 15
        assert cfg2.agents.goals.classifier_enabled is False
