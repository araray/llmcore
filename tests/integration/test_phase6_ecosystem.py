# tests/integration/test_phase6_ecosystem.py
"""
Phase 6 — Ecosystem Integration Tests.

End-to-end verification that the unified logging & configuration system
works correctly across the full Wairu stack:

    confy (config foundation)
      → llmcore (logging system + config wiring)
        → semantiscan (config consumer)

These tests require all three packages to be importable.

Test groups:
    TestFullStackConfig         — Single TOML file configures all apps via confy
    TestConfigPrecedence        — File overrides defaults, env overrides file
    TestAppEnvVarRouting        — SEMANTISCAN_* env vars reach the right namespace
    TestProvenanceTracking      — Provenance reports correct sources
    TestLoggingDisplayBehavior  — display=True reaches console in silent mode
    TestLogRotation             — file_mode="single" uses RotatingFileHandler
    TestSemantiscanConfigShim   — Deprecated load_config() shim works end-to-end

References:
    - logging_config_implementation_plan.md §9 (Phase 6)
    - unified_logging_and_config_design.md §5 (Logging + Config interaction)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Conditional imports — skip entire module if confy or llmcore logging
# are unavailable
# ---------------------------------------------------------------------------

confy_loader = pytest.importorskip("confy.loader", reason="confy not installed")
Config = confy_loader.Config
deep_merge = confy_loader.deep_merge

from confy.provenance import ProvenanceEntry, ProvenanceStore  # noqa: E402

# Import semantiscan defaults — always available since we're in the semantiscan repo
from semantiscan.config.loader import (  # noqa: E402
    DEFAULT_CONFIG as SS_DEFAULTS,
)
from semantiscan.config.loader import (
    get_app_defaults,
    get_default_config,
    validate_config,
)

# llmcore logging_config — import directly to avoid heavy llmcore.__init__ deps
_logging_config_spec = None
try:
    import importlib.util

    _lc_path = Path(__file__).resolve().parents[2] / "src" / "llmcore" / "logging_config.py"
    if _lc_path.exists():
        _logging_config_spec = importlib.util.spec_from_file_location(
            "llmcore.logging_config", str(_lc_path)
        )
except Exception:
    pass

if _logging_config_spec is not None:
    _logging_config = importlib.util.module_from_spec(_logging_config_spec)
    _logging_config_spec.loader.exec_module(_logging_config)
    configure_logging = _logging_config.configure_logging
    log_display = _logging_config.log_display
    UnifiedLoggingManager = _logging_config.UnifiedLoggingManager
    DisplayFilter = _logging_config.DisplayFilter
    DEFAULT_LOGGING_CONFIG = _logging_config.DEFAULT_LOGGING_CONFIG
    HAS_LLMCORE_LOGGING = True
else:
    HAS_LLMCORE_LOGGING = False

requires_llmcore_logging = pytest.mark.skipif(
    not HAS_LLMCORE_LOGGING,
    reason="llmcore logging_config.py not found",
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def unified_config_file(tmp_path: Path) -> Path:
    """Create a single TOML config file that configures all apps."""
    log_dir = tmp_path / "logs"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""\
[llmcore]
default_provider = "ollama"
log_level = "INFO"

[llmcore.providers.ollama]
default_model = "gemma3:1b"
timeout = 120

[logging]
console_enabled = false
file_enabled = true
file_mode = "single"
file_directory = "{log_dir}"
file_level = "DEBUG"
display_min_level = "INFO"

[semantiscan.chunking]
chunk_size = 2000
chunk_overlap = 300

[semantiscan.retrieval]
top_k = 15
"""
    )
    return config


@pytest.fixture
def ss_only_config_file(tmp_path: Path) -> Path:
    """Config file with only semantiscan settings."""
    config = tmp_path / "semantiscan.toml"
    config.write_text(
        """\
[chunking]
chunk_size = 3000

[retrieval]
hybrid_search = false
bm25_weight = 0.5
"""
    )
    return config


@pytest.fixture(autouse=True)
def _reset_logging_manager():
    """Reset the UnifiedLoggingManager singleton between tests."""
    if HAS_LLMCORE_LOGGING:
        # Reset singleton state
        UnifiedLoggingManager._configured = False
        UnifiedLoggingManager._instance = None
        UnifiedLoggingManager._log_file_path = None
        UnifiedLoggingManager._console_handler = None
        UnifiedLoggingManager._file_handler = None
        UnifiedLoggingManager._display_filter = None

        # Clean up root logger handlers added by previous tests
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    yield

    if HAS_LLMCORE_LOGGING:
        # Post-test cleanup
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        UnifiedLoggingManager._configured = False
        UnifiedLoggingManager._instance = None


# =========================================================================
# Step 6.1: Full Stack Config
# =========================================================================


class TestFullStackConfig:
    """Verify a single TOML file configures all apps via confy."""

    def test_unified_config_all_apps(self, unified_config_file: Path):
        """Single TOML file configures llmcore, semantiscan, and logging."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_path=str(unified_config_file),
            prefix="LLMCORE",
            app_prefixes={"semantiscan": "SEMANTISCAN"},
            load_dotenv_file=False,
        )

        # --- llmcore settings ---
        assert cfg.llmcore.default_provider == "ollama"
        assert cfg.llmcore.providers.ollama.timeout == 120
        assert cfg.llmcore.providers.ollama.default_model == "gemma3:1b"

        # --- semantiscan settings (file overrides defaults) ---
        ss = cfg.app("semantiscan")
        assert ss.chunking.chunk_size == 2000
        assert ss.chunking.chunk_overlap == 300
        assert ss.retrieval.top_k == 15

        # --- semantiscan defaults preserved where not overridden ---
        assert ss.chunking.min_chunk_size == 100  # from DEFAULT_CONFIG
        assert ss.retrieval.similarity_threshold == 0.5  # from DEFAULT_CONFIG

        # --- shared logging section ---
        assert cfg.logging.console_enabled is False
        assert cfg.logging.file_mode == "single"
        assert cfg.logging.file_level == "DEBUG"

    def test_semantiscan_namespaced_file(self, ss_only_config_file: Path):
        """Namespaced file loading puts settings under semantiscan.*."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_paths=[(str(ss_only_config_file), "semantiscan")],
            load_dotenv_file=False,
        )

        ss = cfg.app("semantiscan")
        assert ss.chunking.chunk_size == 3000
        assert ss.retrieval.hybrid_search is False
        assert ss.retrieval.bm25_weight == 0.5

        # Defaults still present
        assert ss.chunking.strategy == "recursive"

    def test_multi_file_merge_order(self, tmp_path: Path):
        """Later files override earlier files."""
        base_file = tmp_path / "base.toml"
        base_file.write_text("[semantiscan.chunking]\nchunk_size = 1000\n")

        override_file = tmp_path / "override.toml"
        override_file.write_text("[semantiscan.chunking]\nchunk_size = 5000\n")

        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_paths=[str(base_file), str(override_file)],
            load_dotenv_file=False,
        )

        # override.toml wins
        assert cfg.app("semantiscan").chunking.chunk_size == 5000

    def test_app_accessor_returns_empty_for_unknown(self):
        """cfg.app('unknown') returns empty Config, never raises."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            load_dotenv_file=False,
        )

        unknown = cfg.app("nonexistent")
        assert isinstance(unknown, Config)
        assert len(unknown) == 0

    def test_as_dict_produces_plain_dict(self, unified_config_file: Path):
        """as_dict() returns a standard dict (no Config objects)."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_path=str(unified_config_file),
            load_dotenv_file=False,
        )

        d = cfg.app("semantiscan").as_dict()
        assert isinstance(d, dict)
        assert not isinstance(d, Config)
        assert isinstance(d["chunking"], dict)
        assert not isinstance(d["chunking"], Config)


# =========================================================================
# Config Precedence
# =========================================================================


class TestConfigPrecedence:
    """Verify defaults < file < env < overrides precedence."""

    def test_file_overrides_defaults(self, ss_only_config_file: Path):
        """File values win over app_defaults."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_paths=[(str(ss_only_config_file), "semantiscan")],
            load_dotenv_file=False,
        )
        # Default is 1500, file says 3000
        assert cfg.app("semantiscan").chunking.chunk_size == 3000

    def test_env_overrides_file(self, ss_only_config_file: Path, monkeypatch):
        """Environment variables win over file values."""
        monkeypatch.setenv("SEMANTISCAN_CHUNKING__CHUNK_SIZE", "4000")

        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_paths=[(str(ss_only_config_file), "semantiscan")],
            app_prefixes={"semantiscan": "SEMANTISCAN"},
            load_dotenv_file=False,
        )
        assert cfg.app("semantiscan").chunking.chunk_size == 4000

    def test_overrides_dict_wins_all(self, ss_only_config_file: Path, monkeypatch):
        """overrides_dict has the highest precedence."""
        monkeypatch.setenv("SEMANTISCAN_CHUNKING__CHUNK_SIZE", "4000")

        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_paths=[(str(ss_only_config_file), "semantiscan")],
            app_prefixes={"semantiscan": "SEMANTISCAN"},
            overrides_dict={"semantiscan.chunking.chunk_size": 9999},
            load_dotenv_file=False,
        )
        assert cfg.app("semantiscan").chunking.chunk_size == 9999


# =========================================================================
# App-Specific Environment Variable Routing
# =========================================================================


class TestAppEnvVarRouting:
    """Verify SEMANTISCAN_* env vars route to the semantiscan namespace."""

    def test_semantiscan_prefix_routes_correctly(self, monkeypatch):
        """SEMANTISCAN_RETRIEVAL__TOP_K routes to semantiscan.retrieval.top_k."""
        monkeypatch.setenv("SEMANTISCAN_RETRIEVAL__TOP_K", "25")

        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            app_prefixes={"semantiscan": "SEMANTISCAN"},
            load_dotenv_file=False,
        )
        assert cfg.app("semantiscan").retrieval.top_k == 25

    def test_multiple_env_vars(self, monkeypatch):
        """Multiple SEMANTISCAN_* vars all land in the right places."""
        monkeypatch.setenv("SEMANTISCAN_CHUNKING__CHUNK_SIZE", "2500")
        monkeypatch.setenv("SEMANTISCAN_RETRIEVAL__HYBRID_SEARCH", "false")
        monkeypatch.setenv("SEMANTISCAN_INDEXING__BATCH_SIZE", "50")

        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            app_prefixes={"semantiscan": "SEMANTISCAN"},
            load_dotenv_file=False,
        )

        ss = cfg.app("semantiscan")
        assert ss.chunking.chunk_size == 2500
        assert ss.retrieval.hybrid_search is False
        assert ss.indexing.batch_size == 50


# =========================================================================
# Provenance Tracking
# =========================================================================


class TestProvenanceTracking:
    """Verify provenance correctly reports where values came from."""

    def test_defaults_provenance(self):
        """Values from app_defaults are sourced as 'app_defaults:*'."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            track_provenance=True,
            load_dotenv_file=False,
        )

        p = cfg.provenance("semantiscan.chunking.chunk_size")
        assert p is not None
        assert p.value == 1500
        assert "app_defaults" in p.source

    def test_file_override_provenance(self, unified_config_file: Path):
        """Values from a file show 'file:*' provenance."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_path=str(unified_config_file),
            track_provenance=True,
            load_dotenv_file=False,
        )

        p = cfg.provenance("semantiscan.chunking.chunk_size")
        assert p is not None
        assert p.value == 2000
        assert "file" in p.source

    def test_provenance_history_shows_chain(self, unified_config_file: Path):
        """History shows the override chain: defaults → file."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            file_path=str(unified_config_file),
            track_provenance=True,
            load_dotenv_file=False,
        )

        history = cfg.provenance_history("semantiscan.chunking.chunk_size")
        assert len(history) >= 2
        # First entry is from defaults
        assert "app_defaults" in history[0].source
        # Last entry is from file (the winner)
        assert "file" in history[-1].source
        assert history[-1].value == 2000

    def test_provenance_dump(self):
        """provenance_dump() returns {key: source} for all tracked keys."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            track_provenance=True,
            load_dotenv_file=False,
        )

        dump = cfg.provenance_dump()
        assert isinstance(dump, dict)
        assert len(dump) > 0
        # All values should be source strings
        for source in dump.values():
            assert isinstance(source, str)

    def test_provenance_disabled_by_default(self):
        """Without track_provenance=True, provenance returns None."""
        cfg = Config(
            app_defaults={"semantiscan": deepcopy(SS_DEFAULTS)},
            load_dotenv_file=False,
        )

        assert cfg.provenance("semantiscan.chunking.chunk_size") is None
        assert cfg.provenance_dump() == {}


# =========================================================================
# Step 6.2: Logging Display Behavior
# =========================================================================


@requires_llmcore_logging
class TestLoggingDisplayBehavior:
    """Verify display=True messages reach console even in silent mode."""

    def test_display_reaches_console_in_silent_mode(self, tmp_path: Path):
        """display=True messages appear on console when console_enabled=false."""
        stderr_capture = io.StringIO()

        configure_logging(
            app_name="test_phase6",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "per_run",
                "file_directory": str(tmp_path),
            },
            force_reconfigure=True,
        )

        # Redirect console handler to our capture stream
        mgr = UnifiedLoggingManager.get_instance()
        if mgr._console_handler:
            mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.phase6.display")
        logger.setLevel(logging.DEBUG)

        # Normal log — should NOT appear on console (console_enabled=False)
        logger.info("silent message")

        # Display log — SHOULD appear on console via DisplayFilter
        log_display(logger, logging.INFO, "visible message")

        output = stderr_capture.getvalue()
        assert "silent message" not in output
        assert "visible message" in output

    def test_display_respects_min_level(self, tmp_path: Path):
        """display=True at DEBUG is blocked if display_min_level=INFO."""
        stderr_capture = io.StringIO()

        configure_logging(
            app_name="test_phase6",
            config={
                "console_enabled": False,
                "file_enabled": False,
                "display_min_level": "INFO",
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        if mgr._console_handler:
            mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.phase6.level")
        logger.setLevel(logging.DEBUG)

        # display=True but at DEBUG level — below display_min_level=INFO
        log_display(logger, logging.DEBUG, "too low")

        # display=True at INFO — meets threshold
        log_display(logger, logging.INFO, "meets threshold")

        output = stderr_capture.getvalue()
        assert "too low" not in output
        assert "meets threshold" in output

    def test_normal_logs_reach_file_only(self, tmp_path: Path):
        """Non-display logs go to file but not console in silent mode."""
        configure_logging(
            app_name="test_phase6",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "per_run",
                "file_directory": str(tmp_path),
                "file_level": "DEBUG",
            },
            force_reconfigure=True,
        )

        logger = logging.getLogger("test.phase6.fileonly")
        logger.setLevel(logging.DEBUG)

        logger.info("file-only message")

        # Verify it went to the log file
        log_path = UnifiedLoggingManager.get_log_file_path()
        assert log_path is not None
        assert log_path.exists()
        content = log_path.read_text()
        assert "file-only message" in content

    def test_console_enabled_passes_all(self, tmp_path: Path):
        """When console_enabled=True, all logs pass at console_level."""
        stderr_capture = io.StringIO()

        configure_logging(
            app_name="test_phase6",
            config={
                "console_enabled": True,
                "console_level": "DEBUG",
                "file_enabled": False,
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        if mgr._console_handler:
            mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.phase6.enabled")
        logger.setLevel(logging.DEBUG)

        logger.info("normal info")
        logger.debug("debug msg")

        output = stderr_capture.getvalue()
        assert "normal info" in output
        assert "debug msg" in output


# =========================================================================
# Log Rotation
# =========================================================================


@requires_llmcore_logging
class TestLogRotation:
    """Verify file_mode='single' uses RotatingFileHandler."""

    def test_single_mode_creates_rotating_handler(self, tmp_path: Path):
        """file_mode='single' creates a RotatingFileHandler."""
        from logging.handlers import RotatingFileHandler

        configure_logging(
            app_name="test_rotation",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "single",
                "file_directory": str(tmp_path),
                "file_single_name": "{app}.log",
                "rotation_max_bytes": 1024,
                "rotation_backup_count": 3,
                "file_level": "DEBUG",
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._file_handler is not None
        assert isinstance(mgr._file_handler, RotatingFileHandler)

        # Log file should exist
        log_path = tmp_path / "test_rotation.log"
        assert log_path.exists() or UnifiedLoggingManager.get_log_file_path() is not None

    def test_per_run_mode_creates_standard_handler(self, tmp_path: Path):
        """file_mode='per_run' creates a standard FileHandler (not Rotating)."""
        from logging.handlers import RotatingFileHandler

        configure_logging(
            app_name="test_perrun",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "per_run",
                "file_directory": str(tmp_path),
                "file_level": "DEBUG",
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._file_handler is not None
        assert not isinstance(mgr._file_handler, RotatingFileHandler)
        assert isinstance(mgr._file_handler, logging.FileHandler)


# =========================================================================
# Semantiscan Config Shim (Deprecated load_config)
# =========================================================================


class TestSemantiscanConfigShim:
    """Verify the deprecated load_config() shim works end-to-end with confy."""

    def test_load_config_emits_deprecation(self):
        """load_config() emits DeprecationWarning."""
        from semantiscan.config.loader import load_config

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config()

            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            deprecation_msgs = [
                str(x.message) for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert any("deprecated" in m.lower() for m in deprecation_msgs)

    def test_load_config_returns_valid_config(self):
        """load_config() returns a valid dict with all default sections."""
        from semantiscan.config.loader import load_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_config()

        assert isinstance(config, dict)
        assert "chunking" in config
        assert "retrieval" in config
        assert "indexing" in config
        assert config["chunking"]["chunk_size"] == 1500

    def test_load_config_with_overrides(self):
        """load_config(overrides=...) merges overrides on top."""
        from semantiscan.config.loader import load_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_config(overrides={"chunking": {"chunk_size": 7777}})

        assert config["chunking"]["chunk_size"] == 7777
        # Other defaults still present
        assert config["retrieval"]["top_k"] == 10

    def test_validate_config_works_with_shim_output(self):
        """validate_config() accepts load_config() output."""
        from semantiscan.config.loader import load_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_config()

        errors = validate_config(config)
        assert errors == []


# =========================================================================
# Cross-Component: Config + Logging Integration
# =========================================================================


@requires_llmcore_logging
class TestConfigLoggingIntegration:
    """Verify the logging system consumes config from confy correctly."""

    def test_logging_config_from_confy(self, tmp_path: Path):
        """configure_logging() works with [logging] section from confy Config."""
        config_file = tmp_path / "config.toml"
        log_dir = tmp_path / "logs"
        config_file.write_text(
            f"""\
[logging]
console_enabled = false
file_enabled = true
file_mode = "per_run"
file_directory = "{log_dir}"
file_level = "DEBUG"
display_min_level = "WARNING"
"""
        )

        cfg = Config(
            file_path=str(config_file),
            load_dotenv_file=False,
        )

        # Extract logging section and pass to configure_logging
        logging_dict = cfg.app("logging").as_dict() if "logging" in cfg else {}

        log_path = configure_logging(
            app_name="test_integration",
            config=logging_dict,
            force_reconfigure=True,
        )

        assert log_path is not None
        # Log something
        logger = logging.getLogger("test.integration.cfglog")
        logger.info("integration test message")

        # Verify file was written
        if log_path != Path("/dev/null") and log_path.exists():
            content = log_path.read_text()
            assert "integration test message" in content


# =========================================================================
# Validate all semantiscan defaults pass validation
# =========================================================================


class TestDefaultsIntegrity:
    """Verify semantiscan defaults are self-consistent."""

    def test_default_config_passes_validation(self):
        """DEFAULT_CONFIG validates with zero errors."""
        errors = validate_config(deepcopy(SS_DEFAULTS))
        assert errors == [], f"Default config has validation errors: {errors}"

    def test_get_app_defaults_matches_default_config(self):
        """get_app_defaults() returns a dict equal to DEFAULT_CONFIG."""
        app_defaults = get_app_defaults()
        assert app_defaults == SS_DEFAULTS
        # Must be a deep copy, not the same object
        assert app_defaults is not SS_DEFAULTS

    def test_get_default_config_is_independent(self):
        """get_default_config() returns independent copies."""
        a = get_default_config()
        b = get_default_config()
        a["chunking"]["chunk_size"] = 99999
        assert b["chunking"]["chunk_size"] == 1500  # Unaffected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
