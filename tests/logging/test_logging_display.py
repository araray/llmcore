# tests/logging/test_logging_display.py
# tests/test_logging_display.py
"""
Tests for Phase 3 logging enhancements: DisplayFilter, log_display(),
RotatingFileHandler support, and console gating behavior.

These tests verify:
- DisplayFilter behavior matrix (verbose vs silent, display flag, min level)
- log_display() convenience function (extra merging, record attributes)
- File rotation mode selection (_create_file_handler dispatching)
- End-to-end console gating with display=True records in silent mode
- Default config updates (new keys present and correct)
"""

import io
import logging
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from llmcore.logging_config import (
    DEFAULT_LOGGING_CONFIG,
    DisplayFilter,
    UnifiedLoggingManager,
    configure_logging,
    log_display,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_logging_manager():
    """Reset the logging manager singleton between tests."""
    UnifiedLoggingManager._instance = None
    UnifiedLoggingManager._configured = False
    UnifiedLoggingManager._log_file_path = None
    UnifiedLoggingManager._console_handler = None
    UnifiedLoggingManager._file_handler = None
    UnifiedLoggingManager._display_filter = None

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    yield

    UnifiedLoggingManager._instance = None
    UnifiedLoggingManager._configured = False
    UnifiedLoggingManager._log_file_path = None
    UnifiedLoggingManager._console_handler = None
    UnifiedLoggingManager._file_handler = None
    UnifiedLoggingManager._display_filter = None

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


def _make_record(
    level: int = logging.INFO,
    msg: str = "test message",
    display: bool | None = None,
) -> logging.LogRecord:
    """Helper to create a LogRecord with optional display attribute."""
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if display is not None:
        record.display = display
    return record


# ===========================================================================
# DisplayFilter unit tests
# ===========================================================================


class TestDisplayFilter:
    """Unit tests for the DisplayFilter class."""

    # --- Silent mode (console_globally_enabled=False) ---

    def test_display_true_passes_in_silent_mode(self):
        """display=True record at or above min level passes in silent mode."""
        f = DisplayFilter(console_globally_enabled=False)
        record = _make_record(level=logging.INFO, display=True)
        assert f.filter(record) is True

    def test_no_display_blocked_in_silent_mode(self):
        """Record without display flag is blocked in silent mode."""
        f = DisplayFilter(console_globally_enabled=False)
        record = _make_record(level=logging.INFO)
        assert f.filter(record) is False

    def test_display_false_blocked_in_silent_mode(self):
        """Record with explicit display=False is blocked in silent mode."""
        f = DisplayFilter(console_globally_enabled=False)
        record = _make_record(level=logging.INFO, display=False)
        assert f.filter(record) is False

    def test_display_below_min_level_blocked(self):
        """display=True record below display_min_level is blocked."""
        f = DisplayFilter(
            console_globally_enabled=False,
            display_min_level=logging.WARNING,
        )
        record = _make_record(level=logging.INFO, display=True)
        assert f.filter(record) is False  # INFO < WARNING min

    def test_display_at_min_level_passes(self):
        """display=True record exactly at display_min_level passes."""
        f = DisplayFilter(
            console_globally_enabled=False,
            display_min_level=logging.WARNING,
        )
        record = _make_record(level=logging.WARNING, display=True)
        assert f.filter(record) is True

    def test_display_above_min_level_passes(self):
        """display=True record above display_min_level passes."""
        f = DisplayFilter(
            console_globally_enabled=False,
            display_min_level=logging.WARNING,
        )
        record = _make_record(level=logging.ERROR, display=True)
        assert f.filter(record) is True

    # --- Verbose mode (console_globally_enabled=True) ---

    def test_all_pass_in_verbose_mode(self):
        """All records pass in verbose mode regardless of display flag."""
        f = DisplayFilter(console_globally_enabled=True)
        record = _make_record(level=logging.INFO)
        assert f.filter(record) is True

    def test_display_true_passes_in_verbose_mode(self):
        """display=True also passes in verbose mode."""
        f = DisplayFilter(console_globally_enabled=True)
        record = _make_record(level=logging.INFO, display=True)
        assert f.filter(record) is True

    def test_debug_passes_in_verbose_mode(self):
        """Even DEBUG records pass in verbose mode (handler level decides)."""
        f = DisplayFilter(console_globally_enabled=True)
        record = _make_record(level=logging.DEBUG)
        assert f.filter(record) is True

    # --- Edge cases ---

    def test_default_params(self):
        """Default DisplayFilter: console disabled, min level INFO."""
        f = DisplayFilter()
        assert f.console_globally_enabled is False
        assert f.display_min_level == logging.INFO

    def test_display_debug_with_debug_min_level(self):
        """display=True at DEBUG passes when display_min_level is DEBUG."""
        f = DisplayFilter(
            console_globally_enabled=False,
            display_min_level=logging.DEBUG,
        )
        record = _make_record(level=logging.DEBUG, display=True)
        assert f.filter(record) is True


# ===========================================================================
# log_display() unit tests
# ===========================================================================


class TestLogDisplay:
    """Tests for the log_display() convenience function."""

    def test_log_display_adds_extra(self, caplog):
        """log_display sets display=True on the log record."""
        logger = logging.getLogger("test.display")
        with caplog.at_level(logging.DEBUG):
            log_display(logger, logging.INFO, "Test message %s", "arg")
        assert "Test message arg" in caplog.text

    def test_log_display_sets_display_attribute(self, caplog):
        """Verify the display attribute is set on the record."""
        logger = logging.getLogger("test.display.attr")
        with caplog.at_level(logging.DEBUG):
            log_display(logger, logging.INFO, "msg")
        record = caplog.records[-1]
        assert record.display is True

    def test_log_display_preserves_existing_extra(self, caplog):
        """Caller's extra dict is merged, not replaced."""
        logger = logging.getLogger("test.display.merge")
        with caplog.at_level(logging.DEBUG):
            log_display(logger, logging.INFO, "msg", extra={"custom": "val"})
        record = caplog.records[-1]
        assert record.display is True
        assert record.custom == "val"

    def test_log_display_none_extra(self, caplog):
        """Works when extra is not provided."""
        logger = logging.getLogger("test.display.none")
        with caplog.at_level(logging.DEBUG):
            log_display(logger, logging.WARNING, "warning %d", 42)
        assert "warning 42" in caplog.text
        assert caplog.records[-1].display is True

    def test_log_display_respects_level(self, caplog):
        """Record carries the requested level."""
        logger = logging.getLogger("test.display.level")
        with caplog.at_level(logging.DEBUG):
            log_display(logger, logging.ERROR, "error msg")
        assert caplog.records[-1].levelno == logging.ERROR


# ===========================================================================
# File rotation (handler creation) tests
# ===========================================================================


class TestFileRotation:
    """Tests for _create_file_handler dispatching between modes."""

    def test_single_file_mode_creates_rotating_handler(self, temp_log_dir):
        """file_mode='single' creates a RotatingFileHandler."""
        config = {
            "file_enabled": True,
            "file_mode": "single",
            "file_directory": str(temp_log_dir),
            "file_single_name": "test.log",
            "rotation_max_bytes": 1024,
            "rotation_backup_count": 3,
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, path = mgr._create_file_handler(config, "test")
        assert isinstance(handler, RotatingFileHandler)
        assert path == temp_log_dir / "test.log"
        handler.close()

    def test_per_run_mode_creates_standard_handler(self, temp_log_dir):
        """file_mode='per_run' creates a standard FileHandler."""
        config = {
            "file_enabled": True,
            "file_mode": "per_run",
            "file_directory": str(temp_log_dir),
            "file_name_pattern": "{app}_{timestamp:%Y%m%d}.log",
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, path = mgr._create_file_handler(config, "test")
        assert isinstance(handler, logging.FileHandler)
        assert not isinstance(handler, RotatingFileHandler)
        handler.close()

    def test_default_mode_is_per_run(self, temp_log_dir):
        """Missing file_mode defaults to per_run."""
        config = {
            "file_directory": str(temp_log_dir),
            "file_name_pattern": "{app}_{timestamp:%Y%m%d}.log",
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, path = mgr._create_file_handler(config, "test")
        assert isinstance(handler, logging.FileHandler)
        assert not isinstance(handler, RotatingFileHandler)
        handler.close()

    def test_rotating_handler_params(self, temp_log_dir):
        """RotatingFileHandler uses configured max_bytes and backup_count."""
        config = {
            "file_mode": "single",
            "file_directory": str(temp_log_dir),
            "file_single_name": "app.log",
            "rotation_max_bytes": 5000,
            "rotation_backup_count": 2,
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, _ = mgr._create_file_handler(config, "test")
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == 5000
        assert handler.backupCount == 2
        handler.close()

    def test_single_name_format(self, temp_log_dir):
        """file_single_name supports {app} placeholder."""
        config = {
            "file_mode": "single",
            "file_directory": str(temp_log_dir),
            "file_single_name": "{app}_persistent.log",
            "rotation_max_bytes": 1024,
            "rotation_backup_count": 1,
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, path = mgr._create_file_handler(config, "myapp")
        assert path == temp_log_dir / "myapp_persistent.log"
        handler.close()

    def test_invalid_directory_returns_none(self, tmp_path):
        """Unwritable directory returns (None, None)."""
        # Create a read-only directory so mkdir inside it fails
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        import os

        os.chmod(readonly_dir, 0o444)

        config = {
            "file_directory": str(readonly_dir / "logs"),
            "file_mode": "per_run",
            "file_name_pattern": "{app}.log",
            "file_level": "DEBUG",
            "file_format": "%(message)s",
        }
        mgr = UnifiedLoggingManager.__new__(UnifiedLoggingManager)
        handler, path = mgr._create_file_handler(config, "test")

        # Restore permissions for cleanup
        os.chmod(readonly_dir, 0o755)

        # On root, mkdir may succeed anyway — skip assertion if so
        if handler is not None:
            handler.close()
            pytest.skip("Running as root; cannot create unwritable directories")
        assert handler is None
        assert path is None


# ===========================================================================
# Console gating integration tests
# ===========================================================================


class TestConsoleGating:
    """Integration tests verifying the full console gating flow."""

    def test_console_handler_always_created(self, reset_logging_manager, temp_log_dir):
        """Console handler exists even when console_enabled=False."""
        config = {
            "console_enabled": False,
            "file_directory": str(temp_log_dir),
        }
        configure_logging(config=config)
        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._console_handler is not None

    def test_console_handler_has_display_filter(self, reset_logging_manager, temp_log_dir):
        """Console handler has DisplayFilter attached."""
        config = {
            "console_enabled": False,
            "file_directory": str(temp_log_dir),
        }
        configure_logging(config=config)
        mgr = UnifiedLoggingManager.get_instance()
        filters = mgr._console_handler.filters
        assert any(isinstance(f, DisplayFilter) for f in filters)

    def test_display_filter_silent_when_console_disabled(self, reset_logging_manager, temp_log_dir):
        """DisplayFilter is in silent mode when console_enabled=False."""
        config = {
            "console_enabled": False,
            "file_directory": str(temp_log_dir),
        }
        configure_logging(config=config)
        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._display_filter is not None
        assert mgr._display_filter.console_globally_enabled is False

    def test_display_filter_verbose_when_console_enabled(self, reset_logging_manager, temp_log_dir):
        """DisplayFilter is in verbose mode when console_enabled=True."""
        config = {
            "console_enabled": True,
            "file_directory": str(temp_log_dir),
        }
        configure_logging(config=config)
        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._display_filter is not None
        assert mgr._display_filter.console_globally_enabled is True

    def test_display_reaches_console_in_silent_mode(self, reset_logging_manager, temp_log_dir):
        """display=True messages appear on console even when console_enabled=false."""
        # Configure in silent mode
        configure_logging(
            app_name="test",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "per_run",
                "file_directory": str(temp_log_dir),
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._console_handler is not None

        # Redirect console handler's stream to capture output
        stderr_capture = io.StringIO()
        mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.display.e2e")
        logger.setLevel(logging.DEBUG)

        # Normal log - should NOT appear on console (blocked by DisplayFilter)
        logger.info("silent message")

        # Display log - SHOULD appear on console
        log_display(logger, logging.INFO, "visible message")

        output = stderr_capture.getvalue()
        assert "silent message" not in output
        assert "visible message" in output

    def test_all_messages_reach_console_in_verbose_mode(self, reset_logging_manager, temp_log_dir):
        """All messages at or above handler level appear when console_enabled=True."""
        configure_logging(
            app_name="test",
            config={
                "console_enabled": True,
                "console_level": "INFO",
                "file_enabled": True,
                "file_directory": str(temp_log_dir),
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        stderr_capture = io.StringIO()
        mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.verbose.e2e")
        logger.setLevel(logging.DEBUG)

        logger.info("normal info message")
        logger.warning("normal warning message")

        output = stderr_capture.getvalue()
        assert "normal info message" in output
        assert "normal warning message" in output

    def test_display_min_level_configurable(self, reset_logging_manager, temp_log_dir):
        """display_min_level from config is honoured."""
        configure_logging(
            app_name="test",
            config={
                "console_enabled": False,
                "display_min_level": "WARNING",
                "file_enabled": True,
                "file_directory": str(temp_log_dir),
            },
            force_reconfigure=True,
        )

        mgr = UnifiedLoggingManager.get_instance()
        stderr_capture = io.StringIO()
        mgr._console_handler.stream = stderr_capture

        logger = logging.getLogger("test.minlevel.e2e")
        logger.setLevel(logging.DEBUG)

        # INFO display=True should be blocked (below WARNING min)
        log_display(logger, logging.INFO, "info display")

        # WARNING display=True should pass
        log_display(logger, logging.WARNING, "warning display")

        output = stderr_capture.getvalue()
        assert "info display" not in output
        assert "warning display" in output


# ===========================================================================
# Default config key tests
# ===========================================================================


class TestDefaultConfigKeys:
    """Verify all new keys are present in DEFAULT_LOGGING_CONFIG."""

    def test_console_enabled_default_false(self):
        assert DEFAULT_LOGGING_CONFIG["console_enabled"] is False

    def test_file_mode_default(self):
        assert DEFAULT_LOGGING_CONFIG["file_mode"] == "per_run"

    def test_file_single_name_default(self):
        assert DEFAULT_LOGGING_CONFIG["file_single_name"] == "{app}.log"

    def test_rotation_max_bytes_default(self):
        assert DEFAULT_LOGGING_CONFIG["rotation_max_bytes"] == 10 * 1024 * 1024

    def test_rotation_backup_count_default(self):
        assert DEFAULT_LOGGING_CONFIG["rotation_backup_count"] == 5

    def test_display_min_level_default(self):
        assert DEFAULT_LOGGING_CONFIG["display_min_level"] == "INFO"

    def test_all_original_keys_preserved(self):
        """All keys from the original config are still present."""
        expected_keys = {
            "console_enabled",
            "console_level",
            "console_format",
            "file_enabled",
            "file_level",
            "file_directory",
            "file_name_pattern",
            "file_format",
            "components",
        }
        assert expected_keys.issubset(set(DEFAULT_LOGGING_CONFIG.keys()))


# ===========================================================================
# Enable/disable console with display filter
# ===========================================================================


class TestConsoleToggleWithFilter:
    """Tests for enable/disable with the new display filter behavior."""

    def test_enable_console_upgrades_filter(self, reset_logging_manager, temp_log_dir):
        """enable_console_logging upgrades display filter to globally enabled."""
        configure_logging(config={"console_enabled": False, "file_directory": str(temp_log_dir)})

        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._display_filter.console_globally_enabled is False

        from llmcore.logging_config import enable_console_logging

        enable_console_logging("DEBUG")

        assert mgr._display_filter.console_globally_enabled is True

    def test_disable_then_enable_console(self, reset_logging_manager, temp_log_dir):
        """Full disable → enable cycle works correctly."""
        configure_logging(config={"console_enabled": True, "file_directory": str(temp_log_dir)})

        mgr = UnifiedLoggingManager.get_instance()
        assert mgr._console_handler is not None

        from llmcore.logging_config import disable_console_logging, enable_console_logging

        disable_console_logging()
        assert mgr._console_handler is None

        enable_console_logging("WARNING")
        assert mgr._console_handler is not None
        assert mgr._display_filter.console_globally_enabled is True


# ===========================================================================
# File rotation integration
# ===========================================================================


class TestFileRotationIntegration:
    """Integration tests for file rotation through configure_logging."""

    def test_configure_with_single_mode(self, reset_logging_manager, temp_log_dir):
        """configure_logging with file_mode='single' creates RotatingFileHandler."""
        configure_logging(
            app_name="rottest",
            config={
                "console_enabled": False,
                "file_enabled": True,
                "file_mode": "single",
                "file_directory": str(temp_log_dir),
                "file_single_name": "rottest.log",
                "rotation_max_bytes": 2048,
                "rotation_backup_count": 2,
            },
        )

        mgr = UnifiedLoggingManager.get_instance()
        assert isinstance(mgr._file_handler, RotatingFileHandler)
        assert mgr._log_file_path == temp_log_dir / "rottest.log"

    def test_configure_default_mode_is_per_run(self, reset_logging_manager, temp_log_dir):
        """Default config uses per-run FileHandler, not RotatingFileHandler."""
        configure_logging(
            config={
                "console_enabled": False,
                "file_directory": str(temp_log_dir),
            },
        )

        mgr = UnifiedLoggingManager.get_instance()
        assert isinstance(mgr._file_handler, logging.FileHandler)
        assert not isinstance(mgr._file_handler, RotatingFileHandler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
