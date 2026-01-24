# tests/test_logging_config.py
"""
Comprehensive tests for the llmcore.logging_config module.

Tests the UnifiedLoggingManager singleton, console/file handlers,
runtime configuration, and component log levels.
"""

import logging
import tempfile
from pathlib import Path

import pytest

from llmcore.logging_config import (
    DEFAULT_LOGGING_CONFIG,
    UnifiedLoggingManager,
    configure_logging,
    get_log_file_path,
    set_console_level,
    set_file_level,
    set_component_level,
    disable_console_logging,
    enable_console_logging,
)


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

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


class TestDefaultLoggingConfig:
    """Tests for default logging configuration."""

    def test_console_enabled_by_default(self):
        """Test console logging is enabled by default."""
        assert DEFAULT_LOGGING_CONFIG["console_enabled"] is True

    def test_file_enabled_by_default(self):
        """Test file logging is enabled by default."""
        assert DEFAULT_LOGGING_CONFIG["file_enabled"] is True

    def test_console_level_is_warning(self):
        """Test default console level is WARNING."""
        assert DEFAULT_LOGGING_CONFIG["console_level"] == "WARNING"

    def test_file_level_is_debug(self):
        """Test default file level is DEBUG."""
        assert DEFAULT_LOGGING_CONFIG["file_level"] == "DEBUG"

    def test_components_defined(self):
        """Test component log levels are defined."""
        components = DEFAULT_LOGGING_CONFIG["components"]
        assert "llmchat" in components
        assert "llmcore" in components


class TestUnifiedLoggingManager:
    """Tests for UnifiedLoggingManager singleton."""

    def test_singleton_pattern(self, reset_logging_manager):
        """Test that only one instance is created."""
        manager1 = UnifiedLoggingManager()
        manager2 = UnifiedLoggingManager()
        assert manager1 is manager2

    def test_get_instance(self, reset_logging_manager):
        """Test get_instance returns singleton."""
        instance = UnifiedLoggingManager.get_instance()
        assert instance is not None
        assert instance is UnifiedLoggingManager.get_instance()

    def test_is_configured_initially_false(self, reset_logging_manager):
        """Test is_configured returns False initially."""
        assert UnifiedLoggingManager.is_configured() is False


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_returns_path(self, reset_logging_manager, temp_log_dir):
        """Test configure_logging returns a Path."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": False}
        result = configure_logging(config=config)
        assert isinstance(result, Path)

    def test_console_only(self, reset_logging_manager):
        """Test console-only logging."""
        config = {"file_enabled": False, "console_enabled": True}
        result = configure_logging(config=config)
        assert str(result) == "/dev/null"

    def test_component_levels(self, reset_logging_manager, temp_log_dir):
        """Test component log levels are set."""
        config = {
            "file_directory": str(temp_log_dir),
            "console_enabled": False,
            "components": {"test_component": "ERROR"}
        }
        configure_logging(config=config)
        component_logger = logging.getLogger("test_component")
        assert component_logger.level == logging.ERROR


class TestRuntimeLevelChanges:
    """Tests for runtime log level adjustments."""

    def test_set_console_level_string(self, reset_logging_manager, temp_log_dir):
        """Test setting console level with string."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": True}
        configure_logging(config=config)
        set_console_level("ERROR")
        manager = UnifiedLoggingManager.get_instance()
        if manager._console_handler:
            assert manager._console_handler.level == logging.ERROR

    def test_set_file_level_string(self, reset_logging_manager, temp_log_dir):
        """Test setting file level with string."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": False}
        configure_logging(config=config)
        set_file_level("WARNING")
        manager = UnifiedLoggingManager.get_instance()
        if manager._file_handler:
            assert manager._file_handler.level == logging.WARNING

    def test_set_component_level(self, reset_logging_manager, temp_log_dir):
        """Test setting component level."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": False}
        configure_logging(config=config)
        set_component_level("my_component", "CRITICAL")
        assert logging.getLogger("my_component").level == logging.CRITICAL


class TestConsoleToggle:
    """Tests for enabling/disabling console logging."""

    def test_disable_console(self, reset_logging_manager, temp_log_dir):
        """Test disabling console logging."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": True}
        configure_logging(config=config)
        disable_console_logging()
        manager = UnifiedLoggingManager.get_instance()
        assert manager._console_handler is None

    def test_enable_console(self, reset_logging_manager, temp_log_dir):
        """Test re-enabling console logging."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": False}
        configure_logging(config=config)
        enable_console_logging("DEBUG")
        manager = UnifiedLoggingManager.get_instance()
        assert manager._console_handler is not None


class TestGetLogFilePath:
    """Tests for get_log_file_path function."""

    def test_returns_none_before_configure(self, reset_logging_manager):
        """Test returns None before configuration."""
        result = get_log_file_path()
        assert result is None

    def test_returns_path_after_configure(self, reset_logging_manager, temp_log_dir):
        """Test returns Path after configuration."""
        config = {"file_directory": str(temp_log_dir), "console_enabled": False}
        configure_logging(config=config)
        result = get_log_file_path()
        assert result is not None
        assert isinstance(result, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
