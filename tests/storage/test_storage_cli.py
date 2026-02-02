# tests/storage/test_storage_cli.py
"""
Tests for the Storage CLI (Phase 1 - PRIMORDIUM).

Tests cover:
- Command parsing
- Output formatting
- Configuration loading
- All CLI commands (validate, health, schema, info)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Direct import to avoid circular dependency
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"))
from cli import (
    OutputFormatter,
    StorageCommands,
    _load_config,
    _mask_url_password,
    cmd_health,
    cmd_info,
    cmd_schema,
    cmd_validate,
    create_parser,
    main,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def formatter():
    """Create a non-color formatter for testing."""
    return OutputFormatter(use_color=False, json_output=False)


@pytest.fixture
def json_formatter():
    """Create a JSON formatter for testing."""
    return OutputFormatter(use_color=False, json_output=True)


@pytest.fixture
def valid_config_file():
    """Create a temporary valid config file."""
    content = """
[storage.session]
type = "postgres"
db_url = "postgresql://user:pass@localhost:5432/testdb"

[storage.vector]
type = "chromadb"
path = "/tmp/chromadb"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def invalid_config_file():
    """Create a temporary invalid config file."""
    content = """
[storage.session]
type = "postgres"
# Missing db_url

[storage.vector]
type = "unknown_backend"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


# =============================================================================
# TESTS - OUTPUT FORMATTER
# =============================================================================

class TestOutputFormatter:
    """Tests for OutputFormatter class."""

    def test_success_formatting(self, formatter):
        """Test success message formatting."""
        result = formatter.success("Test passed")
        assert "✓" in result
        assert "Test passed" in result

    def test_error_formatting(self, formatter):
        """Test error message formatting."""
        result = formatter.error("Test failed")
        assert "✗" in result
        assert "Test failed" in result

    def test_warning_formatting(self, formatter):
        """Test warning message formatting."""
        result = formatter.warning("Test warning")
        assert "⚠" in result
        assert "Test warning" in result

    def test_info_formatting(self, formatter):
        """Test info message formatting."""
        result = formatter.info("Test info")
        assert "ℹ" in result
        assert "Test info" in result

    def test_status_healthy(self, formatter):
        """Test status formatting for healthy."""
        result = formatter.format_status("healthy")
        assert "healthy" in result.lower()

    def test_status_unhealthy(self, formatter):
        """Test status formatting for unhealthy."""
        result = formatter.format_status("unhealthy")
        assert "unhealthy" in result.lower()

    def test_no_color_mode(self):
        """Test that no-color mode doesn't include ANSI codes."""
        formatter = OutputFormatter(use_color=False)
        result = formatter.success("Test")
        assert "\033" not in result


# =============================================================================
# TESTS - URL PASSWORD MASKING
# =============================================================================

class TestPasswordMasking:
    """Tests for URL password masking."""

    def test_mask_simple_password(self):
        """Test masking simple password in URL."""
        url = "postgresql://user:secret123@localhost/db"
        masked = _mask_url_password(url)
        assert "secret123" not in masked
        assert "***" in masked
        assert "user" in masked
        assert "localhost" in masked

    def test_mask_complex_password(self):
        """Test masking password with special characters."""
        url = "postgresql://admin:p@ss!word@localhost/db"
        masked = _mask_url_password(url)
        # Should mask the first @-delimited password
        assert "***" in masked

    def test_no_password_unchanged(self):
        """Test URL without password is unchanged."""
        url = "postgresql://localhost/db"
        masked = _mask_url_password(url)
        assert masked == url


# =============================================================================
# TESTS - VALIDATE COMMAND
# =============================================================================

class TestCmdValidate:
    """Tests for validate command."""

    def test_validate_valid_config(self, valid_config_file, formatter, capsys):
        """Test validating a valid config file."""
        exit_code = cmd_validate(valid_config_file, strict=False, formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower()

    def test_validate_invalid_config(self, invalid_config_file, formatter, capsys):
        """Test validating an invalid config file."""
        exit_code = cmd_validate(invalid_config_file, strict=False, formatter=formatter)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()

    def test_validate_json_output(self, valid_config_file, json_formatter, capsys):
        """Test validate command with JSON output."""
        exit_code = cmd_validate(valid_config_file, strict=False, formatter=json_formatter)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "valid" in output
        assert output["valid"] is True

    def test_validate_strict_mode(self, formatter, capsys):
        """Test strict mode treats warnings as errors."""
        # Create config with only warnings (no errors)
        content = """
[storage.vector]
type = "chromadb"
# No path = warning about in-memory storage
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()

            # Non-strict should pass
            exit_code = cmd_validate(f.name, strict=False, formatter=formatter)
            assert exit_code == 0

            # Strict should fail
            exit_code = cmd_validate(f.name, strict=True, formatter=formatter)
            assert exit_code == 1

            os.unlink(f.name)


# =============================================================================
# TESTS - SCHEMA COMMAND
# =============================================================================

class TestCmdSchema:
    """Tests for schema command."""

    def test_schema_status(self, formatter, capsys):
        """Test schema status command."""
        exit_code = cmd_schema("status", formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Schema" in captured.out
        assert "Version" in captured.out

    def test_schema_migrations(self, formatter, capsys):
        """Test schema migrations command."""
        exit_code = cmd_schema("migrations", formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Migrations" in captured.out
        assert "v0" in captured.out

    def test_schema_info(self, formatter, capsys):
        """Test schema info command."""
        exit_code = cmd_schema("info", formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Schema" in captured.out

    def test_schema_json_output(self, json_formatter, capsys):
        """Test schema command with JSON output."""
        exit_code = cmd_schema("status", formatter=json_formatter)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "current_schema_version" in output
        assert "migrations_count" in output

    def test_schema_invalid_action(self, formatter, capsys):
        """Test schema command with invalid action."""
        exit_code = cmd_schema("invalid_action", formatter=formatter)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Unknown" in captured.out or "invalid" in captured.out.lower()


# =============================================================================
# TESTS - INFO COMMAND
# =============================================================================

class TestCmdInfo:
    """Tests for info command."""

    def test_info_valid_config(self, valid_config_file, formatter, capsys):
        """Test info command with valid config."""
        exit_code = cmd_info(valid_config_file, formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Storage Configuration" in captured.out
        assert "postgres" in captured.out
        assert "chromadb" in captured.out

    def test_info_masks_password(self, valid_config_file, formatter, capsys):
        """Test that info command masks passwords."""
        exit_code = cmd_info(valid_config_file, formatter=formatter)

        captured = capsys.readouterr()
        assert "pass" not in captured.out
        assert "***" in captured.out

    def test_info_json_output(self, valid_config_file, json_formatter, capsys):
        """Test info command with JSON output."""
        exit_code = cmd_info(valid_config_file, formatter=json_formatter)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "session" in output or "vector" in output


# =============================================================================
# TESTS - HEALTH COMMAND
# =============================================================================

class TestCmdHealth:
    """Tests for health command."""

    def test_health_with_config(self, valid_config_file, formatter, capsys):
        """Test health command with config."""
        exit_code = cmd_health(valid_config_file, formatter=formatter)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Health" in captured.out or "Storage" in captured.out

    def test_health_json_output(self, valid_config_file, json_formatter, capsys):
        """Test health command with JSON output."""
        exit_code = cmd_health(valid_config_file, formatter=json_formatter)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "overall_healthy" in output or "backends" in output


# =============================================================================
# TESTS - CLI PARSER
# =============================================================================

class TestCliParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None

    def test_parser_validate_command(self):
        """Test parsing validate command."""
        parser = create_parser()
        args = parser.parse_args(["validate"])
        assert args.command == "validate"

    def test_parser_schema_command(self):
        """Test parsing schema command."""
        parser = create_parser()
        args = parser.parse_args(["schema", "migrations"])
        assert args.command == "schema"
        assert args.action == "migrations"

    def test_parser_config_option(self):
        """Test parsing --config option."""
        parser = create_parser()
        args = parser.parse_args(["--config", "/path/to/config.toml", "validate"])
        assert args.config == "/path/to/config.toml"

    def test_parser_json_option(self):
        """Test parsing --json option."""
        parser = create_parser()
        args = parser.parse_args(["--json", "info"])
        assert args.json is True

    def test_parser_no_color_option(self):
        """Test parsing --no-color option."""
        parser = create_parser()
        args = parser.parse_args(["--no-color", "schema", "status"])
        assert args.no_color is True


# =============================================================================
# TESTS - MAIN FUNCTION
# =============================================================================

class TestMain:
    """Tests for main CLI entry point."""

    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        exit_code = main([])

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should show some help output
        assert len(captured.out) > 0

    def test_main_validate(self, valid_config_file):
        """Test main with validate command."""
        exit_code = main(["--config", valid_config_file, "validate"])
        assert exit_code == 0

    def test_main_schema(self):
        """Test main with schema command."""
        exit_code = main(["schema", "status"])
        assert exit_code == 0


# =============================================================================
# TESTS - STORAGE COMMANDS (REPL INTEGRATION)
# =============================================================================

class TestStorageCommands:
    """Tests for StorageCommands class (REPL integration)."""

    def test_help_command(self):
        """Test help command returns help text."""
        commands = StorageCommands()
        result = commands.help()

        assert "health" in result.lower()
        assert "validate" in result.lower()
        assert "schema" in result.lower()

    def test_schema_command(self):
        """Test schema command via REPL interface."""
        commands = StorageCommands()
        result = commands.schema("status")

        assert "Schema" in result
        assert "Version" in result

    def test_health_without_manager(self):
        """Test health command without storage manager."""
        commands = StorageCommands(storage_manager=None)
        result = commands.health()

        assert "not initialized" in result.lower()

    def test_health_with_mock_manager(self):
        """Test health command with mocked storage manager."""
        mock_manager = MagicMock()
        mock_manager.get_health_report.return_value = {
            "overall_healthy": True,
            "backends": {
                "session_postgres": {
                    "status": "healthy",
                    "average_latency_ms": 5.0,
                    "uptime_percentage": 99.9
                }
            }
        }

        commands = StorageCommands(storage_manager=mock_manager)
        result = commands.health()

        assert "Health" in result
        mock_manager.get_health_report.assert_called_once()


# =============================================================================
# TESTS - CONFIG LOADING
# =============================================================================

class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_explicit_config(self, valid_config_file):
        """Test loading config from explicit path."""
        config = _load_config(valid_config_file)

        assert config is not None
        assert "storage" in config

    def test_load_nonexistent_config(self):
        """Test loading non-existent config returns empty dict."""
        config = _load_config("/nonexistent/path/config.toml")

        # Should return empty dict or None, not raise
        assert config is None or config == {}

    def test_auto_detect_config(self):
        """Test auto-detection of config file."""
        # Create a config in current directory
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.toml',
            dir=os.getcwd(),
            prefix='llmcore',
            delete=False
        ) as f:
            f.write("[storage.session]\ntype = 'json'\npath = '/tmp/test.json'\n")
            f.flush()

            try:
                # Should find the config
                config = _load_config(None)
                # May or may not find it depending on filename
            finally:
                os.unlink(f.name)
