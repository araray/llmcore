# src/llmcore/storage/cli.py
"""
Storage CLI Commands for LLMCore.

This module provides command-line interface commands for storage management:
- Health checks and status reporting
- Configuration validation
- Schema management
- Diagnostic information

These commands are available via:
- `llmcore storage <command>` (standalone CLI)
- `/storage <command>` (llmchat REPL integration)

STORAGE SYSTEM V2 (Phase 1 - PRIMORDIUM):
Provides operational visibility and management capabilities.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle imports for both package and standalone usage
try:
    from .config_validator import (
        StorageConfigValidator,
        ValidationResult,
        ValidationSeverity,
        validate_storage_config,
    )
    from .health import (
        HealthStatus,
        CircuitState,
        HealthConfig,
        HealthCheckResult,
        StorageHealthReport,
    )
    from .schema_manager import (
        CURRENT_SCHEMA_VERSION,
        SCHEMA_TABLE_NAME,
        MIGRATIONS,
        SchemaBackend,
    )
except ImportError:
    # Standalone usage - add parent to path
    _parent = Path(__file__).parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

    from config_validator import (
        StorageConfigValidator,
        ValidationResult,
        ValidationSeverity,
        validate_storage_config,
    )
    from health import (
        HealthStatus,
        CircuitState,
        HealthConfig,
        HealthCheckResult,
        StorageHealthReport,
    )
    from schema_manager import (
        CURRENT_SCHEMA_VERSION,
        SCHEMA_TABLE_NAME,
        MIGRATIONS,
        SchemaBackend,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

class OutputFormatter:
    """Formats CLI output in various styles."""

    def __init__(self, use_color: bool = True, json_output: bool = False):
        self.use_color = use_color and sys.stdout.isatty()
        self.json_output = json_output

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        if not self.use_color:
            return text

        colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'bold': '\033[1m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def success(self, text: str) -> str:
        return self._color(f"✓ {text}", 'green')

    def error(self, text: str) -> str:
        return self._color(f"✗ {text}", 'red')

    def warning(self, text: str) -> str:
        return self._color(f"⚠ {text}", 'yellow')

    def info(self, text: str) -> str:
        return self._color(f"ℹ {text}", 'blue')

    def header(self, text: str) -> str:
        return self._color(text, 'bold')

    def format_status(self, status: str) -> str:
        """Format health status with color."""
        status_lower = status.lower()
        if status_lower in ('healthy', 'ok', 'passed'):
            return self._color(status, 'green')
        elif status_lower in ('degraded', 'warning'):
            return self._color(status, 'yellow')
        elif status_lower in ('unhealthy', 'error', 'failed', 'circuit_open'):
            return self._color(status, 'red')
        else:
            return self._color(status, 'blue')


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_validate(config_path: Optional[str] = None, strict: bool = False,
                 formatter: OutputFormatter = None) -> int:
    """
    Validate storage configuration.

    Args:
        config_path: Path to config file (default: auto-detect)
        strict: Treat warnings as errors
        formatter: Output formatter

    Returns:
        Exit code (0 = valid, 1 = invalid)
    """
    formatter = formatter or OutputFormatter()

    # Load configuration
    config = _load_config(config_path)
    if config is None:
        print(formatter.error("Could not load configuration"))
        return 1

    # Validate
    validator = StorageConfigValidator(strict=strict)
    result = validator.validate(config)

    if formatter.json_output:
        output = {
            "valid": result.valid,
            "errors": [str(e) for e in result.errors],
            "warnings": [str(w) for w in result.warnings],
            "info": [str(i) for i in result.issues if i.severity == ValidationSeverity.INFO]
        }
        print(json.dumps(output, indent=2))
    else:
        print(formatter.header("Storage Configuration Validation"))
        print("=" * 45)
        print()

        if result.valid:
            print(formatter.success("Configuration is valid"))
        else:
            print(formatter.error(f"Configuration has {len(result.errors)} error(s)"))

        if result.errors:
            print(f"\n{formatter.header('Errors:')}")
            for err in result.errors:
                print(f"  {formatter.error(str(err))}")

        if result.warnings:
            print(f"\n{formatter.header('Warnings:')}")
            for warn in result.warnings:
                print(f"  {formatter.warning(str(warn))}")

        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        if info_issues:
            print(f"\n{formatter.header('Info:')}")
            for info in info_issues:
                print(f"  {formatter.info(str(info))}")

    return 0 if result.valid else 1


def cmd_health(config_path: Optional[str] = None, backend: Optional[str] = None,
               formatter: OutputFormatter = None) -> int:
    """
    Check storage health status.

    Args:
        config_path: Path to config file
        backend: Specific backend to check (optional)
        formatter: Output formatter

    Returns:
        Exit code (0 = healthy, 1 = unhealthy)
    """
    formatter = formatter or OutputFormatter()

    # This requires an initialized StorageManager
    # For now, provide simulated output based on config validation
    config = _load_config(config_path)
    if config is None:
        print(formatter.error("Could not load configuration"))
        return 1

    storage_config = config.get("storage", {})

    if formatter.json_output:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_healthy": True,
            "backends": {}
        }

        session_cfg = storage_config.get("session", {})
        if session_cfg.get("type"):
            output["backends"]["session"] = {
                "type": session_cfg.get("type"),
                "status": "configured",
                "note": "Run with initialized StorageManager for live health checks"
            }

        vector_cfg = storage_config.get("vector", {})
        if vector_cfg.get("type"):
            output["backends"]["vector"] = {
                "type": vector_cfg.get("type"),
                "status": "configured",
                "note": "Run with initialized StorageManager for live health checks"
            }

        print(json.dumps(output, indent=2))
    else:
        print(formatter.header("Storage Health Status"))
        print("=" * 45)
        print()

        session_cfg = storage_config.get("session", {})
        if session_cfg.get("type"):
            print(f"Session Storage: {formatter._color(session_cfg.get('type'), 'cyan')}")
            print(f"  Status: {formatter.format_status('configured')}")
        else:
            print(f"Session Storage: {formatter.warning('not configured')}")

        print()

        vector_cfg = storage_config.get("vector", {})
        if vector_cfg.get("type"):
            print(f"Vector Storage: {formatter._color(vector_cfg.get('type'), 'cyan')}")
            print(f"  Status: {formatter.format_status('configured')}")
        else:
            print(f"Vector Storage: {formatter.warning('not configured')}")

        print()
        print(formatter.info("Note: For live health checks, use from an initialized LLMCore instance"))

    return 0


def cmd_schema(action: str = "status", target_version: Optional[int] = None,
               formatter: OutputFormatter = None) -> int:
    """
    Manage storage schema.

    Args:
        action: Action to perform (status, migrations, info)
        target_version: Target schema version (for migrate action)
        formatter: Output formatter

    Returns:
        Exit code
    """
    formatter = formatter or OutputFormatter()

    if action == "status" or action == "info":
        if formatter.json_output:
            output = {
                "current_schema_version": CURRENT_SCHEMA_VERSION,
                "schema_table_name": SCHEMA_TABLE_NAME,
                "supported_backends": [b.value for b in SchemaBackend],
                "migrations_count": len(MIGRATIONS)
            }
            print(json.dumps(output, indent=2))
        else:
            print(formatter.header("Schema Information"))
            print("=" * 45)
            print()
            print(f"Current Schema Version: {formatter._color(str(CURRENT_SCHEMA_VERSION), 'cyan')}")
            print(f"Schema Table Name: {SCHEMA_TABLE_NAME}")
            print(f"Supported Backends: {', '.join(b.value for b in SchemaBackend)}")
            print(f"Total Migrations: {len(MIGRATIONS)}")

    elif action == "migrations":
        if formatter.json_output:
            output = {
                "current_version": CURRENT_SCHEMA_VERSION,
                "migrations": [
                    {
                        "from_version": m.from_version,
                        "to_version": m.to_version,
                        "description": m.description
                    }
                    for m in MIGRATIONS
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print(formatter.header("Schema Migrations"))
            print("=" * 45)
            print()
            for m in MIGRATIONS:
                version_str = f"v{m.from_version} → v{m.to_version}"
                print(f"  {formatter._color(version_str, 'cyan')}: {m.description}")

    else:
        print(formatter.error(f"Unknown schema action: {action}"))
        print("Available actions: status, migrations, info")
        return 1

    return 0


def cmd_info(config_path: Optional[str] = None, formatter: OutputFormatter = None) -> int:
    """
    Display storage configuration information.

    Args:
        config_path: Path to config file
        formatter: Output formatter

    Returns:
        Exit code
    """
    formatter = formatter or OutputFormatter()

    config = _load_config(config_path)
    if config is None:
        print(formatter.error("Could not load configuration"))
        return 1

    storage_config = config.get("storage", {})

    if formatter.json_output:
        print(json.dumps(storage_config, indent=2, default=str))
    else:
        print(formatter.header("Storage Configuration"))
        print("=" * 45)
        print()

        session_cfg = storage_config.get("session", {})
        if session_cfg:
            print(formatter.header("Session Storage:"))
            for key, value in session_cfg.items():
                # Mask sensitive values
                if 'password' in key.lower() or 'secret' in key.lower():
                    value = '***'
                elif 'url' in key.lower() and '@' in str(value):
                    # Mask password in URLs
                    value = _mask_url_password(str(value))
                print(f"  {key}: {value}")
        else:
            print(f"Session Storage: {formatter.warning('not configured')}")

        print()

        vector_cfg = storage_config.get("vector", {})
        if vector_cfg:
            print(formatter.header("Vector Storage:"))
            for key, value in vector_cfg.items():
                if 'password' in key.lower() or 'secret' in key.lower():
                    value = '***'
                elif 'url' in key.lower() and '@' in str(value):
                    value = _mask_url_password(str(value))
                print(f"  {key}: {value}")
        else:
            print(f"Vector Storage: {formatter.warning('not configured')}")

    return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load configuration from file.

    Args:
        config_path: Explicit path, or None to auto-detect

    Returns:
        Configuration dictionary, or None if not found
    """
    # Try explicit path first
    if config_path:
        return _load_toml_config(config_path)

    # Auto-detect config locations
    search_paths = [
        Path.cwd() / "llmcore.toml",
        Path.cwd() / "config.toml",
        Path.home() / ".config" / "llmcore" / "config.toml",
        Path.home() / ".llmcore" / "config.toml",
    ]

    # Check LLMCORE_CONFIG env var
    env_path = os.environ.get("LLMCORE_CONFIG")
    if env_path:
        search_paths.insert(0, Path(env_path))

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found config at: {path}")
            return _load_toml_config(str(path))

    # Return empty config as fallback
    logger.warning("No config file found; using empty config")
    return {}


def _load_toml_config(path: str) -> Optional[Dict[str, Any]]:
    """Load a TOML configuration file."""
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            logger.error("Neither tomllib (Python 3.11+) nor toml package available")
            return None

    try:
        with open(path, "rb") as f:
            if hasattr(tomllib, 'load'):
                return tomllib.load(f)
            else:
                # toml package uses text mode
                f.close()
                with open(path, "r") as f2:
                    return tomllib.load(f2)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return None


def _mask_url_password(url: str) -> str:
    """Mask password in a URL string."""
    import re
    # Match :password@ pattern and replace password with ***
    return re.sub(r':([^@:]+)@', ':***@', url)


# =============================================================================
# REPL INTEGRATION
# =============================================================================

class StorageCommands:
    """
    Storage commands for REPL integration.

    This class provides methods that can be called from llmchat's REPL
    with the /storage prefix.

    Usage in llmchat REPL:
        /storage health
        /storage validate
        /storage schema status
        /storage info
    """

    def __init__(self, storage_manager: Optional[Any] = None):
        """
        Initialize storage commands.

        Args:
            storage_manager: Optional StorageManager instance for live checks
        """
        self.storage_manager = storage_manager
        self.formatter = OutputFormatter()

    def health(self, backend: Optional[str] = None) -> str:
        """Check storage health status."""
        if self.storage_manager is not None:
            # Use live health data from manager
            report = self.storage_manager.get_health_report(backend)
            return self._format_health_report(report)
        else:
            return "Storage manager not initialized. Run /storage health after initialization."

    def validate(self, config_path: Optional[str] = None, strict: bool = False) -> str:
        """Validate storage configuration."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_validate(config_path, strict, self.formatter)
        return f.getvalue()

    def schema(self, action: str = "status") -> str:
        """Show schema information."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_schema(action, formatter=self.formatter)
        return f.getvalue()

    def info(self, config_path: Optional[str] = None) -> str:
        """Show storage configuration info."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_info(config_path, self.formatter)
        return f.getvalue()

    def help(self) -> str:
        """Show available storage commands."""
        return """
Storage Commands:
  /storage health [backend]  - Check storage health status
  /storage validate          - Validate storage configuration
  /storage schema [action]   - Schema management (status, migrations)
  /storage info              - Show storage configuration
  /storage help              - Show this help message
"""

    def _format_health_report(self, report: Dict[str, Any]) -> str:
        """Format health report for REPL output."""
        lines = ["Storage Health Report", "=" * 40, ""]

        overall = report.get("overall_healthy", False)
        status = "HEALTHY" if overall else "UNHEALTHY"
        lines.append(f"Overall Status: {self.formatter.format_status(status)}")
        lines.append("")

        for name, backend_report in report.get("backends", {}).items():
            lines.append(f"{name}:")
            lines.append(f"  Status: {self.formatter.format_status(backend_report.get('status', 'unknown'))}")
            if backend_report.get('average_latency_ms'):
                lines.append(f"  Avg Latency: {backend_report['average_latency_ms']:.1f}ms")
            if backend_report.get('uptime_percentage'):
                lines.append(f"  Uptime: {backend_report['uptime_percentage']:.1f}%")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for storage CLI."""
    parser = argparse.ArgumentParser(
        prog="llmcore-storage",
        description="LLMCore Storage Management CLI"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--json",
        help="Output in JSON format",
        action="store_true"
    )
    parser.add_argument(
        "--no-color",
        help="Disable colored output",
        action="store_true"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate storage configuration")
    validate_parser.add_argument(
        "--strict",
        help="Treat warnings as errors",
        action="store_true"
    )

    # health command
    health_parser = subparsers.add_parser("health", help="Check storage health")
    health_parser.add_argument(
        "--backend", "-b",
        help="Specific backend to check",
        default=None
    )

    # schema command
    schema_parser = subparsers.add_parser("schema", help="Schema management")
    schema_parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "migrations", "info"],
        help="Action to perform"
    )

    # info command
    subparsers.add_parser("info", help="Show storage configuration info")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for storage CLI.

    Args:
        args: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    formatter = OutputFormatter(
        use_color=not parsed.no_color,
        json_output=parsed.json
    )

    if parsed.command == "validate":
        return cmd_validate(
            config_path=parsed.config,
            strict=parsed.strict,
            formatter=formatter
        )
    elif parsed.command == "health":
        return cmd_health(
            config_path=parsed.config,
            backend=parsed.backend,
            formatter=formatter
        )
    elif parsed.command == "schema":
        return cmd_schema(
            action=parsed.action,
            formatter=formatter
        )
    elif parsed.command == "info":
        return cmd_info(
            config_path=parsed.config,
            formatter=formatter
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
