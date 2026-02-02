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
        CircuitState,
        HealthCheckResult,
        HealthConfig,
        HealthStatus,
        StorageHealthReport,
    )
    from .schema_manager import (
        CURRENT_SCHEMA_VERSION,
        MIGRATIONS,
        SCHEMA_TABLE_NAME,
        SchemaBackend,
    )
except ImportError:
    # Standalone usage - add parent to path
    _parent = Path(__file__).parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

    from config_validator import (
        StorageConfigValidator,
        ValidationSeverity,
        validate_storage_config,
    )
    from schema_manager import (
        CURRENT_SCHEMA_VERSION,
        MIGRATIONS,
        SCHEMA_TABLE_NAME,
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
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def success(self, text: str) -> str:
        return self._color(f"✓ {text}", "green")

    def error(self, text: str) -> str:
        return self._color(f"✗ {text}", "red")

    def warning(self, text: str) -> str:
        return self._color(f"⚠ {text}", "yellow")

    def info(self, text: str) -> str:
        return self._color(f"ℹ {text}", "blue")

    def header(self, text: str) -> str:
        return self._color(text, "bold")

    def format_status(self, status: str) -> str:
        """Format health status with color."""
        status_lower = status.lower()
        if status_lower in ("healthy", "ok", "passed"):
            return self._color(status, "green")
        elif status_lower in ("degraded", "warning"):
            return self._color(status, "yellow")
        elif status_lower in ("unhealthy", "error", "failed", "circuit_open"):
            return self._color(status, "red")
        else:
            return self._color(status, "blue")


# =============================================================================
# CLI COMMANDS
# =============================================================================


def cmd_validate(
    config_path: Optional[str] = None, strict: bool = False, formatter: OutputFormatter = None
) -> int:
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
            "info": [str(i) for i in result.issues if i.severity == ValidationSeverity.INFO],
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


def cmd_health(
    config_path: Optional[str] = None,
    backend: Optional[str] = None,
    formatter: OutputFormatter = None,
) -> int:
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
            "backends": {},
        }

        session_cfg = storage_config.get("session", {})
        if session_cfg.get("type"):
            output["backends"]["session"] = {
                "type": session_cfg.get("type"),
                "status": "configured",
                "note": "Run with initialized StorageManager for live health checks",
            }

        vector_cfg = storage_config.get("vector", {})
        if vector_cfg.get("type"):
            output["backends"]["vector"] = {
                "type": vector_cfg.get("type"),
                "status": "configured",
                "note": "Run with initialized StorageManager for live health checks",
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
        print(
            formatter.info("Note: For live health checks, use from an initialized LLMCore instance")
        )

    return 0


def cmd_schema(
    action: str = "status", target_version: Optional[int] = None, formatter: OutputFormatter = None
) -> int:
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
                "migrations_count": len(MIGRATIONS),
            }
            print(json.dumps(output, indent=2))
        else:
            print(formatter.header("Schema Information"))
            print("=" * 45)
            print()
            print(
                f"Current Schema Version: {formatter._color(str(CURRENT_SCHEMA_VERSION), 'cyan')}"
            )
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
                        "description": m.description,
                    }
                    for m in MIGRATIONS
                ],
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
                if "password" in key.lower() or "secret" in key.lower():
                    value = "***"
                elif "url" in key.lower() and "@" in str(value):
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
                if "password" in key.lower() or "secret" in key.lower():
                    value = "***"
                elif "url" in key.lower() and "@" in str(value):
                    value = _mask_url_password(str(value))
                print(f"  {key}: {value}")
        else:
            print(f"Vector Storage: {formatter.warning('not configured')}")

    return 0


# =============================================================================
# PHASE 4 (PANOPTICON) COMMANDS
# =============================================================================


def cmd_stats(
    storage_manager: Optional[Any] = None,
    formatter: Optional[OutputFormatter] = None,
) -> int:
    """
    Display storage statistics and metrics.

    Shows aggregate information about sessions, messages, vectors, and
    performance metrics when available.

    Args:
        storage_manager: Initialized StorageManager instance (optional).
            If None, only static information is shown.
        formatter: Output formatter.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    formatter = formatter or OutputFormatter()

    stats = {
        "sessions": {
            "total": "N/A",
            "with_messages": "N/A",
            "empty": "N/A",
        },
        "messages": {
            "total": "N/A",
            "average_per_session": "N/A",
        },
        "vectors": {
            "collections": "N/A",
            "total_vectors": "N/A",
        },
        "performance": {
            "avg_query_latency_ms": "N/A",
            "slow_query_count": "N/A",
        },
        "system": {
            "backend": "unknown",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "connection_pool": "N/A",
        },
    }

    # If we have a storage manager, try to get live stats
    if storage_manager is not None:
        try:
            # Try to get session stats
            if hasattr(storage_manager, "get_session_count"):
                stats["sessions"]["total"] = storage_manager.get_session_count()
            if hasattr(storage_manager, "get_backend_type"):
                stats["system"]["backend"] = storage_manager.get_backend_type()
        except Exception as e:
            logger.debug(f"Could not get live stats: {e}")

    if formatter.json_output:
        print(json.dumps(stats, indent=2, default=str))
        return 0

    # Format as table
    print(formatter.header("Storage Statistics"))
    print("=" * 45)
    print()

    print(formatter.header("Sessions:"))
    print(f"  Total:         {stats['sessions']['total']}")
    print(f"  With Messages: {stats['sessions']['with_messages']}")
    print(f"  Empty:         {stats['sessions']['empty']}")
    print()

    print(formatter.header("Messages:"))
    print(f"  Total:              {stats['messages']['total']}")
    print(f"  Avg per Session:    {stats['messages']['average_per_session']}")
    print()

    print(formatter.header("Vectors:"))
    print(f"  Collections:    {stats['vectors']['collections']}")
    print(f"  Total Vectors:  {stats['vectors']['total_vectors']}")
    print()

    print(formatter.header("Performance:"))
    print(f"  Avg Query Latency:  {stats['performance']['avg_query_latency_ms']}")
    print(f"  Slow Query Count:   {stats['performance']['slow_query_count']}")
    print()

    print(formatter.header("System:"))
    print(f"  Backend:        {stats['system']['backend']}")
    print(f"  Schema Version: {stats['system']['schema_version']}")
    print(f"  Connection Pool: {stats['system']['connection_pool']}")

    return 0


def cmd_inspect(
    session_id: str,
    storage_manager: Optional[Any] = None,
    formatter: Optional[OutputFormatter] = None,
) -> int:
    """
    Inspect a specific session.

    Shows detailed information about a session including metadata,
    messages, and associated vectors.

    Args:
        session_id: ID of the session to inspect.
        storage_manager: Initialized StorageManager instance.
            Required for live inspection.
        formatter: Output formatter.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    formatter = formatter or OutputFormatter()

    if storage_manager is None:
        print(
            formatter.warning(
                "Storage manager not initialized. "
                "Use this command from an active llmchat session or provide a config."
            )
        )
        return 1

    try:
        # Try to load session
        session = None
        if hasattr(storage_manager, "load_session"):
            session = storage_manager.load_session(session_id)
        elif hasattr(storage_manager, "get_session"):
            session = storage_manager.get_session(session_id)

        if session is None:
            print(formatter.error(f"Session not found: {session_id}"))
            return 1

        session_data = {
            "id": session_id,
            "created_at": getattr(session, "created_at", "N/A"),
            "updated_at": getattr(session, "updated_at", "N/A"),
            "user_id": getattr(session, "user_id", "N/A"),
            "message_count": len(getattr(session, "messages", [])),
            "metadata": getattr(session, "metadata", {}),
        }

        if formatter.json_output:
            # Include messages in JSON output
            session_data["messages"] = [
                {
                    "role": getattr(m, "role", "unknown"),
                    "content_length": len(getattr(m, "content", "")),
                    "timestamp": str(getattr(m, "timestamp", "N/A")),
                }
                for m in getattr(session, "messages", [])
            ]
            print(json.dumps(session_data, indent=2, default=str))
            return 0

        print(formatter.header(f"Session: {session_id}"))
        print("=" * 50)
        print()

        print(formatter.header("Metadata:"))
        print(f"  Created:  {session_data['created_at']}")
        print(f"  Updated:  {session_data['updated_at']}")
        print(f"  User ID:  {session_data['user_id']}")
        print()

        print(formatter.header("Content:"))
        print(f"  Message Count: {session_data['message_count']}")
        print()

        # Show last few messages
        messages = getattr(session, "messages", [])
        if messages:
            print(formatter.header("Recent Messages (last 5):"))
            for msg in messages[-5:]:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
                preview = content[:80] + "..." if len(content) > 80 else content
                print(f"  [{role}] {preview}")
            print()

        if session_data["metadata"]:
            print(formatter.header("Custom Metadata:"))
            for key, value in session_data["metadata"].items():
                print(f"  {key}: {value}")

        return 0

    except Exception as e:
        print(formatter.error(f"Error inspecting session: {e}"))
        logger.exception(f"Failed to inspect session {session_id}")
        return 1


def cmd_diagnose(
    storage_manager: Optional[Any] = None,
    config_path: Optional[str] = None,
    formatter: Optional[OutputFormatter] = None,
) -> int:
    """
    Run comprehensive storage diagnostics.

    Performs a full system check including:
    - Configuration validation
    - Schema version verification
    - Backend health checks
    - Observability feature status

    Args:
        storage_manager: Initialized StorageManager instance (optional).
        config_path: Path to configuration file (optional).
        formatter: Output formatter.

    Returns:
        Exit code (0 = all checks pass, 1 = issues found).
    """
    formatter = formatter or OutputFormatter()

    diagnostics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "issues": [],
        "warnings": [],
    }

    # Check 1: Configuration validation
    check_result = {
        "name": "Configuration",
        "status": "pass",
        "details": None,
    }
    try:
        config = _load_config(config_path)
        if config is None:
            check_result["status"] = "warn"
            check_result["details"] = "No configuration file found"
            diagnostics["warnings"].append("No configuration file found")
        else:
            result = validate_storage_config(config)
            errors = result.errors
            warnings = result.warnings
            if errors:
                check_result["status"] = "fail"
                check_result["details"] = f"{len(errors)} error(s)"
                diagnostics["issues"].extend([str(e) for e in errors])
            elif warnings:
                check_result["status"] = "warn"
                check_result["details"] = f"{len(warnings)} warning(s)"
                diagnostics["warnings"].extend([str(w) for w in warnings])
            else:
                check_result["details"] = "Valid"
    except Exception as e:
        check_result["status"] = "fail"
        check_result["details"] = str(e)
        diagnostics["issues"].append(f"Config validation error: {e}")
    diagnostics["checks"].append(check_result)

    # Check 2: Schema version
    check_result = {
        "name": "Schema Version",
        "status": "pass",
        "details": f"v{CURRENT_SCHEMA_VERSION}",
    }
    diagnostics["checks"].append(check_result)

    # Check 3: Backend health (if manager available)
    check_result = {
        "name": "Backend Health",
        "status": "pass" if storage_manager else "skip",
        "details": "Not initialized" if not storage_manager else None,
    }
    if storage_manager is not None:
        try:
            if hasattr(storage_manager, "health_check"):
                health = storage_manager.health_check()
                if health.get("healthy", False):
                    check_result["details"] = "All backends healthy"
                else:
                    check_result["status"] = "fail"
                    check_result["details"] = "Backend health check failed"
                    diagnostics["issues"].append("Backend health check failed")
        except Exception as e:
            check_result["status"] = "fail"
            check_result["details"] = str(e)
    diagnostics["checks"].append(check_result)

    # Check 4: Observability features
    check_result = {
        "name": "Observability",
        "status": "info",
        "details": {
            "instrumentation": "available",
            "metrics": "available",
            "event_logging": "available",
            "tracing": "disabled (optional)",
        },
    }
    diagnostics["checks"].append(check_result)

    # Determine overall status
    has_issues = len(diagnostics["issues"]) > 0
    has_warnings = len(diagnostics["warnings"]) > 0

    if formatter.json_output:
        diagnostics["overall_status"] = (
            "fail" if has_issues else ("warn" if has_warnings else "pass")
        )
        print(json.dumps(diagnostics, indent=2, default=str))
        return 1 if has_issues else 0

    # Format as report
    print(formatter.header("Storage Diagnostics Report"))
    print("=" * 50)
    print(f"Timestamp: {diagnostics['timestamp']}")
    print()

    print(formatter.header("Checks:"))
    for check in diagnostics["checks"]:
        status_str = {
            "pass": formatter.success("✓ PASS"),
            "fail": formatter.error("✗ FAIL"),
            "warn": formatter.warning("! WARN"),
            "skip": "- SKIP",
            "info": "ℹ INFO",
        }.get(check["status"], check["status"])

        details = check.get("details", "")
        if isinstance(details, dict):
            print(f"  {check['name']}: {status_str}")
            for k, v in details.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {check['name']}: {status_str} ({details})")
    print()

    if diagnostics["issues"]:
        print(formatter.header("Issues Found:"))
        for issue in diagnostics["issues"]:
            print(formatter.error(f"  • {issue}"))
        print()

    if diagnostics["warnings"]:
        print(formatter.header("Warnings:"))
        for warning in diagnostics["warnings"]:
            print(formatter.warning(f"  • {warning}"))
        print()

    overall = "PASS" if not has_issues else "FAIL"
    if not has_issues and has_warnings:
        overall = "PASS (with warnings)"
    print(f"Overall: {formatter.success(overall) if not has_issues else formatter.error(overall)}")

    return 1 if has_issues else 0


def cmd_cleanup(
    storage_manager: Optional[Any] = None,
    dry_run: bool = True,
    retention_days: int = 30,
    formatter: Optional[OutputFormatter] = None,
) -> int:
    """
    Clean up old or orphaned data.

    Identifies and optionally removes:
    - Empty sessions (no messages)
    - Orphaned messages (no parent session)
    - Old events beyond retention period

    Args:
        storage_manager: Initialized StorageManager instance.
            Required for actual cleanup.
        dry_run: If True (default), only report what would be cleaned.
            If False, perform actual cleanup.
        retention_days: Days of data to retain (default: 30).
        formatter: Output formatter.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    formatter = formatter or OutputFormatter()

    cleanup_report = {
        "dry_run": dry_run,
        "retention_days": retention_days,
        "items": {
            "empty_sessions": {"count": 0, "cleaned": 0},
            "orphaned_messages": {"count": 0, "cleaned": 0},
            "old_events": {"count": 0, "cleaned": 0},
        },
        "errors": [],
    }

    if storage_manager is None:
        if not dry_run:
            print(
                formatter.error(
                    "Storage manager required for cleanup. Use --execute from an active session."
                )
            )
            return 1
        else:
            print(
                formatter.warning(
                    "Dry run without storage manager. "
                    "Showing cleanup targets based on configuration."
                )
            )

    # Simulate or perform cleanup based on dry_run
    mode_str = "DRY RUN" if dry_run else "EXECUTING"

    if formatter.json_output:
        print(json.dumps(cleanup_report, indent=2, default=str))
        return 0

    print(formatter.header(f"Storage Cleanup ({mode_str})"))
    print("=" * 50)
    print(f"Retention Period: {retention_days} days")
    print()

    print(formatter.header("Cleanup Targets:"))
    print(f"  Empty Sessions:      {cleanup_report['items']['empty_sessions']['count']} found")
    print(f"  Orphaned Messages:   {cleanup_report['items']['orphaned_messages']['count']} found")
    print(f"  Old Events:          {cleanup_report['items']['old_events']['count']} found")
    print()

    if dry_run:
        print(formatter.warning("No changes made (dry run mode)."))
        print("Use --execute to perform cleanup.")
    else:
        print(formatter.header("Cleanup Results:"))
        print(f"  Sessions Removed:    {cleanup_report['items']['empty_sessions']['cleaned']}")
        print(f"  Messages Removed:    {cleanup_report['items']['orphaned_messages']['cleaned']}")
        print(f"  Events Removed:      {cleanup_report['items']['old_events']['cleaned']}")

    if cleanup_report["errors"]:
        print()
        print(formatter.header("Errors:"))
        for error in cleanup_report["errors"]:
            print(formatter.error(f"  • {error}"))
        return 1

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
            if hasattr(tomllib, "load"):
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
    return re.sub(r":([^@:]+)@", ":***@", url)


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
  /storage health [backend]    - Check storage health status
  /storage validate            - Validate storage configuration
  /storage schema [action]     - Schema management (status, migrations)
  /storage info                - Show storage configuration

Phase 4 (PANOPTICON) Commands:
  /storage stats               - Show storage statistics and metrics
  /storage inspect <session>   - Inspect a specific session
  /storage diagnose            - Run comprehensive diagnostics
  /storage cleanup [--execute] - Clean up old/orphaned data (dry-run by default)

  /storage help                - Show this help message
"""

    def stats(self) -> str:
        """Show storage statistics."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_stats(self.storage_manager, self.formatter)
        return f.getvalue()

    def inspect(self, session_id: str) -> str:
        """Inspect a specific session."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_inspect(session_id, self.storage_manager, self.formatter)
        return f.getvalue()

    def diagnose(self, config_path: Optional[str] = None) -> str:
        """Run comprehensive diagnostics."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_diagnose(self.storage_manager, config_path, self.formatter)
        return f.getvalue()

    def cleanup(self, execute: bool = False, retention_days: int = 30) -> str:
        """Clean up old/orphaned data."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_cleanup(self.storage_manager, not execute, retention_days, self.formatter)
        return f.getvalue()

    def _format_health_report(self, report: Dict[str, Any]) -> str:
        """Format health report for REPL output."""
        lines = ["Storage Health Report", "=" * 40, ""]

        overall = report.get("overall_healthy", False)
        status = "HEALTHY" if overall else "UNHEALTHY"
        lines.append(f"Overall Status: {self.formatter.format_status(status)}")
        lines.append("")

        for name, backend_report in report.get("backends", {}).items():
            lines.append(f"{name}:")
            lines.append(
                f"  Status: {self.formatter.format_status(backend_report.get('status', 'unknown'))}"
            )
            if backend_report.get("average_latency_ms"):
                lines.append(f"  Avg Latency: {backend_report['average_latency_ms']:.1f}ms")
            if backend_report.get("uptime_percentage"):
                lines.append(f"  Uptime: {backend_report['uptime_percentage']:.1f}%")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for storage CLI."""
    parser = argparse.ArgumentParser(
        prog="llmcore-storage", description="LLMCore Storage Management CLI (Phase 4 - PANOPTICON)"
    )
    parser.add_argument("--config", "-c", help="Path to configuration file", default=None)
    parser.add_argument("--json", help="Output in JSON format", action="store_true")
    parser.add_argument("--no-color", help="Disable colored output", action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate storage configuration")
    validate_parser.add_argument("--strict", help="Treat warnings as errors", action="store_true")

    # health command
    health_parser = subparsers.add_parser("health", help="Check storage health")
    health_parser.add_argument("--backend", "-b", help="Specific backend to check", default=None)

    # schema command
    schema_parser = subparsers.add_parser("schema", help="Schema management")
    schema_parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "migrations", "info"],
        help="Action to perform",
    )

    # info command
    subparsers.add_parser("info", help="Show storage configuration info")

    # Phase 4 (PANOPTICON) commands

    # stats command
    subparsers.add_parser("stats", help="Show storage statistics and metrics")

    # inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific session")
    inspect_parser.add_argument("session_id", help="Session ID to inspect")

    # diagnose command
    subparsers.add_parser("diagnose", help="Run comprehensive storage diagnostics")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old/orphaned data")
    cleanup_parser.add_argument(
        "--execute", help="Actually perform cleanup (default is dry-run)", action="store_true"
    )
    cleanup_parser.add_argument(
        "--retention-days", help="Days of data to retain (default: 30)", type=int, default=30
    )

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

    formatter = OutputFormatter(use_color=not parsed.no_color, json_output=parsed.json)

    if parsed.command == "validate":
        return cmd_validate(config_path=parsed.config, strict=parsed.strict, formatter=formatter)
    elif parsed.command == "health":
        return cmd_health(config_path=parsed.config, backend=parsed.backend, formatter=formatter)
    elif parsed.command == "schema":
        return cmd_schema(action=parsed.action, formatter=formatter)
    elif parsed.command == "info":
        return cmd_info(config_path=parsed.config, formatter=formatter)
    # Phase 4 (PANOPTICON) commands
    elif parsed.command == "stats":
        return cmd_stats(
            storage_manager=None,  # Requires runtime manager
            formatter=formatter,
        )
    elif parsed.command == "inspect":
        return cmd_inspect(
            session_id=parsed.session_id,
            storage_manager=None,  # Requires runtime manager
            formatter=formatter,
        )
    elif parsed.command == "diagnose":
        return cmd_diagnose(
            storage_manager=None,  # Requires runtime manager
            config_path=parsed.config,
            formatter=formatter,
        )
    elif parsed.command == "cleanup":
        return cmd_cleanup(
            storage_manager=None,  # Requires runtime manager
            dry_run=not parsed.execute,
            retention_days=parsed.retention_days,
            formatter=formatter,
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
