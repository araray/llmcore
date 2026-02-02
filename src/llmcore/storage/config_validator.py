# src/llmcore/storage/config_validator.py
"""
Storage Configuration Validator for LLMCore.

This module provides comprehensive validation of storage configuration at startup,
ensuring that all settings are correct and consistent before attempting to
initialize storage backends.

Key Validations:
- Required fields present for each backend type
- Connection string format validation
- Path existence/writability checks
- Conflicting configuration detection
- Environment variable resolution

Design Philosophy:
- Fail fast with clear error messages
- Validate early (before any connection attempts)
- Provide actionable suggestions for fixing issues
"""

import logging
import os
import pathlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION TYPES
# =============================================================================


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Configuration is invalid, cannot proceed
    WARNING = "warning"  # Configuration may cause issues
    INFO = "info"  # Informational notice


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: ValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        result = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            result += f"\n  → Suggestion: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    resolved_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False

    def format_report(self) -> str:
        """Format validation issues as a human-readable report."""
        if not self.issues:
            return "✓ Storage configuration is valid"

        lines = ["Storage Configuration Validation Report", "=" * 45]

        errors = self.errors
        if errors:
            lines.append(f"\n❌ {len(errors)} Error(s):")
            for issue in errors:
                lines.append(f"  • {issue}")

        warnings = self.warnings
        if warnings:
            lines.append(f"\n⚠ {len(warnings)} Warning(s):")
            for issue in warnings:
                lines.append(f"  • {issue}")

        info_issues = [i for i in self.issues if i.severity == ValidationSeverity.INFO]
        if info_issues:
            lines.append(f"\nℹ {len(info_issues)} Notice(s):")
            for issue in info_issues:
                lines.append(f"  • {issue}")

        return "\n".join(lines)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references in a value.

    Supports:
    - ${VAR_NAME} syntax
    - $VAR_NAME syntax (at start of string)
    - Nested dict/list resolution
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        resolved = re.sub(pattern, replacer, value)

        # Also handle $VAR_NAME at start
        if resolved.startswith("$") and not resolved.startswith("${"):
            var_name = resolved[1:].split()[0]  # Get first word
            env_value = os.environ.get(var_name)
            if env_value:
                resolved = env_value + resolved[1 + len(var_name) :]

        return resolved

    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]

    return value


def _validate_postgres_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate PostgreSQL connection URL format.

    Returns:
        (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        parsed = urlparse(url)

        # Check scheme
        valid_schemes = ["postgresql", "postgres", "postgresql+psycopg", "postgresql+asyncpg"]
        if parsed.scheme not in valid_schemes:
            return False, f"Invalid scheme '{parsed.scheme}'. Expected one of: {valid_schemes}"

        # Check host
        if not parsed.hostname:
            return False, "Missing hostname in connection URL"

        # Check for common mistakes
        if "@" in (parsed.username or "") or "@" in (parsed.password or ""):
            return False, "Username/password contains '@' - ensure it's URL-encoded"

        if parsed.password and parsed.password == "password":
            # Just a warning, not an error
            return True, None

        return True, None

    except Exception as e:
        return False, f"Invalid URL format: {e}"


def _validate_path(
    path: str, must_exist: bool = False, must_be_writable: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate a filesystem path.

    Returns:
        (is_valid, error_message)
    """
    if not path:
        return False, "Path is empty"

    try:
        expanded = pathlib.Path(os.path.expanduser(path))

        if must_exist and not expanded.exists():
            return False, f"Path does not exist: {expanded}"

        # Check parent directory
        parent = expanded.parent
        if must_be_writable:
            if parent.exists() and not os.access(parent, os.W_OK):
                return False, f"Parent directory is not writable: {parent}"

        return True, None

    except Exception as e:
        return False, f"Invalid path: {e}"


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================


class StorageConfigValidator:
    """
    Validates storage configuration for LLMCore.

    Usage:
        validator = StorageConfigValidator()
        result = validator.validate(config)

        if not result.valid:
            logger.error(result.format_report())
            raise ConfigError("Invalid storage configuration")
    """

    # Valid backend types
    VALID_SESSION_TYPES = {"json", "sqlite", "postgres"}
    VALID_VECTOR_TYPES = {"chromadb", "pgvector"}

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete storage configuration.

        Args:
            config: Raw configuration dictionary (typically from TOML)

        Returns:
            ValidationResult with issues and resolved config
        """
        result = ValidationResult(valid=True)

        # Resolve environment variables first
        resolved = _resolve_env_vars(config)

        # Get storage section
        storage_config = resolved.get("storage", {})

        if not storage_config:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="storage",
                    message="No storage configuration found",
                    suggestion="Add [storage.session] and [storage.vector] sections to your config",
                )
            )
            result.resolved_config = resolved
            return result

        # Validate session storage
        session_config = storage_config.get("session", {})
        self._validate_session_storage(session_config, result)

        # Validate vector storage
        vector_config = storage_config.get("vector", {})
        self._validate_vector_storage(vector_config, result)

        # Check for conflicting configurations
        self._validate_cross_backend_consistency(session_config, vector_config, result)

        # Store resolved config
        result.resolved_config = resolved

        # In strict mode, warnings become errors
        if self.strict:
            for issue in result.warnings:
                issue.severity = ValidationSeverity.ERROR
                result.valid = False

        return result

    def _validate_session_storage(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate session storage configuration."""
        session_type = config.get("type", "").lower()

        if not session_type:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="storage.session.type",
                    message="No session storage type specified",
                    suggestion="Set storage.session.type to 'sqlite', 'json', or 'postgres'",
                )
            )
            return

        if session_type not in self.VALID_SESSION_TYPES:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.session.type",
                    message=f"Invalid session storage type: '{session_type}'",
                    suggestion=f"Valid types: {', '.join(sorted(self.VALID_SESSION_TYPES))}",
                )
            )
            return

        # Type-specific validation
        if session_type == "postgres":
            self._validate_postgres_session(config, result)
        elif session_type in ("sqlite", "json"):
            self._validate_file_session(config, session_type, result)

    def _validate_postgres_session(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate PostgreSQL session storage config."""
        db_url = config.get("db_url", "")

        # Check environment variable fallback
        if not db_url:
            db_url = os.environ.get("LLMCORE_STORAGE_SESSION_DB_URL", "")
            if db_url:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        field="storage.session.db_url",
                        message="Using db_url from LLMCORE_STORAGE_SESSION_DB_URL environment variable",
                    )
                )

        if not db_url:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.session.db_url",
                    message="PostgreSQL session storage requires db_url",
                    suggestion="Set storage.session.db_url or LLMCORE_STORAGE_SESSION_DB_URL env var",
                )
            )
            return

        is_valid, error = _validate_postgres_url(db_url)
        if not is_valid:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.session.db_url",
                    message=f"Invalid PostgreSQL URL: {error}",
                    suggestion="Format: postgresql://user:password@host:port/database",
                )
            )

        # Validate pool settings if present
        min_pool = config.get("min_pool_size")
        max_pool = config.get("max_pool_size")

        if min_pool is not None and max_pool is not None:
            if min_pool > max_pool:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field="storage.session.min_pool_size",
                        message=f"min_pool_size ({min_pool}) > max_pool_size ({max_pool})",
                    )
                )

    def _validate_file_session(
        self, config: Dict[str, Any], session_type: str, result: ValidationResult
    ) -> None:
        """Validate file-based session storage config (SQLite/JSON)."""
        path = config.get("path", "")

        if not path:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.session.path",
                    message=f"{session_type.upper()} session storage requires path",
                    suggestion="Set storage.session.path to a file path (e.g., ~/.llmcore/sessions.db)",
                )
            )
            return

        is_valid, error = _validate_path(path, must_exist=False, must_be_writable=True)
        if not is_valid:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR, field="storage.session.path", message=error
                )
            )

    def _validate_vector_storage(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate vector storage configuration."""
        vector_type = config.get("type", "").lower()

        if not vector_type:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="storage.vector.type",
                    message="No vector storage type specified (RAG will be unavailable)",
                    suggestion="Set storage.vector.type to 'chromadb' or 'pgvector'",
                )
            )
            return

        if vector_type not in self.VALID_VECTOR_TYPES:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.vector.type",
                    message=f"Invalid vector storage type: '{vector_type}'",
                    suggestion=f"Valid types: {', '.join(sorted(self.VALID_VECTOR_TYPES))}",
                )
            )
            return

        # Type-specific validation
        if vector_type == "pgvector":
            self._validate_pgvector(config, result)
        elif vector_type == "chromadb":
            self._validate_chromadb(config, result)

    def _validate_pgvector(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate pgvector configuration."""
        db_url = config.get("db_url", "")

        # Check environment variable fallback
        if not db_url:
            db_url = os.environ.get("LLMCORE_STORAGE_VECTOR_DB_URL", "")
            if db_url:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        field="storage.vector.db_url",
                        message="Using db_url from LLMCORE_STORAGE_VECTOR_DB_URL environment variable",
                    )
                )

        if not db_url:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.vector.db_url",
                    message="pgvector storage requires db_url",
                    suggestion="Set storage.vector.db_url or LLMCORE_STORAGE_VECTOR_DB_URL env var",
                )
            )
            return

        is_valid, error = _validate_postgres_url(db_url)
        if not is_valid:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="storage.vector.db_url",
                    message=f"Invalid PostgreSQL URL: {error}",
                )
            )

        # Validate dimension if specified
        dimension = config.get("default_vector_dimension")
        if dimension is not None:
            if not isinstance(dimension, int) or dimension <= 0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field="storage.vector.default_vector_dimension",
                        message=f"Invalid vector dimension: {dimension}",
                        suggestion="Must be a positive integer (common values: 384, 768, 1536)",
                    )
                )
            elif dimension > 4096:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field="storage.vector.default_vector_dimension",
                        message=f"Very high vector dimension ({dimension}) may impact performance",
                    )
                )

    def _validate_chromadb(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate ChromaDB configuration."""
        path = config.get("path", "")

        if not path:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="storage.vector.path",
                    message="No path specified for ChromaDB; using in-memory storage",
                    suggestion="Set storage.vector.path for persistent storage",
                )
            )
            return

        is_valid, error = _validate_path(path, must_exist=False, must_be_writable=True)
        if not is_valid:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR, field="storage.vector.path", message=error
                )
            )

    def _validate_cross_backend_consistency(
        self,
        session_config: Dict[str, Any],
        vector_config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate consistency between session and vector storage configs."""
        session_type = session_config.get("type", "").lower()
        vector_type = vector_config.get("type", "").lower()

        # Check if both are PostgreSQL with same URL (recommended)
        if session_type == "postgres" and vector_type == "pgvector":
            session_url = session_config.get("db_url", "")
            vector_url = vector_config.get("db_url", "")

            if session_url and vector_url and session_url != vector_url:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        field="storage",
                        message="Session and vector storage using different PostgreSQL databases",
                        suggestion="Consider using the same database for simpler management",
                    )
                )
            elif session_url == vector_url:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        field="storage",
                        message="Session and vector storage sharing same PostgreSQL database (recommended)",
                    )
                )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def validate_storage_config(config: Dict[str, Any], strict: bool = False) -> ValidationResult:
    """
    Validate storage configuration.

    Args:
        config: Configuration dictionary
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with issues and resolved config
    """
    validator = StorageConfigValidator(strict=strict)
    return validator.validate(config)
