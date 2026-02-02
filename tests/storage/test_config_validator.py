# tests/storage/test_config_validator.py
"""
Tests for the Storage Configuration Validator (Phase 1 - PRIMORDIUM).

Tests cover:
- Required field validation
- PostgreSQL connection URL validation
- File path validation
- Environment variable resolution
- Cross-backend consistency checks
- Validation severity levels
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

# Add storage module to path for direct imports (avoids llmcore import chain issues)
_storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
if str(_storage_path) not in sys.path:
    sys.path.insert(0, str(_storage_path))

from config_validator import (
    StorageConfigValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    _resolve_env_vars,
    _validate_path,
    _validate_postgres_url,
    validate_storage_config,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def valid_postgres_config() -> Dict[str, Any]:
    """Valid PostgreSQL storage configuration."""
    return {
        "storage": {
            "session": {
                "type": "postgres",
                "db_url": "postgresql://user:pass@localhost:5432/llmcore",
                "min_pool_size": 2,
                "max_pool_size": 10,
            },
            "vector": {
                "type": "pgvector",
                "db_url": "postgresql://user:pass@localhost:5432/llmcore",
                "default_vector_dimension": 1536,
            },
        }
    }


@pytest.fixture
def valid_sqlite_config() -> Dict[str, Any]:
    """Valid SQLite storage configuration."""
    return {
        "storage": {
            "session": {"type": "sqlite", "path": "/tmp/llmcore_test.db"},
            "vector": {"type": "chromadb", "path": "/tmp/llmcore_chroma"},
        }
    }


@pytest.fixture
def empty_config() -> Dict[str, Any]:
    """Empty configuration."""
    return {}


# =============================================================================
# TESTS - ENVIRONMENT VARIABLE RESOLUTION
# =============================================================================


class TestEnvVarResolution:
    """Tests for environment variable resolution."""

    def test_resolve_simple_env_var(self):
        """Test resolving ${VAR_NAME} syntax."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _resolve_env_vars("prefix_${TEST_VAR}_suffix")
            assert result == "prefix_test_value_suffix"

    def test_resolve_env_var_with_default(self):
        """Test resolving ${VAR_NAME:-default} syntax."""
        # Variable exists
        with patch.dict(os.environ, {"TEST_VAR": "real_value"}):
            result = _resolve_env_vars("${TEST_VAR:-default}")
            assert result == "real_value"

        # Variable doesn't exist, use default
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_env_vars("${NONEXISTENT:-default_value}")
            assert result == "default_value"

    def test_resolve_nested_dict(self):
        """Test resolving env vars in nested dictionaries."""
        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432"}):
            config = {"database": {"host": "${DB_HOST}", "port": "${DB_PORT}"}}
            result = _resolve_env_vars(config)

            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == "5432"

    def test_resolve_list(self):
        """Test resolving env vars in lists."""
        with patch.dict(os.environ, {"ITEM1": "a", "ITEM2": "b"}):
            items = ["${ITEM1}", "${ITEM2}", "literal"]
            result = _resolve_env_vars(items)

            assert result == ["a", "b", "literal"]

    def test_no_env_var(self):
        """Test that strings without env vars pass through unchanged."""
        result = _resolve_env_vars("plain string")
        assert result == "plain string"

    def test_non_string_passthrough(self):
        """Test that non-string values pass through unchanged."""
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(3.14) == 3.14
        assert _resolve_env_vars(True) is True
        assert _resolve_env_vars(None) is None


# =============================================================================
# TESTS - POSTGRES URL VALIDATION
# =============================================================================


class TestPostgresUrlValidation:
    """Tests for PostgreSQL connection URL validation."""

    def test_valid_postgresql_url(self):
        """Test valid postgresql:// URL."""
        is_valid, error = _validate_postgres_url("postgresql://user:pass@localhost:5432/db")
        assert is_valid is True
        assert error is None

    def test_valid_postgres_url(self):
        """Test valid postgres:// URL."""
        is_valid, error = _validate_postgres_url("postgres://user:pass@localhost:5432/db")
        assert is_valid is True

    def test_valid_asyncpg_url(self):
        """Test valid postgresql+asyncpg:// URL."""
        is_valid, error = _validate_postgres_url("postgresql+asyncpg://user:pass@localhost/db")
        assert is_valid is True

    def test_invalid_scheme(self):
        """Test URL with invalid scheme."""
        is_valid, error = _validate_postgres_url("mysql://user:pass@localhost/db")
        assert is_valid is False
        assert "Invalid scheme" in error

    def test_missing_hostname(self):
        """Test URL without hostname."""
        is_valid, error = _validate_postgres_url("postgresql:///db")
        assert is_valid is False
        assert "hostname" in error.lower()

    def test_empty_url(self):
        """Test empty URL."""
        is_valid, error = _validate_postgres_url("")
        assert is_valid is False
        assert "empty" in error.lower()


# =============================================================================
# TESTS - PATH VALIDATION
# =============================================================================


class TestPathValidation:
    """Tests for filesystem path validation."""

    def test_valid_path(self, tmp_path):
        """Test valid path with writable parent."""
        file_path = str(tmp_path / "test.db")
        is_valid, error = _validate_path(file_path)
        assert is_valid is True
        assert error is None

    def test_expanduser_path(self):
        """Test that ~ is expanded."""
        # This should not fail even if directory doesn't exist
        is_valid, error = _validate_path("~/llmcore/test.db", must_exist=False)
        assert is_valid is True

    def test_empty_path(self):
        """Test empty path."""
        is_valid, error = _validate_path("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_nonexistent_required_path(self):
        """Test path that must exist but doesn't."""
        is_valid, error = _validate_path("/nonexistent/path/file.db", must_exist=True)
        assert is_valid is False
        assert "does not exist" in error.lower()


# =============================================================================
# TESTS - VALIDATION RESULT
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert len(result.issues) == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        """Test that adding an error makes result invalid."""
        result = ValidationResult(valid=True)
        result.add_issue(
            ValidationIssue(severity=ValidationSeverity.ERROR, field="test", message="Test error")
        )

        assert result.valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Test that warnings don't make result invalid."""
        result = ValidationResult(valid=True)
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING, field="test", message="Test warning"
            )
        )

        assert result.valid is True
        assert len(result.warnings) == 1

    def test_format_report(self):
        """Test report formatting."""
        result = ValidationResult(valid=True)
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="db_url",
                message="Missing required field",
                suggestion="Add storage.session.db_url",
            )
        )
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="path",
                message="Path may not be writable",
            )
        )

        report = result.format_report()

        assert "Error" in report
        assert "Warning" in report
        assert "db_url" in report
        assert "Missing required field" in report


# =============================================================================
# TESTS - STORAGE CONFIG VALIDATOR
# =============================================================================


class TestStorageConfigValidator:
    """Tests for StorageConfigValidator class."""

    def test_valid_postgres_config(self, valid_postgres_config):
        """Test validation of valid PostgreSQL config."""
        validator = StorageConfigValidator()
        result = validator.validate(valid_postgres_config)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_sqlite_config(self, valid_sqlite_config):
        """Test validation of valid SQLite config."""
        validator = StorageConfigValidator()
        result = validator.validate(valid_sqlite_config)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_empty_config_warns(self, empty_config):
        """Test that empty config generates warnings."""
        validator = StorageConfigValidator()
        result = validator.validate(empty_config)

        # Should be valid but with warnings
        assert result.valid is True
        assert len(result.warnings) > 0

    def test_invalid_session_type(self):
        """Test detection of invalid session storage type."""
        config = {
            "storage": {
                "session": {
                    "type": "mongodb"  # Invalid
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("Invalid session storage type" in str(e) for e in result.errors)

    def test_invalid_vector_type(self):
        """Test detection of invalid vector storage type."""
        config = {
            "storage": {
                "vector": {
                    "type": "pinecone"  # Invalid
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("Invalid vector storage type" in str(e) for e in result.errors)

    def test_postgres_missing_db_url(self):
        """Test detection of missing db_url for PostgreSQL."""
        config = {
            "storage": {
                "session": {
                    "type": "postgres"
                    # Missing db_url
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("db_url" in str(e) for e in result.errors)

    def test_sqlite_missing_path(self):
        """Test detection of missing path for SQLite."""
        config = {
            "storage": {
                "session": {
                    "type": "sqlite"
                    # Missing path
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("path" in str(e) for e in result.errors)

    def test_invalid_pool_size(self):
        """Test detection of invalid pool size configuration."""
        config = {
            "storage": {
                "session": {
                    "type": "postgres",
                    "db_url": "postgresql://localhost/db",
                    "min_pool_size": 10,
                    "max_pool_size": 5,  # min > max
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("min_pool_size" in str(e) for e in result.errors)

    def test_invalid_vector_dimension(self):
        """Test detection of invalid vector dimension."""
        config = {
            "storage": {
                "vector": {
                    "type": "pgvector",
                    "db_url": "postgresql://localhost/db",
                    "default_vector_dimension": -100,  # Invalid
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is False
        assert any("dimension" in str(e).lower() for e in result.errors)

    def test_high_vector_dimension_warning(self):
        """Test warning for very high vector dimension."""
        config = {
            "storage": {
                "vector": {
                    "type": "pgvector",
                    "db_url": "postgresql://localhost/db",
                    "default_vector_dimension": 8192,  # Very high
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is True  # Valid but with warning
        assert any("dimension" in str(w).lower() for w in result.warnings)

    def test_chromadb_no_path_warning(self):
        """Test warning when ChromaDB has no path (in-memory)."""
        config = {
            "storage": {
                "vector": {
                    "type": "chromadb"
                    # No path = in-memory
                }
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        assert result.valid is True
        assert any("in-memory" in str(w).lower() for w in result.warnings)

    def test_same_database_recommendation(self, valid_postgres_config):
        """Test info message when session and vector use same DB."""
        validator = StorageConfigValidator()
        result = validator.validate(valid_postgres_config)

        # Should have info about shared database
        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert any("same" in str(i).lower() and "database" in str(i).lower() for i in info_issues)

    def test_different_database_info(self):
        """Test info message when session and vector use different DBs."""
        config = {
            "storage": {
                "session": {"type": "postgres", "db_url": "postgresql://localhost/sessions"},
                "vector": {
                    "type": "pgvector",
                    "db_url": "postgresql://localhost/vectors",  # Different
                },
            }
        }

        validator = StorageConfigValidator()
        result = validator.validate(config)

        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        assert any("different" in str(i).lower() for i in info_issues)

    def test_strict_mode(self):
        """Test that strict mode treats warnings as errors."""
        config = {
            "storage": {
                "vector": {
                    "type": "chromadb"
                    # No path - normally just a warning
                }
            }
        }

        validator = StorageConfigValidator(strict=True)
        result = validator.validate(config)

        # In strict mode, the warning becomes an error
        assert result.valid is False

    def test_env_var_in_config(self):
        """Test that environment variables are resolved."""
        with patch.dict(os.environ, {"TEST_DB_URL": "postgresql://localhost/test"}):
            config = {"storage": {"session": {"type": "postgres", "db_url": "${TEST_DB_URL}"}}}

            validator = StorageConfigValidator()
            result = validator.validate(config)

            # Should be valid after env var resolution
            assert result.valid is True

    def test_env_var_fallback(self):
        """Test db_url fallback from environment variable."""
        with patch.dict(
            os.environ, {"LLMCORE_STORAGE_SESSION_DB_URL": "postgresql://localhost/fallback"}
        ):
            config = {
                "storage": {
                    "session": {
                        "type": "postgres"
                        # No db_url in config, should fall back to env var
                    }
                }
            }

            validator = StorageConfigValidator()
            result = validator.validate(config)

            # Should find the env var and be valid
            assert result.valid is True
            info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
            assert any("environment variable" in str(i).lower() for i in info_issues)


# =============================================================================
# TESTS - CONVENIENCE FUNCTION
# =============================================================================


class TestValidateStorageConfig:
    """Tests for the validate_storage_config convenience function."""

    def test_basic_usage(self, valid_sqlite_config):
        """Test basic usage of convenience function."""
        result = validate_storage_config(valid_sqlite_config)

        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_strict_parameter(self):
        """Test strict parameter is passed through."""
        config = {
            "storage": {
                "vector": {"type": "chromadb"}  # Warning: no path
            }
        }

        # Non-strict should be valid
        result = validate_storage_config(config, strict=False)
        assert result.valid is True

        # Strict should fail
        result = validate_storage_config(config, strict=True)
        assert result.valid is False


# =============================================================================
# TESTS - VALIDATION ISSUE
# =============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue class."""

    def test_str_without_suggestion(self):
        """Test string representation without suggestion."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR, field="db_url", message="Missing required field"
        )

        result = str(issue)

        assert "[ERROR]" in result
        assert "db_url" in result
        assert "Missing required field" in result

    def test_str_with_suggestion(self):
        """Test string representation with suggestion."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="db_url",
            message="Missing required field",
            suggestion="Add storage.session.db_url to config",
        )

        result = str(issue)

        assert "Suggestion" in result
        assert "Add storage.session.db_url" in result

    def test_severity_levels(self):
        """Test all severity levels."""
        for severity in ValidationSeverity:
            issue = ValidationIssue(severity=severity, field="test", message="Test")
            result = str(issue)
            assert severity.value.upper() in result
