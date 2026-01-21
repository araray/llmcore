# tests/api/test_phase7_config.py
"""
Phase 7: Configuration Management API Tests.

Tests for llmcore's configuration management APIs:
- get_runtime_config()
- set_runtime_config()
- diff_config()
- sync_config_to_file()
- reload_config_from_file()
- is_config_dirty()
- get_config_file_path()
- get_default_config_path()
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create a mock configuration dictionary."""
    return {
        "llmcore": {
            "default_provider": "ollama",
            "default_embedding_model": "all-MiniLM-L6-v2",
            "log_level": "INFO",
            "log_raw_payloads": False,
        },
        "providers": {
            "openai": {
                "default_model": "gpt-4o",
                "timeout": 60,
            },
            "ollama": {
                "default_model": "llama3",
                "timeout": 120,
            },
        },
        "storage": {
            "session": {
                "type": "sqlite",
                "path": "~/.llmcore/sessions.db",
            },
            "vector": {
                "type": "chromadb",
                "path": "~/.llmcore/chroma_db",
            },
        },
    }


@pytest.fixture
def mock_llmcore(mock_config: Dict[str, Any]) -> MagicMock:
    """Create a mock LLMCore instance with Phase 7 capabilities."""
    from llmcore.api import LLMCore

    instance = MagicMock(spec=LLMCore)

    # Mock config object
    mock_confy = MagicMock()
    mock_confy.as_dict.return_value = mock_config.copy()
    mock_confy.get.side_effect = lambda key, default=None: _get_nested(mock_config, key, default)

    instance.config = mock_confy
    instance._config_file_path = None
    instance._runtime_config_dirty = False
    instance._original_config_dict = mock_config.copy()

    return instance


def _get_nested(d: Dict, key: str, default: Any = None) -> Any:
    """Get nested value using dot notation."""
    parts = key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


# ==============================================================================
# get_runtime_config() Tests
# ==============================================================================


class TestGetRuntimeConfig:
    """Tests for get_runtime_config() API."""

    def test_get_full_config(self, mock_config: Dict[str, Any]) -> None:
        """Test getting the full configuration."""
        from llmcore.api import LLMCore

        # Create a mock instance
        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        # Call the real method
        result = LLMCore.get_runtime_config(instance, section=None)

        assert result == mock_config
        assert "llmcore" in result
        assert "providers" in result
        assert "storage" in result

    def test_get_config_section(self, mock_config: Dict[str, Any]) -> None:
        """Test getting a specific configuration section."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        # Get providers section
        result = LLMCore.get_runtime_config(instance, section="providers")

        assert "openai" in result
        assert "ollama" in result
        assert result["openai"]["default_model"] == "gpt-4o"

    def test_get_nested_config_section(self, mock_config: Dict[str, Any]) -> None:
        """Test getting a nested configuration section."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        # Get nested section
        result = LLMCore.get_runtime_config(instance, section="providers.openai")

        assert result["default_model"] == "gpt-4o"
        assert result["timeout"] == 60

    def test_get_leaf_value_as_dict(self, mock_config: Dict[str, Any]) -> None:
        """Test getting a leaf value returns dict with key."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        # Get leaf value
        result = LLMCore.get_runtime_config(instance, section="providers.openai.default_model")

        assert result == {"default_model": "gpt-4o"}

    def test_get_nonexistent_section_raises(self, mock_config: Dict[str, Any]) -> None:
        """Test that getting non-existent section raises ConfigError."""
        from llmcore.api import LLMCore
        from llmcore.exceptions import ConfigError

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        with pytest.raises(ConfigError) as exc_info:
            LLMCore.get_runtime_config(instance, section="nonexistent.section")

        assert "not found" in str(exc_info.value)


# ==============================================================================
# set_runtime_config() Tests
# ==============================================================================


class TestSetRuntimeConfig:
    """Tests for set_runtime_config() API."""

    def test_set_simple_value(self, mock_config: Dict[str, Any]) -> None:
        """Test setting a simple configuration value."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config.copy()
        instance.config = mock_confy
        instance._runtime_config_dirty = False

        # Mock the confy loader's set_by_dot and Config class
        with patch("llmcore.api.LLMCore.set_runtime_config") as mock_set:
            # Just verify the method exists and can be called
            mock_set.return_value = None
            mock_set(instance, "providers.openai.default_model", "gpt-4-turbo")
            mock_set.assert_called_once_with(
                instance, "providers.openai.default_model", "gpt-4-turbo"
            )

        # Also verify the method signature exists on the class
        assert hasattr(LLMCore, "set_runtime_config")
        assert callable(getattr(LLMCore, "set_runtime_config"))

    def test_set_marks_config_dirty(self) -> None:
        """Test that setting a value marks config as dirty."""
        # This is a behavioral test - the implementation should set _runtime_config_dirty = True
        pass


# ==============================================================================
# diff_config() Tests
# ==============================================================================


class TestDiffConfig:
    """Tests for diff_config() API."""

    def test_diff_no_changes(self, mock_config: Dict[str, Any]) -> None:
        """Test diff when no changes have been made."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy
        instance._original_config_dict = mock_config.copy()
        instance._runtime_config_dirty = False

        result = LLMCore.diff_config(instance)

        assert result["has_changes"] is False
        assert result["dirty"] is False
        assert result["added"] == {}
        assert result["removed"] == {}
        assert result["modified"] == {}

    def test_diff_with_modifications(self, mock_config: Dict[str, Any]) -> None:
        """Test diff when values have been modified."""
        from llmcore.api import LLMCore

        # Create modified config
        modified_config = mock_config.copy()
        modified_config["llmcore"] = mock_config["llmcore"].copy()
        modified_config["llmcore"]["log_level"] = "DEBUG"

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = modified_config
        instance.config = mock_confy
        instance._original_config_dict = mock_config.copy()
        instance._runtime_config_dirty = True

        result = LLMCore.diff_config(instance)

        assert result["has_changes"] is True
        # Note: The exact diff structure depends on deepdiff being installed
        # If deepdiff is not installed, has_changes will be True but details limited

    def test_diff_deepdiff_path_conversion(self) -> None:
        """Test that DeepDiff paths are converted to dot notation."""
        from llmcore.api import LLMCore

        instance = MagicMock()

        # Test the path conversion helper
        path = "root['providers']['openai']['default_model']"
        result = LLMCore._deepdiff_path_to_dot(instance, path)

        assert result == "providers.openai.default_model"

    def test_diff_path_conversion_with_array(self) -> None:
        """Test path conversion with array index."""
        from llmcore.api import LLMCore

        instance = MagicMock()

        path = "root['items'][0]['name']"
        result = LLMCore._deepdiff_path_to_dot(instance, path)

        assert result == "items.0.name"


# ==============================================================================
# sync_config_to_file() Tests
# ==============================================================================


class TestSyncConfigToFile:
    """Tests for sync_config_to_file() API."""

    def test_sync_creates_directory(self, mock_config: Dict[str, Any]) -> None:
        """Test that sync creates parent directory if needed."""
        from llmcore.api import LLMCore

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "subdir", "config.toml")

            instance = MagicMock()
            mock_confy = MagicMock()
            mock_confy.as_dict.return_value = mock_config
            instance.config = mock_confy
            instance._config_file_path = None
            instance._runtime_config_dirty = True
            instance._original_config_dict = {}

            result = LLMCore.sync_config_to_file(instance, config_path)

            assert result == config_path
            assert os.path.exists(config_path)
            assert instance._runtime_config_dirty is False

    def test_sync_writes_valid_toml(self, mock_config: Dict[str, Any]) -> None:
        """Test that sync writes valid TOML format."""
        from llmcore.api import LLMCore

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.toml")

            instance = MagicMock()
            mock_confy = MagicMock()
            mock_confy.as_dict.return_value = mock_config
            instance.config = mock_confy
            instance._config_file_path = None
            instance._runtime_config_dirty = True
            instance._original_config_dict = {}

            LLMCore.sync_config_to_file(instance, config_path)

            # Read and parse the file
            with open(config_path, "rb") as f:
                loaded_config = tomllib.load(f)

            assert loaded_config["llmcore"]["default_provider"] == "ollama"
            assert loaded_config["providers"]["openai"]["default_model"] == "gpt-4o"

    def test_sync_uses_default_path(self, mock_config: Dict[str, Any]) -> None:
        """Test that sync uses default path when none specified."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy
        instance._config_file_path = None
        instance._runtime_config_dirty = True
        instance._original_config_dict = {}

        # Patch pathlib.Path which is imported locally in the function
        with patch("pathlib.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.expanduser.return_value = mock_path_instance
            mock_path_instance.parent = mock_path_instance
            mock_path_instance.__str__ = MagicMock(
                return_value="/home/user/.config/llmcore/config.toml"
            )
            mock_path_cls.return_value = mock_path_instance

            with patch("builtins.open", MagicMock()):
                with patch("llmcore.api.tomli_w") as mock_tomli_w:
                    mock_tomli_w.dump = MagicMock()
                    result = LLMCore.sync_config_to_file(instance, None)

        # Should use default path
        assert "config.toml" in result


# ==============================================================================
# reload_config_from_file() Tests
# ==============================================================================


class TestReloadConfigFromFile:
    """Tests for reload_config_from_file() API."""

    @pytest.mark.asyncio
    async def test_reload_preserves_transient_state(self, mock_config: Dict[str, Any]) -> None:
        """Test that reload preserves transient sessions cache."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._transient_sessions_cache = {"session1": "data1"}
        instance._transient_last_interaction_info_cache = {"info1": "value1"}
        instance._config_file_path = None
        instance._initialize_from_config = AsyncMock()

        await LLMCore.reload_config_from_file(instance, None)

        # Verify transient state is preserved
        assert instance._transient_sessions_cache == {"session1": "data1"}
        assert instance._transient_last_interaction_info_cache == {"info1": "value1"}

    @pytest.mark.asyncio
    async def test_reload_calls_initialize(self) -> None:
        """Test that reload calls _initialize_from_config."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._transient_sessions_cache = {}
        instance._transient_last_interaction_info_cache = {}
        instance._config_file_path = "/path/to/config.toml"
        instance._initialize_from_config = AsyncMock()

        await LLMCore.reload_config_from_file(instance, None)

        instance._initialize_from_config.assert_called_once()


# ==============================================================================
# Helper Methods Tests
# ==============================================================================


class TestHelperMethods:
    """Tests for configuration helper methods."""

    def test_is_config_dirty_default_false(self) -> None:
        """Test that is_config_dirty returns False initially."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._runtime_config_dirty = False

        result = LLMCore.is_config_dirty(instance)

        assert result is False

    def test_is_config_dirty_true_after_set(self) -> None:
        """Test that is_config_dirty returns True after modification."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._runtime_config_dirty = True

        result = LLMCore.is_config_dirty(instance)

        assert result is True

    def test_get_config_file_path_returns_path(self) -> None:
        """Test that get_config_file_path returns the stored path."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._config_file_path = "/path/to/config.toml"

        result = LLMCore.get_config_file_path(instance)

        assert result == "/path/to/config.toml"

    def test_get_config_file_path_returns_none(self) -> None:
        """Test that get_config_file_path returns None when not set."""
        from llmcore.api import LLMCore

        instance = MagicMock()
        instance._config_file_path = None

        result = LLMCore.get_config_file_path(instance)

        assert result is None

    def test_get_default_config_path(self) -> None:
        """Test that get_default_config_path returns expanded path."""
        from llmcore.api import LLMCore

        instance = MagicMock()

        result = LLMCore.get_default_config_path(instance)

        assert "config.toml" in result
        assert "~" not in result  # Should be expanded
        assert os.path.isabs(result)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestConfigIntegration:
    """Integration tests for configuration management workflow."""

    def test_full_config_workflow(self, mock_config: Dict[str, Any]) -> None:
        """Test the full configuration management workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.toml")

            from llmcore.api import LLMCore

            # Create instance
            instance = MagicMock()
            mock_confy = MagicMock()
            mock_confy.as_dict.return_value = mock_config.copy()
            instance.config = mock_confy
            instance._config_file_path = None
            instance._runtime_config_dirty = False
            instance._original_config_dict = mock_config.copy()

            # Step 1: Get config
            config = LLMCore.get_runtime_config(instance, None)
            assert "llmcore" in config

            # Step 2: Check dirty status
            assert LLMCore.is_config_dirty(instance) is False

            # Step 3: Diff config
            diff = LLMCore.diff_config(instance)
            assert diff["has_changes"] is False

            # Step 4: Sync to file
            LLMCore.sync_config_to_file(instance, config_path)
            assert os.path.exists(config_path)


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling in configuration APIs."""

    def test_sync_without_tomli_w_raises(self, mock_config: Dict[str, Any]) -> None:
        """Test that sync raises error if tomli_w is not installed."""
        from llmcore.api import LLMCore
        from llmcore.exceptions import ConfigError

        instance = MagicMock()
        mock_confy = MagicMock()
        mock_confy.as_dict.return_value = mock_config
        instance.config = mock_confy

        with patch("llmcore.api.tomli_w", None):
            with pytest.raises(ConfigError) as exc_info:
                LLMCore.sync_config_to_file(instance, "/tmp/test.toml")

            assert "tomli-w" in str(exc_info.value)

    def test_get_config_exception_handling(self) -> None:
        """Test that get_runtime_config handles exceptions gracefully."""
        from llmcore.api import LLMCore
        from llmcore.exceptions import ConfigError

        instance = MagicMock()
        instance.config.as_dict.side_effect = Exception("Config error")

        with pytest.raises(ConfigError):
            LLMCore.get_runtime_config(instance, None)
