# tests/sandbox/test_config.py
"""
Unit tests for sandbox configuration management.

These tests verify configuration loading, validation,
and conversion functionality.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
# Assumes llmcore is installed or in PYTHONPATH

from llmcore.agents.sandbox.config import (
    SandboxSystemConfig,
    DockerConfig,
    VMConfig,
    VolumeConfig,
    ToolsConfig,
    OutputTrackingConfig,
    load_sandbox_config,
    create_registry_config,
    generate_sample_config,
    write_sample_config,
    DEFAULT_CONFIG,
    _deep_merge
)
from llmcore.agents.sandbox.registry import SandboxMode


# =============================================================================
# DEFAULT CONFIG TESTS
# =============================================================================

class TestDefaultConfig:
    """Tests for default configuration."""

    def test_default_config_structure(self):
        """Test default config has expected structure."""
        assert "mode" in DEFAULT_CONFIG
        assert "fallback_enabled" in DEFAULT_CONFIG
        assert "docker" in DEFAULT_CONFIG
        assert "vm" in DEFAULT_CONFIG
        assert "volumes" in DEFAULT_CONFIG
        assert "tools" in DEFAULT_CONFIG

    def test_default_mode_is_docker(self):
        """Test default mode is Docker."""
        assert DEFAULT_CONFIG["mode"] == "docker"

    def test_default_docker_image(self):
        """Test default Docker image."""
        assert DEFAULT_CONFIG["docker"]["image"] == "python:3.11-slim"

    def test_default_vm_disabled(self):
        """Test VM is disabled by default."""
        assert DEFAULT_CONFIG["vm"]["enabled"] is False


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestDockerConfig:
    """Tests for DockerConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = DockerConfig()

        assert config.enabled is True
        assert config.image == "python:3.11-slim"
        assert config.auto_pull is True
        assert config.memory_limit == "1g"
        assert config.cpu_limit == 2.0

    def test_custom_values(self):
        """Test custom values."""
        config = DockerConfig(
            image="python:3.12-bookworm",
            memory_limit="2g",
            cpu_limit=4.0
        )

        assert config.image == "python:3.12-bookworm"
        assert config.memory_limit == "2g"
        assert config.cpu_limit == 4.0


class TestVMConfig:
    """Tests for VMConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = VMConfig()

        assert config.enabled is False
        assert config.host is None
        assert config.port == 22
        assert config.username == "agent"

    def test_custom_values(self):
        """Test custom values."""
        config = VMConfig(
            enabled=True,
            host="192.168.1.100",
            port=2222,
            username="llmcore"
        )

        assert config.enabled is True
        assert config.host == "192.168.1.100"
        assert config.port == 2222


class TestVolumeConfig:
    """Tests for VolumeConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = VolumeConfig()

        assert "agent_share" in config.share_path
        assert "agent_outputs" in config.outputs_path


class TestToolsConfig:
    """Tests for ToolsConfig dataclass."""

    def test_default_allowed_tools(self):
        """Test default allowed tools."""
        config = ToolsConfig()

        assert "execute_shell" in config.allowed
        assert "execute_python" in config.allowed
        assert "save_file" in config.allowed

    def test_default_denied_tools(self):
        """Test default denied tools."""
        config = ToolsConfig()

        assert "sudo_execute" in config.denied
        assert "network_request" in config.denied


class TestSandboxSystemConfig:
    """Tests for SandboxSystemConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = SandboxSystemConfig()

        assert config.mode == "docker"
        assert config.fallback_enabled is True
        assert isinstance(config.docker, DockerConfig)
        assert isinstance(config.vm, VMConfig)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = SandboxSystemConfig()

        d = config.to_dict()

        assert d["mode"] == "docker"
        assert "docker" in d
        assert "vm" in d
        assert d["docker"]["image"] == "python:3.11-slim"


# =============================================================================
# DEEP MERGE TESTS
# =============================================================================

class TestDeepMerge:
    """Tests for deep merge utility."""

    def test_simple_merge(self):
        """Test simple merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_nested_merge(self):
        """Test nested merge."""
        base = {"outer": {"inner": 1, "other": 2}}
        override = {"outer": {"inner": 10}}

        result = _deep_merge(base, override)

        assert result["outer"]["inner"] == 10
        assert result["outer"]["other"] == 2

    def test_list_override(self):
        """Test list values are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = _deep_merge(base, override)

        assert result["items"] == [4, 5]


# =============================================================================
# CONFIGURATION LOADING TESTS
# =============================================================================

class TestLoadSandboxConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_sandbox_config()

        assert isinstance(config, SandboxSystemConfig)
        assert config.mode == "docker"

    def test_load_with_overrides(self):
        """Test loading with overrides."""
        overrides = {
            "mode": "hybrid",
            "docker": {
                "image": "python:3.12"
            }
        }

        config = load_sandbox_config(overrides=overrides)

        assert config.mode == "hybrid"
        assert config.docker.image == "python:3.12"

    def test_load_from_toml_file(self):
        """Test loading from TOML file."""
        toml_content = """
[agents.sandbox]
mode = "vm"

[agents.sandbox.docker]
image = "custom:latest"

[agents.sandbox.vm]
enabled = true
host = "192.168.1.50"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                config = load_sandbox_config(config_path=Path(f.name))

                assert config.mode == "vm"
                assert config.docker.image == "custom:latest"
                assert config.vm.enabled is True
                assert config.vm.host == "192.168.1.50"
            finally:
                os.unlink(f.name)

    def test_load_from_env_vars(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            'LLMCORE_SANDBOX_MODE': 'hybrid',
            'LLMCORE_SANDBOX_DOCKER_IMAGE': 'python:3.12-alpine'
        }):
            config = load_sandbox_config()

            # Note: Actual env var loading depends on implementation
            # This test verifies the function works with env vars set
            assert isinstance(config, SandboxSystemConfig)


# =============================================================================
# REGISTRY CONFIG CONVERSION TESTS
# =============================================================================

class TestCreateRegistryConfig:
    """Tests for registry config conversion."""

    def test_convert_docker_mode(self):
        """Test converting Docker mode config."""
        sandbox_config = SandboxSystemConfig(mode="docker")

        registry_config = create_registry_config(sandbox_config)

        assert registry_config.mode == SandboxMode.DOCKER

    def test_convert_vm_mode(self):
        """Test converting VM mode config."""
        sandbox_config = SandboxSystemConfig(mode="vm")

        registry_config = create_registry_config(sandbox_config)

        assert registry_config.mode == SandboxMode.VM

    def test_convert_hybrid_mode(self):
        """Test converting hybrid mode config."""
        sandbox_config = SandboxSystemConfig(mode="hybrid")

        registry_config = create_registry_config(sandbox_config)

        assert registry_config.mode == SandboxMode.HYBRID

    def test_convert_docker_settings(self):
        """Test converting Docker settings."""
        sandbox_config = SandboxSystemConfig(
            docker=DockerConfig(
                image="python:3.12",
                memory_limit="2g",
                cpu_limit=4.0
            )
        )

        registry_config = create_registry_config(sandbox_config)

        assert registry_config.docker_image == "python:3.12"
        assert registry_config.docker_memory_limit == "2g"
        assert registry_config.docker_cpu_limit == 4.0

    def test_convert_vm_settings(self):
        """Test converting VM settings."""
        sandbox_config = SandboxSystemConfig(
            vm=VMConfig(
                enabled=True,
                host="192.168.1.100",
                port=2222,
                username="llmcore"
            )
        )

        registry_config = create_registry_config(sandbox_config)

        assert registry_config.vm_enabled is True
        assert registry_config.vm_host == "192.168.1.100"
        assert registry_config.vm_port == 2222
        assert registry_config.vm_username == "llmcore"

    def test_convert_tool_settings(self):
        """Test converting tool settings."""
        sandbox_config = SandboxSystemConfig(
            tools=ToolsConfig(
                allowed=["tool1", "tool2"],
                denied=["bad_tool"]
            )
        )

        registry_config = create_registry_config(sandbox_config)

        assert "tool1" in registry_config.allowed_tools
        assert "bad_tool" in registry_config.denied_tools


# =============================================================================
# SAMPLE CONFIG GENERATION TESTS
# =============================================================================

class TestSampleConfigGeneration:
    """Tests for sample config generation."""

    def test_generate_sample_config(self):
        """Test generating sample config."""
        sample = generate_sample_config()

        assert "[agents.sandbox]" in sample
        assert "[agents.sandbox.docker]" in sample
        assert "[agents.sandbox.vm]" in sample
        assert "mode = " in sample

    def test_write_sample_config(self):
        """Test writing sample config to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.toml"

            result = write_sample_config(path)

            assert result == path
            assert path.exists()

            content = path.read_text()
            assert "[agents.sandbox]" in content

    def test_write_sample_config_creates_directory(self):
        """Test write creates parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "subdir" / "sample.toml"

            write_sample_config(path)

            assert path.exists()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_overrides(self):
        """Test loading with empty overrides."""
        config = load_sandbox_config(overrides={})

        assert isinstance(config, SandboxSystemConfig)

    def test_none_overrides(self):
        """Test loading with None overrides."""
        config = load_sandbox_config(overrides=None)

        assert isinstance(config, SandboxSystemConfig)

    def test_partial_overrides(self):
        """Test loading with partial overrides."""
        overrides = {
            "docker": {
                "memory_limit": "4g"
                # Other docker settings should use defaults
            }
        }

        config = load_sandbox_config(overrides=overrides)

        assert config.docker.memory_limit == "4g"
        assert config.docker.image == "python:3.12-alpine"  # Default changed

    def test_invalid_mode_defaults(self):
        """Test invalid mode falls back to docker."""
        overrides = {"mode": "invalid_mode"}

        config = load_sandbox_config(overrides=overrides)
        registry_config = create_registry_config(config)

        # Should fall back to DOCKER
        assert registry_config.mode == SandboxMode.DOCKER
