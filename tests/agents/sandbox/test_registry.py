# tests/agents/sandbox/test_registry.py
# tests/sandbox/test_registry.py
"""
Unit tests for the SandboxRegistry class.

Tests:
    - Registry configuration
    - Sandbox creation and tracking
    - Tool access control
    - Cleanup operations
    - Health checks
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llmcore.agents.sandbox.base import SandboxAccessLevel, SandboxConfig
from llmcore.agents.sandbox.exceptions import SandboxInitializationError
from llmcore.agents.sandbox.registry import SandboxMode, SandboxRegistry, SandboxRegistryConfig


class TestSandboxMode:
    """Tests for SandboxMode enum."""

    def test_docker_mode(self):
        """Test DOCKER mode value."""
        assert SandboxMode.DOCKER.value == "docker"

    def test_vm_mode(self):
        """Test VM mode value."""
        assert SandboxMode.VM.value == "vm"

    def test_hybrid_mode(self):
        """Test HYBRID mode value."""
        assert SandboxMode.HYBRID.value == "hybrid"


class TestSandboxRegistryConfig:
    """Tests for SandboxRegistryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxRegistryConfig()

        assert config.mode == SandboxMode.DOCKER
        assert config.fallback_enabled is True
        assert config.docker_enabled is True
        assert config.docker_image == "python:3.11-slim"
        assert config.vm_enabled is False

    def test_docker_mode_validation(self):
        """Test that docker mode requires docker enabled."""
        with pytest.raises(ValueError, match="docker_enabled"):
            SandboxRegistryConfig(mode=SandboxMode.DOCKER, docker_enabled=False)

    def test_vm_mode_validation(self):
        """Test that VM mode requires VM enabled and host set."""
        with pytest.raises(ValueError, match="vm_enabled"):
            SandboxRegistryConfig(mode=SandboxMode.VM, vm_enabled=False)

        with pytest.raises(ValueError, match="vm_host"):
            SandboxRegistryConfig(mode=SandboxMode.VM, vm_enabled=True, vm_host=None)

    def test_valid_vm_config(self):
        """Test valid VM configuration."""
        config = SandboxRegistryConfig(
            mode=SandboxMode.VM, vm_enabled=True, vm_host="192.168.1.100", docker_enabled=False
        )

        assert config.mode == SandboxMode.VM
        assert config.vm_host == "192.168.1.100"


class TestSandboxRegistry:
    """Tests for SandboxRegistry class."""

    @pytest.fixture
    def registry_config(self, temp_dir: Path) -> SandboxRegistryConfig:
        """Create registry config for testing."""
        return SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            docker_image="python:3.11-slim",
            docker_image_whitelist=["python:3.*-slim"],
            share_path=str(temp_dir / "share"),
            outputs_path=str(temp_dir / "outputs"),
            allowed_tools=["execute_shell", "save_file"],
            denied_tools=["sudo_execute"],
        )

    @pytest.fixture
    def registry(self, registry_config) -> SandboxRegistry:
        """Create registry instance for testing."""
        return SandboxRegistry(registry_config)

    def test_registry_creation(self, registry):
        """Test registry is created successfully."""
        assert registry is not None
        assert registry.get_active_count() == 0

    def test_get_config(self, registry, registry_config):
        """Test getting registry configuration."""
        config = registry.get_config()

        assert config.mode == registry_config.mode
        assert config.docker_image == registry_config.docker_image

    @pytest.mark.asyncio
    async def test_create_sandbox_docker_not_available(self, registry: SandboxRegistry):
        """Test sandbox creation fails gracefully when Docker unavailable."""
        sandbox_config = SandboxConfig()

        # Mock Docker provider to fail
        with patch(
            "llmcore.agents.sandbox.docker_provider.DockerSandboxProvider._connect_docker"
        ) as mock_connect:
            mock_connect.side_effect = Exception("Docker not available")

            with pytest.raises(SandboxInitializationError):
                await registry.create_sandbox(sandbox_config)

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_not_found(self, registry: SandboxRegistry):
        """Test cleanup returns False for non-existent sandbox."""
        result = await registry.cleanup_sandbox("non-existent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_sandbox_not_found(self, registry: SandboxRegistry):
        """Test getting non-existent sandbox returns None."""
        sandbox = await registry.get_sandbox("non-existent-id")

        assert sandbox is None

    @pytest.mark.asyncio
    async def test_get_sandbox_status_not_found(self, registry: SandboxRegistry):
        """Test getting status for non-existent sandbox returns None."""
        status = await registry.get_sandbox_status("non-existent-id")

        assert status is None

    def test_list_active_sandboxes_empty(self, registry: SandboxRegistry):
        """Test listing active sandboxes when none exist."""
        sandboxes = registry.list_active_sandboxes()

        assert sandboxes == []

    @pytest.mark.asyncio
    async def test_cleanup_all_empty(self, registry: SandboxRegistry):
        """Test cleanup_all with no active sandboxes."""
        results = await registry.cleanup_all()

        assert results == {}

    @pytest.mark.asyncio
    async def test_health_check_all_empty(self, registry: SandboxRegistry):
        """Test health_check_all with no active sandboxes."""
        results = await registry.health_check_all()

        assert results == {}


class TestToolAccessControl:
    """Tests for tool access control in SandboxRegistry."""

    @pytest.fixture
    def registry_with_tools(self, temp_dir: Path) -> SandboxRegistry:
        """Create registry with specific tool configuration."""
        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            share_path=str(temp_dir / "share"),
            outputs_path=str(temp_dir / "outputs"),
            allowed_tools=["execute_shell", "execute_python", "save_file", "load_file"],
            denied_tools=["sudo_execute", "network_request"],
        )
        return SandboxRegistry(config)

    def test_tool_allowed_full_access(self, registry_with_tools):
        """Test that full access bypasses tool restrictions."""
        # All tools should be allowed for FULL access
        assert registry_with_tools.is_tool_allowed("sudo_execute", SandboxAccessLevel.FULL) is True

        assert (
            registry_with_tools.is_tool_allowed("network_request", SandboxAccessLevel.FULL) is True
        )

        assert registry_with_tools.is_tool_allowed("random_tool", SandboxAccessLevel.FULL) is True

    def test_tool_allowed_restricted(self, registry_with_tools):
        """Test tool allowance for restricted access."""
        # Allowed tools should be allowed
        assert (
            registry_with_tools.is_tool_allowed("execute_shell", SandboxAccessLevel.RESTRICTED)
            is True
        )

        assert (
            registry_with_tools.is_tool_allowed("save_file", SandboxAccessLevel.RESTRICTED) is True
        )

    def test_tool_denied_restricted(self, registry_with_tools):
        """Test tool denial for restricted access."""
        # Denied tools should be denied
        assert (
            registry_with_tools.is_tool_allowed("sudo_execute", SandboxAccessLevel.RESTRICTED)
            is False
        )

        assert (
            registry_with_tools.is_tool_allowed("network_request", SandboxAccessLevel.RESTRICTED)
            is False
        )

    def test_tool_not_in_allowed_list(self, registry_with_tools):
        """Test tool not in allowed list is denied for restricted."""
        assert (
            registry_with_tools.is_tool_allowed("random_tool", SandboxAccessLevel.RESTRICTED)
            is False
        )

    def test_get_allowed_tools_full(self, registry_with_tools):
        """Test getting allowed tools for full access."""
        tools = registry_with_tools.get_allowed_tools(SandboxAccessLevel.FULL)

        # Full access gets all tools
        assert "*" in tools or len(tools) > 0

    def test_get_allowed_tools_restricted(self, registry_with_tools):
        """Test getting allowed tools for restricted access."""
        tools = registry_with_tools.get_allowed_tools(SandboxAccessLevel.RESTRICTED)

        assert "execute_shell" in tools
        assert "save_file" in tools
        assert "sudo_execute" not in tools


class TestRegistryWithMockProvider:
    """Integration tests using mock provider."""

    @pytest.fixture
    def mock_provider(self, mock_sandbox_provider):
        """Get mock provider from conftest."""
        return mock_sandbox_provider

    @pytest.mark.asyncio
    async def test_registry_tracks_sandbox(
        self, mock_sandbox_provider, sandbox_config, temp_dir: Path
    ):
        """Test that registry tracks created sandboxes."""
        # Create registry
        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            share_path=str(temp_dir / "share"),
            outputs_path=str(temp_dir / "outputs"),
        )
        registry = SandboxRegistry(config)

        # Manually add mock sandbox to registry
        await mock_sandbox_provider.initialize(sandbox_config)
        sandbox_id = sandbox_config.sandbox_id
        registry._active_sandboxes[sandbox_id] = mock_sandbox_provider

        # Verify tracking
        assert registry.get_active_count() == 1

        sandbox = await registry.get_sandbox(sandbox_id)
        assert sandbox is mock_sandbox_provider

        # Cleanup
        result = await registry.cleanup_sandbox(sandbox_id)
        assert result is True
        assert registry.get_active_count() == 0

    @pytest.mark.asyncio
    async def test_health_check_with_mock(
        self, mock_sandbox_provider, sandbox_config, temp_dir: Path
    ):
        """Test health check with mock provider."""
        # Create registry
        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            share_path=str(temp_dir / "share"),
            outputs_path=str(temp_dir / "outputs"),
        )
        registry = SandboxRegistry(config)

        # Initialize and add mock
        await mock_sandbox_provider.initialize(sandbox_config)
        sandbox_id = sandbox_config.sandbox_id
        registry._active_sandboxes[sandbox_id] = mock_sandbox_provider

        # Check health
        healthy = await registry.health_check(sandbox_id)
        assert healthy is True

        # Cleanup
        await mock_sandbox_provider.cleanup()

        # After cleanup, should be unhealthy
        healthy = await registry.health_check(sandbox_id)
        assert healthy is False
