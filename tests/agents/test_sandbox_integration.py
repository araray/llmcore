# tests/agents/test_sandbox_integration.py
"""
Tests for sandbox integration in SingleAgentMode.

This module tests the _setup_sandbox() and _cleanup_sandbox() methods
in SingleAgentMode to ensure they properly wire to the sandbox infrastructure.

Tests cover:
    - Sandbox creation via SandboxRegistry
    - Configuration loading from TOML/environment
    - Mode selection (Docker, VM)
    - Graceful fallback when Docker unavailable
    - Proper cleanup on execution completion
    - Error handling scenarios
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test fixtures from sandbox conftest
from llmcore.agents.sandbox.base import (
    SandboxAccessLevel,
    SandboxConfig,
    SandboxProvider,
    SandboxStatus,
)
from llmcore.agents.sandbox.config import (
    DockerConfig,
    OutputTrackingConfig,
    SandboxSystemConfig,
    ToolsConfig,
    VMConfig,
    VolumeConfig,
)
from llmcore.agents.sandbox.registry import SandboxMode, SandboxRegistry, SandboxRegistryConfig

# ==============================================================================
# Event Loop Configuration
# ==============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# Mock Fixtures
# ==============================================================================


class MockSandboxProvider(SandboxProvider):
    """Minimal mock sandbox provider for testing."""

    def __init__(self, sandbox_id: str = "test-sandbox"):
        self._sandbox_id = sandbox_id
        self._config: SandboxConfig = None
        self._status = SandboxStatus.CREATED
        self._access_level = SandboxAccessLevel.RESTRICTED
        self._cleaned_up = False

    async def initialize(self, config: SandboxConfig) -> None:
        self._config = config
        self._status = SandboxStatus.READY

    async def execute_shell(self, command, timeout=None, working_dir=None):
        pass

    async def execute_python(self, code, timeout=None, working_dir=None):
        pass

    async def write_file(self, path, content, mode="w"):
        return True

    async def read_file(self, path):
        return None

    async def write_file_binary(self, path, content):
        return True

    async def read_file_binary(self, path):
        return None

    async def list_files(self, path=".", recursive=False):
        return []

    async def file_exists(self, path):
        return False

    async def delete_file(self, path):
        return True

    async def create_directory(self, path):
        return True

    async def cleanup(self) -> None:
        self._status = SandboxStatus.TERMINATED
        self._cleaned_up = True

    async def is_healthy(self):
        return self._status == SandboxStatus.READY

    def get_access_level(self):
        return self._access_level

    def get_status(self):
        return self._status

    def get_config(self):
        return self._config

    def get_info(self):
        return {
            "provider": "mock",
            "sandbox_id": self._sandbox_id,
        }


@pytest.fixture
def mock_sandbox_provider():
    """Create a mock sandbox provider."""
    return MockSandboxProvider()


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory(prefix="llmcore_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_sandbox_system_config(temp_config_dir: Path) -> SandboxSystemConfig:
    """Create a mock SandboxSystemConfig for testing."""
    return SandboxSystemConfig(
        mode="docker",
        fallback_enabled=True,
        docker=DockerConfig(
            enabled=True,
            image="python:3.11-slim",
            image_whitelist=["python:3.*-slim"],
            memory_limit="512m",
            timeout_seconds=60,
        ),
        vm=VMConfig(enabled=False),
        volumes=VolumeConfig(
            share_path=str(temp_config_dir / "share"),
            outputs_path=str(temp_config_dir / "outputs"),
        ),
        tools=ToolsConfig(),
        output_tracking=OutputTrackingConfig(),
    )


@pytest.fixture
def mock_provider_manager():
    """Create a mock ProviderManager."""
    manager = MagicMock()
    manager.get_provider.return_value = MagicMock()
    manager.get_default_provider_name.return_value = "mock"
    manager.get_default_model.return_value = "mock-model"
    manager._config = None
    return manager


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    return MagicMock()


@pytest.fixture
def mock_storage_manager():
    """Create a mock StorageManager."""
    return MagicMock()


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager."""
    manager = MagicMock()
    manager.get_tool_definitions.return_value = []
    manager.get_tool_names.return_value = []
    return manager


# ==============================================================================
# SingleAgentMode Setup Tests
# ==============================================================================


class TestSandboxSetup:
    """Tests for _setup_sandbox() method."""

    @pytest.mark.asyncio
    async def test_setup_sandbox_returns_none_when_disabled(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that _setup_sandbox returns None when sandbox is disabled in config."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Create config with sandbox disabled
        disabled_config = SandboxSystemConfig(
            mode="docker",
            docker=DockerConfig(enabled=False),
            vm=VMConfig(enabled=False),
        )

        # Patch at the source module where the function is defined
        with patch(
            "llmcore.agents.sandbox.config.load_sandbox_config",
            return_value=disabled_config,
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            sandbox = await agent._setup_sandbox(None)
            assert sandbox is None

    @pytest.mark.asyncio
    async def test_setup_sandbox_uses_config_default_mode(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
        mock_sandbox_provider,
    ):
        """Test that _setup_sandbox uses config default mode when not specified."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(return_value=mock_sandbox_provider)

        # Patch at source modules
        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            sandbox = await agent._setup_sandbox(None)
            assert sandbox is not None
            mock_registry.create_sandbox.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_sandbox_override_mode(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
        mock_sandbox_provider,
    ):
        """Test that _setup_sandbox respects sandbox_type override."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(return_value=mock_sandbox_provider)

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            sandbox = await agent._setup_sandbox("docker")
            assert sandbox is not None

            # Verify the call used docker mode
            call_kwargs = mock_registry.create_sandbox.call_args.kwargs
            assert call_kwargs.get("prefer_mode") == SandboxMode.DOCKER

    @pytest.mark.asyncio
    async def test_setup_sandbox_invalid_mode_falls_back(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
        mock_sandbox_provider,
    ):
        """Test that invalid sandbox_type falls back to config default."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(return_value=mock_sandbox_provider)

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            # Invalid mode should fall back to config default
            sandbox = await agent._setup_sandbox("invalid_mode")
            assert sandbox is not None
            mock_registry.create_sandbox.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_sandbox_stores_registry_reference(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
        mock_sandbox_provider,
    ):
        """Test that _setup_sandbox stores registry reference for cleanup."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(return_value=mock_sandbox_provider)

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            await agent._setup_sandbox("docker")
            assert agent._sandbox_registry is mock_registry

    @pytest.mark.asyncio
    async def test_setup_sandbox_fallback_on_creation_failure(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
    ):
        """Test that _setup_sandbox falls back gracefully when creation fails."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(side_effect=Exception("Docker not available"))

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            # Should return None (fallback) instead of raising
            sandbox = await agent._setup_sandbox("docker")
            assert sandbox is None

    @pytest.mark.asyncio
    async def test_setup_sandbox_raises_when_fallback_disabled(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        temp_config_dir,
    ):
        """Test that _setup_sandbox raises when fallback is disabled."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Config with fallback disabled
        strict_config = SandboxSystemConfig(
            mode="docker",
            fallback_enabled=False,
            docker=DockerConfig(enabled=True),
            vm=VMConfig(enabled=False),
            volumes=VolumeConfig(
                share_path=str(temp_config_dir / "share"),
                outputs_path=str(temp_config_dir / "outputs"),
            ),
        )

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(side_effect=Exception("Docker not available"))

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=strict_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            # Should raise when fallback is disabled
            with pytest.raises(Exception, match="Docker not available"):
                await agent._setup_sandbox("docker")

    @pytest.mark.asyncio
    async def test_setup_sandbox_config_load_failure(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that _setup_sandbox returns None when config loading fails."""
        from llmcore.agents.single_agent import SingleAgentMode

        with patch(
            "llmcore.agents.sandbox.config.load_sandbox_config",
            side_effect=Exception("Config load failed"),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            sandbox = await agent._setup_sandbox("docker")
            assert sandbox is None


# ==============================================================================
# SingleAgentMode Cleanup Tests
# ==============================================================================


class TestSandboxCleanup:
    """Tests for _cleanup_sandbox() method."""

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_none_is_noop(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that cleanup with None sandbox is a no-op."""
        from llmcore.agents.single_agent import SingleAgentMode

        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        # Should not raise
        await agent._cleanup_sandbox(None)

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_via_registry(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_provider,
    ):
        """Test that cleanup uses registry when available."""
        from llmcore.agents.single_agent import SingleAgentMode

        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        # Setup mock registry
        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.cleanup_sandbox = AsyncMock(return_value=True)
        agent._sandbox_registry = mock_registry

        # Initialize the mock provider with config
        config = SandboxConfig(sandbox_id="test-sandbox-123")
        await mock_sandbox_provider.initialize(config)

        await agent._cleanup_sandbox(mock_sandbox_provider)
        mock_registry.cleanup_sandbox.assert_called_once_with("test-sandbox-123")

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_direct_fallback(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_provider,
    ):
        """Test that cleanup falls back to direct cleanup when no registry."""
        from llmcore.agents.single_agent import SingleAgentMode

        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        # No registry set
        agent._sandbox_registry = None

        # Initialize the mock provider with config
        config = SandboxConfig(sandbox_id="test-sandbox-456")
        await mock_sandbox_provider.initialize(config)

        await agent._cleanup_sandbox(mock_sandbox_provider)
        assert mock_sandbox_provider._cleaned_up is True

    @pytest.mark.asyncio
    async def test_cleanup_sandbox_error_does_not_raise(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that cleanup errors don't propagate."""
        from llmcore.agents.single_agent import SingleAgentMode

        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        # Mock sandbox that raises on cleanup
        failing_sandbox = MagicMock(spec=SandboxProvider)
        failing_sandbox.get_config.return_value = SandboxConfig(sandbox_id="test")
        failing_sandbox.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))
        agent._sandbox_registry = None

        # Should not raise
        await agent._cleanup_sandbox(failing_sandbox)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestSandboxIntegrationWithRun:
    """Integration tests for sandbox with run() method."""

    @pytest.mark.asyncio
    async def test_run_with_sandbox_creates_and_cleans_up(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_system_config,
        mock_sandbox_provider,
    ):
        """Test that run() creates sandbox and cleans up on completion."""
        from llmcore.agents.single_agent import SingleAgentMode

        mock_registry = MagicMock(spec=SandboxRegistry)
        mock_registry.create_sandbox = AsyncMock(return_value=mock_sandbox_provider)
        mock_registry.cleanup_sandbox = AsyncMock(return_value=True)

        # Initialize provider with config
        config = SandboxConfig(sandbox_id="run-test-sandbox")
        await mock_sandbox_provider.initialize(config)

        # Mock the cognitive cycle
        mock_cognitive_cycle = MagicMock()
        mock_cognitive_cycle.run_until_complete = AsyncMock(return_value="Task completed")

        with (
            patch(
                "llmcore.agents.sandbox.config.load_sandbox_config",
                return_value=mock_sandbox_system_config,
            ),
            patch(
                "llmcore.agents.sandbox.config.create_registry_config",
                return_value=MagicMock(spec=SandboxRegistryConfig),
            ),
            patch(
                "llmcore.agents.sandbox.registry.SandboxRegistry",
                return_value=mock_registry,
            ),
            patch(
                "llmcore.agents.cognitive.CognitiveCycle",
                return_value=mock_cognitive_cycle,
            ),
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            result = await agent.run(
                goal="Test goal",
                use_sandbox=True,
                sandbox_type="docker",
                skip_goal_classification=True,
            )

            # Verify sandbox was created
            mock_registry.create_sandbox.assert_called_once()

            # Verify cleanup was called
            mock_registry.cleanup_sandbox.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_without_sandbox_skips_sandbox_ops(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that run() skips sandbox operations when use_sandbox=False."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Mock the cognitive cycle
        mock_cognitive_cycle = MagicMock()
        mock_cognitive_cycle.run_until_complete = AsyncMock(return_value="Task completed")

        with (
            patch(
                "llmcore.agents.cognitive.CognitiveCycle",
                return_value=mock_cognitive_cycle,
            ),
            patch.object(
                SingleAgentMode,
                "_setup_sandbox",
                new_callable=AsyncMock,
            ) as mock_setup,
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            await agent.run(
                goal="Test goal",
                use_sandbox=False,
                skip_goal_classification=True,
            )

            # Verify _setup_sandbox was NOT called
            mock_setup.assert_not_called()


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestSandboxConfiguration:
    """Tests for sandbox configuration handling."""

    @pytest.mark.asyncio
    async def test_sandbox_config_from_environment(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        mock_sandbox_provider,
        monkeypatch,
    ):
        """Test that sandbox configuration respects environment variables."""
        from llmcore.agents.sandbox.config import load_sandbox_config

        # Set environment variables
        monkeypatch.setenv("LLMCORE_SANDBOX_MODE", "docker")
        monkeypatch.setenv("LLMCORE_SANDBOX_DOCKER_IMAGE", "python:3.12-slim")

        # Load config and verify env vars are applied
        config = load_sandbox_config()
        assert config.mode == "docker"
        assert config.docker.image == "python:3.12-slim"

    def test_sandbox_system_config_to_dict(self, mock_sandbox_system_config):
        """Test SandboxSystemConfig serialization."""
        config_dict = mock_sandbox_system_config.to_dict()

        assert config_dict["mode"] == "docker"
        assert config_dict["fallback_enabled"] is True
        assert config_dict["docker"]["enabled"] is True
        assert config_dict["docker"]["image"] == "python:3.11-slim"
        assert config_dict["vm"]["enabled"] is False


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestSandboxEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_setup_sandbox_with_none_config(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test handling when sandbox config returns None-like values."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Config with everything disabled
        empty_config = SandboxSystemConfig(
            mode="docker",
            docker=DockerConfig(enabled=False),
            vm=VMConfig(enabled=False),
        )

        with patch(
            "llmcore.agents.sandbox.config.load_sandbox_config",
            return_value=empty_config,
        ):
            agent = SingleAgentMode(
                provider_manager=mock_provider_manager,
                memory_manager=mock_memory_manager,
                storage_manager=mock_storage_manager,
                tool_manager=mock_tool_manager,
            )

            result = await agent._setup_sandbox("docker")
            assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_config(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test cleanup handles sandbox with no config."""
        from llmcore.agents.single_agent import SingleAgentMode

        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        # Mock sandbox with no config
        sandbox = MagicMock(spec=SandboxProvider)
        sandbox.get_config.return_value = None
        sandbox.cleanup = AsyncMock()

        agent._sandbox_registry = None

        # Should not raise
        await agent._cleanup_sandbox(sandbox)
        sandbox.cleanup.assert_called_once()
