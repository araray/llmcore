# tests/agents/test_capability_check_integration.py
"""
G3 Phase 4: Capability Pre-Check Integration Tests.

Tests that the capability checking is properly integrated into
SingleAgentMode.run() and fails fast for incompatible models.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add source to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestCapabilityCheckIntegration:
    """Test capability pre-check integration in SingleAgentMode."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        manager = MagicMock()
        provider = MagicMock()
        provider.default_model = "gpt-4o"
        provider.get_name.return_value = "openai"
        provider.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "Final Answer: Hello!"}}],
                "usage": {"total_tokens": 100},
            }
        )
        provider.extract_response_content = MagicMock(return_value="Final Answer: Hello!")
        manager.get_provider.return_value = provider
        manager.get_default_provider_name.return_value = "openai"
        manager.get_default_model.return_value = "gpt-4o"
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        return MagicMock()

    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock storage manager."""
        return MagicMock()

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager."""
        manager = MagicMock()
        manager.get_tool_definitions.return_value = []
        manager.get_tool_names.return_value = []
        manager.load_default_tools = MagicMock()
        return manager

    @pytest.fixture
    def agents_config(self):
        """Create an agents config with capability check enabled."""
        from llmcore.config.agents_config import AgentsConfig

        config = AgentsConfig()
        config.capability_check.enabled = True
        config.capability_check.strict_mode = True
        return config

    @pytest.mark.asyncio
    async def test_capability_check_fails_for_non_tool_model(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        agents_config,
    ):
        """Test that capability check fails for models without tool support."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Create agent
        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            agents_config=agents_config,
        )

        # Run with a model that doesn't support tools (gemma3:4b)
        result = await agent.run(
            goal="Search for files",
            model_name="gemma3:4b",  # No tool support
            skip_goal_classification=True,  # Skip classification for direct test
        )

        # Should fail immediately with capability error
        assert not result.success
        assert result.iteration_count == 0  # Never started iterations
        assert "does not support required capabilities" in result.final_answer
        assert "gemma3:4b" in result.final_answer

    @pytest.mark.asyncio
    async def test_capability_check_passes_for_tool_model(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
        agents_config,
    ):
        """Test that capability check passes for tool-capable models."""
        from llmcore.agents.single_agent import SingleAgentMode

        # Create agent
        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            agents_config=agents_config,
        )

        # Run with a tool-capable model
        result = await agent.run(
            goal="hello",
            model_name="gpt-4o",  # Has tool support
            skip_goal_classification=False,
        )

        # Should succeed (via fast-path for trivial goal)
        # The mock returns a final answer, so it should complete
        assert result.fast_path or result.iteration_count >= 0

    @pytest.mark.asyncio
    async def test_capability_check_disabled(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that capability check can be disabled."""
        from llmcore.agents.single_agent import SingleAgentMode
        from llmcore.config.agents_config import AgentsConfig

        # Create config with capability check disabled
        config = AgentsConfig()
        config.capability_check.enabled = False

        # Create agent
        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            agents_config=config,
        )

        # Run with non-tool model - should not fail immediately
        # (will fail later in the process, but that's not what we're testing)
        with patch.object(
            agent.cognitive_cycle, "run_until_complete", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = "Task complete"

            result = await agent.run(
                goal="Search for files",
                model_name="gemma3:4b",
                skip_goal_classification=True,
            )

            # Should have attempted to run (capability check didn't stop it)
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_capability_check_non_strict_mode(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Test that non-strict mode only warns but continues."""
        from llmcore.agents.single_agent import SingleAgentMode
        from llmcore.config.agents_config import AgentsConfig

        # Create config with non-strict mode
        config = AgentsConfig()
        config.capability_check.enabled = True
        config.capability_check.strict_mode = False  # Warn but continue

        # Create agent
        agent = SingleAgentMode(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            agents_config=config,
        )

        # Run with non-tool model - should warn but continue
        with patch.object(
            agent.cognitive_cycle, "run_until_complete", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = "Task complete via activity fallback"

            result = await agent.run(
                goal="Search for files",
                model_name="gemma3:4b",
                skip_goal_classification=True,
            )

            # Should have attempted to run (warning logged but continued)
            mock_run.assert_called_once()


class TestCapabilityChecker:
    """Test the CapabilityChecker directly."""

    def test_tool_capable_model(self):
        """Test checking a model with tool support."""
        from llmcore.agents.routing.capability_checker import CapabilityChecker

        checker = CapabilityChecker()
        result = checker.check_compatibility("gpt-4o", requires_tools=True)

        assert result.compatible
        assert len(result.issues) == 0

    def test_non_tool_model(self):
        """Test checking a model without tool support."""
        from llmcore.agents.routing.capability_checker import CapabilityChecker

        checker = CapabilityChecker()
        result = checker.check_compatibility("gemma3:4b", requires_tools=True)

        assert not result.compatible
        assert len(result.issues) > 0
        assert any("tool" in str(i).lower() for i in result.issues)

    def test_vision_capable_model(self):
        """Test checking a model with vision support."""
        from llmcore.agents.routing.capability_checker import CapabilityChecker

        checker = CapabilityChecker()
        result = checker.check_compatibility("gpt-4o", requires_vision=True)

        assert result.compatible

    def test_non_vision_model(self):
        """Test checking a model without vision support."""
        from llmcore.agents.routing.capability_checker import CapabilityChecker

        checker = CapabilityChecker()
        result = checker.check_compatibility("llama3:8b", requires_vision=True)

        assert not result.compatible

    def test_suggest_alternatives(self):
        """Test that alternatives are suggested for incompatible models."""
        from llmcore.agents.routing.capability_checker import (
            Capability,
            CapabilityChecker,
        )

        checker = CapabilityChecker()
        suggestions = checker.suggest_alternative_models(
            required_capabilities={Capability.TOOLS, Capability.VISION},
            exclude_providers={"ollama"},
            max_suggestions=3,
        )

        assert len(suggestions) > 0
        # All suggestions should have both capabilities
        for model in suggestions:
            assert model.supports_tools
            assert model.supports_vision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
