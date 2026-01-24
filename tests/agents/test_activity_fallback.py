# tests/agents/test_activity_fallback.py
"""
G3 Phase 6: Activity Fallback Tests.

Tests that the activity system fallback is properly integrated into
the think_phase and act_phase when native tools fail.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add source to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestActivityFallback:
    """Test activity fallback integration."""

    @pytest.fixture
    def agents_config(self):
        """Create an agents config with activities enabled."""
        from llmcore.config.agents_config import AgentsConfig
        config = AgentsConfig()
        config.activities.enabled = True
        config.activities.fallback_to_native_tools = True
        return config

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.default_model = "gemma3:4b"
        provider.get_name.return_value = "ollama"
        return provider

    @pytest.fixture
    def agent_state(self):
        """Create a mock agent state."""
        from llmcore.agents.cognitive.models import EnhancedAgentState
        return EnhancedAgentState(
            goal="Test goal",
            session_id="test-session",
        )

    @pytest.mark.asyncio
    async def test_activity_fallback_on_tool_error(
        self, agents_config, mock_provider, agent_state
    ):
        """Test that activity fallback activates on tool support error."""
        from llmcore.agents.cognitive.phases.think import think_phase
        from llmcore.agents.cognitive.models import ThinkInput

        # Mock provider to fail with tool error, then succeed without tools
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if 'tools' in kwargs and kwargs['tools']:
                raise Exception("Model gemma3:4b does not support tools")
            # Activity-based response
            return {
                "choices": [{
                    "message": {
                        "content": """I'll search for files using the activity system.

<activity_request>
    <activity>file_search</activity>
    <parameters>
        <pattern>*.py</pattern>
        <directory>/home</directory>
    </parameters>
    <reasoning>Searching for Python files</reasoning>
</activity_request>"""
                    }
                }],
                "usage": {"total_tokens": 100},
            }

        mock_provider.chat_completion = AsyncMock(side_effect=mock_chat)
        mock_provider.extract_response_content = MagicMock(
            return_value="""I'll search for files using the activity system.

<activity_request>
    <activity>file_search</activity>
    <parameters>
        <pattern>*.py</pattern>
        <directory>/home</directory>
    </parameters>
    <reasoning>Searching for Python files</reasoning>
</activity_request>"""
        )

        # Include tool definitions to trigger the tools path
        mock_tool_defs = [
            {"name": "file_search", "description": "Search files", "parameters": {}}
        ]

        think_input = ThinkInput(
            goal="Search for Python files",
            current_step="Find files",
            available_tools=mock_tool_defs,
        )

        # Mock managers
        mock_provider_manager = MagicMock()
        mock_provider_manager.get_provider.return_value = mock_provider

        mock_memory_manager = MagicMock()
        mock_tool_manager = MagicMock()
        # Return tool definitions so tools_param is not None
        mock_tool_manager.get_tool_definitions.return_value = mock_tool_defs

        output = await think_phase(
            agent_state=agent_state,
            think_input=think_input,
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            tool_manager=mock_tool_manager,
            agents_config=agents_config,
        )

        # Should have used activity fallback
        assert output.using_activity_fallback
        # Should have set working memory
        assert agent_state.get_working_memory("using_activity_fallback")

    @pytest.mark.asyncio
    async def test_activity_fallback_disabled(
        self, mock_provider, agent_state
    ):
        """Test that activity fallback doesn't activate when disabled."""
        from llmcore.config.agents_config import AgentsConfig
        from llmcore.agents.cognitive.phases.think import think_phase
        from llmcore.agents.cognitive.models import ThinkInput

        # Disable activities
        config = AgentsConfig()
        config.activities.enabled = False

        # Mock provider to fail with tool error
        mock_provider.chat_completion = AsyncMock(
            side_effect=Exception("Model does not support tools")
        )

        think_input = ThinkInput(
            goal="Search for files",
            current_step="Find files",
            available_tools=[],
        )

        mock_provider_manager = MagicMock()
        mock_provider_manager.get_provider.return_value = mock_provider

        output = await think_phase(
            agent_state=agent_state,
            think_input=think_input,
            provider_manager=mock_provider_manager,
            memory_manager=MagicMock(),
            tool_manager=MagicMock(),
            agents_config=config,
        )

        # Should have returned error output (not activity fallback)
        assert not output.using_activity_fallback
        assert "Error" in output.thought

    @pytest.mark.asyncio
    async def test_act_phase_with_activities(self, agent_state):
        """Test act_phase with activity fallback state set.

        Note: Full activity execution integration is a TODO.
        This test verifies the act_phase handles tool execution correctly
        when activity fallback state is present.
        """
        from llmcore.agents.cognitive.phases.act import act_phase
        from llmcore.agents.cognitive.models import ActInput, ValidationResult
        from llmcore.models import ToolCall, ToolResult

        # Set up activity fallback state (as would be set by think_phase)
        agent_state.set_working_memory("using_activity_fallback", True)
        agent_state.set_working_memory(
            "pending_activities_text",
            """<activity_request>
    <activity>final_answer</activity>
    <parameters>
        <r>The answer is 42</r>
    </parameters>
    <reasoning>Task complete</reasoning>
</activity_request>"""
        )

        act_input = ActInput(
            tool_call=ToolCall(
                id="test",
                name="activity:final_answer",
                arguments={"result": "The answer is 42"},
            ),
            validation_result=ValidationResult.APPROVED,
        )

        # Create a mock tool manager that handles the activity
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool = AsyncMock(
            return_value=ToolResult(
                tool_call_id="test",
                content="The answer is 42",
                is_error=False,
            )
        )

        output = await act_phase(
            agent_state=agent_state,
            act_input=act_input,
            tool_manager=mock_tool_manager,
        )

        # Verify tool execution happened
        assert output.success
        assert "42" in output.tool_result.content


class TestActivityPrompts:
    """Test activity prompt generation."""

    def test_activity_system_prompt(self):
        """Test that activity system prompt contains required elements."""
        from llmcore.agents.activities.prompts import ACTIVITY_SYSTEM_PROMPT

        # Should contain XML format instructions
        assert "<activity_request>" in ACTIVITY_SYSTEM_PROMPT
        assert "<activity>" in ACTIVITY_SYSTEM_PROMPT
        assert "<parameters>" in ACTIVITY_SYSTEM_PROMPT
        assert "<reasoning>" in ACTIVITY_SYSTEM_PROMPT

        # Should list available activities
        assert "file_read" in ACTIVITY_SYSTEM_PROMPT
        assert "file_write" in ACTIVITY_SYSTEM_PROMPT
        assert "execute_python" in ACTIVITY_SYSTEM_PROMPT
        assert "final_answer" in ACTIVITY_SYSTEM_PROMPT

    def test_generate_activity_prompt(self):
        """Test activity prompt generation."""
        from llmcore.agents.activities.prompts import generate_activity_prompt

        prompt = generate_activity_prompt(
            goal="Find Python files",
            current_step="Search directory",
            history="Previous: Listed directory",
            context="Working in /home/user",
        )

        assert "Find Python files" in prompt
        assert "Search directory" in prompt
        assert "Listed directory" in prompt
        assert "/home/user" in prompt
        assert "<activity_request>" in prompt


class TestActivityParser:
    """Test activity request parsing."""

    def test_parse_activity_request(self):
        """Test parsing activity request from LLM output."""
        from llmcore.agents.activities.parser import ActivityRequestParser

        parser = ActivityRequestParser()

        text = """I'll search for files.

<activity_request>
    <activity>file_search</activity>
    <parameters>
        <pattern>*.py</pattern>
        <directory>/home</directory>
    </parameters>
    <reasoning>Searching for Python files</reasoning>
</activity_request>
"""

        result = parser.parse(text)

        assert result.has_requests
        assert len(result.requests) == 1
        assert result.requests[0].activity == "file_search"
        assert result.requests[0].parameters.get("pattern") == "*.py"

    def test_parse_final_answer(self):
        """Test detecting final answer."""
        from llmcore.agents.activities.parser import ActivityRequestParser

        parser = ActivityRequestParser()

        text = """Task complete.

<activity_request>
    <activity>final_answer</activity>
    <parameters>
        <result>The calculation result is 42</result>
    </parameters>
    <reasoning>Providing final answer</reasoning>
</activity_request>
"""

        assert parser.is_final_answer(text)
        answer = parser.extract_final_answer(text)
        assert "42" in answer or answer is not None

    def test_parse_multiple_activities(self):
        """Test parsing multiple activity requests."""
        from llmcore.agents.activities.parser import ActivityRequestParser

        parser = ActivityRequestParser()

        text = """I'll do multiple things.

<activity_request>
    <activity>file_read</activity>
    <parameters>
        <path>/etc/hosts</path>
    </parameters>
</activity_request>

<activity_request>
    <activity>file_read</activity>
    <parameters>
        <path>/etc/passwd</path>
    </parameters>
</activity_request>
"""

        result = parser.parse(text)

        assert result.has_requests
        assert len(result.requests) == 2

    def test_parse_no_activities(self):
        """Test parsing text without activities."""
        from llmcore.agents.activities.parser import ActivityRequestParser

        parser = ActivityRequestParser()

        text = "Just some thinking without any activities..."

        result = parser.parse(text)

        assert not result.has_requests
        assert len(result.requests) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
