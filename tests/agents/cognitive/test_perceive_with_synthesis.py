# tests/agents/cognitive/test_perceive_with_synthesis.py
"""
Integration tests for PERCEIVE phase with ContextSynthesizer.

These tests verify that the perceive_phase function correctly integrates
with the ContextSynthesizer for multi-source context assembly.

Test Coverage:
    - Legacy mode (backward compatibility with memory_manager)
    - Synthesis mode with all 5 context sources
    - Graceful degradation when synthesis fails
    - Factory function for default synthesizer configuration

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
    - Roadmap Task 3.9 (Darwin Integration)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.agents.cognitive.models import (
    EnhancedAgentState,
    PerceiveInput,
    PerceiveOutput,
)
from llmcore.agents.cognitive.phases.perceive import (
    _parse_synthesized_content,
    _TaskProxy,
    create_default_synthesizer,
    perceive_phase,
)
from llmcore.context import ContextChunk, ContextSynthesizer, SynthesizedContext

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def agent_state() -> EnhancedAgentState:
    """Create a minimal EnhancedAgentState for testing."""
    return EnhancedAgentState(
        goal="Test goal: implement feature X",
        plan=[],
        working_memory={"key1": "value1", "key2": 42},
    )


@pytest.fixture
def perceive_input() -> PerceiveInput:
    """Create a PerceiveInput for testing."""
    return PerceiveInput(
        goal="Test goal: implement feature X",
        context_query=None,
        force_refresh=False,
    )


@pytest.fixture
def mock_memory_manager() -> AsyncMock:
    """Create a mock memory manager."""
    mm = AsyncMock()
    mm.retrieve_relevant_context = AsyncMock(
        return_value=[
            MagicMock(content="Legacy context item 1"),
            MagicMock(content="Legacy context item 2"),
        ]
    )
    return mm


@pytest.fixture
def mock_synthesizer() -> MagicMock:
    """Create a mock ContextSynthesizer."""
    synth = MagicMock(spec=ContextSynthesizer)
    synth.synthesize = AsyncMock(
        return_value=SynthesizedContext(
            content="## GOALS\n\nGoal content here\n\n---\n\n## RECENT\n\nRecent history",
            total_tokens=500,
            max_tokens=100_000,
            sources_included=["goals", "recent"],
            sources_truncated=[],
            compression_applied=False,
            synthesis_time_ms=10.5,
        )
    )
    return synth


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock sandbox provider."""
    sandbox = MagicMock()
    sandbox.get_info.return_value = {
        "provider": "docker",
        "status": "running",
        "access_level": "full",
    }
    return sandbox


# =============================================================================
# LEGACY MODE TESTS (Backward Compatibility)
# =============================================================================


class TestPerceivePhaseLegacyMode:
    """Tests for perceive_phase in legacy mode (without synthesizer)."""

    @pytest.mark.asyncio
    async def test_legacy_mode_retrieves_context(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Legacy mode should use memory_manager for context retrieval."""
        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
        )

        # Verify memory_manager was called
        mock_memory_manager.retrieve_relevant_context.assert_called_once()

        # Verify output structure
        assert isinstance(output, PerceiveOutput)
        assert len(output.retrieved_context) == 2
        assert "Legacy context item 1" in output.retrieved_context
        assert "Legacy context item 2" in output.retrieved_context

    @pytest.mark.asyncio
    async def test_legacy_mode_captures_working_memory(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Legacy mode should capture working memory snapshot."""
        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
        )

        assert output.working_memory_snapshot == {"key1": "value1", "key2": 42}

    @pytest.mark.asyncio
    async def test_legacy_mode_captures_environmental_state(
        self, agent_state, perceive_input, mock_memory_manager, mock_sandbox
    ):
        """Legacy mode should capture environmental state including sandbox."""
        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
            sandbox=mock_sandbox,
        )

        assert output.environmental_state["has_sandbox"] is True
        assert output.environmental_state["sandbox_provider"] == "docker"
        assert output.environmental_state["sandbox_status"] == "running"

    @pytest.mark.asyncio
    async def test_legacy_mode_handles_memory_failure_gracefully(self, agent_state, perceive_input):
        """Legacy mode should return empty context on memory failure."""
        mm = AsyncMock()
        mm.retrieve_relevant_context = AsyncMock(side_effect=Exception("Memory retrieval failed"))

        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mm,
        )

        # Should return empty context, not raise
        assert output.retrieved_context == []


# =============================================================================
# SYNTHESIS MODE TESTS
# =============================================================================


class TestPerceivePhaseSynthesisMode:
    """Tests for perceive_phase with ContextSynthesizer."""

    @pytest.mark.asyncio
    async def test_synthesis_mode_uses_synthesizer(
        self, agent_state, perceive_input, mock_memory_manager, mock_synthesizer
    ):
        """Synthesis mode should use synthesizer instead of memory_manager."""
        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
            context_synthesizer=mock_synthesizer,
        )

        # Verify synthesizer was called
        mock_synthesizer.synthesize.assert_called_once()

        # Verify memory_manager was NOT called
        mock_memory_manager.retrieve_relevant_context.assert_not_called()

        # Verify output contains synthesized content
        assert len(output.retrieved_context) == 2
        assert any("GOALS" in ctx for ctx in output.retrieved_context)
        assert any("RECENT" in ctx for ctx in output.retrieved_context)

    @pytest.mark.asyncio
    async def test_synthesis_mode_passes_task_description(
        self, agent_state, perceive_input, mock_memory_manager, mock_synthesizer
    ):
        """Synthesis mode should pass task with description to synthesizer."""
        await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
            context_synthesizer=mock_synthesizer,
        )

        # Get the task that was passed to synthesize
        call_args = mock_synthesizer.synthesize.call_args
        task = call_args.kwargs.get("current_task") or call_args.args[0] if call_args.args else None

        # If task wasn't passed as positional arg, check kwargs
        if task is None and call_args.kwargs:
            task = call_args.kwargs.get("current_task")

        assert task is not None
        assert hasattr(task, "description")
        assert task.description == perceive_input.goal

    @pytest.mark.asyncio
    async def test_synthesis_mode_uses_context_query_when_provided(
        self, agent_state, mock_memory_manager, mock_synthesizer
    ):
        """Synthesis mode should use context_query as description when provided."""
        input_with_query = PerceiveInput(
            goal="Test goal",
            context_query="Specific search query",
            force_refresh=False,
        )

        await perceive_phase(
            agent_state=agent_state,
            perceive_input=input_with_query,
            memory_manager=mock_memory_manager,
            context_synthesizer=mock_synthesizer,
        )

        call_args = mock_synthesizer.synthesize.call_args
        task = call_args.kwargs.get("current_task")
        assert task.description == "Specific search query"

    @pytest.mark.asyncio
    async def test_synthesis_mode_handles_empty_content(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Synthesis mode should handle empty synthesized content."""
        synth = MagicMock(spec=ContextSynthesizer)
        synth.synthesize = AsyncMock(
            return_value=SynthesizedContext(
                content="",
                total_tokens=0,
                max_tokens=100_000,
                sources_included=[],
            )
        )

        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
            context_synthesizer=synth,
        )

        assert output.retrieved_context == []

    @pytest.mark.asyncio
    async def test_synthesis_mode_handles_failure_gracefully(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Synthesis mode should return empty context on synthesis failure."""
        synth = MagicMock(spec=ContextSynthesizer)
        synth.synthesize = AsyncMock(side_effect=Exception("Synthesis failed"))

        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
            context_synthesizer=synth,
        )

        # Should return empty context, not raise
        assert output.retrieved_context == []


# =============================================================================
# CONTENT PARSING TESTS
# =============================================================================


class TestParseSynthesizedContent:
    """Tests for _parse_synthesized_content helper function."""

    def test_parse_single_section(self):
        """Should parse a single section correctly."""
        content = "## GOALS\n\nGoal content here"
        result = _parse_synthesized_content(content)

        assert len(result) == 1
        assert "GOALS" in result[0]
        assert "Goal content here" in result[0]

    def test_parse_multiple_sections(self):
        """Should split on section dividers."""
        content = "## GOALS\n\nGoal content\n\n---\n\n## RECENT\n\nRecent content"
        result = _parse_synthesized_content(content)

        assert len(result) == 2
        assert any("GOALS" in s for s in result)
        assert any("RECENT" in s for s in result)

    def test_parse_empty_content(self):
        """Should return empty list for empty content."""
        assert _parse_synthesized_content("") == []
        assert _parse_synthesized_content("   ") == []

    def test_parse_content_with_extra_whitespace(self):
        """Should handle extra whitespace gracefully."""
        content = "## GOALS\n\nContent\n\n---\n\n\n\n---\n\n## RECENT\n\nMore"
        result = _parse_synthesized_content(content)

        # Should skip empty sections between dividers
        non_empty = [s for s in result if s.strip()]
        assert len(non_empty) == 2


# =============================================================================
# TASK PROXY TESTS
# =============================================================================


class TestTaskProxy:
    """Tests for _TaskProxy helper class."""

    def test_task_proxy_has_description(self):
        """TaskProxy should have description attribute."""
        proxy = _TaskProxy(description="Test desc", goal="Test goal")
        assert proxy.description == "Test desc"

    def test_task_proxy_has_goal(self):
        """TaskProxy should have goal attribute."""
        proxy = _TaskProxy(description="Test desc", goal="Test goal")
        assert proxy.goal == "Test goal"

    def test_task_proxy_str_returns_description(self):
        """str(TaskProxy) should return description."""
        proxy = _TaskProxy(description="Test desc", goal="Test goal")
        assert str(proxy) == "Test desc"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateDefaultSynthesizer:
    """Tests for create_default_synthesizer factory function."""

    def test_creates_synthesizer_with_defaults(self):
        """Should create synthesizer with specified max_tokens."""
        synth = create_default_synthesizer(max_tokens=50_000)

        assert isinstance(synth, ContextSynthesizer)
        assert synth.max_tokens == 50_000

    def test_registers_recent_source_by_default(self):
        """Should register RecentContextSource by default."""
        synth = create_default_synthesizer()

        priorities = synth.list_sources()
        assert "recent" in priorities
        assert priorities["recent"] == 80

    def test_registers_episodic_source_by_default(self):
        """Should register EpisodicContextSource by default."""
        synth = create_default_synthesizer()

        priorities = synth.list_sources()
        assert "episodic" in priorities
        assert priorities["episodic"] == 40

    def test_registers_goal_source_when_goal_manager_provided(self):
        """Should register GoalContextSource when goal_manager is provided."""
        mock_gm = MagicMock()
        synth = create_default_synthesizer(goal_manager=mock_gm)

        priorities = synth.list_sources()
        assert "goals" in priorities
        assert priorities["goals"] == 100

    def test_does_not_register_goal_source_without_goal_manager(self):
        """Should not register GoalContextSource when goal_manager is None."""
        synth = create_default_synthesizer(goal_manager=None)

        priorities = synth.list_sources()
        assert "goals" not in priorities

    def test_registers_skill_source_when_skill_loader_provided(self):
        """Should register SkillContextSource when skill_loader is provided."""
        mock_sl = MagicMock()
        synth = create_default_synthesizer(skill_loader=mock_sl)

        priorities = synth.list_sources()
        assert "skills" in priorities
        assert priorities["skills"] == 50

    def test_registers_semantic_source_when_retrieval_fn_provided(self):
        """Should register SemanticContextSource when retrieval_fn is provided."""

        async def mock_retrieval(query: str, top_k: int = 10):
            return []

        synth = create_default_synthesizer(retrieval_fn=mock_retrieval)

        priorities = synth.list_sources()
        assert "semantic" in priorities
        assert priorities["semantic"] == 60

    def test_respects_compression_threshold(self):
        """Should set compression_threshold correctly."""
        synth = create_default_synthesizer(compression_threshold=0.9)
        assert synth.compression_threshold == 0.9


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPerceivePhaseIntegration:
    """Integration tests using real ContextSynthesizer with mock sources."""

    @pytest.mark.asyncio
    async def test_full_synthesis_with_real_synthesizer(self, agent_state, perceive_input):
        """Full integration test with real ContextSynthesizer and mock sources."""
        # Create real synthesizer
        synth = ContextSynthesizer(max_tokens=10_000)

        # Create mock sources that return real ContextChunks
        class MockGoalSource:
            async def get_context(self, task=None, max_tokens=2000):
                return ContextChunk(
                    source="goals",
                    content="# Goals\n\nComplete the test implementation",
                    tokens=50,
                    priority=100,
                    relevance=1.0,
                    recency=1.0,
                )

        class MockRecentSource:
            async def get_context(self, task=None, max_tokens=2000):
                return ContextChunk(
                    source="recent",
                    content="# Recent\n\nUser: Run the tests\nAssistant: Running...",
                    tokens=40,
                    priority=80,
                    relevance=0.9,
                    recency=1.0,
                )

        # Register sources
        synth.add_source("goals", MockGoalSource(), priority=100)
        synth.add_source("recent", MockRecentSource(), priority=80)

        # Mock memory manager (shouldn't be used)
        mm = AsyncMock()
        mm.retrieve_relevant_context = AsyncMock(return_value=[])

        # Run perceive phase
        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mm,
            context_synthesizer=synth,
        )

        # Verify output
        assert len(output.retrieved_context) == 2
        assert any("Goals" in ctx for ctx in output.retrieved_context)
        assert any("Recent" in ctx for ctx in output.retrieved_context)

        # Memory manager should NOT have been called
        mm.retrieve_relevant_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesis_with_empty_sources(self, agent_state, perceive_input):
        """Synthesizer should handle sources that return empty content."""
        synth = ContextSynthesizer(max_tokens=10_000)

        class EmptySource:
            async def get_context(self, task=None, max_tokens=2000):
                return ContextChunk(
                    source="empty",
                    content="",
                    tokens=0,
                    priority=50,
                )

        synth.add_source("empty", EmptySource(), priority=50)

        mm = AsyncMock()
        mm.retrieve_relevant_context = AsyncMock(return_value=[])

        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mm,
            context_synthesizer=synth,
        )

        # Should handle gracefully (empty sources filtered out)
        assert isinstance(output, PerceiveOutput)

    @pytest.mark.asyncio
    async def test_perceived_at_timestamp_is_set(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Output should have perceived_at timestamp."""
        before = datetime.utcnow()

        output = await perceive_phase(
            agent_state=agent_state,
            perceive_input=perceive_input,
            memory_manager=mock_memory_manager,
        )

        after = datetime.utcnow()

        assert output.perceived_at >= before
        assert output.perceived_at <= after


# =============================================================================
# TRACING TESTS
# =============================================================================


class TestPerceivePhaseTracing:
    """Tests for OpenTelemetry tracing in perceive_phase."""

    @pytest.mark.asyncio
    async def test_tracing_attributes_legacy_mode(
        self, agent_state, perceive_input, mock_memory_manager
    ):
        """Should add correct tracing attributes in legacy mode."""
        # The create_span is imported inside the function from llmcore.tracing
        # We need to patch it at the source
        with patch("llmcore.tracing.create_span") as mock_create:
            mock_span = MagicMock()
            mock_create.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_create.return_value.__exit__ = MagicMock(return_value=None)

            with patch("llmcore.tracing.add_span_attributes") as mock_add:
                await perceive_phase(
                    agent_state=agent_state,
                    perceive_input=perceive_input,
                    memory_manager=mock_memory_manager,
                    tracer=MagicMock(),  # Provide a tracer to trigger span creation
                )

                # Verify add_span_attributes was called
                mock_add.assert_called_once()
                call_args = mock_add.call_args[0]
                attrs = call_args[1]

                assert "perceive.context_items" in attrs
                assert "perceive.synthesis_mode" in attrs
                assert attrs["perceive.synthesis_mode"] is False

    @pytest.mark.asyncio
    async def test_synthesis_mode_tracing_attributes(
        self, agent_state, perceive_input, mock_memory_manager, mock_synthesizer
    ):
        """Should set synthesis_mode=True in tracing attributes when using synthesizer."""
        with patch("llmcore.tracing.create_span") as mock_create:
            mock_span = MagicMock()
            mock_create.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_create.return_value.__exit__ = MagicMock(return_value=None)

            with patch("llmcore.tracing.add_span_attributes") as mock_add:
                await perceive_phase(
                    agent_state=agent_state,
                    perceive_input=perceive_input,
                    memory_manager=mock_memory_manager,
                    context_synthesizer=mock_synthesizer,
                    tracer=MagicMock(),
                )

                mock_add.assert_called_once()
                call_args = mock_add.call_args[0]
                attrs = call_args[1]

                assert attrs["perceive.synthesis_mode"] is True
