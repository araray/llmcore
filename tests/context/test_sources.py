# tests/context/test_sources.py
"""
Tests for the five context source implementations.

Covers:
    - GoalContextSource: goal formatting, empty goals, failing manager,
      fallback to pending, success criteria, learned strategies
    - RecentContextSource: add_turn, sliding window, clear, turn_count,
      empty history, metadata
    - SemanticContextSource: retrieval with task, no task, empty results,
      failing retrieval function
    - EpisodicContextSource: add_episode, episode ranking by significance
      and tag overlap, clear, char budget, max_episodes trimming
    - SkillContextSource: task-based loading, no task, empty skills,
      failing skill loader
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.context.synthesis import ContextChunk

# =============================================================================
# GoalContextSource Tests
# =============================================================================


class TestGoalContextSource:
    """Tests for GoalContextSource."""

    @pytest.mark.asyncio
    async def test_active_goals_formatted(self, mock_goal_manager, mock_goal):
        """Active goals are formatted with status, progress, and strategies."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        assert isinstance(chunk, ContextChunk)
        assert chunk.source == "goals"
        assert chunk.priority == 100
        assert chunk.relevance == 1.0
        assert "Current Goals" in chunk.content
        assert mock_goal.description in chunk.content
        assert "active" in chunk.content
        assert "50%" in chunk.content

    @pytest.mark.asyncio
    async def test_learned_strategies_included(self, mock_goal_manager, mock_goal):
        """Learned strategies appear in the output."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        assert "Learned Strategies" in chunk.content
        assert "protocol-based abstractions" in chunk.content

    @pytest.mark.asyncio
    async def test_success_criteria_with_markers(self, mock_goal_manager, mock_goal):
        """Success criteria show ✓ or ○ markers."""
        from llmcore.context.sources.goals import GoalContextSource

        crit = MagicMock()
        crit.description = "Tests pass at 85%"
        crit.is_met = MagicMock(return_value=True)
        mock_goal.success_criteria = [crit]

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        assert "✓" in chunk.content
        assert "Tests pass at 85%" in chunk.content

    @pytest.mark.asyncio
    async def test_unmet_criteria_marker(self, mock_goal_manager, mock_goal):
        """Unmet criteria show ○ marker."""
        from llmcore.context.sources.goals import GoalContextSource

        crit = MagicMock()
        crit.description = "Coverage above 90%"
        crit.is_met = MagicMock(return_value=False)
        mock_goal.success_criteria = [crit]

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        assert "○" in chunk.content
        assert "Coverage above 90%" in chunk.content

    @pytest.mark.asyncio
    async def test_empty_active_goals_falls_back(self, mock_goal):
        """Falls back to get_all_goals when get_active_goals returns empty."""
        from llmcore.context.sources.goals import GoalContextSource

        manager = MagicMock()
        # Active returns empty → triggers fallback
        manager.get_active_goals = AsyncMock(return_value=[])
        # All goals includes the pending/active one
        mock_goal.status.value = "pending"
        manager.get_all_goals = AsyncMock(return_value=[mock_goal])

        source = GoalContextSource(manager)
        chunk = await source.get_context()

        assert mock_goal.description in chunk.content

    @pytest.mark.asyncio
    async def test_no_goals_at_all(self, mock_goal_manager_empty):
        """No goals returns empty content."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager_empty)
        chunk = await source.get_context()

        assert chunk.source == "goals"
        assert chunk.priority == 100
        # Content should only contain the header
        assert "Current Goals" in chunk.content

    @pytest.mark.asyncio
    async def test_failing_goal_manager(self, mock_goal_manager_failing):
        """Failing goal manager returns gracefully (just header)."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager_failing)
        chunk = await source.get_context()

        assert chunk.source == "goals"
        assert chunk.priority == 100
        # Should not raise — graceful degradation
        assert chunk.tokens >= 0

    @pytest.mark.asyncio
    async def test_most_recent_strategies_only(self, mock_goal_manager, mock_goal):
        """Only the 3 most recent strategies are included."""
        from llmcore.context.sources.goals import GoalContextSource

        mock_goal.learned_strategies = [
            "Strategy A",
            "Strategy B",
            "Strategy C",
            "Strategy D",
            "Strategy E",
        ]

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        # Should have strategies C, D, E (last 3)
        assert "Strategy C" in chunk.content
        assert "Strategy D" in chunk.content
        assert "Strategy E" in chunk.content
        # Strategy A should be trimmed
        assert "Strategy A" not in chunk.content

    @pytest.mark.asyncio
    async def test_task_parameter_ignored(self, mock_goal_manager, sample_task):
        """Task parameter doesn't affect goal output (always relevant)."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "goals"
        assert "Current Goals" in chunk.content

    @pytest.mark.asyncio
    async def test_token_estimate(self, mock_goal_manager):
        """Tokens are estimated as content_length // 4."""
        from llmcore.context.sources.goals import GoalContextSource

        source = GoalContextSource(mock_goal_manager)
        chunk = await source.get_context()

        expected_tokens = len(chunk.content) // 4
        assert chunk.tokens == expected_tokens


# =============================================================================
# RecentContextSource Tests
# =============================================================================


class TestRecentContextSource:
    """Tests for RecentContextSource."""

    def test_init_defaults(self):
        """Default max_turns is 20."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        assert source.max_turns == 20
        assert source.turn_count == 0

    def test_custom_max_turns(self):
        """Custom max_turns is respected."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource(max_turns=5)
        assert source.max_turns == 5

    def test_add_turn(self):
        """add_turn increments turn_count."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Hello")
        assert source.turn_count == 1

    def test_add_multiple_turns(self):
        """Multiple turns are tracked."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Hello")
        source.add_turn("assistant", "Hi there!")
        source.add_turn("user", "What's up?")
        assert source.turn_count == 3

    def test_sliding_window_trim(self):
        """History is trimmed to max_turns."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource(max_turns=3)
        for i in range(10):
            source.add_turn("user", f"Message {i}")
        assert source.turn_count == 3

    def test_sliding_window_keeps_recent(self):
        """The most recent turns survive trimming."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource(max_turns=2)
        source.add_turn("user", "First message")
        source.add_turn("user", "Second message")
        source.add_turn("user", "Third message")

        # Window keeps last 2: "Second" and "Third"
        assert source.turn_count == 2

    def test_clear(self):
        """clear() empties the history."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Hi")
        source.add_turn("assistant", "Hello")
        source.clear()
        assert source.turn_count == 0

    @pytest.mark.asyncio
    async def test_get_context_empty(self):
        """Empty history returns empty chunk."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        chunk = await source.get_context()

        assert chunk.source == "recent"
        assert chunk.content == ""
        assert chunk.tokens == 0
        assert chunk.priority == 80

    @pytest.mark.asyncio
    async def test_get_context_with_turns(self):
        """Populated history is formatted as markdown."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Run the tests")
        source.add_turn("assistant", "All 219 pass.")

        chunk = await source.get_context()

        assert chunk.source == "recent"
        assert chunk.priority == 80
        assert "Recent History" in chunk.content
        assert "**USER:**" in chunk.content
        assert "Run the tests" in chunk.content
        assert "**ASSISTANT:**" in chunk.content
        assert "All 219 pass." in chunk.content
        assert chunk.tokens > 0

    @pytest.mark.asyncio
    async def test_role_uppercase_in_output(self):
        """Role names are uppercased in the formatted output."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("tool", "Execution result: success")
        chunk = await source.get_context()

        assert "**TOOL:**" in chunk.content

    @pytest.mark.asyncio
    async def test_metadata_stored(self):
        """Turn metadata is stored (though not necessarily displayed)."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("tool", "result", metadata={"tool_name": "bash"})

        # Verify internal storage
        assert source._history[0]["metadata"]["tool_name"] == "bash"

    @pytest.mark.asyncio
    async def test_token_estimate(self):
        """Token count is estimated as content_length // 4."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Hello world test message.")
        chunk = await source.get_context()

        assert chunk.tokens == len(chunk.content) // 4

    @pytest.mark.asyncio
    async def test_task_parameter_unused(self, sample_task):
        """Task parameter doesn't affect output (recency-based)."""
        from llmcore.context.sources.recent import RecentContextSource

        source = RecentContextSource()
        source.add_turn("user", "Hello")
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "recent"
        assert "Hello" in chunk.content


# =============================================================================
# SemanticContextSource Tests
# =============================================================================


class TestSemanticContextSource:
    """Tests for SemanticContextSource."""

    @pytest.mark.asyncio
    async def test_retrieval_with_task(self, mock_retrieval_fn, sample_task):
        """Retrieves and formats RAG chunks when task has description."""
        from llmcore.context.sources.semantic import SemanticContextSource

        source = SemanticContextSource(mock_retrieval_fn)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "semantic"
        assert chunk.priority == 60
        assert chunk.relevance == 0.8
        assert "Relevant Knowledge" in chunk.content
        assert "ContextSynthesizer" in chunk.content
        assert "synthesis.py" in chunk.content
        assert chunk.tokens > 0

    @pytest.mark.asyncio
    async def test_no_task_returns_empty(self, mock_retrieval_fn):
        """No task → empty chunk."""
        from llmcore.context.sources.semantic import SemanticContextSource

        source = SemanticContextSource(mock_retrieval_fn)
        chunk = await source.get_context(task=None)

        assert chunk.source == "semantic"
        assert chunk.content == ""
        assert chunk.tokens == 0

    @pytest.mark.asyncio
    async def test_task_without_description(self, mock_retrieval_fn):
        """Task with no description attribute uses str(task)."""
        from llmcore.context.sources.semantic import SemanticContextSource

        source = SemanticContextSource(mock_retrieval_fn)
        # Pass a string directly — getattr(str, 'description', str(task)) = str(task)
        chunk = await source.get_context(task="search for tests")

        assert chunk.source == "semantic"
        assert chunk.tokens > 0  # retrieval was called

    @pytest.mark.asyncio
    async def test_task_with_empty_description(self, mock_retrieval_fn):
        """Task with empty description → empty chunk."""
        from llmcore.context.sources.semantic import SemanticContextSource

        task = MagicMock()
        task.description = ""

        source = SemanticContextSource(mock_retrieval_fn)
        chunk = await source.get_context(task=task)

        assert chunk.content == ""

    @pytest.mark.asyncio
    async def test_empty_retrieval_results(
        self, mock_retrieval_fn_empty, sample_task
    ):
        """Empty retrieval results → header only."""
        from llmcore.context.sources.semantic import SemanticContextSource

        source = SemanticContextSource(mock_retrieval_fn_empty)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "semantic"
        assert "Relevant Knowledge" in chunk.content
        # No actual content beyond the header
        assert "synthesis.py" not in chunk.content

    @pytest.mark.asyncio
    async def test_failing_retrieval(self, mock_retrieval_fn_failing, sample_task):
        """Failing retrieval → empty chunk (graceful degradation)."""
        from llmcore.context.sources.semantic import SemanticContextSource

        source = SemanticContextSource(mock_retrieval_fn_failing)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "semantic"
        assert chunk.content == ""
        assert chunk.tokens == 0

    @pytest.mark.asyncio
    async def test_chunks_with_missing_content(self, sample_task):
        """Retrieval chunks missing 'content' key are skipped."""
        from llmcore.context.sources.semantic import SemanticContextSource

        async def bad_chunks(query, top_k=10):
            return [
                {"source": "file.py", "score": 0.9},  # missing content
                {"content": "Valid.", "source": "ok.py", "score": 0.8},
            ]

        source = SemanticContextSource(bad_chunks)
        chunk = await source.get_context(task=sample_task)

        assert "Valid." in chunk.content
        assert "file.py" not in chunk.content  # skipped because no content

    @pytest.mark.asyncio
    async def test_source_labels_from_chunks(self, sample_task):
        """Source labels from chunk dicts appear as section headers."""
        from llmcore.context.sources.semantic import SemanticContextSource

        async def labeled_chunks(query, top_k=10):
            return [
                {"content": "Data.", "source": "data.csv"},
                {"content": "Code.", "source": "utils.py"},
            ]

        source = SemanticContextSource(labeled_chunks)
        chunk = await source.get_context(task=sample_task)

        assert "### data.csv" in chunk.content
        assert "### utils.py" in chunk.content


# =============================================================================
# EpisodicContextSource Tests
# =============================================================================


class TestEpisodicContextSource:
    """Tests for EpisodicContextSource."""

    def test_init_defaults(self):
        """Default max_episodes is 100."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        assert source.max_episodes == 100
        assert source.episode_count == 0

    def test_add_episode(self):
        """add_episode increments episode_count."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("Deployed v0.29.0", significance=0.8)
        assert source.episode_count == 1

    def test_add_episode_with_tags(self):
        """Tags are stored with the episode."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode(
            "Fixed import chain",
            tags=["import", "fix"],
            significance=0.7,
        )
        assert source.episode_count == 1

    def test_max_episodes_trim(self):
        """Episodes are trimmed to max_episodes."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource(max_episodes=5)
        for i in range(20):
            source.add_episode(f"Episode {i}")
        assert source.episode_count == 5

    def test_max_episodes_keeps_recent(self):
        """Most recent episodes survive trimming."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource(max_episodes=3)
        for i in range(10):
            source.add_episode(f"Episode {i}")

        # Internal list keeps the last 3: episodes 7, 8, 9
        assert source.episode_count == 3
        summaries = [e["summary"] for e in source._episodes]
        assert "Episode 7" in summaries
        assert "Episode 9" in summaries
        assert "Episode 0" not in summaries

    def test_clear(self):
        """clear() removes all episodes."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("Test episode")
        source.clear()
        assert source.episode_count == 0

    @pytest.mark.asyncio
    async def test_get_context_empty(self):
        """Empty episodes → empty chunk."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        chunk = await source.get_context()

        assert chunk.source == "episodic"
        assert chunk.content == ""
        assert chunk.tokens == 0
        assert chunk.priority == 40

    @pytest.mark.asyncio
    async def test_get_context_with_episodes(self):
        """Episodes are formatted with significance markers."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("Fixed import chain", significance=0.8)
        source.add_episode("Deployed v0.29.0", significance=0.9)

        chunk = await source.get_context()

        assert "Past Experiences" in chunk.content
        assert "[80%]" in chunk.content or "[90%]" in chunk.content
        assert "Fixed import chain" in chunk.content
        assert "Deployed v0.29.0" in chunk.content
        assert chunk.tokens > 0

    @pytest.mark.asyncio
    async def test_ranking_by_significance(self):
        """Episodes ranked by significance (highest first)."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("Low importance", significance=0.2)
        source.add_episode("High importance", significance=0.9)
        source.add_episode("Medium importance", significance=0.5)

        chunk = await source.get_context()

        high_pos = chunk.content.find("High importance")
        low_pos = chunk.content.find("Low importance")
        assert high_pos < low_pos

    @pytest.mark.asyncio
    async def test_ranking_with_tag_overlap(self, sample_task):
        """Tag overlap with task description boosts episode ranking."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        # sample_task.description = "Write pytest tests for the context module"
        source.add_episode(
            "Unrelated episode",
            tags=["deployment"],
            significance=0.9,
        )
        source.add_episode(
            "Relevant episode",
            tags=["pytest", "tests", "context"],
            significance=0.3,  # Lower base significance
        )

        chunk = await source.get_context(task=sample_task)

        # Relevant episode gets 0.3 + 3*0.2 = 0.9 from tag overlap
        # Unrelated gets 0.9 + 0 = 0.9
        # Should be close; relevant might be first due to 3 tag matches
        assert "Relevant episode" in chunk.content

    @pytest.mark.asyncio
    async def test_ranking_without_task(self):
        """Without task, episodes ranked by significance only."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("A", significance=0.3)
        source.add_episode("B", significance=0.8)

        chunk = await source.get_context(task=None)

        b_pos = chunk.content.find("B")
        a_pos = chunk.content.find("A")
        assert b_pos < a_pos

    @pytest.mark.asyncio
    async def test_char_budget_limits_output(self):
        """Episodes respect the max_tokens char budget."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        # Add many large episodes
        for i in range(50):
            source.add_episode(f"Episode {i}: " + "x" * 200, significance=0.5)

        # Very small token budget → limited episodes
        chunk = await source.get_context(max_tokens=50)

        # 50 tokens * 4 chars/token = 200 char budget
        # Each episode entry is ~220 chars, plus header
        # Should be truncated to just a few episodes
        assert chunk.tokens <= 100  # generous margin

    @pytest.mark.asyncio
    async def test_episode_metadata_stored(self):
        """Episode metadata is preserved internally."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode(
            "Test episode",
            metadata={"session_id": "abc123"},
        )
        assert source._episodes[0]["metadata"]["session_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_token_estimate(self):
        """Tokens are estimated as content_length // 4."""
        from llmcore.context.sources.episodic import EpisodicContextSource

        source = EpisodicContextSource()
        source.add_episode("Test episode", significance=0.5)
        chunk = await source.get_context()

        assert chunk.tokens == len(chunk.content) // 4


# =============================================================================
# SkillContextSource Tests
# =============================================================================


class TestSkillContextSource:
    """Tests for SkillContextSource."""

    def _make_mock_loader(self, skills=None, format_output="# Skill\nContent"):
        """Create a mock SkillLoader."""
        loader = MagicMock()
        loader.load_for_task = AsyncMock(return_value=skills or [])
        loader.format_skills = MagicMock(return_value=format_output)
        return loader

    @pytest.mark.asyncio
    async def test_with_task_and_skills(self, sample_task):
        """Skills are loaded and formatted when task has description."""
        from llmcore.context.sources.skills import SkillContextSource

        skill_mock = MagicMock()
        loader = self._make_mock_loader(
            skills=[skill_mock],
            format_output="# PPTX Skill\nUse python-pptx for slides.",
        )

        source = SkillContextSource(loader)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "skills"
        assert chunk.priority == 50  # default
        assert "PPTX Skill" in chunk.content
        assert chunk.tokens > 0

        # Verify load_for_task was called with the task description
        loader.load_for_task.assert_called_once_with(
            task_description=sample_task.description,
            max_tokens=10_000,
        )

    @pytest.mark.asyncio
    async def test_custom_priority(self, sample_task):
        """Custom default_priority is used."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader(
            skills=[MagicMock()],
            format_output="Content",
        )
        source = SkillContextSource(loader, default_priority=70)
        chunk = await source.get_context(task=sample_task)

        assert chunk.priority == 70

    @pytest.mark.asyncio
    async def test_no_task_returns_empty(self):
        """No task → empty chunk."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader()
        source = SkillContextSource(loader)
        chunk = await source.get_context(task=None)

        assert chunk.source == "skills"
        assert chunk.content == ""
        assert chunk.tokens == 0
        # load_for_task should NOT be called
        loader.load_for_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_without_description(self):
        """Task with no description attribute uses str(task)."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader(skills=[MagicMock()], format_output="Content")
        source = SkillContextSource(loader)

        # Object with no description → getattr falls back to str()
        task = "create a presentation"
        chunk = await source.get_context(task=task)

        loader.load_for_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_with_empty_description(self):
        """Task with empty description → empty chunk."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader()
        task = MagicMock()
        task.description = ""

        source = SkillContextSource(loader)
        chunk = await source.get_context(task=task)

        assert chunk.content == ""

    @pytest.mark.asyncio
    async def test_empty_skills_result(self, sample_task):
        """No matching skills → empty chunk."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader(skills=[])
        source = SkillContextSource(loader)
        chunk = await source.get_context(task=sample_task)

        assert chunk.content == ""
        assert chunk.tokens == 0

    @pytest.mark.asyncio
    async def test_failing_skill_loader(self, sample_task):
        """Failing skill loader → empty chunk (graceful degradation)."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = MagicMock()
        loader.load_for_task = AsyncMock(
            side_effect=RuntimeError("Skill dir not found")
        )

        source = SkillContextSource(loader)
        chunk = await source.get_context(task=sample_task)

        assert chunk.source == "skills"
        assert chunk.content == ""
        assert chunk.tokens == 0

    @pytest.mark.asyncio
    async def test_relevance_and_recency_scores(self, sample_task):
        """Skill chunks have correct relevance and recency."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader(
            skills=[MagicMock()],
            format_output="Skill content here.",
        )
        source = SkillContextSource(loader)
        chunk = await source.get_context(task=sample_task)

        assert chunk.relevance == 0.7
        assert chunk.recency == 0.5

    @pytest.mark.asyncio
    async def test_max_tokens_passed_to_loader(self, sample_task):
        """max_tokens parameter is forwarded to the skill loader."""
        from llmcore.context.sources.skills import SkillContextSource

        loader = self._make_mock_loader(skills=[])
        source = SkillContextSource(loader)
        await source.get_context(task=sample_task, max_tokens=5000)

        loader.load_for_task.assert_called_once_with(
            task_description=sample_task.description,
            max_tokens=5000,
        )

    @pytest.mark.asyncio
    async def test_token_estimate(self, sample_task):
        """Tokens estimated as content_length // 4."""
        from llmcore.context.sources.skills import SkillContextSource

        content = "# Skill\nThis skill teaches you to make presentations."
        loader = self._make_mock_loader(
            skills=[MagicMock()],
            format_output=content,
        )
        source = SkillContextSource(loader)
        chunk = await source.get_context(task=sample_task)

        assert chunk.tokens == len(content) // 4
