# tests/autonomous/test_skills.py
"""
Tests for the Skill Loading System.

Covers:
    - SkillMetadata / Skill data models
    - SkillLoader initialization
    - Directory scanning (add_skill_directory, load_metadata)
    - Metadata extraction from YAML frontmatter, headings, implicit tags
    - Task-based skill selection (_score_skill, load_for_task)
    - LRU content caching with eviction
    - Section parsing (_parse_sections, ## headings)
    - Skill formatting (format_skills, include_sections)
    - get_available_skills introspection
    - Frontmatter extraction helpers (_extract_list_field, _extract_int_field)
    - Edge cases: missing directories, malformed YAML, empty task
"""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.autonomous.skills import (
    Skill,
    SkillLoader,
    SkillMetadata,
    _extract_int_field,
    _extract_list_field,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_skill_dir(tmp_path):
    """Create a temporary skill directory with sample skill files."""
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()

    # Python testing skill with full frontmatter
    python_dir = skill_dir / "python"
    python_dir.mkdir()
    (python_dir / "testing.md").write_text(
        textwrap.dedent("""\
            ---
            tags: [python, testing, pytest]
            triggers: [test, pytest, unittest, assertion]
            priority: 70
            ---
            # Python Testing Skill

            ## When to Use
            Use this skill when writing or debugging Python tests.

            ## Best Practices
            - Use pytest over unittest
            - Write descriptive test names
            - Use fixtures for setup

            ## Common Patterns
            Parametrize tests for multiple inputs.
        """),
        encoding="utf-8",
    )

    # Async skill — no frontmatter
    (python_dir / "async.md").write_text(
        textwrap.dedent("""\
            # Async Programming Skill

            ## When to Use
            Use when working with asyncio or async/await.

            ## Key Concepts
            - Event loop
            - Coroutines
            - Tasks and futures
        """),
        encoding="utf-8",
    )

    # Git workflow skill
    git_dir = skill_dir / "git"
    git_dir.mkdir()
    (git_dir / "workflow.md").write_text(
        textwrap.dedent("""\
            ---
            tags: [git, version-control]
            triggers: [commit, branch, merge, rebase]
            priority: 60
            ---
            # Git Workflow Skill

            ## Branching Strategy
            Use feature branches.

            ## Commit Messages
            Follow Conventional Commits format.
        """),
        encoding="utf-8",
    )

    return skill_dir


@pytest.fixture
def loader():
    """Create a fresh SkillLoader."""
    return SkillLoader(cache_size=10)


@pytest.fixture
async def loaded_loader(loader, tmp_skill_dir):
    """SkillLoader with skills already loaded from temp directory."""
    loader.add_skill_directory(tmp_skill_dir)
    await loader.load_metadata()
    return loader


# =============================================================================
# Data Model Tests
# =============================================================================


class TestSkillMetadata:
    """Tests for the SkillMetadata dataclass."""

    def test_defaults(self):
        """Default values for optional fields."""
        meta = SkillMetadata(name="test", path=Path("/tmp/test.md"))
        assert meta.tags == []
        assert meta.triggers == []
        assert meta.priority == 50
        assert meta.token_estimate == 0

    def test_full_construction(self):
        """All fields are populated correctly."""
        meta = SkillMetadata(
            name="Python Testing",
            path=Path("/skills/python/testing.md"),
            tags=["python", "testing"],
            triggers=["pytest", "test"],
            priority=70,
            token_estimate=250,
        )
        assert meta.name == "Python Testing"
        assert meta.priority == 70
        assert "pytest" in meta.triggers


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_construction(self):
        """Skill with metadata, content, and sections."""
        meta = SkillMetadata(name="test", path=Path("/tmp/test.md"))
        skill = Skill(
            metadata=meta,
            content="# Test\n\n## Section\nContent",
            sections={"section": "Content"},
        )
        assert skill.metadata is meta
        assert "## Section" in skill.content
        assert "section" in skill.sections

    def test_default_sections(self):
        """Default sections is empty dict."""
        meta = SkillMetadata(name="test", path=Path("/tmp/test.md"))
        skill = Skill(metadata=meta, content="Content")
        assert skill.sections == {}


# =============================================================================
# SkillLoader Initialization
# =============================================================================


class TestSkillLoaderInit:
    """Tests for SkillLoader initialization."""

    def test_default_cache_size(self):
        """Default cache_size is 50."""
        loader = SkillLoader()
        assert loader.cache_size == 50

    def test_custom_cache_size(self):
        """Custom cache_size is respected."""
        loader = SkillLoader(cache_size=100)
        assert loader.cache_size == 100

    def test_initial_state(self):
        """Loader starts with no dirs, no metadata, not loaded."""
        loader = SkillLoader()
        assert loader._skill_dirs == []
        assert loader._metadata_cache == {}
        assert loader._content_cache == {}
        assert loader._loaded is False


# =============================================================================
# Directory Management
# =============================================================================


class TestDirectoryManagement:
    """Tests for add_skill_directory."""

    def test_add_valid_directory(self, loader, tmp_skill_dir):
        """Valid directory is added to the search list."""
        loader.add_skill_directory(tmp_skill_dir)
        assert len(loader._skill_dirs) == 1
        assert loader._loaded is False  # not loaded yet

    def test_add_duplicate_directory(self, loader, tmp_skill_dir):
        """Adding the same directory twice is idempotent."""
        loader.add_skill_directory(tmp_skill_dir)
        loader.add_skill_directory(tmp_skill_dir)
        assert len(loader._skill_dirs) == 1

    def test_add_nonexistent_directory(self, loader, tmp_path):
        """Non-existent directory is silently ignored."""
        loader.add_skill_directory(tmp_path / "nonexistent")
        assert len(loader._skill_dirs) == 0

    def test_add_file_as_directory(self, loader, tmp_path):
        """A file path (not dir) is ignored."""
        f = tmp_path / "file.txt"
        f.write_text("hello")
        loader.add_skill_directory(f)
        assert len(loader._skill_dirs) == 0

    def test_add_multiple_directories(self, loader, tmp_path):
        """Multiple valid directories are tracked."""
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        loader.add_skill_directory(d1)
        loader.add_skill_directory(d2)
        assert len(loader._skill_dirs) == 2

    def test_tilde_expansion(self, loader, tmp_path):
        """~ in path is expanded."""
        with patch("pathlib.Path.expanduser", return_value=tmp_path):
            with patch("pathlib.Path.resolve", return_value=tmp_path):
                # This is tricky to test — just verify it doesn't crash
                loader.add_skill_directory("~/skills")

    def test_adding_dir_resets_loaded_flag(self, loader, tmp_skill_dir):
        """Adding a new directory marks metadata as stale."""
        loader._loaded = True
        loader.add_skill_directory(tmp_skill_dir)
        assert loader._loaded is False


# =============================================================================
# Metadata Loading
# =============================================================================


class TestMetadataLoading:
    """Tests for load_metadata and _extract_metadata."""

    @pytest.mark.asyncio
    async def test_load_metadata_finds_skills(self, loader, tmp_skill_dir):
        """load_metadata discovers all .md files in the directory tree."""
        loader.add_skill_directory(tmp_skill_dir)
        await loader.load_metadata()

        assert loader._loaded is True
        # 3 skill files: testing.md, async.md, workflow.md
        assert len(loader._metadata_cache) == 3

    @pytest.mark.asyncio
    async def test_metadata_name_from_heading(self, loaded_loader):
        """Skill name is extracted from the first # heading."""
        names = [m.name for m in loaded_loader._metadata_cache.values()]
        assert "Python Testing Skill" in names
        assert "Async Programming Skill" in names
        assert "Git Workflow Skill" in names

    @pytest.mark.asyncio
    async def test_metadata_tags_from_frontmatter(self, loaded_loader):
        """Tags are extracted from YAML frontmatter."""
        # Find the testing skill
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        assert "python" in testing.tags
        assert "testing" in testing.tags

    @pytest.mark.asyncio
    async def test_metadata_triggers_from_frontmatter(self, loaded_loader):
        """Triggers are extracted from YAML frontmatter."""
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        assert "pytest" in testing.triggers
        assert "test" in testing.triggers

    @pytest.mark.asyncio
    async def test_metadata_priority_from_frontmatter(self, loaded_loader):
        """Priority is extracted from YAML frontmatter."""
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        assert testing.priority == 70

    @pytest.mark.asyncio
    async def test_metadata_default_priority(self, loaded_loader):
        """Skills without explicit priority get default 50."""
        async_skill = [m for m in loaded_loader._metadata_cache.values() if "Async" in m.name][0]
        assert async_skill.priority == 50

    @pytest.mark.asyncio
    async def test_implicit_tags_from_directory(self, loaded_loader):
        """Parent directory name added as implicit tag."""
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        assert "python" in testing.tags  # parent dir = "python"

    @pytest.mark.asyncio
    async def test_implicit_triggers_from_name(self, loaded_loader):
        """Name words added as implicit triggers."""
        git_skill = [m for m in loaded_loader._metadata_cache.values() if "Git Workflow" in m.name][
            0
        ]
        # "git", "workflow", "skill" from "Git Workflow Skill"
        assert "git" in git_skill.triggers
        assert "workflow" in git_skill.triggers

    @pytest.mark.asyncio
    async def test_token_estimate(self, loaded_loader):
        """Token estimate is len(content) // 4."""
        for meta in loaded_loader._metadata_cache.values():
            assert meta.token_estimate > 0

    @pytest.mark.asyncio
    async def test_load_metadata_clears_cache(self, loader, tmp_skill_dir):
        """load_metadata clears the metadata cache before rescanning."""
        loader.add_skill_directory(tmp_skill_dir)
        await loader.load_metadata()
        first_count = len(loader._metadata_cache)

        # Reload
        await loader.load_metadata()
        assert len(loader._metadata_cache) == first_count

    @pytest.mark.asyncio
    async def test_empty_directory(self, loader, tmp_path):
        """Empty directory results in no skills."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loader.add_skill_directory(empty_dir)
        await loader.load_metadata()

        assert loader._loaded is True
        assert len(loader._metadata_cache) == 0

    @pytest.mark.asyncio
    async def test_malformed_frontmatter(self, loader, tmp_path):
        """Malformed frontmatter is handled gracefully (no crash)."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "bad.md").write_text(
            textwrap.dedent("""\
                ---
                tags: [unclosed bracket
                ---
                # Bad Skill
                Content here.
            """),
            encoding="utf-8",
        )
        loader.add_skill_directory(skill_dir)
        await loader.load_metadata()

        # Should still load (with no tags from frontmatter)
        assert len(loader._metadata_cache) == 1
        skill = list(loader._metadata_cache.values())[0]
        assert skill.name == "Bad Skill"

    @pytest.mark.asyncio
    async def test_skill_without_heading(self, loader, tmp_path):
        """Skill file with no # heading uses filename as name."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "no_heading.md").write_text(
            "Just some content without a heading.",
            encoding="utf-8",
        )
        loader.add_skill_directory(skill_dir)
        await loader.load_metadata()

        skill = list(loader._metadata_cache.values())[0]
        assert skill.name == "no_heading"  # filename stem


# =============================================================================
# Task-Based Loading
# =============================================================================


class TestLoadForTask:
    """Tests for load_for_task and _score_skill."""

    @pytest.mark.asyncio
    async def test_matching_skills_returned(self, loaded_loader):
        """Task with matching keywords returns relevant skills."""
        skills = await loaded_loader.load_for_task("Write pytest tests")
        assert len(skills) >= 1
        # Python Testing Skill should match (triggers: pytest, test)
        skill_names = [s.metadata.name for s in skills]
        assert "Python Testing Skill" in skill_names

    @pytest.mark.asyncio
    async def test_no_matching_keywords(self, loaded_loader):
        """Task with no keyword overlap still returns skills (priority bonus).

        The scoring formula always adds priority/10, so every known
        skill gets a non-zero score.  We verify that the scores are
        purely from priority bonus (no keyword matches).
        """
        task_words = {"make", "a", "sandwich"}
        for meta in loaded_loader._metadata_cache.values():
            score = loaded_loader._score_skill(meta, task_words)
            # Score should equal just the priority bonus (no keyword match)
            assert score == meta.priority / 10.0

    @pytest.mark.asyncio
    async def test_score_includes_triggers(self, loaded_loader):
        """Trigger keyword matches score +10 each."""
        task_words = {"pytest", "test"}
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        score = loaded_loader._score_skill(testing, task_words)

        # trigger matches: "pytest" and "test" → +20
        # plus tag/name matches and priority bonus
        assert score >= 20

    @pytest.mark.asyncio
    async def test_score_includes_tag_matches(self, loaded_loader):
        """Tag keyword matches score +5 each."""
        task_words = {"python"}
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        score = loaded_loader._score_skill(testing, task_words)

        # "python" is both a tag and trigger
        assert score >= 5  # At minimum from tag match

    @pytest.mark.asyncio
    async def test_score_includes_priority_bonus(self, loaded_loader):
        """Priority / 10 is added as a bonus."""
        task_words = set()
        testing = [m for m in loaded_loader._metadata_cache.values() if "Python Testing" in m.name][
            0
        ]
        score = loaded_loader._score_skill(testing, task_words)

        # Only priority bonus: 70 / 10 = 7.0
        assert score == testing.priority / 10.0

    @pytest.mark.asyncio
    async def test_max_skills_limit(self, loaded_loader):
        """At most max_skills are returned."""
        skills = await loaded_loader.load_for_task(
            "python test git commit pytest async",
            max_skills=2,
        )
        assert len(skills) <= 2

    @pytest.mark.asyncio
    async def test_token_budget_respected(self, loaded_loader):
        """Skills that would exceed token budget are skipped."""
        # Very small budget → only 1 or 0 skills
        skills = await loaded_loader.load_for_task(
            "pytest test",
            max_tokens=10,  # very small
        )
        # Token estimates are content_len // 4, likely > 10 for any skill
        assert len(skills) == 0

    @pytest.mark.asyncio
    async def test_auto_loads_metadata(self, loader, tmp_skill_dir):
        """load_for_task calls load_metadata on first invocation."""
        loader.add_skill_directory(tmp_skill_dir)
        assert loader._loaded is False

        skills = await loader.load_for_task("pytest test")
        assert loader._loaded is True

    @pytest.mark.asyncio
    async def test_sorted_by_relevance(self, loaded_loader):
        """Skills are returned sorted by relevance score."""
        skills = await loaded_loader.load_for_task(
            "python test pytest async commit",
            max_skills=5,
        )
        if len(skills) >= 2:
            # First skill should have highest score
            task_words = {"python", "test", "pytest", "async", "commit"}
            scores = [loaded_loader._score_skill(s.metadata, task_words) for s in skills]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Content Caching
# =============================================================================


class TestContentCaching:
    """Tests for LRU content caching."""

    @pytest.mark.asyncio
    async def test_cache_populated_on_load(self, loaded_loader):
        """Loading skills populates the content cache."""
        await loaded_loader.load_for_task("pytest test")
        # At least one skill should be cached
        assert len(loaded_loader._content_cache) >= 1

    @pytest.mark.asyncio
    async def test_cache_hit(self, loaded_loader):
        """Second load of the same skill uses cache."""
        skills1 = await loaded_loader.load_for_task("pytest test")
        cache_size_after_first = len(loaded_loader._content_cache)

        skills2 = await loaded_loader.load_for_task("pytest test")
        # Cache size should not increase
        assert len(loaded_loader._content_cache) == cache_size_after_first

    @pytest.mark.asyncio
    async def test_cache_eviction(self, tmp_path):
        """Oldest cache entries are evicted when cache_size is exceeded."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()

        # Create 5 skill files
        for i in range(5):
            (skill_dir / f"skill_{i}.md").write_text(
                f"# Skill {i}\n\nContent for skill {i}.",
                encoding="utf-8",
            )

        loader = SkillLoader(cache_size=3)
        loader.add_skill_directory(skill_dir)
        await loader.load_metadata()

        # Load all skills (forcing 5 entries into a cache of size 3)
        for key, meta in loader._metadata_cache.items():
            await loader._load_skill(key, meta)

        # Cache should not exceed cache_size
        assert len(loader._content_cache) <= 3


# =============================================================================
# Section Parsing
# =============================================================================


class TestSectionParsing:
    """Tests for _parse_sections."""

    def test_basic_sections(self):
        """## headings create sections."""
        content = textwrap.dedent("""\
            # Main Title

            ## When to Use
            Use this for testing.

            ## Best Practices
            Follow these practices.
        """)
        sections = SkillLoader._parse_sections(content)

        assert "intro" in sections  # Content before first ##
        assert "when_to_use" in sections
        assert "best_practices" in sections

    def test_section_content(self):
        """Section content is captured correctly."""
        content = "## My Section\nLine 1\nLine 2"
        sections = SkillLoader._parse_sections(content)

        assert "my_section" in sections
        assert "Line 1" in sections["my_section"]
        assert "Line 2" in sections["my_section"]

    def test_no_sections(self):
        """Content with no ## headings goes entirely to intro."""
        content = "Just some plain text."
        sections = SkillLoader._parse_sections(content)

        assert "intro" in sections
        assert sections["intro"].strip() == "Just some plain text."

    def test_section_name_normalization(self):
        """Section names are lowercase and underscored."""
        content = "## Best Practices\nContent"
        sections = SkillLoader._parse_sections(content)

        assert "best_practices" in sections

    def test_multiple_sections(self):
        """Multiple sections are parsed independently."""
        content = textwrap.dedent("""\
            ## Section A
            Content A

            ## Section B
            Content B

            ## Section C
            Content C
        """)
        sections = SkillLoader._parse_sections(content)

        assert len(sections) >= 3
        assert "section_a" in sections
        assert "section_b" in sections
        assert "section_c" in sections

    def test_empty_content(self):
        """Empty content produces empty or minimal sections."""
        sections = SkillLoader._parse_sections("")
        # "intro" with empty string
        assert len(sections) <= 1


# =============================================================================
# Formatting
# =============================================================================


class TestFormatSkills:
    """Tests for format_skills."""

    def _make_skill(self, name="Test Skill", content="Test content", sections=None):
        """Helper to create a Skill with a given name and content."""
        meta = SkillMetadata(name=name, path=Path("/tmp/test.md"))
        return Skill(
            metadata=meta,
            content=content,
            sections=sections or {},
        )

    def test_single_skill(self):
        """Single skill is formatted with header."""
        loader = SkillLoader()
        skill = self._make_skill()
        result = loader.format_skills([skill])

        assert "# Relevant Skills" in result
        assert "## Test Skill" in result
        assert "Test content" in result

    def test_multiple_skills(self):
        """Multiple skills are separated."""
        loader = SkillLoader()
        skills = [
            self._make_skill(name="Skill A", content="Content A"),
            self._make_skill(name="Skill B", content="Content B"),
        ]
        result = loader.format_skills(skills)

        assert "## Skill A" in result
        assert "## Skill B" in result
        assert "---" in result

    def test_empty_skills_list(self):
        """Empty skills list → empty string."""
        loader = SkillLoader()
        result = loader.format_skills([])
        assert result == ""

    def test_include_sections_filter(self):
        """include_sections selects specific sections."""
        loader = SkillLoader()
        skill = self._make_skill(
            sections={
                "when_to_use": "Use for testing.",
                "best_practices": "Write good tests.",
                "examples": "Example code here.",
            },
            content="Full content",
        )
        result = loader.format_skills(
            [skill],
            include_sections=["when_to_use", "best_practices"],
        )

        assert "Use for testing." in result
        assert "Write good tests." in result
        assert "Example code here." not in result

    def test_include_sections_none_includes_all(self):
        """include_sections=None includes full content."""
        loader = SkillLoader()
        content = "Full content with everything."
        skill = self._make_skill(content=content)
        result = loader.format_skills([skill])

        assert content in result


# =============================================================================
# Introspection
# =============================================================================


class TestIntrospection:
    """Tests for get_available_skills."""

    @pytest.mark.asyncio
    async def test_get_available_skills(self, loaded_loader):
        """Returns metadata for all discovered skills."""
        skills = loaded_loader.get_available_skills()
        assert len(skills) == 3
        assert all(isinstance(s, SkillMetadata) for s in skills)

    @pytest.mark.asyncio
    async def test_get_available_skills_empty(self, loader):
        """Empty loader returns empty list."""
        skills = loader.get_available_skills()
        assert skills == []


# =============================================================================
# Frontmatter Extraction Helpers
# =============================================================================


class TestFrontmatterHelpers:
    """Tests for _extract_list_field and _extract_int_field."""

    def test_extract_list_field_basic(self):
        """Basic list extraction from frontmatter."""
        content = "tags: [python, testing, pytest]"
        result = _extract_list_field(content, "tags")
        assert result == ["python", "testing", "pytest"]

    def test_extract_list_field_quoted(self):
        """Quoted items are unquoted."""
        content = "triggers: ['test', \"pytest\", assertion]"
        result = _extract_list_field(content, "triggers")
        assert "test" in result
        assert "pytest" in result
        assert "assertion" in result

    def test_extract_list_field_missing(self):
        """Missing field returns empty list."""
        content = "nothing: here"
        result = _extract_list_field(content, "tags")
        assert result == []

    def test_extract_list_field_empty_brackets(self):
        """Empty brackets return empty list."""
        content = "tags: []"
        result = _extract_list_field(content, "tags")
        assert result == []

    def test_extract_list_field_spaces(self):
        """Spaces around items are stripped."""
        content = "tags: [ python ,  testing , pytest ]"
        result = _extract_list_field(content, "tags")
        assert result == ["python", "testing", "pytest"]

    def test_extract_int_field_basic(self):
        """Basic integer extraction."""
        content = "priority: 70"
        result = _extract_int_field(content, "priority")
        assert result == 70

    def test_extract_int_field_missing(self):
        """Missing field returns default."""
        content = "nothing: here"
        result = _extract_int_field(content, "priority", default=50)
        assert result == 50

    def test_extract_int_field_custom_default(self):
        """Custom default is returned when field is missing."""
        content = ""
        result = _extract_int_field(content, "priority", default=99)
        assert result == 99

    def test_extract_int_field_in_frontmatter(self):
        """Int field within --- delimiters."""
        content = "---\ntags: [a]\npriority: 85\n---\n# Title"
        result = _extract_int_field(content, "priority")
        assert result == 85


# =============================================================================
# End-to-End Integration
# =============================================================================


class TestEndToEnd:
    """End-to-end integration tests for the skill loading workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_skill_dir):
        """Full workflow: add dir → load metadata → load for task → format."""
        loader = SkillLoader()
        loader.add_skill_directory(tmp_skill_dir)

        # Load for task
        skills = await loader.load_for_task("Write pytest tests for the module")

        # Should include the testing skill
        assert len(skills) >= 1
        skill_names = [s.metadata.name for s in skills]
        assert "Python Testing Skill" in skill_names

        # Format
        formatted = loader.format_skills(skills)
        assert "# Relevant Skills" in formatted
        assert "Python Testing Skill" in formatted
        assert "Best Practices" in formatted

    @pytest.mark.asyncio
    async def test_section_selective_loading(self, tmp_skill_dir):
        """Load skills and format with specific sections only."""
        loader = SkillLoader()
        loader.add_skill_directory(tmp_skill_dir)

        skills = await loader.load_for_task("Write pytest tests")
        formatted = loader.format_skills(skills, include_sections=["when_to_use"])

        # Should include "when_to_use" section content
        assert "writing or debugging" in formatted
        # Should NOT include "best_practices" section
        assert "descriptive test names" not in formatted

    @pytest.mark.asyncio
    async def test_multiple_directories(self, tmp_path):
        """Skills from multiple directories are discovered."""
        dir_a = tmp_path / "skills_a"
        dir_b = tmp_path / "skills_b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "skill_a.md").write_text("# Skill A\nContent A.", encoding="utf-8")
        (dir_b / "skill_b.md").write_text("# Skill B\nContent B.", encoding="utf-8")

        loader = SkillLoader()
        loader.add_skill_directory(dir_a)
        loader.add_skill_directory(dir_b)
        await loader.load_metadata()

        assert len(loader._metadata_cache) == 2
        names = [m.name for m in loader._metadata_cache.values()]
        assert "Skill A" in names
        assert "Skill B" in names
