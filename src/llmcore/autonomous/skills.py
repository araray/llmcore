# src/llmcore/autonomous/skills.py
"""
Skill Loading System for Autonomous Agents.

Skills are markdown files containing specialized knowledge that agents
can load on-demand based on the current task.  This mirrors how
AI assistants use SKILL.md files for domain-specific guidance.

Skill file format (markdown with optional YAML frontmatter)::

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
    ...

Default skill directory: ``~/.local/share/llmcore/skills/``

Example::

    loader = SkillLoader()
    loader.add_skill_directory("~/.local/share/llmcore/skills")

    # Load relevant skills for a task
    skills = await loader.load_for_task(
        "Write pytest tests for the auth module"
    )

    # Format for context injection
    skill_context = loader.format_skills(skills)

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §13 (Skill Loading System)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SkillMetadata:
    """
    Metadata extracted from a skill file.

    Attributes:
        name: Skill name (from first ``#`` heading or filename).
        path: Absolute path to the skill file.
        tags: Tags for categorization.
        triggers: Keywords that trigger this skill.
        priority: Loading priority (higher = loaded first).
        token_estimate: Estimated tokens when fully loaded.
    """

    name: str
    path: Path
    tags: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    priority: int = 50
    token_estimate: int = 0


@dataclass
class Skill:
    """
    A loaded skill ready for use.

    Attributes:
        metadata: Skill metadata.
        content: Full skill content (markdown).
        sections: Parsed sections keyed by heading (lowercase, underscored).
    """

    metadata: SkillMetadata
    content: str
    sections: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# SkillLoader
# =============================================================================


class SkillLoader:
    """
    Loads and manages skill files for agents.

    Skills are markdown files organized in directories::

        skills/
        ├── python/
        │   ├── testing.md
        │   ├── async.md
        │   └── debugging.md
        ├── git/
        │   └── workflow.md
        └── general/
            └── problem_solving.md

    The loader supports:
    - Multi-directory scanning (``add_skill_directory``)
    - YAML frontmatter extraction (``tags``, ``triggers``, ``priority``)
    - Keyword-based task matching
    - LRU content caching
    - Section-level selective loading

    Example::

        loader = SkillLoader(cache_size=50)
        loader.add_skill_directory("/path/to/skills")

        skills = await loader.load_for_task("Fix the async test")
        context = loader.format_skills(skills)
    """

    def __init__(self, cache_size: int = 50) -> None:
        """
        Initialize the skill loader.

        Args:
            cache_size: Maximum number of skill contents to cache in memory.
        """
        self.cache_size = cache_size

        self._skill_dirs: List[Path] = []
        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._content_cache: Dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def add_skill_directory(self, path: str | Path) -> None:
        """
        Add a directory to search for skill files.

        Only ``.md`` files within the directory (recursive) are considered.

        Args:
            path: Directory path (``~`` expansion supported).
        """
        skill_path = Path(path).expanduser().resolve()
        if skill_path.exists() and skill_path.is_dir():
            if skill_path not in self._skill_dirs:
                self._skill_dirs.append(skill_path)
                self._loaded = False
                logger.info("Added skill directory: %s", skill_path)
        else:
            logger.warning("Skill directory not found: %s", path)

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    async def load_metadata(self) -> None:
        """
        Scan all registered directories and load skill metadata.

        Clears the metadata cache and rescans.  This is called
        automatically on the first ``load_for_task`` if needed.
        """
        self._metadata_cache.clear()

        for skill_dir in self._skill_dirs:
            for md_file in skill_dir.rglob("*.md"):
                try:
                    metadata = self._extract_metadata(md_file, skill_dir)
                    key = str(md_file)
                    self._metadata_cache[key] = metadata
                except Exception as exc:
                    logger.warning("Failed to load skill metadata %s: %s", md_file, exc)

        self._loaded = True
        logger.info("Loaded metadata for %d skills", len(self._metadata_cache))

    def _extract_metadata(self, path: Path, base_dir: Path) -> SkillMetadata:
        """
        Extract metadata from a skill markdown file.

        Metadata sources (in order of precedence):
        1. YAML frontmatter (``tags``, ``triggers``, ``priority``)
        2. First ``#`` heading → skill name
        3. Parent directory name → implicit tag
        4. Name words → implicit triggers

        Args:
            path: Path to the markdown file.
            base_dir: Root skill directory for relative path calculation.

        Returns:
            SkillMetadata instance.
        """
        content = path.read_text(encoding="utf-8")

        # Extract name from first # heading or filename
        name_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        name = name_match.group(1).strip() if name_match else path.stem

        # Extract YAML frontmatter fields
        tags = _extract_list_field(content, "tags")
        triggers = _extract_list_field(content, "triggers")
        priority = _extract_int_field(content, "priority", default=50)

        # Add parent directory name as implicit tag
        try:
            relative = path.relative_to(base_dir)
            if relative.parent.name:
                tags.append(relative.parent.name)
        except ValueError:
            pass

        # Add name words as implicit triggers
        triggers.extend(name.lower().split())
        triggers.extend(t.lower() for t in tags)

        # Deduplicate
        tags = list(set(tags))
        triggers = list(set(t.lower() for t in triggers))

        # Estimate tokens
        token_estimate = len(content) // 4

        return SkillMetadata(
            name=name,
            path=path,
            tags=tags,
            triggers=triggers,
            priority=priority,
            token_estimate=token_estimate,
        )

    # ------------------------------------------------------------------
    # Task-based loading
    # ------------------------------------------------------------------

    async def load_for_task(
        self,
        task_description: str,
        max_skills: int = 5,
        max_tokens: int = 20_000,
    ) -> List[Skill]:
        """
        Load skills relevant to a task description.

        Scores each known skill against the task using keyword matching
        on triggers, tags, and name words, then loads the top matches
        within the token budget.

        Args:
            task_description: Natural language description of the current task.
            max_skills: Maximum number of skills to load.
            max_tokens: Maximum total tokens for all loaded skills.

        Returns:
            List of loaded ``Skill`` objects, sorted by relevance score.
        """
        if not self._loaded:
            await self.load_metadata()

        # Score skills by relevance
        task_words = set(task_description.lower().split())
        scored: List[tuple] = []

        for key, metadata in self._metadata_cache.items():
            score = self._score_skill(metadata, task_words)
            if score > 0:
                scored.append((score, key, metadata))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Load top skills within budget
        skills: List[Skill] = []
        total_tokens = 0

        for score, key, metadata in scored[: max_skills * 2]:
            if len(skills) >= max_skills:
                break
            if total_tokens + metadata.token_estimate > max_tokens:
                continue

            skill = await self._load_skill(key, metadata)
            if skill is not None:
                skills.append(skill)
                total_tokens += metadata.token_estimate

        logger.debug(
            "Loaded %d skills (%d tokens) for task: %s",
            len(skills),
            total_tokens,
            task_description[:50],
        )

        return skills

    def _score_skill(
        self,
        metadata: SkillMetadata,
        task_words: Set[str],
    ) -> float:
        """
        Calculate relevance score for a skill.

        Scoring components:
        - Trigger keyword matches: +10 each
        - Tag matches: +5 each
        - Name word matches: +3 each
        - Priority bonus: priority / 10

        Args:
            metadata: Skill metadata to score.
            task_words: Lowercased words from the task description.

        Returns:
            Non-negative score (0.0 = no match).
        """
        score = 0.0

        # Trigger matches
        trigger_matches = len(task_words & set(metadata.triggers))
        score += trigger_matches * 10

        # Tag matches
        tag_matches = len(task_words & {t.lower() for t in metadata.tags})
        score += tag_matches * 5

        # Name word matches
        name_words = set(metadata.name.lower().split())
        name_matches = len(task_words & name_words)
        score += name_matches * 3

        # Priority bonus
        score += metadata.priority / 10.0

        return score

    # ------------------------------------------------------------------
    # Content loading and caching
    # ------------------------------------------------------------------

    async def _load_skill(
        self,
        key: str,
        metadata: SkillMetadata,
    ) -> Optional[Skill]:
        """
        Load a skill's full content from disk or cache.

        Args:
            key: Cache key (string path).
            metadata: Skill metadata.

        Returns:
            Loaded ``Skill``, or ``None`` on read failure.
        """
        # Check cache
        if key in self._content_cache:
            content = self._content_cache[key]
        else:
            try:
                content = metadata.path.read_text(encoding="utf-8")

                # Cache with LRU eviction
                if len(self._content_cache) >= self.cache_size:
                    # Evict oldest entry
                    oldest_key = next(iter(self._content_cache))
                    del self._content_cache[oldest_key]

                self._content_cache[key] = content
            except Exception as exc:
                logger.error("Failed to read skill %s: %s", key, exc)
                return None

        # Parse sections
        sections = self._parse_sections(content)

        return Skill(
            metadata=metadata,
            content=content,
            sections=sections,
        )

    @staticmethod
    def _parse_sections(content: str) -> Dict[str, str]:
        """
        Parse markdown content into sections by ``##`` headings.

        Args:
            content: Full markdown content.

        Returns:
            Dict mapping section name (lowercase, underscored) → content.
        """
        sections: Dict[str, str] = {}
        current_section = "intro"
        current_lines: List[str] = []

        for line in content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_lines:
                    sections[current_section] = "\n".join(current_lines)

                # Start new section
                current_section = (
                    line[3:].strip().lower().replace(" ", "_")
                )
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections[current_section] = "\n".join(current_lines)

        return sections

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_skills(
        self,
        skills: List[Skill],
        include_sections: Optional[List[str]] = None,
    ) -> str:
        """
        Format loaded skills into a context string.

        Args:
            skills: Skills to format.
            include_sections: Only include these section names.
                ``None`` includes full content.

        Returns:
            Formatted markdown string ready for context injection.
        """
        if not skills:
            return ""

        parts = ["# Relevant Skills\n"]

        for skill in skills:
            parts.append(f"## {skill.metadata.name}\n")

            if include_sections:
                for section_name in include_sections:
                    if section_name in skill.sections:
                        parts.append(skill.sections[section_name])
            else:
                parts.append(skill.content)

            parts.append("\n---\n")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_available_skills(self) -> List[SkillMetadata]:
        """
        Get metadata for all discovered skills.

        Returns:
            List of ``SkillMetadata`` for all registered skills.
        """
        return list(self._metadata_cache.values())


# =============================================================================
# Frontmatter extraction helpers
# =============================================================================


def _extract_list_field(content: str, field_name: str) -> List[str]:
    """
    Extract a list field from YAML frontmatter or inline comments.

    Supports both ``tags: [a, b, c]`` and ``tags: [a, 'b', "c"]``.
    """
    pattern = rf"{field_name}\s*:\s*\[([^\]]*)\]"
    match = re.search(pattern, content)
    if not match:
        return []
    raw = match.group(1)
    items = [
        item.strip().strip("'\"")
        for item in raw.split(",")
        if item.strip()
    ]
    return items


def _extract_int_field(content: str, field_name: str, default: int = 0) -> int:
    """Extract an integer field from YAML frontmatter."""
    pattern = rf"{field_name}\s*:\s*(\d+)"
    match = re.search(pattern, content)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default
