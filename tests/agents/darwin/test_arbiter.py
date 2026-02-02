# tests/agents/darwin/test_arbiter.py
"""
Comprehensive tests for Darwin Multi-Attempt Arbiter system.

Tests candidate generation, evaluation, selection, and edge cases.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List

import pytest

from llmcore.agents.darwin.arbiter import (
    ArbiterConfig,
    ArbiterDecision,
    ArbiterPrompts,
    Candidate,
    CandidateScore,
    EvaluationCriteria,
    MultiAttemptArbiter,
)

# =============================================================================
# Mock LLM Responses
# =============================================================================


class MockLLMResponse:
    """Mock LLM response object."""

    def __init__(self, content: str):
        self.content = content


def create_mock_llm(responses: Dict[str, str] = None):
    """
    Create a mock LLM callable.

    Args:
        responses: Map of prompt keywords to responses.
                   If a keyword is found in the prompt, return that response.
    """
    responses = responses or {}

    async def mock_llm(messages: List[Dict], **kwargs) -> MockLLMResponse:
        # Get the user message content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Check for specific response patterns
        for keyword, response in responses.items():
            if keyword.lower() in user_content.lower():
                return MockLLMResponse(response)

        # Default responses based on prompt type
        if "evaluate" in user_content.lower():
            return MockLLMResponse(
                json.dumps(
                    {
                        "scores": {
                            "correctness": {"score": 8.0, "justification": "Good"},
                            "clarity": {"score": 7.0, "justification": "Clear"},
                            "efficiency": {"score": 7.5, "justification": "Efficient"},
                            "completeness": {"score": 8.0, "justification": "Complete"},
                        },
                        "overall_feedback": "Solid implementation",
                    }
                )
            )
        elif "selecting the best" in user_content.lower():
            return MockLLMResponse(
                json.dumps(
                    {
                        "selected_id": "candidate_0",
                        "reasoning": "Best overall score and quality",
                        "confidence": 0.85,
                    }
                )
            )
        else:
            # Generation response
            return MockLLMResponse(
                """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
            )

    return mock_llm


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Provide a basic mock LLM callable."""
    return create_mock_llm()


@pytest.fixture
def arbiter(mock_llm):
    """Provide a configured arbiter for testing."""
    config = ArbiterConfig(num_candidates=3)
    arb = MultiAttemptArbiter(llm_client=mock_llm, config=config)
    return arb


@pytest.fixture
def sample_candidate():
    """Provide a sample candidate for testing."""
    return Candidate(
        id="test_candidate_1",
        content="def example(): pass",
        temperature=0.5,
        prompt_variant="Focus on simplicity",
    )


@pytest.fixture
def sample_candidates():
    """Provide multiple sample candidates."""
    return [
        Candidate(
            id="candidate_0",
            content="def add(a, b): return a + b",
            temperature=0.3,
            prompt_variant="Focus on simplicity",
        ),
        Candidate(
            id="candidate_1",
            content="def add(a, b):\n    # Add two numbers\n    return a + b",
            temperature=0.5,
            prompt_variant="Focus on clarity",
        ),
        Candidate(
            id="candidate_2",
            content="def add(a: int, b: int) -> int:\n    return a + b",
            temperature=0.7,
            prompt_variant="Focus on robustness",
        ),
    ]


@pytest.fixture
def sample_scores():
    """Provide sample scores for candidates."""
    return [
        CandidateScore(
            candidate_id="candidate_0",
            criteria_scores={
                "correctness": 8.0,
                "clarity": 6.0,
                "efficiency": 9.0,
                "completeness": 7.0,
            },
            weighted_total=7.38,
        ),
        CandidateScore(
            candidate_id="candidate_1",
            criteria_scores={
                "correctness": 8.0,
                "clarity": 9.0,
                "efficiency": 8.0,
                "completeness": 8.0,
            },
            weighted_total=8.25,
        ),
        CandidateScore(
            candidate_id="candidate_2",
            criteria_scores={
                "correctness": 9.0,
                "clarity": 8.0,
                "efficiency": 7.0,
                "completeness": 9.0,
            },
            weighted_total=8.38,
        ),
    ]


# =============================================================================
# Data Model Tests
# =============================================================================


class TestCandidate:
    """Tests for Candidate data model."""

    def test_candidate_creation(self):
        """Test basic candidate creation."""
        candidate = Candidate(
            id="test_1",
            content="def foo(): pass",
            temperature=0.5,
            prompt_variant="test variant",
        )
        assert candidate.id == "test_1"
        assert candidate.content == "def foo(): pass"
        assert candidate.temperature == 0.5
        assert candidate.prompt_variant == "test variant"

    def test_candidate_auto_id(self):
        """Test candidate auto-generates ID if not provided."""
        candidate = Candidate(
            content="def bar(): pass",
            temperature=0.3,
            prompt_variant="variant",
        )
        assert candidate.id.startswith("candidate_")

    def test_candidate_metadata(self):
        """Test candidate metadata handling."""
        candidate = Candidate(
            id="test_2",
            content="code",
            temperature=0.5,
            prompt_variant="variant",
            metadata={"custom": "value", "index": 1},
        )
        assert candidate.metadata["custom"] == "value"
        assert candidate.metadata["index"] == 1

    def test_candidate_timestamp(self):
        """Test candidate has created_at timestamp."""
        from datetime import timezone

        before = datetime.now(timezone.utc)
        candidate = Candidate(
            content="code",
            temperature=0.5,
            prompt_variant="variant",
        )
        after = datetime.now(timezone.utc)
        assert before <= candidate.created_at <= after

    def test_candidate_serialization(self):
        """Test candidate JSON serialization."""
        candidate = Candidate(
            id="test_3",
            content="def test(): pass",
            temperature=0.5,
            prompt_variant="variant",
        )
        data = candidate.model_dump(mode="json")
        assert data["id"] == "test_3"
        assert data["content"] == "def test(): pass"
        assert "created_at" in data


class TestEvaluationCriteria:
    """Tests for EvaluationCriteria data model."""

    def test_criteria_creation(self):
        """Test basic criteria creation."""
        criteria = EvaluationCriteria(
            name="test",
            description="Test criterion",
            weight=2.0,
        )
        assert criteria.name == "test"
        assert criteria.description == "Test criterion"
        assert criteria.weight == 2.0

    def test_criteria_defaults(self):
        """Test criteria default values."""
        criteria = EvaluationCriteria(
            name="default",
            description="Default values",
        )
        assert criteria.weight == 1.0
        assert criteria.min_score == 0.0
        assert criteria.max_score == 10.0

    def test_validate_score(self):
        """Test score validation/clamping."""
        criteria = EvaluationCriteria(
            name="test",
            description="Test",
            min_score=0.0,
            max_score=10.0,
        )
        assert criteria.validate_score(5.0) == 5.0
        assert criteria.validate_score(-1.0) == 0.0
        assert criteria.validate_score(15.0) == 10.0


class TestCandidateScore:
    """Tests for CandidateScore data model."""

    def test_score_creation(self):
        """Test basic score creation."""
        score = CandidateScore(
            candidate_id="test_1",
            criteria_scores={"correctness": 8.0, "clarity": 7.0},
            weighted_total=7.5,
        )
        assert score.candidate_id == "test_1"
        assert score.criteria_scores["correctness"] == 8.0
        assert score.weighted_total == 7.5

    def test_score_with_feedback(self):
        """Test score with arbiter feedback."""
        score = CandidateScore(
            candidate_id="test_2",
            criteria_scores={"correctness": 9.0},
            weighted_total=9.0,
            arbiter_feedback="Excellent implementation",
        )
        assert score.arbiter_feedback == "Excellent implementation"


class TestArbiterDecision:
    """Tests for ArbiterDecision data model."""

    def test_decision_creation(self, sample_candidates, sample_scores):
        """Test basic decision creation."""
        decision = ArbiterDecision(
            selected_id="candidate_1",
            selected_content="def add(a, b): return a + b",
            all_scores=sample_scores,
            all_candidates=sample_candidates,
            reasoning="Best overall quality",
            confidence=0.85,
        )
        assert decision.selected_id == "candidate_1"
        assert decision.confidence == 0.85
        assert len(decision.all_scores) == 3
        assert len(decision.all_candidates) == 3

    def test_get_candidate_by_id(self, sample_candidates, sample_scores):
        """Test retrieving candidate by ID."""
        decision = ArbiterDecision(
            selected_id="candidate_1",
            selected_content="content",
            all_scores=sample_scores,
            all_candidates=sample_candidates,
            reasoning="test",
            confidence=0.8,
        )
        candidate = decision.get_candidate_by_id("candidate_1")
        assert candidate is not None
        assert candidate.id == "candidate_1"

        missing = decision.get_candidate_by_id("nonexistent")
        assert missing is None

    def test_get_score_by_id(self, sample_candidates, sample_scores):
        """Test retrieving score by candidate ID."""
        decision = ArbiterDecision(
            selected_id="candidate_1",
            selected_content="content",
            all_scores=sample_scores,
            all_candidates=sample_candidates,
            reasoning="test",
            confidence=0.8,
        )
        score = decision.get_score_by_id("candidate_1")
        assert score is not None
        assert score.candidate_id == "candidate_1"


class TestArbiterConfig:
    """Tests for ArbiterConfig data model."""

    def test_config_defaults(self):
        """Test config default values."""
        config = ArbiterConfig()
        assert config.num_candidates == 3
        assert len(config.temperatures) == 3
        assert len(config.criteria) == 4
        assert config.min_acceptable_score == 5.0

    def test_config_custom_criteria(self):
        """Test config with custom criteria."""
        custom_criteria = [
            EvaluationCriteria(name="accuracy", description="Accurate results", weight=5.0),
            EvaluationCriteria(name="speed", description="Fast execution", weight=2.0),
        ]
        config = ArbiterConfig(criteria=custom_criteria)
        assert len(config.criteria) == 2
        assert config.criteria[0].name == "accuracy"

    def test_config_total_weight(self):
        """Test total weight calculation."""
        config = ArbiterConfig()
        expected = sum(c.weight for c in config.criteria)
        assert config.total_weight == expected

    def test_get_criterion(self):
        """Test getting criterion by name."""
        config = ArbiterConfig()
        criterion = config.get_criterion("correctness")
        assert criterion is not None
        assert criterion.name == "correctness"

        missing = config.get_criterion("nonexistent")
        assert missing is None


# =============================================================================
# MultiAttemptArbiter Tests
# =============================================================================


class TestMultiAttemptArbiterInitialization:
    """Tests for arbiter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        arbiter = MultiAttemptArbiter()
        assert arbiter.config is not None
        assert arbiter._llm_client is None

    def test_init_with_llm(self, mock_llm):
        """Test initialization with LLM client."""
        arbiter = MultiAttemptArbiter(llm_client=mock_llm)
        assert arbiter._llm_client is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ArbiterConfig(num_candidates=5)
        arbiter = MultiAttemptArbiter(config=config)
        assert arbiter.config.num_candidates == 5

    def test_set_llm_client(self, mock_llm):
        """Test setting LLM client after initialization."""
        arbiter = MultiAttemptArbiter()
        assert arbiter._llm_client is None
        arbiter.set_llm_client(mock_llm)
        assert arbiter._llm_client is not None


class TestCandidateGeneration:
    """Tests for candidate generation."""

    @pytest.mark.asyncio
    async def test_generate_candidates(self, arbiter):
        """Test generating multiple candidates."""
        candidates = await arbiter.generate_candidates(
            task="Implement binary search",
            context="Handle edge cases",
        )
        assert len(candidates) == 3
        for i, candidate in enumerate(candidates):
            assert candidate.id == f"candidate_{i}"
            assert candidate.content
            assert candidate.temperature in arbiter.config.temperatures

    @pytest.mark.asyncio
    async def test_generate_with_custom_variants(self, arbiter):
        """Test generation with custom prompt variants."""
        custom_variants = [
            "Be concise",
            "Be verbose",
            "Use functional style",
        ]
        candidates = await arbiter.generate_candidates(
            task="Implement sort",
            context="",
            prompt_variants=custom_variants,
        )
        assert len(candidates) == 3
        # Check variants were used (cycling through)
        for i, candidate in enumerate(candidates):
            assert candidate.prompt_variant == custom_variants[i % len(custom_variants)]

    @pytest.mark.asyncio
    async def test_generate_without_llm_raises(self):
        """Test that generation without LLM raises error."""
        arbiter = MultiAttemptArbiter()
        with pytest.raises(ValueError, match="LLM client not configured"):
            await arbiter.generate_candidates(task="test")

    @pytest.mark.asyncio
    async def test_generate_handles_partial_failures(self):
        """Test that generation continues despite some failures."""
        call_count = 0

        async def failing_llm(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Generation failed")
            return MockLLMResponse("def example(): pass")

        arbiter = MultiAttemptArbiter(
            llm_client=failing_llm,
            config=ArbiterConfig(num_candidates=3),
        )
        candidates = await arbiter.generate_candidates(task="test")
        # Should have 2 successful candidates (one failed)
        assert len(candidates) == 2


class TestCandidateEvaluation:
    """Tests for candidate evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_candidate(self, arbiter, sample_candidate):
        """Test evaluating a single candidate."""
        score = await arbiter.evaluate_candidate(
            candidate=sample_candidate,
            task="Implement a function",
        )
        assert score.candidate_id == sample_candidate.id
        assert "correctness" in score.criteria_scores
        assert 0 <= score.weighted_total <= 10

    @pytest.mark.asyncio
    async def test_evaluate_without_llm_raises(self, sample_candidate):
        """Test that evaluation without LLM raises error."""
        arbiter = MultiAttemptArbiter()
        with pytest.raises(ValueError, match="LLM client not configured"):
            await arbiter.evaluate_candidate(
                candidate=sample_candidate,
                task="test",
            )

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(self, sample_candidate):
        """Test evaluation handles malformed JSON response."""

        async def bad_json_llm(messages, **kwargs):
            return MockLLMResponse("not valid json {{{")

        arbiter = MultiAttemptArbiter(llm_client=bad_json_llm)
        score = await arbiter.evaluate_candidate(
            candidate=sample_candidate,
            task="test",
        )
        # Should return default scores
        assert score.weighted_total == 5.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_criteria(self, sample_candidate):
        """Test evaluation handles missing criteria in response."""

        async def partial_json_llm(messages, **kwargs):
            return MockLLMResponse(
                json.dumps(
                    {
                        "scores": {
                            "correctness": {"score": 9.0, "justification": "Good"},
                            # Missing other criteria
                        },
                        "overall_feedback": "Partial",
                    }
                )
            )

        arbiter = MultiAttemptArbiter(llm_client=partial_json_llm)
        score = await arbiter.evaluate_candidate(
            candidate=sample_candidate,
            task="test",
        )
        # correctness should be 9.0, others default to 5.0
        assert score.criteria_scores["correctness"] == 9.0
        assert score.criteria_scores.get("clarity", 5.0) == 5.0


class TestCandidateSelection:
    """Tests for candidate selection."""

    @pytest.mark.asyncio
    async def test_select_best(self, arbiter, sample_candidates, sample_scores):
        """Test selecting the best candidate."""
        decision = await arbiter.select_best(
            candidates=sample_candidates,
            scores=sample_scores,
            task="Implement addition",
        )
        assert decision.selected_id == "candidate_0"
        assert decision.reasoning
        assert 0 <= decision.confidence <= 1

    @pytest.mark.asyncio
    async def test_select_without_llm_raises(self, sample_candidates, sample_scores):
        """Test that selection without LLM raises error."""
        arbiter = MultiAttemptArbiter()
        with pytest.raises(ValueError, match="LLM client not configured"):
            await arbiter.select_best(
                candidates=sample_candidates,
                scores=sample_scores,
                task="test",
            )

    @pytest.mark.asyncio
    async def test_select_fallback_on_error(self, sample_candidates, sample_scores):
        """Test fallback selection when LLM fails."""

        async def failing_llm(messages, **kwargs):
            return MockLLMResponse("invalid json")

        config = ArbiterConfig(fallback_on_error=True)
        arbiter = MultiAttemptArbiter(llm_client=failing_llm, config=config)

        decision = await arbiter.select_best(
            candidates=sample_candidates,
            scores=sample_scores,
            task="test",
        )
        # Should fall back to highest weighted score (candidate_2)
        assert decision.selected_id == "candidate_2"
        assert "fallback" in decision.reasoning.lower()


class TestGenerateAndSelect:
    """Tests for the full generate_and_select workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, arbiter):
        """Test the complete generate and select workflow."""
        decision = await arbiter.generate_and_select(
            task="Implement binary search",
            context="Must handle empty arrays",
        )
        assert decision.selected_id
        assert decision.selected_content
        assert len(decision.all_scores) == 3
        assert len(decision.all_candidates) == 3
        assert decision.reasoning
        assert 0 <= decision.confidence <= 1
        assert decision.total_time_ms is not None

    @pytest.mark.asyncio
    async def test_workflow_without_llm_raises(self):
        """Test workflow without LLM raises error."""
        arbiter = MultiAttemptArbiter()
        with pytest.raises(ValueError, match="LLM client not configured"):
            await arbiter.generate_and_select(task="test")

    @pytest.mark.asyncio
    async def test_workflow_with_custom_config(self, mock_llm):
        """Test workflow with custom configuration."""
        custom_criteria = [
            EvaluationCriteria(
                name="accuracy",
                description="Code is accurate",
                weight=5.0,
            ),
        ]
        config = ArbiterConfig(
            num_candidates=2,
            temperatures=[0.2, 0.8],
            criteria=custom_criteria,
        )
        arbiter = MultiAttemptArbiter(llm_client=mock_llm, config=config)

        decision = await arbiter.generate_and_select(
            task="Test task",
            context="Test context",
        )
        assert len(decision.all_candidates) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_candidate(self, mock_llm):
        """Test handling of single candidate."""
        config = ArbiterConfig(num_candidates=1)
        arbiter = MultiAttemptArbiter(llm_client=mock_llm, config=config)

        decision = await arbiter.generate_and_select(
            task="Test task",
        )
        assert len(decision.all_candidates) == 1
        assert decision.selected_id == "candidate_0"
        assert "only one candidate" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_all_low_scores(self, mock_llm):
        """Test handling when all candidates have low scores."""

        async def low_score_llm(messages, **kwargs):
            user_content = messages[-1].get("content", "")
            if "evaluate" in user_content.lower():
                return MockLLMResponse(
                    json.dumps(
                        {
                            "scores": {
                                "correctness": {"score": 2.0, "justification": "Poor"},
                                "clarity": {"score": 2.0, "justification": "Poor"},
                                "efficiency": {"score": 2.0, "justification": "Poor"},
                                "completeness": {"score": 2.0, "justification": "Poor"},
                            },
                            "overall_feedback": "All candidates are poor",
                        }
                    )
                )
            elif "selecting" in user_content.lower():
                return MockLLMResponse(
                    json.dumps(
                        {
                            "selected_id": "candidate_0",
                            "reasoning": "Least bad option",
                            "confidence": 0.3,
                        }
                    )
                )
            return MockLLMResponse("def bad(): pass")

        arbiter = MultiAttemptArbiter(llm_client=low_score_llm)
        decision = await arbiter.generate_and_select(task="Test")

        # Should still select something
        assert decision.selected_id
        assert decision.confidence <= 0.5  # Low confidence

    @pytest.mark.asyncio
    async def test_empty_context(self, arbiter):
        """Test handling of empty context."""
        decision = await arbiter.generate_and_select(
            task="Implement a function",
            context="",
        )
        assert decision.selected_id
        assert decision.selected_content

    @pytest.mark.asyncio
    async def test_long_task_description(self, arbiter):
        """Test handling of long task descriptions."""
        long_task = "Implement a function that " + "does something important " * 100
        decision = await arbiter.generate_and_select(
            task=long_task,
            context="Additional context",
        )
        assert decision.selected_id

    @pytest.mark.asyncio
    async def test_special_characters_in_task(self, arbiter):
        """Test handling of special characters in task."""
        special_task = 'Implement a function that handles "quotes" and <tags> and {braces}'
        decision = await arbiter.generate_and_select(
            task=special_task,
            context="Handle edge cases",
        )
        assert decision.selected_id


class TestCodeExtraction:
    """Tests for code extraction from LLM responses."""

    def test_extract_code_simple(self, arbiter):
        """Test extracting code without markdown."""
        content = "def foo(): pass"
        result = arbiter._extract_code(content)
        assert result == "def foo(): pass"

    def test_extract_code_with_markdown(self, arbiter):
        """Test extracting code from markdown fences."""
        content = "Here's the code:\n```python\ndef foo(): pass\n```"
        result = arbiter._extract_code(content)
        assert result == "def foo(): pass"

    def test_extract_code_with_language_tag(self, arbiter):
        """Test extracting code with language tag."""
        content = "```javascript\nfunction foo() {}\n```"
        result = arbiter._extract_code(content)
        assert result == "function foo() {}"

    def test_extract_code_multiple_blocks(self, arbiter):
        """Test extracting code from multiple blocks."""
        content = (
            "```python\ndef first(): pass\n```\n\nSome text\n\n```python\ndef second(): pass\n```"
        )
        result = arbiter._extract_code(content)
        # Should return first non-empty block
        assert result == "def first(): pass"


class TestJsonParsing:
    """Tests for JSON parsing from LLM responses."""

    def test_parse_json_simple(self, arbiter):
        """Test parsing simple JSON."""
        content = '{"key": "value"}'
        result = arbiter._parse_json(content)
        assert result["key"] == "value"

    def test_parse_json_with_markdown(self, arbiter):
        """Test parsing JSON from markdown."""
        content = '```json\n{"key": "value"}\n```'
        result = arbiter._parse_json(content)
        assert result["key"] == "value"

    def test_parse_json_with_text(self, arbiter):
        """Test parsing JSON with surrounding text."""
        content = 'Here is the result:\n{"key": "value"}\nDone.'
        result = arbiter._parse_json(content)
        assert result["key"] == "value"

    def test_parse_json_invalid_raises(self, arbiter):
        """Test that invalid JSON raises error."""
        content = "not valid json"
        with pytest.raises(json.JSONDecodeError):
            arbiter._parse_json(content)


class TestPrompts:
    """Tests for prompt templates."""

    def test_generation_prompt_format(self):
        """Test generation prompt formatting."""
        prompt = ArbiterPrompts.GENERATION_PROMPT.format(
            task="Test task",
            context="Test context",
            additional_instructions="Be concise",
        )
        assert "Test task" in prompt
        assert "Test context" in prompt
        assert "Be concise" in prompt

    def test_evaluation_prompt_format(self):
        """Test evaluation prompt formatting."""
        prompt = ArbiterPrompts.EVALUATION_PROMPT.format(
            task="Test task",
            code="def foo(): pass",
            criteria_list="- correctness: Is it correct?",
        )
        assert "Test task" in prompt
        assert "def foo(): pass" in prompt
        assert "correctness" in prompt

    def test_selection_prompt_format(self):
        """Test selection prompt formatting."""
        prompt = ArbiterPrompts.SELECTION_PROMPT.format(
            task="Test task",
            candidates_summary="candidate_0: score 8.0\ncandidate_1: score 7.0",
        )
        assert "Test task" in prompt
        assert "candidate_0" in prompt


class TestIntegrationWithOtherComponents:
    """Integration tests with other Darwin components."""

    @pytest.mark.asyncio
    async def test_arbiter_with_evaluation_results(self, mock_llm):
        """Test using arbiter evaluation results for learning."""
        arbiter = MultiAttemptArbiter(llm_client=mock_llm)
        decision = await arbiter.generate_and_select(
            task="Implement sorting",
            context="Must be stable",
        )

        # Verify we can extract useful data for failure learning
        for score in decision.all_scores:
            assert score.candidate_id
            assert score.criteria_scores
            assert score.weighted_total >= 0

        # Verify we can identify below-threshold candidates
        min_score = 5.0
        low_score_candidates = [s for s in decision.all_scores if s.weighted_total < min_score]
        # These could be logged as failures for learning

    @pytest.mark.asyncio
    async def test_arbiter_decision_serialization(self, arbiter):
        """Test that arbiter decisions can be serialized."""
        decision = await arbiter.generate_and_select(
            task="Test task",
            context="Test context",
        )

        # Should be serializable to JSON
        data = decision.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["selected_id"]
        assert data["reasoning"]

        # Should be deserializable
        restored = ArbiterDecision.model_validate(data)
        assert restored.selected_id == decision.selected_id


class TestConcurrencyAndTimeout:
    """Tests for concurrent execution and timeouts."""

    @pytest.mark.asyncio
    async def test_parallel_generation(self, mock_llm):
        """Test parallel candidate generation."""
        config = ArbiterConfig(
            num_candidates=5,
            parallel_generation=True,
        )
        arbiter = MultiAttemptArbiter(llm_client=mock_llm, config=config)

        candidates = await arbiter.generate_candidates(task="Test")
        assert len(candidates) == 5

    @pytest.mark.asyncio
    async def test_sequential_generation(self, mock_llm):
        """Test sequential candidate generation."""
        config = ArbiterConfig(
            num_candidates=3,
            parallel_generation=False,
        )
        arbiter = MultiAttemptArbiter(llm_client=mock_llm, config=config)

        candidates = await arbiter.generate_candidates(task="Test")
        assert len(candidates) == 3

    @pytest.mark.asyncio
    async def test_generation_timeout(self):
        """Test generation timeout handling."""

        async def slow_llm(messages, **kwargs):
            await asyncio.sleep(10)  # Sleep longer than timeout
            return MockLLMResponse("code")

        config = ArbiterConfig(
            num_candidates=1,
            timeout_per_generation_s=0.1,  # Very short timeout
        )
        arbiter = MultiAttemptArbiter(llm_client=slow_llm, config=config)

        # Should handle timeout gracefully
        candidates = await arbiter.generate_candidates(task="Test")
        # Candidate should fail due to timeout
        assert len(candidates) == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
