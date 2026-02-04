# src/llmcore/agents/darwin/arbiter.py
"""
Multi-Attempt Arbiter System for Darwin Agents.

Generates N candidates for a task and uses an arbiter to select the best one.
This produces higher quality outputs by exploring multiple solution approaches
and using LLM-based evaluation to identify the best result.

Features:
    - Generate with different temperatures/prompts for diversity
    - Score candidates based on multiple configurable criteria
    - Use LLM as arbiter for final selection
    - Track selection decisions for learning and analysis
    - Configurable evaluation criteria and weights

Workflow:
    TASK → GENERATE_CANDIDATES → EVALUATE_EACH → ARBITER_SELECT → BEST_RESULT

Architecture:
    - ArbiterConfig: Configuration for generation and evaluation
    - MultiAttemptArbiter: High-level API for multi-attempt generation with selection
    - Data models: Candidate, EvaluationCriteria, CandidateScore, ArbiterDecision

Usage:
    from llmcore.agents.darwin.arbiter import MultiAttemptArbiter, ArbiterConfig

    # Initialize with LLM client
    arbiter = MultiAttemptArbiter(llm_client=my_llm, config=ArbiterConfig(num_candidates=3))

    # Generate multiple candidates and select the best
    decision = await arbiter.generate_and_select(
        task="Implement a binary search function",
        context="Must handle edge cases like empty arrays and single elements",
    )

    print(f"Selected: {decision.selected_id}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Confidence: {decision.confidence}")
    best_code = decision.selected_content

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 6.3
    - Phase 6.1: Failure Learning System (pattern reference)
    - Phase 6.2: TDD Support (pattern reference)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone, UTC
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class Candidate(BaseModel):
    """
    A single generation candidate.

    Represents one possible solution generated with specific parameters.

    Attributes:
        id: Unique identifier for this candidate
        content: Generated content (code, text, etc.)
        temperature: Temperature used for generation
        prompt_variant: Description of prompt variant used
        generation_time_ms: Time taken to generate (milliseconds)
        metadata: Additional metadata about generation
        created_at: When this candidate was generated
    """

    id: str = Field(default_factory=lambda: f"candidate_{uuid.uuid4().hex[:8]}")
    content: str
    temperature: float
    prompt_variant: str
    generation_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class EvaluationCriteria(BaseModel):
    """
    Criteria for evaluating candidates.

    Defines a single evaluation dimension with description and weight.

    Attributes:
        name: Identifier name for this criterion
        description: Human-readable description of what this criterion measures
        weight: Importance weight (higher = more important)
        min_score: Minimum possible score
        max_score: Maximum possible score
    """

    name: str
    description: str
    weight: float = 1.0
    min_score: float = 0.0
    max_score: float = 10.0

    def validate_score(self, score: float) -> float:
        """Clamp score to valid range."""
        return max(self.min_score, min(self.max_score, score))


class CandidateScore(BaseModel):
    """
    Score for a candidate on all criteria.

    Aggregates scores across all evaluation criteria for a single candidate.

    Attributes:
        candidate_id: ID of the candidate being scored
        criteria_scores: Map of criterion name to score
        weighted_total: Weighted average of all scores
        arbiter_feedback: Optional feedback from the evaluator
        evaluation_time_ms: Time taken to evaluate (milliseconds)
        created_at: When this evaluation was performed
    """

    candidate_id: str
    criteria_scores: dict[str, float]  # criterion name -> score
    weighted_total: float
    arbiter_feedback: str | None = None
    evaluation_time_ms: float | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class ArbiterDecision(BaseModel):
    """
    Final arbiter decision.

    Records the result of the multi-attempt selection process.

    Attributes:
        selected_id: ID of the selected candidate
        selected_content: Content of the selected candidate
        all_scores: Scores for all candidates
        all_candidates: All generated candidates
        reasoning: Explanation of why this candidate was selected
        confidence: Confidence score (0-1)
        selection_time_ms: Time taken to make selection (milliseconds)
        total_time_ms: Total time for entire process (milliseconds)
        created_at: When this decision was made
    """

    selected_id: str
    selected_content: str
    all_scores: list[CandidateScore]
    all_candidates: list[Candidate] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    selection_time_ms: float | None = None
    total_time_ms: float | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    def get_candidate_by_id(self, candidate_id: str) -> Candidate | None:
        """Get a candidate by ID."""
        for candidate in self.all_candidates:
            if candidate.id == candidate_id:
                return candidate
        return None

    def get_score_by_id(self, candidate_id: str) -> CandidateScore | None:
        """Get a score by candidate ID."""
        for score in self.all_scores:
            if score.candidate_id == candidate_id:
                return score
        return None


class ArbiterConfig(BaseModel):
    """
    Configuration for the multi-attempt arbiter.

    Controls candidate generation and evaluation parameters.

    Attributes:
        num_candidates: Number of candidates to generate
        temperatures: Temperature values for each candidate
        criteria: Evaluation criteria with weights
        min_acceptable_score: Minimum score to consider a candidate acceptable
        parallel_generation: Whether to generate candidates in parallel
        parallel_evaluation: Whether to evaluate candidates in parallel
        timeout_per_generation_s: Timeout for each generation (seconds)
        timeout_per_evaluation_s: Timeout for each evaluation (seconds)
        fallback_on_error: Whether to fall back to highest score on selection error
    """

    num_candidates: int = Field(default=3, ge=1, le=10)
    temperatures: list[float] = Field(default_factory=lambda: [0.3, 0.5, 0.7])
    criteria: list[EvaluationCriteria] = Field(
        default_factory=lambda: [
            EvaluationCriteria(
                name="correctness",
                description="Code is functionally correct and handles all cases",
                weight=3.0,
            ),
            EvaluationCriteria(
                name="clarity",
                description="Code is readable, well-structured, and self-documenting",
                weight=2.0,
            ),
            EvaluationCriteria(
                name="efficiency",
                description="Code is reasonably efficient without premature optimization",
                weight=1.0,
            ),
            EvaluationCriteria(
                name="completeness",
                description="All requirements are addressed with proper error handling",
                weight=2.0,
            ),
        ]
    )
    min_acceptable_score: float = Field(default=5.0, ge=0.0, le=10.0)
    parallel_generation: bool = True
    parallel_evaluation: bool = True
    timeout_per_generation_s: float = 60.0
    timeout_per_evaluation_s: float = 30.0
    fallback_on_error: bool = True

    @property
    def total_weight(self) -> float:
        """Get sum of all criterion weights."""
        return sum(c.weight for c in self.criteria)

    def get_criterion(self, name: str) -> EvaluationCriteria | None:
        """Get a criterion by name."""
        for c in self.criteria:
            if c.name == name:
                return c
        return None


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================


class ArbiterPrompts:
    """Prompt templates for the arbiter system."""

    GENERATION_PROMPT = """You are an expert software developer.

Task: {task}

Context: {context}

{additional_instructions}

Generate high-quality, production-ready code. Include:
1. Clear, readable implementation
2. Proper error handling
3. Helpful comments where needed
4. Type hints (if applicable)

Output only the code, no explanations or markdown code fences."""

    EVALUATION_PROMPT = """You are an expert code reviewer. Evaluate the following code candidate.

Task: {task}

Code:
```
{code}
```

Evaluate on these criteria (score 0-10 for each):

{criteria_list}

For each criterion, provide:
- Score (0-10)
- Brief justification (1 sentence)

Output as JSON:
{{
    "scores": {{
        "criterion_name": {{"score": N, "justification": "..."}},
        ...
    }},
    "overall_feedback": "Brief overall assessment"
}}

Output ONLY valid JSON, no explanations or markdown."""

    SELECTION_PROMPT = """You are selecting the best code candidate from multiple options.

Task: {task}

Candidates and their scores:
{candidates_summary}

Based on the scores and the task requirements, select the best candidate.
Consider:
- Weighted scores (higher weight = more important)
- Overall quality and adherence to requirements
- Trade-offs between candidates

Output as JSON:
{{
    "selected_id": "candidate_X",
    "reasoning": "Why this candidate is best",
    "confidence": 0.X
}}

Output ONLY valid JSON, no explanations or markdown."""


# =============================================================================
# MULTI-ATTEMPT ARBITER
# =============================================================================


class MultiAttemptArbiter:
    """
    Generates multiple candidates and selects the best one.

    This class orchestrates the multi-attempt generation process:
    1. Generates N candidates with different temperatures/prompts
    2. Evaluates each candidate against defined criteria
    3. Uses LLM-based arbiter to select the best candidate

    Usage:
        arbiter = MultiAttemptArbiter(llm_client, config)

        decision = await arbiter.generate_and_select(
            task="Implement a binary search function",
            context="Must handle edge cases...",
        )

        best_code = decision.selected_content
    """

    # Default prompt variants for diversity
    DEFAULT_VARIANTS = [
        "Focus on simplicity and readability. Write clean, idiomatic code.",
        "Focus on efficiency and performance. Optimize for speed where possible.",
        "Focus on robustness and error handling. Handle all edge cases.",
    ]

    def __init__(
        self,
        llm_client: Callable | None = None,
        config: ArbiterConfig | None = None,
    ):
        """
        Initialize the multi-attempt arbiter.

        Args:
            llm_client: Async callable for LLM generation.
                        Signature: async (messages: List[Dict], **kwargs) -> response
                        Response should have a `.content` attribute or be a string.
            config: Arbiter configuration (uses defaults if not provided)
        """
        self._llm_client = llm_client
        self.config = config or ArbiterConfig()
        self._prompts = ArbiterPrompts()

    def set_llm_client(self, llm_client: Callable) -> None:
        """Set the LLM client for generation and evaluation."""
        self._llm_client = llm_client

    async def generate_and_select(
        self,
        task: str,
        context: str = "",
        prompt_variants: list[str] | None = None,
    ) -> ArbiterDecision:
        """
        Generate multiple candidates and select the best.

        This is the main entry point for the arbiter. It:
        1. Generates N candidates with different parameters
        2. Evaluates each candidate against all criteria
        3. Selects the best candidate using LLM-based arbiter

        Args:
            task: The task to complete (what to implement)
            context: Additional context and requirements
            prompt_variants: Optional custom prompt variants for diversity

        Returns:
            ArbiterDecision with selected candidate and all scores

        Raises:
            ValueError: If LLM client not configured
            RuntimeError: If generation or selection fails
        """
        if not self._llm_client:
            raise ValueError("LLM client not configured. Call set_llm_client() first.")

        total_start = _utc_now()

        # Generate candidates
        logger.info(f"Generating {self.config.num_candidates} candidates for task")
        candidates = await self._generate_candidates(task, context, prompt_variants)
        logger.info(f"Generated {len(candidates)} candidates")

        if not candidates:
            raise RuntimeError("No candidates generated")

        # Evaluate each candidate
        logger.info("Evaluating candidates")
        scores = await self._evaluate_candidates(candidates, task)
        logger.info(f"Evaluated {len(scores)} candidates")

        # Select best
        logger.info("Selecting best candidate")
        decision = await self._select_best(candidates, scores, task)

        total_end = _utc_now()
        total_ms = (total_end - total_start).total_seconds() * 1000
        decision.total_time_ms = total_ms
        decision.all_candidates = candidates

        logger.info(
            f"Selected candidate {decision.selected_id} with confidence "
            f"{decision.confidence:.2f} (total time: {total_ms:.0f}ms)"
        )

        return decision

    async def generate_candidates(
        self,
        task: str,
        context: str = "",
        prompt_variants: list[str] | None = None,
    ) -> list[Candidate]:
        """
        Generate multiple candidates without evaluation.

        Useful when you want to generate candidates but use custom evaluation.

        Args:
            task: The task to complete
            context: Additional context
            prompt_variants: Custom prompt variants

        Returns:
            List of generated Candidates
        """
        if not self._llm_client:
            raise ValueError("LLM client not configured. Call set_llm_client() first.")

        return await self._generate_candidates(task, context, prompt_variants)

    async def evaluate_candidate(
        self,
        candidate: Candidate,
        task: str,
    ) -> CandidateScore:
        """
        Evaluate a single candidate.

        Useful for evaluating individual candidates or custom evaluation flows.

        Args:
            candidate: The candidate to evaluate
            task: The original task

        Returns:
            CandidateScore with all criteria scores
        """
        if not self._llm_client:
            raise ValueError("LLM client not configured. Call set_llm_client() first.")

        return await self._evaluate_single(candidate, task)

    async def select_best(
        self,
        candidates: list[Candidate],
        scores: list[CandidateScore],
        task: str,
    ) -> ArbiterDecision:
        """
        Select the best candidate from pre-evaluated candidates.

        Useful when you have already generated and evaluated candidates.

        Args:
            candidates: List of candidates
            scores: Corresponding scores
            task: The original task

        Returns:
            ArbiterDecision with selection
        """
        if not self._llm_client:
            raise ValueError("LLM client not configured. Call set_llm_client() first.")

        return await self._select_best(candidates, scores, task)

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    async def _generate_candidates(
        self,
        task: str,
        context: str,
        prompt_variants: list[str] | None,
    ) -> list[Candidate]:
        """Generate N candidates with different temperatures."""
        variants = prompt_variants or self.DEFAULT_VARIANTS

        # Ensure we have enough variants by cycling
        while len(variants) < self.config.num_candidates:
            variants = variants + variants

        candidates = []
        tasks = []

        for i in range(self.config.num_candidates):
            temp = self.config.temperatures[i % len(self.config.temperatures)]
            variant = variants[i % len(variants)]
            tasks.append(self._generate_single(task, context, temp, variant, i))

        if self.config.parallel_generation:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for t in tasks:
                try:
                    result = await t
                    results.append(result)
                except Exception as e:
                    results.append(e)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Candidate {i} generation failed: {result}")
                continue
            candidates.append(result)

        return candidates

    async def _generate_single(
        self,
        task: str,
        context: str,
        temperature: float,
        variant: str,
        index: int,
    ) -> Candidate:
        """Generate a single candidate."""
        start_time = _utc_now()

        prompt = self._prompts.GENERATION_PROMPT.format(
            task=task,
            context=context,
            additional_instructions=variant,
        )

        try:
            response = await asyncio.wait_for(
                self._llm_client(
                    [{"role": "user", "content": prompt}],
                    temperature=temperature,
                ),
                timeout=self.config.timeout_per_generation_s,
            )

            content = self._extract_content(response)
            code = self._extract_code(content)

            end_time = _utc_now()
            gen_time_ms = (end_time - start_time).total_seconds() * 1000

            return Candidate(
                id=f"candidate_{index}",
                content=code,
                temperature=temperature,
                prompt_variant=variant,
                generation_time_ms=gen_time_ms,
                metadata={"index": index},
                created_at=start_time,
            )

        except TimeoutError:
            raise RuntimeError(
                f"Generation timeout for candidate {index} "
                f"(>{self.config.timeout_per_generation_s}s)"
            )

    async def _evaluate_candidates(
        self,
        candidates: list[Candidate],
        task: str,
    ) -> list[CandidateScore]:
        """Evaluate all candidates."""
        if self.config.parallel_evaluation:
            tasks = [self._evaluate_single(c, task) for c in candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for c in candidates:
                try:
                    result = await self._evaluate_single(c, task)
                    results.append(result)
                except Exception as e:
                    results.append(e)

        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Candidate {i} evaluation failed: {result}")
                # Create default score for failed evaluation
                scores.append(
                    CandidateScore(
                        candidate_id=candidates[i].id,
                        criteria_scores={c.name: 5.0 for c in self.config.criteria},
                        weighted_total=5.0,
                        arbiter_feedback=f"Evaluation failed: {result}",
                    )
                )
            else:
                scores.append(result)

        return scores

    async def _evaluate_single(
        self,
        candidate: Candidate,
        task: str,
    ) -> CandidateScore:
        """Evaluate a single candidate."""
        start_time = _utc_now()

        criteria_list = "\n".join(
            [f"- {c.name}: {c.description} (weight: {c.weight})" for c in self.config.criteria]
        )

        prompt = self._prompts.EVALUATION_PROMPT.format(
            task=task,
            code=candidate.content,
            criteria_list=criteria_list,
        )

        try:
            response = await asyncio.wait_for(
                self._llm_client(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert code reviewer. Output valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                ),
                timeout=self.config.timeout_per_evaluation_s,
            )

            content = self._extract_content(response)
            data = self._parse_json(content)

            criteria_scores = {}
            weighted_total = 0.0

            for criterion in self.config.criteria:
                score_data = data.get("scores", {}).get(criterion.name, {})
                if isinstance(score_data, dict):
                    score = float(score_data.get("score", 5.0))
                else:
                    score = float(score_data) if score_data else 5.0
                score = criterion.validate_score(score)
                criteria_scores[criterion.name] = score
                weighted_total += score * criterion.weight

            weighted_total /= self.config.total_weight

            end_time = _utc_now()
            eval_time_ms = (end_time - start_time).total_seconds() * 1000

            return CandidateScore(
                candidate_id=candidate.id,
                criteria_scores=criteria_scores,
                weighted_total=weighted_total,
                arbiter_feedback=data.get("overall_feedback"),
                evaluation_time_ms=eval_time_ms,
                created_at=start_time,
            )

        except TimeoutError:
            logger.warning(
                f"Evaluation timeout for {candidate.id} (>{self.config.timeout_per_evaluation_s}s)"
            )
            return CandidateScore(
                candidate_id=candidate.id,
                criteria_scores={c.name: 5.0 for c in self.config.criteria},
                weighted_total=5.0,
                arbiter_feedback="Evaluation timed out",
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation for {candidate.id}: {e}")
            return CandidateScore(
                candidate_id=candidate.id,
                criteria_scores={c.name: 5.0 for c in self.config.criteria},
                weighted_total=5.0,
                arbiter_feedback=f"Parse error: {e}",
            )

    async def _select_best(
        self,
        candidates: list[Candidate],
        scores: list[CandidateScore],
        task: str,
    ) -> ArbiterDecision:
        """Select the best candidate."""
        start_time = _utc_now()

        # Handle edge cases
        if len(candidates) == 0:
            raise RuntimeError("No candidates to select from")

        if len(candidates) == 1:
            return ArbiterDecision(
                selected_id=candidates[0].id,
                selected_content=candidates[0].content,
                all_scores=scores,
                reasoning="Only one candidate available",
                confidence=scores[0].weighted_total / 10.0 if scores else 0.5,
            )

        # Build summary for selection
        summary_parts = []
        for candidate, score in zip(candidates, scores):
            parts = [f"\n{candidate.id} (temp={candidate.temperature}):"]
            parts.append(f"  Weighted Score: {score.weighted_total:.2f}")
            for name, s in score.criteria_scores.items():
                parts.append(f"  - {name}: {s:.1f}")
            if score.arbiter_feedback:
                parts.append(f"  Feedback: {score.arbiter_feedback}")
            summary_parts.append("\n".join(parts))

        prompt = self._prompts.SELECTION_PROMPT.format(
            task=task,
            candidates_summary="\n".join(summary_parts),
        )

        try:
            response = await self._llm_client(
                [
                    {"role": "system", "content": "Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            content = self._extract_content(response)
            data = self._parse_json(content)
            selected_id = data.get("selected_id", "candidate_0")

            # Find selected candidate
            selected = next(
                (c for c in candidates if c.id == selected_id),
                None,
            )

            if selected is None:
                # Fallback to highest score
                logger.warning(f"Selected ID '{selected_id}' not found, using fallback")
                selected = self._fallback_select(candidates, scores)
                selected_id = selected.id

            end_time = _utc_now()
            selection_time_ms = (end_time - start_time).total_seconds() * 1000

            return ArbiterDecision(
                selected_id=selected.id,
                selected_content=selected.content,
                all_scores=scores,
                reasoning=data.get("reasoning", "Highest weighted score"),
                confidence=float(data.get("confidence", 0.7)),
                selection_time_ms=selection_time_ms,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Selection parse failed, using fallback: {e}")
            if self.config.fallback_on_error:
                selected = self._fallback_select(candidates, scores)
                return ArbiterDecision(
                    selected_id=selected.id,
                    selected_content=selected.content,
                    all_scores=scores,
                    reasoning="Selected based on highest weighted score (fallback)",
                    confidence=0.5,
                )
            raise RuntimeError(f"Failed to select best candidate: {e}")

    def _fallback_select(
        self,
        candidates: list[Candidate],
        scores: list[CandidateScore],
    ) -> Candidate:
        """Fallback selection: choose candidate with highest weighted score."""
        if not scores:
            return candidates[0]

        # Map candidate IDs to scores
        score_map = {s.candidate_id: s.weighted_total for s in scores}

        # Find candidate with highest score
        best_candidate = max(
            candidates,
            key=lambda c: score_map.get(c.id, 0.0),
        )
        return best_candidate

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, dict):
            return response.get("content", "") or response.get("text", "")
        return str(response)

    def _extract_code(self, content: str) -> str:
        """Extract code from response, handling markdown fences."""
        content = content.strip()

        # Check for markdown code blocks
        if "```" in content:
            # Find all code blocks
            code_blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
            if code_blocks:
                # Return the first non-empty code block
                for block in code_blocks:
                    if block.strip():
                        return block.strip()

        return content

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON from content, handling markdown fences."""
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        # Try to find JSON object
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        return json.loads(content)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data models
    "Candidate",
    "EvaluationCriteria",
    "CandidateScore",
    "ArbiterDecision",
    "ArbiterConfig",
    # Prompts
    "ArbiterPrompts",
    # Main class
    "MultiAttemptArbiter",
]
