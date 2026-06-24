"""Tests for LLMCore-owned prompt budget helpers."""

from __future__ import annotations

from llmcore.context import (
    build_context_budget,
    estimate_tool_schema_tokens,
    should_compress_prompt,
)
from llmcore.models import Tool
from llmcore.tokens import EstimateCounter


def test_context_budget_reserves_output_tools_and_margin() -> None:
    budget = build_context_budget(
        context_window_tokens=1000,
        reserved_output_tokens=200,
        tool_schema_tokens=125,
        safety_margin_tokens=25,
    )

    assert budget.reserved_non_prompt_tokens == 350
    assert budget.prompt_tokens_available == 650
    assert budget.reserve_overflows_context is False


def test_context_budget_reports_reserve_overflow() -> None:
    budget = build_context_budget(
        context_window_tokens=100,
        reserved_output_tokens=120,
        tool_schema_tokens=40,
    )

    assert budget.prompt_tokens_available == 0
    assert budget.reserve_overflows_context is True
    assert budget.reserve_overflow_tokens == 60


def test_estimate_tool_schema_tokens_counts_provider_visible_schema() -> None:
    tool = Tool(
        name="lookup_document",
        description="Lookup a document by identifier.",
        parameters={
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "required": ["document_id"],
        },
    )

    tokens = estimate_tool_schema_tokens(
        [tool],
        counter=EstimateCounter(chars_per_token=1),
    )

    assert tokens > len(tool.name)
    assert tokens > len(tool.description)


def test_should_compress_prompt_uses_prompt_budget_threshold() -> None:
    assert (
        should_compress_prompt(
            prompt_tokens=790,
            prompt_budget_tokens=1000,
            threshold=0.8,
        )
        is False
    )
    assert (
        should_compress_prompt(
            prompt_tokens=801,
            prompt_budget_tokens=1000,
            threshold=0.8,
        )
        is True
    )
