# tests/integration/phase10/__init__.py
"""
Phase 10: Final Integration & Polish Test Suite

This package contains comprehensive integration tests for validating
cross-project functionality across the llmcore ecosystem:

- llmcore: Core LLM infrastructure library
- llmchat: CLI/REPL interface
- semantiscan: RAG/code analysis engine

Test Categories:
- test_hitl_integration.py: Human-in-the-loop approval workflows
- test_rag_integration.py: RAG pipeline end-to-end flows
- test_observability_integration.py: Metrics, events, and cost tracking
- test_config_validation.py: Configuration system validation
- test_performance_benchmarks.py: Performance and overhead measurements

Usage:
    # Run all Phase 10 tests
    pytest tests/integration/phase10/ -v

    # Run specific category
    pytest tests/integration/phase10/test_hitl_integration.py -v

    # Run with coverage
    pytest tests/integration/phase10/ --cov=llmcore --cov-report=term-missing

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Section 12 (Phase 10)
    - LLMCORE_CONTINUATION_GUIDE_v6.md Section 4
"""

__all__ = [
    "test_hitl_integration",
    "test_rag_integration",
    "test_observability_integration",
    "test_config_validation",
    "test_performance_benchmarks",
]
