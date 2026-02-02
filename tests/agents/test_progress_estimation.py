# tests/agents/test_progress_estimation.py
"""
Unit tests for Phase 2: Cognitive Loop Fixes - Progress Estimation & Circuit Breaker.

This module tests:
1. Progress estimation from content (_estimate_progress_from_content)
2. Progress extraction with explicit values (_extract_progress)
3. Circuit breaker step_completed handling
4. Integration between reflect phase and circuit breaker

The main bug being fixed: "STEP_COMPLETED: No" was incorrectly returning 85%
progress because "completed" was matched as a substring.

References:
    - LLMCORE_CORRECTION_MASTER_PLAN.md Phase 2
    - Technical Spec: Section 5.3.7 (REFLECT Phase)
"""


import pytest

from llmcore.agents.cognitive.phases.reflect import (
    _estimate_progress_from_content,
    _extract_progress,
)
from llmcore.agents.resilience.circuit_breaker import (
    AgentCircuitBreaker,
    TripReason,
)

# =============================================================================
# TEST: _estimate_progress_from_content - False Positive Prevention
# =============================================================================


class TestProgressEstimationFalsePositivePrevention:
    """Tests that verify false positives are prevented in progress estimation."""

    def test_step_completed_no_does_not_trigger_high_progress(self):
        """
        CRITICAL FIX: 'STEP_COMPLETED: No' should NOT return 85%.

        This was the main bug - the word 'completed' was matched as a
        substring, causing false high progress readings.
        """
        text = "**STEP_COMPLETED:** No. This step is incomplete."
        progress = _estimate_progress_from_content(text)
        assert progress < 0.5, f"False positive! Got {progress:.0%}, expected < 50%"

    def test_step_completed_no_variations(self):
        """Test various formats of STEP_COMPLETED: No."""
        test_cases = [
            "STEP_COMPLETED: No",
            "step_completed: no",
            "STEP_COMPLETED: false",
            "step_completed: False",
            "STEP COMPLETED: No",
            "step completed: no",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress < 0.5, f"False positive for '{text}'! Got {progress:.0%}"

    def test_not_completed_patterns(self):
        """Test that 'not completed' patterns are handled correctly."""
        test_cases = [
            "The task is not completed yet.",
            "This step is not finished.",
            "Work is not done.",
            "The action hasn't been completed.",
            "We have not completed the task.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress < 0.5, f"False positive for '{text}'! Got {progress:.0%}"

    def test_incomplete_patterns(self):
        """Test that 'incomplete' patterns are handled correctly."""
        test_cases = [
            "This step is incomplete.",
            "The task remains incomplete.",
            "Incomplete - need more work.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress < 0.5, f"False positive for '{text}'! Got {progress:.0%}"

    def test_still_working_patterns(self):
        """Test that continuation indicators don't trigger high progress."""
        test_cases = [
            "Still working on the task.",
            "Still in progress.",
            "Continuing to work on this.",
            "Continue with the next step.",
            "Proceeding to the next action.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress < 0.5, f"False positive for '{text}'! Got {progress:.0%}"


# =============================================================================
# TEST: _estimate_progress_from_content - Correct Positive Detection
# =============================================================================


class TestProgressEstimationPositiveDetection:
    """Tests that verify correct detection of positive progress indicators."""

    def test_step_completed_yes_triggers_high_progress(self):
        """Test that 'STEP_COMPLETED: Yes' correctly returns high progress."""
        test_cases = [
            "STEP_COMPLETED: Yes",
            "step_completed: yes",
            "STEP_COMPLETED: true",
            "step_completed: True",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress >= 0.8, f"Expected high progress for '{text}', got {progress:.0%}"

    def test_task_completed_successfully_patterns(self):
        """Test that clear completion indicators work."""
        test_cases = [
            "Task completed successfully.",
            "Task is completed.",
            "Step completed.",
            "Successfully completed the task.",
            "All steps completed.",
            "All tasks done.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress >= 0.8, f"Expected high progress for '{text}', got {progress:.0%}"

    def test_blocked_patterns(self):
        """Test that blocked patterns return very low progress."""
        test_cases = [
            "Task is blocked.",
            "Stuck on this step.",
            "Cannot proceed further.",
            "Error occurred.",
            "Failed to complete.",
            "No progress made.",
            "Unable to proceed.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress <= 0.1, f"Expected very low progress for '{text}', got {progress:.0%}"

    def test_low_progress_patterns(self):
        """Test that early-stage indicators return low progress."""
        test_cases = [
            "Just started the task.",
            "Beginning the first step.",
            "Initial step in progress.",
            "Getting started with the work.",
            "Early stages of the project.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress <= 0.2, f"Expected low progress for '{text}', got {progress:.0%}"

    def test_medium_progress_patterns(self):
        """Test that medium progress indicators work correctly."""
        test_cases = [
            "Making progress on the task.",
            "Some progress has been made.",
            "Partial completion.",
            "In progress.",
            "Working on it.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert 0.3 <= progress <= 0.6, (
                f"Expected medium progress for '{text}', got {progress:.0%}"
            )

    def test_medium_high_progress_patterns(self):
        """Test that medium-high progress indicators work correctly."""
        test_cases = [
            "Good progress on the task.",
            "Significant progress made.",
            "Most of the work done.",
            "Nearly complete.",
            "Almost there.",
            "Halfway through.",
        ]
        for text in test_cases:
            progress = _estimate_progress_from_content(text)
            assert progress >= 0.6, (
                f"Expected medium-high progress for '{text}', got {progress:.0%}"
            )


# =============================================================================
# TEST: _extract_progress - Explicit Value Extraction
# =============================================================================


class TestProgressExtraction:
    """Tests for extracting explicit progress values."""

    def test_extract_labeled_progress_percentage(self):
        """Test extraction of PROGRESS: XX% format."""
        test_cases = [
            ("PROGRESS: 30%", 0.30),
            ("PROGRESS: 75%", 0.75),
            ("progress: 50%", 0.50),
            ("PROGRESS: 100%", 1.0),
            ("PROGRESS: 0%", 0.0),
        ]
        for text, expected in test_cases:
            progress = _extract_progress(text)
            assert abs(progress - expected) < 0.05, (
                f"Expected {expected}, got {progress} for '{text}'"
            )

    def test_extract_xml_format_progress(self):
        """Test extraction of <progress>XX</progress> format."""
        test_cases = [
            ("<progress>45</progress>", 0.45),
            ("<progress>75</progress>", 0.75),
            ("<PROGRESS>50</PROGRESS>", 0.50),
        ]
        for text, expected in test_cases:
            progress = _extract_progress(text)
            assert abs(progress - expected) < 0.05, (
                f"Expected {expected}, got {progress} for '{text}'"
            )

    def test_extract_natural_language_percentage(self):
        """Test extraction of natural language percentages."""
        test_cases = [
            ("We are 45% complete", 0.45),
            ("Progress is 30%", 0.30),
            ("Approximately 60% done", 0.60),
            ("About 75% finished", 0.75),
        ]
        for text, expected in test_cases:
            progress = _extract_progress(text)
            assert abs(progress - expected) < 0.05, (
                f"Expected {expected}, got {progress} for '{text}'"
            )

    def test_extract_fraction_format(self):
        """Test extraction of step X of Y format."""
        test_cases = [
            ("Step 2 of 4", 0.50),
            ("Completed 3 of 5 steps", 0.60),
            ("Step 1 of 10", 0.10),
        ]
        for text, expected in test_cases:
            progress = _extract_progress(text)
            assert abs(progress - expected) < 0.1, (
                f"Expected ~{expected}, got {progress} for '{text}'"
            )

    def test_fallback_to_content_estimation(self):
        """Test that extraction falls back to content estimation when no explicit value."""
        text = "Making good progress on the task."
        progress = _extract_progress(text)
        # Should use _estimate_progress_from_content as fallback
        assert 0.0 < progress < 1.0


# =============================================================================
# TEST: Circuit Breaker step_completed Handling
# =============================================================================


class TestCircuitBreakerStepCompleted:
    """Tests for circuit breaker's step_completed parameter handling."""

    def test_step_completed_resets_stall_counter(self):
        """Test that step_completed=True resets the progress stall counter."""
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        # Run iterations with same progress (would normally stall)
        for i in range(5):
            result = breaker.check(
                iteration=i,
                progress=0.5,  # Same progress
                step_completed=True,  # But step is completing
            )
            assert not result.tripped, f"Shouldn't trip when step_completed=True (iteration {i})"

    def test_step_completed_false_does_not_reset_stall(self):
        """Test that step_completed=False doesn't prevent stall detection."""
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        # Run iterations with same progress
        for i in range(10):
            result = breaker.check(
                iteration=i,
                progress=0.5,  # Same progress
                step_completed=False,  # Step not completing
            )
            if result.tripped:
                assert result.reason == TripReason.PROGRESS_STALL
                assert i >= 3, f"Tripped too early at iteration {i}"
                break
        else:
            pytest.fail("Should have tripped due to progress stall")

    def test_step_completed_none_uses_progress_tracking(self):
        """Test that step_completed=None falls back to normal progress tracking."""
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        # Run iterations with same progress (None step_completed)
        for i in range(10):
            result = breaker.check(
                iteration=i,
                progress=0.5,  # Same progress
                step_completed=None,  # Not specified
            )
            if result.tripped:
                assert result.reason == TripReason.PROGRESS_STALL
                assert i >= 3, f"Tripped too early at iteration {i}"
                break
        else:
            pytest.fail("Should have tripped due to progress stall")

    def test_mixed_step_completed_values(self):
        """Test behavior with mixed step_completed values across iterations."""
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        # Iterations: False, False, True (reset), False, False, False (should trip)
        step_completed_sequence = [False, False, True, False, False, False, False]

        tripped = False
        trip_iteration = -1
        for i, step_done in enumerate(step_completed_sequence):
            result = breaker.check(
                iteration=i,
                progress=0.5,
                step_completed=step_done,
            )
            if result.tripped:
                tripped = True
                trip_iteration = i
                break

        assert tripped, "Should have tripped eventually"
        # After reset at i=2, should trip at i=6 (3 stalls after reset)
        assert trip_iteration >= 5, f"Tripped at iteration {trip_iteration}, expected >= 5"


# =============================================================================
# TEST: Circuit Breaker API Compatibility
# =============================================================================


class TestCircuitBreakerAPICompatibility:
    """Tests to ensure backward compatibility of circuit breaker API."""

    def test_check_without_step_completed_works(self):
        """Test that check() works without step_completed parameter (backward compat)."""
        breaker = AgentCircuitBreaker(max_iterations=5)
        breaker.start()

        # Old-style call without step_completed
        result = breaker.check(
            iteration=0,
            progress=0.5,
            error=None,
            cost=0.1,
        )
        assert not result.tripped

    def test_check_with_all_parameters(self):
        """Test that check() works with all parameters including step_completed."""
        breaker = AgentCircuitBreaker(max_iterations=5)
        breaker.start()

        result = breaker.check(
            iteration=0,
            progress=0.5,
            error=None,
            cost=0.1,
            context={"test": "data"},
            step_completed=True,
        )
        assert not result.tripped


# =============================================================================
# TEST: Integration - Progress Estimation with Circuit Breaker
# =============================================================================


class TestProgressCircuitBreakerIntegration:
    """Integration tests for progress estimation and circuit breaker."""

    def test_step_completed_no_with_circuit_breaker(self):
        """
        Integration test: 'STEP_COMPLETED: No' should not cause
        circuit breaker to trip on false high progress.
        """
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        # Simulate iterations where model says "STEP_COMPLETED: No"
        # Progress estimation should return < 0.5, not 0.85
        for i in range(5):
            text = f"Iteration {i}: STEP_COMPLETED: No. Still working on the task."
            progress = _estimate_progress_from_content(text)

            assert progress < 0.5, (
                f"Progress should be < 50% for 'STEP_COMPLETED: No', got {progress:.0%}"
            )

            # Since step_completed is False based on parsing, pass it
            result = breaker.check(
                iteration=i,
                progress=progress,
                step_completed=False,
            )

            # May or may not trip depending on stall threshold
            # But progress should NOT be stuck at 85%

    def test_realistic_agent_scenario(self):
        """
        Test a realistic scenario where agent makes progress across iterations.
        """
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=5,
            max_iterations=10,
        )
        breaker.start()

        # Simulate realistic iteration progression
        iterations = [
            ("STEP_COMPLETED: No. Just starting.", False, 0.35),
            ("Making some progress. In progress.", None, 0.45),
            ("STEP_COMPLETED: Yes. Moving to next step.", True, 0.85),
            ("STEP_COMPLETED: No. Starting new step.", False, 0.35),
            ("Good progress on this step.", None, 0.65),
            ("STEP_COMPLETED: Yes. Task almost done.", True, 0.85),
            ("Task completed successfully.", True, 0.85),
        ]

        for i, (text, expected_step_completed, expected_range_center) in enumerate(iterations):
            progress = _estimate_progress_from_content(text)

            # Verify progress is in expected range
            assert abs(progress - expected_range_center) < 0.3, (
                f"Progress {progress:.0%} not near {expected_range_center:.0%} for '{text}'"
            )

            result = breaker.check(
                iteration=i,
                progress=progress,
                step_completed=expected_step_completed,
            )

            # Should not trip during normal progression
            assert not result.tripped, f"Should not trip on iteration {i} with text: {text}"


# =============================================================================
# TEST: Edge Cases
# =============================================================================


class TestProgressEstimationEdgeCases:
    """Tests for edge cases in progress estimation."""

    def test_empty_string(self):
        """Test handling of empty string."""
        progress = _estimate_progress_from_content("")
        assert progress == 0.35, "Empty string should return default moderate progress"

    def test_mixed_signals(self):
        """Test handling of text with mixed positive and negative signals."""
        # Negative context should take precedence
        text = "Good progress but task is not completed yet."
        progress = _estimate_progress_from_content(text)
        assert progress < 0.5, (
            f"Mixed signals with 'not completed' should be < 50%, got {progress:.0%}"
        )

    def test_long_text_with_one_keyword(self):
        """Test that keywords are found in long text."""
        text = """
        This is a very long reflection text that contains a lot of filler content.
        We have been working on this task for quite some time now.
        There have been many steps and many observations.
        After all this work, the task is finally completed successfully.
        We achieved our goal.
        """
        progress = _estimate_progress_from_content(text)
        assert progress >= 0.8, (
            f"Long text with 'completed successfully' should be high, got {progress:.0%}"
        )

    def test_case_insensitivity(self):
        """Test that detection is case insensitive."""
        test_cases = [
            "COMPLETED",
            "Completed",
            "completed",
            "CoMpLeTeD",
        ]
        for text in test_cases:
            # Note: We're testing the base word, which should trigger
            # high progress only if there's no negative context
            # Adding positive context to ensure detection
            text_with_context = f"Task {text} successfully."
            progress = _estimate_progress_from_content(text_with_context)
            assert progress >= 0.8, f"Case variation '{text}' with context should be detected"

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        text = "Tâche terminée avec succès. Task completed. ✅"
        progress = _estimate_progress_from_content(text)
        # Should detect "completed"
        assert progress >= 0.8


# =============================================================================
# TEST: Regression Tests for the Specific Bug
# =============================================================================


class TestRegressionBugFixes:
    """Regression tests for specific bugs that were fixed."""

    def test_bug_step_completed_no_returning_85_percent(self):
        """
        REGRESSION TEST: The original bug was that text containing
        'STEP_COMPLETED: No' returned 85% progress because 'completed'
        was matched as a substring.

        This test ensures the bug does not regress.
        """
        problematic_texts = [
            "**STEP_COMPLETED:** No. This step is incomplete.",
            "STEP_COMPLETED: No\nThe task needs more work.",
            "Evaluation:\n- Progress: some\n- STEP_COMPLETED: No",
            "Analysis shows STEP_COMPLETED: No, we need to continue.",
        ]

        for text in problematic_texts:
            progress = _estimate_progress_from_content(text)
            # The bug would return 0.85 for these
            assert progress != 0.85, f"Bug regression: got 85% for '{text[:50]}...'"
            assert progress < 0.5, f"Should be < 50% for negative context, got {progress:.0%}"

    def test_bug_circuit_breaker_false_stall_on_step_completed(self):
        """
        REGRESSION TEST: Circuit breaker should not trip on progress stall
        when step_completed=True is passed, even if progress number doesn't change.

        This was happening because the old code only looked at progress delta.
        """
        breaker = AgentCircuitBreaker(
            progress_stall_threshold=2,  # Low threshold to catch bug
            max_iterations=100,
        )
        breaker.start()

        # All same progress, but step_completed=True each time
        for i in range(10):
            result = breaker.check(
                iteration=i,
                progress=0.5,  # Never changes
                step_completed=True,  # But step IS completing
            )
            assert not result.tripped, (
                f"Bug regression: tripped at iteration {i} despite step_completed=True"
            )


# =============================================================================
# RUN SELF-TESTS
# =============================================================================


if __name__ == "__main__":
    print("Running Phase 2 Progress Estimation tests...\n")

    # Quick smoke test
    test_text = "**STEP_COMPLETED:** No. This step is incomplete."
    progress = _estimate_progress_from_content(test_text)

    if progress < 0.5:
        print(f"✅ Bug fix verified: '{test_text[:40]}...' → {progress:.0%} (expected < 50%)")
    else:
        print(f"❌ Bug NOT fixed: '{test_text[:40]}...' → {progress:.0%} (should be < 50%)")
        exit(1)

    # Test circuit breaker
    breaker = AgentCircuitBreaker(progress_stall_threshold=2, max_iterations=100)
    breaker.start()

    for i in range(5):
        result = breaker.check(iteration=i, progress=0.5, step_completed=True)
        if result.tripped:
            print(f"❌ Circuit breaker tripped incorrectly at iteration {i}")
            exit(1)

    print("✅ Circuit breaker step_completed fix verified")

    print("\n" + "=" * 60)
    print("All smoke tests passed! Run pytest for full test suite.")
