---
id: llmcore/darwin/tdd/spec_generation
name: TDD spec generation
version: 1.0.0
tags: [llmcore.builtin, darwin, tdd]
description: Generate test specifications (JSON output).
variables:
  framework:
    type: multiline
    required: true
  language:
    type: multiline
    required: true
  min_tests:
    type: multiline
    required: true
  requirements:
    type: multiline
    required: true
---

# SYSTEM
You are an expert test engineer. Output valid JSON only.

# USER
You are an expert test engineer. Generate comprehensive test specifications for the following requirements.

Requirements:
{{ requirements }}

Language: {{ language }}
Test Framework: {{ framework }}
Minimum Tests: {{ min_tests }}

Generate test specifications covering:
1. Happy path / normal operation (at least 2 tests)
2. Edge cases (empty inputs, large inputs, boundary values) (at least 2 tests)
3. Error handling (invalid inputs, exceptions) (at least 1 test)
4. Integration scenarios (if applicable)

For each test, provide a JSON object with:
- name: A descriptive test name starting with test_ (e.g., test_add_positive_numbers)
- description: What the test verifies
- test_type: One of "unit", "integration", "edge_case", "error"
- inputs: The test inputs as a dict (e.g., {"a": 1, "b": 2})
- expected_output: What the function should return (can be null for void functions)
- expected_behavior: Any side effects or state changes to verify
- expected_exception: Exception type if testing error handling (e.g., "ValueError")
- priority: 1 (must pass), 2 (should pass), or 3 (nice to have)

Output as a JSON array of test specifications. Output ONLY valid JSON, no explanations or markdown.
