---
id: llmcore/darwin/tdd/test_generation
name: TDD test generation
version: 1.0.0
tags: [llmcore.builtin, darwin, tdd]
description: Generate executable test code for one spec.
variables:
  description:
    type: multiline
    required: true
  expected_behavior:
    type: multiline
    required: true
  expected_exception:
    type: multiline
    required: true
  expected_output:
    type: multiline
    required: true
  framework:
    type: multiline
    required: true
  inputs:
    type: multiline
    required: true
  language:
    type: multiline
    required: true
  name:
    type: multiline
    required: true
  test_type:
    type: multiline
    required: true
---

# SYSTEM
You are an expert {{ language }} developer. Output only code.

# USER
Generate executable test code for the following specification.

Specification:
- Name: {{ name }}
- Description: {{ description }}
- Type: {{ test_type }}
- Inputs: {{ inputs }}
- Expected Output: {{ expected_output }}
- Expected Behavior: {{ expected_behavior }}
- Expected Exception: {{ expected_exception }}

Language: {{ language }}
Framework: {{ framework }}

Generate a complete, runnable test function. Include:
1. All necessary imports at the top
2. Any required fixtures or setup
3. The test function with clear assertions
4. Helpful error messages on assertion failure

For pytest:
- Use pytest.raises() for exception testing
- Use descriptive assertion messages
- Use fixtures where appropriate

Output only the Python code, no explanations or markdown backticks.
