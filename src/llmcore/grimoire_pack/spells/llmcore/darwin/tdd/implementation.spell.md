---
id: llmcore/darwin/tdd/implementation
name: TDD implementation generation
version: 1.0.0
tags: [llmcore.builtin, darwin, tdd]
description: Generate implementation code that passes the tests.
variables:
  language:
    type: multiline
    required: true
  previous_implementation:
    type: multiline
    required: true
  requirements:
    type: multiline
    required: true
  test_failures:
    type: multiline
    required: true
  test_file:
    type: multiline
    required: true
---

# SYSTEM
You are an expert {{ language }} developer. Output only implementation code.

# USER
Generate implementation code that passes the following tests.

Requirements:
{{ requirements }}

Test File Content:
```{{ language }}
{{ test_file }}
```

Previous Implementation (if any):
{{ previous_implementation }}

Test Failures (if any):
{{ test_failures }}

Generate implementation code that:
1. Satisfies all the requirements
2. Passes all the tests above
3. Follows best practices for {{ language }}
4. Includes proper error handling
5. Has clear docstrings/comments

Output only the implementation code, no explanations or markdown backticks.
