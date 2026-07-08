---
id: llmcore/activity/execute
name: Activity execution prompt
version: 1.0.0
tags: [llmcore.builtin, activity]
description: Per-iteration prompt for activity-based (XML) execution.
variables:
  goal:
    type: multiline
    required: true
    description: The task goal
  current_step:
    type: string
    required: true
    description: Current step being executed
  activities_section:
    type: multiline
    required: false
    default: ""
    description: Pre-formatted AVAILABLE ACTIVITIES block (else empty)
  history_section:
    type: multiline
    required: false
    default: ""
    description: Pre-formatted RECENT HISTORY block (else empty)
  context_section:
    type: multiline
    required: false
    default: ""
    description: Pre-formatted RELEVANT CONTEXT block (else empty)
---

# USER
You are solving this task using the activity system:

GOAL: {{ goal }}

CURRENT STEP: {{ current_step }}
{{ activities_section }}{{ history_section }}{{ context_section }}

Use the activity system to accomplish your goal. Format your response as:

1. Think about what to do next
2. Request an activity using the XML format

Example:
I need to read the configuration file to understand the current settings.

<activity_request>
    <activity>file_read</activity>
    <parameters>
        <path>/path/to/config.yaml</path>
    </parameters>
    <reasoning>Reading configuration to understand current settings</reasoning>
</activity_request>

OR if the task is complete:

I have completed the task successfully.

<activity_request>
    <activity>final_answer</activity>
    <parameters>
        <answer>The calculation result is 42.</answer>
    </parameters>
    <reasoning>Task complete, providing final answer</reasoning>
</activity_request>

Respond now:
