from __future__ import annotations

from llmcore.agents.cognitive.phases.think import _parse_think_response


def test_parse_think_response_extracts_openai_native_tool_call() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/tmp/example.txt"}',
                            },
                        }
                    ]
                }
            }
        ],
        "usage": {"total_tokens": 12},
    }

    output = _parse_think_response(
        response_text="Thought: I should inspect the file.",
        response_dict=response,
        tool_manager=object(),
    )

    assert output.proposed_action is not None
    assert output.proposed_action.id == "call_abc"
    assert output.proposed_action.name == "read_file"
    assert output.proposed_action.arguments == {"path": "/tmp/example.txt"}
    assert output.reasoning_tokens == 12


def test_parse_think_response_extracts_ollama_native_tool_call() -> None:
    response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd", "timeout": 5},
                    }
                }
            ],
        }
    }

    output = _parse_think_response(
        response_text="",
        response_dict=response,
        tool_manager=object(),
    )

    assert output.proposed_action is not None
    assert output.proposed_action.id == "call_native_0"
    assert output.proposed_action.name == "run_command"
    assert output.proposed_action.arguments == {"command": "pwd", "timeout": 5}


def test_parse_think_response_wraps_malformed_native_arguments() -> None:
    response = {
        "tool_calls": [
            {
                "name": "search",
                "arguments": "not-json but still useful",
            }
        ]
    }

    output = _parse_think_response(
        response_text="Thought: search with the raw input.",
        response_dict=response,
        tool_manager=object(),
    )

    assert output.proposed_action is not None
    assert output.proposed_action.name == "search"
    assert output.proposed_action.arguments == {"input": "not-json but still useful"}


def test_parse_think_response_prefers_native_tool_call_over_text_action() -> None:
    response = {
        "tool_calls": [
            {
                "name": "native_tool",
                "arguments": {"value": "native"},
            }
        ]
    }

    output = _parse_think_response(
        response_text="""Thought: use a tool.
Action: text_tool
Action Input: {"value": "text"}
""",
        response_dict=response,
        tool_manager=object(),
    )

    assert output.proposed_action is not None
    assert output.proposed_action.name == "native_tool"
    assert output.proposed_action.arguments == {"value": "native"}
