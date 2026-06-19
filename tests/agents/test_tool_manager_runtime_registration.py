from unittest.mock import MagicMock

import pytest

from llmcore.agents.tools import ToolManager
from llmcore.models import Tool, ToolCall


@pytest.mark.asyncio
async def test_register_runtime_tool_executes_async_callable():
    manager = ToolManager(MagicMock(), MagicMock())

    async def echo(value: str) -> str:
        return f"echo:{value}"

    manager.register_runtime_tool(
        Tool(
            name="echo",
            description="Echo a value",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        ),
        "tests.runtime.echo",
        echo,
    )

    result = await manager.execute_tool(
        ToolCall(id="call-1", name="echo", arguments={"value": "x"})
    )

    assert result.is_error is False
    assert result.content == "echo:x"


def test_register_runtime_tool_rejects_duplicates_when_replace_false():
    manager = ToolManager(MagicMock(), MagicMock())
    tool = Tool(name="echo", description="Echo", parameters={})

    manager.register_runtime_tool(tool, "tests.runtime.duplicate", lambda: "one")

    with pytest.raises(ValueError, match="Implementation key"):
        manager.register_runtime_tool(
            tool,
            "tests.runtime.duplicate",
            lambda: "two",
            replace=False,
        )

    with pytest.raises(ValueError, match="Tool 'echo'"):
        manager.register_runtime_tool(
            tool,
            "tests.runtime.other",
            lambda: "two",
            replace=False,
        )


@pytest.mark.asyncio
async def test_register_runtime_tool_replaces_existing_tool_mapping():
    manager = ToolManager(MagicMock(), MagicMock())
    tool = Tool(name="echo", description="Echo", parameters={})

    manager.register_runtime_tool(tool, "tests.runtime.replace.one", lambda: "one")
    manager.register_runtime_tool(tool, "tests.runtime.replace.two", lambda: "two")

    result = await manager.execute_tool(ToolCall(id="call-1", name="echo", arguments={}))

    assert result.is_error is False
    assert result.content == "two"
    assert manager.get_implementation_key("echo") == "tests.runtime.replace.two"



def test_tool_inventory_is_lightweight_and_selects_full_schemas():
    manager = ToolManager(MagicMock(), MagicMock())
    tool = Tool(
        name="long_tool",
        description="A" * 240,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    )
    manager.register_runtime_tool(tool, "tests.runtime.inventory", lambda: "ok")

    inventory = manager.get_tool_inventory(max_description_chars=60)

    assert inventory == [
        {
            "name": "long_tool",
            "description": "A" * 46 + "...[truncated]",
            "implementation_key": "tests.runtime.inventory",
            "parameter_names": ["path", "limit"],
            "required_parameters": ["path"],
        }
    ]
    assert manager.get_tool_definitions(["long_tool"]) == [tool]
    assert manager.get_tool_definitions(["missing"]) == []


def test_tool_inventory_can_include_full_parameters_on_request():
    manager = ToolManager(MagicMock(), MagicMock())
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}
    manager.register_runtime_tool(
        Tool(name="echo", description="Echo", parameters=parameters),
        "tests.runtime.inventory.parameters",
        lambda: "ok",
    )

    inventory = manager.get_tool_inventory(include_parameters=True)

    assert inventory[0]["parameters"] == parameters


def test_tool_inventory_includes_runtime_metadata_without_changing_tool_schema():
    manager = ToolManager(MagicMock(), MagicMock())
    tool = Tool(name="delete_file", description="Delete a file", parameters={})

    manager.register_runtime_tool(
        tool,
        "tests.runtime.inventory.metadata",
        lambda: "ok",
        metadata={
            "requires_approval": True,
            "risk_level": "high",
            "owasp": ["A01:2021-Broken Access Control"],
        },
    )

    inventory = manager.get_tool_inventory()

    assert manager.get_tool_definitions(["delete_file"]) == [tool]
    assert inventory[0]["requires_approval"] is True
    assert inventory[0]["risk_level"] == "high"
    assert inventory[0]["owasp"] == ["A01:2021-Broken Access Control"]
    assert inventory[0]["metadata"]["requires_approval"] is True
