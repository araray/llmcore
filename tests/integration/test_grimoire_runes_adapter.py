from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llmcore.agents.tools import ToolManager
from llmcore.integration import (
    register_rune_tools,
    rune_command_metadata,
    rune_command_to_tool,
)
from llmcore.models import ToolCall


def test_grimoire_rune_command_converts_to_llmcore_tool() -> None:
    from grimoire.models import CommandSpec, ParamSpec, Permission, RiskLevel, RuneSpec

    rune = RuneSpec(
        id="devtools/git",
        name="Git diagnostics",
        tags=["devtools", "vcs"],
        permissions=[Permission.READ_FS],
        risk_level=RiskLevel.LOW,
        commands=[
            CommandSpec(
                name="status",
                summary="Show working tree status",
                params=[ParamSpec(name="porcelain", type="bool", default=True)],
                execution_target="local",
            )
        ],
        mappings={"wairu.tool_name": "git_status"},
    )
    command = rune.commands[0]

    tool = rune_command_to_tool(rune, command)
    metadata = rune_command_metadata(rune, command)

    assert tool.name == "devtools__git__status"
    assert tool.description == "Show working tree status"
    assert tool.parameters["properties"]["porcelain"]["type"] == "boolean"
    assert metadata == {
        "source": "grimoire.rune",
        "rune_id": "devtools/git",
        "command_name": "status",
        "risk_level": "low",
        "requires_approval": False,
        "permissions": ["read_fs"],
        "tags": ["devtools", "vcs"],
        "side_effects": [],
        "execution_target": "local",
        "content_hash": rune.content_hash,
        "mappings": {"wairu.tool_name": "git_status"},
    }


@pytest.mark.asyncio
async def test_register_rune_tools_registers_runtime_implementation() -> None:
    from grimoire.models import CommandSpec, ParamSpec, Permission, RiskLevel, RuneSpec

    rune = RuneSpec(
        id="wairu/shell",
        name="Shell",
        tags=["shell"],
        permissions=[Permission.EXEC, Permission.WRITE_FS],
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
        commands=[
            CommandSpec(
                name="run",
                summary="Run a command",
                params=[ParamSpec(name="command", type="string", required=True)],
                requires_approval=True,
                execution_target="sandbox",
            )
        ],
    )
    tool_manager = ToolManager(MagicMock(), MagicMock())

    def run_shell(command: str) -> str:
        return f"ran: {command}"

    count = register_rune_tools(
        tool_manager,
        [rune],
        {("wairu/shell", "run"): run_shell},
    )

    assert count == 1
    assert tool_manager.is_tool_loaded("wairu__shell__run")
    inventory = tool_manager.get_tool_inventory()
    assert inventory[0]["name"] == "wairu__shell__run"
    assert inventory[0]["source"] == "grimoire.rune"
    assert inventory[0]["risk_level"] == "high"
    assert inventory[0]["requires_approval"] is True
    assert inventory[0]["permissions"] == ["exec", "write_fs"]
    assert inventory[0]["execution_target"] == "sandbox"

    result = await tool_manager.execute_tool(
        ToolCall(
            id="call-1",
            name="wairu__shell__run",
            arguments={"command": "echo ok"},
        )
    )

    assert result.is_error is False
    assert result.content == "ran: echo ok"


def test_register_rune_tools_requires_host_implementation() -> None:
    from grimoire.models import CommandSpec, RuneSpec

    rune = RuneSpec(
        id="devtools/git",
        name="Git diagnostics",
        commands=[CommandSpec(name="status")],
    )

    with pytest.raises(ValueError, match="No implementation supplied"):
        register_rune_tools(ToolManager(MagicMock(), MagicMock()), [rune], {})
