# tests/integration/test_commandcode_live.py
"""Live tests for the Command Code provider (opt-in; needs ``CMD_API_KEY``).

Skipped automatically when no key is set (CI has none). Run locally with::

    set -a; source /path/to/your/.env; set +a
    pytest tests/integration/test_commandcode_live.py -m integration -q

Cost: chat tests use Command Code's free models where they can
(``inkling-2-small:free``, 200 requests/day). Claude tests use
``claude-haiku-5-5`` ($0.10 / $0.50 per 1M). One full run costs well under a
cent. Both transports are exercised.
"""

from __future__ import annotations

import os

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers.commandcode_provider import API_KEY_ENV_VARS, CommandCodeProvider
from llmcore.providers.typesafe_provider import Noul

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        not any(os.environ.get(name) for name in API_KEY_ENV_VARS),
        reason="no Command Code key (CMD_API_KEY / COMMAND_CODE_API_KEY / COMMANDCODE_API_KEY)",
    ),
]

FREE = "inkling-2-small:free"
CLAUDE = "claude-haiku-5-5"
CTX = [
    Message(role=Role.SYSTEM, content="Be terse."),
    Message(role=Role.USER, content="What is 17*23? Answer with the number only."),
]
WEATHER = Tool(
    name="get_weather",
    description="Current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@pytest.fixture(params=["httpx", "sdk"])
async def provider(request):
    p = CommandCodeProvider({"transport": request.param})
    yield p
    await p.close()


async def test_catalog(provider):
    count = await provider.refresh_catalog()
    assert count > 50
    assert provider.route_for(CLAUDE) == "messages"
    assert provider.route_for(FREE) == "chat"
    assert provider.resolve_model("kimi-k3") == "moonshotai/Kimi-K3"


async def test_chat_open_model(provider):
    resp = await provider.chat_completion(CTX, FREE, max_tokens=400)
    assert "391" in provider.extract_response_content(resp)
    usage = provider.extract_usage_details(resp)
    assert usage["prompt_tokens"] > 0 and usage["route"] == "chat"


async def test_chat_open_model_streaming(provider):
    stream = await provider.chat_completion(CTX, FREE, stream=True, max_tokens=400)
    assert "391" in "".join([provider.extract_delta_content(c) async for c in stream])


async def test_claude(provider):
    resp = await provider.chat_completion(CTX, CLAUDE, max_tokens=32, effort="low")
    assert "391" in provider.extract_response_content(resp)
    assert provider.extract_usage_details(resp)["route"] == "messages"


async def test_claude_streaming(provider):
    stream = await provider.chat_completion(CTX, CLAUDE, stream=True, max_tokens=32)
    assert "391" in "".join([provider.extract_delta_content(c) async for c in stream])


async def test_claude_tools(provider):
    ctx = [Message(role=Role.USER, content="What's the weather in Lisbon? Use the tool.")]
    resp = await provider.chat_completion(ctx, CLAUDE, tools=[WEATHER], max_tokens=200)
    calls = provider.extract_tool_calls(resp)
    assert calls and calls[0].name == "get_weather"


async def test_errors_are_mapped(provider):
    with pytest.raises(ProviderError) as info:
        await provider.chat_completion(CTX, "nope/not-a-model", max_tokens=8)
    assert info.value.failure_kind == "model_not_found"
    with pytest.raises(ProviderError) as info:
        # Free models have no zero-data-retention upstream.
        await provider.chat_completion(CTX, FREE, max_tokens=8, zdr=True)
    assert info.value.error_code == "cmd_zdr_no_providers"


async def test_bad_key_is_an_auth_failure():
    p = CommandCodeProvider({"api_key": "definitely-not-a-key"})
    try:
        with pytest.raises(ProviderError) as info:
            await p.chat_completion(CTX, FREE, max_tokens=8)
        assert info.value.failure_kind == "auth"
    finally:
        await p.close()


async def test_system_one():
    p = CommandCodeProvider({})
    try:
        res = await p.system_one(
            "The user said: the build is green and all tests pass.",
            {"done": Noul(instructions="Is the work finished?")},
        )
        assert 0.0 <= res.answers_dict()["done"]["noul"] <= 1.0
    finally:
        await p.close()
