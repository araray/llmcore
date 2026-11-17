# tests/providers/test_huggingface_current_api.py
"""HuggingFaceProvider against the live router (huggingface_hub 2.4, 2026-11-16).

- the router's /v1/models gives each served model's live providers with their
  context windows (Llama-3.3-70B: 12,288 on one of three);
- reasoning comes back as ``reasoning``, and the router takes it back as
  ``reasoning``; ``reasoning_content`` is a 400 on the default route;
- reasoning_effort / top_k have no SDK keyword and must go in extra_body;
- the packaged config's scalar keys used to sit after a subtable header, so
  TOML put them inside ``endpoints``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("huggingface_hub")

from llmcore.models import Message, Role  # noqa: E402
from llmcore.providers.huggingface_provider import HuggingFaceProvider  # noqa: E402

LLAMA = {"id": "meta-llama/Llama-3.3-70B-Instruct", "owned_by": "meta-llama",
         "architecture": {"input_modalities": ["text"]},
         "providers": [{"provider": "novita", "status": "live", "context_length": 12288, "supports_tools": False},
                       {"provider": "ovhcloud", "status": "live", "context_length": 131072, "supports_tools": True},
                       {"provider": "gone", "status": "error", "context_length": 999999}]}


@pytest.fixture
def provider() -> HuggingFaceProvider:
    return HuggingFaceProvider({"api_key": "hf_x"})


def test_router_entry_drives_context_per_route(provider):
    details = provider._router_details(LLAMA)
    assert details.context_length == 12288 and details.supports_tools  # any live provider
    assert provider.get_max_context_length("meta-llama/Llama-3.3-70B-Instruct") == 12288
    assert provider.get_max_context_length("meta-llama/Llama-3.3-70B-Instruct:ovhcloud") == 131072
    assert provider.get_max_context_length("meta-llama/Llama-3.3-70B-Instruct:cheapest") == 12288


async def test_extra_body_carries_effort_and_top_k_and_streams_request_usage(provider):
    provider._client.chat_completion = AsyncMock(return_value={"choices": []})
    await provider.chat_completion([Message(role=Role.USER, content="hi")], effort="low", top_k=5, temperature=0.2)
    kwargs = provider._client.chat_completion.await_args.kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "low", "top_k": 5}
    assert kwargs["temperature"] == 0.2

    async def empty():
        if False:
            yield

    provider._client.chat_completion = AsyncMock(return_value=empty())
    await provider.chat_completion([Message(role=Role.USER, content="hi")], stream=True)
    assert provider._client.chat_completion.await_args.kwargs["stream_options"] == {"include_usage": True}


def test_tool_calls_and_reasoning_replay(provider):
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    response = {"choices": [{"message": {"role": "assistant", "content": None, "reasoning": "use f",
                                         "tool_calls": [call]}}]}
    meta = provider.assistant_replay_metadata(response)
    payload = provider._build_message_payload(Message(role=Role.ASSISTANT, content="", tool_calls=[call], metadata=meta))
    assert payload["tool_calls"] == [call]  # first-class tool_calls used to be ignored
    assert payload["reasoning"] == "use f" and "reasoning_content" not in payload


async def test_stream_reasoning_lands_on_the_finish_chunk(provider):
    async def chunks():
        yield {"choices": [{"index": 0, "delta": {"reasoning": "use "}}]}
        yield {"choices": [{"index": 0, "delta": {"reasoning": "f"}, "finish_reason": "tool_calls"}]}

    out = [c async for c in provider._stream_wrapper(chunks())]
    assert provider.assistant_replay_metadata(out[-1]) == {"reasoning_content": "use f"}


def test_packaged_config_keys_land_on_the_provider():
    config = Path(__file__).resolve().parents[2] / "src" / "llmcore" / "config" / "default_config.toml"
    hf = tomllib.loads(config.read_text())["providers"]["huggingface"]
    assert hf["default_model"] == "openai/gpt-oss-120b"
    assert hf["api_key_env_var"] == "HF_TOKEN"
    assert hf["endpoints"] == {}
