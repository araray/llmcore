# tests/integration/providers/test_vllm_live.py
"""
Live integration tests for :class:`VLLMProvider`.

These tests exercise the provider against a real vLLM server. They are
**skipped by default** and enabled only when the ``VLLM_TEST_SERVER_URL``
environment variable is set to the server's base URL (including the
``/v1`` suffix).

Running locally
===============
Spin up a small model under vLLM::

    docker run --gpus all --rm -p 8000:8000 \\
        -v ~/.cache/huggingface:/root/.cache/huggingface \\
        vllm/vllm-openai:latest \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes

Then in a separate shell::

    export VLLM_TEST_SERVER_URL=http://localhost:8000/v1
    export VLLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct
    pytest tests/integration/providers/test_vllm_live.py -v

Optional environment variables
==============================
``VLLM_TEST_SERVER_URL``
    Base URL including ``/v1``. Required to activate this suite.

``VLLM_TEST_MODEL``
    Model identifier the server is serving. Defaults to
    ``"Qwen/Qwen2.5-0.5B-Instruct"``.

``VLLM_TEST_API_KEY``
    Bearer token, only if the server was launched with ``--api-key``.

Nothing about these tests is destructive: they issue single-turn chat
completions, a streamed completion, a models list, and (if tool-calling
is enabled on the server) one tool-calling probe.
"""

from __future__ import annotations

import os

import pytest

from llmcore.models import Message, Role, Tool
from llmcore.providers.vllm_provider import VLLMProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("VLLM_TEST_SERVER_URL"),
    reason="VLLM_TEST_SERVER_URL not set — live integration tests skipped.",
)


@pytest.fixture
def server_url() -> str:
    return os.environ["VLLM_TEST_SERVER_URL"].rstrip("/")


@pytest.fixture
def model_id() -> str:
    return os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture
def provider(server_url: str, model_id: str) -> VLLMProvider:
    """Construct a real VLLMProvider against the live server."""
    config: dict = {
        "base_url": server_url,
        "default_model": model_id,
        "timeout": 120,
    }
    api_key = os.environ.get("VLLM_TEST_API_KEY")
    if api_key:
        config["api_key"] = api_key
    return VLLMProvider(config)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_models_list(provider: VLLMProvider, model_id: str) -> None:
    """``/v1/models`` returns at least the served model, with a
    positive ``context_length`` drawn from ``max_model_len``."""
    models = await provider.get_models_details()

    assert len(models) >= 1
    ids = [m.id for m in models]
    assert model_id in ids, (
        f"Configured VLLM_TEST_MODEL={model_id!r} not present in server model list {ids}"
    )

    served = next(m for m in models if m.id == model_id)
    assert served.provider_name == "vllm"
    assert served.supports_streaming is True
    # max_model_len is authoritative; if the server reports it, trust it
    assert served.context_length > 0
    # Cache should be populated as a side effect
    assert provider._vllm_model_cache.get(model_id) == served.context_length


@pytest.mark.asyncio
async def test_live_get_max_context_length_uses_cache(
    provider: VLLMProvider, model_id: str
) -> None:
    """After discovery, get_max_context_length returns the cached value
    (not the 4096 fallback)."""
    await provider.get_models_details()
    ctx = provider.get_max_context_length(model_id)
    assert ctx > 0
    # 4096 is the pessimistic last-resort fallback; real vLLM models
    # always advertise more than that.
    assert ctx != 4096


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_chat_completion_single_turn(provider: VLLMProvider) -> None:
    """A minimal single-turn chat returns a non-empty assistant message."""
    messages = [
        Message(role=Role.USER, content="Reply with exactly the single word: pong."),
    ]
    response = await provider.chat_completion(messages, max_tokens=16, temperature=0.0)

    assert isinstance(response, dict)
    content = provider.extract_response_content(response)
    assert content, "Empty response content from vLLM"
    # We don't assert on exact text (model behaviour varies) — just that
    # something plausibly related arrived.
    assert isinstance(content, str)


@pytest.mark.asyncio
async def test_live_chat_completion_streaming(provider: VLLMProvider) -> None:
    """Streaming mode yields at least one non-empty delta chunk."""
    messages = [
        Message(role=Role.USER, content="Count: 1, 2, 3."),
    ]
    stream = await provider.chat_completion(messages, stream=True, max_tokens=32, temperature=0.0)

    deltas: list[str] = []
    async for chunk in stream:
        delta = provider.extract_delta_content(chunk)
        if delta:
            deltas.append(delta)

    assert deltas, "No deltas received from streaming chat completion"
    full = "".join(deltas)
    assert full.strip(), "Concatenated deltas are empty after stripping whitespace"


@pytest.mark.asyncio
async def test_live_usage_fields_populated(provider: VLLMProvider) -> None:
    """vLLM reports prompt_tokens and completion_tokens in usage."""
    messages = [Message(role=Role.USER, content="Say hi.")]
    response = await provider.chat_completion(messages, max_tokens=8, temperature=0.0)
    usage = provider.extract_usage_details(response)

    assert usage.get("prompt_tokens") is not None
    assert usage.get("prompt_tokens") > 0
    assert usage.get("completion_tokens") is not None


# ---------------------------------------------------------------------------
# Structured outputs (vLLM-specific; best-effort)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_guided_choice(provider: VLLMProvider) -> None:
    """vLLM's ``guided_choice`` constrains the output to one of the given
    strings. Works on any vLLM build with a guided-decoding backend
    available; skipped if the server rejects the parameter."""
    messages = [
        Message(
            role=Role.USER,
            content=(
                "Output exactly one of: yes, no, maybe. Question: Is the sky blue on a clear day?"
            ),
        ),
    ]
    try:
        response = await provider.chat_completion(
            messages,
            max_tokens=8,
            temperature=0.0,
            guided_choice=["yes", "no", "maybe"],
        )
    except Exception as e:
        pytest.skip(f"Server does not support guided_choice: {e}")

    content = provider.extract_response_content(response).strip().lower()
    assert content in {"yes", "no", "maybe"}, (
        f"guided_choice constraint was not enforced: got {content!r}"
    )


# ---------------------------------------------------------------------------
# Tool calling (skipped unless the server supports it)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_tool_calling_if_supported(provider: VLLMProvider) -> None:
    """If the server was launched with ``--enable-auto-tool-choice`` and
    the model supports tool use, a get_weather tool should be invoked."""
    tool = Tool(
        name="get_weather",
        description="Get the current weather in a given city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
    )

    messages = [
        Message(
            role=Role.USER,
            content="What's the weather in Paris? Use the get_weather tool.",
        ),
    ]
    try:
        response = await provider.chat_completion(
            messages,
            tools=[tool],
            tool_choice="auto",
            max_tokens=64,
            temperature=0.0,
        )
    except Exception as e:
        pytest.skip(
            f"Server does not support tool calling (likely missing --enable-auto-tool-choice): {e}"
        )

    tool_calls = provider.extract_tool_calls(response)
    if not tool_calls:
        pytest.skip(
            "Server accepted tools but model produced no tool_calls; "
            "this is a model/capability issue, not a provider issue."
        )

    call = tool_calls[0]
    assert call.name == "get_weather"
    assert isinstance(call.arguments, dict)
    assert "city" in call.arguments
