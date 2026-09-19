# tests/moderation/test_openai_gateway.py
"""
Tests for OpenAIModerationGateway with a mocked SDK client (SF-1).

Covers:
- Request payload (model + input) sent to the moderations endpoint
- Response mapping: flagged, snake_case category normalization, provider flags
- Real-SDK pydantic response objects (model_dump path)
- Typed ModerationError on timeout / API fault / malformed response
- Fail-safe integration: broken gateway + enabled policy => BLOCK
- Construction: key resolution precedence, missing key, missing SDK
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APITimeoutError
from openai.types.moderation import Categories, CategoryScores, Moderation

from llmcore.exceptions import ConfigError, ModerationError
from llmcore.moderation import (
    DEFAULT_MODERATION_MODEL,
    ModerationPolicy,
    OpenAIModerationGateway,
    moderate,
)
from llmcore.moderation import openai_gateway as openai_gateway_module


class _FakeModerations:
    """Stand-in for ``AsyncOpenAI().moderations``."""

    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _gateway(response: Any = None, error: Exception | None = None, **config: Any):
    """Build a gateway with an explicit key and a mocked client."""
    gateway = OpenAIModerationGateway({"api_key": "sk-test", **config})
    fake = _FakeModerations(response=response, error=error)
    gateway._client = SimpleNamespace(moderations=fake, close=_noop_close)
    return gateway, fake


async def _noop_close() -> None:
    return None


def _dict_response(
    *,
    flagged: bool = True,
    scores: dict[str, float] | None = None,
    flags: dict[str, bool] | None = None,
    model: str = "omni-moderation-2024-09-26",
) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                flagged=flagged,
                category_scores=scores if scores is not None else {},
                categories=flags if flags is not None else {},
            )
        ],
        model=model,
    )


# =============================================================================
# PAYLOAD + RESPONSE MAPPING
# =============================================================================


async def test_check_sends_model_and_input():
    gateway, fake = _gateway(response=_dict_response(flagged=False))
    await gateway.check("hello world", context="input")
    assert fake.calls == [{"model": DEFAULT_MODERATION_MODEL, "input": "hello world"}]


async def test_check_uses_configured_model():
    gateway, fake = _gateway(response=_dict_response(flagged=False), model="text-moderation-stable")
    await gateway.check("hi")
    assert fake.calls[0]["model"] == "text-moderation-stable"


async def test_response_mapping_normalizes_categories():
    gateway, _ = _gateway(
        response=_dict_response(
            flagged=True,
            scores={"violence": 0.91, "self-harm/intent": 0.2, "harassment/threatening": 0.05},
            flags={"violence": True, "self-harm/intent": False, "harassment/threatening": False},
        )
    )
    result = await gateway.check("bad text")
    assert result.flagged is True
    assert result.categories == {
        "violence": 0.91,
        "self_harm_intent": 0.2,
        "harassment_threatening": 0.05,
    }
    assert result.flagged_categories == ("violence",)
    assert result.provider == "openai"
    assert result.model == "omni-moderation-2024-09-26"
    assert result.action_hint is not None


async def test_response_mapping_with_real_sdk_types():
    """The pydantic model_dump path used with the real openai SDK response."""
    moderation = Moderation.model_construct(
        flagged=True,
        categories=Categories.model_construct(violence=True, harassment=False),
        category_scores=CategoryScores.model_construct(violence=0.95, harassment=0.01),
    )
    response = SimpleNamespace(results=[moderation], model="omni-moderation-latest")
    gateway, _ = _gateway(response=response)
    result = await gateway.check("bad text")
    assert result.flagged is True
    assert result.categories == {"violence": 0.95, "harassment": 0.01}
    assert result.flagged_categories == ("violence",)
    assert result.model == "omni-moderation-latest"


async def test_clean_response_maps_to_unflagged():
    gateway, _ = _gateway(
        response=_dict_response(flagged=False, scores={"violence": 0.001}, flags={"violence": False})
    )
    result = await gateway.check("hello")
    assert result.flagged is False
    assert result.flagged_categories == ()
    assert result.action_hint is None


# =============================================================================
# TYPED ERRORS
# =============================================================================


async def test_timeout_raises_moderation_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    gateway, _ = _gateway(error=APITimeoutError(request=request))
    with pytest.raises(ModerationError, match="openai"):
        await gateway.check("hello")


async def test_unexpected_error_raises_moderation_error():
    gateway, _ = _gateway(error=RuntimeError("socket exploded"))
    with pytest.raises(ModerationError, match="socket exploded"):
        await gateway.check("hello")


async def test_empty_results_raises_moderation_error():
    gateway, _ = _gateway(response=SimpleNamespace(results=[], model="m"))
    with pytest.raises(ModerationError, match="no results"):
        await gateway.check("hello")


async def test_unrecognized_result_shape_raises_moderation_error():
    """A non-empty result item without a ``flagged`` field must not ALLOW.

    A compatible proxy (base_url) returning a 200 whose ``results[0]``
    lacks the expected fields must surface as a ModerationError, never a
    silent ``flagged=False`` allow. Covers both the object and dict paths.
    """
    gateway, _ = _gateway(response=SimpleNamespace(results=[SimpleNamespace()], model="m"))
    with pytest.raises(ModerationError, match="unrecognized result shape"):
        await gateway.check("hello")

    gateway, _ = _gateway(response={"results": [{}], "model": "m"})
    with pytest.raises(ModerationError, match="unrecognized result shape"):
        await gateway.check("hello")


async def test_unrecognized_result_shape_blocks_through_policy_fail_safe():
    """Acceptance: unrecognized 200 body + enabled policy => block (fail-safe)."""
    gateway, _ = _gateway(response={"results": [{}], "model": "m"})
    decision = await moderate(gateway, ModerationPolicy(), "hello", context="input")
    assert decision.allowed is False
    assert decision.fail_safe is True


async def test_gateway_error_blocks_through_policy_fail_safe():
    """Acceptance: gateway-down + enabled => block."""
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    gateway, _ = _gateway(error=APITimeoutError(request=request))
    decision = await moderate(gateway, ModerationPolicy(), "hello", context="input")
    assert decision.allowed is False
    assert decision.fail_safe is True


# =============================================================================
# CONSTRUCTION / CONFIG
# =============================================================================


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API key"):
        OpenAIModerationGateway({})


def test_api_key_env_var_resolution(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_MODERATION_KEY", "sk-from-env")
    gateway = OpenAIModerationGateway({"api_key_env_var": "MY_MODERATION_KEY"})
    assert gateway._client.api_key == "sk-from-env"


def test_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    gateway = OpenAIModerationGateway({"api_key": "sk-explicit"})
    assert gateway._client.api_key == "sk-explicit"


def test_timeout_and_retries_from_config():
    gateway = OpenAIModerationGateway({"api_key": "sk-test", "timeout": 5, "max_retries": 0})
    assert gateway.timeout == 5.0
    assert gateway.max_retries == 0


def test_missing_sdk_raises_import_error(monkeypatch):
    monkeypatch.setattr(openai_gateway_module, "openai_available", False)
    with pytest.raises(ImportError, match="OpenAI library not installed"):
        OpenAIModerationGateway({"api_key": "sk-test"})


async def test_close_is_idempotent():
    gateway, _ = _gateway(response=_dict_response(flagged=False))
    await gateway.close()
    assert gateway._client is None
    await gateway.close()  # second close must not raise
