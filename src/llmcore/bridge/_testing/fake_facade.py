"""A deterministic, offline ``LLMCoreFacade`` for conformance testing.

Behaviour is fully reproducible and requires no provider keys or network:

* ``chat`` / ``chat_with_usage`` echo the prompt as ``"echo: <message>"`` and
  compute token usage by whitespace splitting. The streaming form yields fixed
  8-character slices so ``"".join(chunks) == unary_text`` exactly.
* Scriptable triggers (carried in the message) exercise non-happy paths:
  - ``__error__:<key>``     — raise a mapped llmcore exception at call setup.
  - ``__error_mid__:<key>`` — yield one chunk, then raise mid-stream.
  - ``__cancel__``          — yield two chunks, then await forever (cancellation).

Supported ``<key>`` values: ``provider_rate_limited``, ``provider_unauthorized``,
``config``, ``context_length``, ``not_found``, ``embedding``, ``storage``,
``unsupported``, ``internal``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from llmcore.exceptions import (
    ConfigError,
    ContextLengthError,
    EmbeddingError,
    LLMCoreError,
    ProviderError,
    SessionNotFoundError,
    StorageError,
)
from llmcore.models import ModelDetails
from llmcore.usage import ChatUsage

_ERROR_PREFIX = "__error__:"
_ERROR_MID_PREFIX = "__error_mid__:"
_CANCEL_TOKEN = "__cancel__"


def fake_count_tokens(text: str | None, model: str | None = None) -> int:
    """Deterministic token counter: number of whitespace-delimited words."""
    return len((text or "").split())


def _raise_for_key(key: str) -> None:
    key = key.strip()
    if key == "provider_rate_limited":
        raise ProviderError(
            "fake", "rate limited", model_name="fake-1", status_code=429, retry_after_seconds=2.0
        )
    if key == "provider_unauthorized":
        raise ProviderError("fake", "invalid api key", model_name="fake-1", status_code=401)
    if key == "config":
        raise ConfigError("bad configuration value")
    if key == "context_length":
        raise ContextLengthError("fake-1", 8192, 9000, "prompt too long")
    if key == "not_found":
        raise SessionNotFoundError("sess-does-not-exist")
    if key == "embedding":
        raise EmbeddingError("embed-1", "embedding backend failure")
    if key == "storage":
        raise StorageError("session store unavailable")
    if key == "unsupported":
        raise NotImplementedError("provider does not support this capability")
    if key == "internal":
        raise LLMCoreError("unexpected internal failure")
    raise LLMCoreError(f"unknown error trigger: {key}")


class FakeFacade:
    """Deterministic stand-in implementing the ``LLMCoreFacade`` protocol."""

    PROVIDER = "fake"
    MODEL = "fake-1"

    # -- chat ------------------------------------------------------------- #
    async def chat_with_usage(self, message: str, **kw: Any) -> tuple[str, ChatUsage]:
        if message.startswith(_ERROR_PREFIX):
            _raise_for_key(message[len(_ERROR_PREFIX):])
        text = "echo: " + message
        usage = ChatUsage(
            prompt_tokens=fake_count_tokens(message),
            completion_tokens=fake_count_tokens(text),
            total_tokens=fake_count_tokens(message) + fake_count_tokens(text),
            provider=kw.get("provider_name") or self.PROVIDER,
            model=kw.get("model_name") or self.MODEL,
        )
        return text, usage

    async def chat(
        self, message: str, *, stream: bool = False, **kw: Any
    ) -> "str | AsyncGenerator[str, None]":
        if message.startswith(_ERROR_PREFIX):
            _raise_for_key(message[len(_ERROR_PREFIX):])
        text = "echo: " + message
        if not stream:
            return text
        return self._stream(message, text)

    def _stream(self, message: str, text: str) -> AsyncGenerator[str, None]:
        async def gen() -> AsyncGenerator[str, None]:
            if message.strip() == _CANCEL_TOKEN:
                yield text[:4]
                yield text[4:8]
                await asyncio.Event().wait()  # never set -> cancellation point
                return
            if message.startswith(_ERROR_MID_PREFIX):
                yield text[:4]
                _raise_for_key(message[len(_ERROR_MID_PREFIX):])
                return
            for i in range(0, len(text), 8):
                yield text[i : i + 8]

        return gen()

    # -- cost / catalog --------------------------------------------------- #
    def estimate_cost(
        self,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> Any:
        from llmcore.models import CostEstimate

        in_rate, out_rate = 1.0, 2.0  # USD per 1M tokens
        input_cost = prompt_tokens / 1_000_000 * in_rate
        output_cost = completion_tokens / 1_000_000 * out_rate
        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            currency="USD",
            pricing_source="model_card",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            input_price_per_million=in_rate,
            output_price_per_million=out_rate,
            model_id=model_name,
            provider=provider_name,
        )

    def get_provider_details(self, provider_name: str | None = None) -> ModelDetails:
        return ModelDetails(
            id=self.MODEL,
            provider_name=provider_name or self.PROVIDER,
            display_name="Fake Model 1",
            context_length=8192,
            max_output_tokens=4096,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            supports_reasoning=False,
            family="Fake",
            model_type="chat",
            metadata={"vendor": "fake"},
        )

    def get_available_providers(self) -> list[str]:
        return [self.PROVIDER]

    def get_models_for_provider(self, provider_name: str) -> list[str]:
        if provider_name in (self.PROVIDER, "", None):
            return ["fake-1", "fake-2"]
        return []

    # -- lifecycle -------------------------------------------------------- #
    async def reload_config(self) -> None:
        return None

    async def close(self) -> None:
        return None
