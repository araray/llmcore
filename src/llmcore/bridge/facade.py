"""The narrow ``LLMCore`` surface the bridge depends on (spec §4.2, §17.3).

``BridgeCore`` is written against :class:`LLMCoreFacade` — a structural
(``typing.Protocol``) view of the *exact* public methods it calls. The real
``llmcore.LLMCore`` satisfies this protocol by duck typing, and the test
``FakeFacade`` satisfies it too, so the adapter code is identical for both.

This explicit, minimal dependency is what the contract-vs-reality CI guard
(``bindings/scripts/contract_guard.py``) verifies against the live codebase, so
the bridge can never silently drift from ``LLMCore``.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Protocol, runtime_checkable


@runtime_checkable
class LLMCoreFacade(Protocol):
    """Structural type for the subset of ``LLMCore`` the bridge uses.

    Signatures mirror the verified facade (api.py). Async/sync split is
    significant: ``chat``/``chat_with_usage``/``reload_config``/``close`` are
    coroutines; ``estimate_cost``/``get_provider_details``/
    ``get_available_providers``/``get_models_for_provider`` are synchronous.
    """

    async def chat(
        self,
        message: str,
        *,
        session_id: str | None = ...,
        system_message: str | None = ...,
        provider_name: str | None = ...,
        model_name: str | None = ...,
        stream: bool = ...,
        save_session: bool = ...,
        enable_rag: bool = ...,
        tools: list[Any] | None = ...,
        tool_choice: str | None = ...,
        **provider_kwargs: Any,
    ) -> "str | AsyncGenerator[str, None]":
        ...

    async def chat_with_usage(
        self,
        message: str,
        *,
        session_id: str | None = ...,
        system_message: str | None = ...,
        provider_name: str | None = ...,
        model_name: str | None = ...,
        save_session: bool = ...,
        enable_rag: bool = ...,
        tools: list[Any] | None = ...,
        tool_choice: str | None = ...,
        **provider_kwargs: Any,
    ) -> "tuple[str, Any]":
        ...

    def estimate_cost(
        self,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = ...,
        reasoning_tokens: int = ...,
    ) -> Any:
        ...

    def get_provider_details(self, provider_name: str | None = ...) -> Any:
        ...

    def get_available_providers(self) -> list[str]:
        ...

    def get_models_for_provider(self, provider_name: str) -> list[str]:
        ...

    async def reload_config(self) -> None:
        ...

    async def close(self) -> None:
        ...


async def build_facade(
    *,
    config_overrides: dict[str, Any] | None = None,
    config_file_path: str | None = None,
    env_prefix: str | None = "LLMCORE",
) -> LLMCoreFacade:
    """Construct a real ``LLMCore`` instance (lazy import).

    ``llmcore`` (and at runtime ``confy``) are imported here, not at module
    load, so ``import llmcore.bridge`` works in environments without the full
    runtime (e.g. doc builds, codegen).

    Args:
        config_overrides: Optional dict merged over file/env config.
        config_file_path: Optional path to a TOML config consumed by confy.
        env_prefix: Environment-variable prefix (default ``"LLMCORE"``).

    Returns:
        A fully initialized ``LLMCore`` satisfying :class:`LLMCoreFacade`.
    """
    from llmcore import LLMCore  # local import by design

    return await LLMCore.create(
        config_overrides=config_overrides,
        config_file_path=config_file_path,
        env_prefix=env_prefix,
    )
