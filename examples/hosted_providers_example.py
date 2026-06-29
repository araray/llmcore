"""Hosted provider matrix: Poe, OpenRouter, DeepSeek, Kimi, DeepInfra, Mistral.

This example demonstrates explicitly selecting several hosted providers from
one LLMCore instance. It keeps the prompts small, uses provider-specific model
environment overrides, and skips providers whose keys are not available.

Run:
    set -a
    source /av/data/dbs/.env
    set +a
    python examples/hosted_providers_example.py

Optional:
    LLMCORE_EXAMPLE_PROVIDERS=poe,openrouter,deepseek python examples/hosted_providers_example.py
    LLMCORE_EXAMPLE_DEEPSEEK_MODEL=deepseek-v4-flash python examples/hosted_providers_example.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("google.genai", "google_genai", "httpx", "httpcore", "llmcore"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


@dataclass(frozen=True)
class ProviderExample:
    """Small, live-safe example configuration for a hosted provider."""

    name: str
    key_envs: tuple[str, ...]
    model_env: str
    model: str
    prompt: str
    kwargs: dict[str, Any] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderExample] = {
    "poe": ProviderExample(
        name="poe",
        key_envs=("POE_API_KEY",),
        model_env="LLMCORE_EXAMPLE_POE_MODEL",
        model="GPT-4o-Mini",
        prompt="In one sentence, explain what Poe adds as an LLMCore provider.",
        kwargs={"max_tokens": 80},
    ),
    "openrouter": ProviderExample(
        name="openrouter",
        key_envs=("OPENROUTER_API_KEY",),
        model_env="LLMCORE_EXAMPLE_OPENROUTER_MODEL",
        model="openai/gpt-4o-mini",
        prompt="In one sentence, explain what OpenRouter adds as an LLMCore provider.",
        kwargs={"max_tokens": 80},
    ),
    "deepseek": ProviderExample(
        name="deepseek",
        key_envs=("DEEPSEEK_API_KEY",),
        model_env="LLMCORE_EXAMPLE_DEEPSEEK_MODEL",
        model="deepseek-v4-flash",
        prompt="In one sentence, explain what DeepSeek adds as an LLMCore provider.",
        kwargs={"max_tokens": 80, "thinking": {"type": "disabled"}},
    ),
    "kimi": ProviderExample(
        name="kimi",
        key_envs=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        model_env="LLMCORE_EXAMPLE_KIMI_MODEL",
        model="kimi-k2.6",
        prompt="In one sentence, explain what Kimi adds as an LLMCore provider.",
        kwargs={"max_tokens": 80, "thinking": {"type": "disabled"}},
    ),
    "deepinfra": ProviderExample(
        name="deepinfra",
        key_envs=("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"),
        model_env="LLMCORE_EXAMPLE_DEEPINFRA_MODEL",
        model="deepseek-ai/DeepSeek-V3",
        prompt="In one sentence, explain what DeepInfra adds as an LLMCore provider.",
        kwargs={"max_tokens": 80},
    ),
    "mistral": ProviderExample(
        name="mistral",
        key_envs=("MISTRAL_API_KEY",),
        model_env="LLMCORE_EXAMPLE_MISTRAL_MODEL",
        model="mistral-small-latest",
        prompt="In one sentence, explain what Mistral adds as an LLMCore provider.",
        kwargs={"max_tokens": 80},
    ),
}


def _has_key(example: ProviderExample) -> bool:
    return any(os.environ.get(env_name) for env_name in example.key_envs)


def _requested_examples() -> list[ProviderExample]:
    requested = os.environ.get("LLMCORE_EXAMPLE_PROVIDERS")
    if requested:
        names = [name.strip().lower() for name in requested.split(",") if name.strip()]
        return [PROVIDERS[name] for name in names if name in PROVIDERS]
    return [example for example in PROVIDERS.values() if _has_key(example)]


def _config_overrides() -> dict[str, object]:
    providers: dict[str, dict[str, str]] = {}
    if os.environ.get("KIMI_API_KEY") and not os.environ.get("MOONSHOT_API_KEY"):
        providers["kimi"] = {"api_key_env_var": "KIMI_API_KEY"}
    if os.environ.get("DEEPINFRA_API_KEY") and not os.environ.get("DEEPINFRA_TOKEN"):
        providers["deepinfra"] = {"api_key_env_var": "DEEPINFRA_API_KEY"}

    overrides: dict[str, object] = {"llmcore": {"log_level": "ERROR"}}
    if providers:
        overrides["providers"] = providers
    return overrides


async def _run_provider(llm: LLMCore, example: ProviderExample) -> bool:
    if not _has_key(example):
        logger.warning("SKIP %s: missing one of %s", example.name, ", ".join(example.key_envs))
        return True

    model_name = os.environ.get(example.model_env, example.model)
    try:
        async with asyncio.timeout(90):
            response = await llm.chat(
                example.prompt,
                provider_name=example.name,
                model_name=model_name,
                save_session=False,
                system_message="You are concise and factual.",
                **example.kwargs,
            )
    except (ConfigError, ProviderError, LLMCoreError, TimeoutError) as exc:
        logger.error("FAIL %s/%s: %s", example.name, model_name, exc)
        return False

    logger.info("OK %s/%s: %s", example.name, model_name, response.strip())
    return True


async def main() -> None:
    examples = _requested_examples()
    if not examples:
        logger.error("No requested hosted provider keys found.")
        logger.error("Set one of: %s", ", ".join(env for item in PROVIDERS.values() for env in item.key_envs))
        raise SystemExit(1)

    async with await LLMCore.create(config_overrides=_config_overrides()) as llm:
        available = set(llm.get_available_providers())
        logger.info("Available providers: %s", sorted(available))

        ok = True
        for example in examples:
            if example.name not in available:
                logger.warning("SKIP %s: provider is not loaded.", example.name)
                continue
            ok = await _run_provider(llm, example) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
