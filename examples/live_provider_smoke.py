"""Concise live smoke checks for configured LLM providers.

This example is intentionally smaller than the provider walkthroughs. It makes
one short non-streaming call per hosted provider that has a key in the
environment, and one streaming call for Gemini when available.

Run:
    set -a
    source /av/data/dbs/.env
    set +a
    python examples/live_provider_smoke.py

Optional:
    LLMCORE_EXAMPLE_PROVIDERS=openai,gemini python examples/live_provider_smoke.py
    LLMCORE_RUN_OLLAMA=1 python examples/live_provider_smoke.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterable

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("google.genai", "google_genai", "httpx", "httpcore", "llmcore"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

MODEL_BY_PROVIDER = {
    "openai": ("LLMCORE_EXAMPLE_OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": ("LLMCORE_EXAMPLE_GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    "poe": ("LLMCORE_EXAMPLE_POE_MODEL", "GPT-4o-Mini"),
    "openrouter": ("LLMCORE_EXAMPLE_OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    "deepseek": ("LLMCORE_EXAMPLE_DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "kimi": ("LLMCORE_EXAMPLE_KIMI_MODEL", "kimi-k2.6"),
    "deepinfra": ("LLMCORE_EXAMPLE_DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3"),
    "mistral": ("LLMCORE_EXAMPLE_MISTRAL_MODEL", "mistral-small-latest"),
    "ollama": ("LLMCORE_EXAMPLE_OLLAMA_MODEL", "llama3"),
}

KEY_ENVS_BY_PROVIDER = {
    "openai": ("OPENAI_API_KEY", "LLMCORE_PROVIDERS__OPENAI__API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLMCORE_PROVIDERS__GEMINI__API_KEY"),
    "poe": ("POE_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "kimi": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "deepinfra": ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
}

PROVIDER_KWARGS = {
    "openai": {"max_tokens": 40},
    "poe": {"max_tokens": 40},
    "openrouter": {"max_tokens": 40},
    "deepseek": {"max_tokens": 40, "thinking": {"type": "disabled"}},
    "kimi": {"max_tokens": 40, "thinking": {"type": "disabled"}},
    "deepinfra": {"max_tokens": 40},
    "mistral": {"max_tokens": 40},
    "gemini": {"max_output_tokens": 40},
}


def _configured_provider_names() -> list[str]:
    requested = os.environ.get("LLMCORE_EXAMPLE_PROVIDERS")
    if requested:
        return [name.strip().lower() for name in requested.split(",") if name.strip()]

    names: list[str] = []
    for provider_name, env_names in KEY_ENVS_BY_PROVIDER.items():
        if any(os.environ.get(env_name) for env_name in env_names):
            names.append(provider_name)
    if os.environ.get("LLMCORE_RUN_OLLAMA") == "1":
        names.append("ollama")
    return names


def _model_for(provider_name: str) -> str | None:
    env_name, fallback = MODEL_BY_PROVIDER.get(provider_name, ("", ""))
    return os.environ.get(env_name, fallback) or None


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


async def _consume_stream(chunks: AsyncIterable[str]) -> str:
    text = ""
    async with asyncio.timeout(60):
        async for chunk in chunks:  # type: ignore[attr-defined]
            text += chunk
    return text


async def main() -> None:
    provider_names = _configured_provider_names()
    if not provider_names:
        logger.error("No live provider keys found. Set provider API keys and try again.")
        raise SystemExit(1)

    failures: list[str] = []

    try:
        async with await LLMCore.create(config_overrides=_config_overrides()) as llm:
            available = set(llm.get_available_providers())
            logger.info("Available providers: %s", sorted(available))

            for provider_name in provider_names:
                if provider_name not in available:
                    logger.warning("SKIP %s: provider is not loaded.", provider_name)
                    continue

                model_name = _model_for(provider_name)
                try:
                    async with asyncio.timeout(60):
                        response = await llm.chat(
                            "Answer with exactly five words: what is LLMCore?",
                            provider_name=provider_name,
                            model_name=model_name,
                            save_session=False,
                            system_message="You answer tersely.",
                            **PROVIDER_KWARGS.get(provider_name, {}),
                        )
                    logger.info("OK %s/%s: %s", provider_name, model_name, response.strip())
                except (ProviderError, ConfigError, LLMCoreError, TimeoutError) as exc:
                    failures.append(f"{provider_name}: {exc}")
                    logger.error("FAIL %s/%s: %s", provider_name, model_name, exc)

            if "gemini" in provider_names and "gemini" in available:
                model_name = _model_for("gemini")
                try:
                    stream = await llm.chat(
                        "Stream exactly three words about provider health.",
                        provider_name="gemini",
                        model_name=model_name,
                        stream=True,
                        save_session=False,
                    )
                    streamed = await _consume_stream(stream)
                    logger.info("OK gemini streaming/%s: %s", model_name, streamed.strip())
                except (ProviderError, ConfigError, LLMCoreError, TimeoutError) as exc:
                    failures.append(f"gemini streaming: {exc}")
                    logger.error("FAIL gemini streaming/%s: %s", model_name, exc)

    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
