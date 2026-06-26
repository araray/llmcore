"""Concise live smoke checks for search providers.

This example makes one small query per configured provider. Bright Data SERP is
only run when both the token and the required SERP zone are available.

Run:
    set -a
    source /av/data/dbs/.env
    set +a
    python examples/live_search_smoke.py

Optional:
    LLMCORE_EXAMPLE_SEARCH_PROVIDERS=serper,serpapi python examples/live_search_smoke.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from llmcore import ConfigError, LLMCore, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("google.genai", "google_genai", "httpx", "httpcore", "llmcore"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


def _requested_search_providers() -> list[str]:
    requested = os.environ.get("LLMCORE_EXAMPLE_SEARCH_PROVIDERS")
    if requested:
        return [name.strip().lower() for name in requested.split(",") if name.strip()]

    names: list[str] = []
    if os.environ.get("SERPER_API_KEY"):
        names.append("serper")
    if os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY"):
        names.append("serpapi")
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY"):
        names.append("semanticscholar")
    elif os.environ.get("LLMCORE_RUN_KEYLESS_SEARCH") == "1":
        names.append("semanticscholar")
    if os.environ.get("BRIGHTDATA_API_TOKEN") and os.environ.get("BRIGHTDATA_SERP_ZONE"):
        names.append("brightdata")
    return names


def _config_overrides() -> dict[str, object]:
    brightdata: dict[str, str] = {}
    if os.environ.get("BRIGHTDATA_SERP_ZONE"):
        brightdata["serp_zone"] = os.environ["BRIGHTDATA_SERP_ZONE"]
    if os.environ.get("BRIGHTDATA_UNLOCKER_ZONE"):
        brightdata["unlocker_zone"] = os.environ["BRIGHTDATA_UNLOCKER_ZONE"]

    overrides: dict[str, object] = {"llmcore": {"log_level": "ERROR"}}
    if brightdata:
        overrides["search_providers"] = {"brightdata": brightdata}
    return overrides


async def _run_search(llm: LLMCore, provider: str) -> None:
    kwargs: dict[str, object] = {"provider": provider, "count": 2, "language": "en"}
    if provider in {"serper", "serpapi", "brightdata"}:
        kwargs["country"] = "us"
    if provider == "semanticscholar":
        kwargs.update(
            {
                "fields": "title,abstract,url,venue,year,authors,citationCount",
                "year": "2020-2026",
            }
        )

    async with asyncio.timeout(60):
        result = await llm.web_search("retrieval augmented generation", **kwargs)

    if not result.success:
        raise SearchProviderError(provider, result.error or "Search failed")

    first = result.items[0].title if result.items else "no title"
    logger.info("OK %s: %d results; first=%s", provider, len(result.items), first)


async def main() -> None:
    providers = _requested_search_providers()
    if not providers:
        logger.error("No live search providers selected from environment keys.")
        logger.error("Set SERPER_API_KEY, SERPAPI_API_KEY, or SEMANTIC_SCHOLAR_API_KEY.")
        raise SystemExit(1)

    if os.environ.get("BRIGHTDATA_API_TOKEN") and not os.environ.get("BRIGHTDATA_SERP_ZONE"):
        logger.warning("SKIP brightdata: BRIGHTDATA_SERP_ZONE is required for SERP search.")

    failures: list[str] = []
    try:
        async with await LLMCore.create(config_overrides=_config_overrides()) as llm:
            available = set(llm.get_available_search_providers())
            logger.info("Available search providers: %s", sorted(available))
            for provider in providers:
                if provider not in available:
                    logger.warning("SKIP %s: provider is not loaded.", provider)
                    continue
                try:
                    await _run_search(llm, provider)
                except (SearchProviderError, ConfigError, TimeoutError) as exc:
                    failures.append(f"{provider}: {exc}")
                    logger.error("FAIL %s: %s", provider, exc)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
