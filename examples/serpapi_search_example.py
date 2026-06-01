# examples/serpapi_search_example.py
"""Example demonstrating web search via the SerpApi search provider.

This script shows how to:
1. Initialize LLMCore (the [search_providers.serpapi] section is in the packaged
   defaults; you only need to supply SERPAPI_API_KEY).
2. Run a standard Google web search and read normalized organic results, plus
   the rich SERP extras (knowledge_graph / answer_box / related_questions /
   serpapi_pagination) that SerpApi preserves on ``result.raw``.
3. Switch engines/verticals with a single ``engine`` argument
   (google_news / google_scholar / bing / ...).
4. Run an async search (submit + poll the Search Archive).
5. Run a batched multi-query search (client-side concurrent fan-out).
6. Use provider-specific helpers: the **free** Account API and the Locations API.
7. Introspect provider capabilities and check connectivity (free — no credit).

Search is OPTIONAL in LLMCore: if you do not configure a search provider, the
LLM features keep working and only the search methods raise a clear error.

To run this example:
- Install the search extra:  pip install "llmcore[serpapi]"
- Export your SerpApi key:
      export SERPAPI_API_KEY="your-serpapi-key"   # SERPAPI_KEY is also honored

Notes:
- SerpApi authenticates with an ``api_key`` QUERY PARAMETER (not a header) and
  needs no zones.
- SerpApi has no server-side batch endpoint, so ``batch_web_search`` issues
  concurrent requests bounded by ``max_concurrency`` (each consumes one credit).
- This example performs real network calls and will consume SerpApi credits
  (except ``health_check`` / ``account`` / ``locations``, which are free).
"""

import asyncio
import logging

from llmcore import ConfigError, LLMCore, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the SerpApi search examples."""
    # If multiple search providers are configured, name the provider explicitly
    # (here we force SerpApi). The SERPAPI_API_KEY env var is picked up automatically.
    llm = await LLMCore.create()
    try:
        available = llm.get_available_search_providers()
        logger.info("Available search providers: %s", available)
        if "serpapi" not in available:
            logger.error("SerpApi not configured. Set SERPAPI_API_KEY and try again.")
            return

        provider = llm.get_search_provider("serpapi")
        logger.info(
            "Provider: %s | capabilities: %s",
            provider.get_name(),
            sorted(provider.get_capabilities()),
        )

        # --- Connectivity check (FREE — uses the Account API, no credit) ---
        healthy = await provider.health_check()
        logger.info("Connectivity/credential check: %s", "OK" if healthy else "FAILED")

        # --- Account / quota (FREE) ---
        try:
            acct = await provider.account()
            logger.info(
                "Plan: %s | searches left: %s | this-month usage: %s",
                acct.get("plan_name"),
                acct.get("total_searches_left"),
                acct.get("this_month_usage"),
            )
        except SearchProviderError as e:  # pragma: no cover - network-dependent
            logger.warning("account() failed: %s", e)

        # --- 1) Standard Google web search ---
        logger.info("\n--- web_search (google) ---")
        res = await llm.web_search(
            "apple inc", provider="serpapi", count=10, country="us", language="en"
        )
        if res.success:
            for item in res.items:
                logger.info("  [%s] %s — %s", item.position, item.title, item.url)
            kg = res.raw.get("knowledge_graph", {}) if isinstance(res.raw, dict) else {}
            if kg:
                logger.info("  knowledge_graph: %s (%s)", kg.get("title"), kg.get("type"))
            rq = res.raw.get("related_questions", []) if isinstance(res.raw, dict) else []
            logger.info("  related_questions: %d", len(rq))
            if res.total_results is not None:
                logger.info("  total_results: %s", res.total_results)
        else:
            logger.warning("web_search failed: %s", res.error)

        # --- 2) Switch engine: Google News ---
        logger.info("\n--- web_search (engine=google_news) ---")
        news = await llm.web_search(
            "artificial intelligence", provider="serpapi", engine="google_news"
        )
        for item in news.items[:5]:
            logger.info("  %s — %s", item.title, item.url)

        # --- 3) Switch engine: Google Scholar (this engine honors num) ---
        logger.info("\n--- web_search (engine=google_scholar) ---")
        scholar = await llm.web_search(
            "graph neural networks", provider="serpapi", engine="google_scholar", num=5
        )
        for item in scholar.items[:5]:
            logger.info("  %s — %s", item.title, item.url)

        # --- 4) Async search (submit + poll the Search Archive) ---
        logger.info("\n--- web_search (mode=async) ---")
        async_res = await llm.web_search("openai", provider="serpapi", mode="async")
        logger.info("  async ok=%s, %d items", async_res.success, len(async_res.items))

        # --- 5) Batched multi-query search (client-side concurrent fan-out) ---
        logger.info("\n--- batch_web_search ---")
        batch = await llm.batch_web_search(
            ["apple inc", "tesla inc", "google inc"],
            provider="serpapi",
            country="us",
        )
        for r in batch:
            logger.info("  %-12s -> %d results", r.query, len(r.items))

        # --- 6) Locations API (FREE) — canonical names for the `location` param ---
        logger.info("\n--- locations ---")
        try:
            locs = await provider.locations(q="Austin", limit=3)
            for loc in locs:
                logger.info("  %s", loc.get("canonical_name") or loc.get("name"))
        except SearchProviderError as e:  # pragma: no cover - network-dependent
            logger.warning("locations() failed: %s", e)

    except SearchProviderError as e:
        logger.error("Search provider error: %s", e)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
