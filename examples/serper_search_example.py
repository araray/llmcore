# examples/serper_search_example.py
"""Example demonstrating web search via the Serper.dev search provider.

This script shows how to:
1. Initialize LLMCore (the [search_providers.serper] section is in the packaged
   defaults; you only need to supply SERPER_API_KEY).
2. Run a standard Google web search and read normalized organic results, plus
   the rich SERP extras (knowledgeGraph / peopleAlsoAsk / relatedSearches) that
   Serper preserves on ``result.raw``.
3. Query a vertical (news/scholar/patents/...) with a time filter.
4. Run a **batched** multi-query search in a single request (Serper's strength).
5. Scrape a URL (markdown) via the separate scrape.serper.dev host.
6. Introspect provider capabilities and check connectivity.

Search is OPTIONAL in LLMCore: if you do not configure a search provider, the
LLM features keep working and only the search methods raise a clear error.

To run this example:
- Install the search extra:  pip install "llmcore[serper]"
- Export your Serper API key:
      export SERPER_API_KEY="your-serper-key"

Notes:
- Serper authenticates with an X-API-KEY header (not a Bearer token) and needs
  no zones.
- This example performs real network calls and will consume Serper credits.
"""

import asyncio
import logging

from llmcore import ConfigError, LLMCore, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the Serper.dev search examples."""
    # If both Bright Data and Serper are configured, name the provider explicitly
    # (here we force Serper). The SERPER_API_KEY env var is picked up automatically.
    llm = await LLMCore.create()
    try:
        available = llm.get_available_search_providers()
        logger.info("Available search providers: %s", available)
        if "serper" not in available:
            logger.error("Serper not configured. Set SERPER_API_KEY and try again.")
            return

        provider = llm.get_search_provider("serper")
        logger.info(
            "Provider: %s | capabilities: %s",
            provider.get_name(),
            sorted(provider.get_capabilities()),
        )

        # --- Connectivity check (consumes 1 credit) ---
        healthy = await provider.health_check()
        logger.info("Connectivity/credential check: %s", "OK" if healthy else "FAILED")

        # --- 1) Standard web search ---
        logger.info("\n--- web_search (search) ---")
        res = await llm.web_search(
            "apple inc", provider="serper", count=10, country="us", language="en"
        )
        if res.success:
            for item in res.items:
                logger.info("  [%s] %s — %s", item.position, item.title, item.url)
            kg = res.raw.get("knowledgeGraph", {}) if isinstance(res.raw, dict) else {}
            if kg:
                logger.info("  knowledgeGraph: %s (%s)", kg.get("title"), kg.get("type"))
            paa = res.raw.get("peopleAlsoAsk", []) if isinstance(res.raw, dict) else []
            logger.info("  peopleAlsoAsk: %d questions", len(paa))
        else:
            logger.warning("web_search failed: %s", res.error)

        # --- 2) Vertical + time filter (news, past day) ---
        logger.info("\n--- web_search (news, qdr:d) ---")
        news = await llm.web_search(
            "artificial intelligence", provider="serper", search_type="news", time_range="d"
        )
        for item in news.items[:5]:
            logger.info("  %s — %s", item.title, item.url)

        # --- 3) Batched multi-query search (one request) ---
        logger.info("\n--- batch_web_search ---")
        batch = await llm.batch_web_search(
            ["apple inc", "tesla inc", "google inc"],
            provider="serper",
            country="us",
            time_range="m",
        )
        for r in batch:
            logger.info("  %-12s -> %d results", r.query, len(r.items))

        # --- 4) Scrape a URL (markdown) ---
        logger.info("\n--- scrape_url (markdown) ---")
        page = await llm.scrape_url("https://example.com", provider="serper")
        if page.success:
            logger.info(
                "Scraped %s as %s (%s chars); metadata keys: %s",
                page.url,
                page.response_format,
                page.content_char_size,
                list(page.raw.get("metadata", {}).keys()) if isinstance(page.raw, dict) else [],
            )
        else:
            logger.warning("scrape failed: %s", page.error)

    except SearchProviderError as e:
        logger.error("Search provider error: %s", e)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
