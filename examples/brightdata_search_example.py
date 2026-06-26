# examples/brightdata_search_example.py
"""Example demonstrating web/data search via the Bright Data search provider.

This script shows how to:
1. Initialize LLMCore (the [search_providers.brightdata] section is already in
   the packaged defaults; you only need to supply a token + zone names).
2. Run a web search (SERP) and read normalized organic results.
3. Scrape a single URL through the Web Unlocker.
4. Run an AI-relevance-ranked Discover search.
5. List datasets and collect structured records from one.
6. Introspect provider capabilities and check connectivity.

Search is OPTIONAL in LLMCore: if you do not configure a search provider, the
LLM features keep working and only the search methods raise a clear error.

To run this example:
- Install the search extra:  pip install "llmcore[brightdata]"
- Export your Bright Data token:
      export BRIGHTDATA_API_TOKEN="bd_..."
- Provide your zone names (created at https://brightdata.com/cp/zones) via
  environment variables or in ~/.config/llmcore/config.toml:
      export BRIGHTDATA_SERP_ZONE="my_serp_zone"
      export BRIGHTDATA_UNLOCKER_ZONE="my_unlocker_zone"

  Equivalent TOML:
      [search_providers.brightdata]
      serp_zone = "my_serp_zone"
      unlocker_zone = "my_unlocker_zone"

Notes:
- Zones are NOT auto-created (unlike the vendor SDK) to avoid mutating your
  account. Dataset and Discover operations do not require a zone.
- This example performs real network calls and will incur Bright Data usage.
"""

import asyncio
import logging
import os

from llmcore import ConfigError, LLMCore, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


async def main() -> None:
    """Run the Bright Data search examples."""
    # Supply zone names without editing your config file. The token is read from
    # BRIGHTDATA_API_TOKEN automatically by the packaged default config.
    serp_zone = os.environ.get("BRIGHTDATA_SERP_ZONE")
    unlocker_zone = os.environ.get("BRIGHTDATA_UNLOCKER_ZONE")
    overrides = {
        "search_providers": {
            "brightdata": {
                **({"serp_zone": serp_zone} if serp_zone else {}),
                **({"unlocker_zone": unlocker_zone} if unlocker_zone else {}),
            }
        }
    }

    llm = await LLMCore.create(config_overrides=overrides)
    try:
        # --- Capability introspection ---
        available = llm.get_available_search_providers()
        logger.info("Available search providers: %s", available)
        if "brightdata" not in available:
            logger.error("Bright Data is not configured. Set BRIGHTDATA_API_TOKEN and try again.")
            return

        provider = llm.get_search_provider("brightdata")
        logger.info("Search provider: %s", provider.get_name())
        logger.info("Capabilities: %s", sorted(provider.get_capabilities()))

        # --- Connectivity check ---
        healthy = await provider.health_check()
        logger.info("Connectivity/credential check: %s", "OK" if healthy else "FAILED")

        # --- 1) Web search (SERP) ---
        logger.info("\n--- web_search ---")
        if not serp_zone:
            logger.warning("Skipping web_search: set BRIGHTDATA_SERP_ZONE to run SERP calls.")
        else:
            result = await llm.web_search(
                "best vector databases 2026", provider="brightdata", count=5, country="US"
            )
            if result.success:
                logger.info(
                    "Got %d organic results (total≈%s):", len(result.items), result.total_results
                )
                for item in result.items:
                    logger.info("  [%s] %s — %s", item.position, item.title, item.url)
            else:
                logger.warning("web_search failed: %s", result.error)

        # --- 2) Scrape a URL (Web Unlocker) ---
        logger.info("\n--- scrape_url ---")
        if not unlocker_zone:
            logger.warning(
                "Skipping scrape_url: set BRIGHTDATA_UNLOCKER_ZONE to run Web Unlocker calls."
            )
        else:
            scrape = await llm.scrape_url(
                "https://example.com", provider="brightdata", response_format="raw"
            )
            if scrape.success:
                logger.info(
                    "Scraped %s (%s chars) from domain %s",
                    scrape.url,
                    scrape.content_char_size,
                    scrape.root_domain,
                )
            else:
                logger.warning("scrape failed: %s", scrape.error)

        # --- 3) Discover (AI-ranked search) ---
        logger.info("\n--- discover ---")
        disc = await llm.discover(
            "agentic AI frameworks",
            provider="brightdata",
            intent="compare open-source multi-agent orchestration libraries",
            count=5,
        )
        if disc.success:
            for item in disc.items:
                score = f"{item.relevance_score:.2f}" if item.relevance_score is not None else "n/a"
                logger.info("  [%s] %s — %s", score, item.title, item.url)
        else:
            logger.warning("discover failed: %s", disc.error)

        # --- 4) Datasets: list + collect ---
        logger.info("\n--- datasets ---")
        datasets = await llm.list_datasets(provider="brightdata")
        logger.info("Found %d datasets (showing up to 5):", len(datasets))
        for ds in datasets[:5]:
            logger.info("  %s — %s (size≈%s)", ds.id, ds.name, ds.size)

        if datasets:
            ds = datasets[0]
            logger.info("Collecting a small sample from '%s'...", ds.name)
            meta = await llm.get_dataset_metadata(ds.id, provider="brightdata")
            if meta.fields:
                first_field = meta.field_names()[0]
                snapshot = await llm.dataset_search(
                    ds.id,
                    {"name": first_field, "operator": "is_not_null"},
                    provider="brightdata",
                    records_limit=3,
                )
                if snapshot.success:
                    logger.info("Downloaded %d records.", snapshot.record_count)
                else:
                    logger.warning("dataset_search failed: %s", snapshot.error)

    except SearchProviderError as e:
        logger.error("Search provider error: %s", e)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
