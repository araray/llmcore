# examples/semanticscholar_search_example.py
"""Example: academic search via the Semantic Scholar (S2) search provider.

This script shows how to:
1. Initialize LLMCore with Semantic Scholar enabled (it works WITHOUT an API
   key — the [search_providers.semanticscholar] section is shipped commented out
   in the packaged defaults, so this example registers it programmatically).
2. Run paper search across the four S2 flavors: relevance, bulk (with a
   continuation token), title match, and text-snippet search (great for RAG).
3. Traverse the citation graph and look up papers/authors in batch.
4. Get paper recommendations (single-seed and from positive/negative lists).
5. Inspect the bulk-corpus Datasets API (release/dataset download links).
6. Introspect provider capabilities and check connectivity.

Search is OPTIONAL in LLMCore: if you do not configure a search provider, the
LLM features keep working and only the search methods raise a clear error.

To run this example:
- Install the search extra:  pip install "llmcore[semanticscholar]"
- (OPTIONAL) Export an API key for higher rate limits:
      export SEMANTIC_SCHOLAR_API_KEY="your-s2-key"   # S2_API_KEY also honored

Notes:
- The API key is OPTIONAL. Without one you share the public (rate-limited) pool;
  S2 requires exponential backoff, which the provider applies automatically.
- These are real network calls against the Semantic Scholar APIs (free), so be
  considerate of the shared rate limits when running without a key.
"""

import asyncio
import logging

from llmcore import ConfigError, LLMCore, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The packaged defaults ship the S2 block commented out (so search stays empty
# by default). Enable it here without requiring a key. In a real app you would
# instead uncomment [search_providers.semanticscholar] in your config.toml.
S2_CONFIG_OVERRIDE = {
    "search_providers": {
        "semanticscholar": {
            "default_search_type": "relevance",
            "max_concurrency": 1,  # be gentle on the shared rate limit
        }
    }
}


async def main() -> None:
    """Run the Semantic Scholar search examples."""
    llm = await LLMCore.create(config_overrides=S2_CONFIG_OVERRIDE)
    try:
        available = llm.get_available_search_providers()
        logger.info("Available search providers: %s", available)
        if "semanticscholar" not in available:
            logger.error("Semantic Scholar not loaded. Check your installation/config.")
            return

        s2 = llm.get_search_provider("semanticscholar")
        logger.info(
            "Provider: %s | capabilities: %s",
            s2.get_name(),
            sorted(s2.get_capabilities()),
        )

        # --- Connectivity check (tiny autocomplete probe) ---
        healthy = await s2.health_check()
        logger.info("Connectivity check: %s", "OK" if healthy else "FAILED")

        # --- 1) Relevance paper search with filters ---
        logger.info("\n--- web_search (relevance) ---")
        res = await llm.web_search(
            "graph neural networks",
            provider="semanticscholar",
            count=10,
            year="2018-2024",
            fieldsOfStudy="Computer Science",
            openAccessPdf=True,
        )
        if res.success:
            for item in res.items[:10]:
                logger.info("  [%s] %s — %s", item.position, item.title, item.url)
            logger.info("  approx total matches: %s", res.total_results)
        else:
            logger.warning("relevance search failed: %s", res.error)

        # --- 2) Bulk search (paginate via the continuation token) ---
        logger.info("\n--- web_search (bulk) ---")
        bulk = await llm.web_search(
            "transformer architecture",
            provider="semanticscholar",
            search_type="bulk",
            count=100,
            sort="citationCount:desc",
        )
        logger.info("  bulk page items: %d | next token: %s", len(bulk.items),
                    bulk.raw.get("token") if isinstance(bulk.raw, dict) else None)

        # --- 3) Title match — resolve a title/citation to one canonical paper ---
        logger.info("\n--- paper_match ---")
        match = await s2.paper_match("Attention is all you need")
        seed_id = None
        if match.success and match.items:
            seed_id = match.raw["data"][0]["paperId"]
            logger.info("  matched: %s (%s)", match.items[0].title, seed_id)

        # --- 4) Snippet search (text passages — ideal for RAG) ---
        logger.info("\n--- snippet_search ---")
        snips = await s2.snippet_search("retrieval augmented generation", limit=5)
        for item in snips.items[:5]:
            logger.info("  %s :: %s", item.title, item.description[:90])

        # --- 5) Citation graph + batch lookup (seed from the matched paper) ---
        if seed_id:
            logger.info("\n--- citations / references / batch ---")
            citing = await s2.paper_citations(seed_id, limit=5)
            refs = await s2.paper_references(seed_id, limit=5)
            logger.info("  %d citing, %d references", len(citing.items), len(refs.items))
            batch = await s2.paper_batch([seed_id, "ARXIV:1810.04805"])
            logger.info("  batch resolved %d papers", len(batch.items))

            # --- 6) Recommendations from the seed paper ---
            logger.info("\n--- recommend_papers ---")
            recs = await s2.recommend_papers(seed_id, limit=5, pool="recent")
            for item in recs.items[:5]:
                logger.info("  %s", item.title)

        # --- 7) Author search ---
        logger.info("\n--- author_search ---")
        authors = await s2.author_search("Yoshua Bengio")
        for item in authors.items[:3]:
            logger.info("  %s — %s", item.title, item.url)

        # --- 8) Datasets API (bulk-corpus download links) ---
        logger.info("\n--- datasets ---")
        releases = await s2.list_releases()
        if releases:
            logger.info("  %d releases (latest: %s)", len(releases), releases[-1])
            latest = await s2.get_release("latest")
            names = [d.get("name") for d in latest.get("datasets", [])]
            logger.info("  datasets in latest release: %s", names[:8])

    except SearchProviderError as e:
        logger.error("Search provider error: %s", e)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
