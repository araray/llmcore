# examples/search_to_chat_external_rag.py
"""External-RAG style search-to-chat flow using ``llmcore.search``.

This example uses Semantic Scholar as a keyless search provider, formats the
search results as prompt context, and sends the final prompt through
``chat_with_usage``. It demonstrates the recommended external RAG pattern:
retrieval/ranking is controlled outside LLMCore's vector store, while LLMCore
handles provider abstraction, prompt execution, and usage accounting.

Run:
    pip install -e .
    pip install "llmcore[semanticscholar]"
    python examples/search_to_chat_external_rag.py

Notes:
- Semantic Scholar works without an API key, but an optional
  SEMANTIC_SCHOLAR_API_KEY gives higher rate limits.
- The chat step uses your configured default LLM provider.
"""

from __future__ import annotations

import asyncio
import logging

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError, SearchProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

S2_CONFIG_OVERRIDE = {
    "search_providers": {
        "semanticscholar": {
            "default_search_type": "relevance",
            "max_concurrency": 1,
        }
    }
}


def _format_papers_for_prompt(items: list[object]) -> str:
    """Build compact citation-like context from Semantic Scholar result items."""
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        title = getattr(item, "title", "Untitled")
        url = getattr(item, "url", "")
        description = getattr(item, "description", "") or ""
        lines.append(f"[{index}] {title}\nURL: {url}\nAbstract/Snippet: {description[:900]}")
    return "\n\n".join(lines)


async def main() -> None:
    topic = "retrieval augmented generation evaluation methods"

    try:
        async with await LLMCore.create(config_overrides=S2_CONFIG_OVERRIDE) as llm:
            if "semanticscholar" not in llm.get_available_search_providers():
                logger.error("Semantic Scholar provider did not load.")
                return

            logger.info("Searching Semantic Scholar for: %s", topic)
            result = await llm.web_search(
                topic,
                provider="semanticscholar",
                count=5,
                fields="title,abstract,url,venue,year,authors,citationCount,openAccessPdf",
                year="2020-2026",
            )
            if not result.success or not result.items:
                logger.error("Search failed or returned no papers: %s", result.error)
                return

            context = _format_papers_for_prompt(result.items[:5])
            prompt = (
                "Use only the paper context below. Summarize the main evaluation "
                "approaches for retrieval augmented generation, then list two "
                "open questions.\n\n"
                f"Paper context:\n{context}"
            )

            answer, usage = await llm.chat_with_usage(
                prompt,
                save_session=False,
                system_message=(
                    "You are a careful research assistant. Cite papers with bracketed "
                    "numbers from the provided context."
                ),
            )

            print("\n=== Answer ===")
            print(answer)
            print("\n=== Usage ===")
            if usage.is_available:
                print(
                    f"provider={usage.provider} model={usage.model} "
                    f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                    f"total={usage.total_tokens}"
                )
            else:
                print(f"Usage unavailable for provider={usage.provider} model={usage.model}")

    except SearchProviderError as exc:
        logger.error("Search provider error: %s", exc)
    except ProviderError as exc:
        logger.error("LLM provider error: %s", exc)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
    except LLMCoreError as exc:
        logger.error("LLMCore error: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
