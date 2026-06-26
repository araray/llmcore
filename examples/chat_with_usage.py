# examples/chat_with_usage.py
"""Per-call token usage with ``LLMCore.chat_with_usage``.

This example shows how to:
1. Call the normal chat path while receiving a ``ChatUsage`` value.
2. Meter a transient call without saving a session.
3. Use an explicit session id when you want the turn saved.
4. Run concurrent usage-returning calls safely.

Run:
    pip install -e .
    python examples/chat_with_usage.py

The example uses your configured default provider. For a local default, make
sure Ollama is running. For hosted providers, export the relevant API key.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError
from llmcore.usage import ChatUsage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _format_usage(usage: ChatUsage) -> str:
    """Return compact usage text that handles unavailable counts."""
    if not usage.is_available:
        return f"usage unavailable (provider={usage.provider}, model={usage.model})"
    return (
        f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
        f"total={usage.total_tokens}, provider={usage.provider}, model={usage.model}"
    )


async def main() -> None:
    """Run the usage examples."""
    try:
        async with await LLMCore.create() as llm:
            # --- 1) Transient metered call ---------------------------------
            text, usage = await llm.chat_with_usage(
                "Give a one-sentence explanation of token metering.",
                save_session=False,
            )
            print("\n=== Transient call ===")
            print(text)
            print(_format_usage(usage))

            # --- 2) Saved session call -------------------------------------
            session_id = f"usage_example_{uuid.uuid4().hex[:8]}"
            text, usage = await llm.chat_with_usage(
                "Remember that my deployment codename is Aurora.",
                session_id=session_id,
                system_message="You are concise and operational.",
                save_session=True,
            )
            print("\n=== Saved session call ===")
            print(text)
            print(_format_usage(usage))

            session = await llm.get_session(session_id)
            if session:
                logger.info("Saved session %s with %d messages.", session_id, len(session.messages))

            # --- 3) Concurrent metered calls -------------------------------
            prompts = [
                "Name one benefit of structured logging.",
                "Name one benefit of explicit config overrides.",
                "Name one benefit of per-call usage data.",
            ]
            results = await asyncio.gather(
                *(llm.chat_with_usage(prompt, save_session=False) for prompt in prompts)
            )
            print("\n=== Concurrent calls ===")
            for prompt, (answer, call_usage) in zip(prompts, results, strict=True):
                print(f"\nQ: {prompt}")
                print(f"A: {answer}")
                print(_format_usage(call_usage))

    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
    except ProviderError as exc:
        logger.error("Provider error: %s", exc)
    except LLMCoreError as exc:
        logger.error("LLMCore error: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
