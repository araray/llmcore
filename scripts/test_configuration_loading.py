# scripts/test_configuration_loading.py
import asyncio

from llmcore.api import LLMCore


async def test_config():
    core = await LLMCore.create()
    # Test dot notation access
    db_type = core.config.get("semantiscan.database.type")
    print(f"Database type: {db_type}")
    # Should print: Database type: chromadb


asyncio.run(test_config())
