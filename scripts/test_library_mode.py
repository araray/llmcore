#!/usr/bin/env python3
"""
Simple functional test for llmcore in pure library mode.
Tests that LLMCore can be instantiated without service dependencies.
"""

import asyncio
import sys
import tempfile
from pathlib import Path


async def test_library_instantiation():
    """Test that LLMCore can be created in library mode without database/Redis."""
    print("🧪 Testing LLMCore library mode instantiation...")
    print("")

    try:
        from llmcore import LLMCore
        print("✅ Successfully imported LLMCore")
    except ImportError as e:
        print(f"❌ Failed to import LLMCore: {e}")
        return False

    # Create a minimal config override for testing
    # Use JSON session storage to avoid database dependencies
    config_overrides = {
        "llmcore.default_provider": "openai",  # Will fail gracefully without API key
        "storage.session.type": "json",
        "storage.session.path": str(Path(tempfile.gettempdir()) / "test_sessions"),
    }

    try:
        print("🔧 Creating LLMCore instance with JSON storage (no DB required)...")
        llm = await LLMCore.create(config_overrides=config_overrides)
        print("✅ LLMCore instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create LLMCore instance: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test basic methods
    try:
        print("🔍 Testing get_available_providers()...")
        providers = llm.get_available_providers()
        print(f"✅ Available providers: {providers if providers else 'None (expected without API keys)'}")
    except Exception as e:
        print(f"❌ Failed to get available providers: {e}")
        return False

    # Test that we can access the configuration
    try:
        print("🔍 Testing configuration access...")
        log_level = llm.config.get('llmcore.log_level', 'INFO')
        print(f"✅ Configuration accessible (log_level: {log_level})")
    except Exception as e:
        print(f"❌ Failed to access configuration: {e}")
        return False

    # Test session listing (should work with JSON storage)
    try:
        print("🔍 Testing list_sessions()...")
        sessions = await llm.list_sessions()
        print(f"✅ Sessions listed: {len(sessions)} session(s) found")
    except Exception as e:
        print(f"❌ Failed to list sessions: {e}")
        return False

    # Cleanup
    try:
        await llm.close()
        print("✅ LLMCore instance closed successfully")
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")

    print("")
    print("🎉 All library mode tests passed!")
    return True


async def test_agent_availability():
    """Test that AgentManager is still available for advanced users."""
    print("")
    print("🧪 Testing AgentManager availability (optional feature)...")
    print("")

    try:
        from llmcore import AgentManager, ToolManager
        print("✅ AgentManager can be imported")
        print("✅ ToolManager can be imported")
        print("ℹ️  Note: AgentManager is available but not auto-initialized in LLMCore")
        return True
    except ImportError as e:
        print(f"❌ Failed to import agent components: {e}")
        return False


async def main():
    """Run all tests."""
    print("═══════════════════════════════════════════════════")
    print("LLMCore Library Mode Functional Test")
    print("═══════════════════════════════════════════════════")
    print("")

    test1 = await test_library_instantiation()
    test2 = await test_agent_availability()

    print("")
    print("═══════════════════════════════════════════════════")

    if test1 and test2:
        print("✅ All tests passed - llmcore is ready for library use!")
        sys.exit(0)
    else:
        print("❌ Some tests failed - please review errors above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
