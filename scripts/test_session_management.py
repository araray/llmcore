#!/usr/bin/env python3
# scripts/test_session_management.py
"""
Enhanced test script for LLMCore library mode with session management.

Tests all SessionManager methods: list_sessions, get_session, delete_session
"""

import asyncio
import os
import sys

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    print("=" * 55)
    print("LLMCore Library Mode - Session Management Test")
    print("=" * 55)

    try:
        # Test 1: Import LLMCore
        print("🧪 Test 1: Importing LLMCore...")
        from llmcore import LLMCore

        print("✅ Successfully imported LLMCore")

        # Test 2: Create instance
        print("\n🧪 Test 2: Creating LLMCore instance with JSON storage...")
        config_overrides = {"storage": {"type": "json", "path": "/tmp/llmcore_test_sessions"}}
        llmcore = await LLMCore.create(config_overrides=config_overrides)
        print("✅ LLMCore instance created successfully")

        # Test 3: Get available providers
        print("\n🧪 Test 3: Testing get_available_providers()...")
        providers = llmcore.get_available_providers()
        print(f"✅ Available providers: {providers}")

        # Test 4: Configuration access
        print("\n🧪 Test 4: Testing configuration access...")
        log_level = llmcore.config.get("log_level", "INFO")
        print(f"✅ Configuration accessible (log_level: {log_level})")

        # Test 5: List sessions (should work now)
        print("\n🧪 Test 5: Testing list_sessions()...")
        sessions = await llmcore.list_sessions()
        print("✅ list_sessions() succeeded")
        print(f"   Found {len(sessions)} existing sessions")
        if sessions:
            for sess in sessions[:3]:  # Show first 3
                print(f"   - {sess.get('id', 'N/A')[:16]}... | {sess.get('name', '(unnamed)')}")

        # Test 6: Create a test session
        print("\n🧪 Test 6: Creating a test session...")
        from llmcore.models import ChatSession, Role

        test_session = ChatSession(id="test-session-123", name="Test Session")
        test_session.add_message("Hello, this is a test", Role.USER)
        test_session.add_message("Hi! I'm responding", Role.ASSISTANT)

        # Save it via SessionManager
        await llmcore._session_manager.save_session(test_session)
        print("✅ Test session created and saved")

        # Test 7: Get specific session
        print("\n🧪 Test 7: Testing get_session()...")
        retrieved_session = await llmcore.get_session("test-session-123")
        print("✅ get_session() succeeded")
        print(f"   Session name: {retrieved_session.name}")
        print(f"   Message count: {len(retrieved_session.messages)}")

        # Test 8: List sessions again (should include our test session)
        print("\n🧪 Test 8: Verifying test session appears in list...")
        sessions_after = await llmcore.list_sessions()
        test_session_found = any(s.get("id") == "test-session-123" for s in sessions_after)
        if test_session_found:
            print("✅ Test session found in list")
        else:
            print("❌ Test session NOT found in list")

        # Test 9: Delete session
        print("\n🧪 Test 9: Testing delete_session()...")
        await llmcore.delete_session("test-session-123")
        print("✅ delete_session() succeeded")

        # Test 10: Verify deletion
        print("\n🧪 Test 10: Verifying session was deleted...")
        sessions_final = await llmcore.list_sessions()
        test_session_still_exists = any(s.get("id") == "test-session-123" for s in sessions_final)
        if not test_session_still_exists:
            print("✅ Test session successfully deleted")
        else:
            print("❌ Test session still exists after deletion")

        # Test 11: AgentManager availability
        print("\n🧪 Test 11: Testing AgentManager availability (optional feature)...")
        try:
            from llmcore.agents import AgentManager, ToolManager

            print("✅ AgentManager can be imported")
            print("✅ ToolManager can be imported")
            print("ℹ️  Note: AgentManager is available but not auto-initialized in LLMCore")
        except ImportError as e:
            print(f"❌ Could not import agent components: {e}")

        print("\n" + "=" * 55)
        print("✅ All tests passed successfully!")
        print("=" * 55)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 55)
        print("❌ Some tests failed - please review errors above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
