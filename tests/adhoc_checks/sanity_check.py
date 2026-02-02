#!/usr/bin/env python3
"""
Quick Sanity Check for Darwin Layer 2 Integration.

This is a fast verification script that checks the essential imports
and components are working. For comprehensive verification, use
verify_darwin_layer2.py instead.

Usage:
    python sanity_check.py
"""

import sys


def main():
    """Run quick sanity check."""
    print("🔍 Darwin Layer 2 Quick Sanity Check\n")

    errors = []

    # Check 1: Core imports
    print("1. Testing core imports...")
    try:
        from llmcore.agents import (
            AgentMode,
            AgentResult,
            EnhancedAgentManager,
            SingleAgentMode,
        )

        print("   ✅ Core imports OK")
    except ImportError as e:
        errors.append(f"Core imports failed: {e}")
        print(f"   ❌ Core imports FAILED: {e}")

    # Check 2: Cognitive cycle
    print("2. Testing cognitive cycle imports...")
    try:
        from llmcore.agents import (
            CognitiveCycle,
            CognitivePhase,
            EnhancedAgentState,
        )

        # Verify 8 phases
        phases = list(CognitivePhase)
        assert len(phases) == 8, f"Expected 8 phases, got {len(phases)}"
        print(f"   ✅ Cognitive cycle OK ({len(phases)} phases)")
    except (ImportError, AssertionError) as e:
        errors.append(f"Cognitive cycle failed: {e}")
        print(f"   ❌ Cognitive cycle FAILED: {e}")

    # Check 3: Persona system
    print("3. Testing persona system imports...")
    try:
        from llmcore.agents import (
            PersonaManager,
        )

        manager = PersonaManager()
        personas = manager.list_personas()
        print(f"   ✅ Persona system OK ({len(personas)} built-in personas)")
    except Exception as e:
        errors.append(f"Persona system failed: {e}")
        print(f"   ❌ Persona system FAILED: {e}")

    # Check 4: Prompt library
    print("4. Testing prompt library imports...")
    try:
        from llmcore.agents import (
            PromptRegistry,
        )

        registry = PromptRegistry()
        print("   ✅ Prompt library OK")
    except Exception as e:
        errors.append(f"Prompt library failed: {e}")
        print(f"   ❌ Prompt library FAILED: {e}")

    # Check 5: Memory integration
    print("5. Testing memory integration imports...")
    try:
        from llmcore.agents import CognitiveMemoryIntegrator

        print("   ✅ Memory integration OK")
    except ImportError as e:
        errors.append(f"Memory integration failed: {e}")
        print(f"   ❌ Memory integration FAILED: {e}")

    # Check 6: Backward compatibility
    print("6. Testing backward compatibility...")
    try:
        from llmcore.agents import EnhancedAgentManager
        from llmcore.agents.manager import AgentManager

        assert issubclass(EnhancedAgentManager, AgentManager)
        print("   ✅ Backward compatibility OK (EnhancedAgentManager extends AgentManager)")
    except Exception as e:
        errors.append(f"Backward compatibility failed: {e}")
        print(f"   ❌ Backward compatibility FAILED: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"❌ FAILED - {len(errors)} error(s) found:")
        for err in errors:
            print(f"   • {err}")
        return 1
    else:
        print("✅ ALL CHECKS PASSED")
        print("   Darwin Layer 2 is properly integrated!")
        print("   llmcore is ready for use.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
