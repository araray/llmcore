#!/usr/bin/env python3
"""
Import Chain Diagnostic for Darwin Layer 2.

This script traces through the import chain step by step to identify
exactly where imports are failing.

Usage:
    cd /av/data/repos/llmcore
    python diagnose_imports.py
"""

import sys
from pathlib import Path


def test_import(module_path: str, names: list = None) -> tuple:
    """
    Test importing from a module.
    
    Returns:
        (success: bool, error: str or None)
    """
    try:
        if names:
            exec(f"from {module_path} import {', '.join(names)}")
        else:
            exec(f"import {module_path}")
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("  Darwin Layer 2 - Import Chain Diagnostic")
    print("=" * 70)
    
    # Add src to path
    src_path = Path("src")
    if src_path.exists():
        sys.path.insert(0, str(src_path.absolute()))
        print(f"\n📁 Added to path: {src_path.absolute()}")
    
    # Clear any cached imports
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('llmcore')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    print(f"   Cleared {len(modules_to_clear)} cached modules")
    
    # Define the import chain tests
    tests = [
        # Level 0: Base llmcore
        ("llmcore.models", ["AgentState", "ToolCall"]),
        ("llmcore.exceptions", ["LLMCoreError"]),
        
        # Level 1: Agents base
        ("llmcore.agents.tools", ["ToolManager"]),
        ("llmcore.agents.manager", ["AgentManager"]),
        
        # Level 2: Sandbox exceptions
        ("llmcore.agents.sandbox.exceptions", ["SandboxError"]),
        
        # Level 3: Sandbox base
        ("llmcore.agents.sandbox.base", ["SandboxProvider", "SandboxConfig"]),
        
        # Level 4: Sandbox providers
        ("llmcore.agents.sandbox.docker_provider", ["DockerSandboxProvider"]),
        ("llmcore.agents.sandbox.vm_provider", ["VMSandboxProvider"]),
        
        # Level 5: Sandbox registry
        ("llmcore.agents.sandbox.registry", ["SandboxRegistry"]),
        
        # Level 6: Sandbox __init__
        ("llmcore.agents.sandbox", ["SandboxError", "SandboxRegistry"]),
        
        # Level 7: Cognitive models
        ("llmcore.agents.cognitive.models", ["CognitivePhase", "ActInput", "EnhancedAgentState"]),
        
        # Level 8: Cognitive phases (individual)
        ("llmcore.agents.cognitive.phases.perceive", ["perceive_phase"]),
        ("llmcore.agents.cognitive.phases.act", ["act_phase"]),
        
        # Level 9: Cognitive phases __init__
        ("llmcore.agents.cognitive.phases", ["perceive_phase", "act_phase"]),
        
        # Level 10: Cognitive cycle
        ("llmcore.agents.cognitive.phases.cycle", ["CognitiveCycle"]),
        
        # Level 11: Cognitive __init__
        ("llmcore.agents.cognitive", ["CognitivePhase", "ActInput", "CognitiveCycle"]),
        
        # Level 12: Persona
        ("llmcore.agents.persona", ["PersonaManager", "AgentPersona"]),
        
        # Level 13: Prompts
        ("llmcore.agents.prompts", ["PromptRegistry", "PromptTemplate"]),
        
        # Level 14: Memory integration
        ("llmcore.agents.memory", ["CognitiveMemoryIntegrator"]),
        
        # Level 15: Enhanced manager
        ("llmcore.agents.manager", ["EnhancedAgentManager", "AgentMode"]),
        
        # Level 16: Single agent
        ("llmcore.agents.single_agent", ["SingleAgentMode", "AgentResult"]),
        
        # Level 17: Agents __init__
        ("llmcore.agents", ["AgentManager", "SandboxError", "CognitiveCycle"]),
        
        # Level 18: llmcore __init__
        ("llmcore", ["LLMCore", "AgentManager", "SandboxError"]),
    ]
    
    print("\n" + "-" * 70)
    print("  Testing Import Chain")
    print("-" * 70)
    
    failed = []
    last_success_level = -1
    
    for i, (module, names) in enumerate(tests):
        success, error = test_import(module, names)
        
        if success:
            print(f"  ✅ {module}")
            if names:
                print(f"      └─ {', '.join(names)}")
            last_success_level = i
        else:
            print(f"  ❌ {module}")
            if names:
                print(f"      └─ {', '.join(names)}")
            print(f"      ERROR: {error}")
            failed.append((module, names, error))
            
            # Stop at first failure to identify root cause
            print(f"\n  ⛔ Stopped at first failure to identify root cause")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if not failed:
        print("\n  ✅ All imports successful!")
        print("     Darwin Layer 2 integration is working.")
    else:
        print(f"\n  ❌ Import failed at:")
        module, names, error = failed[0]
        print(f"     Module: {module}")
        if names:
            print(f"     Names:  {', '.join(names)}")
        print(f"     Error:  {error}")
        
        # Provide specific fix advice
        print("\n  🔧 FIX:")
        if "SandboxError" in error:
            print("     The sandbox/__init__.py is missing exception exports.")
            print("     Run: python fix_sandbox_exports.py")
        elif "phases.models" in error:
            print("     The phases/ files have wrong import paths.")
            print("     Run: python fix_darwin_imports.py")
        elif "cognitive" in module.lower():
            print("     The cognitive package has import issues.")
            print("     Run: python fix_darwin_imports.py")
        else:
            print(f"     Check the {module} module for missing exports.")
    
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
