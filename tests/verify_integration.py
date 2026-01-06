#!/usr/bin/env python3
"""
llmcore Integration Verification Script
========================================

This script performs comprehensive checks to verify that llmcore is 
properly integrated and usable, with a focus on Darwin Layer 2 components.

Checks Performed:
1. Core llmcore imports work
2. Sandbox system (Layer 1) is accessible
3. Layer 2 components are accessible (if present)
4. Test collection works
5. Key component instantiation works

Usage:
    cd /path/to/llmcore
    python verify_llmcore_integration.py

Exit Codes:
    0 - All checks passed
    1 - Some checks failed
    2 - Critical import failure

Author: Darwin Integration
Date: December 2025
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Core imports that must always work
CORE_IMPORTS = [
    ("llmcore", "LLMCore"),
    ("llmcore", "AgentManager"),
    ("llmcore", "ToolManager"),
    ("llmcore.models", "ChatSession"),
    ("llmcore.models", "Message"),
    ("llmcore.models", "AgentState"),
    ("llmcore.models", "AgentTask"),
    ("llmcore.exceptions", "LLMCoreError"),
]

# Sandbox (Layer 1) imports
SANDBOX_IMPORTS = [
    ("llmcore.agents.sandbox", "SandboxProvider"),
    ("llmcore.agents.sandbox", "SandboxRegistry"),
    ("llmcore.agents.sandbox", "SandboxRegistryConfig"),
    ("llmcore.agents.sandbox", "SandboxConfig"),
    ("llmcore.agents.sandbox", "SandboxMode"),
    ("llmcore.agents.sandbox", "SandboxAccessLevel"),
    ("llmcore.agents.sandbox", "SandboxStatus"),
    ("llmcore.agents.sandbox", "ExecutionResult"),
    ("llmcore.agents.sandbox", "DockerSandboxProvider"),
    ("llmcore.agents.sandbox", "VMSandboxProvider"),
    ("llmcore.agents.sandbox", "EphemeralResourceManager"),
    ("llmcore.agents.sandbox", "OutputTracker"),
    ("llmcore.agents.sandbox", "set_active_sandbox"),
    ("llmcore.agents.sandbox", "clear_active_sandbox"),
    ("llmcore.agents.sandbox.exceptions", "SandboxError"),
]

# Layer 2 imports (may not exist yet)
LAYER2_IMPORTS = [
    ("llmcore.agents", "EnhancedAgentManager"),
    ("llmcore.agents", "AgentMode"),
    ("llmcore.agents", "SingleAgentMode"),
    ("llmcore.agents", "PersonaManager"),
    ("llmcore.agents", "PromptRegistry"),
    ("llmcore.agents", "CognitiveCycle"),
    ("llmcore.agents", "EnhancedAgentState"),
    ("llmcore.agents.cognitive", "CognitivePhase"),
    ("llmcore.agents.persona", "AgentPersona"),
    ("llmcore.agents.prompts", "PromptTemplate"),
]


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def green(text: str) -> str:
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def red(text: str) -> str:
    return f"{Colors.RED}{text}{Colors.RESET}"


def yellow(text: str) -> str:
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def blue(text: str) -> str:
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def check_import(module_name: str, attribute: str) -> Tuple[bool, Optional[str]]:
    """
    Check if an import works.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        module = importlib.import_module(module_name)
        if attribute:
            getattr(module, attribute)
        return True, None
    except ImportError as e:
        return False, f"ImportError: {e}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_imports(imports: List[Tuple[str, str]], category: str) -> Tuple[int, int]:
    """
    Check a list of imports.
    
    Returns:
        Tuple of (passed, failed)
    """
    print(f"\n{bold(category)}")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for module_name, attribute in imports:
        full_name = f"{module_name}.{attribute}" if attribute else module_name
        success, error = check_import(module_name, attribute)
        
        if success:
            print(f"  {green('✓')} {full_name}")
            passed += 1
        else:
            print(f"  {red('✗')} {full_name}")
            print(f"      {red(error)}")
            failed += 1
    
    return passed, failed


def check_python_path() -> bool:
    """Check if src directory is in Python path."""
    cwd = Path.cwd()
    src_path = cwd / "src"
    
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        return True
    return False


def check_test_collection() -> Tuple[bool, str]:
    """Check if pytest can collect tests."""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/agents/", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Count tests
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'test' in line.lower() and 'selected' in line.lower():
                    return True, line
            return True, f"Tests collected successfully"
        else:
            return False, result.stderr or result.stdout
    except FileNotFoundError:
        return False, "pytest not found"
    except subprocess.TimeoutExpired:
        return False, "Test collection timed out"
    except Exception as e:
        return False, str(e)


def check_sandbox_instantiation() -> Tuple[bool, str]:
    """Check if sandbox components can be instantiated."""
    try:
        from llmcore.agents.sandbox import SandboxConfig, SandboxRegistryConfig, SandboxMode
        
        # Create basic config
        config = SandboxConfig()
        assert config.sandbox_id is not None
        
        # Create registry config
        reg_config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True
        )
        assert reg_config.mode == SandboxMode.DOCKER
        
        return True, "SandboxConfig and SandboxRegistryConfig created successfully"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_layer2_instantiation() -> Tuple[bool, str]:
    """Check if Layer 2 components can be instantiated."""
    try:
        # Try to import Layer 2 components
        from llmcore.agents import PersonaManager
        
        # Create persona manager
        manager = PersonaManager()
        personas = manager.list_personas()
        
        return True, f"PersonaManager created with {len(personas)} personas"
    except ImportError:
        return False, "Layer 2 components not found (may not be integrated yet)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(bold("llmcore Integration Verification"))
    print("=" * 70)
    
    # Check and add src to path
    added_src = check_python_path()
    if added_src:
        print(f"\n{yellow('Note:')} Added src/ to Python path")
    
    total_passed = 0
    total_failed = 0
    
    # Check core imports
    passed, failed = check_imports(CORE_IMPORTS, "Core llmcore Imports")
    total_passed += passed
    total_failed += failed
    
    if failed > 0:
        print(f"\n{red('CRITICAL:')} Core imports failed. llmcore is not usable.")
        sys.exit(2)
    
    # Check sandbox imports
    passed, failed = check_imports(SANDBOX_IMPORTS, "Sandbox System (Layer 1) Imports")
    total_passed += passed
    total_failed += failed
    
    # Check Layer 2 imports (optional)
    print(f"\n{bold('Darwin Layer 2 Imports (Optional)')}")
    print("-" * 60)
    layer2_passed = 0
    layer2_failed = 0
    
    for module_name, attribute in LAYER2_IMPORTS:
        full_name = f"{module_name}.{attribute}" if attribute else module_name
        success, error = check_import(module_name, attribute)
        
        if success:
            print(f"  {green('✓')} {full_name}")
            layer2_passed += 1
        else:
            print(f"  {yellow('○')} {full_name} (not found)")
            layer2_failed += 1
    
    # Component instantiation checks
    print(f"\n{bold('Component Instantiation Checks')}")
    print("-" * 60)
    
    # Sandbox
    success, msg = check_sandbox_instantiation()
    if success:
        print(f"  {green('✓')} Sandbox: {msg}")
    else:
        print(f"  {red('✗')} Sandbox: {msg}")
        total_failed += 1
    
    # Layer 2
    success, msg = check_layer2_instantiation()
    if success:
        print(f"  {green('✓')} Layer 2: {msg}")
    else:
        print(f"  {yellow('○')} Layer 2: {msg}")
    
    # Test collection check
    print(f"\n{bold('Test Collection Check')}")
    print("-" * 60)
    success, msg = check_test_collection()
    if success:
        print(f"  {green('✓')} pytest: {msg}")
    else:
        print(f"  {red('✗')} pytest: {msg}")
        total_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(bold("Summary"))
    print("=" * 70)
    print(f"  Core imports:    {green(f'{len(CORE_IMPORTS)} passed')}")
    print(f"  Sandbox imports: {green(str(passed)) if failed == 0 else f'{green(str(passed))} passed, {red(str(failed))} failed'}")
    print(f"  Layer 2 imports: {green(str(layer2_passed))} found, {yellow(str(layer2_failed))} not found")
    print()
    
    if total_failed == 0:
        print(green(bold("✓ All critical checks passed!")))
        print()
        print("llmcore is usable. You can now:")
        print("  1. Run tests: pytest tests/ -v")
        print("  2. Run sandbox tests: pytest tests/agents/sandbox/ -v")
        if layer2_passed > 0:
            print("  3. Use Darwin Layer 2 components")
        sys.exit(0)
    else:
        print(red(bold(f"✗ {total_failed} check(s) failed")))
        print()
        print("Please fix the issues above before using llmcore.")
        sys.exit(1)


if __name__ == "__main__":
    main()
