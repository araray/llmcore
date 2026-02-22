#!/usr/bin/env python3
# scripts/verify_phase4_ast.py
"""
AST-based verification script for Phase 4 implementation.

Verifies that:
1. All 7 methods are implemented (not stubs)
2. Both tenant and legacy mode methods exist
3. Proper imports are present
4. Methods have expected structure
"""

import ast
import sys
from pathlib import Path


def verify_phase4():
    """Run AST verification for Phase 4 PostgreSQL storage implementation."""
    src_path = Path("src/llmcore/storage/postgres_session_storage.py")

    if not src_path.exists():
        print(f"❌ File not found: {src_path}")
        return False

    with open(src_path) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"❌ Syntax error in {src_path}: {e}")
        return False

    print("=" * 60)
    print("Phase 4: PostgreSQL Storage Backend - AST Verification")
    print("=" * 60)

    # Find the class
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PostgresSessionStorage":
            class_node = node
            break

    if not class_node:
        print("❌ PostgresSessionStorage class not found")
        return False

    print("✅ PostgresSessionStorage class found")

    # Get all method names
    methods = {}
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = item

    # Required methods to verify
    required_methods = {
        "save_context_preset": [
            "_save_context_preset_tenant_mode",
            "_save_context_preset_legacy_mode",
        ],
        "get_context_preset": [
            "_get_context_preset_tenant_mode",
            "_get_context_preset_legacy_mode",
        ],
        "list_context_presets": [
            "_list_context_presets_tenant_mode",
            "_list_context_presets_legacy_mode",
        ],
        "delete_context_preset": [
            "_delete_context_preset_tenant_mode",
            "_delete_context_preset_legacy_mode",
        ],
        "rename_context_preset": [
            "_rename_context_preset_tenant_mode",
            "_rename_context_preset_legacy_mode",
        ],
        "add_episode": ["_add_episode_tenant_mode", "_add_episode_legacy_mode"],
        "get_episodes": ["_get_episodes_tenant_mode", "_get_episodes_legacy_mode"],
    }

    all_ok = True

    # Verify each required method
    print("\n1. Checking public methods...")
    for method_name, helper_methods in required_methods.items():
        if method_name in methods:
            method = methods[method_name]
            # Check if it's async
            is_async = isinstance(method, ast.AsyncFunctionDef)

            # Check if it's a stub (just 'pass' or 'return None/[]')
            is_stub = len(method.body) == 1 and (
                isinstance(method.body[0], ast.Pass)
                or (
                    isinstance(method.body[0], ast.Return)
                    and isinstance(method.body[0].value, (ast.Constant, ast.List))
                    and (
                        method.body[0].value.value in (None, False)
                        if isinstance(method.body[0].value, ast.Constant)
                        else True
                    )
                )
            )

            if is_stub:
                print(f"   ❌ {method_name} - still a stub!")
                all_ok = False
            else:
                print(f"   ✅ {method_name} - implemented {'(async)' if is_async else ''}")
        else:
            print(f"   ❌ {method_name} - missing!")
            all_ok = False

    # Verify helper methods exist
    print("\n2. Checking helper methods...")
    for method_name, helper_methods in required_methods.items():
        for helper in helper_methods:
            if helper in methods:
                print(f"   ✅ {helper}")
            else:
                print(f"   ❌ {helper} - missing!")
                all_ok = False

    # Check imports
    print("\n3. Checking imports...")
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports[alias.name] = node.module

    required_imports = ["ContextPreset", "ContextPresetItem", "Episode", "EpisodeType"]
    for imp in required_imports:
        if imp in imports:
            print(f"   ✅ {imp} imported from {imports[imp]}")
        else:
            print(f"   ⚠️ {imp} not found in direct imports (may be in models import)")

    # Check for proper error handling patterns
    print("\n4. Checking implementation patterns...")

    # Check for SessionStorageError usage
    source_lower = source.lower()
    if "sessionstorageerror" in source_lower:
        print("   ✅ SessionStorageError used for error handling")
    else:
        print("   ⚠️ SessionStorageError not found in code")

    # Check for _tenant_session checks
    if "_tenant_session" in source and "hasattr" in source:
        print("   ✅ Tenant mode checking pattern present")
    else:
        print("   ⚠️ Tenant mode checking pattern not found")

    # Check for _pool usage
    if "_pool.connection" in source:
        print("   ✅ Legacy pool connection pattern present")
    else:
        print("   ⚠️ Legacy pool connection pattern not found")

    # Check for proper transaction handling
    if "transaction" in source:
        print("   ✅ Transaction handling present")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ PHASE 4 VERIFICATION PASSED")
        print("   All 7 methods implemented with tenant/legacy mode support")
    else:
        print("❌ PHASE 4 VERIFICATION FAILED")
        print("   Some methods still missing or are stubs")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = verify_phase4()
    sys.exit(0 if success else 1)
