#!/usr/bin/env python3
"""
Darwin Layer 2 Diagnostic Script.

Run this to identify what's missing or broken in the cognitive package.

Usage:
    cd /av/data/repos/llmcore
    source venv/bin/activate
    python diagnose_darwin.py
"""

import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  Darwin Layer 2 Diagnostic Tool")
    print("=" * 60)

    # Find llmcore source directory
    possible_paths = [
        Path("src/llmcore"),
        Path("llmcore"),
        Path("."),
    ]

    llmcore_path = None
    for p in possible_paths:
        if (p / "agents").is_dir():
            llmcore_path = p
            break

    if not llmcore_path:
        print("\n❌ Could not find llmcore source directory")
        print("   Run this from your llmcore repository root")
        return 1

    agents_path = llmcore_path / "agents"
    cognitive_path = agents_path / "cognitive"
    phases_path = cognitive_path / "phases"

    print(f"\n📁 llmcore path: {llmcore_path.absolute()}")
    print(f"📁 agents path: {agents_path.absolute()}")
    print(f"📁 cognitive path: {cognitive_path.absolute()}")

    # Check directory structure
    print("\n" + "-" * 60)
    print("  Directory Structure Check")
    print("-" * 60)

    dirs_to_check = [
        agents_path,
        cognitive_path,
        phases_path,
        agents_path / "persona",
        agents_path / "prompts",
        agents_path / "memory",
    ]

    for d in dirs_to_check:
        if d.exists():
            print(f"  ✅ {d.relative_to(llmcore_path)}")
        else:
            print(f"  ❌ {d.relative_to(llmcore_path)} - MISSING")

    # Check files
    print("\n" + "-" * 60)
    print("  Critical Files Check")
    print("-" * 60)

    files_to_check = [
        # Cognitive package
        (cognitive_path / "__init__.py", "cognitive/__init__.py"),
        (cognitive_path / "models.py", "cognitive/models.py"),
        (phases_path / "__init__.py", "cognitive/phases/__init__.py"),
        (phases_path / "cycle.py", "cognitive/phases/cycle.py"),
        (phases_path / "perceive.py", "cognitive/phases/perceive.py"),
        (phases_path / "plan.py", "cognitive/phases/plan.py"),
        (phases_path / "think.py", "cognitive/phases/think.py"),
        (phases_path / "validate.py", "cognitive/phases/validate.py"),
        (phases_path / "act.py", "cognitive/phases/act.py"),
        (phases_path / "observe.py", "cognitive/phases/observe.py"),
        (phases_path / "reflect.py", "cognitive/phases/reflect.py"),
        (phases_path / "update.py", "cognitive/phases/update.py"),
        # Other packages
        (agents_path / "manager.py", "manager.py"),
        (agents_path / "single_agent.py", "single_agent.py"),
        (agents_path / "persona" / "__init__.py", "persona/__init__.py"),
        (agents_path / "prompts" / "__init__.py", "prompts/__init__.py"),
        (agents_path / "memory" / "__init__.py", "memory/__init__.py"),
    ]

    missing_files = []
    for path, name in files_to_check:
        if path.exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - MISSING")
            missing_files.append((path, name))

    # Check cognitive/__init__.py exports
    print("\n" + "-" * 60)
    print("  cognitive/__init__.py Export Check")
    print("-" * 60)

    cog_init = cognitive_path / "__init__.py"
    if cog_init.exists():
        content = cog_init.read_text()

        required_exports = [
            "ActInput",
            "ActOutput",
            "ObserveInput",
            "ObserveOutput",
            "PerceiveInput",
            "PerceiveOutput",
            "PlanInput",
            "PlanOutput",
            "ReflectInput",
            "ReflectOutput",
            "ThinkInput",
            "ThinkOutput",
            "UpdateInput",
            "UpdateOutput",
            "ValidateInput",
            "ValidateOutput",
            "CognitivePhase",
            "IterationStatus",
            "ValidationResult",
            "ConfidenceLevel",
            "CycleIteration",
            "EnhancedAgentState",
            "CognitiveCycle",
            "perceive_phase",
            "plan_phase",
            "think_phase",
            "validate_phase",
            "act_phase",
            "observe_phase",
            "reflect_phase",
            "update_phase",
        ]

        missing_exports = []
        for export in required_exports:
            if export in content:
                print(f"  ✅ {export}")
            else:
                print(f"  ❌ {export} - NOT EXPORTED")
                missing_exports.append(export)

        if missing_exports:
            print(f"\n  ⚠️ Missing {len(missing_exports)} exports in cognitive/__init__.py")
            print("  Use the fix files provided to correct this.")
    else:
        print("  ❌ cognitive/__init__.py does not exist!")
        missing_exports = ["ALL"]

    # Check phases/__init__.py
    print("\n" + "-" * 60)
    print("  phases/__init__.py Export Check")
    print("-" * 60)

    phases_init = phases_path / "__init__.py"
    if phases_init.exists():
        content = phases_init.read_text()

        phase_exports = [
            "perceive_phase",
            "plan_phase",
            "think_phase",
            "validate_phase",
            "act_phase",
            "observe_phase",
            "reflect_phase",
            "update_phase",
        ]

        missing_phase_exports = []
        for export in phase_exports:
            if export in content:
                print(f"  ✅ {export}")
            else:
                print(f"  ❌ {export} - NOT EXPORTED")
                missing_phase_exports.append(export)
    else:
        print("  ❌ phases/__init__.py does not exist!")
        missing_phase_exports = ["ALL"]

    # Try actual imports
    print("\n" + "-" * 60)
    print("  Import Test")
    print("-" * 60)

    try:
        # Add src to path if needed
        src_path = str(
            llmcore_path.parent if llmcore_path.name == "llmcore" else llmcore_path.parent / "src"
        )
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from llmcore.agents.cognitive import CognitivePhase

        print("  ✅ CognitivePhase imports successfully")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")

    try:
        from llmcore.agents.cognitive import ActInput

        print("  ✅ ActInput imports successfully")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")

    try:
        from llmcore.agents.cognitive import act_phase

        print("  ✅ act_phase imports successfully")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")

    try:
        from llmcore.agents.cognitive import CognitiveCycle

        print("  ✅ CognitiveCycle imports successfully")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 60)

    issues = []
    if missing_files:
        issues.append(f"Missing {len(missing_files)} file(s)")
    if missing_exports:
        issues.append("Missing exports in cognitive/__init__.py")
    if missing_phase_exports:
        issues.append("Missing exports in phases/__init__.py")

    if issues:
        print("\n  ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"     • {issue}")
        print("\n  FIX INSTRUCTIONS:")
        print("     1. Copy cognitive_init_fix.py to:")
        print(f"        {cognitive_path / '__init__.py'}")
        print("     2. Copy phases_init_fix.py to:")
        print(f"        {phases_path / '__init__.py'}")
        print("     3. Reinstall: pip install -e .")
        print("     4. Re-run sanity_check.py")
    else:
        print("\n  ✅ All diagnostics passed!")
        print("     Run sanity_check.py to confirm.")

    print()
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
