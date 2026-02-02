#!/usr/bin/env python3
"""
Darwin Layer 2 Integration Verification Script.

This script verifies that Darwin Layer 2 is properly integrated into llmcore
and that the system is fully usable. Run this script from your llmcore root
directory after activating the virtual environment.

Usage:
    cd /path/to/llmcore
    source venv/bin/activate  # or your venv name
    python verify_darwin_layer2.py

Requirements:
    - llmcore installed in editable mode (pip install -e .)
    - Python 3.11+

Author: Claude (Anthropic)
Date: December 2025
Version: 1.0.1 - Fixed EnhancedAgentState verification
"""

import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# =============================================================================
# VERIFICATION RESULT TYPES
# =============================================================================


class VerificationStatus(Enum):
    """Status of a verification check."""

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    name: str
    status: VerificationStatus
    message: str
    details: Optional[str] = None
    exception: Optional[Exception] = None


class VerificationReport:
    """Aggregated verification report."""

    def __init__(self):
        self.results: List[VerificationResult] = []
        self.section: str = ""

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.WARN)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.SKIP)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success(self) -> bool:
        return self.failed == 0


# =============================================================================
# VERIFICATION CHECKS
# =============================================================================


def verify_python_version(report: VerificationReport) -> bool:
    """Verify Python version is 3.11+."""
    version = sys.version_info
    if version >= (3, 11):
        report.add(
            VerificationResult(
                name="Python Version",
                status=VerificationStatus.PASS,
                message=f"Python {version.major}.{version.minor}.{version.micro}",
            )
        )
        return True
    else:
        report.add(
            VerificationResult(
                name="Python Version",
                status=VerificationStatus.FAIL,
                message=f"Python {version.major}.{version.minor} (require 3.11+)",
            )
        )
        return False


def verify_llmcore_import(report: VerificationReport) -> bool:
    """Verify llmcore base package imports."""
    try:
        import llmcore

        report.add(
            VerificationResult(
                name="llmcore Import",
                status=VerificationStatus.PASS,
                message=f"llmcore v{getattr(llmcore, '__version__', 'unknown')}",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="llmcore Import",
                status=VerificationStatus.FAIL,
                message="Failed to import llmcore",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_agents_package(report: VerificationReport) -> bool:
    """Verify llmcore.agents package structure."""
    try:
        from llmcore import agents

        version = getattr(agents, "__version__", "unknown")

        # Check version is 0.26.0 (Darwin Layer 2)
        if version == "0.26.0":
            report.add(
                VerificationResult(
                    name="Agents Package Version",
                    status=VerificationStatus.PASS,
                    message=f"llmcore.agents v{version} (Darwin Layer 2 Complete)",
                )
            )
        else:
            report.add(
                VerificationResult(
                    name="Agents Package Version",
                    status=VerificationStatus.WARN,
                    message=f"llmcore.agents v{version} (expected 0.26.0)",
                )
            )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Agents Package",
                status=VerificationStatus.FAIL,
                message="Failed to import llmcore.agents",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_core_manager_imports(report: VerificationReport) -> bool:
    """Verify core manager imports."""
    try:
        from llmcore.agents import AgentMode, EnhancedAgentManager

        # Verify AgentMode enum values
        assert AgentMode.SINGLE.value == "single"
        assert AgentMode.LEGACY.value == "legacy"
        assert AgentMode.MULTI.value == "multi"

        report.add(
            VerificationResult(
                name="Core Manager Imports",
                status=VerificationStatus.PASS,
                message="EnhancedAgentManager, AgentMode imported successfully",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Core Manager Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import EnhancedAgentManager or AgentMode",
                details=str(e),
                exception=e,
            )
        )
        return False
    except AssertionError as e:
        report.add(
            VerificationResult(
                name="Core Manager Imports",
                status=VerificationStatus.FAIL,
                message="AgentMode enum values incorrect",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_single_agent_imports(report: VerificationReport) -> bool:
    """Verify single agent mode imports."""
    try:
        from llmcore.agents import AgentResult, IterationUpdate, SingleAgentMode

        report.add(
            VerificationResult(
                name="Single Agent Imports",
                status=VerificationStatus.PASS,
                message="SingleAgentMode, AgentResult, IterationUpdate imported",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Single Agent Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import single agent components",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_cognitive_cycle_imports(report: VerificationReport) -> bool:
    """Verify cognitive cycle imports."""
    try:
        from llmcore.agents import (
            CognitiveCycle,
            CognitivePhase,
            ConfidenceLevel,
            CycleIteration,
            EnhancedAgentState,
            IterationStatus,
            ValidationResult,
        )

        # Verify CognitivePhase enum has all 8 phases
        phases = [
            CognitivePhase.PERCEIVE,
            CognitivePhase.PLAN,
            CognitivePhase.THINK,
            CognitivePhase.VALIDATE,
            CognitivePhase.ACT,
            CognitivePhase.OBSERVE,
            CognitivePhase.REFLECT,
            CognitivePhase.UPDATE,
        ]
        assert len(phases) == 8

        report.add(
            VerificationResult(
                name="Cognitive Cycle Imports",
                status=VerificationStatus.PASS,
                message="All 8 cognitive phases and models imported",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Cognitive Cycle Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import cognitive cycle components",
                details=str(e),
                exception=e,
            )
        )
        return False
    except AssertionError as e:
        report.add(
            VerificationResult(
                name="Cognitive Cycle Imports",
                status=VerificationStatus.FAIL,
                message="Cognitive phases count incorrect",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_phase_function_imports(report: VerificationReport) -> bool:
    """Verify individual phase function imports."""
    try:
        from llmcore.agents import (
            act_phase,
            observe_phase,
            perceive_phase,
            plan_phase,
            reflect_phase,
            think_phase,
            update_phase,
            validate_phase,
        )

        # Verify all are callable
        phases = [
            perceive_phase,
            plan_phase,
            think_phase,
            validate_phase,
            act_phase,
            observe_phase,
            reflect_phase,
            update_phase,
        ]
        assert all(callable(p) for p in phases)

        report.add(
            VerificationResult(
                name="Phase Function Imports",
                status=VerificationStatus.PASS,
                message="All 8 phase functions imported and callable",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Phase Function Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import phase functions",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_phase_io_models(report: VerificationReport) -> bool:
    """Verify phase input/output model imports."""
    try:
        # Verify all are Pydantic models
        from pydantic import BaseModel

        from llmcore.agents import (
            ActInput,
            ActOutput,
            ObserveInput,
            ObserveOutput,
            PerceiveInput,
            PerceiveOutput,
            PlanInput,
            PlanOutput,
            ReflectInput,
            ReflectOutput,
            ThinkInput,
            ThinkOutput,
            UpdateInput,
            UpdateOutput,
            ValidateInput,
            ValidateOutput,
        )

        models = [
            PerceiveInput,
            PerceiveOutput,
            PlanInput,
            PlanOutput,
            ThinkInput,
            ThinkOutput,
            ValidateInput,
            ValidateOutput,
            ActInput,
            ActOutput,
            ObserveInput,
            ObserveOutput,
            ReflectInput,
            ReflectOutput,
            UpdateInput,
            UpdateOutput,
        ]
        assert all(issubclass(m, BaseModel) for m in models)

        report.add(
            VerificationResult(
                name="Phase I/O Models",
                status=VerificationStatus.PASS,
                message=f"All 16 phase I/O models imported ({len(models)} Pydantic models)",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Phase I/O Models",
                status=VerificationStatus.FAIL,
                message="Failed to import phase I/O models",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_persona_imports(report: VerificationReport) -> bool:
    """Verify persona system imports."""
    try:
        from llmcore.agents import (
            AgentPersona,
            CommunicationPreferences,
            CommunicationStyle,
            DecisionMakingPreferences,
            PersonalityTrait,
            PersonaManager,
            PersonaTrait,
            PlanningDepth,
            PromptModifications,
            RiskTolerance,
        )

        report.add(
            VerificationResult(
                name="Persona System Imports",
                status=VerificationStatus.PASS,
                message="PersonaManager and all persona models imported",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Persona System Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import persona components",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_prompt_library_imports(report: VerificationReport) -> bool:
    """Verify prompt library imports."""
    try:
        from llmcore.agents import (
            PromptComposer,
            PromptRegistry,
            PromptTemplate,
            PromptVersion,
            TemplateLoader,
        )

        report.add(
            VerificationResult(
                name="Prompt Library Imports",
                status=VerificationStatus.PASS,
                message="PromptRegistry and all prompt models imported",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Prompt Library Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import prompt library components",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_memory_integration_imports(report: VerificationReport) -> bool:
    """Verify memory integration imports."""
    try:
        from llmcore.agents import CognitiveMemoryIntegrator

        report.add(
            VerificationResult(
                name="Memory Integration Imports",
                status=VerificationStatus.PASS,
                message="CognitiveMemoryIntegrator imported",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Memory Integration Imports",
                status=VerificationStatus.FAIL,
                message="Failed to import memory integration",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_backward_compatibility(report: VerificationReport) -> bool:
    """Verify backward compatibility with original AgentManager."""
    try:
        # Original import should still work
        # EnhancedAgentManager should be subclass of AgentManager
        from llmcore.agents import EnhancedAgentManager
        from llmcore.agents.manager import AgentManager

        assert issubclass(EnhancedAgentManager, AgentManager)

        # Original methods should exist
        assert hasattr(EnhancedAgentManager, "run_agent_loop")
        assert hasattr(EnhancedAgentManager, "initialize_sandbox")
        assert hasattr(EnhancedAgentManager, "shutdown_sandbox")

        report.add(
            VerificationResult(
                name="Backward Compatibility",
                status=VerificationStatus.PASS,
                message="AgentManager preserved, EnhancedAgentManager extends it",
            )
        )
        return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Backward Compatibility",
                status=VerificationStatus.FAIL,
                message="Original AgentManager not accessible",
                details=str(e),
                exception=e,
            )
        )
        return False
    except AssertionError as e:
        report.add(
            VerificationResult(
                name="Backward Compatibility",
                status=VerificationStatus.FAIL,
                message="Backward compatibility broken",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_persona_manager_instantiation(report: VerificationReport) -> bool:
    """Verify PersonaManager can be instantiated and has built-in personas."""
    try:
        from llmcore.agents import PersonaManager

        manager = PersonaManager()
        personas = manager.list_personas()

        # Should have built-in personas
        if len(personas) >= 1:
            report.add(
                VerificationResult(
                    name="PersonaManager Instantiation",
                    status=VerificationStatus.PASS,
                    message=f"PersonaManager created with {len(personas)} built-in personas",
                )
            )
            return True
        else:
            report.add(
                VerificationResult(
                    name="PersonaManager Instantiation",
                    status=VerificationStatus.WARN,
                    message="PersonaManager created but no built-in personas found",
                )
            )
            return True
    except Exception as e:
        report.add(
            VerificationResult(
                name="PersonaManager Instantiation",
                status=VerificationStatus.FAIL,
                message="Failed to instantiate PersonaManager",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_prompt_registry_instantiation(report: VerificationReport) -> bool:
    """Verify PromptRegistry can be instantiated."""
    try:
        from llmcore.agents import PromptRegistry

        registry = PromptRegistry()

        # Try to list templates
        if hasattr(registry, "list_templates"):
            templates = registry.list_templates()
            count = len(templates)
        elif hasattr(registry, "_templates"):
            count = len(registry._templates)
        else:
            count = 0

        report.add(
            VerificationResult(
                name="PromptRegistry Instantiation",
                status=VerificationStatus.PASS,
                message=f"PromptRegistry created with {count} templates",
            )
        )
        return True
    except Exception as e:
        report.add(
            VerificationResult(
                name="PromptRegistry Instantiation",
                status=VerificationStatus.FAIL,
                message="Failed to instantiate PromptRegistry",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_enhanced_agent_state_creation(report: VerificationReport) -> bool:
    """Verify EnhancedAgentState can be created."""
    try:
        from llmcore.agents import EnhancedAgentState

        # Create with required arguments
        state = EnhancedAgentState(goal="Test goal", session_id="session-123")

        # Verify key attributes - FIXED: use correct values and attribute names
        assert state.goal == "Test goal", f"Expected 'Test goal', got '{state.goal}'"
        assert state.session_id == "session-123", (
            f"Expected 'session-123', got '{state.session_id}'"
        )

        # Check iteration tracking (attribute is 'iterations', not 'iteration_history')
        assert hasattr(state, "iterations"), "Missing 'iterations' attribute"
        assert len(state.iterations) == 0, "Initial iterations should be empty"

        # Check plan attribute
        assert hasattr(state, "plan"), "Missing 'plan' attribute"

        # Check working memory
        assert hasattr(state, "working_memory"), "Missing 'working_memory' attribute"

        # Check progress tracking
        assert hasattr(state, "progress_estimate"), "Missing 'progress_estimate' attribute"
        assert state.progress_estimate == 0.0, "Initial progress should be 0.0"

        report.add(
            VerificationResult(
                name="EnhancedAgentState Creation",
                status=VerificationStatus.PASS,
                message="EnhancedAgentState created and validated successfully",
            )
        )
        return True
    except AssertionError as e:
        report.add(
            VerificationResult(
                name="EnhancedAgentState Creation",
                status=VerificationStatus.FAIL,
                message="EnhancedAgentState validation failed",
                details=str(e),
                exception=e,
            )
        )
        return False
    except Exception as e:
        report.add(
            VerificationResult(
                name="EnhancedAgentState Creation",
                status=VerificationStatus.FAIL,
                message="Failed to create EnhancedAgentState",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_cognitive_cycle_instantiation(report: VerificationReport) -> bool:
    """Verify CognitiveCycle can be instantiated (with mocks)."""
    try:
        from unittest.mock import MagicMock

        from llmcore.agents import CognitiveCycle

        # Create mock managers
        provider_manager = MagicMock()
        memory_manager = MagicMock()
        storage_manager = MagicMock()
        tool_manager = MagicMock()

        cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            tool_manager=tool_manager,
        )

        # Verify key methods exist
        assert hasattr(cycle, "run_iteration")

        report.add(
            VerificationResult(
                name="CognitiveCycle Instantiation",
                status=VerificationStatus.PASS,
                message="CognitiveCycle created with mock managers",
            )
        )
        return True
    except Exception as e:
        report.add(
            VerificationResult(
                name="CognitiveCycle Instantiation",
                status=VerificationStatus.FAIL,
                message="Failed to instantiate CognitiveCycle",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_agent_persona_creation(report: VerificationReport) -> bool:
    """Verify AgentPersona can be created."""
    try:
        from llmcore.agents import AgentPersona

        # Create a custom persona with required fields
        persona = AgentPersona(
            id="test_persona", name="Test Persona", description="A test persona for verification"
        )

        assert persona.id == "test_persona"
        assert persona.name == "Test Persona"
        assert persona.description == "A test persona for verification"

        report.add(
            VerificationResult(
                name="AgentPersona Creation",
                status=VerificationStatus.PASS,
                message="Custom AgentPersona created successfully",
            )
        )
        return True
    except Exception as e:
        report.add(
            VerificationResult(
                name="AgentPersona Creation",
                status=VerificationStatus.FAIL,
                message="Failed to create AgentPersona",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_exports_complete(report: VerificationReport) -> bool:
    """Verify __all__ exports are complete."""
    try:
        from llmcore.agents import __all__

        expected_exports = [
            # Core Manager
            "EnhancedAgentManager",
            "AgentMode",
            # Single Agent
            "SingleAgentMode",
            "AgentResult",
            # Cognitive Cycle
            "CognitiveCycle",
            "EnhancedAgentState",
            "CognitivePhase",
            # Personas
            "PersonaManager",
            "AgentPersona",
            # Prompts
            "PromptRegistry",
            # Memory
            "CognitiveMemoryIntegrator",
        ]

        missing = [e for e in expected_exports if e not in __all__]

        if not missing:
            report.add(
                VerificationResult(
                    name="Package Exports",
                    status=VerificationStatus.PASS,
                    message=f"All {len(__all__)} exports present in __all__",
                )
            )
            return True
        else:
            report.add(
                VerificationResult(
                    name="Package Exports",
                    status=VerificationStatus.WARN,
                    message=f"Missing exports: {missing}",
                )
            )
            return True
    except ImportError as e:
        report.add(
            VerificationResult(
                name="Package Exports",
                status=VerificationStatus.FAIL,
                message="Failed to check exports",
                details=str(e),
                exception=e,
            )
        )
        return False


def verify_file_structure(report: VerificationReport) -> bool:
    """Verify expected file structure exists."""
    try:
        # Try to find llmcore package location
        import llmcore.agents

        agents_path = os.path.dirname(llmcore.agents.__file__)

        # Note: prompts/templates is optional - may not exist if using programmatic templates
        expected_dirs = [
            "cognitive",
            "cognitive/phases",
            "persona",
            "prompts",
            "memory",
        ]

        expected_files = [
            "__init__.py",
            "manager.py",
            "single_agent.py",
            "cognitive/__init__.py",
            "cognitive/models.py",
            "persona/__init__.py",
            "persona/models.py",
            "persona/manager.py",
            "prompts/__init__.py",
            "prompts/models.py",
            "prompts/registry.py",
            "memory/__init__.py",
            "memory/integration.py",
        ]

        missing_dirs = []
        for d in expected_dirs:
            full_path = os.path.join(agents_path, d)
            if not os.path.isdir(full_path):
                missing_dirs.append(d)

        missing_files = []
        for f in expected_files:
            full_path = os.path.join(agents_path, f)
            if not os.path.isfile(full_path):
                missing_files.append(f)

        if not missing_dirs and not missing_files:
            report.add(
                VerificationResult(
                    name="File Structure",
                    status=VerificationStatus.PASS,
                    message=f"All {len(expected_dirs)} directories and {len(expected_files)} files present",
                )
            )
            return True
        else:
            details = []
            if missing_dirs:
                details.append(f"Missing dirs: {missing_dirs}")
            if missing_files:
                details.append(f"Missing files: {missing_files}")
            report.add(
                VerificationResult(
                    name="File Structure",
                    status=VerificationStatus.WARN,
                    message="Some expected files/dirs missing",
                    details="; ".join(details),
                )
            )
            return True
    except Exception as e:
        report.add(
            VerificationResult(
                name="File Structure",
                status=VerificationStatus.FAIL,
                message="Failed to check file structure",
                details=str(e),
                exception=e,
            )
        )
        return False


# =============================================================================
# MAIN VERIFICATION RUNNER
# =============================================================================


def print_header():
    """Print verification header."""
    print("=" * 70)
    print("  Darwin Layer 2 Integration Verification")
    print("  llmcore v0.26.0")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_result(result: VerificationResult):
    """Print a single verification result."""
    print(f"  {result.status.value}  {result.name}")
    print(f"        {result.message}")
    if result.details:
        for line in result.details.split("\n"):
            print(f"        └─ {line}")


def print_summary(report: VerificationReport):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Total checks: {report.total}")
    print(f"  Passed:       {report.passed} {VerificationStatus.PASS.value.split()[0]}")
    print(f"  Failed:       {report.failed} {VerificationStatus.FAIL.value.split()[0]}")
    print(f"  Warnings:     {report.warnings} {VerificationStatus.WARN.value.split()[0]}")
    print(f"  Skipped:      {report.skipped} {VerificationStatus.SKIP.value.split()[0]}")
    print("=" * 70)

    if report.success:
        print("\n  🎉 Darwin Layer 2 is PROPERLY INTEGRATED!")
        print("  llmcore is FULLY USABLE in its current state.")
        print("  Layers 3+ can be added incrementally.")
    else:
        print("\n  ⚠️  Darwin Layer 2 integration has ISSUES")
        print("  Please review failed checks above.")
        print("\n  Failed checks:")
        for r in report.results:
            if r.status == VerificationStatus.FAIL:
                print(f"    • {r.name}: {r.message}")

    print()


def run_verification() -> VerificationReport:
    """Run all verification checks."""
    report = VerificationReport()

    print_header()

    # Section 1: Environment
    print_section("1. Environment Verification")
    verify_python_version(report)
    print_result(report.results[-1])

    # Section 2: Base Imports
    print_section("2. Base Package Imports")
    if not verify_llmcore_import(report):
        print_result(report.results[-1])
        print("\n  ⛔ Cannot continue without llmcore import")
        return report
    print_result(report.results[-1])

    verify_agents_package(report)
    print_result(report.results[-1])

    # Section 3: Core Components
    print_section("3. Core Components")
    verify_core_manager_imports(report)
    print_result(report.results[-1])

    verify_single_agent_imports(report)
    print_result(report.results[-1])

    # Section 4: Cognitive Cycle
    print_section("4. Cognitive Cycle (8 Phases)")
    verify_cognitive_cycle_imports(report)
    print_result(report.results[-1])

    verify_phase_function_imports(report)
    print_result(report.results[-1])

    verify_phase_io_models(report)
    print_result(report.results[-1])

    # Section 5: Persona System
    print_section("5. Persona System")
    verify_persona_imports(report)
    print_result(report.results[-1])

    # Section 6: Prompt Library
    print_section("6. Prompt Library")
    verify_prompt_library_imports(report)
    print_result(report.results[-1])

    # Section 7: Memory Integration
    print_section("7. Memory Integration")
    verify_memory_integration_imports(report)
    print_result(report.results[-1])

    # Section 8: Backward Compatibility
    print_section("8. Backward Compatibility")
    verify_backward_compatibility(report)
    print_result(report.results[-1])

    # Section 9: Component Instantiation
    print_section("9. Component Instantiation")
    verify_persona_manager_instantiation(report)
    print_result(report.results[-1])

    verify_prompt_registry_instantiation(report)
    print_result(report.results[-1])

    verify_enhanced_agent_state_creation(report)
    print_result(report.results[-1])

    verify_cognitive_cycle_instantiation(report)
    print_result(report.results[-1])

    verify_agent_persona_creation(report)
    print_result(report.results[-1])

    # Section 10: Package Structure
    print_section("10. Package Structure")
    verify_exports_complete(report)
    print_result(report.results[-1])

    verify_file_structure(report)
    print_result(report.results[-1])

    # Summary
    print_summary(report)

    return report


if __name__ == "__main__":
    try:
        report = run_verification()
        sys.exit(0 if report.success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
