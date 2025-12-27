#!/usr/bin/env python3
"""
Darwin Layer 2 Test Runner.

Runs all Darwin Layer 2 tests and generates a comprehensive report.

Usage:
    cd /path/to/llmcore
    source venv/bin/activate
    python run_darwin_tests.py [OPTIONS]

Options:
    --quick     Run quick tests only (no coverage)
    --coverage  Generate coverage report
    --verbose   Verbose output
    --report    Generate HTML report
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


# Darwin Layer 2 test paths
DARWIN_TEST_PATHS = [
    "tests/agents/cognitive/test_cognitive_system.py",
    "tests/agents/cognitive/test_cognitive_phases_advanced.py",
    "tests/agents/prompts/test_prompt_library.py",
    "tests/agents/test_persona_and_single_agent.py",
    "tests/agents/test_memory_and_manager.py",
]

# Fallback: test entire agents directory
FALLBACK_TEST_PATH = "tests/agents/"


def check_prerequisites():
    """Check that prerequisites are met."""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 11):
        errors.append(f"Python 3.11+ required (got {sys.version})")
    
    # Check pytest
    try:
        import pytest
    except ImportError:
        errors.append("pytest not installed (pip install pytest)")
    
    # Check llmcore
    try:
        import llmcore
    except ImportError:
        errors.append("llmcore not installed (pip install -e .)")
    
    return errors


def find_existing_tests():
    """Find which Darwin test files exist."""
    existing = []
    missing = []
    
    for path in DARWIN_TEST_PATHS:
        if os.path.exists(path):
            existing.append(path)
        else:
            missing.append(path)
    
    return existing, missing


def run_tests(test_paths, coverage=False, verbose=False):
    """Run pytest on specified test paths."""
    cmd = ["pytest"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend([
            "--cov=src/llmcore/agents",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov_darwin",
        ])
    
    cmd.extend(test_paths)
    
    print(f"\n🔬 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def generate_report(results: dict):
    """Generate a summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              Darwin Layer 2 Test Report                          ║
║              {timestamp}                         ║
╚══════════════════════════════════════════════════════════════════╝

SUMMARY
-------
  Tests found:    {results.get('tests_found', 0)}
  Tests missing:  {results.get('tests_missing', 0)}
  Exit code:      {results.get('exit_code', 'N/A')}
  Status:         {"✅ PASSED" if results.get('exit_code') == 0 else "❌ FAILED"}

TEST FILES
----------
  Found:
"""
    
    for path in results.get('existing_tests', []):
        report += f"    ✅ {path}\n"
    
    if results.get('missing_tests'):
        report += "\n  Missing:\n"
        for path in results.get('missing_tests', []):
            report += f"    ⚠️ {path}\n"
    
    report += f"""
RECOMMENDATIONS
---------------
"""
    
    if results.get('exit_code') == 0:
        report += """  • All tests passed! Darwin Layer 2 is properly integrated.
  • You can proceed with Layer 3 implementation.
  • Consider running with --coverage for detailed coverage report.
"""
    else:
        report += """  • Some tests failed. Review the output above.
  • Fix failing tests before proceeding.
  • Run with --verbose for more details.
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Darwin Layer 2 Test Runner")
    parser.add_argument("--quick", action="store_true", help="Quick run without coverage")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    args = parser.parse_args()
    
    print("═" * 60)
    print("  Darwin Layer 2 Test Runner")
    print("═" * 60)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    errors = check_prerequisites()
    if errors:
        print("\n❌ Prerequisites not met:")
        for err in errors:
            print(f"    • {err}")
        return 1
    print("   ✅ All prerequisites met")
    
    # Find existing tests
    print("\n🔍 Looking for Darwin Layer 2 tests...")
    existing, missing = find_existing_tests()
    
    if existing:
        print(f"   ✅ Found {len(existing)} test files")
        for path in existing:
            print(f"      • {path}")
    else:
        print(f"   ⚠️ No specific Darwin test files found")
        print(f"   📁 Falling back to: {FALLBACK_TEST_PATH}")
        existing = [FALLBACK_TEST_PATH]
    
    if missing:
        print(f"\n   ⚠️ Missing {len(missing)} test files (may need to be created):")
        for path in missing:
            print(f"      • {path}")
    
    # Run tests
    print("\n" + "─" * 60)
    exit_code = run_tests(
        existing,
        coverage=args.coverage and not args.quick,
        verbose=args.verbose
    )
    print("─" * 60)
    
    # Generate report if requested
    results = {
        'tests_found': len(existing),
        'tests_missing': len(missing),
        'existing_tests': existing,
        'missing_tests': missing,
        'exit_code': exit_code,
    }
    
    if args.report:
        report = generate_report(results)
        print(report)
        
        # Save report to file
        report_file = f"darwin_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")
    
    # Summary
    print("\n" + "═" * 60)
    if exit_code == 0:
        print("  ✅ ALL TESTS PASSED")
        print("  Darwin Layer 2 is properly integrated!")
        print("  llmcore is ready for use.")
    else:
        print("  ❌ SOME TESTS FAILED")
        print("  Review output above and fix failing tests.")
    print("═" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
