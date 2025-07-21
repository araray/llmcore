#!/usr/bin/env python3
"""
Test runner script for llmcore API server tests.

This script provides various testing options including unit tests,
functional tests, performance tests, and coverage reporting.
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run llmcore API server tests")
    parser.add_argument(
        "--test-type",
        choices=["unit", "functional", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Run tests with specific pytest markers (e.g., 'not slow')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test-results",
        help="Directory for test outputs and reports"
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    success = True

    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])

    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])

    # Add markers
    if args.markers:
        pytest_cmd.extend(["-m", args.markers])

    # Configure coverage
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src/llmcore/api_server",
            f"--cov-report=html:{output_dir}/coverage-html",
            f"--cov-report=xml:{output_dir}/coverage.xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])

    # Add JUnit XML output for CI
    pytest_cmd.extend([f"--junit-xml={output_dir}/junit.xml"])

    # Run different test types
    if args.test_type in ["unit", "all"]:
        unit_cmd = pytest_cmd + [
            "tests/api_server/test_api_server.py",
            "-m", "unit or not (integration or slow)"
        ]
        if not run_command(unit_cmd, "Unit Tests"):
            success = False

    if args.test_type in ["functional", "all"]:
        functional_cmd = pytest_cmd + [
            "tests/api_server/test_functional.py",
            "-m", "integration or api"
        ]
        if not run_command(functional_cmd, "Functional Tests"):
            success = False

    if args.test_type in ["performance", "all"]:
        performance_cmd = pytest_cmd + [
            "tests/api_server/test_functional.py",
            "-m", "slow"
        ]
        if not run_command(performance_cmd, "Performance Tests"):
            success = False

    # Run linting and code quality checks
    print(f"\n{'='*60}")
    print("Running Code Quality Checks")
    print(f"{'='*60}")

    # Run ruff linting
    ruff_cmd = ["python", "-m", "ruff", "check", "src/llmcore/api_server/"]
    if not run_command(ruff_cmd, "Ruff Linting"):
        print("⚠️ Linting issues found (not blocking)")

    # Run type checking with mypy
    mypy_cmd = ["python", "-m", "mypy", "src/llmcore/api_server/", "--ignore-missing-imports"]
    if not run_command(mypy_cmd, "MyPy Type Checking"):
        print("⚠️ Type checking issues found (not blocking)")

    # Generate test summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    if success:
        print("✅ All tests passed!")
        if args.coverage:
            print(f"📊 Coverage report available at: {output_dir}/coverage-html/index.html")
        print(f"📋 JUnit report available at: {output_dir}/junit.xml")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

    # Additional development helpers
    print(f"\n{'='*60}")
    print("Development Commands")
    print(f"{'='*60}")
    print("To run the API server:")
    print("  uvicorn llmcore.api_server.main:app --reload")
    print("\nTo run tests with specific markers:")
    print("  pytest tests/api_server/ -m 'not slow'")
    print("  pytest tests/api_server/ -m 'unit'")
    print("  pytest tests/api_server/ -m 'integration'")
    print("\nTo run a specific test:")
    print("  pytest tests/api_server/test_api_server.py::TestChatEndpoint::test_chat_non_streaming_success -v")


if __name__ == "__main__":
    main()
