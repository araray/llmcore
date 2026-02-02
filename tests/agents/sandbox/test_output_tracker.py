# tests/sandbox/test_output_tracker.py
"""
Unit tests for OutputTracker.

These tests verify output tracking, run management, and
persistence functionality.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.agents.sandbox.base import ExecutionResult

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.output_tracker import ExecutionLog, OutputTracker, RunMetadata

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def output_tracker(temp_dir):
    """Create an OutputTracker with temp directory."""
    return OutputTracker(
        base_path=temp_dir,
        max_log_entries=1000,
        log_input_preview_length=100,
        log_output_preview_length=200
    )


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox provider."""
    sandbox = MagicMock()
    config = MagicMock()
    config.sandbox_id = "test-sandbox-123"
    sandbox.get_config.return_value = config
    sandbox.get_info.return_value = {"provider": "docker"}
    return sandbox


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestOutputTrackerInit:
    """Tests for OutputTracker initialization."""

    def test_init_creates_directory(self, temp_dir):
        """Test initialization creates base directory."""
        tracker = OutputTracker(base_path=temp_dir)

        assert Path(temp_dir).exists()

    def test_init_with_home_expansion(self):
        """Test initialization with home directory expansion."""
        tracker = OutputTracker(base_path="~/.llmcore/test_outputs")

        expected = Path.home() / ".llmcore" / "test_outputs"
        assert tracker._base_path == expected


# =============================================================================
# RUN MANAGEMENT TESTS
# =============================================================================

class TestRunManagement:
    """Tests for run management."""

    @pytest.mark.asyncio
    async def test_create_run(self, output_tracker):
        """Test creating a new run."""
        run_id = await output_tracker.create_run(
            sandbox_id="sandbox-abc123",
            sandbox_type="docker",
            access_level="restricted",
            task_description="Test task",
            metadata={"custom": "value"}
        )

        assert run_id == "sandbox-abc123"

        # Verify directory structure
        run_path = output_tracker._get_run_path(run_id)
        assert run_path.exists()
        assert (run_path / "outputs").exists()
        assert (run_path / "logs").exists()
        assert (run_path / "state").exists()
        assert (run_path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_create_run_metadata(self, output_tracker):
        """Test run metadata is correctly saved."""
        run_id = await output_tracker.create_run(
            sandbox_id="test-run",
            sandbox_type="vm",
            access_level="full",
            task_description="Integration test",
            metadata={"env": "production"}
        )

        metadata = await output_tracker.get_run_metadata(run_id)

        assert metadata["run_id"] == "test-run"
        assert metadata["sandbox_type"] == "vm"
        assert metadata["access_level"] == "full"
        assert metadata["task_description"] == "Integration test"
        assert metadata["custom_metadata"]["env"] == "production"

    @pytest.mark.asyncio
    async def test_get_run_metadata_nonexistent(self, output_tracker):
        """Test getting metadata for nonexistent run."""
        metadata = await output_tracker.get_run_metadata("nonexistent")

        assert metadata is None


# =============================================================================
# EXECUTION LOGGING TESTS
# =============================================================================

class TestExecutionLogging:
    """Tests for execution logging."""

    @pytest.mark.asyncio
    async def test_log_execution(self, output_tracker):
        """Test logging an execution."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        result = ExecutionResult(
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            execution_time_seconds=0.5
        )

        await output_tracker.log_execution(
            run_id=run_id,
            tool_name="execute_shell",
            input_data="echo 'Hello, World!'",
            result=result
        )

        # Verify log was added
        logs = await output_tracker.get_execution_logs(run_id)
        assert len(logs) == 1
        assert logs[0]["tool_name"] == "execute_shell"
        assert logs[0]["success"] is True

    @pytest.mark.asyncio
    async def test_log_execution_truncation(self, output_tracker):
        """Test input/output truncation in logs."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        long_input = "x" * 500
        long_output = "y" * 500

        result = ExecutionResult(
            exit_code=0,
            stdout=long_output,
            stderr=""
        )

        await output_tracker.log_execution(
            run_id=run_id,
            tool_name="test",
            input_data=long_input,
            result=result
        )

        logs = await output_tracker.get_execution_logs(run_id)
        assert len(logs[0]["input_summary"]) <= output_tracker._input_preview_length + 10
        assert len(logs[0]["output_preview"]) <= output_tracker._output_preview_length + 10

    @pytest.mark.asyncio
    async def test_log_execution_max_entries(self, temp_dir):
        """Test max log entries limit."""
        tracker = OutputTracker(base_path=temp_dir, max_log_entries=5)
        run_id = await tracker.create_run(sandbox_id="test-run")

        # Add more than max entries
        for i in range(10):
            result = ExecutionResult(exit_code=0, stdout=f"output {i}")
            await tracker.log_execution(run_id, "test", f"input {i}", result)

        logs = await tracker.get_execution_logs(run_id)
        assert len(logs) <= 5


# =============================================================================
# FILE TRACKING TESTS
# =============================================================================

class TestFileTracking:
    """Tests for file tracking."""

    @pytest.mark.asyncio
    async def test_track_file(self, output_tracker):
        """Test tracking a file."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        await output_tracker.track_file(
            run_id=run_id,
            file_path="/workspace/output.py",
            size_bytes=1024,
            description="Generated code"
        )

        files = await output_tracker.get_tracked_files(run_id)
        assert len(files) == 1
        assert files[0]["path"] == "/workspace/output.py"
        assert files[0]["size_bytes"] == 1024

    @pytest.mark.asyncio
    async def test_track_multiple_files(self, output_tracker):
        """Test tracking multiple files."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        await output_tracker.track_file(run_id, "/file1.py", 100)
        await output_tracker.track_file(run_id, "/file2.py", 200)
        await output_tracker.track_file(run_id, "/file3.py", 300)

        files = await output_tracker.get_tracked_files(run_id)
        assert len(files) == 3


# =============================================================================
# STATE PERSISTENCE TESTS
# =============================================================================

class TestStatePersistence:
    """Tests for state persistence."""

    @pytest.mark.asyncio
    async def test_save_final_state(self, output_tracker):
        """Test saving final state."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        state = {
            "iteration": 5,
            "plan": ["step1", "step2", "step3"],
            "completed": True
        }

        await output_tracker.save_final_state(run_id, state)

        # Verify state file exists
        state_path = output_tracker._get_state_path(run_id) / "final_state.json"
        assert state_path.exists()

        with open(state_path) as f:
            saved_state = json.load(f)

        assert saved_state["iteration"] == 5
        assert saved_state["completed"] is True


# =============================================================================
# RUN FINALIZATION TESTS
# =============================================================================

class TestRunFinalization:
    """Tests for run finalization."""

    @pytest.mark.asyncio
    async def test_finalize_run_success(self, output_tracker):
        """Test finalizing a successful run."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        await output_tracker.finalize_run(
            run_id=run_id,
            success=True
        )

        metadata = await output_tracker.get_run_metadata(run_id)
        assert metadata["status"] == "completed"
        assert metadata["success"] is True
        assert metadata["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_finalize_run_failure(self, output_tracker):
        """Test finalizing a failed run."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        await output_tracker.finalize_run(
            run_id=run_id,
            success=False,
            error_message="Task failed due to timeout"
        )

        metadata = await output_tracker.get_run_metadata(run_id)
        assert metadata["status"] == "failed"
        assert metadata["success"] is False
        assert metadata["error_message"] == "Task failed due to timeout"

    @pytest.mark.asyncio
    async def test_finalize_run_with_sandbox(self, output_tracker, mock_sandbox):
        """Test finalizing run with sandbox state preservation."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        # Mock ephemeral manager
        with patch('llmcore.agents.sandbox.ephemeral.EphemeralResourceManager') as MockEphemeral:
            mock_ephemeral = MagicMock()
            mock_ephemeral.export_state = AsyncMock(return_value={
                "state": {"key": "value"},
                "logs": [],
                "files": []
            })
            MockEphemeral.return_value = mock_ephemeral

            await output_tracker.finalize_run(
                run_id=run_id,
                sandbox=mock_sandbox,
                success=True,
                preserve_state=True
            )

        # Verify state was preserved
        state_path = output_tracker._get_state_path(run_id) / "final_state.json"
        assert state_path.exists()


# =============================================================================
# RUN LISTING TESTS
# =============================================================================

class TestRunListing:
    """Tests for run listing."""

    @pytest.mark.asyncio
    async def test_list_runs(self, output_tracker):
        """Test listing runs."""
        # Create multiple runs
        for i in range(5):
            await output_tracker.create_run(sandbox_id=f"run-{i}")

        runs = await output_tracker.list_runs()

        assert len(runs) == 5

    @pytest.mark.asyncio
    async def test_list_runs_with_limit(self, output_tracker):
        """Test listing runs with limit."""
        for i in range(10):
            await output_tracker.create_run(sandbox_id=f"run-{i}")

        runs = await output_tracker.list_runs(limit=5)

        assert len(runs) == 5

    @pytest.mark.asyncio
    async def test_list_runs_with_status_filter(self, output_tracker):
        """Test listing runs with status filter."""
        run1 = await output_tracker.create_run(sandbox_id="run-1")
        run2 = await output_tracker.create_run(sandbox_id="run-2")

        await output_tracker.finalize_run(run1, success=True)
        await output_tracker.finalize_run(run2, success=False)

        completed_runs = await output_tracker.list_runs(status_filter="completed")
        failed_runs = await output_tracker.list_runs(status_filter="failed")

        assert len(completed_runs) == 1
        assert len(failed_runs) == 1


# =============================================================================
# RUN DELETION TESTS
# =============================================================================

class TestRunDeletion:
    """Tests for run deletion."""

    @pytest.mark.asyncio
    async def test_delete_run(self, output_tracker):
        """Test deleting a run."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        result = await output_tracker.delete_run(run_id)

        assert result is True
        assert not output_tracker._get_run_path(run_id).exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_run(self, output_tracker):
        """Test deleting nonexistent run."""
        result = await output_tracker.delete_run("nonexistent")

        assert result is False


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanup:
    """Tests for old run cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_old_runs(self, output_tracker):
        """Test cleaning up old runs."""
        # Create runs with different ages
        for i in range(10):
            run_id = await output_tracker.create_run(sandbox_id=f"run-{i}")

        # Should keep at least keep_min_runs
        deleted = await output_tracker.cleanup_old_runs(
            max_age_days=0,  # All runs are "old"
            keep_min_runs=5
        )

        remaining = await output_tracker.list_runs()
        assert len(remaining) >= 5

    @pytest.mark.asyncio
    async def test_cleanup_preserves_recent_runs(self, output_tracker):
        """Test cleanup preserves recent runs."""
        for i in range(5):
            await output_tracker.create_run(sandbox_id=f"run-{i}")

        # All runs are recent, none should be deleted
        deleted = await output_tracker.cleanup_old_runs(
            max_age_days=30,
            keep_min_runs=3
        )

        remaining = await output_tracker.list_runs()
        assert len(remaining) == 5


# =============================================================================
# OUTPUT PATH TESTS
# =============================================================================

class TestOutputPaths:
    """Tests for output path handling."""

    @pytest.mark.asyncio
    async def test_get_run_outputs_path(self, output_tracker):
        """Test getting run outputs path."""
        run_id = await output_tracker.create_run(sandbox_id="test-run")

        path = await output_tracker.get_run_outputs_path(run_id)

        assert path is not None
        assert path.exists()
        assert str(path).endswith("outputs")

    @pytest.mark.asyncio
    async def test_get_run_outputs_path_nonexistent(self, output_tracker):
        """Test getting outputs path for nonexistent run."""
        path = await output_tracker.get_run_outputs_path("nonexistent")

        assert path is None


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_execution_log_to_dict(self):
        """Test ExecutionLog serialization."""
        log = ExecutionLog(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            tool_name="execute_shell",
            input_summary="echo test",
            exit_code=0,
            execution_time=0.5,
            success=True,
            output_preview="test"
        )

        d = log.to_dict()

        assert d["tool_name"] == "execute_shell"
        assert d["exit_code"] == 0
        assert d["success"] is True

    def test_run_metadata_to_dict(self):
        """Test RunMetadata serialization."""
        metadata = RunMetadata(
            run_id="test-123",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            sandbox_type="docker",
            access_level="restricted",
            task_description="Test task"
        )

        d = metadata.to_dict()

        assert d["run_id"] == "test-123"
        assert d["sandbox_type"] == "docker"
        assert d["task_description"] == "Test task"
