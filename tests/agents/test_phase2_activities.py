# tests/agents/test_phase2_activities.py
"""
Phase 2 Activity System Tests.

Tests the Activity System components:
- Activity schema and data models
- Activity request parser (XML parsing)
- Activity registry
- Activity executor
- Activity loop integration

Run with:
    PYTHONPATH=src:$PYTHONPATH pytest tests/agents/test_phase2_activities.py -v
"""

import os
import tempfile
from pathlib import Path

import pytest

# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestActivitySchema:
    """Tests for activity schema data models."""

    def test_activity_category_enum(self):
        """Test ActivityCategory enum values."""
        from llmcore.agents.activities.schema import ActivityCategory

        assert ActivityCategory.FILE_OPERATIONS.value == "file_operations"
        assert ActivityCategory.CODE_EXECUTION.value == "code_execution"
        assert ActivityCategory.CONTROL.value == "control"

    def test_risk_level_enum(self):
        """Test RiskLevel enum values and ordering."""
        from llmcore.agents.activities.schema import RiskLevel

        levels = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 5
        assert levels[0].value == "none"
        assert levels[-1].value == "critical"

    def test_activity_definition_creation(self):
        """Test creating an ActivityDefinition."""
        from llmcore.agents.activities.schema import (
            ActivityCategory,
            ActivityDefinition,
            ParameterSchema,
            ParameterType,
            RiskLevel,
        )

        defn = ActivityDefinition(
            name="test_activity",
            category=ActivityCategory.FILE_OPERATIONS,
            description="A test activity",
            parameters=[
                ParameterSchema(
                    name="path",
                    type=ParameterType.STRING,
                    description="File path",
                    required=True,
                ),
                ParameterSchema(
                    name="encoding",
                    type=ParameterType.STRING,
                    description="Encoding",
                    default="utf-8",
                ),
            ],
            risk_level=RiskLevel.LOW,
        )

        assert defn.name == "test_activity"
        assert defn.category == ActivityCategory.FILE_OPERATIONS
        assert len(defn.parameters) == 2
        assert defn.get_required_parameters()[0].name == "path"

    def test_activity_definition_to_prompt_format(self):
        """Test formatting activity for prompts."""
        from llmcore.agents.activities.schema import (
            ActivityCategory,
            ActivityDefinition,
            ParameterSchema,
            ParameterType,
            RiskLevel,
        )

        defn = ActivityDefinition(
            name="file_read",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Read a file",
            parameters=[
                ParameterSchema(name="path", type=ParameterType.STRING, required=True),
            ],
            risk_level=RiskLevel.LOW,
        )

        prompt = defn.to_prompt_format()
        assert "file_read" in prompt
        assert "Read a file" in prompt
        assert "path" in prompt

    def test_activity_request_creation(self):
        """Test creating an ActivityRequest."""
        from llmcore.agents.activities.schema import ActivityRequest, ExecutionTarget

        request = ActivityRequest(
            activity="file_read",
            parameters={"path": "/etc/hosts"},
            target=ExecutionTarget.DOCKER,
            reason="Check network config",
        )

        assert request.activity == "file_read"
        assert request.parameters["path"] == "/etc/hosts"
        assert request.target == ExecutionTarget.DOCKER

    def test_activity_result_to_observation(self):
        """Test formatting result as observation."""
        from llmcore.agents.activities.schema import ActivityResult, ActivityStatus

        result = ActivityResult(
            activity="file_read",
            status=ActivityStatus.SUCCESS,
            output="127.0.0.1 localhost",
            duration_ms=50,
        )

        obs = result.to_observation()
        assert "file_read" in obs
        assert "SUCCESS" in obs.upper() or "success" in obs
        assert "127.0.0.1" in obs


# =============================================================================
# PARSER TESTS
# =============================================================================


class TestActivityParser:
    """Tests for activity request parser."""

    @pytest.fixture
    def parser(self):
        from llmcore.agents.activities.parser import ActivityRequestParser

        return ActivityRequestParser()

    def test_parse_single_activity(self, parser):
        """Test parsing a single activity request."""
        text = """
        I'll read the file.
        <activity_request>
            <activity>file_read</activity>
            <parameters>
                <path>/etc/hosts</path>
            </parameters>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].activity == "file_read"
        assert result.requests[0].parameters["path"] == "/etc/hosts"
        assert "I'll read the file" in result.remaining_text

    def test_parse_multiple_activities(self, parser):
        """Test parsing multiple activity requests."""
        text = """
        <activity_request>
            <activity>file_read</activity>
            <parameters><path>/tmp/a.txt</path></parameters>
        </activity_request>
        <activity_request>
            <activity>file_write</activity>
            <parameters>
                <path>/tmp/b.txt</path>
                <content>Hello</content>
            </parameters>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 2
        assert result.requests[0].activity == "file_read"
        assert result.requests[1].activity == "file_write"

    def test_parse_with_reason(self, parser):
        """Test parsing activity with reason."""
        text = """
        <activity_request>
            <activity>file_search</activity>
            <parameters>
                <path>/var/log</path>
                <pattern>*.log</pattern>
            </parameters>
            <reason>Find application logs for debugging</reason>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].reason == "Find application logs for debugging"

    def test_parse_no_activities(self, parser):
        """Test parsing text with no activities."""
        text = "Just some regular text without any activities."

        result = parser.parse(text)

        assert len(result.requests) == 0
        assert result.remaining_text == text

    def test_parse_malformed_xml_fallback(self, parser):
        """Test regex fallback for malformed XML."""
        text = """
        <activity_request>
        <activity>python_exec</activity>
        <parameters>
        <code>print("hello")</code>
        </parameters>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].activity == "python_exec"

    def test_is_final_answer(self, parser):
        """Test final answer detection."""
        assert parser.is_final_answer("<final_answer>The result is 42.</final_answer>")
        assert parser.is_final_answer("FINAL ANSWER: The result is 42.")
        assert not parser.is_final_answer("This is just normal text.")

    def test_extract_final_answer(self, parser):
        """Test extracting final answer content."""
        text = "<final_answer>The answer is 42.</final_answer>"
        answer = parser.extract_final_answer(text)
        assert answer == "The answer is 42."

    def test_parse_with_target(self, parser):
        """Test parsing activity with execution target."""
        from llmcore.agents.activities.schema import ExecutionTarget

        text = """
        <activity_request>
            <activity>bash_exec</activity>
            <parameters><command>ls -la</command></parameters>
            <target>docker:sandbox</target>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].target == ExecutionTarget.DOCKER

    def test_parse_boolean_parameter(self, parser):
        """Test parsing boolean parameters."""
        text = """
        <activity_request>
            <activity>file_search</activity>
            <parameters>
                <path>/tmp</path>
                <recursive>true</recursive>
            </parameters>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].parameters["recursive"] is True

    def test_parse_integer_parameter(self, parser):
        """Test parsing integer parameters."""
        text = """
        <activity_request>
            <activity>file_search</activity>
            <parameters>
                <path>/tmp</path>
                <max_results>50</max_results>
            </parameters>
        </activity_request>
        """

        result = parser.parse(text)

        assert len(result.requests) == 1
        assert result.requests[0].parameters["max_results"] == 50


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestActivityRegistry:
    """Tests for activity registry."""

    @pytest.fixture
    def registry(self):
        from llmcore.agents.activities.registry import ActivityRegistry

        return ActivityRegistry(auto_register_builtins=True)

    def test_builtin_activities_registered(self, registry):
        """Test that built-in activities are registered."""
        assert len(registry) > 0
        assert "file_read" in registry
        assert "file_write" in registry
        assert "python_exec" in registry
        assert "final_answer" in registry

    def test_get_activity(self, registry):
        """Test getting an activity by name."""
        activity = registry.get("file_read")

        assert activity is not None
        assert activity.definition.name == "file_read"
        assert activity.source == "builtin"

    def test_get_nonexistent_activity(self, registry):
        """Test getting a nonexistent activity."""
        activity = registry.get("nonexistent_activity")
        assert activity is None

    def test_filter_by_category(self, registry):
        """Test filtering activities by category."""
        from llmcore.agents.activities.schema import ActivityCategory

        file_ops = registry.filter_by_category(ActivityCategory.FILE_OPERATIONS)

        assert len(file_ops) > 0
        assert all(a.category == ActivityCategory.FILE_OPERATIONS for a in file_ops)

    def test_filter_by_risk_level(self, registry):
        """Test filtering activities by risk level."""
        from llmcore.agents.activities.schema import RiskLevel

        safe_activities = registry.filter_by_risk_level(RiskLevel.LOW)

        assert len(safe_activities) > 0
        # All should be NONE or LOW risk
        for activity in safe_activities:
            assert activity.definition.risk_level in [RiskLevel.NONE, RiskLevel.LOW]

    def test_format_for_prompt(self, registry):
        """Test formatting activities for prompt."""
        from llmcore.agents.activities.schema import ActivityCategory

        prompt = registry.format_for_prompt(
            categories=[ActivityCategory.FILE_OPERATIONS]
        )

        assert "file_read" in prompt
        assert "file_write" in prompt
        assert "activity_request" in prompt.lower()

    def test_register_custom_activity(self, registry):
        """Test registering a custom activity."""
        from llmcore.agents.activities.schema import (
            ActivityCategory,
            ActivityDefinition,
            RiskLevel,
        )

        custom = ActivityDefinition(
            name="my_custom_activity",
            category=ActivityCategory.CUSTOM,
            description="A custom activity",
            risk_level=RiskLevel.LOW,
        )

        registry.register(custom, source="user")

        assert "my_custom_activity" in registry
        assert registry.get("my_custom_activity").source == "user"

    def test_unregister_activity(self, registry):
        """Test unregistering an activity."""
        from llmcore.agents.activities.schema import (
            ActivityCategory,
            ActivityDefinition,
            RiskLevel,
        )

        custom = ActivityDefinition(
            name="to_remove",
            category=ActivityCategory.CUSTOM,
            description="Will be removed",
            risk_level=RiskLevel.NONE,
        )

        registry.register(custom)
        assert "to_remove" in registry

        result = registry.unregister("to_remove")
        assert result is True
        assert "to_remove" not in registry

    def test_enable_disable_activity(self, registry):
        """Test enabling/disabling activities."""
        assert registry.is_enabled("file_read")

        registry.set_enabled("file_read", False)
        assert not registry.is_enabled("file_read")

        registry.set_enabled("file_read", True)
        assert registry.is_enabled("file_read")


# =============================================================================
# EXECUTOR TESTS
# =============================================================================


class TestActivityExecutor:
    """Tests for activity executor."""

    @pytest.fixture
    def executor(self):
        from llmcore.agents.activities.executor import ActivityExecutor

        return ActivityExecutor()

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, executor):
        """Test validating a valid request."""
        from llmcore.agents.activities.schema import ActivityRequest

        request = ActivityRequest(
            activity="file_read",
            parameters={"path": "/etc/hosts"},
        )

        result = executor.validator.validate(request)

        assert result.valid
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_unknown_activity(self, executor):
        """Test validating unknown activity."""
        from llmcore.agents.activities.schema import ActivityRequest

        request = ActivityRequest(
            activity="unknown_activity",
            parameters={},
        )

        result = executor.validator.validate(request)

        assert not result.valid
        assert any("Unknown activity" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_missing_required_param(self, executor):
        """Test validating request missing required parameter."""
        from llmcore.agents.activities.schema import ActivityRequest

        request = ActivityRequest(
            activity="file_read",
            parameters={},  # Missing required 'path'
        )

        result = executor.validator.validate(request)

        assert not result.valid
        assert any("Missing required parameter" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_execute_file_read(self, executor):
        """Test executing file_read activity."""
        from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            request = ActivityRequest(
                activity="file_read",
                parameters={"path": temp_path},
            )

            result = await executor.execute(request)

            assert result.status == ActivityStatus.SUCCESS
            assert "test content" in result.output
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_execute_file_write(self, executor):
        """Test executing file_write activity."""
        from llmcore.agents.activities.executor import HITLApprover
        from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus, RiskLevel

        # Allow medium risk for this test
        executor.hitl_approver = HITLApprover(risk_threshold=RiskLevel.HIGH)

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "test_output.txt")

        try:
            request = ActivityRequest(
                activity="file_write",
                parameters={
                    "path": temp_path,
                    "content": "Hello World",
                },
            )

            result = await executor.execute(request)

            assert result.status == ActivityStatus.SUCCESS
            assert Path(temp_path).exists()
            assert Path(temp_path).read_text() == "Hello World"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_execute_json_query(self, executor):
        """Test executing json_query activity."""
        from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus

        request = ActivityRequest(
            activity="json_query",
            parameters={
                "data": '{"users": [{"name": "Alice"}, {"name": "Bob"}]}',
                "query": "users.0.name",
            },
        )

        result = await executor.execute(request)

        assert result.status == ActivityStatus.SUCCESS
        assert "Alice" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, executor):
        """Test that execution respects timeout."""
        from llmcore.agents.activities.executor import HITLApprover
        from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus, RiskLevel

        # Allow high risk for this test
        executor.hitl_approver = HITLApprover(risk_threshold=RiskLevel.CRITICAL)

        request = ActivityRequest(
            activity="python_exec",
            parameters={
                "code": "import time; time.sleep(10); print('done')",
                "timeout": 1,
            },
            timeout_seconds=1,
        )

        result = await executor.execute(request)

        assert result.status == ActivityStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_hitl_rejection(self, executor):
        """Test HITL rejection of risky activity."""
        from llmcore.agents.activities.executor import HITLApprover
        from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus, RiskLevel

        # Set low threshold so file_delete requires approval
        executor.hitl_approver = HITLApprover(risk_threshold=RiskLevel.LOW)

        request = ActivityRequest(
            activity="bash_exec",  # HIGH risk
            parameters={"command": "echo hello"},
        )

        # No approval callback = rejection
        result = await executor.execute(request, approval_callback=None)

        assert result.status == ActivityStatus.REJECTED


# =============================================================================
# LOOP TESTS
# =============================================================================


class TestActivityLoop:
    """Tests for activity loop."""

    @pytest.fixture
    def loop(self):
        from llmcore.agents.activities.executor import HITLApprover
        from llmcore.agents.activities.loop import ActivityLoop, ActivityLoopConfig
        from llmcore.agents.activities.schema import RiskLevel

        config = ActivityLoopConfig(max_per_iteration=5, max_total=20)
        loop = ActivityLoop(config=config)
        # Allow all for testing
        loop.executor.hitl_approver = HITLApprover(risk_threshold=RiskLevel.CRITICAL)
        return loop

    @pytest.mark.asyncio
    async def test_process_single_activity(self, loop):
        """Test processing single activity."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test data")
            temp_path = f.name

        try:
            llm_output = f"""
            <activity_request>
                <activity>file_read</activity>
                <parameters><path>{temp_path}</path></parameters>
            </activity_request>
            """

            result = await loop.process_output(llm_output)

            assert len(result.executions) == 1
            assert result.executions[0].success
            assert "test data" in result.observation
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_process_no_activities(self, loop):
        """Test processing text with no activities."""
        llm_output = "Just thinking about the problem..."

        result = await loop.process_output(llm_output)

        assert len(result.executions) == 0
        assert result.should_continue

    @pytest.mark.asyncio
    async def test_process_final_answer(self, loop):
        """Test processing final answer."""
        llm_output = "<final_answer>The answer is 42.</final_answer>"

        result = await loop.process_output(llm_output)

        assert result.is_final_answer
        assert not result.should_continue
        assert "42" in result.observation

    @pytest.mark.asyncio
    async def test_activity_limit(self, loop):
        """Test that activity limit is enforced."""
        loop.config.max_per_iteration = 2

        llm_output = """
        <activity_request>
            <activity>memory_store</activity>
            <parameters><key>k1</key><value>v1</value></parameters>
        </activity_request>
        <activity_request>
            <activity>memory_store</activity>
            <parameters><key>k2</key><value>v2</value></parameters>
        </activity_request>
        <activity_request>
            <activity>memory_store</activity>
            <parameters><key>k3</key><value>v3</value></parameters>
        </activity_request>
        """

        result = await loop.process_output(llm_output)

        # Should only execute 2 of 3
        assert len(result.executions) == 2

    @pytest.mark.asyncio
    async def test_session_tracking(self, loop):
        """Test session tracking."""
        loop.start_session()

        assert loop.total_executed == 0
        assert loop.remaining_budget == 20

        llm_output = """
        <activity_request>
            <activity>think_aloud</activity>
            <parameters><thought>Testing session tracking</thought></parameters>
        </activity_request>
        """

        await loop.process_output(llm_output)

        assert loop.total_executed == 1
        assert loop.remaining_budget == 19


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_imports(self):
        """Test that all Phase 2 components can be imported."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from LLM output to observation."""
        from llmcore.agents.activities import (
            ActivityLoop,
            ActivityLoopConfig,
        )
        from llmcore.agents.activities.executor import HITLApprover
        from llmcore.agents.activities.schema import RiskLevel

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello from the test file!")
            temp_path = f.name

        try:
            # Setup loop
            config = ActivityLoopConfig()
            loop = ActivityLoop(config=config)
            loop.executor.hitl_approver = HITLApprover(risk_threshold=RiskLevel.CRITICAL)

            # Simulate LLM output
            llm_output = f"""
            I'll read the test file to see its contents.

            <activity_request>
                <activity>file_read</activity>
                <parameters>
                    <path>{temp_path}</path>
                </parameters>
                <reason>Read test file contents</reason>
            </activity_request>
            """

            # Process
            result = await loop.process_output(llm_output)

            # Verify
            assert len(result.executions) == 1
            assert result.executions[0].success
            assert "Hello from the test file!" in result.observation
            assert result.should_continue
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
