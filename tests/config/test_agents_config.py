# tests/config/test_agents_config.py
"""
Tests for the agent configuration module.

Tests cover:
1. Default configuration values
2. Configuration override mechanisms
3. TOML loading
4. Environment variable overrides
5. Validation constraints
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add source directory to path to import directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.config.agents_config import (
    ActivitiesConfig,
    AgentsConfig,
    CapabilityCheckConfig,
    CircuitBreakerConfig,
    FastPathConfig,
    GoalsConfig,
    HITLConfig,
    RiskLevel,
    RoutingConfig,
    RoutingStrategy,
    TimeoutPolicy,
    load_agents_config,
)

# =============================================================================
# TEST: Default Values
# =============================================================================


class TestAgentsConfigDefaults:
    """Test that default configuration values are sensible."""

    def test_root_config_defaults(self):
        """Root config should have sensible defaults."""
        config = AgentsConfig()

        assert config.max_iterations == 10
        assert config.default_timeout == 600
        assert config.default_persona == "assistant"
        assert config.memory_enabled is True

    def test_goals_config_defaults(self):
        """Goals config should have sensible defaults."""
        config = GoalsConfig()

        assert config.classifier_enabled is True
        assert config.use_llm_fallback is False
        assert config.heuristic_confidence_threshold == 0.9
        assert config.trivial_max_iterations == 1
        assert config.simple_max_iterations == 5
        assert config.moderate_max_iterations == 10
        assert config.complex_max_iterations == 15

    def test_fast_path_config_defaults(self):
        """Fast-path config should have sensible defaults."""
        config = FastPathConfig()

        assert config.enabled is True
        assert config.cache_enabled is True
        assert config.cache_max_entries == 100
        assert config.cache_ttl_seconds == 3600
        assert config.templates_enabled is True
        assert config.max_response_time_ms == 5000
        assert config.temperature == 0.7
        assert config.max_tokens == 500
        assert config.fallback_on_timeout is True

    def test_circuit_breaker_config_defaults(self):
        """Circuit breaker config should have sensible defaults."""
        config = CircuitBreakerConfig()

        assert config.enabled is True
        assert config.max_iterations == 15
        assert config.max_same_errors == 3
        assert config.max_execution_time_seconds == 300
        assert config.max_total_cost == 1.0
        assert config.progress_stall_threshold == 5
        assert config.progress_stall_tolerance == 0.01

    def test_activities_config_defaults(self):
        """Activities config should have sensible defaults."""
        config = ActivitiesConfig()

        assert config.enabled is True
        assert config.fallback_to_native_tools is True
        assert config.max_per_iteration == 10
        assert config.max_total == 100
        assert config.default_timeout_seconds == 60
        assert config.total_timeout_seconds == 300
        assert config.stop_on_error is False
        assert config.parallel_execution is False
        assert config.max_observation_length == 4000
        assert config.include_reasoning is True

    def test_capability_check_config_defaults(self):
        """Capability check config should have sensible defaults."""
        config = CapabilityCheckConfig()

        assert config.enabled is True
        assert config.use_model_cards is True
        assert config.use_runtime_query is True
        assert config.strict_mode is True
        assert config.suggest_alternatives is True

    def test_hitl_config_defaults(self):
        """HITL config should have sensible defaults."""
        config = HITLConfig()

        assert config.enabled is True
        assert config.global_risk_threshold == RiskLevel.MEDIUM
        assert config.default_timeout_seconds == 300
        assert config.timeout_policy == TimeoutPolicy.REJECT
        assert "final_answer" in config.safe_tools
        assert "bash_exec" in config.high_risk_tools
        assert config.batch_similar_requests is True
        assert config.audit_logging_enabled is True

    def test_routing_config_defaults(self):
        """Routing config should have sensible defaults."""
        config = RoutingConfig()

        assert config.enabled is True
        assert config.strategy == RoutingStrategy.COST_OPTIMIZED
        assert config.fallback_enabled is True
        assert "gpt-4o-mini" in config.tiers.fast
        assert "gpt-4o" in config.tiers.balanced
        assert "gpt-4-turbo" in config.tiers.capable

    def test_nested_configs_in_root(self):
        """Root config should contain all nested configs."""
        config = AgentsConfig()

        assert isinstance(config.goals, GoalsConfig)
        assert isinstance(config.fast_path, FastPathConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.activities, ActivitiesConfig)
        assert isinstance(config.capability_check, CapabilityCheckConfig)
        assert isinstance(config.hitl, HITLConfig)
        assert isinstance(config.routing, RoutingConfig)


# =============================================================================
# TEST: Configuration Override
# =============================================================================


class TestAgentsConfigOverride:
    """Test configuration override mechanisms."""

    def test_override_root_values(self):
        """Should be able to override root-level values."""
        config = AgentsConfig(
            max_iterations=20,
            default_timeout=1200,
            default_persona="developer",
            memory_enabled=False,
        )

        assert config.max_iterations == 20
        assert config.default_timeout == 1200
        assert config.default_persona == "developer"
        assert config.memory_enabled is False

    def test_override_nested_config(self):
        """Should be able to override nested config values."""
        config = AgentsConfig(
            goals=GoalsConfig(
                classifier_enabled=False,
                trivial_max_iterations=2,
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_same_errors=5,
            ),
        )

        assert config.goals.classifier_enabled is False
        assert config.goals.trivial_max_iterations == 2
        assert config.circuit_breaker.max_same_errors == 5

    def test_partial_override_preserves_defaults(self):
        """Partial overrides should preserve other defaults."""
        config = AgentsConfig(
            goals=GoalsConfig(classifier_enabled=False)
        )

        # Overridden value
        assert config.goals.classifier_enabled is False

        # Preserved defaults
        assert config.goals.use_llm_fallback is False
        assert config.goals.heuristic_confidence_threshold == 0.9

        # Other configs unchanged
        assert config.fast_path.enabled is True
        assert config.circuit_breaker.enabled is True


# =============================================================================
# TEST: Validation
# =============================================================================


class TestAgentsConfigValidation:
    """Test configuration validation constraints."""

    def test_max_iterations_must_be_positive(self):
        """max_iterations must be >= 1."""
        with pytest.raises(ValueError):
            AgentsConfig(max_iterations=0)

        with pytest.raises(ValueError):
            AgentsConfig(max_iterations=-1)

    def test_max_iterations_upper_limit(self):
        """max_iterations must be <= 1000."""
        with pytest.raises(ValueError):
            AgentsConfig(max_iterations=1001)

    def test_confidence_threshold_range(self):
        """heuristic_confidence_threshold must be 0.0-1.0."""
        with pytest.raises(ValueError):
            GoalsConfig(heuristic_confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            GoalsConfig(heuristic_confidence_threshold=1.1)

    def test_progress_stall_tolerance_range(self):
        """progress_stall_tolerance must be 0.0-1.0."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(progress_stall_tolerance=-0.1)

        with pytest.raises(ValueError):
            CircuitBreakerConfig(progress_stall_tolerance=1.5)

    def test_temperature_range(self):
        """temperature must be 0.0-2.0."""
        with pytest.raises(ValueError):
            FastPathConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            FastPathConfig(temperature=2.5)

    def test_valid_boundary_values(self):
        """Boundary values should be accepted."""
        # Should not raise
        GoalsConfig(heuristic_confidence_threshold=0.0)
        GoalsConfig(heuristic_confidence_threshold=1.0)
        CircuitBreakerConfig(progress_stall_tolerance=0.0)
        CircuitBreakerConfig(progress_stall_tolerance=1.0)
        FastPathConfig(temperature=0.0)
        FastPathConfig(temperature=2.0)


# =============================================================================
# TEST: Config Loading
# =============================================================================


class TestLoadAgentsConfig:
    """Test config loading from various sources."""

    def test_load_with_no_sources(self):
        """Loading with no sources should return defaults."""
        config = load_agents_config()

        assert config.max_iterations == 10
        assert config.goals.classifier_enabled is True

    def test_load_from_dict(self):
        """Should load configuration from dictionary."""
        config_dict = {
            "agents": {
                "max_iterations": 20,
                "goals": {
                    "classifier_enabled": False,
                },
            }
        }

        config = load_agents_config(config_dict=config_dict)

        assert config.max_iterations == 20
        assert config.goals.classifier_enabled is False
        # Preserved defaults
        assert config.fast_path.enabled is True

    def test_load_from_toml_file(self):
        """Should load configuration from TOML file."""
        toml_content = """
[agents]
max_iterations = 25

[agents.circuit_breaker]
max_same_errors = 5
max_execution_time_seconds = 600
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            config = load_agents_config(config_path=config_file)

            assert config.max_iterations == 25
            assert config.circuit_breaker.max_same_errors == 5
            assert config.circuit_breaker.max_execution_time_seconds == 600
            # Preserved defaults
            assert config.circuit_breaker.enabled is True
        finally:
            config_file.unlink()

    def test_load_nonexistent_file_falls_back_to_defaults(self):
        """Loading from nonexistent file should fall back to defaults."""
        config = load_agents_config(config_path=Path("/nonexistent/path/config.toml"))

        assert config.max_iterations == 10
        assert config.goals.classifier_enabled is True

    def test_load_with_runtime_overrides(self):
        """Runtime overrides should take precedence."""
        config_dict = {
            "agents": {
                "max_iterations": 20,
            }
        }
        overrides = {
            "max_iterations": 30,
            "goals": {
                "classifier_enabled": False,
            },
        }

        config = load_agents_config(config_dict=config_dict, overrides=overrides)

        assert config.max_iterations == 30
        assert config.goals.classifier_enabled is False

    def test_invalid_config_falls_back_to_defaults(self):
        """Invalid config should fall back to defaults."""
        # Invalid: max_iterations can't be negative
        config_dict = {
            "agents": {
                "max_iterations": -1,
            }
        }

        config = load_agents_config(config_dict=config_dict)

        # Should use defaults when validation fails
        assert config.max_iterations == 10


# =============================================================================
# TEST: Environment Variable Overrides
# =============================================================================


class TestEnvVarOverrides:
    """Test environment variable override mechanism."""

    def test_env_override_boolean(self, monkeypatch):
        """Environment variables should override boolean values."""
        monkeypatch.setenv("LLMCORE_AGENTS__GOALS__CLASSIFIER_ENABLED", "false")

        config = load_agents_config()

        assert config.goals.classifier_enabled is False

    def test_env_override_integer(self, monkeypatch):
        """Environment variables should override integer values."""
        monkeypatch.setenv("LLMCORE_AGENTS__CIRCUIT_BREAKER__MAX_SAME_ERRORS", "7")

        config = load_agents_config()

        assert config.circuit_breaker.max_same_errors == 7

    def test_env_override_float(self, monkeypatch):
        """Environment variables should override float values."""
        monkeypatch.setenv(
            "LLMCORE_AGENTS__GOALS__HEURISTIC_CONFIDENCE_THRESHOLD", "0.85"
        )

        config = load_agents_config()

        assert config.goals.heuristic_confidence_threshold == 0.85

    def test_env_override_multiple(self, monkeypatch):
        """Multiple environment variables should all be applied."""
        monkeypatch.setenv("LLMCORE_AGENTS__GOALS__CLASSIFIER_ENABLED", "false")
        monkeypatch.setenv("LLMCORE_AGENTS__CIRCUIT_BREAKER__MAX_SAME_ERRORS", "10")
        monkeypatch.setenv("LLMCORE_AGENTS__FAST_PATH__ENABLED", "false")

        config = load_agents_config()

        assert config.goals.classifier_enabled is False
        assert config.circuit_breaker.max_same_errors == 10
        assert config.fast_path.enabled is False

    def test_env_override_takes_precedence_over_dict(self, monkeypatch):
        """Environment variables should take precedence over config dict."""
        monkeypatch.setenv("LLMCORE_AGENTS__GOALS__CLASSIFIER_ENABLED", "true")

        config_dict = {
            "agents": {
                "goals": {
                    "classifier_enabled": False,
                }
            }
        }

        config = load_agents_config(config_dict=config_dict)

        # Env var wins
        assert config.goals.classifier_enabled is True


# =============================================================================
# TEST: Enum Handling
# =============================================================================


class TestEnumHandling:
    """Test enum value handling in configuration."""

    def test_risk_level_enum(self):
        """RiskLevel enum values should be accepted."""
        config = HITLConfig(global_risk_threshold=RiskLevel.HIGH)
        assert config.global_risk_threshold == RiskLevel.HIGH

    def test_risk_level_string(self):
        """RiskLevel as string should be accepted."""
        config = HITLConfig(global_risk_threshold="low")
        assert config.global_risk_threshold == RiskLevel.LOW

    def test_timeout_policy_enum(self):
        """TimeoutPolicy enum values should be accepted."""
        config = HITLConfig(timeout_policy=TimeoutPolicy.APPROVE)
        assert config.timeout_policy == TimeoutPolicy.APPROVE

    def test_routing_strategy_enum(self):
        """RoutingStrategy enum values should be accepted."""
        config = RoutingConfig(strategy=RoutingStrategy.QUALITY_OPTIMIZED)
        assert config.strategy == RoutingStrategy.QUALITY_OPTIMIZED

    def test_routing_strategy_string(self):
        """RoutingStrategy as string should be accepted."""
        config = RoutingConfig(strategy="latency_optimized")
        assert config.strategy == RoutingStrategy.LATENCY_OPTIMIZED


# =============================================================================
# TEST: Serialization
# =============================================================================


class TestConfigSerialization:
    """Test configuration serialization/deserialization."""

    def test_model_dump(self):
        """Config should be serializable to dict."""
        config = AgentsConfig(max_iterations=20)
        dumped = config.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["max_iterations"] == 20
        assert "goals" in dumped
        assert dumped["goals"]["classifier_enabled"] is True

    def test_roundtrip(self):
        """Config should survive roundtrip serialization."""
        original = AgentsConfig(
            max_iterations=20,
            goals=GoalsConfig(classifier_enabled=False),
            circuit_breaker=CircuitBreakerConfig(max_same_errors=5),
        )

        dumped = original.model_dump()
        restored = AgentsConfig(**dumped)

        assert restored.max_iterations == original.max_iterations
        assert restored.goals.classifier_enabled == original.goals.classifier_enabled
        assert (
            restored.circuit_breaker.max_same_errors
            == original.circuit_breaker.max_same_errors
        )


# =============================================================================
# TEST: Integration with TOML Loading
# =============================================================================


class TestTOMLIntegration:
    """Test integration with full TOML configuration files."""

    def test_full_config_file(self):
        """Test loading a comprehensive config file."""
        toml_content = """
[agents]
max_iterations = 15
default_timeout = 900
default_persona = "developer"
memory_enabled = true

[agents.goals]
classifier_enabled = true
use_llm_fallback = true
heuristic_confidence_threshold = 0.85
trivial_max_iterations = 2
simple_max_iterations = 6

[agents.fast_path]
enabled = true
cache_enabled = false
max_response_time_ms = 3000

[agents.circuit_breaker]
enabled = true
max_same_errors = 5
max_execution_time_seconds = 600

[agents.activities]
enabled = true
max_per_iteration = 15
stop_on_error = true

[agents.capability_check]
enabled = true
strict_mode = false

[agents.hitl]
enabled = true
global_risk_threshold = "high"
timeout_policy = "approve"

[agents.routing]
enabled = true
strategy = "quality_optimized"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            config_file = Path(f.name)

        try:
            config = load_agents_config(config_path=config_file)

            # Root values
            assert config.max_iterations == 15
            assert config.default_timeout == 900
            assert config.default_persona == "developer"

            # Goals
            assert config.goals.classifier_enabled is True
            assert config.goals.use_llm_fallback is True
            assert config.goals.heuristic_confidence_threshold == 0.85
            assert config.goals.trivial_max_iterations == 2

            # Fast path
            assert config.fast_path.enabled is True
            assert config.fast_path.cache_enabled is False
            assert config.fast_path.max_response_time_ms == 3000

            # Circuit breaker
            assert config.circuit_breaker.max_same_errors == 5
            assert config.circuit_breaker.max_execution_time_seconds == 600

            # Activities
            assert config.activities.max_per_iteration == 15
            assert config.activities.stop_on_error is True

            # Capability check
            assert config.capability_check.strict_mode is False

            # HITL
            assert config.hitl.global_risk_threshold == RiskLevel.HIGH
            assert config.hitl.timeout_policy == TimeoutPolicy.APPROVE

            # Routing
            assert config.routing.strategy == RoutingStrategy.QUALITY_OPTIMIZED

        finally:
            config_file.unlink()
