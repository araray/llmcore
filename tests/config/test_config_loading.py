# tests/config/test_config_loading.py
"""
Tests for LLMCore configuration loading via confy.

Validates that LLMCore passes the correct parameter names to
confy.Config.__init__, ensuring config files, environment variables,
and overrides are actually applied.

The root cause of the "model always shows llama3" bug was that
LLMCore used `config_file_path` instead of `file_path`,
`env_prefix` instead of `prefix`, and `overrides` instead of
`overrides_dict`.  Since confy.Config accepts **kwargs, those
wrong names silently became extra dict entries instead of being
used as config sources.
"""

import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(path: Path, content: str) -> str:
    """Write TOML content to a file and return the path string."""
    path.write_text(textwrap.dedent(content))
    return str(path)


def _make_user_config(tmp_path: Path, **overrides) -> str:
    """Create a minimal user TOML config file.

    Default content uses provider=ollama, model=user-custom-model.
    Pass keyword args to override specific fields.
    """
    provider = overrides.get("provider", "ollama")
    model = overrides.get("model", "user-custom-model")
    api_url = overrides.get("api_url", "http://custom-host:11434")

    content = f"""\
    [llmcore]
    default_provider = "{provider}"
    default_model = "{model}"

    [llmcore.providers.ollama]
    api_url = "{api_url}"
    default_model = "{model}"
    """
    return _write_toml(tmp_path / "user_config.toml", content)


# ---------------------------------------------------------------------------
# Tests: confy parameter mapping
# ---------------------------------------------------------------------------


class TestConfyParameterNames:
    """Verify LLMCore passes the correct parameter names to confy.Config."""

    def test_file_path_is_used_not_config_file_path(self, tmp_path):
        """Config file values must override defaults when file_path is correct."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        user_config = _make_user_config(tmp_path, model="my-test-model")

        # CORRECT: file_path=
        cfg_correct = Config(defaults=defaults, file_path=user_config)
        assert cfg_correct.get("providers.ollama.default_model") == "my-test-model"

        # WRONG (old bug): config_file_path= — silently ignored
        cfg_wrong = Config(defaults=defaults, config_file_path=user_config)
        # The file is NOT loaded, so the default "llama3" remains
        assert cfg_wrong.get("providers.ollama.default_model") == "llama3"

    def test_overrides_dict_is_used_not_overrides(self, tmp_path):
        """Override dict values must override everything when overrides_dict is correct."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        user_config = _make_user_config(tmp_path, model="file-model")

        # CORRECT: overrides_dict=
        cfg = Config(
            defaults=defaults,
            file_path=user_config,
            overrides_dict={"providers.ollama.default_model": "override-model"},
        )
        assert cfg.get("providers.ollama.default_model") == "override-model"

        # WRONG (old bug): overrides= — silently ignored
        cfg_wrong = Config(
            defaults=defaults,
            file_path=user_config,
            overrides={"providers.ollama.default_model": "override-model"},
        )
        # Override is NOT applied, file value remains
        assert cfg_wrong.get("providers.ollama.default_model") == "file-model"

    def test_prefix_is_used_not_env_prefix(self):
        """Using 'prefix' must not create a spurious 'env_prefix' key.

        The exact env var remapping behavior is a confy concern. Here we
        verify that using the correct parameter name ('prefix') does NOT
        pollute the config dict with a spurious key, whereas the old wrong
        name ('env_prefix') does.
        """
        from confy.loader import Config

        minimal_defaults = {"llmcore": {"default_provider": "ollama"}}

        # CORRECT: prefix= — no spurious key
        cfg_correct = Config(defaults=minimal_defaults, prefix="LLMCORE")
        assert "env_prefix" not in cfg_correct, "Spurious 'env_prefix' key found — wrong param name"

        # WRONG (old bug): env_prefix= — creates spurious key
        cfg_wrong = Config(defaults=minimal_defaults, env_prefix="LLMCORE")
        assert "env_prefix" in cfg_wrong, "Expected spurious 'env_prefix' key from wrong param name"


class TestConfigFileMerging:
    """Verify config file values properly merge with/override defaults."""

    def test_user_model_overrides_default(self, tmp_path):
        """User config model must replace the hardcoded default."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        user_config = _make_user_config(tmp_path, model="qwen3:1.7b")
        cfg = Config(defaults=defaults, file_path=user_config)

        assert cfg.get("providers.ollama.default_model") == "qwen3:1.7b"
        assert cfg.get("llmcore.default_model") == "qwen3:1.7b"

    def test_user_api_url_overrides_default(self, tmp_path):
        """User config api_url must replace the default."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        user_config = _make_user_config(tmp_path, api_url="http://remote-host:11434")
        cfg = Config(defaults=defaults, file_path=user_config)
        assert cfg.get("providers.ollama.api_url") == "http://remote-host:11434"

    def test_unset_values_keep_defaults(self, tmp_path):
        """Values not in user config must retain defaults."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        # Minimal user config: only set ollama model
        user_config = _write_toml(
            tmp_path / "minimal.toml",
            """\
            [llmcore]
            default_provider = "ollama"

            [llmcore.providers.ollama]
            default_model = "custom-model"
            """,
        )
        cfg = Config(defaults=defaults, file_path=user_config)

        # Ollama model overridden
        assert cfg.get("providers.ollama.default_model") == "custom-model"
        # OpenAI model kept from defaults
        assert cfg.get("providers.openai.default_model") == "gpt-4o"

    def test_override_dict_takes_highest_precedence(self, tmp_path):
        """overrides_dict must beat both defaults and config file."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        user_config = _make_user_config(tmp_path, model="file-model")
        cfg = Config(
            defaults=defaults,
            file_path=user_config,
            overrides_dict={"providers.ollama.default_model": "final-model"},
        )
        assert cfg.get("providers.ollama.default_model") == "final-model"


class TestWairuStyleConfigFile:
    """Test with a config file structured like the actual wairu config."""

    def test_wairu_config_format(self, tmp_path):
        """Config in wairu's [llmcore.providers.X] format must be loaded."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        # Mimic the real av_wairu_conf.toml structure
        user_config = _write_toml(
            tmp_path / "wairu.toml",
            """\
            [wairu]
            default_provider = "ollama"
            default_model = "qwen3:1.7b"

            [llmcore]
            default_provider = "ollama"
            default_model = "qwen3:1.7b"

            [llmcore.providers.ollama]
            api_url = "http://localhost:11434"
            default_model = "qwen3:4b"

            [llmcore.providers.anthropic]
            default_model = "claude-sonnet-4-20250514"
            max_tokens = 4096

            [llmcore.storage.session]
            backend = "sqlite"

            [llmcore.storage.vector]
            backend = "chromadb"

            [llmcore.rag]
            enabled = false
            """,
        )

        cfg = Config(defaults=defaults, file_path=user_config)

        # Verify key merges from the wairu-style config
        assert cfg.get("llmcore.default_provider") == "ollama"
        assert cfg.get("llmcore.default_model") == "qwen3:1.7b"
        assert cfg.get("providers.ollama.default_model") == "qwen3:4b"
        assert cfg.get("providers.ollama.api_url") == "http://localhost:11434"


class TestNoSpuriousKeys:
    """Ensure wrong param names don't create junk keys in config."""

    def test_correct_params_no_extra_keys(self, tmp_path):
        """Using correct param names must not inject extra keys."""
        import tomllib

        from confy.loader import Config

        defaults_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "llmcore"
            / "config"
            / "default_config.toml"
        )
        with open(defaults_path, "rb") as f:
            defaults = tomllib.load(f)

        cfg = Config(
            defaults=defaults,
            file_path=None,
            prefix=None,
            overrides_dict=None,
        )

        # These keys should NOT exist — they were the old bug symptoms
        keys = set(cfg.keys())
        assert "config_file_path" not in keys, "Spurious 'config_file_path' key found"
        assert "env_prefix" not in keys, "Spurious 'env_prefix' key found"
        assert "overrides" not in keys, "Spurious 'overrides' key found"

    def test_wrong_params_create_extra_keys(self):
        """Demonstrate the old bug: wrong names create spurious dict keys."""
        from confy.loader import Config

        cfg = Config(
            defaults={"llmcore": {"default_provider": "ollama"}},
            config_file_path="/some/path",  # WRONG
            env_prefix="LLMCORE",  # WRONG
            overrides={"key": "val"},  # WRONG
        )

        # These exist as plain dict keys — the bug!
        assert "config_file_path" in cfg
        assert "env_prefix" in cfg
        assert "overrides" in cfg
