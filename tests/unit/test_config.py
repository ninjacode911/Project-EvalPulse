"""Tests for EvalPulse configuration system."""

import pytest
import yaml

from evalpulse.config import EvalPulseConfig, get_config, reset_config


class TestEvalPulseConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Config should have sensible defaults when no file exists."""
        config = EvalPulseConfig()
        assert config.app_name == "default"
        assert config.storage_backend == "sqlite"
        assert config.sqlite_path == "evalpulse.db"
        assert config.modules.hallucination is True
        assert config.thresholds.hallucination == 0.3
        assert config.thresholds.drift == 0.15

    def test_load_from_yaml(self, tmp_path):
        """Config should load values from a YAML file."""
        config_data = {
            "app_name": "my-app",
            "storage_backend": "sqlite",
            "sqlite_path": "custom.db",
            "thresholds": {"hallucination": 0.5, "drift": 0.2},
        }
        config_path = tmp_path / "evalpulse.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = EvalPulseConfig.load(str(config_path))
        assert config.app_name == "my-app"
        assert config.sqlite_path == "custom.db"
        assert config.thresholds.hallucination == 0.5
        assert config.thresholds.drift == 0.2
        # Non-overridden defaults still work
        assert config.thresholds.toxicity == 0.05

    def test_load_missing_file_returns_defaults(self):
        """Loading from a nonexistent file should return defaults."""
        config = EvalPulseConfig.load("/nonexistent/path/config.yml")
        assert config.app_name == "default"
        assert config.storage_backend == "sqlite"

    def test_env_var_override(self, monkeypatch):
        """Environment variables should override file values."""
        monkeypatch.setenv("EVALPULSE_APP_NAME", "env-app")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
        config = EvalPulseConfig.load()
        assert config.app_name == "env-app"
        assert config.groq_api_key is not None
        assert config.groq_api_key.get_secret_value() == "gsk_test_key"

    def test_invalid_values_raise_error(self, tmp_path):
        """Invalid config values should raise ValidationError."""
        config_data = {"thresholds": {"hallucination": "not_a_number"}}
        config_path = tmp_path / "evalpulse.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(Exception):
            EvalPulseConfig.load(str(config_path))

    def test_singleton_behavior(self, tmp_config):
        """get_config should return the same instance on repeated calls."""
        config1 = get_config(tmp_config)
        config2 = get_config()
        assert config1 is config2

    def test_reset_clears_singleton(self, tmp_config):
        """reset_config should clear the cached config."""
        config1 = get_config(tmp_config)
        reset_config()
        # After reset, should reload
        config2 = get_config(tmp_config)
        assert config1 is not config2

    def test_modules_config(self):
        """Module enable/disable should work."""
        config = EvalPulseConfig(modules={"hallucination": False, "drift": True})
        assert config.modules.hallucination is False
        assert config.modules.drift is True
        assert config.modules.rag_quality is True  # default


class TestConfigEdgeCases:
    """Edge cases for configuration."""

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file should use defaults."""
        config_path = tmp_path / "evalpulse.yml"
        config_path.write_text("")
        config = EvalPulseConfig.load(str(config_path))
        assert config.app_name == "default"

    def test_yaml_with_extra_fields(self, tmp_path):
        """Extra fields in YAML should be ignored."""
        config_data = {"app_name": "test", "unknown_field": "value"}
        config_path = tmp_path / "evalpulse.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        # Should not raise
        config = EvalPulseConfig.load(str(config_path))
        assert config.app_name == "test"
