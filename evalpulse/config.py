"""EvalPulse configuration loader and validation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr


class ModuleConfig(BaseModel):
    """Enable/disable individual evaluation modules."""

    hallucination: bool = True
    drift: bool = True
    rag_quality: bool = True
    response_quality: bool = True
    regression: bool = True


class ThresholdConfig(BaseModel):
    """Alert threshold configuration."""

    hallucination: float = 0.3
    drift: float = 0.15
    rag_groundedness_min: float = 0.65
    toxicity: float = 0.05
    regression_fail_rate: float = 0.10


class NotificationConfig(BaseModel):
    """Notification channel configuration."""

    email: str | None = None
    slack_webhook: str | None = None


class EvalPulseConfig(BaseModel):
    """Main EvalPulse configuration."""

    app_name: str = "default"
    storage_backend: str = "sqlite"
    sqlite_path: str = "evalpulse.db"
    database_url: str | None = None
    groq_api_key: SecretStr | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    modules: ModuleConfig = Field(default_factory=ModuleConfig)
    drift_window_size: int = 50
    baseline_window_days: int = 7
    drift_threshold: float = 0.15
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    alert_cooldown_seconds: int = 300
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> EvalPulseConfig:
        """Load configuration from a YAML file.

        Falls back to defaults if file doesn't exist.
        Environment variables override file values:
          - EVALPULSE_GROQ_API_KEY -> groq_api_key
          - EVALPULSE_APP_NAME -> app_name
          - EVALPULSE_STORAGE_BACKEND -> storage_backend
          - EVALPULSE_SQLITE_PATH -> sqlite_path
          - EVALPULSE_DATABASE_URL -> database_url
        """
        data: dict = {}

        if path is None:
            # Search in current directory, then parent directories
            for candidate in [Path("evalpulse.yml"), Path("evalpulse.yaml")]:
                if candidate.exists():
                    path = candidate
                    break

        if path is not None:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded

        # Environment variable overrides
        env_map = {
            "EVALPULSE_GROQ_API_KEY": "groq_api_key",
            "EVALPULSE_APP_NAME": "app_name",
            "EVALPULSE_STORAGE_BACKEND": "storage_backend",
            "EVALPULSE_SQLITE_PATH": "sqlite_path",
            "EVALPULSE_DATABASE_URL": "database_url",
            "GROQ_API_KEY": "groq_api_key",
        }
        for env_var, config_key in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                data[config_key] = val

        return cls.model_validate(data)


_config: EvalPulseConfig | None = None


def get_config(path: str | Path | None = None) -> EvalPulseConfig:
    """Get the global EvalPulse configuration singleton.

    On first call, loads from the specified path (or auto-discovers evalpulse.yml).
    Subsequent calls return the cached config.
    """
    global _config
    if _config is None:
        _config = EvalPulseConfig.load(path)
    return _config


def reset_config() -> None:
    """Reset the config singleton (useful for testing)."""
    global _config
    _config = None
