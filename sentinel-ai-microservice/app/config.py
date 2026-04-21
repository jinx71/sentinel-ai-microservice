"""Application configuration loaded from environment variables.

Why: pydantic-settings gives typed, validated config with one source of truth.
The same Settings object is read locally (.env), in Docker (-e flags), and in
Kubernetes (env from ConfigMap + Secret) — no code changes between environments.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service identity
    app_name: str = "sentinel-ai-microservice"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Model
    model_path: str = "app/artifacts/model.joblib"
    contamination: float = 0.02  # expected anomaly fraction used at training time
    random_seed: int = 42

    # Optional LLM enrichment (graceful degradation when unset)
    enable_llm_explanations: bool = False
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    @property
    def llm_available(self) -> bool:
        """LLM path is usable only when explicitly enabled AND a key is present."""
        return self.enable_llm_explanations and bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so config is parsed once per process."""
    return Settings()
