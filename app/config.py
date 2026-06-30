"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Embedding Migration Framework"
    environment: str = "development"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    # Migration defaults
    sample_fraction: float = 0.03
    confidence_threshold: float = 0.90

    def ensure_dirs(self) -> None:
        """Create working directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
