"""Centralized application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Mental Health Information Support Assistant"
    app_version: str = "2.0.0"
    environment: str = "development"
    llm_provider: Literal["mock", "openai_compatible"] = "mock"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    dashscope_compatible_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str | None = None
    dashscope_model: str = "qwen-plus"
    enable_optional_classifier: bool = False
    optional_classifier_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    default_region: str = "generic"
    knowledge_manifest_path: Path = Path("knowledge/manifest.json")
    survey_path: Path = Path("surveys/example_wellbeing_survey.json")
    support_resources_path: Path = Path("resources/support_resources.json")
    sqlite_path: Path = Path("data/app.sqlite3")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
