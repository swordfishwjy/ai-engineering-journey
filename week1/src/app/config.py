from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    default_model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    request_timeout: float = 60.0  # seconds, per attempt
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    enable_fault_injection: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
