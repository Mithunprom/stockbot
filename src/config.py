"""Central configuration using pydantic-settings.

All secrets come from environment variables (.env / Docker env).
Never hardcode credentials here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Environment ─────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ─── Market Data ──────────────────────────────────────────────────────────
    polygon_api_key: str = Field(..., description="Polygon.io API key")
    unusual_whales_api_key: str = Field("", description="Unusual Whales API key")
    news_api_key: str = Field("", description="NewsAPI key")
    benzinga_api_key: str = Field("", description="Benzinga API key")

    # ─── Broker ───────────────────────────────────────────────────────────────
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_mode: Literal["paper", "live"] = "paper"
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_live_base_url: str = "https://api.alpaca.markets"

    @property
    def alpaca_base_url(self) -> str:
        return self.alpaca_paper_base_url if self.alpaca_mode == "paper" else self.alpaca_live_base_url

    # ─── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(..., description="Async SQLAlchemy DB URL")
    database_sync_url: str = ""

    @model_validator(mode="after")
    def set_sync_url(self) -> Settings:
        if not self.database_sync_url:
            self.database_sync_url = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
        return self

    # ─── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── AI ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", description="Anthropic API key")

    # ─── A/B Testing ──────────────────────────────────────────────────────────
    ab_test_enabled: bool = False
    ab_capital_split: float = 0.5  # fraction of capital for Pipeline A (rest → B)

    # ─── Trading mode (resolved from yaml config) ─────────────────────────────
    _trading_config: dict | None = None

    def get_trading_config(self) -> dict:
        """Load the YAML config matching the current Alpaca mode."""
        if self._trading_config is None:
            config_path = Path("config") / f"{self.alpaca_mode}.yaml"
            with open(config_path) as f:
                raw = f.read()
            # Expand ${ENV_VAR} references
            for key, val in os.environ.items():
                raw = raw.replace(f"${{{key}}}", val)
            object.__setattr__(self, "_trading_config", yaml.safe_load(raw))
        return self._trading_config  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()  # type: ignore[call-arg]
