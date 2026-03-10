"""
Environment-based settings loaded from .env file.
These are secrets and deployment-level settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Deployment ----
    bot_mode: str = "paper"  # live | paper | backtest

    # ---- Coinbase ----
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    coinbase_api_passphrase: str = ""
    coinbase_rest_url: str = "https://api.coinbase.com"
    coinbase_ws_url: str = "wss://advanced-trade-ws.coinbase.com"

    # ---- Database ----
    database_url: str = "sqlite:///./data/trading_bot.db"

    # ---- Dashboard Auth ----
    dashboard_username: str = "admin"
    dashboard_password: str = "changeme"
    jwt_secret_key: str = "change_this_secret_key_before_deploying"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480  # 8 hours

    # ---- Discord ----
    discord_webhook_url: str = ""

    # ---- Anthropic ----
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-6"

    # ---- Logging ----
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # ---- ML ----
    ml_model_dir: str = "./data/models"

    # ---- Data Cache ----
    data_cache_dir: str = "./data/historical"

    # ---- Paper Trading ----
    paper_initial_balance: float = 10000.0

    # ---- Misc ----
    max_concurrent_requests: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
