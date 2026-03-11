"""
Discord Signals Bot — entry point.

Usage:
    python -m discord_signals_bot.main

Required environment variables (see .env.example):
    DISCORD_BOT_TOKEN
    ENTRY_CHANNEL_ID
    EXIT_CHANNEL_ID
    TRADER_USER_ID
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv

# ── Load .env from the discord_signals_bot directory ─────────────────────────
_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")

# ── Logging setup ─────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return val


def _load_config() -> dict:
    config_path = _HERE / "config" / "signals_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


async def main() -> None:
    from .bot.client import SignalsBot
    from .db.database import init_db, DB_PATH
    from .market_data.binance_client import BinanceClient
    from .signals.generator import SignalGenerator
    from .bot.trade_tracker import TradeTracker

    # ── Validate environment ───────────────────────────────────────────────────
    token = _require_env("DISCORD_BOT_TOKEN")
    entry_channel_id = int(_require_env("ENTRY_CHANNEL_ID"))
    exit_channel_id = int(_require_env("EXIT_CHANNEL_ID"))
    trader_user_id = int(_require_env("TRADER_USER_ID"))
    scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    min_score = float(os.getenv("MIN_SIGNAL_SCORE", "40"))

    logger.info("Starting Discord Signals Bot")
    logger.info("  Entry channel : %d", entry_channel_id)
    logger.info("  Exit channel  : %d", exit_channel_id)
    logger.info("  Trader user   : %d", trader_user_id)
    logger.info("  Scan interval : %ds", scan_interval)
    logger.info("  Min score     : %.0f", min_score)

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = _load_config()

    # Inject runtime overrides from env
    cfg["scoring"]["no_trade_max"] = min_score
    cfg["voting"]["score_thresholds"] = {
        "no_trade_max": min_score,
        "strong_min": cfg["scoring"]["strong_min"],
        "elite_min": cfg["scoring"]["elite_min"],
    }

    # ── Init DB ────────────────────────────────────────────────────────────────
    await init_db(DB_PATH)

    # ── Build components ───────────────────────────────────────────────────────
    tracker = TradeTracker(DB_PATH)

    async with aiohttp.ClientSession() as http_session:
        binance = BinanceClient(session=http_session)
        generator = SignalGenerator(cfg, binance)

        bot = SignalsBot(
            entry_channel_id=entry_channel_id,
            exit_channel_id=exit_channel_id,
            trader_user_id=trader_user_id,
            scan_interval=scan_interval,
            generator=generator,
            tracker=tracker,
        )

        logger.info("Connecting to Discord...")
        try:
            await bot.start(token)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        finally:
            if not bot.is_closed():
                await bot.close()
            logger.info("Bot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
