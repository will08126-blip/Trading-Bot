"""
SQLite database setup using aiosqlite.
Manages the active_trades table and signal log.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "signals_bot.db"

CREATE_ACTIVE_TRADES = """
CREATE TABLE IF NOT EXISTS active_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      TEXT    NOT NULL UNIQUE,   -- Discord message ID of the entry signal
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,          -- 'long' | 'short'
    entry_price     REAL    NOT NULL,
    stop_price      REAL    NOT NULL,
    tp1_price       REAL    NOT NULL,
    tp2_price       REAL    NOT NULL,
    leverage        INTEGER NOT NULL,
    portfolio_pct   INTEGER NOT NULL,
    strategy_names  TEXT    NOT NULL,          -- comma-separated
    score           REAL    NOT NULL,
    tier            TEXT    NOT NULL,
    regime          TEXT    NOT NULL,
    tp1_hit         INTEGER NOT NULL DEFAULT 0,  -- bool
    entered_at      TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'active'  -- 'active' | 'closed'
);
"""

CREATE_SIGNAL_LOG = """
CREATE TABLE IF NOT EXISTS signal_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    score       REAL    NOT NULL,
    tier        TEXT    NOT NULL,
    leverage    INTEGER NOT NULL,
    pct         INTEGER NOT NULL,
    strategies  TEXT    NOT NULL,
    regime      TEXT    NOT NULL,
    logged_at   TEXT    NOT NULL,
    confirmed   INTEGER NOT NULL DEFAULT 0
);
"""

CREATE_EXIT_LOG = """
CREATE TABLE IF NOT EXISTS exit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER NOT NULL,
    symbol      TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    exit_type   TEXT    NOT NULL,
    exit_price  REAL    NOT NULL,
    logged_at   TEXT    NOT NULL
);
"""


async def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_ACTIVE_TRADES)
        await db.execute(CREATE_SIGNAL_LOG)
        await db.execute(CREATE_EXIT_LOG)
        await db.commit()
    logger.info("database_initialised path=%s", db_path)


async def get_db(db_path: Path = DB_PATH) -> aiosqlite.Connection:
    """Open and return an aiosqlite connection. Caller is responsible for closing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    return conn
