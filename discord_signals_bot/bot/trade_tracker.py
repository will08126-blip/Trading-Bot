"""
Trade Tracker — manages the lifecycle of confirmed trades in SQLite.

Flow:
  1. An entry signal is posted in #entry-signals → signal_log row inserted
  2. User reacts ✅ to the message → active_trades row inserted (trade confirmed)
  3. Exit scanner detects TP1/TP2/SL → exit_log row inserted, active_trades updated
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from ..db.database import DB_PATH, get_db

logger = logging.getLogger(__name__)


class TradeTracker:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path

    # ── Entry signal logged (before user confirms) ────────────────────────────

    async def log_signal(
        self,
        message_id: str,
        symbol: str,
        direction: str,
        score: float,
        tier: str,
        leverage: int,
        portfolio_pct: int,
        strategy_names: List[str],
        regime: str,
        entry_price: float,
        stop_price: float,
        tp1_price: float,
        tp2_price: float,
    ) -> None:
        """Log a posted entry signal. Not yet an active trade."""
        now = datetime.now(timezone.utc).isoformat()
        async with await get_db(self._db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO signal_log
                    (symbol, direction, score, tier, leverage, pct,
                     strategies, regime, logged_at, confirmed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (symbol, direction, score, tier, leverage, portfolio_pct,
                 ",".join(strategy_names), regime, now),
            )
            # Also store pending entry in active_trades with status='pending'
            # so we can link the Discord message_id back to full trade details
            await db.execute(
                """
                INSERT OR IGNORE INTO active_trades
                    (message_id, symbol, direction, entry_price, stop_price,
                     tp1_price, tp2_price, leverage, portfolio_pct,
                     strategy_names, score, tier, regime, entered_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (message_id, symbol, direction, entry_price, stop_price,
                 tp1_price, tp2_price, leverage, portfolio_pct,
                 ",".join(strategy_names), score, tier, regime, now),
            )
            await db.commit()
        logger.debug("signal_logged message_id=%s symbol=%s dir=%s", message_id, symbol, direction)

    # ── User confirms entry (reacts ✅) ───────────────────────────────────────

    async def confirm_entry(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark a pending signal as an active trade.
        Returns the trade dict if found, else None.
        """
        async with await get_db(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM active_trades WHERE message_id = ? AND status = 'pending'",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return None

            await db.execute(
                "UPDATE active_trades SET status = 'active' WHERE message_id = ?",
                (message_id,),
            )
            # Mark confirmed in signal_log
            await db.execute(
                "UPDATE signal_log SET confirmed = 1 WHERE symbol = ? AND direction = ? "
                "ORDER BY id DESC LIMIT 1",
                (row["symbol"], row["direction"]),
            )
            await db.commit()

        trade = dict(row)
        logger.info("trade_confirmed message_id=%s symbol=%s dir=%s lev=%sx",
                    message_id, trade["symbol"], trade["direction"], trade["leverage"])
        return trade

    # ── Fetch active trades ───────────────────────────────────────────────────

    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Return all currently active (confirmed) trades."""
        async with await get_db(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM active_trades WHERE status = 'active'",
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_trade_by_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        async with await get_db(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM active_trades WHERE message_id = ?", (message_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return dict(row) if row else None

    # ── Mark TP1 hit (move SL to breakeven) ──────────────────────────────────

    async def mark_tp1_hit(self, trade_id: int) -> None:
        async with await get_db(self._db_path) as db:
            await db.execute(
                "UPDATE active_trades SET tp1_hit = 1 WHERE id = ?", (trade_id,)
            )
            await db.commit()
        logger.info("tp1_hit trade_id=%d", trade_id)

    # ── Close a trade ─────────────────────────────────────────────────────────

    async def close_trade(
        self,
        trade_id: int,
        exit_type: str,
        exit_price: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with await get_db(self._db_path) as db:
            # Fetch symbol/direction for log
            async with db.execute(
                "SELECT symbol, direction FROM active_trades WHERE id = ?", (trade_id,)
            ) as cur:
                row = await cur.fetchone()

            if row:
                await db.execute(
                    "INSERT INTO exit_log (trade_id, symbol, direction, exit_type, exit_price, logged_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (trade_id, row["symbol"], row["direction"], exit_type, exit_price, now),
                )

            await db.execute(
                "UPDATE active_trades SET status = 'closed' WHERE id = ?", (trade_id,)
            )
            await db.commit()
        logger.info("trade_closed trade_id=%d exit_type=%s price=%.4f", trade_id, exit_type, exit_price)

    # ── User manually closes a trade ─────────────────────────────────────────

    async def manual_close(self, symbol: str, direction: str) -> bool:
        """Close the most recent active trade for a symbol/direction. Returns True if found."""
        async with await get_db(self._db_path) as db:
            async with db.execute(
                "SELECT id FROM active_trades WHERE symbol = ? AND direction = ? AND status = 'active' "
                "ORDER BY id DESC LIMIT 1",
                (symbol, direction),
            ) as cur:
                row = await cur.fetchone()

            if not row:
                return False

            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE active_trades SET status = 'closed' WHERE id = ?", (row["id"],)
            )
            await db.execute(
                "INSERT INTO exit_log (trade_id, symbol, direction, exit_type, exit_price, logged_at) "
                "VALUES (?, ?, ?, 'MANUAL', 0.0, ?)",
                (row["id"], symbol, direction, now),
            )
            await db.commit()
        logger.info("manual_close symbol=%s dir=%s", symbol, direction)
        return True

    # ── Stats helpers ─────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        async with await get_db(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) as total FROM signal_log"
            ) as cur:
                total_signals = (await cur.fetchone())["total"]

            async with db.execute(
                "SELECT COUNT(*) as confirmed FROM signal_log WHERE confirmed = 1"
            ) as cur:
                confirmed = (await cur.fetchone())["confirmed"]

            async with db.execute(
                "SELECT COUNT(*) as active FROM active_trades WHERE status = 'active'"
            ) as cur:
                active = (await cur.fetchone())["active"]

            async with db.execute(
                "SELECT COUNT(*) as closed FROM exit_log"
            ) as cur:
                closed = (await cur.fetchone())["closed"]

        return {
            "total_signals": total_signals,
            "confirmed_entries": confirmed,
            "active_trades": active,
            "closed_trades": closed,
        }
