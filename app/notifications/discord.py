"""
Discord notification service.
Sends webhook messages for key trading events.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.models.notification_log import NotificationLog

logger = structlog.get_logger(__name__)


class DiscordNotifier:
    """
    Sends notifications to a Discord webhook.
    Gracefully degrades if webhook URL is not configured.
    """

    def __init__(self, db: Optional[AsyncSession] = None) -> None:
        settings = get_settings()
        self._webhook_url = settings.discord_webhook_url
        self._db = db
        self._enabled = bool(self._webhook_url)
        if not self._enabled:
            logger.info("discord_notifications_disabled", reason="no_webhook_url")

    async def send(
        self,
        event_type: str,
        title: str,
        message: str,
        color: int = 0x0099FF,
        fields: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a Discord embed message."""
        if not self._enabled:
            return False

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "CryptoBot"},
        }

        if fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in fields.items()
            ]

        payload = {"embeds": [embed]}

        success = False
        error_msg = None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                resp.raise_for_status()
            success = True
        except Exception as e:
            error_msg = str(e)[:500]
            logger.warning("discord_send_failed", event=event_type, error=error_msg)

        # Log to DB
        if self._db:
            self._db.add(NotificationLog(
                channel="discord",
                event_type=event_type,
                message=f"{title}: {message[:500]}",
                sent_at=datetime.now(timezone.utc),
                success=success,
                error_message=error_msg,
            ))
            try:
                await self._db.commit()
            except Exception:
                await self._db.rollback()

        return success

    # ------------------------------------------------------------------ event helpers

    async def bot_started(self, mode: str) -> None:
        await self.send(
            "bot_started", "🤖 Bot Started",
            f"Trading bot started in **{mode}** mode.",
            color=0x00FF00,
        )

    async def bot_stopped(self, reason: str = "manual") -> None:
        await self.send(
            "bot_stopped", "🛑 Bot Stopped",
            f"Trading bot stopped. Reason: {reason}",
            color=0xFF6600,
        )

    async def kill_switch_triggered(self, reason: str) -> None:
        await self.send(
            "kill_switch", "🚨 KILL SWITCH TRIGGERED",
            f"Emergency stop activated. Reason: **{reason}**\nAll new trading is halted.",
            color=0xFF0000,
        )

    async def trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_price: float,
        tp_price: float,
        score: float,
        strategy: str,
    ) -> None:
        emoji = "📈" if direction == "long" else "📉"
        await self.send(
            "trade_opened",
            f"{emoji} Trade Opened: {symbol}",
            f"**{direction.upper()}** entry executed",
            color=0x00AA00 if direction == "long" else 0xAA0000,
            fields={
                "Symbol": symbol,
                "Direction": direction.upper(),
                "Entry": f"${entry_price:,.2f}",
                "Size": f"{size:.4f}",
                "Stop": f"${stop_price:,.2f}",
                "TP": f"${tp_price:,.2f}",
                "Score": f"{score:.1f}",
                "Strategy": strategy,
            },
        )

    async def trade_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_minutes: float,
        reason: str,
    ) -> None:
        is_win = pnl > 0
        emoji = "✅" if is_win else "❌"
        color = 0x00AA00 if is_win else 0xAA0000
        await self.send(
            "trade_closed",
            f"{emoji} Trade Closed: {symbol}",
            f"**{reason.upper()}** — {'Profit' if is_win else 'Loss'}: ${pnl:+.2f}",
            color=color,
            fields={
                "Symbol": symbol,
                "Direction": direction.upper(),
                "Entry": f"${entry_price:,.2f}",
                "Exit": f"${exit_price:,.2f}",
                "PnL": f"${pnl:+.2f} ({pnl_pct:+.2f}%)",
                "Hold": f"{hold_minutes:.0f}m",
                "Reason": reason,
            },
        )

    async def unusual_slippage(self, symbol: str, expected_pct: float, actual_pct: float) -> None:
        await self.send(
            "unusual_slippage", "⚠️ Unusual Slippage",
            f"{symbol}: Expected {expected_pct:.3f}% slippage, got {actual_pct:.3f}%",
            color=0xFFAA00,
        )

    async def strategy_disabled(self, strategy_name: str, reason: str, hours: float) -> None:
        await self.send(
            "strategy_disabled", "⏸️ Strategy Disabled",
            f"**{strategy_name}** paused for {hours:.1f}h. Reason: {reason}",
            color=0xFFAA00,
        )

    async def daily_report(self, content: str) -> None:
        await self.send(
            "daily_report", "📊 Daily Report",
            content[:2000],
            color=0x0099FF,
        )

    async def weekly_report(self, content: str) -> None:
        await self.send(
            "weekly_report", "📈 Weekly Report",
            content[:2000],
            color=0x0055FF,
        )

    async def error_alert(self, component: str, error: str) -> None:
        await self.send(
            "error", f"🔴 Error: {component}",
            f"```\n{error[:1500]}\n```",
            color=0xFF0000,
        )

    async def anomaly_summary(self, summary: str) -> None:
        await self.send(
            "anomaly", "🔍 Anomaly Detected",
            summary[:2000],
            color=0xFF6600,
        )
