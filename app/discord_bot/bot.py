"""
Discord bot with slash commands for monitoring the trading bot.

Commands:
  /status   - Full system status (mode, equity, PnL, open positions)
  /positions - List all open positions
  /trades   - Recent closed trades
  /metrics  - Performance metrics (win rate, profit factor, etc.)

Setup:
  1. Create a bot at https://discord.com/developers/applications
  2. Enable "Message Content Intent" under Bot settings
  3. Add bot to your server with scopes: bot, applications.commands
  4. Set DISCORD_BOT_TOKEN and DISCORD_GUILD_ID in .env
  5. Run: python -m app.main --mode discord-bot
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import discord
from discord import app_commands
import structlog

logger = structlog.get_logger(__name__)


def _pnl_color(pnl: float) -> discord.Color:
    if pnl > 0:
        return discord.Color.green()
    if pnl < 0:
        return discord.Color.red()
    return discord.Color.greyple()


class TradingBotClient(discord.Client):
    def __init__(self, guild_id: int = 0) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._guild_id = guild_id
        self._guild: Optional[discord.Object] = (
            discord.Object(id=guild_id) if guild_id else None
        )

    async def setup_hook(self) -> None:
        """Register slash commands. Guild sync is instant; global sync takes ~1 hour."""
        _register_commands(self.tree)
        if self._guild:
            self.tree.copy_global_to(guild=self._guild)
            await self.tree.sync(guild=self._guild)
            logger.info("discord_commands_synced", scope="guild", guild_id=self._guild_id)
        else:
            await self.tree.sync()
            logger.info("discord_commands_synced", scope="global")

    async def on_ready(self) -> None:
        logger.info("discord_bot_ready", user=str(self.user))
        print(f"[Discord Bot] Logged in as {self.user} — slash commands ready.")


# ------------------------------------------------------------------ helpers

async def _fetch_db_metrics() -> dict:
    """Query the SQLite database for current metrics. Returns empty dict on error."""
    try:
        from sqlalchemy import select, desc
        from app.db.database import AsyncSessionLocal
        from app.models.equity_snapshot import EquitySnapshot
        from app.models.trade import Trade
        from app.models.strategy_health import StrategyHealth
        from app.config.settings import get_settings

        settings = get_settings()

        async with AsyncSessionLocal() as db:
            # Latest equity snapshot
            snap_result = await db.execute(
                select(EquitySnapshot).order_by(desc(EquitySnapshot.snapshot_time)).limit(1)
            )
            snap = snap_result.scalar_one_or_none()

            # Open positions
            open_result = await db.execute(
                select(Trade).where(Trade.status == "open").order_by(desc(Trade.entry_time))
            )
            open_trades = open_result.scalars().all()

            # Recent closed trades
            closed_result = await db.execute(
                select(Trade)
                .where(Trade.status == "closed")
                .order_by(desc(Trade.entry_time))
                .limit(5)
            )
            recent_closed = closed_result.scalars().all()

            # All closed for win rate
            all_closed_result = await db.execute(
                select(Trade).where(Trade.status == "closed")
            )
            all_closed = all_closed_result.scalars().all()

            # Strategy health
            sh_result = await db.execute(
                select(StrategyHealth).order_by(
                    StrategyHealth.strategy_name, desc(StrategyHealth.computed_at)
                )
            )
            sh_rows = sh_result.scalars().all()
            seen: set = set()
            strategy_health = []
            for row in sh_rows:
                if row.strategy_name not in seen:
                    seen.add(row.strategy_name)
                    strategy_health.append(row)

            pnls = [t.realized_pnl or 0.0 for t in all_closed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            return {
                "mode": settings.bot_mode,
                "total_equity": snap.total_equity if snap else 0.0,
                "daily_pnl": snap.realized_pnl_today if snap else 0.0,
                "total_pnl": snap.realized_pnl_total if snap else 0.0,
                "drawdown_pct": snap.drawdown_pct if snap else 0.0,
                "open_trades": open_trades,
                "recent_closed": recent_closed,
                "total_trades": len(all_closed),
                "win_rate": len(wins) / len(pnls) if pnls else 0.0,
                "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0.0,
                "avg_win": sum(wins) / len(wins) if wins else 0.0,
                "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                "strategy_health": strategy_health,
                "snapshot_time": snap.snapshot_time if snap else None,
            }
    except Exception as exc:
        logger.warning("discord_db_fetch_failed", error=str(exc))
        return {"error": str(exc)}


# ------------------------------------------------------------------ commands

def _register_commands(tree: app_commands.CommandTree) -> None:

    @tree.command(name="status", description="Full trading bot status: mode, equity, PnL, positions")
    async def status(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        data = await _fetch_db_metrics()

        if "error" in data:
            await interaction.followup.send(
                f"⚠️ Could not fetch status: `{data['error']}`", ephemeral=True
            )
            return

        daily_pnl = data["daily_pnl"]
        total_pnl = data["total_pnl"]
        mode = data["mode"].upper()
        mode_emoji = {"LIVE": "🟢", "PAPER": "🟡", "BACKTEST": "🔵"}.get(mode, "⚪")

        embed = discord.Embed(
            title=f"{mode_emoji} Trading Bot Status — {mode}",
            color=_pnl_color(daily_pnl),
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="💰 Total Equity", value=f"${data['total_equity']:,.2f}", inline=True)
        embed.add_field(name="📅 Daily PnL", value=f"${daily_pnl:+,.2f}", inline=True)
        embed.add_field(name="📈 Total PnL", value=f"${total_pnl:+,.2f}", inline=True)
        embed.add_field(name="📉 Max Drawdown", value=f"{data['drawdown_pct']:.2f}%", inline=True)
        embed.add_field(name="📊 Open Positions", value=str(len(data["open_trades"])), inline=True)
        embed.add_field(name="🔁 Total Trades", value=str(data["total_trades"]), inline=True)

        if data["snapshot_time"]:
            embed.set_footer(text=f"Last snapshot: {data['snapshot_time'].strftime('%Y-%m-%d %H:%M UTC')}")

        await interaction.followup.send(embed=embed)

    @tree.command(name="positions", description="List all currently open positions")
    async def positions(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        data = await _fetch_db_metrics()

        if "error" in data:
            await interaction.followup.send(
                f"⚠️ Could not fetch positions: `{data['error']}`", ephemeral=True
            )
            return

        open_trades = data["open_trades"]
        if not open_trades:
            await interaction.followup.send("📭 No open positions right now.")
            return

        embed = discord.Embed(
            title=f"📋 Open Positions ({len(open_trades)})",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        for t in open_trades[:10]:  # Discord embed field limit
            direction_emoji = "📈" if t.direction == "long" else "📉"
            hold_mins = ""
            if t.entry_time:
                delta = datetime.now(timezone.utc) - t.entry_time.replace(tzinfo=timezone.utc)
                hold_mins = f" | {int(delta.total_seconds() // 60)}m"
            embed.add_field(
                name=f"{direction_emoji} {t.symbol} ({t.direction.upper()})",
                value=(
                    f"Entry: ${t.entry_price:,.2f}\n"
                    f"Size: {t.quantity:.4f} | ${t.notional_usd:,.0f}\n"
                    f"SL: ${t.stop_loss_price:,.2f} | TP: ${t.take_profit_price:,.2f}\n"
                    f"Strategy: {t.strategy_family}{hold_mins}"
                ),
                inline=True,
            )

        if len(open_trades) > 10:
            embed.set_footer(text=f"Showing 10 of {len(open_trades)} positions")

        await interaction.followup.send(embed=embed)

    @tree.command(name="trades", description="Show recent closed trades (last 5)")
    async def trades(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        data = await _fetch_db_metrics()

        if "error" in data:
            await interaction.followup.send(
                f"⚠️ Could not fetch trades: `{data['error']}`", ephemeral=True
            )
            return

        recent = data["recent_closed"]
        if not recent:
            await interaction.followup.send("📭 No closed trades yet.")
            return

        embed = discord.Embed(
            title="🔁 Recent Closed Trades",
            color=discord.Color.blurple(),
            timestamp=datetime.now(timezone.utc),
        )

        for t in recent:
            pnl = t.realized_pnl or 0.0
            emoji = "✅" if pnl > 0 else "❌"
            embed.add_field(
                name=f"{emoji} {t.symbol} ({t.direction.upper() if t.direction else '?'})",
                value=(
                    f"PnL: **${pnl:+,.2f}**\n"
                    f"Entry: ${t.entry_price:,.2f} → Exit: ${t.exit_price:,.2f}\n"
                    f"Reason: {t.exit_reason or 'n/a'} | {t.strategy_family or 'n/a'}"
                ),
                inline=True,
            )

        await interaction.followup.send(embed=embed)

    @tree.command(name="metrics", description="Performance metrics: win rate, profit factor, averages")
    async def metrics(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        data = await _fetch_db_metrics()

        if "error" in data:
            await interaction.followup.send(
                f"⚠️ Could not fetch metrics: `{data['error']}`", ephemeral=True
            )
            return

        embed = discord.Embed(
            title="📊 Performance Metrics",
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="🎯 Win Rate", value=f"{data['win_rate']*100:.1f}%", inline=True)
        embed.add_field(
            name="⚖️ Profit Factor",
            value=f"{data['profit_factor']:.2f}" if data["profit_factor"] else "N/A",
            inline=True,
        )
        embed.add_field(name="🔁 Total Trades", value=str(data["total_trades"]), inline=True)
        embed.add_field(name="✅ Avg Win", value=f"${data['avg_win']:+,.2f}", inline=True)
        embed.add_field(name="❌ Avg Loss", value=f"${data['avg_loss']:+,.2f}", inline=True)
        embed.add_field(name="📉 Drawdown", value=f"{data['drawdown_pct']:.2f}%", inline=True)

        # Strategy health summary
        sh = data.get("strategy_health", [])
        if sh:
            lines = []
            for row in sh[:4]:
                status_icon = "⏸️" if row.is_disabled else "▶️"
                lines.append(
                    f"{status_icon} **{row.strategy_name}**: "
                    f"WR {row.win_rate*100:.0f}% | PF {row.profit_factor:.2f}"
                )
            embed.add_field(
                name="🧠 Strategy Health", value="\n".join(lines), inline=False
            )

        await interaction.followup.send(embed=embed)


# ------------------------------------------------------------------ runner

async def run_discord_bot() -> None:
    """Start the Discord bot. Blocks until the bot disconnects."""
    from app.config.settings import get_settings
    from app.db.database import init_db

    settings = get_settings()

    if not settings.discord_bot_token:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN is not set. "
            "Create a bot at https://discord.com/developers/applications "
            "and add DISCORD_BOT_TOKEN to your .env file."
        )

    await init_db()

    client = TradingBotClient(guild_id=settings.discord_guild_id)

    logger.info("discord_bot_starting", guild_id=settings.discord_guild_id or "global")
    await client.start(settings.discord_bot_token)
