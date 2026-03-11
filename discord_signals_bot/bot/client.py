"""
Discord bot client — the heart of the signals bot.

Two channels:
  • #entry-signals  — entry signal embeds with ✅ reaction to confirm entry
  • #exit-signals   — exit/TP/SL alerts for trades you've confirmed

Slash commands:
  /status           — show active trades and bot stats
  /close <symbol> <long|short>  — manually mark a trade as closed
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import discord
from discord import app_commands

from ..db.database import init_db
from ..market_data.binance_client import BinanceClient
from ..signals.generator import ExitEvent, SignalGenerator
from ..strategies.schema import TradeSignal
from .embeds import build_entry_embed, build_exit_embed, build_status_embed
from .trade_tracker import TradeTracker

logger = logging.getLogger(__name__)

CONFIRM_EMOJI = "✅"


class SignalsBot(discord.Client):
    def __init__(
        self,
        entry_channel_id: int,
        exit_channel_id: int,
        trader_user_id: int,
        scan_interval: int,
        generator: SignalGenerator,
        tracker: TradeTracker,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.guilds = True
        super().__init__(intents=intents)

        self.entry_channel_id = entry_channel_id
        self.exit_channel_id = exit_channel_id
        self.trader_user_id = trader_user_id
        self.scan_interval = scan_interval
        self.generator = generator
        self.tracker = tracker

        # Slash command tree
        self.tree = app_commands.CommandTree(self)

        self._scan_task: Optional[asyncio.Task] = None
        self._entry_channel: Optional[discord.TextChannel] = None
        self._exit_channel: Optional[discord.TextChannel] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup_hook(self) -> None:
        """Register slash commands."""
        self._register_commands()
        await self.tree.sync()
        logger.info("slash_commands_synced")

    async def on_ready(self) -> None:
        logger.info("bot_ready user=%s id=%s", self.user, self.user.id)

        self._entry_channel = self.get_channel(self.entry_channel_id)
        self._exit_channel = self.get_channel(self.exit_channel_id)

        if not self._entry_channel:
            logger.error("entry_channel_not_found id=%d", self.entry_channel_id)
        if not self._exit_channel:
            logger.error("exit_channel_not_found id=%d", self.exit_channel_id)

        # Start background scanner
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("scan_loop_started interval=%ds", self.scan_interval)

        if self._entry_channel:
            await self._entry_channel.send(
                embed=discord.Embed(
                    title="🤖 Signals Bot Online",
                    description=(
                        "Scanning BTC, ETH, SOL, XRP for leveraged trade signals.\n"
                        "React ✅ to any entry signal to confirm you've entered that trade.\n"
                        "Exit signals will appear in the exit channel for your confirmed trades."
                    ),
                    color=0x00C853,
                )
            )

    async def on_close(self) -> None:
        if self._scan_task:
            self._scan_task.cancel()
        logger.info("bot_closed")

    # ── Reaction handling ─────────────────────────────────────────────────────

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        """When the trader reacts ✅ to an entry signal, confirm the trade."""
        # Only care about the trader's own reactions
        if payload.user_id != self.trader_user_id:
            return
        if str(payload.emoji) != CONFIRM_EMOJI:
            return
        # Only in entry channel
        if payload.channel_id != self.entry_channel_id:
            return

        message_id = str(payload.message_id)
        trade = await self.tracker.confirm_entry(message_id)

        if trade is None:
            logger.debug("reaction_no_pending_signal message_id=%s", message_id)
            return

        logger.info(
            "trade_entry_confirmed symbol=%s dir=%s lev=%sx pct=%d%%",
            trade["symbol"], trade["direction"], trade["leverage"], trade["portfolio_pct"],
        )

        # Send confirmation DM or channel reply
        try:
            channel = self.get_channel(payload.channel_id)
            if channel:
                confirm_embed = discord.Embed(
                    title="✅ Trade Confirmed",
                    description=(
                        f"**{trade['symbol']}** {trade['direction'].upper()} @ "
                        f"`{float(trade['entry_price']):,.4f}` confirmed!\n"
                        f"⚡ {trade['leverage']}x leverage | 💼 {trade['portfolio_pct']}% portfolio\n\n"
                        f"I'll watch for TP/SL levels and post exit signals in <#{self.exit_channel_id}>."
                    ),
                    color=0x00C853,
                )
                msg = await channel.fetch_message(payload.message_id)
                await msg.reply(embed=confirm_embed, delete_after=30)
        except Exception as exc:
            logger.warning("confirm_reply_error %s", exc)

    # ── Background scanner ────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        """Main loop: scan for new entry signals + check exits."""
        await asyncio.sleep(5)  # brief startup delay

        while True:
            try:
                await self._run_entry_scan()
                await self._run_exit_check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("scan_loop_error %s", exc, exc_info=True)

            await asyncio.sleep(self.scan_interval)

    async def _run_entry_scan(self) -> None:
        """Run the signal generator and post any new entry signals."""
        if not self._entry_channel:
            return

        signals = await self.generator.scan()
        for signal in signals:
            await self._post_entry_signal(signal)

    async def _post_entry_signal(self, signal: TradeSignal) -> None:
        """Post an entry signal embed to #entry-signals."""
        embed = build_entry_embed(signal)
        try:
            msg = await self._entry_channel.send(embed=embed)
            # Add the ✅ reaction prompt
            await msg.add_reaction(CONFIRM_EMOJI)

            # Log to DB
            await self.tracker.log_signal(
                message_id=str(msg.id),
                symbol=signal.symbol,
                direction=signal.direction.value,
                score=signal.score,
                tier=signal.tier.value,
                leverage=signal.leverage,
                portfolio_pct=signal.portfolio_pct,
                strategy_names=signal.strategy_names,
                regime=signal.regime,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
            )
            logger.info(
                "entry_signal_posted symbol=%s dir=%s score=%.1f tier=%s lev=%sx",
                signal.symbol, signal.direction.value, signal.score,
                signal.tier.value, signal.leverage,
            )
        except discord.HTTPException as exc:
            logger.error("failed_to_post_entry_signal error=%s", exc)

    async def _run_exit_check(self) -> None:
        """Check active trades for TP/SL hits and post exit signals."""
        if not self._exit_channel:
            return

        active_trades = await self.tracker.get_active_trades()
        if not active_trades:
            return

        exit_events = await self.generator.check_exits(active_trades)
        for event in exit_events:
            await self._post_exit_signal(event)

    async def _post_exit_signal(self, event: ExitEvent) -> None:
        """Post an exit signal embed to #exit-signals."""
        embed = build_exit_embed(event)
        try:
            msg = await self._exit_channel.send(embed=embed)
            await msg.add_reaction(CONFIRM_EMOJI)

            # Update DB state
            if event.exit_type == "TP1":
                await self.tracker.mark_tp1_hit(event.trade_id)
                # Don't close yet — still watching for TP2
            else:
                # TP2, SL, or TECHNICAL → full close
                await self.tracker.close_trade(event.trade_id, event.exit_type, event.exit_price)

            logger.info(
                "exit_signal_posted trade_id=%d symbol=%s exit=%s price=%.4f",
                event.trade_id, event.symbol, event.exit_type, event.exit_price,
            )
        except discord.HTTPException as exc:
            logger.error("failed_to_post_exit_signal error=%s", exc)

    # ── Slash commands ────────────────────────────────────────────────────────

    def _register_commands(self) -> None:

        @self.tree.command(name="status", description="Show active trades and bot statistics")
        async def cmd_status(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.trader_user_id:
                await interaction.response.send_message("🚫 Not authorised.", ephemeral=True)
                return
            stats = await self.tracker.get_stats()
            active = await self.tracker.get_active_trades()
            embed = build_status_embed(stats, active)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(
            name="close",
            description="Manually mark a trade as closed (e.g. you exited on your own)"
        )
        @app_commands.describe(
            symbol="Trading pair, e.g. BTCUSDT",
            direction="long or short",
        )
        async def cmd_close(
            interaction: discord.Interaction,
            symbol: str,
            direction: str,
        ) -> None:
            if interaction.user.id != self.trader_user_id:
                await interaction.response.send_message("🚫 Not authorised.", ephemeral=True)
                return

            direction = direction.lower().strip()
            if direction not in ("long", "short"):
                await interaction.response.send_message(
                    "Direction must be `long` or `short`.", ephemeral=True
                )
                return

            symbol = symbol.upper().strip()
            closed = await self.tracker.manual_close(symbol, direction)
            if closed:
                await interaction.response.send_message(
                    f"✅ **{symbol} {direction.upper()}** marked as closed.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"⚠️ No active **{symbol} {direction.upper()}** trade found.", ephemeral=True
                )

        @self.tree.command(name="trades", description="List your currently active trades")
        async def cmd_trades(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.trader_user_id:
                await interaction.response.send_message("🚫 Not authorised.", ephemeral=True)
                return
            active = await self.tracker.get_active_trades()
            if not active:
                await interaction.response.send_message("No active trades.", ephemeral=True)
                return

            lines = []
            for t in active:
                tp1_flag = "✅" if t["tp1_hit"] else "⬜"
                lines.append(
                    f"**{t['symbol']}** {t['direction'].upper()} | "
                    f"Entry `${float(t['entry_price']):,.4f}` | "
                    f"{t['leverage']}x | {t['portfolio_pct']}% | "
                    f"TP1 {tp1_flag}"
                )
            await interaction.response.send_message(
                "\n".join(lines), ephemeral=True
            )
