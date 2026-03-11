"""
Discord embed builders for entry and exit signals.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import discord

from ..strategies.schema import SignalDirection, SignalTier, TradeSignal
from ..signals.generator import ExitEvent

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_LONG = 0x00C853        # bright green
COLOR_SHORT = 0xFF1744       # bright red
COLOR_TP1 = 0x00BCD4         # cyan
COLOR_TP2 = 0x00E676         # green
COLOR_SL = 0xFF6D00          # orange
COLOR_TECHNICAL = 0xAA00FF   # purple
COLOR_STATUS = 0x607D8B      # blue-grey
COLOR_TIER: Dict[str, int] = {
    "medium": 0xFFC400,      # amber
    "strong": 0x00E676,      # green
    "elite": 0xE040FB,       # purple/gold
}

# ── Tier labels ───────────────────────────────────────────────────────────────
TIER_EMOJI: Dict[str, str] = {
    "medium": "🟡",
    "strong": "🟢",
    "elite": "💎",
}

DIRECTION_EMOJI: Dict[str, str] = {
    "long": "📈",
    "short": "📉",
}


def _price_fmt(price: float, symbol: str) -> str:
    """Format price with appropriate decimal places based on magnitude."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    else:
        return f"${price:,.6f}"


def _symbol_display(symbol: str) -> str:
    """BTCUSDT → BTC/USDT"""
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    return symbol


def build_entry_embed(signal: TradeSignal) -> discord.Embed:
    """Build the rich embed for an entry signal."""
    dir_str = signal.direction.value
    tier_str = signal.tier.value
    sym_display = _symbol_display(signal.symbol)

    color = COLOR_LONG if dir_str == "long" else COLOR_SHORT
    dir_emoji = DIRECTION_EMOJI.get(dir_str, "")
    tier_emoji = TIER_EMOJI.get(tier_str, "")

    title = f"{dir_emoji} {sym_display} — {dir_str.upper()} {tier_emoji}"
    description = (
        f"**{tier_str.upper()} SIGNAL** • Score: `{signal.score:.1f}/100`\n"
        f"React with ✅ below to confirm you've entered this trade."
    )

    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=signal.signal_time,
    )

    # ── Core price levels ─────────────────────────────────────────────────────
    embed.add_field(
        name="📍 Entry",
        value=_price_fmt(signal.entry_price, signal.symbol),
        inline=True,
    )
    embed.add_field(
        name="🛑 Stop Loss",
        value=_price_fmt(signal.stop_price, signal.symbol),
        inline=True,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)  # spacer

    embed.add_field(
        name="✅ TP1 (50% exit)",
        value=_price_fmt(signal.tp1_price, signal.symbol),
        inline=True,
    )
    embed.add_field(
        name="🎯 TP2 (close rest)",
        value=_price_fmt(signal.tp2_price, signal.symbol),
        inline=True,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)

    # ── Risk metrics ──────────────────────────────────────────────────────────
    embed.add_field(
        name="⚡ Leverage",
        value=f"**{signal.leverage}x**",
        inline=True,
    )
    embed.add_field(
        name="💼 Portfolio Size",
        value=f"**{signal.portfolio_pct}%**",
        inline=True,
    )
    embed.add_field(
        name="📐 R:R",
        value=f"TP1 {signal.r_to_tp1:.1f}R  |  TP2 {signal.r_to_tp2:.1f}R",
        inline=True,
    )

    # ── Analysis info ─────────────────────────────────────────────────────────
    embed.add_field(
        name="🧠 Strategies",
        value=", ".join(s.replace("_", " ").title() for s in signal.strategy_names),
        inline=True,
    )
    embed.add_field(
        name="🌐 Regime",
        value=signal.regime.replace("_", " ").title(),
        inline=True,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)

    # ── Reason codes (compact) ────────────────────────────────────────────────
    if signal.reason_codes:
        clean_codes = [c.split(":")[-1].replace("_", " ") for c in signal.reason_codes[:6]]
        embed.add_field(
            name="📋 Signals",
            value=" • ".join(clean_codes),
            inline=False,
        )

    embed.set_footer(text=f"🟢 React ✅ to confirm entry  |  {signal.symbol}")
    return embed


def build_exit_embed(event: ExitEvent, entry_signal_url: Optional[str] = None) -> discord.Embed:
    """Build the rich embed for an exit / take-profit signal."""
    dir_str = event.direction.value
    sym_display = _symbol_display(event.symbol)

    # Color by exit type
    color_map = {"TP1": COLOR_TP1, "TP2": COLOR_TP2, "SL": COLOR_SL, "TECHNICAL": COLOR_TECHNICAL}
    color = color_map.get(event.exit_type, COLOR_STATUS)

    # PnL calc (rough — actual depends on leverage and exact fill)
    price_change_pct = ((event.exit_price - event.entry_price) / event.entry_price) * 100
    if dir_str == "short":
        price_change_pct = -price_change_pct
    leveraged_pnl_pct = price_change_pct * event.leverage

    pnl_emoji = "✅" if leveraged_pnl_pct >= 0 else "🔴"
    pnl_str = f"{leveraged_pnl_pct:+.1f}% (leveraged)"

    title = f"{'⚠️' if event.exit_type == 'SL' else '🎯'} EXIT SIGNAL — {sym_display} {dir_str.upper()}"
    description = event.message

    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    embed.add_field(name="📍 Entry Was", value=_price_fmt(event.entry_price, event.symbol), inline=True)
    embed.add_field(name="🏁 Exit Now", value=_price_fmt(event.exit_price, event.symbol), inline=True)
    embed.add_field(name=f"{pnl_emoji} Est. PnL", value=pnl_str, inline=True)

    embed.add_field(name="⚡ Leverage Used", value=f"{event.leverage}x", inline=True)
    embed.add_field(name="💼 Portfolio Deployed", value=f"{event.portfolio_pct}%", inline=True)
    embed.add_field(name="📊 Exit Type", value=event.exit_type, inline=True)

    if event.partial:
        embed.add_field(
            name="ℹ️ Partial Exit",
            value="Close **50%** of your position. Move stop to **breakeven**.",
            inline=False,
        )
    else:
        embed.add_field(
            name="ℹ️ Full Exit",
            value="Close **100%** of your remaining position.",
            inline=False,
        )

    if entry_signal_url:
        embed.add_field(name="🔗 Entry Signal", value=entry_signal_url, inline=False)

    embed.set_footer(text=f"React ✅ to acknowledge  |  {event.symbol}")
    return embed


def build_status_embed(stats: dict, active_trades: list) -> discord.Embed:
    """Build a status/health embed for the /status command."""
    embed = discord.Embed(
        title="📊 Signals Bot Status",
        color=COLOR_STATUS,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="📡 Total Signals", value=str(stats.get("total_signals", 0)), inline=True)
    embed.add_field(name="✅ Confirmed Entries", value=str(stats.get("confirmed_entries", 0)), inline=True)
    embed.add_field(name="📂 Active Trades", value=str(stats.get("active_trades", 0)), inline=True)
    embed.add_field(name="📁 Closed Trades", value=str(stats.get("closed_trades", 0)), inline=True)

    if active_trades:
        trade_lines = []
        for t in active_trades[:5]:
            sym = _symbol_display(t["symbol"])
            trade_lines.append(
                f"• **{sym}** {t['direction'].upper()} @ {_price_fmt(float(t['entry_price']), t['symbol'])} "
                f"| {t['leverage']}x | TP1{'✅' if t['tp1_hit'] else '⬜'}"
            )
        embed.add_field(
            name="🔥 Open Positions",
            value="\n".join(trade_lines),
            inline=False,
        )

    embed.set_footer(text="Last updated")
    return embed
