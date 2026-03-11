"""
Signal Generator — the main analysis brain.

Runs on a periodic timer (default 60s). For every configured symbol:
  1. Fetch multi-timeframe candles from Binance
  2. Compute all technical features
  3. Classify market regime
  4. Run all 4 strategies
  5. Vote on long + short directions
  6. For any actionable vote (MEDIUM/STRONG/ELITE): build a TradeSignal
  7. Enforce cooldown (suppress duplicate signals within window)
  8. Return list of new TradeSignals to post

Exit monitoring:
  - Given a list of active trades, check current price against TP1/TP2/SL
  - Return exit events for trades that hit a level
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..market_data.binance_client import BinanceClient
from ..market_data.features import (
    compute_all_features, classify_regime,
    get_trend_direction, get_rsi_value, get_adx_value, get_macd_hist,
)
from ..strategies.breakout_retest import BreakoutRetestStrategy
from ..strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from ..strategies.schema import SignalDirection, SignalTier, TradeSignal, VoteResult
from ..strategies.trend_pullback import TrendPullbackStrategy
from ..strategies.volatility_expansion import VolatilityExpansionStrategy
from ..voting.engine import VotingEngine
from . import leverage_calc

logger = logging.getLogger(__name__)


@dataclass
class ExitEvent:
    """Triggered when an active trade hits a TP or SL level."""
    trade_id: int
    symbol: str
    direction: SignalDirection
    exit_type: str          # "TP1" | "TP2" | "SL" | "TECHNICAL"
    exit_price: float
    entry_price: float
    leverage: int
    portfolio_pct: int
    partial: bool           # True = take 50%, False = close everything
    message: str


def _get_session_score(utc_hour: int, session_cfg: Dict) -> float:
    """Return trading session quality score for the current UTC hour."""
    for name, params in session_cfg.items():
        if name == "default":
            continue
        if isinstance(params, dict) and "start" in params and "end" in params:
            start, end = params["start"], params["end"]
            if start <= utc_hour < end:
                return float(params.get("score", 0.7))
    return float(session_cfg.get("default", {}).get("score", 0.7))


class SignalGenerator:
    def __init__(self, cfg: Dict[str, Any], binance: BinanceClient) -> None:
        self._cfg = cfg
        self._binance = binance

        # Strategy instances
        self._strategies = [
            TrendPullbackStrategy(),
            BreakoutRetestStrategy(),
            LiquiditySweepReversalStrategy(),
            VolatilityExpansionStrategy(),
        ]

        self._voting_engine = VotingEngine(cfg["voting"])
        self._leverage_cfg = cfg["leverage"]
        self._exit_cfg = cfg["exits"]
        self._symbols: List[str] = cfg["symbols"]
        self._timeframes: List[str] = cfg["timeframes"]
        self._candle_limits: Dict[str, int] = cfg["candle_limits"]
        self._session_cfg: Dict = cfg.get("sessions", {})
        self._score_cfg: Dict = cfg.get("scoring", {})
        self._regime_strategy_map: Dict = cfg.get("regime", {}).get("strategy_map", {})
        self._cooldown_minutes: int = cfg.get("signal_cooldown_minutes", 30)
        self._min_score: float = self._score_cfg.get("no_trade_max", 40.0)

        # Cooldown tracker: (symbol, direction) -> last signal time
        self._last_signal_times: Dict[Tuple[str, str], datetime] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Entry scan
    # ──────────────────────────────────────────────────────────────────────────

    async def scan(self) -> List[TradeSignal]:
        """Scan all symbols for entry signals. Returns new actionable signals."""
        tasks = [self._scan_symbol(sym) for sym in self._symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[TradeSignal] = []
        for sym, result in zip(self._symbols, results):
            if isinstance(result, Exception):
                logger.error("scan_error symbol=%s error=%s", sym, result)
            elif result:
                signals.extend(result)
        return signals

    async def _scan_symbol(self, symbol: str) -> List[TradeSignal]:
        """Run full analysis pipeline for one symbol."""
        # 1. Fetch candles
        candles = await self._binance.get_all_timeframes(
            symbol, self._timeframes, self._candle_limits
        )

        # Validate we have enough data
        df_5m = candles.get("5m")
        df_15m = candles.get("15m")
        df_4h = candles.get("4h")
        if df_5m is None or df_15m is None or df_4h is None:
            logger.warning("insufficient_candles symbol=%s", symbol)
            return []
        if len(df_5m) < 30 or len(df_15m) < 40 or len(df_4h) < 50:
            logger.warning("not_enough_candles symbol=%s", symbol)
            return []

        # 2. Compute features on all timeframes
        candles_with_features = {}
        for tf, df in candles.items():
            if df is not None and len(df) >= 30:
                candles_with_features[tf] = compute_all_features(df)
            else:
                candles_with_features[tf] = df

        # Current price from last 5m close
        current_price = float(candles_with_features["5m"].iloc[-1]["close"])

        # 3. Regime classification
        regime = classify_regime(candles_with_features.get("4h"), candles_with_features.get("15m"))

        # 4. Run strategies (filter by regime)
        allowed = self._regime_strategy_map.get(regime, [
            "trend_pullback", "breakout_retest", "volatility_expansion"
        ])
        strategy_signals = []
        for strat in self._strategies:
            if strat.name not in allowed:
                continue
            try:
                sig = await strat.evaluate(symbol, candles_with_features, current_price, regime)
                if sig is not None:
                    strategy_signals.append(sig)
            except Exception as exc:
                logger.warning("strategy_error strategy=%s symbol=%s error=%s",
                               strat.name, symbol, exc)

        if not strategy_signals:
            return []

        # 5. Build context for voting
        df_5m_feat = candles_with_features["5m"]
        df_4h_feat = candles_with_features["4h"]
        df_15m_feat = candles_with_features["15m"]

        rsi = get_rsi_value(df_5m_feat)
        adx = get_adx_value(df_4h_feat)
        macd_hist = get_macd_hist(df_5m_feat)
        trend_4h = get_trend_direction(df_4h_feat, "ema_20", "ema_50")
        trend_15m = get_trend_direction(df_15m_feat, "ema_20", "ema_50")
        vol_state = str(df_15m_feat.iloc[-1].get("volatility_state", "medium"))

        now_utc = datetime.now(timezone.utc)
        session_score = _get_session_score(now_utc.hour, self._session_cfg)

        # Fetch 24h ticker for volume/liquidity scoring
        ticker = await self._binance.get_ticker_24h(symbol)
        volume_24h_usd = 0.0
        if ticker:
            try:
                volume_24h_usd = float(ticker.get("quoteVolume", 0))
            except (TypeError, ValueError):
                pass

        context = {
            "trend_4h": trend_4h,
            "trend_15m": trend_15m,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "adx": adx,
            "volatility_state": vol_state,
            "volume_24h_usd": volume_24h_usd,
            "session_score": session_score,
            "spread_pct": 0.02,   # conservative estimate (Binance perpetuals typically ~0.01-0.02%)
        }

        # 6. Vote on both directions
        new_signals: List[TradeSignal] = []
        for direction in (SignalDirection.LONG, SignalDirection.SHORT):
            vote: VoteResult = self._voting_engine.vote(
                symbol, strategy_signals, direction, context
            )

            if vote.tier == SignalTier.NO_TRADE:
                continue

            # Cooldown check
            cooldown_key = (symbol, direction.value)
            last_time = self._last_signal_times.get(cooldown_key)
            if last_time and (now_utc - last_time) < timedelta(minutes=self._cooldown_minutes):
                logger.debug("signal_on_cooldown symbol=%s dir=%s", symbol, direction.value)
                continue

            # Build TradeSignal
            trade_signal = self._build_trade_signal(vote, regime)
            if trade_signal:
                self._last_signal_times[cooldown_key] = now_utc
                new_signals.append(trade_signal)

        return new_signals

    def _build_trade_signal(self, vote: VoteResult, regime: str) -> Optional[TradeSignal]:
        if vote.entry_price is None or vote.stop_price is None:
            return None

        lev_rec = leverage_calc.calculate(vote.final_score, vote.tier, self._leverage_cfg)

        # Recalculate TP1 / TP2 using configured R-multiples
        risk = abs(vote.entry_price - vote.stop_price)
        sign = 1.0 if vote.direction == SignalDirection.LONG else -1.0
        tp1 = vote.entry_price + sign * risk * self._exit_cfg["tp1_r_multiple"]
        tp2 = vote.entry_price + sign * risk * self._exit_cfg["tp2_r_multiple"]

        strategy_names = [s.strategy_family for s in vote.contributing_signals]

        return TradeSignal(
            symbol=vote.symbol,
            direction=vote.direction,
            tier=vote.tier,
            score=vote.final_score,
            entry_price=vote.entry_price,
            stop_price=vote.stop_price,
            tp1_price=round(tp1, 6),
            tp2_price=round(tp2, 6),
            leverage=lev_rec.leverage,
            portfolio_pct=lev_rec.portfolio_pct,
            strategy_names=strategy_names,
            reason_codes=vote.reason_codes,
            regime=regime,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Exit monitoring
    # ──────────────────────────────────────────────────────────────────────────

    async def check_exits(self, active_trades: List[Dict]) -> List[ExitEvent]:
        """
        Check if any active trade has hit TP1, TP2, or SL.

        Args:
            active_trades: list of dicts from TradeTracker (DB rows as dicts)

        Returns:
            list of ExitEvent objects to post to the exit channel
        """
        if not active_trades:
            return []

        # Group trades by symbol to minimise API calls
        symbols_needed = list({t["symbol"] for t in active_trades})
        price_map: Dict[str, float] = {}

        for sym in symbols_needed:
            df = await self._binance.get_klines(sym, "1m", limit=3)
            if df is not None and not df.empty:
                price_map[sym] = float(df.iloc[-1]["close"])

        exit_events: List[ExitEvent] = []
        for trade in active_trades:
            sym = trade["symbol"]
            current = price_map.get(sym)
            if current is None:
                continue

            event = self._evaluate_exit(trade, current)
            if event:
                exit_events.append(event)

        return exit_events

    def _evaluate_exit(self, trade: Dict, current_price: float) -> Optional[ExitEvent]:
        direction = trade["direction"]
        entry = float(trade["entry_price"])
        stop = float(trade["stop_price"])
        tp1 = float(trade["tp1_price"])
        tp2 = float(trade["tp2_price"])
        tp1_hit = bool(trade.get("tp1_hit", False))

        is_long = direction == "long"

        # After TP1 is hit, the effective stop moves to breakeven
        effective_stop = entry if tp1_hit else stop

        # ── Check SL ─────────────────────────────────────────────────────────
        sl_hit = (is_long and current_price <= effective_stop) or \
                 (not is_long and current_price >= effective_stop)
        if sl_hit:
            label = "STOP LOSS (Breakeven)" if tp1_hit else "STOP LOSS"
            return ExitEvent(
                trade_id=trade["id"],
                symbol=trade["symbol"],
                direction=SignalDirection(direction),
                exit_type="SL",
                exit_price=current_price,
                entry_price=entry,
                leverage=trade["leverage"],
                portfolio_pct=trade["portfolio_pct"],
                partial=False,
                message=f"🛑 **{label}** hit at `${current_price:,.4f}`",
            )

        # ── Check TP2 (only if TP1 already hit) ──────────────────────────────
        if tp1_hit:
            tp2_hit = (is_long and current_price >= tp2) or \
                      (not is_long and current_price <= tp2)
            if tp2_hit:
                return ExitEvent(
                    trade_id=trade["id"],
                    symbol=trade["symbol"],
                    direction=SignalDirection(direction),
                    exit_type="TP2",
                    exit_price=current_price,
                    entry_price=entry,
                    leverage=trade["leverage"],
                    portfolio_pct=trade["portfolio_pct"],
                    partial=False,
                    message=f"🎯 **TAKE PROFIT 2 — Close remaining position** at `${current_price:,.4f}`",
                )

        # ── Check TP1 ─────────────────────────────────────────────────────────
        if not tp1_hit:
            tp1_reached = (is_long and current_price >= tp1) or \
                          (not is_long and current_price <= tp1)
            if tp1_reached:
                return ExitEvent(
                    trade_id=trade["id"],
                    symbol=trade["symbol"],
                    direction=SignalDirection(direction),
                    exit_type="TP1",
                    exit_price=current_price,
                    entry_price=entry,
                    leverage=trade["leverage"],
                    portfolio_pct=trade["portfolio_pct"],
                    partial=True,
                    message=(
                        f"✅ **TAKE PROFIT 1 — Close 50% of position** at `${current_price:,.4f}`\n"
                        f"🔄 Move stop loss to breakeven (`${entry:,.4f}`)"
                    ),
                )

        return None
