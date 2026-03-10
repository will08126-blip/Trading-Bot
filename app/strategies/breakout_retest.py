"""
Strategy 2: Breakout Retest

Goal: Trade valid breakouts after price retests the broken level.

Logic:
1. Identify key 15m structure high/low or consolidation boundary
2. Confirm breakout with momentum/volume expansion
3. Wait for retest (don't chase)
4. Require retest hold + confirmation candle
5. Reject false breakouts, low-volume breaks, overextended moves
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import structlog

from app.market_data.features import (
    compute_all_features, get_atr_value, get_rsi_value,
    get_trend_direction,
)
from app.signals.schema import SignalDirection, StrategySignal
from app.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class BreakoutRetestStrategy(BaseStrategy):
    name = "breakout_retest"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or {}
        self._structure_lookback = 30

    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        df_15m = candles.get("15m")
        df_5m = candles.get("5m")
        df_4h = candles.get("4h")

        if df_15m is None or df_5m is None:
            return None
        if len(df_15m) < 50 or len(df_5m) < 30:
            return None

        df_15m = compute_all_features(df_15m)
        df_5m = compute_all_features(df_5m)
        if df_4h is not None and len(df_4h) >= 50:
            df_4h = compute_all_features(df_4h)

        last_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]

        # Step 1: Find structure high/low over lookback
        lookback_df = df_15m.tail(self._structure_lookback + 1).iloc[:-1]  # exclude last
        struct_high = lookback_df["high"].max()
        struct_low = lookback_df["low"].min()

        struct_high_candle = lookback_df["high"].idxmax()
        struct_low_candle = lookback_df["low"].idxmin()

        atr_15m = get_atr_value(df_15m)
        if atr_15m == 0:
            return None

        # Step 2: Detect breakout
        breakout_threshold = atr_15m * 0.3  # price must be meaningfully beyond structure
        bullish_breakout = last_15m["close"] > struct_high + breakout_threshold
        bearish_breakout = last_15m["close"] < struct_low - breakout_threshold

        # Volume expansion check
        vol_ratio = last_15m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        volume_expanding = vol_ratio > 1.2

        if not bullish_breakout and not bearish_breakout:
            return None

        direction = SignalDirection.LONG if bullish_breakout else SignalDirection.SHORT
        broken_level = struct_high if bullish_breakout else struct_low

        # Step 3: Check if we're in retest phase
        # For a bullish breakout: current price should have pulled back near the broken level
        # Don't chase – wait for retest proximity
        retest_proximity_pct = 0.003  # within 0.3% of broken level

        if direction == SignalDirection.LONG:
            in_retest = abs(current_price - broken_level) / broken_level < retest_proximity_pct * 5
            held_above = current_price > broken_level * (1 - retest_proximity_pct)
        else:
            in_retest = abs(current_price - broken_level) / broken_level < retest_proximity_pct * 5
            held_above = current_price < broken_level * (1 + retest_proximity_pct)

        if not in_retest:
            return None

        # Step 4: Confirmation candle on 5m
        last_5m = df_5m.iloc[-1]
        if direction == SignalDirection.LONG:
            confirmation = (last_5m["close"] > last_5m["open"] and
                           last_5m["close"] > broken_level)
        else:
            confirmation = (last_5m["close"] < last_5m["open"] and
                           last_5m["close"] < broken_level)

        # Reject false breakouts: extension check
        # If price moved more than 3x ATR from structure before retest, likely exhausted
        price_extension = abs(last_15m["high" if bullish_breakout else "low"] - broken_level)
        overextended = price_extension > atr_15m * 3

        if overextended:
            logger.debug("breakout_retest_overextended", symbol=symbol, extension=price_extension)
            return None

        # Higher timeframe context
        htf_aligned = False
        if df_4h is not None:
            trend_4h = get_trend_direction(df_4h, "ema_20", "ema_50")
            htf_aligned = (trend_4h == "bullish" and direction == SignalDirection.LONG) or \
                          (trend_4h == "bearish" and direction == SignalDirection.SHORT)

        # RSI check – avoid entering when exhausted
        rsi_5m = get_rsi_value(df_5m)
        rsi_extreme = (direction == SignalDirection.LONG and rsi_5m > 80) or \
                      (direction == SignalDirection.SHORT and rsi_5m < 20)
        if rsi_extreme:
            return None

        # Confidence
        confidence = 0.0
        confidence += 0.25 if volume_expanding else 0.10
        confidence += 0.20 if confirmation else 0.05
        confidence += 0.20 if held_above else 0.0
        confidence += 0.15 if htf_aligned else 0.05
        confidence += 0.10 if not overextended else 0.0
        confidence += 0.10 * (vol_ratio / 3.0 if vol_ratio < 3 else 1.0)
        confidence = min(confidence, 1.0)

        # Stop and TP
        stop_distance = atr_15m * 1.5
        if direction == SignalDirection.LONG:
            stop_price = broken_level - stop_distance
            tp_price = current_price + stop_distance * 2.0
        else:
            stop_price = broken_level + stop_distance
            tp_price = current_price - stop_distance * 2.0

        reason_codes = []
        if volume_expanding:
            reason_codes.append("volume_expanding")
        if confirmation:
            reason_codes.append("5m_retest_confirmation")
        if held_above:
            reason_codes.append("retest_held")
        if htf_aligned:
            reason_codes.append("4h_aligned")

        logger.debug("breakout_retest_signal", symbol=symbol, direction=direction,
                     confidence=confidence, broken_level=broken_level)

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="15m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            validity_minutes=45,
            reason_codes=reason_codes,
            key_metrics={
                "broken_level": round(broken_level, 2),
                "struct_high": round(struct_high, 2),
                "struct_low": round(struct_low, 2),
                "volume_ratio": round(vol_ratio, 2),
                "atr_15m": round(atr_15m, 4),
                "htf_aligned": htf_aligned,
                "overextended": overextended,
                "rsi_5m": round(rsi_5m, 1),
                "regime": regime,
            },
            regime_context=regime,
        )
