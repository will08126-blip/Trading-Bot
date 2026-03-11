"""
Strategy 2: Breakout Retest
Trade valid breakouts after price retests the broken level.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from ..market_data.features import (
    compute_all_features, get_atr_value, get_rsi_value,
    get_trend_direction,
)
from .base import BaseStrategy
from .schema import SignalDirection, StrategySignal

logger = logging.getLogger(__name__)


class BreakoutRetestStrategy(BaseStrategy):
    name = "breakout_retest"
    _structure_lookback = 30

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
        atr_15m = get_atr_value(df_15m)
        if atr_15m == 0:
            return None

        # Step 1: Structure high/low over lookback (exclude last candle)
        lookback_df = df_15m.tail(self._structure_lookback + 1).iloc[:-1]
        struct_high = lookback_df["high"].max()
        struct_low = lookback_df["low"].min()

        # Step 2: Detect breakout
        breakout_threshold = atr_15m * 0.3
        bullish_breakout = last_15m["close"] > struct_high + breakout_threshold
        bearish_breakout = last_15m["close"] < struct_low - breakout_threshold

        vol_ratio = last_15m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        volume_expanding = float(vol_ratio) > 1.2

        if not bullish_breakout and not bearish_breakout:
            return None

        direction = SignalDirection.LONG if bullish_breakout else SignalDirection.SHORT
        broken_level = struct_high if bullish_breakout else struct_low

        # Overextension check
        price_extension = abs(last_15m["high" if bullish_breakout else "low"] - broken_level)
        if price_extension > atr_15m * 3:
            return None

        # Step 3: Retest proximity
        retest_proximity_pct = 0.003
        in_retest = abs(current_price - broken_level) / broken_level < retest_proximity_pct * 5
        if direction == SignalDirection.LONG:
            held_above = current_price > broken_level * (1 - retest_proximity_pct)
        else:
            held_above = current_price < broken_level * (1 + retest_proximity_pct)

        if not in_retest:
            return None

        # Step 4: 5m confirmation candle
        last_5m = df_5m.iloc[-1]
        if direction == SignalDirection.LONG:
            confirmation = (last_5m["close"] > last_5m["open"] and
                            last_5m["close"] > broken_level)
        else:
            confirmation = (last_5m["close"] < last_5m["open"] and
                            last_5m["close"] < broken_level)

        # Higher TF alignment bonus
        htf_aligned = False
        if df_4h is not None:
            trend_4h = get_trend_direction(df_4h, "ema_20", "ema_50")
            htf_aligned = ((trend_4h == "bullish" and direction == SignalDirection.LONG) or
                           (trend_4h == "bearish" and direction == SignalDirection.SHORT))

        # RSI extremes filter
        rsi_5m = get_rsi_value(df_5m)
        if direction == SignalDirection.LONG and rsi_5m > 80:
            return None
        if direction == SignalDirection.SHORT and rsi_5m < 20:
            return None

        # Confidence
        confidence = 0.0
        confidence += 0.25 if volume_expanding else 0.10
        confidence += 0.20 if confirmation else 0.05
        confidence += 0.20 if held_above else 0.0
        confidence += 0.15 if htf_aligned else 0.05
        confidence += 0.10 * (float(vol_ratio) / 3.0 if float(vol_ratio) < 3 else 1.0)
        confidence = min(confidence, 1.0)

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

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="15m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 6),
            tp_price=round(tp_price, 6),
            validity_minutes=45,
            reason_codes=reason_codes,
            key_metrics={
                "broken_level": round(broken_level, 6),
                "struct_high": round(struct_high, 6),
                "struct_low": round(struct_low, 6),
                "volume_ratio": round(float(vol_ratio), 2),
                "atr_15m": round(atr_15m, 6),
                "htf_aligned": htf_aligned,
                "rsi_5m": round(rsi_5m, 1),
                "regime": regime,
            },
            regime_context=regime,
        )
