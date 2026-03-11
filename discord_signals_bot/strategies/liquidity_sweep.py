"""
Strategy 3: Liquidity Sweep Reversal
Trade stop-hunts / failed breakdowns / failed breakouts that reverse sharply.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from ..market_data.features import compute_all_features, get_atr_value, get_rsi_value
from .base import BaseStrategy
from .schema import SignalDirection, StrategySignal

logger = logging.getLogger(__name__)


class LiquiditySweepReversalStrategy(BaseStrategy):
    name = "liquidity_sweep_reversal"

    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        # Works best outside strong trending regimes
        if regime == "trending":
            return None

        df_15m = candles.get("15m")
        df_5m = candles.get("5m")

        if df_15m is None or df_5m is None:
            return None
        if len(df_15m) < 40 or len(df_5m) < 20:
            return None

        df_15m = compute_all_features(df_15m)
        df_5m = compute_all_features(df_5m)

        atr_15m = get_atr_value(df_15m)
        if atr_15m == 0:
            return None

        # Step 1: Find recent swing highs/lows on 15m (liquidity pools)
        lookback = df_15m.tail(30)
        swing_highs = lookback[lookback["swing_high"] == True]["high"]
        swing_lows = lookback[lookback["swing_low"] == True]["low"]

        if swing_highs.empty and swing_lows.empty:
            return None

        recent_swing_high = float(swing_highs.max()) if not swing_highs.empty else None
        recent_swing_low = float(swing_lows.min()) if not swing_lows.empty else None

        # Step 2: Detect sweep on 5m
        last_5m = df_5m.iloc[-1]
        wick_threshold = atr_15m * 0.3

        bullish_sweep = False
        bearish_sweep = False

        if recent_swing_low is not None:
            swept_below = last_5m["low"] < recent_swing_low - wick_threshold * 0.2
            reclaimed = last_5m["close"] > recent_swing_low
            strong_reversal = ((last_5m["close"] - last_5m["low"]) >
                               (last_5m["high"] - last_5m["low"]) * 0.6)
            if swept_below and reclaimed and strong_reversal:
                bullish_sweep = True

        if recent_swing_high is not None:
            swept_above = last_5m["high"] > recent_swing_high + wick_threshold * 0.2
            reclaimed = last_5m["close"] < recent_swing_high
            strong_reversal = ((last_5m["high"] - last_5m["close"]) >
                               (last_5m["high"] - last_5m["low"]) * 0.6)
            if swept_above and reclaimed and strong_reversal:
                bearish_sweep = True

        if not bullish_sweep and not bearish_sweep:
            return None

        direction = SignalDirection.LONG if bullish_sweep else SignalDirection.SHORT
        swept_level = recent_swing_low if bullish_sweep else recent_swing_high

        # Step 3: RSI extremes
        rsi_5m = get_rsi_value(df_5m)
        if direction == SignalDirection.LONG and rsi_5m > 45:
            return None
        if direction == SignalDirection.SHORT and rsi_5m < 55:
            return None

        # Step 4: Volume
        vol_ratio = last_5m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        high_volume = float(vol_ratio) > 1.5

        # Confidence
        wick_size = float(last_5m["high"] - last_5m["low"])
        wick_ratio = wick_size / (atr_15m + 1e-8)
        confidence = 0.0
        confidence += 0.30 * min(wick_ratio / 2.0, 1.0)
        confidence += 0.25 if high_volume else 0.10
        confidence += 0.20 if (rsi_5m < 35 or rsi_5m > 65) else 0.10
        confidence += 0.15
        confidence += 0.10 if regime in ("ranging", "compression") else 0.03
        confidence = min(confidence, 1.0)

        stop_distance = atr_15m * 1.0
        if direction == SignalDirection.LONG:
            stop_price = float(last_5m["low"]) - atr_15m * 0.3
            tp_price = current_price + stop_distance * 2.5
        else:
            stop_price = float(last_5m["high"]) + atr_15m * 0.3
            tp_price = current_price - stop_distance * 2.5

        reason_codes = []
        if bullish_sweep:
            reason_codes.append("bullish_sweep_detected")
        if bearish_sweep:
            reason_codes.append("bearish_sweep_detected")
        if high_volume:
            reason_codes.append("high_volume_sweep")
        if rsi_5m < 35:
            reason_codes.append("rsi_oversold")
        if rsi_5m > 65:
            reason_codes.append("rsi_overbought")

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="5m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 6),
            tp_price=round(tp_price, 6),
            validity_minutes=20,
            reason_codes=reason_codes,
            key_metrics={
                "swept_level": round(swept_level, 6) if swept_level else None,
                "rsi_5m": round(rsi_5m, 1),
                "volume_ratio": round(float(vol_ratio), 2),
                "wick_ratio": round(wick_ratio, 2),
                "regime": regime,
            },
            regime_context=regime,
        )
