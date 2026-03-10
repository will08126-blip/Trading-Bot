"""
Strategy 3: Liquidity Sweep Reversal

Goal: Trade stop hunts / failed breakdowns / failed breakouts that reverse.

Logic:
1. Identify recent swing highs/lows (liquidity pools)
2. Detect sweep beyond prior level (wick beyond, then rejection)
3. Require reclaim and reversal confirmation
4. Strong reversal candle required
5. Only active in suitable regimes (ranging/reversal-friendly)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import structlog

from app.market_data.features import (
    compute_all_features, get_atr_value, get_rsi_value,
)
from app.signals.schema import SignalDirection, StrategySignal
from app.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class LiquiditySweepReversalStrategy(BaseStrategy):
    name = "liquidity_sweep_reversal"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or {}

    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        # This strategy performs best in ranging/reversal regimes
        # In strong trending regimes, disable it
        if regime in ("trending",):
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

        recent_swing_high = swing_highs.max() if not swing_highs.empty else None
        recent_swing_low = swing_lows.min() if not swing_lows.empty else None

        # Step 2: Detect sweep on 5m
        last_5m = df_5m.iloc[-1]
        prev_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else last_5m

        wick_threshold = atr_15m * 0.3  # minimum wick to confirm sweep

        bullish_sweep = False  # swept below swing low then reversed
        bearish_sweep = False  # swept above swing high then reversed

        if recent_swing_low is not None:
            # Bullish sweep: last 5m wick went below swing low but closed above
            swept_below = last_5m["low"] < recent_swing_low - wick_threshold * 0.2
            reclaimed = last_5m["close"] > recent_swing_low
            strong_reversal = (last_5m["close"] - last_5m["low"]) > (last_5m["high"] - last_5m["low"]) * 0.6
            if swept_below and reclaimed and strong_reversal:
                bullish_sweep = True

        if recent_swing_high is not None:
            # Bearish sweep: wick went above swing high but closed below
            swept_above = last_5m["high"] > recent_swing_high + wick_threshold * 0.2
            reclaimed = last_5m["close"] < recent_swing_high
            strong_reversal = (last_5m["high"] - last_5m["close"]) > (last_5m["high"] - last_5m["low"]) * 0.6
            if swept_above and reclaimed and strong_reversal:
                bearish_sweep = True

        if not bullish_sweep and not bearish_sweep:
            return None

        direction = SignalDirection.LONG if bullish_sweep else SignalDirection.SHORT
        swept_level = recent_swing_low if bullish_sweep else recent_swing_high

        # Step 3: RSI extremes confirmation
        rsi_5m = get_rsi_value(df_5m)
        if direction == SignalDirection.LONG and rsi_5m > 45:
            return None  # not oversold enough for sweep reversal
        if direction == SignalDirection.SHORT and rsi_5m < 55:
            return None

        # Step 4: Volume on sweep candle
        vol_ratio = last_5m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        high_volume = vol_ratio > 1.5

        # Confidence
        confidence = 0.0
        wick_size = abs(last_5m["high"] - last_5m["low"])
        wick_ratio = wick_size / (atr_15m + 0.0001)
        confidence += 0.30 * min(wick_ratio / 2.0, 1.0)    # bigger sweep = higher confidence
        confidence += 0.25 if high_volume else 0.10
        confidence += 0.20 if rsi_5m < 35 or rsi_5m > 65 else 0.10  # RSI extreme
        confidence += 0.15  # base for finding a sweep pattern
        confidence += 0.10 if regime in ("ranging", "compression") else 0.03

        confidence = min(confidence, 1.0)

        # Rejection near sweep: tight stop
        stop_distance = atr_15m * 1.0  # tighter stop for sweep reversals
        if direction == SignalDirection.LONG:
            stop_price = last_5m["low"] - atr_15m * 0.3
            tp_price = current_price + stop_distance * 2.5
        else:
            stop_price = last_5m["high"] + atr_15m * 0.3
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

        logger.debug("liquidity_sweep_signal", symbol=symbol, direction=direction,
                     confidence=confidence, swept_level=swept_level)

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="5m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            validity_minutes=20,  # shorter validity for reversal plays
            reason_codes=reason_codes,
            key_metrics={
                "swept_level": round(swept_level, 2) if swept_level else None,
                "rsi_5m": round(rsi_5m, 1),
                "volume_ratio": round(vol_ratio, 2),
                "wick_ratio": round(wick_ratio, 2),
                "regime": regime,
            },
            regime_context=regime,
        )
