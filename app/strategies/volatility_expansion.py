"""
Strategy 4: Volatility Expansion

Goal: Trade range compression transitioning to directional expansion.

Logic:
1. Detect volatility compression (low ATR, tight Bollinger Bands)
2. Identify breakout with volume/range expansion
3. Confirm not already overextended
4. Require directional context alignment where applicable
5. Reject immediate exhaustion patterns
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import structlog

from app.market_data.features import (
    compute_all_features, get_atr_value, get_adx_value, get_rsi_value,
    get_trend_direction,
)
from app.signals.schema import SignalDirection, StrategySignal
from app.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class VolatilityExpansionStrategy(BaseStrategy):
    name = "volatility_expansion"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or {}
        self._compression_lookback = 20
        self._compression_atr_ratio = 0.7  # ATR must be below 70% of its MA

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

        # Step 1: Detect compression on 15m
        atr_15m = get_atr_value(df_15m)
        if atr_15m == 0:
            return None

        # Bollinger Band width compression
        last_15m = df_15m.iloc[-1]
        bb_width = last_15m.get("bb_width", None)
        volatility_ratio = last_15m.get("volatility_ratio", 1.0)
        if pd.isna(bb_width):
            return None

        # Check compression history: BB width must be near recent lows
        bb_width_series = df_15m["bb_width"].tail(self._compression_lookback)
        bb_width_pct = (bb_width - bb_width_series.min()) / (bb_width_series.max() - bb_width_series.min() + 1e-8)
        in_compression = bb_width_pct < 0.3  # BB width in bottom 30% of recent range

        volatility_compressed = (
            pd.notna(volatility_ratio) and
            volatility_ratio < self._compression_atr_ratio
        )

        if not (in_compression or volatility_compressed):
            return None

        # Step 2: Detect expansion breakout on 5m
        df_5m_tail = df_5m.tail(5)
        if len(df_5m_tail) < 3:
            return None

        # Look for a large range candle (expansion candle)
        last_5m = df_5m.iloc[-1]
        atr_5m = get_atr_value(df_5m)
        if atr_5m == 0:
            return None

        candle_range_5m = last_5m["high"] - last_5m["low"]
        range_expansion = candle_range_5m > atr_5m * 1.5  # expansion candle is >1.5x ATR

        if not range_expansion:
            return None

        # Determine direction from expansion candle
        candle_bullish = last_5m["close"] > last_5m["open"]
        direction = SignalDirection.LONG if candle_bullish else SignalDirection.SHORT

        # Step 3: Volume expansion
        vol_ratio = last_5m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        volume_expanding = vol_ratio > 1.5

        # Step 4: Overextension check
        # If price moved >3 ATR from center of compression range in one candle, likely exhausted
        bb_mid = last_15m.get("bb_mid", current_price)
        if pd.isna(bb_mid):
            bb_mid = current_price
        price_from_center = abs(current_price - bb_mid)
        overextended = price_from_center > atr_15m * 3

        if overextended:
            return None

        # Step 5: Immediate exhaustion check
        # If the expansion candle has a significant opposing wick (>40% of candle range), reject
        if direction == SignalDirection.LONG:
            upper_wick = last_5m["high"] - last_5m["close"]
            exhausted = upper_wick > candle_range_5m * 0.4
        else:
            lower_wick = last_5m["close"] - last_5m["low"]
            exhausted = lower_wick > candle_range_5m * 0.4

        if exhausted:
            return None

        # Higher timeframe context
        htf_aligned = False
        if df_4h is not None:
            trend_4h = get_trend_direction(df_4h, "ema_20", "ema_50")
            htf_aligned = (trend_4h == "bullish" and direction == SignalDirection.LONG) or \
                          (trend_4h == "bearish" and direction == SignalDirection.SHORT) or \
                          trend_4h == "neutral"

        # Regime alignment bonus
        regime_aligned = regime in ("volatility_expansion", "compression", "unknown")

        # Confidence
        confidence = 0.0
        confidence += 0.20 if in_compression else 0.05
        confidence += 0.10 if volatility_compressed else 0.0
        confidence += 0.20 if range_expansion else 0.0
        confidence += 0.20 if volume_expanding else 0.08
        confidence += 0.15 if htf_aligned else 0.03
        confidence += 0.10 if regime_aligned else 0.02
        confidence += 0.05 * min(candle_range_5m / (atr_5m + 1e-8) / 2.0, 1.0)

        confidence = min(confidence, 1.0)

        # Stop: just outside compression range
        stop_distance = atr_5m * 1.5
        if direction == SignalDirection.LONG:
            stop_price = current_price - stop_distance
            tp_price = current_price + stop_distance * 2.5  # wider TP for expansion plays
        else:
            stop_price = current_price + stop_distance
            tp_price = current_price - stop_distance * 2.5

        reason_codes = []
        if in_compression:
            reason_codes.append("bb_compression_detected")
        if volatility_compressed:
            reason_codes.append("atr_compressed")
        if range_expansion:
            reason_codes.append("expansion_candle")
        if volume_expanding:
            reason_codes.append("volume_expanding")
        if htf_aligned:
            reason_codes.append("4h_aligned")
        if regime_aligned:
            reason_codes.append("regime_aligned")

        logger.debug("volatility_expansion_signal", symbol=symbol, direction=direction,
                     confidence=confidence, bb_width_pct=bb_width_pct)

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="15m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            validity_minutes=30,
            reason_codes=reason_codes,
            key_metrics={
                "bb_width": round(bb_width, 4),
                "bb_width_pct": round(bb_width_pct, 3),
                "volatility_ratio": round(volatility_ratio, 3) if not pd.isna(volatility_ratio) else None,
                "candle_range_ratio": round(candle_range_5m / (atr_5m + 1e-8), 2),
                "volume_ratio": round(vol_ratio, 2),
                "in_compression": in_compression,
                "overextended": overextended,
                "regime": regime,
            },
            regime_context=regime,
        )
