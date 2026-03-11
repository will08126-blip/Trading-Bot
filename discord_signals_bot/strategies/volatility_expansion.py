"""
Strategy 4: Volatility Expansion
Trade range compression transitioning to directional expansion.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from ..market_data.features import (
    compute_all_features, get_atr_value, get_rsi_value, get_trend_direction,
)
from .base import BaseStrategy
from .schema import SignalDirection, StrategySignal

logger = logging.getLogger(__name__)


class VolatilityExpansionStrategy(BaseStrategy):
    name = "volatility_expansion"
    _compression_lookback = 20
    _compression_atr_ratio = 0.7

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

        atr_15m = get_atr_value(df_15m)
        if atr_15m == 0:
            return None

        last_15m = df_15m.iloc[-1]
        bb_width = last_15m.get("bb_width", None)
        volatility_ratio = last_15m.get("volatility_ratio", 1.0)
        if bb_width is None or pd.isna(bb_width):
            return None

        # Step 1: Compression detection
        bb_width_series = df_15m["bb_width"].tail(self._compression_lookback)
        bb_range = bb_width_series.max() - bb_width_series.min()
        bb_width_pct = (float(bb_width) - bb_width_series.min()) / (bb_range + 1e-8)
        in_compression = bb_width_pct < 0.3

        volatility_compressed = (
            pd.notna(volatility_ratio) and
            float(volatility_ratio) < self._compression_atr_ratio
        )

        if not (in_compression or volatility_compressed):
            return None

        # Step 2: Expansion candle on 5m
        last_5m = df_5m.iloc[-1]
        atr_5m = get_atr_value(df_5m)
        if atr_5m == 0:
            return None

        candle_range_5m = float(last_5m["high"] - last_5m["low"])
        range_expansion = candle_range_5m > atr_5m * 1.5
        if not range_expansion:
            return None

        direction = (SignalDirection.LONG if last_5m["close"] > last_5m["open"]
                     else SignalDirection.SHORT)

        # Step 3: Volume
        vol_ratio = last_5m.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        volume_expanding = float(vol_ratio) > 1.5

        # Step 4: Overextension
        bb_mid = last_15m.get("bb_mid", current_price)
        if pd.isna(bb_mid):
            bb_mid = current_price
        if abs(current_price - float(bb_mid)) > atr_15m * 3:
            return None

        # Step 5: Exhaustion check (large opposing wick)
        if direction == SignalDirection.LONG:
            upper_wick = float(last_5m["high"]) - float(last_5m["close"])
            if upper_wick > candle_range_5m * 0.4:
                return None
        else:
            lower_wick = float(last_5m["close"]) - float(last_5m["low"])
            if lower_wick > candle_range_5m * 0.4:
                return None

        # HTF context
        htf_aligned = False
        if df_4h is not None:
            trend_4h = get_trend_direction(df_4h, "ema_20", "ema_50")
            htf_aligned = ((trend_4h == "bullish" and direction == SignalDirection.LONG) or
                           (trend_4h == "bearish" and direction == SignalDirection.SHORT) or
                           trend_4h == "neutral")

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

        stop_distance = atr_5m * 1.5
        if direction == SignalDirection.LONG:
            stop_price = current_price - stop_distance
            tp_price = current_price + stop_distance * 2.5
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

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="15m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 6),
            tp_price=round(tp_price, 6),
            validity_minutes=30,
            reason_codes=reason_codes,
            key_metrics={
                "bb_width_pct": round(bb_width_pct, 3),
                "volatility_ratio": round(float(volatility_ratio), 3) if pd.notna(volatility_ratio) else None,
                "candle_range_ratio": round(candle_range_5m / (atr_5m + 1e-8), 2),
                "volume_ratio": round(float(vol_ratio), 2),
                "in_compression": in_compression,
                "regime": regime,
            },
            regime_context=regime,
        )
