"""
Strategy 1: Trend Pullback

Goal: Trade pullbacks in the direction of the higher timeframe trend.

Logic:
1. Determine 4h trend via EMA slope and structure
2. Require 4h/15m directional alignment
3. Identify an orderly pullback toward MA zone
4. Require 5m confirmation candle close
5. Optional 1m refinement
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import structlog

from app.market_data.features import (
    compute_all_features, get_atr_value, get_rsi_value,
    get_trend_direction, get_adx_value,
)
from app.signals.schema import SignalDirection, StrategySignal
from app.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class TrendPullbackStrategy(BaseStrategy):
    name = "trend_pullback"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or {}

    async def evaluate(
        self,
        symbol: str,
        candles: Dict[str, pd.DataFrame],
        current_price: float,
        regime: str,
    ) -> Optional[StrategySignal]:
        df_4h = candles.get("4h")
        df_15m = candles.get("15m")
        df_5m = candles.get("5m")
        df_1m = candles.get("1m")

        if df_4h is None or df_15m is None or df_5m is None:
            return None
        if len(df_4h) < 50 or len(df_15m) < 50 or len(df_5m) < 30:
            return None

        # Compute features
        df_4h = compute_all_features(df_4h)
        df_15m = compute_all_features(df_15m)
        df_5m = compute_all_features(df_5m)

        # Step 1: Higher timeframe trend
        trend_4h = get_trend_direction(df_4h, "ema_20", "ema_50")
        adx_4h = get_adx_value(df_4h)

        # Need a clear trend (ADX > 20) on 4h
        if adx_4h < 20:
            return None
        if trend_4h == "neutral":
            return None

        direction = SignalDirection.LONG if trend_4h == "bullish" else SignalDirection.SHORT

        # Step 2: 15m alignment
        trend_15m = get_trend_direction(df_15m, "ema_20", "ema_50")
        if trend_15m != trend_4h and trend_15m != "neutral":
            return None  # conflicting structure

        # Step 3: Check pullback on 5m
        last_5m = df_5m.iloc[-1]
        ema20_5m = last_5m.get("ema_20", np.nan)
        ema50_5m = last_5m.get("ema_50", np.nan)
        atr_5m = get_atr_value(df_5m)

        if pd.isna(ema20_5m) or pd.isna(ema50_5m) or atr_5m == 0:
            return None

        # For long: price should be near or just above ema20/ema50 (pullback zone)
        # For short: price should be near or just below ema20/ema50
        pullback_zone_pct = 0.005  # within 0.5% of MA = in pullback zone
        in_pullback = False
        if direction == SignalDirection.LONG:
            ma_level = max(ema20_5m, ema50_5m * 0.995)
            if ema50_5m * (1 - pullback_zone_pct) <= current_price <= ema20_5m * (1 + pullback_zone_pct * 3):
                in_pullback = True
        else:
            ma_level = min(ema20_5m, ema50_5m * 1.005)
            if ema20_5m * (1 - pullback_zone_pct * 3) <= current_price <= ema50_5m * (1 + pullback_zone_pct):
                in_pullback = True

        if not in_pullback:
            return None

        # Step 4: 5m confirmation – look for bullish/bearish candle after pullback
        # Confirmation: last 5m candle closed in trend direction and above/below open
        prev_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else last_5m
        if direction == SignalDirection.LONG:
            confirmation = (last_5m["close"] > last_5m["open"] and
                           last_5m["close"] > prev_5m["close"])
        else:
            confirmation = (last_5m["close"] < last_5m["open"] and
                           last_5m["close"] < prev_5m["close"])

        # RSI filter – avoid overbought/oversold entries in trend direction
        rsi_5m = get_rsi_value(df_5m)
        if direction == SignalDirection.LONG and rsi_5m > 75:
            return None
        if direction == SignalDirection.SHORT and rsi_5m < 25:
            return None

        # Step 5: Optional 1m refinement
        timing_quality = 0.7  # base
        if df_1m is not None and len(df_1m) >= 10:
            df_1m = compute_all_features(df_1m)
            last_1m = df_1m.iloc[-1]
            if direction == SignalDirection.LONG and last_1m.get("close", 0) > last_1m.get("open", 0):
                timing_quality = 0.9
            elif direction == SignalDirection.SHORT and last_1m.get("close", 0) < last_1m.get("open", 0):
                timing_quality = 0.9

        # Compute confidence
        confidence = 0.0
        confidence += 0.25 if adx_4h > 25 else 0.15       # trend strength
        confidence += 0.20 if trend_15m == trend_4h else 0.05  # alignment
        confidence += 0.20 if in_pullback else 0.0
        confidence += 0.15 if confirmation else 0.0
        confidence += 0.10 if timing_quality > 0.8 else 0.05
        confidence += 0.10 * (1 - abs(rsi_5m - 50) / 50)  # RSI centrality bonus

        confidence = min(confidence, 1.0)

        # Stop and TP calculation
        stop_distance = atr_5m * 1.5
        if direction == SignalDirection.LONG:
            stop_price = current_price - stop_distance
            tp_price = current_price + stop_distance * 2.0
        else:
            stop_price = current_price + stop_distance
            tp_price = current_price - stop_distance * 2.0

        reason_codes = []
        if adx_4h > 25:
            reason_codes.append("strong_4h_trend")
        if trend_15m == trend_4h:
            reason_codes.append("4h_15m_aligned")
        if in_pullback:
            reason_codes.append("in_pullback_zone")
        if confirmation:
            reason_codes.append("5m_confirmation")
        if timing_quality > 0.8:
            reason_codes.append("1m_timing_confirmed")

        logger.debug("trend_pullback_signal", symbol=symbol, direction=direction,
                     confidence=confidence, adx_4h=adx_4h)

        return StrategySignal(
            strategy_family=self.name,
            symbol=symbol,
            timeframe="5m",
            direction=direction,
            confidence=round(confidence, 3),
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            validity_minutes=30,
            reason_codes=reason_codes,
            key_metrics={
                "adx_4h": round(adx_4h, 1),
                "trend_4h": trend_4h,
                "trend_15m": trend_15m,
                "rsi_5m": round(rsi_5m, 1),
                "atr_5m": round(atr_5m, 4),
                "in_pullback": in_pullback,
                "confirmation": confirmation,
                "regime": regime,
            },
            regime_context=regime,
        )
