"""
Technical feature / indicator calculation.
Uses pandas-ta where possible.
All functions operate on DataFrames with columns: open, high, low, close, volume.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, fast: int = 20, slow: int = 50, trend: int = 200) -> pd.DataFrame:
    """Add EMA and SMA columns."""
    df = df.copy()
    df[f"ema_{fast}"] = df["close"].ewm(span=fast, adjust=False).mean()
    df[f"ema_{slow}"] = df["close"].ewm(span=slow, adjust=False).mean()
    df[f"ema_{trend}"] = df["close"].ewm(span=trend, adjust=False).mean()
    df[f"sma_{fast}"] = df["close"].rolling(fast).mean()
    df[f"sma_{slow}"] = df["close"].rolling(slow).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    df["atr_pct"] = df[f"atr_{period}"] / df["close"] * 100
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, signal, and histogram."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma * 100
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    return df


def add_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume ratio relative to MA."""
    df = df.copy()
    df["volume_ma"] = df["volume"].rolling(period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
    return df


def add_momentum(df: pd.DataFrame, periods: list[int] = [3, 5, 10, 20]) -> pd.DataFrame:
    """Rate of change for various lookback periods."""
    df = df.copy()
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(p) * 100
    return df


def add_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identify recent swing highs and lows.
    A swing high is a candle whose high is higher than the 'lookback' candles on each side.
    """
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(lookback, len(df) - lookback):
        window_high = df["high"].iloc[i - lookback: i + lookback + 1]
        window_low = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if df["low"].iloc[i] == window_low.min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index for trend strength."""
    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm > 0) & (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm > 0) & (minus_dm < plus_dm)] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_volatility_regime(df: pd.DataFrame, atr_period: int = 14, lookback: int = 50) -> pd.DataFrame:
    """
    Classify volatility regime by comparing current ATR to its rolling average.
    Adds: volatility_state (low | medium | high | extreme)
    """
    df = df.copy()
    if f"atr_{atr_period}" not in df.columns:
        df = add_atr(df, atr_period)
    atr_col = f"atr_{atr_period}"
    atr_ma = df[atr_col].rolling(lookback).mean()
    ratio = df[atr_col] / atr_ma.replace(0, np.nan)
    df["volatility_ratio"] = ratio

    conditions = [
        ratio < 0.7,
        (ratio >= 0.7) & (ratio < 1.2),
        (ratio >= 1.2) & (ratio < 2.0),
        ratio >= 2.0,
    ]
    choices = ["low", "medium", "high", "extreme"]
    df["volatility_state"] = np.select(conditions, choices, default="medium")
    return df


def add_structure_levels(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """
    Calculate key structure levels: recent high, recent low, midpoint.
    """
    df = df.copy()
    df["structure_high"] = df["high"].rolling(lookback).max()
    df["structure_low"] = df["low"].rolling(lookback).min()
    df["structure_mid"] = (df["structure_high"] + df["structure_low"]) / 2
    df["structure_range_pct"] = (df["structure_high"] - df["structure_low"]) / df["close"] * 100
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features needed for strategy evaluation.
    Input: raw OHLCV DataFrame.
    Output: enriched DataFrame with all indicators.
    """
    if df is None or len(df) < 30:
        return df

    df = add_moving_averages(df, fast=20, slow=50, trend=200)
    df = add_atr(df, period=14)
    df = add_rsi(df, period=14)
    df = add_macd(df)
    df = add_bollinger_bands(df, period=20)
    df = add_volume_features(df, period=20)
    df = add_momentum(df, periods=[3, 5, 10, 20])
    df = add_adx(df, period=14)
    df = add_volatility_regime(df)
    df = add_structure_levels(df, lookback=30)
    df = add_swing_highs_lows(df, lookback=5)
    return df


def get_trend_direction(df: pd.DataFrame, fast_col: str = "ema_20", slow_col: str = "ema_50") -> str:
    """
    Returns 'bullish', 'bearish', or 'neutral' based on MA relationships.
    """
    if df is None or len(df) < 2:
        return "neutral"
    last = df.iloc[-1]
    if fast_col not in last or slow_col not in last:
        return "neutral"
    fast = last[fast_col]
    slow = last[slow_col]
    if pd.isna(fast) or pd.isna(slow):
        return "neutral"
    if fast > slow * 1.001:
        return "bullish"
    elif fast < slow * 0.999:
        return "bearish"
    return "neutral"


def get_atr_value(df: pd.DataFrame, period: int = 14) -> float:
    """Return the most recent ATR value, or 0 if unavailable."""
    col = f"atr_{period}"
    if df is None or col not in df.columns or df.empty:
        return 0.0
    val = df[col].iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def get_rsi_value(df: pd.DataFrame, period: int = 14) -> float:
    col = f"rsi_{period}"
    if df is None or col not in df.columns or df.empty:
        return 50.0
    val = df[col].iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def get_adx_value(df: pd.DataFrame) -> float:
    if df is None or "adx" not in df.columns or df.empty:
        return 0.0
    val = df["adx"].iloc[-1]
    return float(val) if not pd.isna(val) else 0.0
