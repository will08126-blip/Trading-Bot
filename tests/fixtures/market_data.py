"""
Test fixtures for market data.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict

import numpy as np
import pandas as pd


def make_candles(
    n: int = 200,
    base_price: float = 50000.0,
    trend: float = 0.0,
    volatility: float = 0.01,
    timeframe: str = "5m",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV candle data.
    trend: pct change per candle (positive = uptrend)
    volatility: pct std dev of price changes
    """
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "4h": 14400}.get(timeframe, 300)

    prices = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))

    records = []
    for i, close in enumerate(prices):
        open_p = prices[i - 1] if i > 0 else close * (1 - np.random.uniform(0, volatility))
        high = max(open_p, close) * (1 + abs(np.random.normal(0, volatility / 2)))
        low = min(open_p, close) * (1 - abs(np.random.normal(0, volatility / 2)))
        volume = np.random.uniform(10, 100) * base_price

        dt = now - timedelta(seconds=tf_seconds * (n - i))
        records.append({
            "open_time": dt,
            "close_time": dt + timedelta(seconds=tf_seconds),
            "open": round(open_p, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume, 2),
        })

    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df.set_index("open_time", inplace=True)
    return df


def make_trending_candles(n: int = 200, direction: str = "up") -> pd.DataFrame:
    """Generate clearly trending candles."""
    trend = 0.002 if direction == "up" else -0.002
    return make_candles(n, trend=trend, volatility=0.005)


def make_ranging_candles(n: int = 200) -> pd.DataFrame:
    """Generate ranging/sideways candles."""
    return make_candles(n, trend=0.0, volatility=0.003)


def make_volatile_candles(n: int = 200) -> pd.DataFrame:
    """Generate high-volatility candles."""
    return make_candles(n, trend=0.0, volatility=0.03)


def make_multi_tf_candles(n_5m: int = 300) -> Dict[str, pd.DataFrame]:
    """Generate candles for all timeframes for testing."""
    return {
        "1m": make_candles(n_5m * 5, volatility=0.005),
        "5m": make_candles(n_5m, volatility=0.008),
        "15m": make_candles(n_5m // 3, volatility=0.01),
        "4h": make_candles(n_5m // 48, volatility=0.02),
    }
