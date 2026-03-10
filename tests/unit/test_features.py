"""
Tests for market data feature calculation.
"""
import pytest
import numpy as np
import pandas as pd

from app.market_data.features import (
    add_moving_averages, add_atr, add_rsi, add_bollinger_bands,
    add_macd, add_adx, compute_all_features, get_trend_direction,
    get_atr_value, get_rsi_value, get_adx_value,
)
from tests.fixtures.market_data import make_candles, make_trending_candles


class TestFeatureCalculation:
    def setup_method(self):
        self.df_trending = make_trending_candles(200, direction="up")
        self.df_flat = make_candles(200, trend=0.0, volatility=0.003)

    def test_moving_averages_computed(self):
        df = add_moving_averages(self.df_trending)
        assert "ema_20" in df.columns
        assert "ema_50" in df.columns
        assert "ema_200" in df.columns
        assert not df["ema_20"].isna().all()

    def test_atr_positive(self):
        df = add_atr(self.df_trending)
        assert "atr_14" in df.columns
        assert (df["atr_14"].dropna() > 0).all()

    def test_rsi_in_range(self):
        df = add_rsi(self.df_trending)
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_bollinger_bands_width_positive(self):
        df = add_bollinger_bands(self.df_trending)
        bb_width = df["bb_width"].dropna()
        assert (bb_width >= 0).all()

    def test_macd_computed(self):
        df = add_macd(self.df_trending)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_adx_in_range(self):
        df = add_adx(self.df_trending)
        adx = df["adx"].dropna()
        assert (adx >= 0).all()
        assert (adx <= 100).all()

    def test_compute_all_features_completes(self):
        df = compute_all_features(self.df_trending)
        assert "ema_20" in df.columns
        assert "atr_14" in df.columns
        assert "rsi_14" in df.columns
        assert "adx" in df.columns
        assert "bb_width" in df.columns

    def test_trend_direction_uptrend(self):
        df = add_moving_averages(self.df_trending)
        # Uptrending: ema_20 should be above ema_50 after warmup
        trend = get_trend_direction(df.tail(100))
        assert trend in ("bullish", "neutral")  # may be neutral early

    def test_get_atr_value_returns_float(self):
        df = compute_all_features(self.df_trending)
        val = get_atr_value(df)
        assert isinstance(val, float)
        assert val >= 0

    def test_get_rsi_returns_float_in_range(self):
        df = compute_all_features(self.df_trending)
        val = get_rsi_value(df)
        assert 0 <= val <= 100

    def test_none_df_handled_gracefully(self):
        assert get_atr_value(None) == 0.0
        assert get_rsi_value(None) == 50.0
        assert get_adx_value(None) == 0.0
        assert get_trend_direction(None) == "neutral"

    def test_short_df_handled_gracefully(self):
        short_df = make_candles(5)
        result = compute_all_features(short_df)
        # Should return the df unchanged or with minimal processing
        assert result is not None


class TestSwingHighsLows:
    def test_swing_high_detected(self):
        from app.market_data.features import add_swing_highs_lows
        df = make_candles(100)
        df = add_swing_highs_lows(df, lookback=5)
        assert "swing_high" in df.columns
        assert "swing_low" in df.columns
        # At least some swing highs/lows should be detected
        assert df["swing_high"].any() or df["swing_low"].any()
