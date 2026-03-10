"""
Tests for strategy modules.
Strategies should return valid StrategySignal objects or None.
"""
import pytest
import asyncio
from typing import Optional

from app.signals.schema import StrategySignal, SignalDirection
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.breakout_retest import BreakoutRetestStrategy
from app.strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from app.strategies.volatility_expansion import VolatilityExpansionStrategy
from tests.fixtures.market_data import (
    make_candles, make_trending_candles, make_ranging_candles,
    make_multi_tf_candles,
)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStrategySignalSchema:
    """Test that strategies return well-formed signals."""

    def _validate_signal(self, sig: Optional[StrategySignal]) -> None:
        if sig is None:
            return
        assert isinstance(sig, StrategySignal)
        assert sig.direction in (SignalDirection.LONG, SignalDirection.SHORT, "long", "short")
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.strategy_family
        assert sig.symbol
        if sig.stop_price and sig.entry_price:
            assert sig.stop_price != sig.entry_price
        assert isinstance(sig.reason_codes, list)

    def test_trend_pullback_bullish(self):
        strategy = TrendPullbackStrategy()
        candles = make_multi_tf_candles(200)
        # Use trending candles for 4h and 15m
        candles["4h"] = make_trending_candles(100, "up")
        candles["15m"] = make_trending_candles(100, "up")
        candles["5m"] = make_trending_candles(150, "up")

        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "trending"))
        self._validate_signal(sig)

    def test_trend_pullback_no_signal_on_short_data(self):
        strategy = TrendPullbackStrategy()
        short_candles = {"4h": make_candles(10), "15m": make_candles(10), "5m": make_candles(10)}
        sig = run_async(strategy.evaluate("BTC-PERP", short_candles, 50000.0, "trending"))
        assert sig is None

    def test_breakout_retest_returns_valid_or_none(self):
        strategy = BreakoutRetestStrategy()
        candles = make_multi_tf_candles(200)
        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "ranging"))
        self._validate_signal(sig)

    def test_liquidity_sweep_blocked_in_trending_regime(self):
        strategy = LiquiditySweepReversalStrategy()
        candles = make_multi_tf_candles(200)
        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "trending"))
        assert sig is None  # Should not trade in trending regime

    def test_liquidity_sweep_in_ranging_regime(self):
        strategy = LiquiditySweepReversalStrategy()
        candles = make_multi_tf_candles(200)
        candles["15m"] = make_ranging_candles(200)
        candles["5m"] = make_ranging_candles(200)
        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "ranging"))
        self._validate_signal(sig)

    def test_volatility_expansion_returns_valid_or_none(self):
        strategy = VolatilityExpansionStrategy()
        candles = make_multi_tf_candles(200)
        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "compression"))
        self._validate_signal(sig)

    def test_strategy_signal_has_required_fields(self):
        """Verify the signal schema has all required fields."""
        sig = StrategySignal(
            strategy_family="trend_pullback",
            symbol="BTC-PERP",
            timeframe="5m",
            direction=SignalDirection.LONG,
            confidence=0.7,
            entry_price=50000.0,
            stop_price=49000.0,
            tp_price=52000.0,
        )
        assert sig.strategy_family == "trend_pullback"
        assert sig.confidence == 0.7
        assert sig.direction == "long"
        assert sig.validity_minutes == 30  # default


class TestStrategyOutputConsistency:
    def test_confidence_bounded(self):
        """All strategies must output confidence in [0, 1]."""
        for _ in range(5):
            candles = make_multi_tf_candles(200)
            for Strategy in [TrendPullbackStrategy, BreakoutRetestStrategy,
                             LiquiditySweepReversalStrategy, VolatilityExpansionStrategy]:
                strategy = Strategy()
                sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "trending"))
                if sig:
                    assert 0.0 <= sig.confidence <= 1.0, f"{Strategy.__name__} confidence out of range"

    def test_stop_direction_correct(self):
        """Stop should be below entry for longs, above for shorts."""
        strategy = TrendPullbackStrategy()
        candles = make_multi_tf_candles(200)
        candles["4h"] = make_trending_candles(100, "up")
        candles["15m"] = make_trending_candles(100, "up")
        candles["5m"] = make_trending_candles(150, "up")

        sig = run_async(strategy.evaluate("BTC-PERP", candles, 50000.0, "trending"))
        if sig and sig.stop_price and sig.entry_price:
            if sig.direction == "long":
                assert sig.stop_price < sig.entry_price
            else:
                assert sig.stop_price > sig.entry_price
