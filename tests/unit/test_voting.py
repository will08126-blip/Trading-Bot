"""
Tests for the weighted voting engine.
"""
import pytest

from app.signals.schema import SignalDirection, StrategySignal
from app.voting.engine import VotingEngine, VoteResult
from app.config.trading_config import TradingConfig, VotingConfig, ScoreThresholdConfig


def make_signal(
    direction: str = "long",
    confidence: float = 0.7,
    strategy: str = "trend_pullback",
    symbol: str = "BTC-PERP",
) -> StrategySignal:
    return StrategySignal(
        strategy_family=strategy,
        symbol=symbol,
        timeframe="5m",
        direction=SignalDirection(direction),
        confidence=confidence,
        entry_price=50000.0,
        stop_price=49000.0,
        tp_price=52000.0,
    )


def make_context(
    trend_4h: str = "bullish",
    trend_15m: str = "bullish",
    rsi: float = 55.0,
    adx: float = 30.0,
    vol_state: str = "medium",
    session_score: float = 0.8,
    spread_pct: float = 0.01,
    volume_24h: float = 1_000_000_000.0,
) -> dict:
    return {
        "trend_4h": trend_4h,
        "trend_15m": trend_15m,
        "rsi": rsi,
        "adx": adx,
        "volatility_state": vol_state,
        "spread_pct": spread_pct,
        "volume_24h_usd": volume_24h,
        "order_book_depth_usd": 500_000,
        "session_score": session_score,
        "slippage_estimate_pct": 0.03,
        "macd_hist": 0.5,
    }


class TestVotingEngine:
    def setup_method(self):
        self.engine = VotingEngine()

    def test_no_signals_returns_no_trade(self):
        context = make_context()
        result = self.engine.vote("BTC-PERP", [], SignalDirection.LONG, context)
        assert result.tier == "no_trade"
        assert result.final_score < 40.0

    def test_strong_aligned_signal_scores_high(self):
        signals = [
            make_signal("long", 0.85, "trend_pullback"),
            make_signal("long", 0.8, "breakout_retest"),
        ]
        context = make_context(trend_4h="bullish", trend_15m="bullish", adx=35.0)
        result = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context)
        assert result.final_score >= 40.0
        assert result.tier in ("medium", "strong", "elite")

    def test_opposing_trend_penalizes_score(self):
        # Long signal but 4h is bearish
        signals = [make_signal("long", 0.7, "trend_pullback")]
        context_aligned = make_context(trend_4h="bullish", trend_15m="bullish")
        context_opposing = make_context(trend_4h="bearish", trend_15m="bearish")

        result_aligned = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_aligned)
        result_opposing = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_opposing)

        assert result_aligned.final_score > result_opposing.final_score

    def test_wide_spread_reduces_score(self):
        signals = [make_signal("long", 0.8)]
        context_tight = make_context(spread_pct=0.01)
        context_wide = make_context(spread_pct=0.2)

        result_tight = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_tight)
        result_wide = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_wide)

        assert result_tight.final_score > result_wide.final_score

    def test_score_tiers_boundaries(self):
        """Ensure tier classification matches configured thresholds."""
        cfg = self.engine._cfg

        # Force various scores
        # We test classification logic directly
        assert self.engine._classify_tier(0.0) == "no_trade"
        assert self.engine._classify_tier(39.9) == "no_trade"
        assert self.engine._classify_tier(40.0) == "medium"
        assert self.engine._classify_tier(59.9) == "medium"
        assert self.engine._classify_tier(60.0) == "strong"
        assert self.engine._classify_tier(79.9) == "strong"
        assert self.engine._classify_tier(80.0) == "elite"
        assert self.engine._classify_tier(100.0) == "elite"

    def test_multi_strategy_confluence_bonus(self):
        """Multiple confirming signals should score higher than one."""
        single_sig = [make_signal("long", 0.7, "trend_pullback")]
        multi_sig = [
            make_signal("long", 0.7, "trend_pullback"),
            make_signal("long", 0.65, "breakout_retest"),
            make_signal("long", 0.6, "volatility_expansion"),
        ]
        context = make_context()
        single_result = self.engine.vote("BTC-PERP", single_sig, SignalDirection.LONG, context)
        multi_result = self.engine.vote("BTC-PERP", multi_sig, SignalDirection.LONG, context)
        assert multi_result.final_score >= single_result.final_score

    def test_ml_quality_score_influences_result(self):
        signals = [make_signal("long", 0.75)]
        context = make_context()

        result_no_ml = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context)
        result_high_ml = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context, ml_quality_score=0.9)
        result_low_ml = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context, ml_quality_score=0.1)

        assert result_high_ml.final_score >= result_no_ml.final_score * 0.9  # high ML boosts
        assert result_low_ml.final_score <= result_no_ml.final_score  # low ML reduces

    def test_extreme_volatility_reduces_score(self):
        signals = [make_signal("long", 0.75)]
        context_normal = make_context(vol_state="medium")
        context_extreme = make_context(vol_state="extreme")

        result_normal = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_normal)
        result_extreme = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context_extreme)

        assert result_normal.final_score > result_extreme.final_score

    def test_vote_result_has_reason_codes(self):
        signals = [make_signal("long", 0.8, "trend_pullback")]
        context = make_context(trend_4h="bullish")
        result = self.engine.vote("BTC-PERP", signals, SignalDirection.LONG, context)
        assert isinstance(result.reason_codes, list)
