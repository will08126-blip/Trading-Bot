"""
Tests for the risk engine.
Critical: these must be deterministic and catch edge cases.
"""
import pytest

from app.risk.engine import RiskEngine, RiskDecision, RiskDecisionResult
from app.signals.schema import SignalDirection
from app.voting.engine import VoteResult
from app.config.trading_config import RiskConfig, LeverageTierConfig


def make_vote_result(
    score: float = 70.0,
    tier: str = "strong",
    direction: str = "long",
    symbol: str = "BTC-PERP",
) -> VoteResult:
    return VoteResult(
        symbol=symbol,
        direction=SignalDirection(direction),
        final_score=score,
        tier=tier,
        contributing_signals=[],
    )


def make_portfolio_state(
    equity: float = 10000.0,
    allocated: float = 0.0,
    daily_pnl: float = 0.0,
    open_positions: list = None,
    consecutive_losses: int = 0,
    drawdown_pct: float = 0.0,
    risk_mode: str = "normal",
) -> dict:
    return {
        "total_equity": equity,
        "allocated_capital": allocated,
        "daily_pnl": daily_pnl,
        "peak_equity": equity,
        "open_positions": open_positions or [],
        "consecutive_losses": consecutive_losses,
        "drawdown_pct": drawdown_pct,
        "risk_mode": risk_mode,
    }


class TestRiskEngine:
    def setup_method(self):
        self.engine = RiskEngine()

    def test_kill_switch_blocks_all_trades(self):
        vote = make_vote_result(score=90.0, tier="elite")
        portfolio = make_portfolio_state(risk_mode="kill_switch")
        result = self.engine.evaluate(vote, 50000.0, 49000.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
        assert "kill_switch" in result.reason

    def test_drawdown_breach_blocks_trade(self):
        vote = make_vote_result(score=80.0, tier="strong")
        portfolio = make_portfolio_state(drawdown_pct=16.0)  # > 15% default
        result = self.engine.evaluate(vote, 50000.0, 49000.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
        assert "drawdown" in result.reason

    def test_daily_loss_blocks_trade(self):
        vote = make_vote_result(score=80.0, tier="strong")
        # daily_pnl = -1100 on 10000 equity = 11% > 10% max
        portfolio = make_portfolio_state(equity=10000.0, daily_pnl=-1100.0)
        result = self.engine.evaluate(vote, 50000.0, 49000.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
        assert "daily_loss" in result.reason

    def test_max_positions_blocks_trade(self):
        vote = make_vote_result(score=80.0, tier="strong")
        # Fill max positions (default 4)
        open_positions = [{"symbol": "BTC-PERP"} for _ in range(4)]
        portfolio = make_portfolio_state(open_positions=open_positions)
        result = self.engine.evaluate(vote, 50000.0, 49000.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT

    def test_stop_too_tight_blocked(self):
        vote = make_vote_result(score=80.0, tier="strong")
        portfolio = make_portfolio_state()
        # Stop distance = 10 on 50000 = 0.02% < 0.15% minimum
        result = self.engine.evaluate(vote, 50000.0, 49990.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
        assert "stop_too_tight" in result.reason

    def test_stop_too_wide_blocked(self):
        vote = make_vote_result(score=80.0, tier="strong")
        portfolio = make_portfolio_state()
        # Stop distance = 2000 on 50000 = 4% > 3% maximum
        result = self.engine.evaluate(vote, 50000.0, 48000.0, 55000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
        assert "stop_too_wide" in result.reason

    def test_high_slippage_blocked(self):
        vote = make_vote_result(score=80.0, tier="strong")
        portfolio = make_portfolio_state()
        result = self.engine.evaluate(
            vote, 50000.0, 49250.0, 52000.0, portfolio,
            {"slippage_estimate_pct": 0.5}  # 0.5% > 0.1% max
        )
        assert result.decision == RiskDecision.REJECT
        assert "slippage" in result.reason

    def test_no_trade_tier_blocked(self):
        vote = make_vote_result(score=35.0, tier="no_trade")
        portfolio = make_portfolio_state()
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT

    def test_valid_trade_approved_with_sizing(self):
        vote = make_vote_result(score=70.0, tier="strong")
        portfolio = make_portfolio_state(equity=10000.0)
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})
        assert result.decision in (RiskDecision.APPROVE, RiskDecision.REDUCE)
        assert result.approved_quantity > 0
        assert result.approved_leverage > 0
        assert result.capital_at_risk > 0
        assert result.capital_at_risk <= portfolio["total_equity"]

    def test_risk_sizing_respects_equity(self):
        """Position size should be based on risk %."""
        vote = make_vote_result(score=70.0, tier="strong")
        portfolio = make_portfolio_state(equity=10000.0)
        # Stop distance = 750 (1.5%)
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})

        if result.decision != RiskDecision.REJECT:
            # Risk per trade default 1% of 10000 = 100 USD
            # But adjusted for "strong" tier (0.8x) = 80 USD
            # stop_distance = 750
            # quantity = 80/750 ≈ 0.107
            assert result.approved_quantity > 0
            assert result.capital_at_risk <= 10000 * 0.02  # max 2%

    def test_leverage_caps_respected(self):
        vote = make_vote_result(score=90.0, tier="elite")
        portfolio = make_portfolio_state(equity=10000.0)
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})
        if result.decision != RiskDecision.REJECT:
            cfg = self.engine._cfg
            assert result.approved_leverage <= cfg.leverage_tiers.max_leverage_cap

    def test_reduced_risk_mode_cuts_size(self):
        vote = make_vote_result(score=70.0, tier="strong")
        portfolio_normal = make_portfolio_state(risk_mode="normal")
        portfolio_reduced = make_portfolio_state(risk_mode="reduced")

        result_normal = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio_normal, {})
        result_reduced = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio_reduced, {})

        if result_normal.decision != RiskDecision.REJECT and result_reduced.decision != RiskDecision.REJECT:
            assert result_reduced.capital_at_risk < result_normal.capital_at_risk

    def test_max_notional_cap(self):
        """Position should be capped at max_notional_per_trade_usd."""
        vote = make_vote_result(score=90.0, tier="elite")
        # Very large equity
        portfolio = make_portfolio_state(equity=10_000_000.0)
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})
        if result.decision != RiskDecision.REJECT:
            cfg = self.engine._cfg
            assert result.approved_size_usd <= cfg.max_notional_per_trade_usd + 1.0

    def test_zero_equity_blocks_trade(self):
        vote = make_vote_result(score=80.0, tier="strong")
        portfolio = make_portfolio_state(equity=0.0)
        result = self.engine.evaluate(vote, 50000.0, 49250.0, 52000.0, portfolio, {})
        assert result.decision == RiskDecision.REJECT
