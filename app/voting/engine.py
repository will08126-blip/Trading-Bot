"""
Weighted Voting Engine

Aggregates evidence across strategy families + context filters.
Produces a final score 0-100 and a tier classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from app.config.trading_config import VotingConfig, get_trading_config
from app.signals.schema import SignalDirection, SignalTier, StrategySignal

logger = structlog.get_logger(__name__)


@dataclass
class VoteResult:
    symbol: str
    direction: SignalDirection
    final_score: float                   # 0–100
    tier: SignalTier
    contributing_signals: List[StrategySignal]

    # Component scores (all 0–100 before weighting)
    trend_alignment_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    session_score: float = 0.0
    strategy_signal_score: float = 0.0

    # Penalties (reduce from score)
    spread_penalty: float = 0.0
    slippage_penalty: float = 0.0

    # Performance factor (multiplier from recent strategy stats)
    performance_factor: float = 1.0

    # ML overlay
    ml_quality_score: Optional[float] = None

    # Explanation
    reason_codes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    voted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "final_score": self.final_score,
            "tier": self.tier,
            "trend_alignment_score": self.trend_alignment_score,
            "momentum_score": self.momentum_score,
            "volatility_score": self.volatility_score,
            "liquidity_score": self.liquidity_score,
            "session_score": self.session_score,
            "strategy_signal_score": self.strategy_signal_score,
            "spread_penalty": self.spread_penalty,
            "slippage_penalty": self.slippage_penalty,
            "performance_factor": self.performance_factor,
            "ml_quality_score": self.ml_quality_score,
            "reason_codes": self.reason_codes,
        }


class VotingEngine:
    """
    Aggregates signals and context into a final score.
    All weights and thresholds are loaded from config.
    """

    def __init__(self, voting_config: Optional[VotingConfig] = None) -> None:
        self._cfg = voting_config or get_trading_config().voting

    def vote(
        self,
        symbol: str,
        signals: List[StrategySignal],
        direction: SignalDirection,
        context: Dict[str, Any],
        strategy_health: Optional[Dict[str, Any]] = None,
        ml_quality_score: Optional[float] = None,
    ) -> VoteResult:
        """
        Compute the final vote score for a direction given signals and context.

        Args:
            symbol: trading symbol
            signals: list of StrategySignals (may include opposing direction signals)
            direction: the direction being scored (long or short)
            context: dict of context data:
                - trend_4h: "bullish" | "bearish" | "neutral"
                - trend_15m: "bullish" | "bearish" | "neutral"
                - rsi: float
                - adx: float
                - volatility_state: "low" | "medium" | "high" | "extreme"
                - spread_pct: float
                - volume_24h_usd: float
                - order_book_depth_usd: float
                - session_score: float (0–1)
                - slippage_estimate_pct: float
                - recent_performance: dict (optional)
            strategy_health: dict keyed by strategy name with health stats
            ml_quality_score: optional ML trade quality score (0–1)

        Returns:
            VoteResult
        """
        cfg = self._cfg
        reason_codes: List[str] = []
        details: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # 0. Hard gate: no signals = no trade
        # Strategy signals are mandatory; context alone cannot generate a trade.
        # ----------------------------------------------------------------
        relevant_signals = [s for s in signals if s.direction == direction]
        if not relevant_signals:
            return VoteResult(
                symbol=symbol,
                direction=direction,
                final_score=0.0,
                tier=SignalTier.NO_TRADE,
                contributing_signals=[],
                reason_codes=["no_strategy_signals"],
            )

        # ----------------------------------------------------------------
        # 1. Strategy signal aggregation
        # ----------------------------------------------------------------
        strategy_signal_score = self._compute_strategy_score(
            relevant_signals, direction, strategy_health, cfg, reason_codes
        )
        details["contributing_strategy_count"] = len(relevant_signals)

        # ----------------------------------------------------------------
        # 2. Trend alignment score
        # ----------------------------------------------------------------
        trend_alignment_score = self._score_trend_alignment(
            direction, context, reason_codes
        )

        # ----------------------------------------------------------------
        # 3. Momentum score
        # ----------------------------------------------------------------
        momentum_score = self._score_momentum(direction, context, reason_codes)

        # ----------------------------------------------------------------
        # 4. Volatility regime score
        # ----------------------------------------------------------------
        volatility_score = self._score_volatility_regime(context, reason_codes)

        # ----------------------------------------------------------------
        # 5. Liquidity score
        # ----------------------------------------------------------------
        liquidity_score = self._score_liquidity(context, reason_codes)

        # ----------------------------------------------------------------
        # 6. Session score
        # ----------------------------------------------------------------
        session_score = self._score_session(context, reason_codes)

        # ----------------------------------------------------------------
        # 7. Penalties
        # ----------------------------------------------------------------
        spread_penalty = self._compute_spread_penalty(context)
        slippage_penalty = self._compute_slippage_penalty(context)

        if spread_penalty > 30:
            reason_codes.append("wide_spread_penalty")
        if slippage_penalty > 30:
            reason_codes.append("high_slippage_penalty")

        # ----------------------------------------------------------------
        # 8. Strategy performance factor
        # ----------------------------------------------------------------
        perf_factor = self._compute_performance_factor(
            relevant_signals, strategy_health, reason_codes
        )

        # ----------------------------------------------------------------
        # 9. Weighted aggregate (before penalties and ML)
        # ----------------------------------------------------------------
        w = cfg
        raw_score = (
            strategy_signal_score * w.strategy_signal_weight
            + trend_alignment_score * w.trend_alignment_weight
            + momentum_score * w.momentum_weight
            + volatility_score * w.volatility_regime_weight
            + liquidity_score * w.liquidity_weight
            + session_score * w.session_weight
        ) * 100  # scale to 0–100

        # Normalize weights sum to 1
        weight_sum = (
            w.strategy_signal_weight + w.trend_alignment_weight + w.momentum_weight
            + w.volatility_regime_weight + w.liquidity_weight + w.session_weight
        )
        if weight_sum > 0:
            raw_score = raw_score / weight_sum

        # ----------------------------------------------------------------
        # 10. Apply penalties
        # ----------------------------------------------------------------
        penalty_total = (
            spread_penalty * w.spread_penalty_weight
            + slippage_penalty * w.slippage_penalty_weight
        ) / (w.spread_penalty_weight + w.slippage_penalty_weight + 1e-8)

        raw_score = max(0.0, raw_score - penalty_total)

        # ----------------------------------------------------------------
        # 11. Apply performance factor
        # ----------------------------------------------------------------
        raw_score *= perf_factor

        # ----------------------------------------------------------------
        # 12. ML quality overlay
        # ----------------------------------------------------------------
        if ml_quality_score is not None:
            # Blend ML score: if ML says low quality, reduce score
            ml_factor = 0.5 + ml_quality_score * 0.5  # 0.5 to 1.0
            raw_score *= ml_factor
            if ml_quality_score < 0.4:
                reason_codes.append("ml_low_quality")
            elif ml_quality_score > 0.7:
                reason_codes.append("ml_high_quality")

        final_score = max(0.0, min(100.0, raw_score))

        # ----------------------------------------------------------------
        # 13. Tier classification
        # ----------------------------------------------------------------
        tier = self._classify_tier(final_score)
        if tier == SignalTier.NO_TRADE:
            reason_codes.append("score_below_threshold")

        if get_trading_config().verbose_voting_logging:
            logger.info(
                "vote_result",
                symbol=symbol,
                direction=direction,
                final_score=round(final_score, 1),
                tier=tier,
                strategy_score=round(strategy_signal_score * 100, 1),
                trend_score=round(trend_alignment_score * 100, 1),
                session_score_raw=round(session_score * 100, 1),
                penalties=round(penalty_total, 1),
            )

        return VoteResult(
            symbol=symbol,
            direction=direction,
            final_score=round(final_score, 2),
            tier=tier,
            contributing_signals=relevant_signals,
            trend_alignment_score=round(trend_alignment_score * 100, 1),
            momentum_score=round(momentum_score * 100, 1),
            volatility_score=round(volatility_score * 100, 1),
            liquidity_score=round(liquidity_score * 100, 1),
            session_score=round(session_score * 100, 1),
            strategy_signal_score=round(strategy_signal_score * 100, 1),
            spread_penalty=round(spread_penalty, 1),
            slippage_penalty=round(slippage_penalty, 1),
            performance_factor=round(perf_factor, 3),
            ml_quality_score=ml_quality_score,
            reason_codes=list(set(reason_codes)),
            details=details,
        )

    # ------------------------------------------------------------------
    # Component scorers (all return 0–1)
    # ------------------------------------------------------------------

    def _compute_strategy_score(
        self,
        signals: List[StrategySignal],
        direction: SignalDirection,
        strategy_health: Optional[Dict],
        cfg: VotingConfig,
        reason_codes: List[str],
    ) -> float:
        if not signals:
            return 0.0

        weights = cfg.strategy_weights
        weight_map = {
            "trend_pullback": weights.trend_pullback,
            "breakout_retest": weights.breakout_retest,
            "liquidity_sweep_reversal": weights.liquidity_sweep_reversal,
            "volatility_expansion": weights.volatility_expansion,
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for sig in signals:
            w = weight_map.get(sig.strategy_family, 1.0)
            # Discount if strategy health is poor
            health_factor = 1.0
            if strategy_health and sig.strategy_family in strategy_health:
                health = strategy_health[sig.strategy_family]
                health_factor = max(0.3, min(1.2, health.get("performance_factor", 1.0)))
            conf = sig.confidence * health_factor
            weighted_confidence += conf * w
            total_weight += w
            reason_codes.extend([f"{sig.strategy_family}:{code}" for code in sig.reason_codes[:3]])

        if total_weight == 0:
            return 0.0

        score = weighted_confidence / total_weight

        # Bonus for multiple confirming strategies
        if len(signals) >= 2:
            score = min(1.0, score * 1.1)
            reason_codes.append("multi_strategy_confluence")
        if len(signals) >= 3:
            score = min(1.0, score * 1.05)
            reason_codes.append("strong_multi_strategy_confluence")

        return score

    def _score_trend_alignment(
        self,
        direction: SignalDirection,
        context: Dict[str, Any],
        reason_codes: List[str],
    ) -> float:
        trend_4h = context.get("trend_4h", "neutral")
        trend_15m = context.get("trend_15m", "neutral")
        adx = context.get("adx", 0.0)

        expected = "bullish" if direction == SignalDirection.LONG else "bearish"

        score = 0.0
        if trend_4h == expected:
            score += 0.5
            reason_codes.append("4h_trend_aligned")
        elif trend_4h == "neutral":
            score += 0.2
        else:
            score += 0.0
            reason_codes.append("4h_trend_opposing")

        if trend_15m == expected:
            score += 0.4
            reason_codes.append("15m_trend_aligned")
        elif trend_15m == "neutral":
            score += 0.15
        else:
            score += 0.0

        # ADX bonus
        if adx > 30:
            score = min(1.0, score * 1.1)
            reason_codes.append("strong_adx")
        elif adx < 20:
            score *= 0.8

        return min(1.0, score)

    def _score_momentum(
        self,
        direction: SignalDirection,
        context: Dict[str, Any],
        reason_codes: List[str],
    ) -> float:
        rsi = context.get("rsi", 50.0)
        macd_hist = context.get("macd_hist", 0.0)

        score = 0.5  # neutral base

        # RSI momentum
        if direction == SignalDirection.LONG:
            if 45 <= rsi <= 65:
                score += 0.2  # ideal momentum
                reason_codes.append("rsi_bullish_zone")
            elif rsi < 45:
                score -= 0.1  # weak upward momentum
            elif rsi > 70:
                score -= 0.2  # overbought
        else:
            if 35 <= rsi <= 55:
                score += 0.2
                reason_codes.append("rsi_bearish_zone")
            elif rsi > 55:
                score -= 0.1
            elif rsi < 30:
                score -= 0.2  # oversold

        # MACD histogram direction
        if direction == SignalDirection.LONG and macd_hist > 0:
            score += 0.15
        elif direction == SignalDirection.SHORT and macd_hist < 0:
            score += 0.15
        elif macd_hist != 0:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _score_volatility_regime(
        self,
        context: Dict[str, Any],
        reason_codes: List[str],
    ) -> float:
        vol_state = context.get("volatility_state", "medium")

        if vol_state == "medium":
            reason_codes.append("healthy_volatility")
            return 0.8
        elif vol_state == "high":
            reason_codes.append("elevated_volatility")
            return 0.6
        elif vol_state == "low":
            reason_codes.append("low_volatility")
            return 0.7
        elif vol_state == "extreme":
            reason_codes.append("extreme_volatility")
            return 0.3
        return 0.5

    def _score_liquidity(
        self,
        context: Dict[str, Any],
        reason_codes: List[str],
    ) -> float:
        volume_24h = context.get("volume_24h_usd", 0.0)
        ob_depth = context.get("order_book_depth_usd", 0.0)

        score = 0.5
        min_vol = 50_000_000.0
        min_depth = 100_000.0

        if volume_24h >= min_vol:
            score += 0.3
            reason_codes.append("adequate_volume")
        elif volume_24h > 0:
            score += 0.1 * (volume_24h / min_vol)

        if ob_depth >= min_depth:
            score += 0.2
        elif ob_depth > 0:
            score += 0.1 * (ob_depth / min_depth)

        return max(0.0, min(1.0, score))

    def _score_session(
        self,
        context: Dict[str, Any],
        reason_codes: List[str],
    ) -> float:
        session_score = context.get("session_score", 0.7)
        if session_score >= 0.9:
            reason_codes.append("prime_session")
        elif session_score < 0.5:
            reason_codes.append("low_session_quality")
        return float(session_score)

    def _compute_spread_penalty(self, context: Dict[str, Any]) -> float:
        spread_pct = context.get("spread_pct", 0.0)
        max_spread = 0.05  # 0.05% = no penalty
        if spread_pct <= max_spread:
            return 0.0
        # Linear penalty up to 100 at spread=0.3%
        return min(100.0, (spread_pct - max_spread) / 0.25 * 100)

    def _compute_slippage_penalty(self, context: Dict[str, Any]) -> float:
        slippage = context.get("slippage_estimate_pct", 0.0)
        max_ok = 0.05
        if slippage <= max_ok:
            return 0.0
        return min(100.0, (slippage - max_ok) / 0.15 * 100)

    def _compute_performance_factor(
        self,
        signals: List[StrategySignal],
        strategy_health: Optional[Dict],
        reason_codes: List[str],
    ) -> float:
        if not strategy_health or not signals:
            return 1.0

        factors = []
        for sig in signals:
            health = strategy_health.get(sig.strategy_family)
            if health:
                ef = health.get("expectancy_factor", 1.0)
                wr = health.get("win_rate", 0.5)
                # Good strategy: >55% win rate, positive expectancy
                f = 0.5 + wr * 0.5 + max(0, ef - 1.0) * 0.2
                factors.append(max(0.3, min(1.3, f)))

        if not factors:
            return 1.0

        avg_factor = sum(factors) / len(factors)
        if avg_factor < 0.6:
            reason_codes.append("degraded_strategy_performance")
        elif avg_factor > 1.1:
            reason_codes.append("strong_strategy_performance")
        return avg_factor

    def _classify_tier(self, score: float) -> SignalTier:
        cfg = self._cfg.score_thresholds
        if score < cfg.no_trade_max:
            return SignalTier.NO_TRADE
        if score < cfg.strong_min:
            return SignalTier.MEDIUM
        if score < cfg.elite_min:
            return SignalTier.STRONG
        return SignalTier.ELITE
