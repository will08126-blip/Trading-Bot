"""
Simplified Voting Engine — no ML, no exchange-specific deps.
Aggregates strategy signals + context into a final score (0-100) and tier.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..strategies.schema import SignalDirection, SignalTier, StrategySignal, VoteResult

logger = logging.getLogger(__name__)


class VotingEngine:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Args:
            cfg: voting section from signals_config.yaml
        """
        self._cfg = cfg
        self._strategy_weights: Dict[str, float] = cfg.get("strategy_weights", {
            "trend_pullback": 1.0,
            "breakout_retest": 1.0,
            "liquidity_sweep_reversal": 0.85,
            "volatility_expansion": 0.90,
        })
        self._w_strategy = cfg.get("strategy_signal_weight", 0.30)
        self._w_trend = cfg.get("trend_alignment_weight", 0.20)
        self._w_momentum = cfg.get("momentum_weight", 0.15)
        self._w_volatility = cfg.get("volatility_regime_weight", 0.10)
        self._w_liquidity = cfg.get("liquidity_weight", 0.15)
        self._w_session = cfg.get("session_weight", 0.10)
        self._score_thresholds = cfg.get("score_thresholds", {
            "no_trade_max": 40.0,
            "strong_min": 60.0,
            "elite_min": 80.0,
        })

    def vote(
        self,
        symbol: str,
        signals: List[StrategySignal],
        direction: SignalDirection,
        context: Dict[str, Any],
    ) -> VoteResult:
        """
        Compute the final vote score for a symbol + direction.

        Context keys expected:
            trend_4h, trend_15m, rsi, macd_hist, adx,
            volatility_state, volume_24h_usd, session_score, spread_pct
        """
        reason_codes: List[str] = []

        # Hard gate: must have at least one strategy signal
        relevant = [s for s in signals if s.direction == direction]
        if not relevant:
            return VoteResult(
                symbol=symbol, direction=direction,
                final_score=0.0, tier=SignalTier.NO_TRADE,
                contributing_signals=[],
                reason_codes=["no_strategy_signals"],
            )

        # ── Component scores (0–1 each) ───────────────────────────────────────
        strategy_score = self._score_strategy(relevant, reason_codes)
        trend_score = self._score_trend(direction, context, reason_codes)
        momentum_score = self._score_momentum(direction, context, reason_codes)
        vol_score = self._score_volatility(context, reason_codes)
        liq_score = self._score_liquidity(context, reason_codes)
        session_score = self._score_session(context, reason_codes)

        # ── Weighted sum (already summing to weight total, then normalise) ────
        weight_sum = (self._w_strategy + self._w_trend + self._w_momentum +
                      self._w_volatility + self._w_liquidity + self._w_session)

        raw_score = (
            strategy_score * self._w_strategy +
            trend_score * self._w_trend +
            momentum_score * self._w_momentum +
            vol_score * self._w_volatility +
            liq_score * self._w_liquidity +
            session_score * self._w_session
        ) * 100 / weight_sum

        # ── Spread penalty ────────────────────────────────────────────────────
        spread_pct = context.get("spread_pct", 0.0)
        spread_penalty = 0.0
        if spread_pct > 0.05:
            spread_penalty = min(20.0, (spread_pct - 0.05) / 0.25 * 20)
            reason_codes.append("spread_penalty_applied")

        raw_score = max(0.0, raw_score - spread_penalty)
        final_score = max(0.0, min(100.0, raw_score))

        tier = self._tier(final_score)
        if tier == SignalTier.NO_TRADE:
            reason_codes.append("score_below_threshold")

        # ── Aggregate price levels from signals ───────────────────────────────
        entry_price = relevant[0].entry_price
        # Use the stop from the highest-confidence signal
        best_sig = max(relevant, key=lambda s: s.confidence)
        stop_price = best_sig.stop_price
        tp_primary = best_sig.tp_price

        # Derive TP1 / TP2 from risk distance
        if entry_price and stop_price:
            risk = abs(entry_price - stop_price)
            sign = 1 if direction == SignalDirection.LONG else -1
            tp1 = entry_price + sign * risk * 1.5
            tp2 = entry_price + sign * risk * 3.0
        else:
            tp1 = tp_primary
            tp2 = tp_primary

        logger.debug(
            "vote_result symbol=%s dir=%s score=%.1f tier=%s",
            symbol, direction.value, final_score, tier.value,
        )

        return VoteResult(
            symbol=symbol,
            direction=direction,
            final_score=round(final_score, 2),
            tier=tier,
            contributing_signals=relevant,
            strategy_signal_score=round(strategy_score * 100, 1),
            trend_alignment_score=round(trend_score * 100, 1),
            momentum_score=round(momentum_score * 100, 1),
            volatility_score=round(vol_score * 100, 1),
            liquidity_score=round(liq_score * 100, 1),
            session_score=round(session_score * 100, 1),
            spread_penalty=round(spread_penalty, 1),
            reason_codes=list(set(reason_codes)),
            entry_price=entry_price,
            stop_price=stop_price,
            tp1_price=round(tp1, 6) if tp1 else None,
            tp2_price=round(tp2, 6) if tp2 else None,
        )

    # ── Component scorers ─────────────────────────────────────────────────────

    def _score_strategy(self, signals: List[StrategySignal], reason_codes: List[str]) -> float:
        total_w = 0.0
        weighted_conf = 0.0
        for sig in signals:
            w = self._strategy_weights.get(sig.strategy_family, 1.0)
            weighted_conf += sig.confidence * w
            total_w += w
            reason_codes.extend(f"{sig.strategy_family}:{c}" for c in sig.reason_codes[:2])

        if total_w == 0:
            return 0.0
        score = weighted_conf / total_w

        if len(signals) >= 2:
            score = min(1.0, score * 1.10)
            reason_codes.append("multi_strategy_confluence")
        if len(signals) >= 3:
            score = min(1.0, score * 1.05)
            reason_codes.append("strong_multi_strategy_confluence")
        return score

    def _score_trend(self, direction: SignalDirection, ctx: Dict, reason_codes: List[str]) -> float:
        trend_4h = ctx.get("trend_4h", "neutral")
        trend_15m = ctx.get("trend_15m", "neutral")
        adx = ctx.get("adx", 0.0)
        expected = "bullish" if direction == SignalDirection.LONG else "bearish"

        score = 0.0
        if trend_4h == expected:
            score += 0.5
            reason_codes.append("4h_trend_aligned")
        elif trend_4h == "neutral":
            score += 0.2
        else:
            reason_codes.append("4h_trend_opposing")

        if trend_15m == expected:
            score += 0.4
            reason_codes.append("15m_trend_aligned")
        elif trend_15m == "neutral":
            score += 0.15

        if adx > 30:
            score = min(1.0, score * 1.1)
            reason_codes.append("strong_adx")
        elif adx < 20:
            score *= 0.8

        return min(1.0, score)

    def _score_momentum(self, direction: SignalDirection, ctx: Dict, reason_codes: List[str]) -> float:
        rsi = ctx.get("rsi", 50.0)
        macd_hist = ctx.get("macd_hist", 0.0)
        score = 0.5

        if direction == SignalDirection.LONG:
            if 45 <= rsi <= 65:
                score += 0.2
                reason_codes.append("rsi_bullish_zone")
            elif rsi > 70:
                score -= 0.2
            elif rsi < 45:
                score -= 0.1
        else:
            if 35 <= rsi <= 55:
                score += 0.2
                reason_codes.append("rsi_bearish_zone")
            elif rsi < 30:
                score -= 0.2
            elif rsi > 55:
                score -= 0.1

        if direction == SignalDirection.LONG and macd_hist > 0:
            score += 0.15
        elif direction == SignalDirection.SHORT and macd_hist < 0:
            score += 0.15
        elif macd_hist != 0:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _score_volatility(self, ctx: Dict, reason_codes: List[str]) -> float:
        state = ctx.get("volatility_state", "medium")
        mapping = {"low": 0.7, "medium": 0.8, "high": 0.6, "extreme": 0.3}
        score = mapping.get(state, 0.5)
        if state == "extreme":
            reason_codes.append("extreme_volatility")
        elif state == "medium":
            reason_codes.append("healthy_volatility")
        return score

    def _score_liquidity(self, ctx: Dict, reason_codes: List[str]) -> float:
        volume_24h = ctx.get("volume_24h_usd", 0.0)
        score = 0.5
        min_vol = 50_000_000.0
        if volume_24h >= min_vol:
            score += 0.4
            reason_codes.append("adequate_volume")
        elif volume_24h > 0:
            score += 0.2 * (volume_24h / min_vol)
        return min(1.0, score)

    def _score_session(self, ctx: Dict, reason_codes: List[str]) -> float:
        ss = ctx.get("session_score", 0.7)
        if ss >= 0.9:
            reason_codes.append("prime_session")
        elif ss < 0.5:
            reason_codes.append("low_session_quality")
        return float(ss)

    def _tier(self, score: float) -> SignalTier:
        t = self._score_thresholds
        if score < t.get("no_trade_max", 40.0):
            return SignalTier.NO_TRADE
        if score < t.get("strong_min", 60.0):
            return SignalTier.MEDIUM
        if score < t.get("elite_min", 80.0):
            return SignalTier.STRONG
        return SignalTier.ELITE
