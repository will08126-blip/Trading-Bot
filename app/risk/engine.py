"""
Risk Engine

Deterministic, strict risk controls.
Must NOT be overridden by ML or LLM.
All rules are derived from config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from app.config.trading_config import RiskConfig, get_trading_config
from app.voting.engine import VoteResult

logger = structlog.get_logger(__name__)


class RiskDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE = "reduce"            # approve with reduced size


@dataclass
class RiskDecisionResult:
    decision: RiskDecision
    reason: str
    reject_reasons: List[str] = field(default_factory=list)

    # If approved, these are set
    approved_size_usd: float = 0.0
    approved_leverage: float = 1.0
    approved_stop_price: float = 0.0
    approved_tp_price: float = 0.0
    approved_quantity: float = 0.0
    capital_at_risk: float = 0.0
    stop_distance_pct: float = 0.0

    details: Dict[str, Any] = field(default_factory=dict)


class RiskEngine:
    """
    Evaluates all risk checks before a trade is approved.
    Also computes position sizing based on risk config.

    This is deterministic and cannot be bypassed by the ML or LLM layers.
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None) -> None:
        self._cfg = risk_config or get_trading_config().risk

    def evaluate(
        self,
        vote_result: VoteResult,
        current_price: float,
        stop_price: float,
        tp_price: float,
        portfolio_state: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> RiskDecisionResult:
        """
        Evaluate all risk controls and compute sizing.

        Args:
            vote_result: VoteResult from voting engine
            current_price: current market price
            stop_price: suggested stop loss price
            tp_price: suggested take profit price
            portfolio_state: dict with keys:
                - total_equity: float
                - allocated_capital: float (currently deployed)
                - daily_pnl: float
                - peak_equity: float
                - open_positions: list of position dicts
                - consecutive_losses: int
                - drawdown_pct: float
                - risk_mode: str (normal | reduced | kill_switch)
            market_context: dict with market conditions

        Returns:
            RiskDecisionResult
        """
        cfg = self._cfg
        reject_reasons: List[str] = []

        # ----------------------------------------------------------------
        # Gate 0: Kill switch / risk mode
        # ----------------------------------------------------------------
        risk_mode = portfolio_state.get("risk_mode", "normal")
        if risk_mode == "kill_switch":
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="kill_switch_active",
                reject_reasons=["kill_switch_active"],
            )

        # ----------------------------------------------------------------
        # Gate 1: Drawdown check
        # ----------------------------------------------------------------
        drawdown_pct = portfolio_state.get("drawdown_pct", 0.0)
        if drawdown_pct >= cfg.max_account_drawdown_pct:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="max_drawdown_exceeded",
                reject_reasons=[f"drawdown={drawdown_pct:.1f}% >= max={cfg.max_account_drawdown_pct:.1f}%"],
            )

        # ----------------------------------------------------------------
        # Gate 2: Daily loss check
        # ----------------------------------------------------------------
        total_equity = portfolio_state.get("total_equity", 0.0)
        daily_pnl = portfolio_state.get("daily_pnl", 0.0)
        if total_equity > 0:
            daily_loss_pct = -daily_pnl / total_equity * 100
            if daily_loss_pct >= cfg.max_daily_loss_pct:
                return RiskDecisionResult(
                    decision=RiskDecision.REJECT,
                    reason="daily_loss_limit_reached",
                    reject_reasons=[f"daily_loss={daily_loss_pct:.1f}% >= max={cfg.max_daily_loss_pct:.1f}%"],
                )

        # ----------------------------------------------------------------
        # Gate 3: Max positions
        # ----------------------------------------------------------------
        open_positions = portfolio_state.get("open_positions", [])
        if len(open_positions) >= cfg.max_simultaneous_positions:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="max_positions_reached",
                reject_reasons=[f"open_positions={len(open_positions)} >= max={cfg.max_simultaneous_positions}"],
            )

        # Per-symbol position cap
        symbol = vote_result.symbol
        symbol_positions = [p for p in open_positions if p.get("symbol") == symbol]
        if len(symbol_positions) >= cfg.max_positions_per_symbol:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="max_positions_per_symbol_reached",
                reject_reasons=[f"{symbol} positions={len(symbol_positions)} >= max={cfg.max_positions_per_symbol}"],
            )

        # ----------------------------------------------------------------
        # Gate 4: Capital deployment cap
        # ----------------------------------------------------------------
        allocated_capital = portfolio_state.get("allocated_capital", 0.0)
        if total_equity > 0:
            allocated_pct = allocated_capital / total_equity * 100
            if allocated_pct >= cfg.max_capital_deployed_pct:
                return RiskDecisionResult(
                    decision=RiskDecision.REJECT,
                    reason="max_capital_deployed",
                    reject_reasons=[f"deployed={allocated_pct:.1f}% >= max={cfg.max_capital_deployed_pct:.1f}%"],
                )

        # ----------------------------------------------------------------
        # Gate 5: Score threshold (minimum)
        # ----------------------------------------------------------------
        if vote_result.tier == "no_trade":
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="score_below_no_trade_threshold",
                reject_reasons=[f"score={vote_result.final_score:.1f}"],
            )

        # ----------------------------------------------------------------
        # Gate 6: Stop distance validation
        # ----------------------------------------------------------------
        if current_price <= 0 or stop_price <= 0:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="invalid_prices",
                reject_reasons=["current_price or stop_price <= 0"],
            )

        stop_distance_pct = abs(current_price - stop_price) / current_price * 100

        if stop_distance_pct < cfg.min_stop_distance_pct:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="stop_too_tight",
                reject_reasons=[f"stop_distance={stop_distance_pct:.3f}% < min={cfg.min_stop_distance_pct:.3f}%"],
            )

        if stop_distance_pct > cfg.max_stop_distance_pct:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="stop_too_wide",
                reject_reasons=[f"stop_distance={stop_distance_pct:.3f}% > max={cfg.max_stop_distance_pct:.3f}%"],
            )

        # ----------------------------------------------------------------
        # Gate 7: Slippage check
        # ----------------------------------------------------------------
        slippage_pct = market_context.get("slippage_estimate_pct", 0.0)
        if slippage_pct > cfg.max_slippage_pct:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="slippage_too_high",
                reject_reasons=[f"slippage={slippage_pct:.3f}% > max={cfg.max_slippage_pct:.3f}%"],
            )

        # ----------------------------------------------------------------
        # Sizing: risk-based position sizing
        # ----------------------------------------------------------------
        base_risk_pct = cfg.risk_per_trade_pct

        # Adjust for risk mode
        if risk_mode == "reduced":
            base_risk_pct *= cfg.reduced_risk_multiplier
            logger.info("reduced_risk_mode_applied", original_pct=cfg.risk_per_trade_pct,
                        reduced_pct=base_risk_pct)

        # Adjust for score tier
        tier = vote_result.tier
        tier_risk_multiplier = self._get_tier_risk_multiplier(tier)
        adjusted_risk_pct = min(
            base_risk_pct * tier_risk_multiplier,
            cfg.max_risk_per_trade_pct
        )

        capital_at_risk = total_equity * (adjusted_risk_pct / 100)

        # Position size = capital_at_risk / stop_distance
        stop_distance_price = abs(current_price - stop_price)
        if stop_distance_price <= 0:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="zero_stop_distance",
                reject_reasons=["stop_distance_price <= 0"],
            )

        quantity = capital_at_risk / stop_distance_price

        # Leverage selection
        leverage = self._select_leverage(tier)

        # Notional check
        notional_usd = quantity * current_price
        if notional_usd > cfg.max_notional_per_trade_usd:
            # Scale down
            quantity = cfg.max_notional_per_trade_usd / current_price
            notional_usd = cfg.max_notional_per_trade_usd
            capital_at_risk = quantity * stop_distance_price
            reject_reasons.append(f"notional_capped_to_{cfg.max_notional_per_trade_usd}")

        # Available capital check
        available_capital = total_equity - allocated_capital
        margin_required = notional_usd / leverage
        if margin_required > available_capital * (cfg.max_capital_deployed_pct / 100):
            # Scale down
            max_margin = available_capital * (cfg.max_capital_deployed_pct / 100)
            quantity = max_margin * leverage / current_price
            notional_usd = quantity * current_price
            capital_at_risk = quantity * stop_distance_price
            reject_reasons.append("scaled_down_for_capital_limit")

        if quantity <= 0 or notional_usd <= 0:
            return RiskDecisionResult(
                decision=RiskDecision.REJECT,
                reason="sizing_resulted_in_zero_quantity",
                reject_reasons=["calculated_quantity_zero"],
            )

        decision = RiskDecision.REDUCE if reject_reasons else RiskDecision.APPROVE
        reason = "approved_with_reduction" if reject_reasons else "approved"

        logger.info(
            "risk_decision",
            symbol=symbol,
            decision=decision,
            tier=tier,
            score=vote_result.final_score,
            quantity=round(quantity, 6),
            notional=round(notional_usd, 2),
            leverage=leverage,
            stop_dist_pct=round(stop_distance_pct, 3),
            capital_at_risk=round(capital_at_risk, 2),
        )

        return RiskDecisionResult(
            decision=decision,
            reason=reason,
            reject_reasons=reject_reasons,
            approved_size_usd=round(notional_usd, 2),
            approved_leverage=leverage,
            approved_stop_price=stop_price,
            approved_tp_price=tp_price,
            approved_quantity=round(quantity, 6),
            capital_at_risk=round(capital_at_risk, 2),
            stop_distance_pct=round(stop_distance_pct, 4),
            details={
                "base_risk_pct": base_risk_pct,
                "tier_multiplier": tier_risk_multiplier,
                "adjusted_risk_pct": adjusted_risk_pct,
                "stop_distance_price": round(stop_distance_price, 4),
            },
        )

    def _get_tier_risk_multiplier(self, tier: str) -> float:
        """Return risk multiplier based on signal quality tier."""
        tier_map = {
            "medium": 0.5,
            "strong": 0.8,
            "elite": 1.0,
        }
        return tier_map.get(tier, 0.5)

    def _select_leverage(self, tier: str) -> float:
        """Select leverage based on signal tier, bounded by config."""
        cfg = self._cfg
        lt = cfg.leverage_tiers
        tier_leverage_map = {
            "medium": lt.medium_leverage,
            "strong": lt.strong_leverage,
            "elite": lt.elite_leverage,
        }
        leverage = tier_leverage_map.get(tier, lt.medium_leverage)
        return min(leverage, lt.max_leverage_cap)

    # ------------------------------------------------------------------
    # Pre-trade context checks (can be called standalone)
    # ------------------------------------------------------------------

    def check_spread(self, spread_pct: float) -> bool:
        from app.config.trading_config import get_trading_config
        max_spread = get_trading_config().universe.symbol_overrides.get(
            "default", {}
        )
        # Fall back to hardcoded default from config
        return spread_pct <= 0.05  # TODO: make this symbol-specific from config

    def check_volume(self, volume_24h_usd: float, min_volume_usd: float = 50_000_000.0) -> bool:
        return volume_24h_usd >= min_volume_usd
