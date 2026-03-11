"""
Leverage and portfolio deployment calculator.

Given a final vote score (0-100) and tier, recommends:
  - Leverage amount (snapped to common values)
  - Portfolio deployment percentage
  - Trade type label
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Dict, List

from ..strategies.schema import SignalTier


# Common leverage values — bot will snap to the nearest one
LEVERAGE_SNAP: List[int] = [5, 10, 15, 20, 25, 50, 75, 100]


@dataclass
class LeverageRecommendation:
    leverage: int
    portfolio_pct: int
    trade_type: str
    tier: str
    conviction: str   # "Moderate" | "High" | "Very High" | "Max Conviction"


def _interpolate(score: float, score_min: float, score_max: float,
                 val_min: float, val_max: float) -> float:
    """Linear interpolation within [score_min, score_max]."""
    t = (score - score_min) / max(score_max - score_min, 1)
    t = max(0.0, min(1.0, t))
    return val_min + t * (val_max - val_min)


def _snap_leverage(raw: float, snap_values: List[int]) -> int:
    """Round raw leverage to the nearest value in snap_values."""
    if not snap_values:
        return max(1, round(raw))
    idx = bisect.bisect_left(snap_values, raw)
    if idx == 0:
        return snap_values[0]
    if idx == len(snap_values):
        return snap_values[-1]
    lo, hi = snap_values[idx - 1], snap_values[idx]
    return lo if (raw - lo) <= (hi - raw) else hi


def calculate(score: float, tier: SignalTier, cfg: Dict[str, Any]) -> LeverageRecommendation:
    """
    Compute leverage and portfolio % from score + tier.

    Args:
        score: 0-100 vote score
        tier:  SignalTier enum value
        cfg:   leverage section from signals_config.yaml

    Returns:
        LeverageRecommendation
    """
    snap_values: List[int] = cfg.get("snap_values", LEVERAGE_SNAP)

    if tier == SignalTier.MEDIUM:
        band = cfg["medium"]
        trade_type = "Scalp"
        conviction = "Moderate"
    elif tier == SignalTier.STRONG:
        band = cfg["strong"]
        trade_type = "Scalp / Swing"
        conviction = "High"
    elif tier == SignalTier.ELITE:
        band = cfg["elite"]
        trade_type = "Swing"
        conviction = "Max Conviction" if score >= 90 else "Very High"
    else:
        # NO_TRADE — shouldn't reach here, but guard anyway
        return LeverageRecommendation(
            leverage=1, portfolio_pct=0,
            trade_type="No Trade", tier="no_trade", conviction="—",
        )

    raw_leverage = _interpolate(
        score,
        band["score_min"], band["score_max"],
        band["leverage_min"], band["leverage_max"],
    )
    raw_pct = _interpolate(
        score,
        band["score_min"], band["score_max"],
        band["portfolio_pct_min"], band["portfolio_pct_max"],
    )

    leverage = _snap_leverage(raw_leverage, snap_values)
    portfolio_pct = max(band["portfolio_pct_min"], min(100, round(raw_pct)))

    return LeverageRecommendation(
        leverage=leverage,
        portfolio_pct=portfolio_pct,
        trade_type=trade_type,
        tier=tier.value,
        conviction=conviction,
    )
