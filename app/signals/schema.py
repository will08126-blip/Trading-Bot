"""
Canonical signal schema output by each strategy module.
All strategies return a StrategySignal (or None for no signal).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalTier(str, Enum):
    NO_TRADE = "no_trade"
    MEDIUM = "medium"
    STRONG = "strong"
    ELITE = "elite"


class StrategySignal(BaseModel):
    """
    Output from a single strategy family evaluation.
    Not yet a trade decision - feeds into the weighted voting engine.
    """
    # Identity
    strategy_family: str
    symbol: str
    timeframe: str  # primary timeframe used

    # Signal
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0, description="0–1 raw confidence from this strategy")

    # Price levels
    entry_price: Optional[float] = None       # suggested entry zone center
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_price: Optional[float] = None        # suggested stop loss
    tp_price: Optional[float] = None          # suggested take profit

    # Validity
    signal_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validity_minutes: int = 30               # how long signal stays relevant

    # Explanations
    reason_codes: List[str] = Field(default_factory=list)
    key_metrics: Dict[str, Any] = Field(default_factory=dict)

    # Regime context the signal was generated under
    regime_context: Optional[str] = None

    class Config:
        use_enum_values = True
