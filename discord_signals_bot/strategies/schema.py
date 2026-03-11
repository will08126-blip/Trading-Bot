"""
Signal schema — dataclasses used throughout the signals bot.
Intentionally kept dependency-free (stdlib only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalTier(str, Enum):
    NO_TRADE = "no_trade"
    MEDIUM = "medium"
    STRONG = "strong"
    ELITE = "elite"


@dataclass
class StrategySignal:
    """Output from a single strategy family evaluation."""
    strategy_family: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    confidence: float                         # 0.0 – 1.0

    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None          # primary TP

    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validity_minutes: int = 30

    reason_codes: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    regime_context: Optional[str] = None


@dataclass
class VoteResult:
    """Output from the voting engine for one symbol + direction."""
    symbol: str
    direction: SignalDirection
    final_score: float                        # 0 – 100
    tier: SignalTier
    contributing_signals: List[StrategySignal]

    trend_alignment_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    session_score: float = 0.0
    strategy_signal_score: float = 0.0

    spread_penalty: float = 0.0
    performance_factor: float = 1.0

    reason_codes: List[str] = field(default_factory=list)
    voted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Aggregated price levels from contributing signals
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp1_price: Optional[float] = None
    tp2_price: Optional[float] = None


@dataclass
class TradeSignal:
    """
    Final signal posted to Discord — combines vote result with leverage recommendation.
    """
    symbol: str
    direction: SignalDirection
    tier: SignalTier
    score: float

    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float

    leverage: int
    portfolio_pct: int

    strategy_names: List[str]
    reason_codes: List[str]
    regime: str

    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def r_to_tp1(self) -> float:
        if self.risk_per_unit == 0:
            return 0.0
        return abs(self.tp1_price - self.entry_price) / self.risk_per_unit

    @property
    def r_to_tp2(self) -> float:
        if self.risk_per_unit == 0:
            return 0.0
        return abs(self.tp2_price - self.entry_price) / self.risk_per_unit
