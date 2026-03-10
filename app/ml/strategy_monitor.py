"""
Strategy Performance Monitor

Tracks rolling performance metrics per strategy family.
Can temporarily disable strategies based on performance thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade import Trade
from app.models.strategy_health import StrategyHealth

logger = structlog.get_logger(__name__)

STRATEGY_NAMES = [
    "trend_pullback",
    "breakout_retest",
    "liquidity_sweep_reversal",
    "volatility_expansion",
]


@dataclass
class StrategyStats:
    name: str
    window_trades: int = 0
    win_rate: float = 0.5
    profit_factor: float = 1.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_losses: int = 0
    is_disabled: bool = False
    disabled_until: Optional[datetime] = None
    performance_factor: float = 1.0   # used by voting engine

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "window_trades": self.window_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "consecutive_losses": self.consecutive_losses,
            "is_disabled": self.is_disabled,
            "performance_factor": self.performance_factor,
        }


class StrategyPerformanceMonitor:
    """
    Monitors rolling strategy performance and enforces cooldowns.
    """

    def __init__(
        self,
        db: AsyncSession,
        mode: str = "paper",
        window_days: int = 14,
        min_trades: int = 5,
        min_win_rate: float = 0.35,
        min_profit_factor: float = 0.6,
        max_consecutive_losses: int = 4,
        cooldown_hours: float = 2.0,
    ) -> None:
        self._db = db
        self._mode = mode
        self._window_days = window_days
        self._min_trades = min_trades
        self._min_win_rate = min_win_rate
        self._min_profit_factor = min_profit_factor
        self._max_consecutive_losses = max_consecutive_losses
        self._cooldown_hours = cooldown_hours
        self._stats: Dict[str, StrategyStats] = {
            name: StrategyStats(name=name) for name in STRATEGY_NAMES
        }

    async def refresh(self) -> None:
        """Recompute stats for all strategies from DB."""
        for name in STRATEGY_NAMES:
            stats = await self._compute_stats(name)
            self._stats[name] = stats
            await self._save_health(stats)

    async def _compute_stats(self, strategy_name: str) -> StrategyStats:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._window_days)
        result = await self._db.execute(
            select(Trade).where(
                Trade.strategy_family == strategy_name,
                Trade.mode == self._mode,
                Trade.status == "closed",
                Trade.exit_time >= cutoff,
            )
        )
        trades = result.scalars().all()

        stats = StrategyStats(name=strategy_name)
        if len(trades) < self._min_trades:
            return stats  # not enough data

        pnls = [t.realized_pnl or 0.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        stats.window_trades = len(trades)
        stats.win_rate = len(wins) / len(pnls) if pnls else 0.5
        stats.avg_win = sum(wins) / len(wins) if wins else 0.0
        stats.avg_loss = sum(losses) / len(losses) if losses else 0.0
        stats.expectancy = stats.win_rate * stats.avg_win + (1 - stats.win_rate) * stats.avg_loss

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        # Count recent consecutive losses
        sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)
        consecutive = 0
        for t in reversed(sorted_trades):
            if (t.realized_pnl or 0.0) <= 0:
                consecutive += 1
            else:
                break
        stats.consecutive_losses = consecutive

        # Check if disabled
        existing_health = await self._db.execute(
            select(StrategyHealth).where(
                StrategyHealth.strategy_name == strategy_name,
                StrategyHealth.symbol == None,
            ).order_by(StrategyHealth.computed_at.desc()).limit(1)
        )
        last_health = existing_health.scalar_one_or_none()
        if last_health and last_health.is_disabled and last_health.disabled_until:
            if last_health.disabled_until > datetime.now(timezone.utc):
                stats.is_disabled = True
                stats.disabled_until = last_health.disabled_until

        # Check if should be disabled
        if not stats.is_disabled:
            if (stats.win_rate < self._min_win_rate or
                stats.profit_factor < self._min_profit_factor or
                stats.consecutive_losses >= self._max_consecutive_losses):
                stats.is_disabled = True
                stats.disabled_until = datetime.now(timezone.utc) + timedelta(hours=self._cooldown_hours)
                logger.warning("strategy_disabled",
                              name=strategy_name,
                              win_rate=stats.win_rate,
                              profit_factor=stats.profit_factor,
                              consecutive_losses=stats.consecutive_losses,
                              until=stats.disabled_until.isoformat())

        # Performance factor for voting engine
        wr_factor = stats.win_rate / 0.5  # 1.0 at 50% win rate
        pf_factor = min(stats.profit_factor / 1.0, 2.0)
        stats.performance_factor = max(0.3, min(1.5, (wr_factor + pf_factor) / 2))

        return stats

    async def _save_health(self, stats: StrategyStats) -> None:
        self._db.add(StrategyHealth(
            strategy_name=stats.name,
            symbol=None,
            window_trades=stats.window_trades,
            win_rate=stats.win_rate,
            profit_factor=stats.profit_factor,
            expectancy=stats.expectancy,
            avg_win=stats.avg_win,
            avg_loss=stats.avg_loss,
            consecutive_losses=stats.consecutive_losses,
            is_disabled=stats.is_disabled,
            disabled_until=stats.disabled_until,
        ))
        try:
            await self._db.commit()
        except Exception:
            await self._db.rollback()

    def get_stats(self, strategy_name: str) -> StrategyStats:
        return self._stats.get(strategy_name, StrategyStats(name=strategy_name))

    def is_disabled(self, strategy_name: str) -> bool:
        stats = self._stats.get(strategy_name)
        if not stats:
            return False
        if not stats.is_disabled:
            return False
        if stats.disabled_until and stats.disabled_until <= datetime.now(timezone.utc):
            stats.is_disabled = False
            return False
        return True

    def get_all_health_dict(self) -> Dict[str, Dict]:
        return {name: stats.to_dict() for name, stats in self._stats.items()}
