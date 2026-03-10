"""
Portfolio state management.
Tracks equity, positions, PnL, drawdown, cooldowns, and risk mode.
Persists to database and recovers on restart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config
from app.models.equity_snapshot import EquitySnapshot
from app.models.cooldown_state import CooldownState
from app.models.trade import Trade

logger = structlog.get_logger(__name__)


@dataclass
class PositionSummary:
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    stop_price: float
    tp_price: float
    leverage: float
    notional_usd: float
    unrealized_pnl: float
    trade_id: str


@dataclass
class PortfolioState:
    mode: str = "paper"
    total_equity: float = 0.0
    peak_equity: float = 0.0
    allocated_capital: float = 0.0
    cash_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    realized_pnl_total: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: List[PositionSummary] = field(default_factory=list)
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    risk_mode: str = "normal"  # normal | reduced | kill_switch
    regime: str = "unknown"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total_equity": self.total_equity,
            "peak_equity": self.peak_equity,
            "allocated_capital": self.allocated_capital,
            "cash_balance": self.cash_balance,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "realized_pnl_total": self.realized_pnl_total,
            "drawdown_pct": self.drawdown_pct,
            "open_position_count": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "risk_mode": self.risk_mode,
            "regime": self.regime,
            "last_updated": self.last_updated.isoformat(),
        }


class PortfolioManager:
    """
    Manages portfolio state persistence and computation.
    Reconciles with exchange state on startup.
    """

    def __init__(self, db: AsyncSession, mode: str = "paper") -> None:
        self._db = db
        self._mode = mode
        self._state = PortfolioState(mode=mode)
        self._cfg = get_trading_config()
        self._risk_cfg = self._cfg.risk
        self._cooldown_cfg = self._cfg.cooldown
        settings = get_settings()
        if mode == "paper":
            self._initial_equity = settings.paper_initial_balance
        else:
            self._initial_equity = 0.0  # will be fetched from exchange

    async def initialize(self, exchange_equity: Optional[float] = None) -> None:
        """Load state from DB and optionally reconcile with exchange."""
        if exchange_equity is not None:
            self._state.total_equity = exchange_equity
            self._state.cash_balance = exchange_equity
        elif self._mode == "paper":
            self._state.total_equity = self._initial_equity
            self._state.cash_balance = self._initial_equity
            self._state.peak_equity = self._initial_equity

        # Load today's PnL from DB
        today_pnl = await self._compute_daily_pnl()
        self._state.realized_pnl_today = today_pnl

        # Load total PnL
        total_pnl = await self._compute_total_pnl()
        self._state.realized_pnl_total = total_pnl

        # Compute drawdown
        self._update_drawdown()

        logger.info("portfolio_initialized",
                    equity=self._state.total_equity,
                    daily_pnl=today_pnl,
                    total_pnl=total_pnl)

    def get_state(self) -> PortfolioState:
        return self._state

    def get_state_dict(self) -> Dict[str, Any]:
        return self._state.to_dict()

    def update_equity(self, new_equity: float) -> None:
        self._state.total_equity = new_equity
        if new_equity > self._state.peak_equity:
            self._state.peak_equity = new_equity
        self._update_drawdown()

    def update_unrealized_pnl(self, unrealized: float) -> None:
        self._state.unrealized_pnl = unrealized

    def add_position(self, position: PositionSummary) -> None:
        self._state.open_positions.append(position)
        self._state.allocated_capital += position.notional_usd / position.leverage

    def remove_position(self, trade_id: str) -> Optional[PositionSummary]:
        for i, pos in enumerate(self._state.open_positions):
            if pos.trade_id == trade_id:
                removed = self._state.open_positions.pop(i)
                self._state.allocated_capital = max(
                    0.0,
                    self._state.allocated_capital - removed.notional_usd / removed.leverage
                )
                return removed
        return None

    def record_trade_result(self, pnl: float) -> None:
        """Update state after a trade closes."""
        self._state.realized_pnl_today += pnl
        self._state.realized_pnl_total += pnl
        self._state.total_equity += pnl

        if self._state.total_equity > self._state.peak_equity:
            self._state.peak_equity = self._state.total_equity

        self._update_drawdown()

        if pnl < 0:
            self._state.consecutive_losses += 1
            self._state.consecutive_wins = 0
        else:
            self._state.consecutive_wins += 1
            self._state.consecutive_losses = 0

        # Check risk mode after each trade
        self._update_risk_mode()

    def _update_drawdown(self) -> None:
        if self._state.peak_equity > 0:
            self._state.drawdown_pct = max(
                0.0,
                (self._state.peak_equity - self._state.total_equity) / self._state.peak_equity * 100
            )

    def _update_risk_mode(self) -> None:
        cfg = self._risk_cfg
        cooldown_cfg = self._cooldown_cfg

        # Kill switch check
        if self._state.drawdown_pct >= cfg.max_account_drawdown_pct:
            self._state.risk_mode = "kill_switch"
            logger.critical("kill_switch_triggered_drawdown",
                           drawdown=self._state.drawdown_pct)
            return

        # Daily loss check
        if self._state.total_equity > 0:
            daily_loss_pct = -self._state.realized_pnl_today / self._state.total_equity * 100
            if daily_loss_pct >= cfg.max_daily_loss_pct:
                self._state.risk_mode = "kill_switch"
                logger.critical("kill_switch_triggered_daily_loss",
                               daily_loss_pct=daily_loss_pct)
                return

        # Reduced risk mode
        if self._state.consecutive_losses >= cooldown_cfg.consecutive_losses_short:
            self._state.risk_mode = "reduced"
        else:
            if self._state.risk_mode == "reduced":
                self._state.risk_mode = "normal"

    def is_in_cooldown(self) -> bool:
        """Check global cooldown status."""
        cfg = self._cooldown_cfg
        losses = self._state.consecutive_losses
        # This is a simplified in-memory check; DB cooldowns are checked separately
        return losses >= cfg.consecutive_losses_short

    async def check_cooldown_active(self, scope: str = "global", scope_key: Optional[str] = None) -> bool:
        """Check DB for active cooldown."""
        now = datetime.now(timezone.utc)
        query = select(CooldownState).where(
            CooldownState.scope == scope,
            CooldownState.is_active == True,
            CooldownState.paused_until > now,
        )
        if scope_key:
            query = query.where(CooldownState.scope_key == scope_key)
        result = await self._db.execute(query)
        return result.scalar_one_or_none() is not None

    async def create_cooldown(
        self,
        scope: str,
        scope_key: Optional[str],
        reason: str,
        pause_hours: float,
        consecutive_losses: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        paused_until = now + timedelta(hours=pause_hours)
        self._db.add(CooldownState(
            scope=scope,
            scope_key=scope_key,
            reason=reason,
            consecutive_losses=consecutive_losses,
            paused_until=paused_until,
            is_active=True,
            triggered_at=now,
        ))
        await self._db.commit()
        logger.warning("cooldown_created", scope=scope, key=scope_key,
                       reason=reason, until=paused_until.isoformat())

    async def save_equity_snapshot(self) -> None:
        state = self._state
        self._db.add(EquitySnapshot(
            snapshot_time=datetime.now(timezone.utc),
            total_equity=state.total_equity,
            allocated_capital=state.allocated_capital,
            cash_balance=state.cash_balance,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl_today=state.realized_pnl_today,
            realized_pnl_total=state.realized_pnl_total,
            drawdown_pct=state.drawdown_pct,
            peak_equity=state.peak_equity,
            open_position_count=len(state.open_positions),
            mode=state.mode,
        ))
        try:
            await self._db.commit()
        except Exception as e:
            await self._db.rollback()
            logger.error("equity_snapshot_save_error", error=str(e))

    async def _compute_daily_pnl(self) -> float:
        today = date.today()
        result = await self._db.execute(
            select(func.sum(Trade.realized_pnl))
            .where(
                Trade.mode == self._mode,
                Trade.status == "closed",
                func.date(Trade.exit_time) == today,
            )
        )
        return float(result.scalar_one_or_none() or 0.0)

    async def _compute_total_pnl(self) -> float:
        result = await self._db.execute(
            select(func.sum(Trade.realized_pnl))
            .where(Trade.mode == self._mode, Trade.status == "closed")
        )
        return float(result.scalar_one_or_none() or 0.0)
