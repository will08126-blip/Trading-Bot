from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class StrategyHealth(Base, TimestampMixin):
    __tablename__ = "strategy_health"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    # None = global, not symbol-specific
    window_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    profit_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    expectancy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_win: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_slippage_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_disabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    disabled_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
