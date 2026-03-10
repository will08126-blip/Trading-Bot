from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Boolean, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class Trade(Base, TimestampMixin):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(10), nullable=False)  # live | paper | backtest
    direction: Mapped[str] = mapped_column(String(5), nullable=False)  # long | short
    strategy_family: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_tier: Mapped[str] = mapped_column(String(20), nullable=False)  # medium|strong|elite

    # Entry
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    notional_usd: Mapped[float] = mapped_column(Float, nullable=False)
    leverage: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # Risk parameters
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_distance_pct: Mapped[float] = mapped_column(Float, nullable=False)
    capital_risked_usd: Mapped[float] = mapped_column(Float, nullable=False)

    # Exit
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # tp|sl|manual|timeout

    # PnL
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")
    # open | closed | cancelled | error

    # Context
    regime_at_entry: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    session_at_entry: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    reason_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    hold_duration_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Exchange order IDs
    entry_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sl_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tp_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
