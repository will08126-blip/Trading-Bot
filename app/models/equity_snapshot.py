from datetime import datetime
from sqlalchemy import String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class EquitySnapshot(Base, TimestampMixin):
    __tablename__ = "equity_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    total_equity: Mapped[float] = mapped_column(Float, nullable=False)
    allocated_capital: Mapped[float] = mapped_column(Float, nullable=False)
    cash_balance: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl_today: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl_total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    open_position_count: Mapped[int] = mapped_column(nullable=False, default=0)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
