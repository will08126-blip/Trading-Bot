from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class Order(Base, TimestampMixin):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    trade_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # market|limit|stop|tp
    side: Mapped[str] = mapped_column(String(5), nullable=False)  # buy|sell
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # None for market
    stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    filled_qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending|open|partial|filled|cancelled|rejected|expired
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="entry")
    # entry|stop_loss|take_profit|close
    placed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
