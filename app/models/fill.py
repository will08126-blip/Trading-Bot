from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class Fill(Base, TimestampMixin):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    trade_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    exchange_order_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    exchange_fill_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(5), nullable=False)  # buy | sell
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_qty: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
