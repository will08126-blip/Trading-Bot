from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Boolean, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class Signal(Base, TimestampMixin):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    strategy_family: Mapped[str] = mapped_column(String(50), nullable=False)
    direction: Mapped[str] = mapped_column(String(5), nullable=False)  # long | short
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_tier: Mapped[str] = mapped_column(String(20), nullable=False)

    # Price levels
    entry_price_suggestion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_suggestion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp_suggestion: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Context
    regime: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    session: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    validity_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Whether the signal led to a trade
    acted_on: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trade_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Detailed breakdown (JSON)
    score_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason_codes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")
