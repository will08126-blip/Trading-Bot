from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class ScoreBreakdown(Base, TimestampMixin):
    __tablename__ = "score_breakdowns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    signal_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(5), nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    trend_alignment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    momentum_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    volatility_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liquidity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    session_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    strategy_signal_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_penalty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage_penalty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    performance_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    ml_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
