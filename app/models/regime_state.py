from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class RegimeState(Base, TimestampMixin):
    __tablename__ = "regime_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    regime: Mapped[str] = mapped_column(String(30), nullable=False)
    # trending | ranging | volatility_expansion | compression | unknown
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    classified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # input features snapshot
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
