from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class CooldownState(Base, TimestampMixin):
    __tablename__ = "cooldown_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    scope: Mapped[str] = mapped_column(String(20), nullable=False)  # global | strategy | symbol
    scope_key: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # strategy name or symbol
    reason: Mapped[str] = mapped_column(String(100), nullable=False)
    consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    paused_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
