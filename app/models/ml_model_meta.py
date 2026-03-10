from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base
from app.models.base import TimestampMixin, new_uuid, utcnow


class MLModelMeta(Base, TimestampMixin):
    __tablename__ = "ml_model_meta"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    training_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    training_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
