from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.schemas.base import DatabaseBase


class ViolationEvent(DatabaseBase):
    __tablename__ = "violation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id", ondelete="SET NULL"), nullable=True, index=True)
    zone_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), index=True)
    source_ref: Mapped[str] = mapped_column(Text)
    track_id: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(12), default="open", index=True)
    helmet_color: Mapped[str] = mapped_column(String(20), default="unknown")
    helmet_role: Mapped[str] = mapped_column(String(80), default="Не определена")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0)
    close_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    start_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    end_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
