from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.schemas.base import DatabaseBase


class Zone(DatabaseBase):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), default="rtsp", index=True)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class ZoneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    source_type: Literal["rtsp", "webcam", "video"] = "rtsp"
    source_uri: str | None = None
    enabled: bool = True


class ZoneUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = None
    source_type: Literal["rtsp", "webcam", "video"] | None = None
    source_uri: str | None = None
    enabled: bool | None = None
