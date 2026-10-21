from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.modules.analytics.analytics import analytics_summary, event_by_id, list_events
from src.modules.helmet_colors import role_config

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/events")
def events(
    zone_id: int | None = None,
    source_type: Literal["rtsp", "webcam", "video"] | None = None,
    status: Literal["open", "closed"] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return list_events(
        zone_id=zone_id, source_type=source_type, status=status, since=since, until=until, limit=limit, offset=offset
    )


@router.get("/events/{event_id}")
def event(event_id: int):
    value = event_by_id(event_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Analytics event not found")
    return value


@router.get("/summary")
def summary():
    return analytics_summary()


@router.get("/helmet-colors")
def helmet_colors():
    return role_config()
