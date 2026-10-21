from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import func, select

from src.config import VIOLATION_IMAGE_URL
from src.modules.db.database import SessionLocal, db_lock
from src.modules.helmet_colors import role_for_color
from src.modules.snapshots import VIOLATION_IMAGE_DIR
from src.schemas.violation import ViolationEvent
from src.schemas.zone import Zone


def zone_dict(zone: Zone) -> dict[str, Any]:
    source_uri = zone.source_uri
    if source_uri:
        parts = urlsplit(source_uri)
        if parts.username or parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            source_uri = urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))
    return {
        "id": zone.id,
        "name": zone.name,
        "description": zone.description,
        "source_type": zone.source_type,
        "source_uri": source_uri,
        "enabled": zone.enabled,
        "created_at": zone.created_at.isoformat(),
        "updated_at": zone.updated_at.isoformat(),
    }


def list_zones() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        return [zone_dict(zone) for zone in session.scalars(select(Zone).order_by(Zone.id))]


def get_zone(zone_id: int) -> dict[str, Any] | None:
    with SessionLocal() as session:
        zone = session.get(Zone, zone_id)
        return zone_dict(zone) if zone else None


def get_zone_source(zone_id: int) -> str | None:
    with SessionLocal() as session:
        zone = session.get(Zone, zone_id)
        return zone.source_uri if zone else None


def create_zone(data: dict[str, Any]) -> dict[str, Any]:
    with db_lock, SessionLocal() as session:
        zone = Zone(**data)
        session.add(zone)
        session.commit()
        session.refresh(zone)
        return zone_dict(zone)


def update_zone(zone_id: int, data: dict[str, Any]) -> dict[str, Any] | None:
    with db_lock, SessionLocal() as session:
        zone = session.get(Zone, zone_id)
        if zone is None:
            return None
        for key, value in data.items():
            setattr(zone, key, value)
        session.commit()
        session.refresh(zone)
        return zone_dict(zone)


def delete_zone(zone_id: int) -> bool:
    with db_lock, SessionLocal() as session:
        zone = session.get(Zone, zone_id)
        if zone is None:
            return False
        for event in session.scalars(select(ViolationEvent).where(ViolationEvent.zone_id == zone_id)):
            event.zone_id = None
        session.delete(zone)
        session.commit()
        return True


def resolve_zone(source_type: str, source_ref: str, zone_id: int | None = None) -> dict[str, Any] | None:
    with SessionLocal() as session:
        zone = (
            session.get(Zone, zone_id)
            if zone_id is not None
            else session.scalar(
                select(Zone).where(
                    Zone.source_type == source_type, Zone.source_uri == source_ref, Zone.enabled.is_(True)
                )
            )
        )
        if zone is None or not zone.enabled or zone.source_type != source_type:
            return None
        return zone_dict(zone)


def safe_source_ref(source: str) -> str:
    parts = urlsplit(source)
    if not parts.scheme:
        return source
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def event_dict(event: ViolationEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "event_type": "helmet_not_detected",
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "source_type": event.source_type,
        "source_ref": event.source_ref,
        "track_id": event.track_id,
        "status": event.status,
        "helmet_color": event.helmet_color,
        "helmet_role": event.helmet_role or role_for_color(event.helmet_color),
        "started_at": event.started_at.isoformat(),
        "last_seen_at": event.last_seen_at.isoformat(),
        "ended_at": event.ended_at.isoformat() if event.ended_at else None,
        "duration_seconds": round(event.duration_seconds, 2),
        "close_reason": event.close_reason,
        "start_image_path": event.start_image_path,
        "end_image_path": event.end_image_path,
    }


def list_events(
    *,
    zone_id: int | None = None,
    source_type: str | None = None,
    status: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    query = select(ViolationEvent)
    if zone_id is not None:
        query = query.where(ViolationEvent.zone_id == zone_id)
    if source_type:
        query = query.where(ViolationEvent.source_type == source_type)
    if status:
        query = query.where(ViolationEvent.status == status)
    if since:
        query = query.where(ViolationEvent.started_at >= since)
    if until:
        query = query.where(ViolationEvent.started_at <= until)
    query = query.order_by(ViolationEvent.started_at.desc()).limit(limit).offset(offset)
    with SessionLocal() as session:
        return [event_dict(event) for event in session.scalars(query)]


def analytics_summary() -> dict[str, Any]:
    with SessionLocal() as session:
        open_count = (
            session.scalar(select(func.count()).select_from(ViolationEvent).where(ViolationEvent.status == "open")) or 0
        )
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = (
            session.scalar(select(func.count()).select_from(ViolationEvent).where(ViolationEvent.started_at >= today))
            or 0
        )
        by_zone = session.execute(
            select(ViolationEvent.zone_name, func.count())
            .where(ViolationEvent.started_at >= today)
            .group_by(ViolationEvent.zone_name)
        ).all()
        return {
            "open_events": open_count,
            "events_today": today_count,
            "events_today_by_zone": {name or "Без зоны": count for name, count in by_zone},
        }


def save_event(
    zone: dict[str, Any] | None,
    source_type: str,
    source_ref: str,
    track_id: int,
    started_at: datetime,
    last_seen_at: datetime,
    color: str,
    role: str | None = None,
    event_id: int | None = None,
    ended_at: datetime | None = None,
    close_reason: str | None = None,
) -> int:
    with db_lock, SessionLocal() as session:
        event = session.get(ViolationEvent, event_id) if event_id else None
        if event is None:
            event = ViolationEvent(
                zone_id=zone["id"] if zone else None,
                zone_name=zone["name"] if zone else None,
                source_type=source_type,
                source_ref=source_ref,
                track_id=track_id,
                started_at=started_at,
                last_seen_at=last_seen_at,
                helmet_color=color,
                helmet_role=role or role_for_color(color),
            )
            session.add(event)
        event.last_seen_at = last_seen_at
        event.helmet_color = color
        event.helmet_role = role or role_for_color(color)
        event.duration_seconds = max(0.0, (last_seen_at - started_at).total_seconds())
        if ended_at:
            event.status = "closed"
            event.ended_at = ended_at
            event.close_reason = close_reason
        session.commit()
        session.refresh(event)
        return event.id


def event_by_id(event_id: int) -> dict[str, Any] | None:
    with SessionLocal() as session:
        event = session.get(ViolationEvent, event_id)
        return event_dict(event) if event else None


def set_event_snapshot(event_id: int, stage: str, path: str | None) -> None:
    attribute = "start_image_path" if stage == "start" else "end_image_path"
    with db_lock, SessionLocal() as session:
        event = session.get(ViolationEvent, event_id)
        if event is not None:
            setattr(event, attribute, path)
            session.commit()


def discard_duplicate_event(event_id: int) -> None:
    snapshot_paths: list[str] = []
    with db_lock, SessionLocal() as session:
        event = session.get(ViolationEvent, event_id)
        if event is None:
            return
        snapshot_paths = [path for path in (event.start_image_path, event.end_image_path) if path is not None]
        session.delete(event)
        session.commit()
    for snapshot in snapshot_paths:
        if snapshot and snapshot.startswith(f"{VIOLATION_IMAGE_URL}/"):
            (VIOLATION_IMAGE_DIR / Path(snapshot).name).unlink(missing_ok=True)
