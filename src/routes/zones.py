from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError

from src.modules.analytics.analytics import create_zone, delete_zone, get_zone, list_zones, update_zone
from src.schemas.zone import ZoneCreate, ZoneUpdate

router = APIRouter(prefix="/api/zones", tags=["zones"])


def validate_source(source_type: str, source_uri: str | None):
    if source_type == "rtsp" and (not source_uri or not source_uri.lower().startswith("rtsp://")):
        raise HTTPException(status_code=422, detail="RTSP zone requires an rtsp:// source_uri")
    if source_type == "webcam" and source_uri not in (None, "", "webcam"):
        raise HTTPException(status_code=422, detail="Webcam source_uri must be empty or 'webcam'")


@router.get("")
def list_all():
    return list_zones()


@router.post("", status_code=201)
def create(payload: ZoneCreate):
    data = payload.model_dump()
    validate_source(data["source_type"], data["source_uri"])
    if data["source_type"] == "webcam":
        data["source_uri"] = None
    try:
        return create_zone(data)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Zone name and source_uri must be unique") from exc


@router.get("/{zone_id}")
def get(zone_id: int):
    zone = get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone


@router.patch("/{zone_id}")
def update(zone_id: int, payload: ZoneUpdate):
    current = get_zone(zone_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    changes = payload.model_dump(exclude_unset=True)
    validate_source(
        changes.get("source_type", current["source_type"]), changes.get("source_uri", current["source_uri"])
    )
    if changes.get("source_type", current["source_type"]) == "webcam" and changes.get("source_uri") == "webcam":
        changes["source_uri"] = None
    try:
        return update_zone(zone_id, changes)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Zone name and source_uri must be unique") from exc


@router.delete("/{zone_id}", status_code=204)
def remove(zone_id: int):
    if not delete_zone(zone_id):
        raise HTTPException(status_code=404, detail="Zone not found")
