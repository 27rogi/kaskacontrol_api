from fastapi import APIRouter, HTTPException

from src import state

router = APIRouter(prefix="/api/rtsp/monitors", tags=["monitors"])


@router.get("")
async def list_monitors():
    return state.rtsp_manager.statuses() if state.rtsp_manager else []


@router.post("/{zone_id}/start")
async def start(zone_id: int):
    if state.rtsp_manager is None:
        raise HTTPException(status_code=503, detail="RTSP manager is not ready")
    try:
        return await state.rtsp_manager.start(zone_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{zone_id}/stop")
async def stop(zone_id: int):
    if state.rtsp_manager is None:
        raise HTTPException(status_code=503, detail="RTSP manager is not ready")
    if not await state.rtsp_manager.stop(zone_id):
        raise HTTPException(status_code=404, detail="RTSP monitor not found")
    return {"status": "stopped", "zone_id": zone_id}
