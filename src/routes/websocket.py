import asyncio
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src import state
from src.auth import equal, token
from src.modules.analytics.analytics import resolve_zone
from src.modules.inference import encode_jpeg
from src.modules.tracker.tracker import UltralyticsTracker, ViolationTracker

router = APIRouter()


async def auth_socket(socket: WebSocket) -> bool:
    value = socket.query_params.get("token") or socket.headers.get("authorization", "").removeprefix("Bearer ")
    if equal(value, token()):
        return True
    await socket.close(code=1008, reason="Authentication required")
    return False


@router.websocket("/api/ws/camera")
async def camera(socket: WebSocket):
    if not await auth_socket(socket):
        return
    await socket.accept()
    if state.service is None:
        await socket.close(code=1013, reason="Model is not loaded")
        return
    zone_id = socket.query_params.get("zone_id")
    zone = resolve_zone("webcam", "webcam", int(zone_id) if zone_id and zone_id.isdigit() else None)
    tracker = ViolationTracker("webcam", f"webcam:{zone['id']}" if zone else f"webcam:{uuid.uuid4().hex}", zone)
    object_tracker = UltralyticsTracker()
    try:
        while True:
            payload = await socket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await socket.send_json({"error": "Invalid JPEG frame"})
                continue
            analysis = await asyncio.to_thread(state.service.analyze, frame, tracker=object_tracker)
            await asyncio.to_thread(tracker.observe, analysis, None, frame)
            annotated = state.service.annotate(frame, analysis)
            analysis["annotated_image"] = "data:image/jpeg;base64," + encode_jpeg(annotated, quality=75)
            await socket.send_json(analysis)
    except WebSocketDisconnect:
        pass
    finally:
        await asyncio.to_thread(tracker.close_all, "source_stopped")


@router.websocket("/api/ws/rtsp")
async def rtsp(socket: WebSocket):
    if not await auth_socket(socket):
        return
    await socket.accept()
    if state.rtsp_manager is None:
        await socket.close(code=1013, reason="RTSP manager is not ready")
        return
    try:
        config = await asyncio.wait_for(socket.receive_json(), timeout=10)
        zone_id = config.get("zone_id")
        if zone_id is not None:
            await state.rtsp_manager.start(int(zone_id))
            key = f"zone:{int(zone_id)}"
        else:
            source = config.get("source")
            if not isinstance(source, str) or not source.lower().startswith("rtsp://"):
                await socket.close(code=1008, reason="Provide zone_id or RTSP source URL")
                return
            monitor = await state.rtsp_manager.start_source(source)
            key = monitor.key
        await socket.send_json({"status": "connected", "monitor": state.rtsp_manager.status(key)})
        seen = 0.0
        while True:
            monitor = state.rtsp_manager.monitors.get(key)
            snapshot = state.rtsp_manager.snapshot(key)
            if monitor and snapshot is not None and monitor.updated_at != seen:
                await socket.send_json(snapshot)
                seen = monitor.updated_at
            await asyncio.sleep(0.08)
    except (TimeoutError, WebSocketDisconnect):
        pass
