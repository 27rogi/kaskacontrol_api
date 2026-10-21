from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import av
import av.error

from src.modules.analytics.analytics import get_zone, get_zone_source, list_zones, safe_source_ref
from src.modules.inference import PyAVFrameSource, encode_jpeg
from src.modules.logger import setup_logger
from src.modules.tracker.tracker import UltralyticsTracker, ViolationTracker

logger = setup_logger()


@dataclass
class RTSPMonitor:
    key: str
    source: str
    zone: dict[str, Any] | None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    latest: dict[str, Any] | None = None
    updated_at: float = 0.0
    status: str = "stopped"
    error: str | None = None


class RTSPMonitorManager:
    def __init__(self, detector):
        self.detector = detector
        self.monitors: dict[str, RTSPMonitor] = {}
        self.lock = asyncio.Lock()

    async def start_enabled(self) -> None:
        for zone in list_zones():
            if zone.get("enabled") and zone.get("source_type") == "rtsp" and zone.get("id"):
                await self.start(zone["id"])

    async def start(self, zone_id: int, source_override: str | None = None) -> dict[str, Any]:
        zone = get_zone(zone_id)
        if not zone or zone.get("source_type") != "rtsp":
            raise ValueError("RTSP zone not found")
        source = source_override or get_zone_source(zone_id)
        if not source:
            raise ValueError("RTSP zone has no source URI")
        key = f"zone:{zone_id}"
        async with self.lock:
            current = self.monitors.get(key)
            if current and current.task and not current.task.done():
                return self.status(key)
            monitor = RTSPMonitor(key, source, zone)
            monitor.status = "starting"
            monitor.task = asyncio.create_task(self.run_monitor(monitor), name=f"rtsp-{zone_id}")
            self.monitors[key] = monitor
            return self.status(key)

    async def start_source(self, source: str) -> RTSPMonitor:
        key = f"source:{safe_source_ref(source)}"
        async with self.lock:
            current = self.monitors.get(key)
            if current and current.task and not current.task.done():
                return current
            monitor = RTSPMonitor(key, source, None)
            monitor.status = "starting"
            monitor.task = asyncio.create_task(self.run_monitor(monitor), name="rtsp-viewer")
            self.monitors[key] = monitor
            return monitor

    async def stop(self, zone_id: int) -> bool:
        return await self.stop_key(f"zone:{zone_id}")

    async def stop_key(self, key: str) -> bool:
        monitor = self.monitors.get(key)
        if not monitor:
            return False
        monitor.stop_event.set()
        if monitor.task and not monitor.task.done():
            monitor.task.cancel()
            try:
                await monitor.task
            except (asyncio.CancelledError, Exception):
                pass
        monitor.status = "stopped"
        return True

    async def shutdown(self) -> None:
        for key in list(self.monitors):
            await self.stop_key(key)

    def status(self, key: str) -> dict[str, Any]:
        monitor = self.monitors.get(key)
        if monitor is None:
            return {"key": key, "status": "stopped", "error": None}
        return {
            "key": key,
            "zone_id": monitor.zone.get("id") if monitor.zone else None,
            "source": safe_source_ref(monitor.source),
            "status": monitor.status,
            "error": monitor.error,
            "updated_at": monitor.updated_at,
        }

    def statuses(self) -> list[dict[str, Any]]:
        return [self.status(key) for key in self.monitors]

    def snapshot(self, key: str) -> dict[str, Any] | None:
        monitor = self.monitors.get(key)
        return monitor.latest if monitor else None

    async def run_monitor(self, monitor: RTSPMonitor) -> None:
        tracker = ViolationTracker("rtsp", monitor.key, monitor.zone)
        object_tracker = UltralyticsTracker()
        source = None
        try:
            while not monitor.stop_event.is_set():
                try:
                    source = await asyncio.to_thread(PyAVFrameSource, monitor.source)
                    monitor.status, monitor.error = "running", None
                    logger.info(
                        "RTSP подключён | транспорт=%s | источник=%s | зона=%s",
                        getattr(source, "transport", "unknown"),
                        safe_source_ref(monitor.source),
                        monitor.zone["name"] if monitor.zone else "Без зоны",
                    )
                    while not monitor.stop_event.is_set():
                        frame = await asyncio.to_thread(source.read)
                        if frame is None:
                            raise RuntimeError("RTSP source ended or returned an invalid frame")
                        analysis = await asyncio.to_thread(self.detector.analyze, frame, tracker=object_tracker)
                        await asyncio.to_thread(tracker.observe, analysis, None, frame)
                        annotated = await asyncio.to_thread(self.detector.annotate, frame, analysis)
                        analysis["annotated_image"] = "data:image/jpeg;base64," + encode_jpeg(annotated, quality=72)
                        monitor.latest = analysis
                        monitor.updated_at = time.monotonic()
                except asyncio.CancelledError:
                    raise
                except (av.error.FFmpegError, OSError, ValueError, RuntimeError) as exc:
                    monitor.status, monitor.error = "error", type(exc).__name__
                    logger.warning(
                        "RTSP монитор | источник=%s | %s: %s",
                        safe_source_ref(monitor.source),
                        type(exc).__name__,
                        exc,
                    )
                    if not monitor.stop_event.is_set():
                        await asyncio.sleep(2)
                except Exception as exc:
                    monitor.status, monitor.error = "error", type(exc).__name__
                    logger.exception(
                        "Ошибка обработки RTSP | источник=%s | %s",
                        safe_source_ref(monitor.source),
                        exc,
                    )
                    if not monitor.stop_event.is_set():
                        await asyncio.sleep(2)
                finally:
                    if source is not None:
                        try:
                            await asyncio.to_thread(source.close)
                        except Exception:
                            pass
                        source = None
        finally:
            tracker.close_all("source_stopped")
            monitor.status = "stopped"
