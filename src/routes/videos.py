import subprocess
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src import state
from src.config import VIDEO_ENCODER
from src.modules.analytics.analytics import get_zone, resolve_zone
from src.modules.inference import OpenCVFrameSource
from src.modules.tracker.tracker import UltralyticsTracker, ViolationTracker

router = APIRouter(prefix="/api/videos", tags=["videos"])
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def process(job_id: str, path: Path, filename: str, zone_id: int | None):
    if state.service is None:
        raise RuntimeError("Inference model is not loaded")
    source = OpenCVFrameSource(str(path))
    fps = source.capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(source.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(source.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output = state.UPLOAD_DIRECTORY / f"{job_id}_annotated.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        source.close()
        raise ValueError("Не удалось создать выходное видео")
    zone = resolve_zone("video", filename, zone_id)
    tracker = ViolationTracker("video", f"zone:{zone['id']}" if zone else f"video:{job_id}", zone)
    object_tracker = UltralyticsTracker()
    started = datetime.now(UTC)
    frame_index = sampled = people = helmets = with_helmet = violations = 0
    last_analysis = None
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            if frame_index % state.VIDEO_FRAME_STEP == 0:
                analysis = state.service.analyze(frame, tracker=object_tracker)
                tracker.observe(analysis, started + timedelta(seconds=frame_index / max(fps, 1)), frame)
                last_analysis = analysis
                counts = analysis["counts"]
                people += counts["people"]
                helmets += counts["helmets"]
                with_helmet += counts["people_with_helmet"]
                violations += counts["people_without_helmet"]
                sampled += 1
            if last_analysis is not None:
                frame = state.service.annotate(frame, last_analysis)
            writer.write(frame)
            frame_index += 1
            if frame_index % 10 == 0:
                with jobs_lock:
                    jobs[job_id]["progress"] = round(frame_index / total, 4) if total else None
                    jobs[job_id]["processed_frames"] = frame_index
    finally:
        tracker.close_all("source_stopped", started + timedelta(seconds=frame_index / max(fps, 1)))
        source.close()
        writer.release()
    h264 = state.UPLOAD_DIRECTORY / f"{job_id}_h264.mp4"
    encoder = VIDEO_ENCODER
    args = ["-c:v", encoder, "-pix_fmt", "yuv420p"]
    args += ["-preset", "p4", "-cq", "23"] if encoder == "h264_nvenc" else ["-preset", "fast", "-crf", "23"]
    try:
        converted = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(output),
                *args,
                "-movflags",
                "+faststart",
                str(h264),
            ],
            capture_output=True,
            timeout=900,
        )
        if converted.returncode != 0 and encoder != "libx264":
            converted = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(output),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(h264),
                ],
                capture_output=True,
                timeout=900,
            )
        if converted.returncode == 0 and h264.exists():
            output.unlink(missing_ok=True)
            h264.replace(output)
        else:
            h264.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        h264.unlink(missing_ok=True)
    with jobs_lock:
        jobs[job_id].update(
            status="completed",
            progress=1.0,
            report={
                "sampled_frames": sampled,
                "video_frames": frame_index,
                "frame_step": state.VIDEO_FRAME_STEP,
                "counts_across_sampled_frames": {
                    "people": people,
                    "helmets": helmets,
                    "people_with_helmet": with_helmet,
                    "people_without_helmet": violations,
                },
            },
            result_url=f"/api/videos/{job_id}/result",
            filename=filename,
        )


def run_job(job_id: str, path: Path, filename: str, zone_id: int | None):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"
        process(job_id, path, filename, zone_id)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))
    finally:
        path.unlink(missing_ok=True)


@router.post("", status_code=202)
async def submit(file: Annotated[UploadFile, File()], zone_id: int | None = Query(None)):
    data = await file.read(state.MAX_UPLOAD_BYTES + 1)
    if len(data) > state.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds upload limit")
    if zone_id is not None and (get_zone(zone_id) or {}).get("source_type") != "video":
        raise HTTPException(status_code=422, detail="zone_id must refer to a video zone")
    job_id = uuid.uuid4().hex
    filename = file.filename or "video.mp4"
    path = state.UPLOAD_DIRECTORY / f"{job_id}{Path(filename).suffix or '.mp4'}"
    path.write_bytes(data)
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "processed_frames": 0}
    state.executor.submit(run_job, job_id, path, filename, zone_id)
    return {"id": job_id, "status_url": f"/api/videos/{job_id}", "result_url": f"/api/videos/{job_id}/result"}


@router.get("/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        return dict(job)


@router.get("/{job_id}/result")
def result(job_id: str, preview: bool = False):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"Video job is {job['status']}")
    return FileResponse(
        state.UPLOAD_DIRECTORY / f"{job_id}_annotated.mp4",
        media_type="video/mp4",
        filename=f"{Path(job['filename']).stem}_annotated.mp4",
        content_disposition_type="inline" if preview else "attachment",
    )
