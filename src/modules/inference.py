from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import av
import av.error
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.config import FONT_PATH, YOLO_WEIGHTS, configured_path
from src.modules.logger import setup_logger

logger = setup_logger()


def check_box_collision(person_box, helmet_box, threshold: float = 0.3) -> bool:
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box
    ix1, iy1, ix2, iy2 = max(px1, hx1), max(py1, hy1), min(px2, hx2), min(py2, hy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    helmet_area = (hx2 - hx1) * (hy2 - hy1)
    return helmet_area > 0 and ((ix2 - ix1) * (iy2 - iy1) / helmet_area) >= threshold


def classify_helmet_color(frame: np.ndarray, box) -> tuple[str, dict | None]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 4 or box_height < 4:
        return "unknown", None

    crop = frame[y1 + box_height // 4 : y2 - box_height // 4, x1 + box_width // 4 : x2 - box_width // 4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv.reshape(-1, 3).T
    white = (s <= 55) & (v >= 150) & (v <= 250)
    if white.mean() >= 0.35:
        return "white", {"h": None, "s": int(np.median(s[white])), "v": int(np.median(v[white]))}

    usable = (s >= 70) & (v >= 55) & (v <= 245)
    if usable.sum() < 3:
        return "unknown", None
    hue = h[usable]
    red = ((hue <= 10) | (hue >= 170)).mean()
    orange = ((hue >= 11) & (hue <= 24)).mean()
    yellow = ((hue >= 25) & (hue <= 38)).mean()
    scores = {"red": red, "yellow": yellow, "orange": orange}
    color, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.25:
        return "unknown", None
    return color, {"h": int(np.median(hue)), "s": int(np.median(s[usable])), "v": int(np.median(v[usable]))}


class FrameSource:
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class OpenCVFrameSource(FrameSource):
    def __init__(self, source):
        self.source = source
        self.capture = self.open_capture()
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("Не удалось открыть источник кадров")

    def open_capture(self):
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def read(self) -> np.ndarray | None:
        for _ in range(3):
            ok, frame = self.capture.read()
            if ok and frame is not None and frame.size and frame.shape[0] > 8 and frame.shape[1] > 8:
                return frame
            time.sleep(0.03)
        self.capture.release()
        self.capture = self.open_capture()
        return None

    def close(self):
        self.capture.release()


class PyAVFrameSource(FrameSource):
    def __init__(self, source: str):
        open_timeout = max(0.5, float(os.getenv("RTSP_OPEN_TIMEOUT_SECONDS", "3")))
        read_timeout = max(0.5, float(os.getenv("RTSP_READ_TIMEOUT_SECONDS", "5")))
        configured_transport = os.getenv("RTSP_TRANSPORT", "auto").lower()
        valid_transports = {"tcp", "udp", "udp_multicast", "http", "https"}
        transports = (
            [configured_transport] if configured_transport in valid_transports else ["tcp", "udp", "udp_multicast"]
        )
        last_error = None
        self.container = None
        for transport in transports:
            try:
                self.container = av.open(
                    source,
                    options={
                        "rtsp_transport": transport,
                        "fflags": "nobuffer+discardcorrupt",
                        "flags": "low_delay",
                        "max_delay": os.getenv("RTSP_MAX_DELAY_US", "500000"),
                        "reorder_queue_size": os.getenv("RTSP_REORDER_QUEUE_SIZE", "0"),
                    },
                    timeout=(open_timeout, read_timeout),
                )
                self.transport = transport
                break
            except Exception as exc:
                last_error = exc
        if self.container is None:
            raise last_error or RuntimeError("Не удалось открыть RTSP источник")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.frames = self.container.decode(self.stream)

    def read(self) -> np.ndarray | None:
        try:
            frame = next(self.frames)
            if frame.is_corrupt:
                return None
            image = frame.to_ndarray(format="bgr24")
            if image.size == 0 or image.shape[0] < 8 or image.shape[1] < 8:
                return None
            return image
        except (StopIteration, av.error.EOFError, av.error.InvalidDataError):
            return None

    def close(self):
        if getattr(self, "container", None) is not None:
            assert self.container is not None
            self.container.close()


@dataclass
class InferenceService:
    model: YOLO
    model_name: str
    lock: threading.Lock

    def analyze(
        self,
        frame: np.ndarray,
        *,
        tracker=None,
    ) -> dict:
        started = time.perf_counter()
        with self.lock:
            result = self.model.predict(source=frame, conf=0.25, verbose=False, classes=[0, 6])[0]
            track_ids = tracker.update(result.boxes, frame) if tracker is not None else {}
        names = result.names
        persons, helmets = [], []
        detections = []
        boxes = cast(Any, result.boxes)
        for box_index in range(len(boxes or [])):
            box = boxes[box_index]
            class_id = int(box.cls[0])
            name = str(names[class_id]).lower()
            coords = [round(float(v), 2) for v in box.xyxy[0].cpu().tolist()]
            confidence = round(float(box.conf[0]), 4)
            detection = {"class": name, "confidence": confidence, "box": coords}
            if track_id := track_ids.get(box_index):
                detection["track_id"] = track_id
            if name == "helmet":
                color, hsv = classify_helmet_color(frame, coords)
                detection["color"] = color
                detection["hsv"] = hsv
                helmets.append((coords, confidence, color))
            elif name == "person":
                persons.append((coords, confidence))
            else:
                continue
            detections.append(detection)

        people_with_helmet = 0
        for person_box, _ in persons:
            candidates = [
                (helmet_box, color) for helmet_box, _, color in helmets if check_box_collision(person_box, helmet_box)
            ]
            if candidates:
                people_with_helmet += 1
        people_count = len(persons)
        elapsed_ms = (time.perf_counter() - started) * 1000
        summary = {
            "model": self.model_name,
            "detections": detections,
            "counts": {
                "people": people_count,
                "helmets": len(helmets),
                "people_with_helmet": people_with_helmet,
                "people_without_helmet": people_count - people_with_helmet,
                "helmet_colors": {
                    color: sum(1 for _, _, found in helmets if found == color)
                    for color in ("white", "red", "yellow", "orange", "unknown")
                },
            },
            "processing_time_ms": round(elapsed_ms, 2),
        }
        return summary

    def annotate(self, frame: np.ndarray, analysis: dict) -> np.ndarray:
        output = frame.copy()
        colors = {
            "white": (245, 245, 245),
            "red": (0, 0, 230),
            "yellow": (0, 220, 255),
            "orange": (0, 140, 255),
            "unknown": (180, 180, 180),
            "person": (70, 210, 80),
        }
        for detection in analysis["detections"]:
            x1, y1, x2, y2 = [int(v) for v in detection["box"]]
            color = colors.get(detection.get("color", detection["class"]), (180, 180, 180))
            label = detection["class"]
            if detection["class"] == "helmet":
                label += f" {detection.get('color', 'unknown')}"
            thickness = max(2, round(min(output.shape[:2]) / 360))
            font_scale = max(0.6, min(output.shape[:2]) / 900)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                output,
                f"{label} {detection['confidence']:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        if analysis["counts"]["people_without_helmet"] > 0:
            warning = "НАЙДЕН ЧЕЛОВЕК БЕЗ КАСКИ"
            banner_width = min(output.shape[1], 760)
            cv2.rectangle(output, (0, 0), (banner_width, 54), (30, 30, 190), -1)
            pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(FONT_PATH, max(20, round(min(output.shape[:2]) / 42)))
            draw.text((14, 10), warning, font=font, fill=(255, 255, 255))
            output = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return output


def load_service() -> InferenceService:
    weights = YOLO_WEIGHTS or configured_path("MODELS_DIRECTORY", "models") / "YOLO26" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Не удалось найти модель по пути: {weights}")
    model = YOLO(str(weights))
    logger.info("YOLO weights loaded: %s", weights)
    return InferenceService(model, "YOLO26", threading.Lock())


def encode_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Не удалось закодировать изображение")
    return buffer.tobytes()


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    return base64.b64encode(encode_jpeg_bytes(frame, quality)).decode("ascii")
