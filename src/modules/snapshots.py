import secrets
from typing import Any

import cv2
import numpy as np

from src.config import UPLOAD_DIRECTORY, VIOLATION_IMAGE_URL
from src.modules.helmet_colors import visual_role_for_color
from src.modules.logger import setup_logger
from src.modules.tracker.tracking import Track, near_duplicate

logger = setup_logger()
VIOLATION_IMAGE_DIR = UPLOAD_DIRECTORY / "violations"
VIOLATION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def write_event_snapshot(
    event_id: int,
    stage: str,
    frame: np.ndarray | None,
    detections: list[dict[str, Any]],
    fallback_track: Track | None = None,
) -> str | None:
    if frame is None:
        return None
    image = frame.copy()
    seen_person_track = False
    helmet_index = 0
    for detection in detections:
        box = detection.get("box")
        track_id = detection.get("track_id")
        if not box:
            continue
        class_name = detection.get("class", "object")
        if class_name == "person":
            if track_id is None:
                continue
            seen_person_track |= fallback_track is not None and track_id == fallback_track.id
            label = f"person (id P{track_id}): {detection.get('confidence', 0):.2f}"
            color = (70, 210, 80)
        else:
            if track_id is None:
                helmet_index += 1
                track_id = f"H{helmet_index}"
            color_name = detection.get("color", "unknown")
            helmet_id = f"H{track_id}" if isinstance(track_id, int) else track_id
            label = f"helmet ({color_name}) (id {helmet_id}): {detection.get('confidence', 0):.2f}"
            label += f"\n{visual_role_for_color(color_name)}"
            color = {
                "unknown": (150, 115, 85),
                "white": (240, 240, 240),
                "red": (0, 0, 230),
                "yellow": (0, 220, 255),
                "orange": (0, 140, 255),
            }.get(color_name, (150, 115, 85))
        draw_labeled_box(image, box, label, color)
    for fallback_box, fallback_id in fallback_track.last_helmet_boxes if fallback_track else []:
        if not any(
            detection.get("box") and near_duplicate(fallback_box, detection["box"])
            for detection in detections
            if detection.get("class") == "helmet"
        ):
            draw_labeled_box(image, fallback_box, f"helmet (id H{fallback_id})", (0, 180, 255))
    if fallback_track is not None and not seen_person_track:
        draw_labeled_box(image, fallback_track.box, f"person (id P{fallback_track.id})", (70, 210, 80))
    name = f"event_{event_id}_{stage}_{secrets.token_hex(16)}.jpg"
    path = VIOLATION_IMAGE_DIR / name
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        logger.warning("Не удалось сохранить фотофиксацию события #%s (%s)", event_id, stage)
        return None
    return f"{VIOLATION_IMAGE_URL}/{name}"


def draw_labeled_box(image: np.ndarray, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    thickness = max(2, round(min(height, width) / 360))
    font_scale = max(0.55, min(height, width) / 900)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    lines = label.splitlines()
    metrics = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness) for line in lines]
    text_width = max((size[0][0] for size in metrics), default=0)
    line_height = max((size[0][1] + size[1] for size in metrics), default=0)
    label_top = max(0, y1 - line_height * len(lines) - 6)
    label_bottom = min(height - 1, label_top + line_height * len(lines) + 6)
    cv2.rectangle(image, (x1, label_top), (min(width - 1, x1 + text_width + 8), label_bottom), color, -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x1 + 4, label_top + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 25, 20),
            thickness,
            cv2.LINE_AA,
        )
