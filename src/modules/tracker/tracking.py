import math
from collections import deque
from datetime import datetime


class Track:
    def __init__(self, track_id: int, box: list[float], now: float):
        self.id = track_id
        self.box = box
        self.last_seen = now
        self.last_helmet_boxes: list[tuple[list[float], int | str]] = []
        self.streak_start: datetime | None = None
        self.samples: deque[tuple[datetime, bool]] = deque(maxlen=240)
        self.helmet_streak = 0
        self.helmet_streak_start: datetime | None = None
        self.colors: deque[str] = deque(maxlen=20)
        self.event_id: int | None = None
        self.event_started: datetime | None = None
        self.last_persisted = 0.0


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-6)


def areas_centers(first: list[float], second: list[float]):
    def geometry(box):
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        return width * height, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), math.hypot(width, height)

    return geometry(first), geometry(second)


def near_duplicate(first: list[float], second: list[float]) -> bool:
    (area_first, center_first, diagonal_first), (area_second, center_second, diagonal_second) = areas_centers(
        first, second
    )
    if min(area_first, area_second) <= 0:
        return False
    overlap = iou(first, second)
    center_delta = math.dist(center_first, center_second) / max(min(diagonal_first, diagonal_second), 1.0)
    if overlap >= 0.4 and center_delta <= 0.28:
        return True
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    contained = max(0.0, x2 - x1) * max(0.0, y2 - y1) / min(area_first, area_second)
    return contained >= 0.72 and center_delta <= 0.42


def association_score(first: list[float], second: list[float]) -> float:
    overlap = iou(first, second)
    if overlap >= 0.08:
        return overlap
    (area_first, center_first, diagonal_first), (area_second, center_second, diagonal_second) = areas_centers(
        first, second
    )
    if min(area_first, area_second) <= 0:
        return 0.0
    size_ratio = min(area_first, area_second) / max(area_first, area_second)
    center_delta = math.dist(center_first, center_second) / max(diagonal_first, diagonal_second, 1.0)
    if size_ratio >= 0.4 and center_delta <= 0.5:
        return 0.08 + (0.5 - center_delta) * 0.1
    return 0.0
