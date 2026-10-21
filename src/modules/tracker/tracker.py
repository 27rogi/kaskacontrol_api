from __future__ import annotations

import math
import time
from collections import Counter, deque
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers import BYTETracker

from src.modules.analytics.analytics import discard_duplicate_event, save_event, set_event_snapshot
from src.modules.helmet_colors import role_for_color
from src.modules.inference import check_box_collision
from src.modules.logger import setup_logger
from src.modules.snapshots import write_event_snapshot
from src.modules.tracker.tracking import Track, association_score, near_duplicate

logger = setup_logger()
VIOLATION_CONFIRM_SECONDS = 1.5
TRACK_LOST_GRACE_SECONDS = 3


class UltralyticsTracker:
    def __init__(self, track_buffer: int = 30):
        self.tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=track_buffer,
                match_thresh=0.8,
                fuse_score=True,
            )
        )

    def update(self, boxes, frame: np.ndarray) -> dict[int, int]:
        if boxes is None:
            self.tracker.update(boxes, frame)
            return {}
        boxes = boxes.cpu() if hasattr(boxes, "cpu") else boxes
        if len(boxes) == 0:
            self.tracker.update(boxes, frame)
            return {}
        output = self.tracker.update(boxes, frame)
        return {int(row[7]): int(row[4]) for row in output if len(row) >= 8}


class ViolationTracker:
    def __init__(
        self,
        source_type: str,
        source_ref: str,
        zone: dict[str, Any] | None = None,
        confirm_seconds: float | None = None,
        lost_grace_seconds: float | None = None,
    ):
        self.source_type = source_type
        self.source_ref = source_ref
        self.zone = zone
        self.confirm_seconds = confirm_seconds if confirm_seconds is not None else VIOLATION_CONFIRM_SECONDS
        self.lost_grace_seconds = lost_grace_seconds if lost_grace_seconds is not None else TRACK_LOST_GRACE_SECONDS
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_source_time = datetime.now(UTC)
        self.explicit_source_clock = False
        self.last_frame: np.ndarray | None = None
        self.last_analysis: dict[str, Any] | None = None
        self.pending_snapshots: list[tuple[int, str, Track, np.ndarray | None, list[dict[str, Any]]]] | None = None

    def observe(self, analysis: dict[str, Any], at: datetime | None = None, frame: np.ndarray | None = None) -> None:
        wall_time = at or datetime.now(UTC)
        self.explicit_source_clock = at is not None
        self.last_source_time = wall_time
        now = wall_time.timestamp() if at is not None else time.monotonic()
        self.collapse_tracks(now)
        if frame is not None:
            self.last_frame = frame
        self.last_analysis = analysis
        self.pending_snapshots = []
        observations, helmets, duplicates = self.collect_observations(analysis)
        assignments = self.assign_tracks(observations, now)
        person_track_ids = {}
        for index, observation in enumerate(observations):
            person, track = self.update_track(observation, assignments.get(index), now, wall_time, frame, analysis)
            person_track_ids[id(person)] = track.id
        for duplicate, keeper in duplicates:
            duplicate["track_id"] = person_track_ids.get(id(keeper))
        self.assign_helmet_tracks(helmets, observations, person_track_ids)
        self.flush_snapshots()
        self.remove_lost(now, wall_time, frame, analysis)

    def collect_observations(self, analysis: dict[str, Any]):
        detections = analysis.get("detections", [])
        raw_people = [item for item in detections if item.get("class") == "person"]
        helmets = [item for item in detections if item.get("class") == "helmet"]
        people, duplicates = [], []
        for person in sorted(raw_people, key=lambda item: item.get("confidence", 0), reverse=True):
            keeper = next((item for item in people if near_duplicate(item["box"], person["box"])), None)
            (people if keeper is None else duplicates).append(person if keeper is None else (person, keeper))
        observations = []
        for person in people:
            associated = []
            for helmet in helmets:
                box = helmet["box"]
                if check_box_collision(person["box"], box):
                    associated.append(helmet)
            colors = [item.get("color", "unknown") for item in associated]
            observations.append((person, person["box"], bool(associated), colors, associated))
        return observations, helmets, duplicates

    def assign_tracks(self, observations, now: float) -> dict[int, int]:
        assignments = {}
        used_tracks = set()
        used_observations = set()
        for index, observation in enumerate(observations):
            model_id = observation[0].get("track_id")
            if isinstance(model_id, int) and model_id in self.tracks:
                assignments[index] = model_id
                used_tracks.add(model_id)
                used_observations.add(index)
        pairs = sorted(
            (
                (association_score(track.box, observation[1]), track_id, index)
                for track_id, track in self.tracks.items()
                if now - track.last_seen <= self.lost_grace_seconds
                for index, observation in enumerate(observations)
            ),
            reverse=True,
        )
        for score, track_id, index in pairs:
            if score <= 0 or track_id in used_tracks or index in used_observations:
                continue
            used_tracks.add(track_id)
            used_observations.add(index)
            assignments[index] = track_id
        return assignments

    def update_track(self, observation, track_id, now, wall_time, frame, analysis):
        person, box, has_helmet, colors, helmets = observation
        if track_id is None:
            model_id = person.get("track_id")
            track_id = model_id if isinstance(model_id, int) else self.next_id
            self.next_id = max(self.next_id, track_id + 1)
            self.tracks[track_id] = Track(track_id, box, now)
        track = self.tracks[track_id]
        track.box, track.last_seen = box, now
        track.last_helmet_boxes = [(item["box"], track.id) for item in helmets]
        person["track_id"] = track.id
        for helmet in helmets:
            helmet["track_id"] = track.id
        track.colors.extend(colors)
        track.samples.append((wall_time, has_helmet))
        while track.samples and track.samples[0][0].timestamp() < wall_time.timestamp() - 2:
            track.samples.popleft()
        if has_helmet:
            self.update_helmet_state(track, wall_time, frame, analysis)
        else:
            self.update_violation_state(track, now, wall_time, frame, analysis)
        return person, track

    def update_helmet_state(self, track, wall_time, frame, analysis):
        track.helmet_streak = track.helmet_streak + 1 if track.event_id else 0
        if track.event_id is None:
            track.helmet_streak_start = None
        elif track.helmet_streak == 1:
            track.helmet_streak_start = wall_time
        if track.event_id and track.helmet_streak >= 3:
            self.close_event(
                track, track.helmet_streak_start or wall_time, "helmet_detected", frame, analysis.get("detections", [])
            )
            track.helmet_streak, track.helmet_streak_start = 0, None

    def update_violation_state(self, track, now, wall_time, frame, analysis):
        track.helmet_streak, track.helmet_streak_start = 0, None
        missing = [sample_time for sample_time, wearing in track.samples if not wearing]
        track.streak_start = missing[0] if len(missing) >= max(4, math.ceil(len(track.samples) * 0.7)) else None
        if track.streak_start is None or wall_time.timestamp() - track.streak_start.timestamp() < self.confirm_seconds:
            return
        color = Counter(track.colors).most_common(1)[0][0] if track.colors else "unknown"
        role = role_for_color(color)
        if track.event_id is None:
            track.event_id = save_event(
                self.zone, self.source_type, self.source_ref, track.id, track.streak_start, wall_time, color, role
            )
            track.event_started, track.last_persisted = track.streak_start, now
            logger.warning(
                "НАРУШЕНИЕ подтверждено | event=%s | track=%s | zone=%s",
                track.event_id,
                track.id,
                self.zone["name"] if self.zone else "Без зоны",
            )
            self.save_or_defer_snapshot(track.event_id, "start", frame, analysis.get("detections", []), track)
        elif now - track.last_persisted >= 2:
            save_event(
                self.zone,
                self.source_type,
                self.source_ref,
                track.id,
                track.event_started or track.streak_start,
                wall_time,
                color,
                role=role,
                event_id=track.event_id,
            )
            track.last_persisted = now

    def assign_helmet_tracks(self, helmets, observations, person_track_ids):
        for index, helmet in enumerate(helmets, start=1):
            matches = []
            for person, box, _, _, _ in observations:
                if check_box_collision(box, helmet["box"]):
                    matches.append((1.0, person_track_ids[id(person)]))
            helmet["track_id"] = max(matches)[1] if matches else f"H{index}"

    def flush_snapshots(self):
        pending = self.pending_snapshots or []
        self.pending_snapshots = None
        for event_id, stage, track, frame, detections in pending:
            self.save_snapshot(event_id, stage, frame, detections, track)

    def remove_lost(self, now, wall_time, frame, analysis):
        for track_id, track in list(self.tracks.items()):
            if now - track.last_seen > self.lost_grace_seconds:
                if track.event_id is not None:
                    self.close_event(track, wall_time, "track_lost", frame, analysis.get("detections", []))
                del self.tracks[track_id]

    def collapse_tracks(self, now: float) -> None:
        changed = True
        while changed:
            changed = False
            track_ids = list(self.tracks)
            for index, first_id in enumerate(track_ids):
                first = self.tracks.get(first_id)
                if first is None:
                    continue
                for second_id in track_ids[index + 1 :]:
                    second = self.tracks.get(second_id)
                    if second is None or abs(first.last_seen - second.last_seen) > 1.0:
                        continue
                    if not near_duplicate(first.box, second.box):
                        continue

                    keep_id, discard_id = first_id, second_id
                    keep, discard = first, second
                    if keep.event_id is None and discard.event_id is not None:
                        keep_id, discard_id = discard_id, keep_id
                        keep, discard = discard, keep
                    elif keep.event_id is not None and discard.event_id is not None:
                        keep_start = keep.event_started or keep.streak_start or datetime.max.replace(tzinfo=UTC)
                        discard_start = (
                            discard.event_started or discard.streak_start or datetime.max.replace(tzinfo=UTC)
                        )
                        if discard_start < keep_start:
                            keep_id, discard_id = discard_id, keep_id
                            keep, discard = discard, keep

                    if keep.event_id is not None and discard.event_id is not None and keep.event_id != discard.event_id:
                        duplicate_event_id = discard.event_id
                        discard_duplicate_event(duplicate_event_id)
                        logger.warning(
                            "Дублирующий трек объединён | удалено событие #%s | оставлен track=%s",
                            duplicate_event_id,
                            keep.id,
                        )
                    elif keep.event_id is None and discard.event_id is not None:
                        keep.event_id = discard.event_id
                        keep.event_started = discard.event_started

                    if discard.last_seen > keep.last_seen:
                        keep.box = discard.box
                        keep.last_seen = discard.last_seen
                        keep.last_helmet_boxes = discard.last_helmet_boxes
                    keep.samples = deque(
                        sorted((*keep.samples, *discard.samples), key=lambda sample: sample[0])[-240:], maxlen=240
                    )
                    keep.colors.extend(discard.colors)
                    starts = [value for value in (keep.streak_start, discard.streak_start) if value is not None]
                    keep.streak_start = min(starts) if starts else None
                    keep.helmet_streak = max(keep.helmet_streak, discard.helmet_streak)
                    keep.last_persisted = max(keep.last_persisted, discard.last_persisted)
                    self.tracks[keep_id] = keep
                    self.tracks.pop(discard_id, None)
                    changed = True
                    break
                if changed:
                    break

    def close_event(
        self,
        track: Track,
        at: datetime,
        reason: str,
        frame: np.ndarray | None = None,
        detections: list[dict[str, Any]] | None = None,
    ) -> None:
        color = Counter(track.colors).most_common(1)[0][0] if track.colors else "unknown"
        role = role_for_color(color)
        event_id = track.event_id
        duration = max(0.0, (at - (track.event_started or track.streak_start or at)).total_seconds())
        save_event(
            self.zone,
            self.source_type,
            self.source_ref,
            track.id,
            track.event_started or track.streak_start or at,
            at,
            color,
            role=role,
            event_id=track.event_id,
            ended_at=at,
            close_reason=reason,
        )
        if event_id is not None:
            self.save_or_defer_snapshot(
                event_id,
                "end",
                frame if frame is not None else self.last_frame,
                detections if detections is not None else (self.last_analysis or {}).get("detections", []),
                track,
            )
        reason_labels = {
            "helmet_detected": "каска обнаружена",
            "track_lost": "трек потерян",
            "source_stopped": "источник остановлен",
        }
        logger.info(
            "НАРУШЕНИЕ закрыто | event=%s | zone=%s | причина=%s | длительность=%.1f с",
            event_id,
            self.zone["name"] if self.zone else "Без зоны",
            reason_labels.get(reason, reason),
            duration,
        )
        track.event_id = None
        track.event_started = None
        if reason == "helmet_detected":
            track.samples.clear()
            track.streak_start = None

    def save_or_defer_snapshot(
        self, event_id: int, stage: str, frame: np.ndarray | None, detections: list[dict[str, Any]], track: Track
    ) -> None:
        if self.pending_snapshots is not None:
            self.pending_snapshots.append((event_id, stage, track, frame, detections))
        else:
            self.save_snapshot(event_id, stage, frame, detections, track)

    @staticmethod
    def save_snapshot(
        event_id: int, stage: str, frame: np.ndarray | None, detections: list[dict[str, Any]], track: Track
    ) -> None:
        path = write_event_snapshot(event_id, stage, frame, detections, track)
        set_event_snapshot(event_id, stage, path)

    def close_all(self, reason: str = "source_stopped", at: datetime | None = None) -> None:
        now = at or (self.last_source_time if self.explicit_source_clock else datetime.now(UTC))
        for track in self.tracks.values():
            if track.event_id is not None:
                self.close_event(track, now, reason, self.last_frame, (self.last_analysis or {}).get("detections", []))
        self.tracks.clear()
