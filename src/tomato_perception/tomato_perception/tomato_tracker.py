from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


BoundingBox = tuple[int, int, int, int]


@dataclass
class TomatoTrack:
    """State retained for one physical tomato across image frames."""

    track_id: int
    bbox: BoundingBox
    missed_frames: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TomatoTracker:
    """
    Lightweight persistent tracker for mostly stationary tomatoes.

    A new detection is matched to an existing track when either:
    - the boxes have enough overlap, or
    - their centers are close enough.

    Matching uses a greedy global cost ordering so one detection cannot be
    assigned to multiple tracks and one track cannot claim multiple detections.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        max_center_distance_px: float = 80.0,
        max_missed_frames: int = 10,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_center_distance_px <= 0.0:
            raise ValueError("max_center_distance_px must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")

        self.iou_threshold = float(iou_threshold)
        self.max_center_distance_px = float(max_center_distance_px)
        self.max_missed_frames = int(max_missed_frames)

        self._next_track_id = 0
        self._tracks: dict[int, TomatoTrack] = {}

    @staticmethod
    def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def center_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        center_a_x = (ax1 + ax2) / 2.0
        center_a_y = (ay1 + ay2) / 2.0
        center_b_x = (bx1 + bx2) / 2.0
        center_b_y = (by1 + by2) / 2.0

        return hypot(center_a_x - center_b_x, center_a_y - center_b_y)

    def _new_track(self, bbox: BoundingBox) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1

        self._tracks[track_id] = TomatoTrack(
            track_id=track_id,
            bbox=bbox,
            missed_frames=0,
        )

        return track_id

    def update(self, detections: Iterable[dict]) -> list[dict]:
        """
        Attach persistent ``track_id`` values to the current detections.

        Each detection dictionary must contain:
            detection["coords"] = (x1, y1, x2, y2)

        A copied list is returned; the caller's dictionaries are not modified.
        """

        current_detections = [dict(detection) for detection in detections]

        if not current_detections:
            self._mark_all_tracks_missed()
            return []

        if not self._tracks:
            for detection in current_detections:
                detection["track_id"] = self._new_track(detection["coords"])
            return sorted(current_detections, key=lambda item: item["track_id"])

        candidate_matches: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(current_detections):
                detection_bbox = detection["coords"]
                overlap = self.compute_iou(track.bbox, detection_bbox)
                distance = self.center_distance(track.bbox, detection_bbox)

                if (
                    overlap < self.iou_threshold
                    and distance > self.max_center_distance_px
                ):
                    continue

                normalized_distance = min(
                    distance / self.max_center_distance_px,
                    1.0,
                )

                # Lower cost is better. IoU is favored slightly more than
                # center distance because tomatoes are mostly stationary.
                match_cost = 0.65 * (1.0 - overlap) + 0.35 * normalized_distance
                candidate_matches.append(
                    (match_cost, track_id, detection_index)
                )

        candidate_matches.sort(key=lambda match: match[0])

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        for _, track_id, detection_index in candidate_matches:
            if track_id in matched_track_ids:
                continue
            if detection_index in matched_detection_indices:
                continue

            detection = current_detections[detection_index]
            detection["track_id"] = track_id

            track = self._tracks[track_id]
            track.bbox = detection["coords"]
            track.missed_frames = 0

            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.missed_frames += 1

        self._remove_expired_tracks()

        for detection_index, detection in enumerate(current_detections):
            if detection_index in matched_detection_indices:
                continue

            detection["track_id"] = self._new_track(detection["coords"])

        return sorted(current_detections, key=lambda item: item["track_id"])

    def _mark_all_tracks_missed(self) -> None:
        for track in self._tracks.values():
            track.missed_frames += 1

        self._remove_expired_tracks()

    def _remove_expired_tracks(self) -> None:
        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]

        for track_id in expired_track_ids:
            del self._tracks[track_id]

    @property
    def active_track_ids(self) -> list[int]:
        return sorted(self._tracks.keys())
