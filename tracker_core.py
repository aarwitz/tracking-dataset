"""Minimal IoU-based tracker – no Kalman filter, no learned features.

Classes:
    Track   – state of a single tracked object
    Tracker – manages active tracks and performs per-frame association
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from anno_measurements import Measurement


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x, y, w, h] bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    bbox_xywh: np.ndarray           # last-updated bbox [x,y,w,h]
    age: int = 0                     # frames since creation
    time_since_update: int = 0       # frames since last matched detection
    hits: int = 1                    # number of successful matches
    confirmed: bool = False
    last_gt_id: int = -1             # debug only

    @property
    def center_xy(self) -> np.ndarray:
        x, y, w, h = self.bbox_xywh
        return np.array([x + w / 2, y + h / 2], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """Greedy IoU tracker (no prediction step)."""

    def __init__(
        self,
        min_iou: float = 0.1,
        max_age: int = 15,
        min_hits: int = 1,
    ):
        self.min_iou = min_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 0

    # ---- internal ----------------------------------------------------------

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # ---- association -------------------------------------------------------

    def _associate(
        self,
        detections: List[Measurement],
    ) -> Tuple[List[Tuple[Track, Measurement]], List[Track], List[Measurement]]:
        """Greedy assignment by descending IoU.

        Returns (matched, unmatched_tracks, unmatched_detections).
        """
        if not self.tracks or not detections:
            return [], list(self.tracks), list(detections)

        n_trk = len(self.tracks)
        n_det = len(detections)

        # build cost matrix (IoU values)
        iou_matrix = np.zeros((n_trk, n_det), dtype=np.float64)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou_xywh(trk.bbox_xywh, det.bbox_xywh)

        # greedy: pick highest IoU first
        matched: List[Tuple[Track, Measurement]] = []
        used_trk = set()
        used_det = set()

        # flatten and sort descending
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape))[0]
        for i, j in indices:
            if i in used_trk or j in used_det:
                continue
            if iou_matrix[i, j] < self.min_iou:
                break  # remaining are all below threshold
            matched.append((self.tracks[i], detections[j]))
            used_trk.add(i)
            used_det.add(j)

        unmatched_tracks = [self.tracks[i] for i in range(n_trk) if i not in used_trk]
        unmatched_dets = [detections[j] for j in range(n_det) if j not in used_det]
        return matched, unmatched_tracks, unmatched_dets

    # ---- public step -------------------------------------------------------

    def step(self, detections: List[Measurement]) -> List[Track]:
        """Process one frame of detections and return active tracks."""

        # 1. age every track
        for trk in self.tracks:
            trk.age += 1

        # 2. associate
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)

        # 3. update matched
        for trk, det in matched:
            trk.bbox_xywh = det.bbox_xywh.copy()
            trk.time_since_update = 0
            trk.hits += 1
            trk.confirmed = trk.hits >= self.min_hits
            trk.last_gt_id = det.gt_id

        # 4. increment time_since_update for unmatched tracks
        for trk in unmatched_tracks:
            trk.time_since_update += 1

        # 5. delete dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. create new tracks for unmatched detections
        for det in unmatched_dets:
            trk = Track(
                track_id=self._new_id(),
                bbox_xywh=det.bbox_xywh.copy(),
                age=1,
                time_since_update=0,
                hits=1,
                confirmed=(1 >= self.min_hits),
                last_gt_id=det.gt_id,
            )
            self.tracks.append(trk)

        # return confirmed tracks only
        return [t for t in self.tracks if t.confirmed]
