"""Load PersonPath22 annotations and produce per-frame detection measurements.

Supports two annotation sources:
  1. Per-video JSON  (anno_visible_2022/uid_vid_XXXXX.mp4.json)
  2. Combined JSON    (anno_visible_2022.json) – metadata only; entities are in
     the per-video files.

Interpolation is linear between annotated frames for each GT id.
Extrapolation outside the annotated range is NOT performed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Measurement dataclass
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """A single detection measurement for one frame."""

    bbox_xywh: np.ndarray          # shape (4,) – [x, y, w, h]
    gt_id: int = -1                # ground-truth id (debug only)

    # --- derived properties (computed lazily) ---

    @property
    def bbox_xyxy(self) -> np.ndarray:
        x, y, w, h = self.bbox_xywh
        return np.array([x, y, x + w, y + h], dtype=np.float64)

    @property
    def center_xy(self) -> np.ndarray:
        x, y, w, h = self.bbox_xywh
        return np.array([x + w / 2, y + h / 2], dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        """2×2 diagonal covariance based on bbox size."""
        _, _, w, h = self.bbox_xywh
        sx = 0.25 * w
        sy = 0.25 * h
        return np.diag([sx * sx, sy * sy])


# ---------------------------------------------------------------------------
# Annotation loading helpers
# ---------------------------------------------------------------------------

def _load_per_video_json(path: str) -> Dict[int, List[Tuple[int, List[float]]]]:
    """Return {gt_id: [(frame_idx, [x,y,w,h]), ...]} sorted by frame_idx."""
    with open(path, "r") as f:
        data = json.load(f)
    by_id: Dict[int, List[Tuple[int, List[float]]]] = {}
    for ent in data.get("entities", []):
        bb = ent.get("bb")
        blob = ent.get("blob", {})
        fi = blob.get("frame_idx")
        if fi is None or bb is None:
            continue
        ent_id = ent.get("id", -1)
        by_id.setdefault(ent_id, []).append((int(fi), [float(v) for v in bb]))
    for k in by_id:
        by_id[k].sort(key=lambda t: t[0])
    return by_id


def resolve_anno_path(anno: str, uid: Optional[str] = None) -> str:
    """Given either a per-video JSON or combined JSON, return the per-video path.

    If *anno* points at the combined JSON (anno_visible_2022.json) and *uid* is
    given, derive the per-video path.  Otherwise return *anno* as-is.
    """
    if os.path.isfile(anno):
        # check if it looks like the combined file
        base = os.path.basename(anno)
        if uid and base in ("anno_visible_2022.json", "anno_amodal_2022.json"):
            # derive per-video
            parent = os.path.splitext(anno)[0]  # directory with same stem
            candidate = os.path.join(parent, uid + ".json")
            if os.path.isfile(candidate):
                return candidate
    return anno


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gt_tracks(anno_path: str, uid: Optional[str] = None) -> Dict[int, List[Tuple[int, List[float]]]]:
    """Load ground-truth tracks keyed by gt_id.

    Returns:
        {gt_id: [(frame_idx, [x,y,w,h]), ...]}  sorted by frame_idx.
    """
    path = resolve_anno_path(anno_path, uid)
    return _load_per_video_json(path)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_measurements(
    gt_tracks: Dict[int, List[Tuple[int, List[float]]]],
    frame_idx: int,
) -> List[Measurement]:
    """Return interpolated measurements for *frame_idx*.

    For each gt_id whose annotated range spans *frame_idx*, produce one
    Measurement via linear interpolation.  Outside the range → nothing.
    """
    measurements: List[Measurement] = []
    for gt_id, keyframes in gt_tracks.items():
        if len(keyframes) == 0:
            continue
        first_fi = keyframes[0][0]
        last_fi = keyframes[-1][0]
        if frame_idx < first_fi or frame_idx > last_fi:
            continue

        # binary-search for the surrounding keyframes
        lo, hi = 0, len(keyframes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if keyframes[mid][0] <= frame_idx:
                lo = mid + 1
            else:
                hi = mid
        # lo is the first index with frame > frame_idx  (or len-1 if exact match at end)
        idx_after = lo
        idx_before = lo - 1 if lo > 0 else 0

        f0, bb0 = keyframes[idx_before]
        f1, bb1 = keyframes[idx_after]

        if f0 == f1 or frame_idx == f0:
            bbox = np.array(bb0, dtype=np.float64)
        elif frame_idx == f1:
            bbox = np.array(bb1, dtype=np.float64)
        else:
            t = float(frame_idx - f0) / float(f1 - f0)
            bbox = np.array([_lerp(a, b, t) for a, b in zip(bb0, bb1)], dtype=np.float64)

        measurements.append(Measurement(bbox_xywh=bbox, gt_id=gt_id))

    return measurements
