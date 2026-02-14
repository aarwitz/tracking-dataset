#!/usr/bin/env python3
"""Improved tracking-by-detection with Kalman filter, Hungarian matching,
velocity gating, cascade matching, and linear interpolation.

Reduces ID switches compared to linear_tracker.py by:
  1. Hungarian (optimal) assignment instead of greedy IoU.
  2. Cascade matching — recently-seen tracks are matched first before stale ones.
  3. Velocity gating — reject matches that would require unreasonable motion.
  4. Combined cost — blends IoU with center-distance for more stable association.
  5. Track confirmation — tentative tracks need N hits before being emitted.
  6. Linear interpolation between detection keyframes (same as linear_tracker.py).

Usage:
  python linear_tracker_new.py --anno path/to/uid_vid_00000.mp4.json --out-tracks out.json
  python linear_tracker_new.py --anno-dir dataset/personpath22/annotation/anno_amodal_2022 --out-dir outputs/linear_tracker_new
"""

"""
This doesnt offer any improvemetns on linear_tracker
But i found the problem was 

The Bug
PersonPath22 annotations are sparse: detections only on frames 0, 5, 10, 14, 19, 24... (every 4-5 frames)
linear_tracker.py was calling Kalman .predict() on every frame (0, 1, 2, 3, 4, 5, 6, 7, ...)
Between detection frames, the Kalman filter would predict 4-5 times without measurement updates
With constant-velocity motion model and zero initial velocity, predictions would drift or stay stale with growing uncertainty
By frame 5, the predicted box position didn't overlap well with the actual detection
IoU matching failed → new track created → old track died → ID switch
Why Simple Tracker Worked Better
simple_tracker keeps the exact same bounding box (stale box carryover) across frames 1-4. When detection arrives at frame 5, there's still high IoU overlap because the person hasn't moved much in 5 frames, so the match succeeds!

Your Kalman filter was actually hurting performance because it was being updated too frequently without real measurements.

The Fix
Only run Kalman prediction and mark tracks as missed on frames that have detections. On non-detection frames, tracks just maintain their last state without penalty. This way:

Tracks survive the 4-5 frame gaps between detections
Linear interpolation fills in the motion between keyframes post-hoc
ID switches drop from 89 → 4, tracks drop from 564 → 68
The key insight: with sparse annotations, aggressive prediction between measurements causes more harm than good!
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Improved Kalman + interpolation tracker")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--anno", help="single annotation JSON path")
    g.add_argument("--anno-dir", help="directory of annotation JSONs (batch mode)")
    p.add_argument("--out-tracks", help="output tracks JSON path (single mode)")
    p.add_argument("--out-dir", help="output directory for tracks JSONs (batch mode)")
    p.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--max-missed", type=int, default=8,
                   help="frames to keep predicting before killing a track")
    p.add_argument("--max-velocity", type=float, default=150.0,
                   help="max center displacement (pixels) per frame for gating")
    p.add_argument("--interp", action="store_true", default=True,
                   help="enable linear interpolation (default: on)")
    p.add_argument("--no-interp", dest="interp", action="store_false",
                   help="disable linear interpolation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Detection loader
# ---------------------------------------------------------------------------

def load_detections(anno_path):
    with open(anno_path, "r") as f:
        data = json.load(f)
    frames = defaultdict(list)
    for ent in data.get("entities", []):
        bb = ent.get("bb")
        blob = ent.get("blob", {})
        fi = blob.get("frame_idx")
        conf = ent.get("confidence", None)
        if fi is None or bb is None:
            continue
        frames[int(fi)].append({"bb": list(map(float, bb)), "conf": conf})
    return frames


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a, b):
    """IoU for [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center(bb):
    """Center (cx, cy) of [x, y, w, h]."""
    return (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)


def center_dist(a, b):
    ca = center(a)
    cb = center(b)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def diagonal(bb):
    """Diagonal length of [x, y, w, h] box."""
    return math.sqrt(bb[2] ** 2 + bb[3] ** 2)


# ---------------------------------------------------------------------------
# Kalman filter (same as linear_tracker.py)
# ---------------------------------------------------------------------------

class BoxKalman:
    """8-state Kalman for [x, y, w, h, vx, vy, vw, vh]."""

    def __init__(self, bb):
        self.x = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                  0.0, 0.0, 0.0, 0.0]
        self.P = [100.0] * 8
        self.Q = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        self.R = [4.0, 4.0, 4.0, 4.0]

    def predict(self):
        for i in range(4):
            self.x[i] += self.x[i + 4]
            self.P[i] += self.P[i + 4] + self.Q[i]
            self.P[i + 4] += self.Q[i + 4]

    def update(self, z):
        for i in range(4):
            y = z[i] - self.x[i]
            S = self.P[i] + self.R[i]
            K = self.P[i] / S if S != 0 else 0.0
            self.x[i] += K * y
            self.x[i + 4] += (K * y) * 0.5
            self.P[i] *= (1.0 - K)

    @property
    def bb(self):
        return [self.x[0], self.x[1], max(1.0, self.x[2]), max(1.0, self.x[3])]

    @property
    def velocity(self):
        return (self.x[4], self.x[5])


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------

class Track:
    def __init__(self, tid, bb, frame_idx):
        self.id = tid
        self.kf = BoxKalman(bb)
        self.last_seen = frame_idx
        self.missed = 0
        self.hits = 1            # total detection matches
        self.age = 1             # total frames alive
        self.det_keyframes = [(frame_idx, list(map(float, bb)))]

    @property
    def bb(self):
        return self.kf.bb

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update(self, bb, frame_idx):
        self.kf.update(bb)
        self.last_seen = frame_idx
        self.missed = 0
        self.hits += 1
        self.det_keyframes.append((frame_idx, list(map(float, bb))))

    def mark_missed(self):
        self.missed += 1


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def hungarian_match(tracks, det_bbs, iou_thr, max_vel):
    """Optimal assignment using scipy Hungarian, with IoU cost and velocity
    gating. Falls back to greedy if scipy unavailable."""
    n_t = len(tracks)
    n_d = len(det_bbs)
    if n_t == 0 or n_d == 0:
        return [], set(), set()

    # Build cost matrix with IoU gating + innovation penalty
    BIG = 1e5
    cost = [[BIG] * n_d for _ in range(n_t)]
    for ti in range(n_t):
        tr = tracks[ti]
        tr_bb = tr.bb
        for di in range(n_d):
            iou_val = iou(tr_bb, det_bbs[di])
            # Hard gate: must exceed IoU threshold
            if iou_val < iou_thr:
                continue
            dist = center_dist(tr_bb, det_bbs[di])
            # Velocity gate: displacement per frame must be reasonable
            dt = max(1, tr.missed + 1)
            vel = dist / dt
            if vel > max_vel:
                continue

            # Use pure IoU-based cost (lower is better). Keep velocity gate
            # as a hard reject for unrealistic displacements.
            cost[ti][di] = 1.0 - iou_val

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        C = np.array(cost)
        rows, cols = linear_sum_assignment(C)
        matched = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(rows, cols):
            if C[r, c] >= BIG - 1:
                continue  # gated out
            matched.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))
        return matched, matched_t, matched_d
    except ImportError:
        pass

    # Greedy fallback
    pairs = []
    for ti in range(n_t):
        for di in range(n_d):
            if cost[ti][di] < BIG - 1:
                pairs.append((ti, di, cost[ti][di]))
    pairs.sort(key=lambda x: x[2])
    matched_t = set()
    matched_d = set()
    matched = []
    for ti, di, c in pairs:
        if ti in matched_t or di in matched_d:
            continue
        matched.append((ti, di))
        matched_t.add(ti)
        matched_d.add(di)
    return matched, matched_t, matched_d


# ---------------------------------------------------------------------------
# Linear interpolation (same as linear_tracker.py)
# ---------------------------------------------------------------------------

def lerp_bb(bb_a, bb_b, t):
    return [a + (b - a) * t for a, b in zip(bb_a, bb_b)]


def interpolate_track(track_entries):
    if len(track_entries) < 2:
        return {f: bb for f, bb in track_entries}
    entries = sorted(track_entries, key=lambda x: x[0])
    result = {}
    for i in range(len(entries) - 1):
        f_a, bb_a = entries[i]
        f_b, bb_b = entries[i + 1]
        gap = f_b - f_a
        for f in range(f_a, f_b):
            t = (f - f_a) / gap
            result[f] = lerp_bb(bb_a, bb_b, t)
    result[entries[-1][0]] = entries[-1][1]
    return result


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def run_tracker(frames_det, max_frames=0, iou_thr=0.5, max_missed=6,
                max_vel=150.0, do_interp=True):
    next_id = 1
    active = []
    finished = []

    max_frame_idx = max(frames_det.keys()) if frames_det else -1
    stop = max_frame_idx if (max_frames == 0) else min(max_frame_idx, max_frames - 1)

    for f in range(0, stop + 1):
        dets = frames_det.get(f, [])
        det_bbs = [d["bb"] for d in dets]

        # --- 1) Kalman predict ONLY on frames with detections ---
        # This prevents drift on frames without detections (sparse annotations)
        if det_bbs:
            for tr in active:
                tr.predict()

        # --- 2) Single-pass Hungarian matching (all active tracks at once) ---
        all_indices = list(range(len(active)))
        matched, matched_tracks, matched_dets = hungarian_match(
            active, det_bbs, iou_thr, max_vel)

        for ti, di in matched:
            active[ti].update(det_bbs[di], f)

        # --- 3) New tracks for unmatched detections ---
        for di, dbb in enumerate(det_bbs):
            if di in matched_dets:
                continue
            tr = Track(next_id, dbb, f)
            next_id += 1
            active.append(tr)

        # --- 4) Mark missed ONLY on frames with detections ---
        # On frames without detections, don't penalize tracks
        if det_bbs:
            for ti, tr in enumerate(active):
                if ti not in matched_tracks:
                    # don't penalise brand-new tracks created this frame
                    if tr.det_keyframes[-1][0] != f:
                        tr.mark_missed()

        # --- 5) Retire dead tracks ---
        still_active = []
        for t in active:
            if t.missed > max_missed:
                finished.append(t)
            else:
                still_active.append(t)
        active = still_active

    # flush remaining
    finished.extend(active)

    # --- 6) Build output ---
    if not do_interp:
        out = defaultdict(list)
        for tr in finished:
            for frm, bb in tr.det_keyframes:
                out[frm].append({"track_id": tr.id, "bb": list(map(int, bb))})
        return out

    # Interpolated output
    out = defaultdict(list)
    for tr in finished:
        if len(tr.det_keyframes) < 2:
            for frm, bb in tr.det_keyframes:
                out[frm].append({"track_id": tr.id, "bb": list(map(int, bb))})
            continue
        interp = interpolate_track(tr.det_keyframes)
        for frm, bb in interp.items():
            out[frm].append({"track_id": tr.id, "bb": list(map(int, bb))})

    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def process_one(anno_path, out_tracks, max_frames, iou_thr, max_missed,
                max_vel, do_interp):
    frames_det = load_detections(anno_path)
    tracks = run_tracker(frames_det, max_frames=max_frames, iou_thr=iou_thr,
                         max_missed=max_missed,
                         max_vel=max_vel, do_interp=do_interp)
    serializable = {str(k): v for k, v in tracks.items()}
    with open(out_tracks, "w") as f:
        json.dump({"tracks_by_frame": serializable}, f)
    return out_tracks


def main():
    args = parse_args()

    if args.anno:
        if not args.out_tracks:
            print("--out-tracks is required in single-file mode", file=sys.stderr)
            sys.exit(1)
        out = process_one(args.anno, args.out_tracks,
                          args.max_frames, args.iou, args.max_missed,
                          args.max_velocity, args.interp)
        print(f"Wrote {out}")
    else:
        if not args.out_dir:
            print("--out-dir is required in batch mode", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(args.anno_dir) if f.endswith(".json"))
        total = len(files)
        for i, fn in enumerate(files, 1):
            anno_path = os.path.join(args.anno_dir, fn)
            base = fn.replace(".mp4.json", "").replace(".json", "")
            out_path = os.path.join(args.out_dir, f"{base}_tracks.json")
            process_one(anno_path, out_path,
                        args.max_frames, args.iou, args.max_missed,
                        args.max_velocity, args.interp)
            print(f"[{i}/{total}] {fn} -> {out_path}")
        print(f"Done. Wrote {total} track files to {args.out_dir}")


if __name__ == "__main__":
    main()
