#!/usr/bin/env python3
"""Enhanced tracking-by-detection with Kalman filter and linear interpolation.

Improvements over simple_tracker.py:
  1. Constant-velocity Kalman filter predicts track positions on frames without
     detections, so IoU matching uses *predicted* boxes instead of stale ones.
  2. After all frames are processed, bounding boxes are linearly interpolated
     between detection keyframes to fill gaps smoothly.

Reads detection JSON (PersonPath22 format) and assigns tracker IDs.
Outputs a tracks JSON (same format as simple_tracker.py).

Usage:
  python linear_tracker.py --anno path/to/uid_vid_00000.mp4.json --out-tracks out.json
  python linear_tracker.py --anno-dir dataset/personpath22/annotation/anno_amodal_2022 --out-dir outputs/linear_tracker
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
    p = argparse.ArgumentParser(description="Kalman + interpolation tracker")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--anno", help="single annotation JSON path")
    g.add_argument("--anno-dir", help="directory of annotation JSONs (batch mode)")
    p.add_argument("--out-tracks", help="output tracks JSON path (single mode)")
    p.add_argument("--out-dir", help="output directory for tracks JSONs (batch mode)")
    p.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--max-missed", type=int, default=8,
                   help="frames to keep predicting before killing a track")
    p.add_argument("--interp", action="store_true", default=True,
                   help="enable linear interpolation of gaps (default: on)")
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
# IoU
# ---------------------------------------------------------------------------

def iou(a, b):
    """Compute IoU for two [x, y, w, h] boxes."""
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


# ---------------------------------------------------------------------------
# Minimal Kalman filter for [x, y, w, h] with constant-velocity model
# ---------------------------------------------------------------------------

class BoxKalman:
    """4-state (position) + 4-state (velocity) Kalman filter for [x, y, w, h].

    State: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]
    """

    def __init__(self, bb):
        # state vector
        self.x = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                   0.0, 0.0, 0.0, 0.0]
        # diagonal covariance (simple)
        self.P = [100.0] * 8
        # process noise
        self.Q = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        # measurement noise
        self.R = [4.0, 4.0, 4.0, 4.0]

    def predict(self):
        """Predict next state (constant velocity)."""
        for i in range(4):
            self.x[i] += self.x[i + 4]          # pos += vel
            self.P[i] += self.P[i + 4] + self.Q[i]
            self.P[i + 4] += self.Q[i + 4]

    def update(self, z):
        """Update with measurement z = [x, y, w, h]."""
        for i in range(4):
            y = z[i] - self.x[i]                 # innovation
            S = self.P[i] + self.R[i]             # innovation covariance
            K = self.P[i] / S if S != 0 else 0.0  # Kalman gain
            self.x[i] += K * y
            self.x[i + 4] += (K * y) * 0.5       # also nudge velocity
            self.P[i] *= (1.0 - K)

    @property
    def bb(self):
        """Current bounding box [x, y, w, h]."""
        return [self.x[0], self.x[1], max(1.0, self.x[2]), max(1.0, self.x[3])]


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------

class Track:
    def __init__(self, tid, bb, frame_idx):
        self.id = tid
        self.kf = BoxKalman(bb)
        self.last_seen = frame_idx
        self.missed = 0
        self.age = 1
        # for interpolation: list of (frame_idx, bb) where actual detections hit
        self.det_keyframes = [(frame_idx, list(map(float, bb)))]

    @property
    def bb(self):
        return self.kf.bb

    def predict(self):
        self.kf.predict()

    def update(self, bb, frame_idx):
        self.kf.update(bb)
        self.last_seen = frame_idx
        self.missed = 0
        self.age += 1
        self.det_keyframes.append((frame_idx, list(map(float, bb))))

    def mark_missed(self):
        self.missed += 1


# ---------------------------------------------------------------------------
# Linear interpolation post-process
# ---------------------------------------------------------------------------

def lerp_bb(bb_a, bb_b, t):
    """Linearly interpolate between two bboxes, t in [0,1]."""
    return [a + (b - a) * t for a, b in zip(bb_a, bb_b)]


def interpolate_track(track_entries):
    """Given a list of (frame_idx, bb) detection keyframes for one track,
    return a dict frame->bb with linearly interpolated boxes filling gaps."""
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
    # last keyframe
    result[entries[-1][0]] = entries[-1][1]
    return result


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def run_tracker(frames_det, max_frames=0, iou_thr=0.3, max_missed=8,
                do_interp=True):
    next_id = 1
    active = []       # list of Track
    finished = []     # tracks removed from active (for interpolation later)
    raw_out = defaultdict(list)  # frame -> [{track_id, bb}]  (from Kalman predict/update)

    max_frame_idx = max(frames_det.keys()) if frames_det else -1
    stop = max_frame_idx if (max_frames == 0) else min(max_frame_idx, max_frames - 1)

    for f in range(0, stop + 1):
        dets = frames_det.get(f, [])
        det_bbs = [d["bb"] for d in dets]

        # --- 1) Kalman predict ONLY on frames with detections ---
        # This prevents drift on frames without detections (which occur every 4-5 frames)
        if det_bbs:
            for tr in active:
                tr.predict()

        # --- 2) IoU matching (predicted box vs detection) ---
        matches = []
        for ti, tr in enumerate(active):
            for di, dbb in enumerate(det_bbs):
                matches.append((ti, di, iou(tr.bb, dbb)))

        matched_tracks = set()
        matched_dets = set()
        matches.sort(key=lambda x: x[2], reverse=True)
        for ti, di, val in matches:
            if val < iou_thr:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            active[ti].update(det_bbs[di], f)
            matched_tracks.add(ti)
            matched_dets.add(di)

        # --- 3) New tracks for unmatched detections ---
        for di, dbb in enumerate(det_bbs):
            if di in matched_dets:
                continue
            tr = Track(next_id, dbb, f)
            next_id += 1
            active.append(tr)

        # --- 4) Mark missed ONLY on frames with detections ---
        # On frames without detections, don't penalize tracks for not matching
        if det_bbs:
            for ti, tr in enumerate(active):
                if ti not in matched_tracks and tr.age > 0:
                    # only mark missed for tracks that existed before this frame step
                    # (new tracks just created won't be in matched_tracks but aren't missed)
                    if len(tr.det_keyframes) > 0 and tr.det_keyframes[-1][0] != f:
                        tr.mark_missed()

        # --- 5) Retire dead tracks, keep alive ones ---
        still_active = []
        for t in active:
            if t.missed > max_missed:
                finished.append(t)
            else:
                still_active.append(t)
        active = still_active

        # --- 6) Record Kalman-predicted box for every active track ---
        for tr in active:
            raw_out[f].append({"track_id": tr.id,
                               "bb": list(map(int, tr.bb))})

    # move remaining active to finished
    finished.extend(active)

    if not do_interp:
        return raw_out

    # --- 7) Post-process: linear interpolation ---
    out = defaultdict(list)
    for tr in finished:
        if len(tr.det_keyframes) < 2:
            # no interpolation possible, just emit raw keyframes
            for frm, bb in tr.det_keyframes:
                out[frm].append({"track_id": tr.id,
                                 "bb": list(map(int, bb))})
            continue
        interp = interpolate_track(tr.det_keyframes)
        for frm, bb in interp.items():
            out[frm].append({"track_id": tr.id,
                             "bb": list(map(int, bb))})

    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def process_one(anno_path, out_tracks, max_frames, iou_thr, max_missed, do_interp):
    frames_det = load_detections(anno_path)
    tracks = run_tracker(frames_det, max_frames=max_frames, iou_thr=iou_thr,
                         max_missed=max_missed, do_interp=do_interp)
    # convert defaultdict keys to str for JSON
    serializable = {str(k): v for k, v in tracks.items()}
    with open(out_tracks, "w") as f:
        json.dump({"tracks_by_frame": serializable}, f)
    return out_tracks


def main():
    args = parse_args()

    if args.anno:
        # single file mode
        if not args.out_tracks:
            print("--out-tracks is required in single-file mode", file=sys.stderr)
            sys.exit(1)
        out = process_one(args.anno, args.out_tracks,
                          args.max_frames, args.iou, args.max_missed, args.interp)
        print(f"Wrote {out}")
    else:
        # batch mode
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
                        args.max_frames, args.iou, args.max_missed, args.interp)
            print(f"[{i}/{total}] {fn} -> {out_path}")
        print(f"Done. Wrote {total} track files to {args.out_dir}")


if __name__ == "__main__":
    main()
