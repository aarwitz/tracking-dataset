#!/usr/bin/env python3
"""Baseline tracking-by-detection on PersonPath22.

Reads annotation JSON, interpolates detections, runs an IoU-based tracker,
and writes an annotated video with tracker IDs overlaid.

Usage:
  python tracker_baseline.py \
      --anno dataset/personpath22/annotation/anno_visible_2022/uid_vid_00000.mp4.json \
      --video dataset/personpath22/raw_data/uid_vid_00000.mp4 \
      --out outputs/uid_vid_00000_tracked.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Baseline IoU tracker for PersonPath22")
    p.add_argument("--anno", required=True, help="annotation JSON path (per-video or combined)")
    p.add_argument("--video", help="video file path (optional; auto-derived from anno)")
    p.add_argument("--uid", help="video uid (e.g. uid_vid_00000.mp4) when using combined JSON")
    p.add_argument("--out", required=True, help="output annotated video path")
    p.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    p.add_argument("--min-iou", type=float, default=0.1, help="min IoU for association (default 0.1)")
    p.add_argument("--max-age", type=int, default=15, help="frames before track deletion (default 15)")
    p.add_argument("--min-hits", type=int, default=1, help="matches to confirm track (default 1)")
    p.add_argument("--show-gt", action="store_true", help="overlay gt id alongside tracker id")
    p.add_argument("--transcode", choices=["h264", "webm", "none"], default="h264")
    return p.parse_args()


# ---- color helper ----------------------------------------------------------

def id_to_color(i: int):
    r = (i * 37 + 100) % 255
    g = (i * 83 + 50) % 255
    b = (i * 151 + 150) % 255
    return (int(b), int(g), int(r))  # BGR


# ---- video path derivation ------------------------------------------------

def derive_video_path(annopath: str) -> str | None:
    bp = os.path.abspath(annopath).split(os.sep)
    try:
        idx = bp.index("annotation")
        bp[idx] = "raw_data"
        bp[-1] = os.path.splitext(bp[-1])[0] + ".mp4"
        candidate = os.sep.join(bp)
        if os.path.exists(candidate):
            return candidate
    except ValueError:
        pass
    return None


# ---- transcode helper -----------------------------------------------------

def transcode(tmp_path: str, out_path: str, mode: str):
    if mode == "none":
        shutil.move(tmp_path, out_path)
        return
    if shutil.which("ffmpeg") is None:
        shutil.move(tmp_path, out_path)
        print("ffmpeg not found – wrote raw mp4v", file=sys.stderr)
        return
    if mode == "h264":
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p",
            "-c:a", "copy", out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", out_path,
        ]
    print("Transcoding …")
    subprocess.run(cmd, check=True)
    try:
        os.remove(tmp_path)
    except OSError:
        pass


# ---- main ------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        import cv2
    except ImportError:
        print("Install opencv-python: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    from anno_measurements import load_gt_tracks, interpolate_measurements
    from tracker_core import Tracker

    # --- resolve paths -------------------------------------------------------
    anno = args.anno
    if not os.path.isfile(anno):
        print(f"Annotation not found: {anno}", file=sys.stderr)
        sys.exit(2)

    video_path = args.video or derive_video_path(anno)
    if not video_path or not os.path.isfile(video_path):
        print("Video not found. Provide --video.", file=sys.stderr)
        sys.exit(2)

    # --- load annotations & create tracker -----------------------------------
    gt_tracks = load_gt_tracks(anno, uid=args.uid)
    tracker = Tracker(min_iou=args.min_iou, max_age=args.max_age, min_hits=args.min_hits)

    # --- open video ----------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmpf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmpf.name, fourcc, fps, (w, h))

    # --- frame loop ----------------------------------------------------------
    frame_idx = 0
    max_frames = args.max_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # get interpolated measurements for this frame
        measurements = interpolate_measurements(gt_tracks, frame_idx)

        # tracker step
        active_tracks = tracker.step(measurements)

        # draw active tracks
        for trk in active_tracks:
            x, y, bw, bh = map(int, trk.bbox_xywh)
            color = id_to_color(trk.track_id)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            label = f"trk:{trk.track_id}"
            if args.show_gt and trk.last_gt_id >= 0:
                label += f" gt:{trk.last_gt_id}"
            cv2.putText(frame, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # HUD overlay
        info = f"frame:{frame_idx}  tracks:{len(active_tracks)}"
        cv2.putText(frame, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()
    print(f"Processed {frame_idx} frames, {tracker._next_id} total track IDs created.")

    # --- transcode -----------------------------------------------------------
    transcode(tmpf.name, args.out, args.transcode)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
