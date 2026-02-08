    #!/usr/bin/env python3
"""Minimal tracking-by-detection baseline (greedy IoU matching).

Reads detection JSON (PersonPath22 format) and assigns tracker IDs.
Outputs a tracks JSON and optionally an annotated video showing tracker IDs.

Usage:
  python simple_tracker.py --anno path/to/uid_vid_00000.mp4.json --out-tracks out.json --video dataset/personpath22/raw_data/uid_vid_00000.mp4 --out-video out.mp4
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anno", required=True, help="annotation JSON path (detections input, ids ignored)")
    p.add_argument("--out-tracks", required=True, help="output tracks JSON path")
    p.add_argument("--video", help="optional video file to produce annotated video")
    p.add_argument("--out-video", help="optional annotated video output path")
    p.add_argument("--max-frames", type=int, default=0, help="limit frames to process (0 = all)")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--max-missed", type=int, default=5, help="frames to allow missing before deleting a track")
    p.add_argument("--transcode", choices=["h264","webm","none"], default="h264")
    return p.parse_args()


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
        frames[fi].append({"bb": bb, "conf": conf})
    return frames


def iou(a, b):
    # a,b = [x,y,w,h]
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
    return float(inter) / float(union)


class Track:
    def __init__(self, tid, bb, frame_idx):
        self.id = tid
        self.bb = list(map(int, bb))
        self.last_seen = frame_idx
        self.missed = 0
        self.age = 1

    def update(self, bb, frame_idx):
        self.bb = list(map(int, bb))
        self.last_seen = frame_idx
        self.missed = 0
        self.age += 1

    def mark_missed(self):
        self.missed += 1


def run_tracker(frames_det, max_frames=0, iou_thr=0.3, max_missed=5):
    tracks = {}
    next_id = 1
    active = []  # list of Track
    out = defaultdict(list)

    max_frame_idx = max(frames_det.keys()) if frames_det else -1
    stop = max_frame_idx if (max_frames == 0) else min(max_frame_idx, max_frames - 1)

    for f in range(0, stop + 1):
        dets = frames_det.get(f, [])
        det_bbs = [d["bb"] for d in dets]

        # compute IoU matrix
        matches = []  # (track_idx, det_idx, iou)
        for ti, tr in enumerate(active):
            for di, dbb in enumerate(det_bbs):
                matches.append((ti, di, iou(tr.bb, dbb)))

        # greedy match by descending IoU
        matched_tracks = set()
        matched_dets = set()
        matches.sort(key=lambda x: x[2], reverse=True)
        for ti, di, val in matches:
            if val < iou_thr:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            # assign
            active[ti].update(det_bbs[di], f)
            matched_tracks.add(ti)
            matched_dets.add(di)

        # create new tracks for unmatched detections
        for di, dbb in enumerate(det_bbs):
            if di in matched_dets:
                continue
            tr = Track(next_id, dbb, f)
            next_id += 1
            active.append(tr)

        # mark missed for unmatched tracks
        for ti, tr in enumerate(list(active)):
            if ti not in matched_tracks:
                tr.mark_missed()

        # remove dead tracks
        active = [t for t in active if t.missed <= max_missed]

        # record output for this frame
        for tr in active:
            out[f].append({"track_id": tr.id, "bb": tr.bb})

    return out


def id_to_color(i):
    r = (i * 37) % 255
    g = (i * 83) % 255
    b = (i * 151) % 255
    return (int(b), int(g), int(r))


def visualize_tracks(video_path, tracks, out_video, max_frames=0, transcode="h264"):
    try:
        import cv2
    except Exception:
        print("Install opencv-python to visualize", file=sys.stderr); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("failed to open video", file=sys.stderr); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import tempfile, subprocess, shutil
    tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmpf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmpf.name, fourcc, fps, (w, h))

    frame_idx = 0
    stop = max(tracks.keys()) if tracks else -1
    if max_frames:
        stop = min(stop, max_frames - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in tracks:
            for t in tracks[frame_idx]:
                bb = t["bb"]
                tid = t["track_id"]
                x, y, bw, bh = map(int, bb)
                color = id_to_color(tid)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, f"tid:{tid}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1
        if max_frames and frame_idx > stop:
            break

    cap.release(); out.release()

    if transcode == "none":
        shutil.move(tmpf.name, out_video); print(f"Wrote {out_video}"); return
    if shutil.which("ffmpeg") is None:
        shutil.move(tmpf.name, out_video); print(f"Wrote {out_video} (no ffmpeg)"); return
    if transcode == "h264":
        cmd = ["ffmpeg","-y","-i",tmpf.name,"-c:v","libx264","-crf","18","-preset","veryfast","-movflags","+faststart","-pix_fmt","yuv420p","-c:a","copy",out_video]
    else:
        cmd = ["ffmpeg","-y","-i",tmpf.name,"-c:v","libvpx-vp9","-b:v","0","-crf","30",out_video]
    subprocess.run(cmd, check=True)
    try: os.remove(tmpf.name)
    except Exception: pass
    print(f"Wrote {out_video}")


def main():
    args = parse_args()
    frames_det = load_detections(args.anno)
    tracks = run_tracker(frames_det, max_frames=args.max_frames, iou_thr=args.iou, max_missed=args.max_missed)
    # save tracks
    with open(args.out_tracks, "w") as f:
        json.dump({"tracks_by_frame": tracks}, f)
    print(f"Wrote tracks JSON: {args.out_tracks}")

    if args.out_video and args.video:
        visualize_tracks(args.video, tracks, args.out_video, max_frames=args.max_frames, transcode=args.transcode)


if __name__ == '__main__':
    main()
