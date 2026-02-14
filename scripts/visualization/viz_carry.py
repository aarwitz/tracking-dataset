#!/usr/bin/env python3
"""Draw annotations and carry-forward the last bbox per id (with optional expiry).
"""
import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anno", required=True)
    p.add_argument("--video", help="optional")
    p.add_argument("--out", required=True)
    p.add_argument("--transcode", choices=["h264","webm","none"], default="h264")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-hold-frames", type=int, default=0, help="0 = infinite")
    return p.parse_args()


def load_annotations(path):
    with open(path, "r") as f:
        data = json.load(f)
    frames = {}
    for ent in data.get("entities", []):
        bb = ent.get("bb")
        blob = ent.get("blob", {})
        fi = blob.get("frame_idx")
        if fi is None or bb is None:
            continue
        ent_id = ent.get("id", -1)
        time = ent.get("time")
        frames.setdefault(fi, []).append((bb, ent_id, time))
    return frames


def id_to_color(i):
    r = (i * 37) % 255
    g = (i * 83) % 255
    b = (i * 151) % 255
    return (int(b), int(g), int(r))


def main():
    args = parse_args()
    try:
        import cv2
    except Exception:
        print("Install opencv-python", file=sys.stderr)
        raise

    annopath = args.anno
    if not os.path.exists(annopath):
        print("anno not found", file=sys.stderr); sys.exit(2)

    video_path = args.video
    if not video_path:
        bp = os.path.abspath(annopath).split(os.sep)
        try:
            idx = bp.index("annotation")
            bp[idx] = "raw_data"
            bp[-1] = os.path.splitext(bp[-1])[0] + ".mp4"
            candidate = os.sep.join(bp)
            if os.path.exists(candidate):
                video_path = candidate
        except ValueError:
            pass

    if not video_path or not os.path.exists(video_path):
        print("video not found", file=sys.stderr); sys.exit(2)

    frames_map = load_annotations(annopath)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("failed to open video", file=sys.stderr); sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import tempfile, subprocess, shutil
    tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmpf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmpf.name, fourcc, fps, (w, h))

    frame_idx = 0
    max_hold = args.max_hold_frames
    last_bb = {}
    last_seen = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_map:
            for bb, ent_id, t in frames_map[frame_idx]:
                try:
                    x, y, bw, bh = map(int, bb)
                except Exception:
                    continue
                last_bb[ent_id] = (x, y, bw, bh)
                last_seen[ent_id] = frame_idx

        for ent_id, (x, y, bw, bh) in list(last_bb.items()):
            seen = last_seen.get(ent_id, -999999)
            if max_hold > 0 and (frame_idx - seen) > max_hold:
                del last_bb[ent_id]; del last_seen[ent_id]; continue
            color = id_to_color(ent_id if ent_id is not None else 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = f"id:{ent_id}"
            cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release(); out.release()

    transcode = args.transcode
    if transcode == "none":
        shutil.move(tmpf.name, args.out); print(f"Wrote {args.out}"); return
    if shutil.which("ffmpeg") is None:
        shutil.move(tmpf.name, args.out); print(f"Wrote {args.out} (no ffmpeg)"); return
    if transcode == "h264":
        cmd = ["ffmpeg","-y","-i",tmpf.name,"-c:v","libx264","-crf","18","-preset","veryfast","-movflags","+faststart","-pix_fmt","yuv420p","-c:a","copy",args.out]
    else:
        cmd = ["ffmpeg","-y","-i",tmpf.name,"-c:v","libvpx-vp9","-b:v","0","-crf","30",args.out]
    subprocess.run(cmd, check=True)
    try: os.remove(tmpf.name)
    except Exception: pass
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
