#!/usr/bin/env python3
"""Draw annotations only on the exact frames listed in the JSON (precise-only).

Usage:
  python viz_precise.py --anno path/to/uid_vid_00000.mp4.json --video path/to/uid_vid_00000.mp4 --out out.mp4
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
                color = id_to_color(ent_id if ent_id is not None else 0)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                lbl = f"id:{ent_id} t:{t}" if t is not None else f"id:{ent_id}"
                cv2.putText(frame, lbl, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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
