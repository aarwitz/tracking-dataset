#!/usr/bin/env python3
"""Draw annotations with linear interpolation between frames for each `id`.

Only interpolates between known detections for the same id. Outside the annotated
range for an id nothing is drawn for that id.
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


def load_by_id(path):
    with open(path, "r") as f:
        data = json.load(f)
    by_id = {}
    for ent in data.get("entities", []):
        bb = ent.get("bb")
        blob = ent.get("blob", {})
        fi = blob.get("frame_idx")
        if fi is None or bb is None:
            continue
        ent_id = ent.get("id", -1)
        by_id.setdefault(ent_id, []).append((fi, bb))
    # sort lists
    for k in by_id:
        by_id[k].sort(key=lambda x: x[0])
    return by_id


def id_to_color(i):
    r = (i * 37) % 255
    g = (i * 83) % 255
    b = (i * 151) % 255
    return (int(b), int(g), int(r))


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    args = parse_args()
    try:
        import cv2
    except Exception:
        print("Install opencv-python", file=sys.stderr); raise

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

    by_id = load_by_id(annopath)

    # prepare per-id pointers to the current segment
    ptr = {k: 0 for k in by_id}

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

        # for each id, advance pointer to the segment containing or just before frame_idx
        for ent_id, lst in by_id.items():
            p = ptr[ent_id]
            # advance while next point's frame <= current
            while p + 1 < len(lst) and lst[p + 1][0] <= frame_idx:
                p += 1
            ptr[ent_id] = p
            # now determine if we can interpolate: need p and p+1 where lst[p][0] <= frame_idx < lst[p+1][0]
            if p + 1 < len(lst) and lst[p][0] <= frame_idx < lst[p + 1][0]:
                f0, bb0 = lst[p]
                f1, bb1 = lst[p + 1]
                if f1 == f0:
                    t = 0.0
                else:
                    t = float(frame_idx - f0) / float(f1 - f0)
                x0, y0, w0, h0 = map(float, bb0)
                x1, y1, w1, h1 = map(float, bb1)
                xi = int(round(lerp(x0, x1, t)))
                yi = int(round(lerp(y0, y1, t)))
                wi = int(round(lerp(w0, w1, t)))
                hi = int(round(lerp(h0, h1, t)))
                color = id_to_color(ent_id if ent_id is not None else 0)
                cv2.rectangle(frame, (xi, yi), (xi + wi, yi + hi), color, 2)
                cv2.putText(frame, f"id:{ent_id}", (xi, max(0, yi - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                # not between two known detections; do nothing (no carry)
                pass

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
