#!/usr/bin/env python3
"""Visualize per-frame bounding boxes from annotation JSON onto video frames.

Usage:
  python viz_annotations.py --anno path/to/uid_vid_00000.mp4.json [--video path/to/uid_vid_00000.mp4] --out out.mp4

If --video is omitted the script will try to find the .mp4 next to the annotation by replacing
the annotation directory with `../raw_data` and switching `.json` -> `.mp4`.
"""
import argparse
import json
import os
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anno", required=True, help="annotation JSON path")
    p.add_argument("--video", help="video file path (optional)")
    p.add_argument("--out", required=True, help="output annotated video path")
    p.add_argument("--transcode", choices=["h264","webm","none"], default="h264", help="transcode final output via ffmpeg (default: h264)")
    p.add_argument("--max-frames", type=int, default=0, help="limit number of frames to process (0 = all)")
    p.add_argument("--max-hold-frames", type=int, default=0, help="max frames to carry-forward a detection (0 = infinite)")
    return p.parse_args()


def load_annotations(path):
    with open(path, "r") as f:
        data = json.load(f)
    # build mapping frame_idx -> list of (x,y,w,h,id,time)
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
    # deterministic color per id
    r = (i * 37) % 255
    g = (i * 83) % 255
    b = (i * 151) % 255
    return (int(b), int(g), int(r))


def main():
    args = parse_args()

    try:
        import cv2
    except Exception as e:
        print("OpenCV (cv2) is required. Install via: pip install opencv-python", file=sys.stderr)
        raise

    annopath = args.anno
    if not os.path.exists(annopath):
        print(f"Annotation not found: {annopath}", file=sys.stderr)
        sys.exit(2)

    # derive video path if not provided
    video_path = args.video
    if not video_path:
        # try to find raw_data sibling: replace /annotation/.*.json -> /raw_data/<same_basename>.mp4
        bp = os.path.abspath(annopath)
        parts = bp.split(os.sep)
        try:
            idx = parts.index("annotation")
            parts[idx] = "raw_data"
            # replace final json file name with .mp4
            parts[-1] = os.path.splitext(parts[-1])[0] + ".mp4"
            candidate = os.sep.join(parts)
            if os.path.exists(candidate):
                video_path = candidate
        except ValueError:
            pass

    if not video_path or not os.path.exists(video_path):
        print("Video file not found. Provide --video or place .mp4 under raw_data next to annotations.")
        sys.exit(2)

    frames_map = load_annotations(annopath)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # write to a temporary mp4 (mp4v) then optionally transcode
    import tempfile, subprocess, shutil

    tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmpf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmpf.name, fourcc, fps, (w, h))

    frame_idx = 0
    max_frames = args.max_frames
    max_hold = args.max_hold_frames

    # carry-forward state: last bbox and last seen frame per id
    last_bb = {}          # id -> (x,y,w,h)
    last_seen = {}        # id -> frame_idx when last updated

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update detections for this exact frame
        if frame_idx in frames_map:
            for bb, ent_id, t in frames_map[frame_idx]:
                try:
                    x, y, bw, bh = map(int, bb)
                except Exception:
                    continue
                last_bb[ent_id] = (x, y, bw, bh)
                last_seen[ent_id] = frame_idx

        # draw carried-forward boxes for all ids that are within hold window
        for ent_id, (x, y, bw, bh) in list(last_bb.items()):
            seen = last_seen.get(ent_id, -999999)
            if max_hold > 0 and (frame_idx - seen) > max_hold:
                # drop expired
                del last_bb[ent_id]
                del last_seen[ent_id]
                continue
            color = id_to_color(ent_id if ent_id is not None else 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = f"id:{ent_id}"
            cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()

    # transcode if requested
    transcode = getattr(args, "transcode", "h264")
    if transcode == "none":
        shutil.move(tmpf.name, args.out)
        print(f"Wrote annotated video: {args.out}")
        return

    # build ffmpeg command
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; install ffmpeg to enable transcoding to H.264", file=sys.stderr)
        shutil.move(tmpf.name, args.out)
        print(f"Wrote annotated video (no transcode): {args.out}")
        return

    if transcode == "h264":
        cmd = [
            "ffmpeg", "-y", "-i", tmpf.name,
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-c:a", "copy", args.out
        ]
    elif transcode == "webm":
        cmd = ["ffmpeg", "-y", "-i", tmpf.name, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", args.out]
    else:
        # unknown option, just move
        shutil.move(tmpf.name, args.out)
        print(f"Wrote annotated video: {args.out}")
        return

    print("Transcoding annotated video with ffmpeg...")
    subprocess.run(cmd, check=True)
    try:
        os.remove(tmpf.name)
    except Exception:
        pass
    print(f"Wrote annotated video: {args.out}")


if __name__ == "__main__":
    main()
