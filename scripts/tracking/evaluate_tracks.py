#!/usr/bin/env python3
"""Evaluate predicted tracks against PersonPath22 ground-truth detections.

Per-frame IoU matching (Hungarian when available), counts TP/FP/FN and ID switches,
and prints a small report. Optionally writes per-frame match records.

Usage:
  python evaluate_tracks.py --gt path/to/uid_vid_00000.mp4.json --pred outputs/uid_vid_00000_tracks.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="ground-truth annotation JSON (uses entities with id and blob.frame_idx)")
    p.add_argument("--pred", required=True, help="predicted tracks JSON (tracks_by_frame mapping from simple_tracker.py)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    p.add_argument("--out-matches", help="optional JSON output of per-frame matchings")
    return p.parse_args()


def load_gt(gt_path):
    with open(gt_path, "r") as f:
        data = json.load(f)
    frames = defaultdict(list)
    for ent in data.get("entities", []):
        bb = ent.get("bb")
        blob = ent.get("blob", {})
        fi = blob.get("frame_idx")
        gid = ent.get("id")
        if fi is None or bb is None or gid is None:
            continue
        frames[int(fi)].append({"bb": bb, "gt_id": gid})
    return frames


def load_pred(pred_path):
    with open(pred_path, "r") as f:
        data = json.load(f)
    # expected format: {"tracks_by_frame": {frame: [ {"track_id":.., "bb": [...]}, ... ]}}
    tb = data.get("tracks_by_frame", data)
    frames = defaultdict(list)
    for k, v in tb.items():
        try:
            fi = int(k)
        except Exception:
            fi = k
        for item in v:
            frames[int(fi)].append({"bb": item["bb"], "pred_id": item["track_id"]})
    return frames


def iou(a, b):
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
    ua = aw * ah
    ub = bw * bh
    union = ua + ub - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def match_frame(gt_list, pred_list, iou_thr=0.5):
    # returns list of (gt_idx, pred_idx, iou) for matched pairs, using Hungarian if available
    n_gt = len(gt_list)
    n_pr = len(pred_list)
    if n_gt == 0 or n_pr == 0:
        return []
    cost = [[0.0 for _ in range(n_pr)] for _ in range(n_gt)]
    for i in range(n_gt):
        for j in range(n_pr):
            cost[i][j] = 1.0 - iou(gt_list[i]["bb"], pred_list[j]["bb"])  # lower cost = better

    try:
        from scipy.optimize import linear_sum_assignment
        import math
        import numpy as np
        C = np.array(cost)
        gt_idx, pred_idx = linear_sum_assignment(C)
        matches = []
        for g, p in zip(gt_idx, pred_idx):
            iou_val = 1.0 - float(C[g, p])
            if iou_val >= iou_thr:
                matches.append((g, p, float(iou_val)))
        return matches
    except Exception:
        # fallback greedy by descending IoU
        pairs = []
        for i in range(n_gt):
            for j in range(n_pr):
                pairs.append((i, j, 1.0 - cost[i][j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        matched_g = set(); matched_p = set(); matches = []
        for g, p, v in pairs:
            if v < iou_thr:
                break
            if g in matched_g or p in matched_p:
                continue
            matched_g.add(g); matched_p.add(p); matches.append((g, p, v))
        return matches


def evaluate(gt_frames, pred_frames, iou_thr=0.5):
    # evaluate only on frames that contain ground-truth annotations
    all_frames = sorted(set(list(gt_frames.keys())))
    TP = 0; FP = 0; FN = 0; IDSW = 0
    gt_total = 0
    prev_assignment = {}  # pred_id -> gt_id
    per_frame_matches = {}

    for f in all_frames:
        gts = gt_frames.get(f, [])
        preds = pred_frames.get(f, [])
        gt_total += len(gts)
        matches = match_frame(gts, preds, iou_thr=iou_thr)
        matched_g = set(); matched_p = set()
        records = []
        for gi, pi, val in matches:
            gi = int(gi); pi = int(pi); val = float(val)
            matched_g.add(gi); matched_p.add(pi)
            TP += 1
            pred_id = preds[pi]["pred_id"]
            gt_id = gts[gi]["gt_id"]
            records.append({"gt_idx": gi, "pred_idx": pi, "gt_id": int(gt_id) if not isinstance(gt_id, str) else gt_id, "pred_id": int(pred_id) if not isinstance(pred_id, str) else pred_id, "iou": val})
            # ID switch: if this pred was previously assigned to a different gt
            prev = prev_assignment.get(pred_id)
            if prev is not None and prev != gt_id:
                IDSW += 1
            prev_assignment[pred_id] = gt_id

        # unmatched predictions -> FP
        for pi, p in enumerate(preds):
            if pi not in matched_p:
                FP += 1
                records.append({"pred_idx": pi, "pred_id": p["pred_id"], "unmatched_pred": True})
        # unmatched GT -> FN
        for gi, g in enumerate(gts):
            if gi not in matched_g:
                FN += 1
                records.append({"gt_idx": gi, "gt_id": g["gt_id"], "unmatched_gt": True})

        per_frame_matches[f] = records

    mota = 1.0 - float(FN + FP + IDSW) / float(gt_total) if gt_total > 0 else 0.0
    results = {"TP": TP, "FP": FP, "FN": FN, "IDSW": IDSW, "GT_total": gt_total, "MOTA": mota}
    return results, per_frame_matches


def main():
    args = parse_args()
    gt = load_gt(args.gt)
    pred = load_pred(args.pred)
    res, per_frame = evaluate(gt, pred, iou_thr=args.iou)
    print("Evaluation:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    if args.out_matches:
        with open(args.out_matches, "w") as f:
            json.dump({"per_frame": per_frame, "summary": res}, f)
        print(f"Wrote per-frame matches to: {args.out_matches}")


if __name__ == '__main__':
    main()
