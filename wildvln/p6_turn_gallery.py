#!/usr/bin/env python3
"""Render turn-step comparison images from eval dumps (CPU only).

Each output jpg: camera frame + GT trace (green) + one colored predicted
trace per model dump. Selects steps whose GT trace net heading >= 20 deg.

Usage:
  python -m wildvln.p6_turn_gallery --out DIR m0=/path/m0.jsonl m2=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wildvln.p4_fullann import overlay_trace
from wildvln.p6_slice import net_heading, TURN_DEG
from wildvln.rigs import rig_for_site

COLORS = {  # BGR
    "m0": (60, 60, 230), "m1": (0, 150, 255),
    "m2": (230, 120, 40), "m3": (200, 40, 200),
    "m4": (0, 165, 255), "kin": (128, 128, 128),
}


def draw_pred(img, pred, rig, color):
    h = rig.lidar_height_m
    pred = np.asarray(pred, float)
    pts_l = np.concatenate([pred, np.full((len(pred), 1), -h)], 1)
    T = np.asarray(rig.T_cam_lidar)
    Xc = pts_l @ T[:3, :3].T + T[:3, 3]
    fx, fy, cx, cy = rig.intrinsics
    ok = Xc[:, 2] > 0.5
    u = (fx * Xc[:, 0] / Xc[:, 2] + cx)[ok]
    v = (fy * Xc[:, 1] / Xc[:, 2] + cy)[ok]
    W, H = rig.image_size
    pts = [(int(a), int(b)) for a, b in zip(u, v)
           if 0 <= a < W and 0 <= b < H]
    for p, q in zip(pts[:-1], pts[1:]):
        cv2.line(img, p, q, color, 3, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test_site")
    ap.add_argument("dumps", nargs="+", help="name=path.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("/data/patelm/ticvla/wildvln/p5/samples.parquet")
    sub = df[(df["mode"] == "chained") & (df.variant == 0)]
    idx = {(r["ep_id"], int(r["step"])): r for _, r in sub.iterrows()}

    preds = {}          # (ep_id, step) -> {model: rec}
    for spec in args.dumps:
        name, path = spec.split("=", 1)
        for line in open(path):
            r = json.loads(line)
            if r["split"] != args.split:
                continue
            preds.setdefault((r["ep_id"], r["step"]), {})[name] = r

    meta = []
    for (ep, step), by_model in sorted(preds.items()):
        any_rec = next(iter(by_model.values()))
        nh = net_heading(any_rec["gt"])
        if nh is None or abs(nh) < TURN_DEG:
            continue
        row = idx[(ep, step)]
        rig = rig_for_site(row["site"])
        img = cv2.imread(row["image"])
        img = overlay_trace(img, np.asarray(any_rec["gt"]), rig)   # green GT
        for name, rec in by_model.items():
            draw_pred(img, rec["pred"], rig,
                      COLORS.get(name, (255, 255, 255)))
        fname = f"{ep}_s{step:02d}.jpg"
        cv2.imwrite(str(out / fname), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        meta.append({
            "img": fname, "ep_id": ep, "step": step,
            "nh_gt": round(nh, 1), "instruction": row["instruction"],
            "gt_cot": row["cot"], "gt_mem": row["memory_out"],
            "models": {n: {"ade": round(r["ade"], 2),
                           "fde": round(r["fde"], 2),
                           "nh_pred": round(net_heading(r["pred"]) or 0, 1)}
                       for n, r in by_model.items()}})
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print("GALLERY_DONE", len(meta), "->", out)


if __name__ == "__main__":
    main()
