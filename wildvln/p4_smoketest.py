#!/usr/bin/env python3
"""P4 smoke test: one real annotation call against the local vLLM server.

Picks a P3 window, overlays its GT future trace on the anchor keyframe
(image-space via official calibration, ground-plane projection), and asks
the annotator model for exactly what the language farm needs: landmark
naming + a navigation instruction, in strict JSON.

Usage: python -m wildvln.p4_smoketest [--n 3] [--port 8117]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

from wildvln.rigs import rig_for_site

P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
P3 = Path("/data/patelm/ticvla/wildvln/p3/windows.parquet")
OUT = Path("/data/patelm/ticvla/wildvln/p4/_smoke")

PROMPT = """You are annotating data for a robot navigation dataset. The image is from a ground robot's forward camera. The green line is a PRIVILEGED HINT for you only: it shows where the robot actually went over the next 10 meters (projected onto the ground). The robot that will be trained on your annotation CANNOT see this line — never mention the line, its color, or any overlay in your outputs.

Write the instruction as a human guide would: imperative, grounded ONLY in distinctive PERMANENT scene elements (signs, buildings, fences, trees, doors, poles). NEVER anchor on people, animals, or vehicles that could move away. It must uniquely determine the executed route: if the robot could plausibly go two ways here (e.g. two walkways, a fork, an open plaza), the instruction must rule the wrong ones out. Avoid vague words like "the path", "the curve", "the area".

Return STRICT JSON, no prose, with keys:
  "scene": one sentence describing the environment,
  "landmarks": list of 2-5 objects, each {"name": short distinctive name, "side": "left"|"right"|"ahead", "distance_m": rough number},
  "instruction": one imperative sentence (max 22 words) that a robot seeing ONLY the raw image could follow to reproduce the executed route,
  "alternatives": name every other visible route the robot could physically take (other walkways, road crossings, openings, grass shortcuts) and how the instruction excludes each; write "none" ONLY if no other route is visible,
  "path_summary": one of "straight", "curve_left", "curve_right", "sharp_left", "sharp_right"."""

def trace_overlay(row, rig):
    site, bag = row["site"], row["bag"]
    img_p = P2B_ROOT / site / bag / "keyframes" / row["kf"]
    img = cv2.imread(str(img_p))
    if img is None:
        return None
    tx = np.array([row[f"tx{i}"] for i in range(10)])
    ty = np.array([row[f"ty{i}"] for i in range(10)])
    # robot frame (x fwd, y left) on ground -> lidar frame -> camera
    h = rig.lidar_height_m
    pts_l = np.stack([tx, -ty * -1 * -1, np.full(10, -h)], 1)  # y_l = +left
    pts_l[:, 1] = ty
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
        cv2.line(img, p, q, (60, 220, 60), 3, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8117)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(P3)
    rng = np.random.default_rng(7)
    picks = df.iloc[rng.choice(len(df), args.n, replace=False)]
    for i, (_, row) in enumerate(picks.iterrows()):
        rig = rig_for_site(row["site"])
        img = trace_overlay(row, rig)
        if img is None:
            continue
        p = OUT / f"smoke{i}_{row['site']}.jpg"
        cv2.imwrite(str(p), img)
        b64 = base64.b64encode(p.read_bytes()).decode()
        r = requests.post(
            f"http://localhost:{args.port}/v1/chat/completions",
            json={"model": "qwen3.5-vl",
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url",
                       "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                      {"type": "text", "text": PROMPT}]}],
                  "max_tokens": 500, "temperature": 0.2},
            timeout=300)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        print(f"=== {row['site']}/{row['bag']} t={row['t']:.1f} -> {p.name}")
        print(txt.strip()[:800])
        try:
            j = json.loads(txt.strip().removeprefix("```json").removesuffix("```").strip())
            print(f"[JSON OK] landmarks={len(j.get('landmarks', []))} "
                  f"path={j.get('path_summary')}")
        except Exception as e:
            print(f"[JSON FAIL] {e}")


if __name__ == "__main__":
    main()
