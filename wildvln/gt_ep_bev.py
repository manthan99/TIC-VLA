#!/usr/bin/env python3
"""Wavemap BEV renders for farm episode steps.

Per step: crop the mission's wavemap occupancy in the SAME motion-derived
ego frame as the trace GT (x up), tri-state colors (occupied/free/
unknown), overlay history (white) + next-10 m GT trace (green).
Query height follows the PATH elevation (nearest path point + CLEAR_M)
so staircases and slopes are sliced sensibly.

Writes farm/img/<ep_id>_sNN_wm.png for the given episodes.

Usage: python -m wildvln.gt_ep_bev EP_ID [EP_ID ...]
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
WM = Path("/data/patelm/ticvla/grandtour/wavemap")
FARM = Path("/data/patelm/ticvla/grandtour/p4/farm")

HALF_M = 12.0
RES = 0.15
CLEAR_M = 0.5           # slice height above local path elevation

OCC = (40, 40, 200)
FREE = (238, 238, 238)
UNK = (128, 128, 128)


@functools.lru_cache(maxsize=2)
def load_mission(bag):
    import pywavemap as wave
    idx = np.load(P2B / bag / "index.npz")
    m = wave.Map.load(str(WM / f"{bag}.wvmp"))
    pose, valid = idx["pose"], idx["valid"]
    stamp2i = {int(t * 1e9): i for i, t in enumerate(idx["t"])}
    path_xy = pose[valid][:, :2, 3]
    path_z = pose[valid][:, 2, 3]
    return m, idx["t"], pose, stamp2i, cKDTree(path_xy), path_z


def ego(pose, i):
    p = pose[i][:3, 3]
    a = pose[max(i - 1, 0)][:2, 3]
    b = pose[min(i + 1, len(pose) - 1)][:2, 3]
    d = b - a
    ang = np.arctan2(d[1], d[0])
    R = np.array([[np.cos(ang), np.sin(ang)],
                  [-np.sin(ang), np.cos(ang)]])
    return R, p


def to_px(pts, n):
    c = (HALF_M - pts[:, 1]) / RES
    r = (HALF_M - pts[:, 0]) / RES
    return np.clip(np.stack([c, r], 1), 0, n - 1).astype(int)


def render_step(bag, kf_name, trace, history):
    import pywavemap as wave
    m, kt, pose, stamp2i, tree, path_z = load_mission(bag)
    ki = stamp2i[int(Path(kf_name).stem)]
    R, p = ego(pose, ki)

    n = int(2 * HALF_M / RES)
    ax = np.linspace(HALF_M - RES / 2, -HALF_M + RES / 2, n)
    gy, gx = np.meshgrid(ax, ax)          # row = x (fwd, top), col = y
    ex, ey = gx.ravel(), gy.ravel()
    world = np.stack([ex, ey], 1) @ R + p[:2]
    _, j = tree.query(world, k=1)
    zq = path_z[j] - 0.55 + CLEAR_M       # local ground + clearance
    q = np.concatenate([world, zq[:, None]], 1).astype(np.float32)

    interp = m.interpolate
    nearest = wave.InterpolationMode.NEAREST
    vals = np.fromiter((interp(pt, nearest) for pt in q), np.float32,
                       count=len(q)).reshape(n, n)
    img = np.full((n, n, 3), UNK, np.uint8)
    img[vals < -0.1] = FREE
    img[vals > 0.1] = OCC

    if history:
        h = np.asarray(history, float)
        hp = to_px(h, n)
        cv2.polylines(img, [hp], False, (200, 200, 200), 1, cv2.LINE_AA)
    t = np.asarray(trace, float)
    tp = to_px(t, n)
    cv2.polylines(img, [tp], False, (60, 200, 60), 2, cv2.LINE_AA)
    c = n // 2
    cv2.drawMarker(img, (c, c), (30, 30, 30), cv2.MARKER_TRIANGLE_UP, 9, 2)
    return cv2.resize(img, (n * 3, n * 3),
                      interpolation=cv2.INTER_NEAREST)


def main() -> None:
    for ep_id in sys.argv[1:]:
        e = json.loads((FARM / f"{ep_id}.json").read_text())
        for si, st in enumerate(e["steps"]):
            img = render_step(e["bag"], st["kf"],
                              st["target"]["trace"],
                              st["input"]["history"])
            out = FARM / "img" / f"{ep_id}_s{si:02d}_wm.png"
            cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"{ep_id}: {len(e['steps'])} wavemap BEVs", flush=True)
    print("GT_EP_BEV_DONE")


if __name__ == "__main__":
    main()
