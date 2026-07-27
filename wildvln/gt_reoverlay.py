#!/usr/bin/env python3
"""Re-paint GrandTour farm overlay jpgs with pitch-aware projection.

The original overlays projected the yaw-aligned ego trace through the
static T_cam_base, which assumes a level base — on a legged robot the
base pitches on stairs/slopes and the painted trace drifts. Here the
3D path stations are expressed in the TRUE full-SE3 keyframe pose
before projection. Annotations (CoT/memory) are left untouched; only
farm/img/<ep_id>_sNN.jpg files are overwritten.

Usage: python -m wildvln.gt_reoverlay            # all episodes
       python -m wildvln.gt_reoverlay EP_ID ...  # subset
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from wildvln.p4_fullann import TRACE_BUDGET_M, TRACE_PTS, overlay_trace

P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
FARM = Path("/data/patelm/ticvla/grandtour/p4/farm")
BASE_H = 0.55


@functools.lru_cache(maxsize=2)
def load_mission(bag):
    idx = np.load(P2B / bag / "index.npz")
    rj = json.loads((P2B / bag / "rig.json").read_text())
    rig = SimpleNamespace(
        intrinsics=(rj["fx"], rj["fy"], rj["cx"], rj["cy"]),
        image_size=(rj["width"], rj["height"]),
        T_cam_lidar=np.asarray(rj["T_cam_base"]),
        lidar_height_m=BASE_H)
    pose, valid = idx["pose"], idx["valid"]
    pv = pose[valid]
    stamp2i = {int(t * 1e9): i for i, t in enumerate(idx["t"][valid])}
    pw = pv[:, :3, 3]
    seg = np.linalg.norm(np.diff(pw[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return rig, pv, pw, s, stamp2i


def repaint(ep_id):
    e = json.loads((FARM / f"{ep_id}.json").read_text())
    rig, pv, pw, s, stamp2i = load_mission(e["bag"])
    for si, st in enumerate(e["steps"]):
        stamp = int(Path(st["kf"]).stem)
        ki = stamp2i[stamp]
        tlen = min(TRACE_BUDGET_M, s[-1] - s[ki])
        if tlen < 0.5:
            continue
        targets = s[ki] + np.linspace(tlen / TRACE_PTS, tlen, TRACE_PTS)
        st3 = np.stack([np.interp(targets, s, pw[:, k])
                        for k in range(3)], 1)
        st3[:, 2] -= rig.lidar_height_m
        Ti = np.linalg.inv(pv[ki])
        pts3d = st3 @ Ti[:3, :3].T + Ti[:3, 3]
        im = cv2.imread(str(P2B / e["bag"] / "keyframes" / st["kf"]))
        ov = overlay_trace(im, None, rig, pts3d=pts3d)
        cv2.imwrite(str(FARM / "img" / f"{ep_id}_s{si:02d}.jpg"), ov,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(e["steps"])


def main() -> None:
    eps = sys.argv[1:] or sorted(
        p.stem for p in FARM.glob("gt_*.json"))
    # group by mission so the lru_cache(2) actually hits
    eps.sort(key=lambda x: x.split("_")[1])
    n = 0
    for ep_id in eps:
        n += repaint(ep_id)
        print(ep_id, flush=True)
    print(f"GT_REOVERLAY_DONE {len(eps)} eps {n} frames")


if __name__ == "__main__":
    main()
