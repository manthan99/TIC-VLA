#!/usr/bin/env python3
"""Render LiDAR / DA3 / hybrid depth comparisons for visual verification.

Per sample, one row of four panels sharing a 0-25 m turbo colormap:
    RGB | LiDAR patch-median (sparse, 16 px patches) | DA3 scale-fitted |
    hybrid (LiDAR patch where returns exist, DA3 elsewhere)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m wildvln.p2b_depth_visuals --per-rig 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from wildvln.p2b_calibrate import load_pairs, project
from wildvln.p2b_depth_eval import load_extrinsic
from wildvln.rigs import UMD, ZED2

CALIB = Path("/data/patelm/ticvla/wildvln/p2b/_calib")
OUT = CALIB / "depth_visuals"
RIGS = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD),
}
PATCH = 16
DMAX = 25.0


def colorize(depth, mask=None):
    d = np.clip(depth, 0.5, DMAX)
    img = cv2.applyColorMap(
        (255 * (1 - (d - 0.5) / (DMAX - 0.5))).astype(np.uint8), cv2.COLORMAP_TURBO)
    if mask is not None:
        img[~mask] = (30, 30, 30)
    return img


def patch_depth(u, v, z, W, H):
    """16-px patch-median LiDAR depth, plus a validity mask, at full res."""
    gw, gh = (W + PATCH - 1) // PATCH, (H + PATCH - 1) // PATCH
    grid = np.full((gh, gw), np.nan, np.float32)
    key = (v // PATCH).astype(int) * gw + (u // PATCH).astype(int)
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    uq, starts = np.unique(ks, return_index=True)
    med = np.array([np.median(zs[a:b]) for a, b in
                    zip(starts, np.append(starts[1:], len(zs)))])
    grid.flat[uq] = med
    full = cv2.resize(grid, (W, H), interpolation=cv2.INTER_NEAREST)
    return full, ~np.isnan(full)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rig", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to("cuda")

    for rig_name, (site, bag, rig) in RIGS.items():
        K, (W, H) = rig.intrinsics, rig.image_size
        fx, fy, cx, cy = K
        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)[None]
        Rcard, params = load_extrinsic(rig_name)
        pairs = [(f, pts @ Rcard.T)
                 for f, pts in load_pairs(site, bag, rig, 60)]
        step = max(1, len(pairs) // args.per_rig)
        picked = pairs[::step][:args.per_rig]

        for i, (f, pts) in enumerate(picked):
            u, v, z, ok = project(pts, params, K, (W, H))
            u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
            if len(z) < 300:
                continue
            lid, lmask = patch_depth(u, v, z, W, H)

            img = np.asarray(Image.open(f).convert("RGB"))
            pred = da3.inference([img], intrinsics=intr)
            d3 = cv2.resize(np.asarray(pred.depth[0], np.float32), (W, H))
            dv = d3[v, u]
            good = dv > 1e-3
            s = float(np.median(z[good] / dv[good]))
            d3s = d3 * s

            hybrid = np.where(lmask, lid, d3s)

            rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            row = np.hstack([
                rgb, colorize(lid, lmask), colorize(d3s), colorize(hybrid)])
            labels = ["RGB", "LiDAR patch (16px)", f"DA3 x{s:.2f}", "hybrid"]
            for j, lab in enumerate(labels):
                cv2.putText(row, lab, (j * W + 8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(str(OUT / f"{rig_name}_{i:02d}.jpg"), row,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{rig_name}: {len(picked)} rows -> {OUT}")


if __name__ == "__main__":
    main()
