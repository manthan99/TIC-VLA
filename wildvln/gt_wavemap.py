#!/usr/bin/env python3
"""GrandTour occupancy GT via wavemap: integrate Hesai scans through the
DLIO poses (T_dlio_map->hesai directly, per devkit convention).

Not shipped with the dataset — we build it. Output: <mission>.wvmp +
a QC slice render (occupied / free / UNKNOWN — the tri-state that the
GND voxel maps could never give us).

Usage: python -m wildvln.gt_wavemap --mission 2024-12-03-13-26-40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/patelm/ticvla/grandtour/raw")
OUT = Path("/data/patelm/ticvla/grandtour/wavemap")

CELL_M = 0.1
MAX_RANGE = 25.0
SCAN_STRIDE = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--stride", type=int, default=SCAN_STRIDE)
    ap.add_argument("--qc-only", action="store_true",
                    help="load stored .wvmp and just redo the QC slice")
    args = ap.parse_args()
    import zarr
    import pywavemap as wave
    from scipy.spatial.transform import Rotation as R

    mdir = ROOT / args.mission
    OUT.mkdir(parents=True, exist_ok=True)
    hes = zarr.open_group(str(mdir / "data" / "hesai_points_undistorted"),
                          mode="r", zarr_format=2)
    od = zarr.open_group(str(mdir / "data" / "dlio_map_odometry"),
                         mode="r", zarr_format=2)
    ots, opos, oq = (od["timestamp"][:], od["pose_pos"][:],
                     od["pose_orien"][:])
    sts = hes["timestamp"][:]

    out_map = OUT / f"{args.mission}.wvmp"
    if args.qc_only:
        the_map = wave.Map.load(str(out_map))
    else:
        the_map = wave.Map.create({
            "type": "hashed_chunked_wavelet_octree",
            "min_cell_width": {"meters": CELL_M}})
    pipeline = wave.Pipeline(the_map)
    pipeline.add_operation({"type": "threshold_map",
                            "once_every": {"seconds": 10.0}})
    pipeline.add_integrator("hesai", {
        # Hesai XT-32: 32 beams, -16..15 deg vertical, 360 deg
        "projection_model": {
            "type": "spherical_projector",
            "elevation": {"num_cells": 64,
                          "min_angle": {"degrees": -16.5},
                          "max_angle": {"degrees": 15.5}},
            "azimuth": {"num_cells": 1024,
                        "min_angle": {"degrees": -180.0},
                        "max_angle": {"degrees": 180.0}}},
        "measurement_model": {
            "type": "continuous_beam",
            "angle_sigma": {"degrees": 0.1},
            "range_sigma": {"meters": 0.05},
            "scaling_free": 0.2,
            "scaling_occupied": 0.4},
        "integration_method": {
            "type": "hashed_chunked_wavelet_integrator",
            "min_range": {"meters": 0.7},
            "max_range": {"meters": MAX_RANGE}}})

    n = len(sts)
    for i in ([] if args.qc_only else range(0, n, args.stride)):
        j = int(np.argmin(np.abs(ots - sts[i])))
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat(oq[j]).as_matrix()
        T[:3, 3] = opos[j]
        pts = hes["points"][i]
        rng = np.linalg.norm(pts, axis=1)
        ok = np.isfinite(rng) & (rng > 0.7) & (rng < MAX_RANGE)
        cloud = wave.Pointcloud(np.asfortranarray(pts[ok].T,
                                                  dtype=np.float32))
        pipeline.run_pipeline(["hesai"], wave.PosedPointcloud(
            wave.Pose(np.asfortranarray(T)), cloud))
        if i % 200 == 0:
            print(f"  scan {i}/{n}", flush=True)

    if not args.qc_only:
        the_map.threshold()
        the_map.prune()
        the_map.store(str(out_map))
    print("map ->", out_map, f"{out_map.stat().st_size/1e6:.1f} MB")

    # QC slice: occupancy at trajectory height + 0.5 m
    zs = np.median(opos[:, 2]) + 0.5
    mn, mx = opos[:, :2].min(0) - 15, opos[:, :2].max(0) + 15
    res = 0.2                       # QC only; interpolate() is per-point
    xs = np.arange(mn[0], mx[0], res)
    ys = np.arange(mn[1], mx[1], res)
    gx, gy = np.meshgrid(xs, ys)
    q = np.stack([gx.ravel(), gy.ravel(),
                  np.full(gx.size, zs)], 1).astype(np.float32)
    interp = the_map.interpolate
    nearest = wave.InterpolationMode.NEAREST
    vals = np.fromiter((interp(p, nearest) for p in q), np.float32,
                       count=len(q)).reshape(gy.shape)
    img = np.full((*vals.shape, 3), 128, np.uint8)      # unknown gray
    img[vals < -0.1] = (255, 255, 255)                   # free white
    img[vals > 0.1] = (40, 40, 200)                      # occupied red
    px = ((opos[:, :2] - mn) / res).astype(int)
    for a, b in zip(px[:-1], px[1:]):
        cv2.line(img, tuple(a), tuple(b), (80, 200, 80), 2)
    qc = OUT / f"{args.mission}_slice.png"
    cv2.imwrite(str(qc), img[::-1])
    unk = float((np.abs(vals) <= 0.1).mean())
    print(f"slice: occupied {(vals > .1).mean()*100:.1f}% "
          f"free {(vals < -.1).mean()*100:.1f}% unknown {unk*100:.1f}%")
    print("WAVEMAP_DONE", qc)


if __name__ == "__main__":
    main()
