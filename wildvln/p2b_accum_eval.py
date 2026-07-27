#!/usr/bin/env python3
"""P2b-4: accumulated-scan LiDAR depth under the OFFICIAL calibration.

The earlier accumulation numbers (UMD +-2 s absrel 0.087, ZED2 +-0.5 s
0.233) were measured under the wrong Velodyne cardinal and stale UMD
intrinsics, and the sky artifacts that killed the idea were solar returns
(now filtered in the UMD dense cache) plus stray-point patches (now handled
by the mode-cluster estimator). Rerun cleanly:

    single          fit-split of the anchor scan only (baseline)
    accum +-W       fit-split + neighbor scans within W seconds, ego-motion
                    compensated through P1c poses, z-buffer culled

Scored on the eval split of the anchor scan (never fed to any grid).
Renders: RGB | accumulated sparse | single patch | accumulated patch.

Usage:
    python -m wildvln.p2b_accum_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from wildvln.liftdepth import (MIN_PTS_ACCUM, MIN_PTS_SINGLE, PATCH,
                               patch_depth_grid, zbuffer_cull)
from wildvln.p2b_depth_eval_official import (colorize, label,
                                             project_official, sparse_overlay)
from wildvln.rigs import UMD, ZED2

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
CALIB = P2B_ROOT / "_calib"
OUT_VIS = CALIB / "depth_visuals7"

RIGS = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2, "AU_dense.npz"),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD,
                "UMD_map1_2_lot9_dense_filt.npz"),
}
WINDOWS = (0.5, 1.0, 2.0)
# Window shown in the QC rows: the measured per-rig sweet spot (UMD holds
# absrel to +-1 s; ZED2 poses are noisier and degrade beyond +-0.5 s).
VIS_WINDOW = {"gnd-zed2": 0.5, "gnd-umd": 1.0}


def load_anchor_sets(site, bag, rig, dense_name):
    """Per anchor: (img, single_pts, {W: neighbor_pts}) in kf lidar frame."""
    dz = np.load(CALIB / dense_name)
    ct, anchors = dz["t"], dz["anchors"]
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kt, kseg = idx["t"], idx["seg_id"]
    pz = np.load(P1_ROOT / site / bag / "poses_repaired.npz")
    pt, pp, pseg = pz["t"], pz["poses"], pz["seg_id"]

    out = []
    for ta in anchors:
        ki = int(np.argmin(np.abs(kt - ta)))
        if abs(kt[ki] - ta) > 0.3 or kseg[ki] < 0:
            continue
        tk = kt[ki]
        pj = int(np.argmin(np.abs(pt - tk)))
        if pseg[pj] < 0:
            continue
        inv_kf = np.linalg.inv(pp[pj])

        def to_kf(ci):
            pi = int(np.argmin(np.abs(pt - ct[ci])))
            if pseg[pi] != pseg[pj]:
                return None
            rel = inv_kf @ pp[pi]
            return dz[f"c{ci}"] @ rel[:3, :3].T + rel[:3, 3]

        ci0 = int(np.argmin(np.abs(ct - tk)))
        single = to_kf(ci0)
        if single is None or len(single) < 400:
            continue
        neigh = {}
        for W in WINDOWS:
            sel = np.where((np.abs(ct - tk) <= W) & (np.arange(len(ct)) != ci0))[0]
            pts = [p for ci in sel if (p := to_kf(ci)) is not None]
            neigh[W] = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
        img = P2B_ROOT / site / bag / "keyframes" / f"{int(tk*1e9)}.jpg"
        if img.exists():
            out.append((str(img), single, neigh))
    return out


def main() -> None:
    OUT_VIS.mkdir(parents=True, exist_ok=True)
    report = {}
    for rig_name, (site, bag, rig, dense_name) in RIGS.items():
        W_img, H_img = rig.image_size
        sets = load_anchor_sets(site, bag, rig, dense_name)
        print(f"{rig_name}: {len(sets)} anchors")

        res = {k: {"errs": [], "cov": []}
               for k in ["single"] + [f"accum+-{W}s" for W in WINDOWS]}
        rng = np.random.default_rng(0)
        for vi, (img, single, neigh) in enumerate(sets):
            u, v, z, ok = project_official(single, rig)
            u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
            fit = rng.random(len(z)) < 0.5
            ue, ve, ze = u[~fit], v[~fit], z[~fit]

            def score(name, grid, gmask):
                pe = grid[ve // PATCH, ue // PATCH]
                good = ~np.isnan(pe)
                if good.sum() < 50:
                    return
                res[name]["errs"].append(float(np.mean(
                    np.abs(pe[good] - ze[good]) / ze[good])))
                res[name]["cov"].append(float(gmask.mean()))

            g1, m1 = patch_depth_grid(u[fit], v[fit], z[fit], W_img, H_img,
                                      MIN_PTS_SINGLE)
            score("single", g1, m1)

            grids = {}
            for Wn in WINDOWS:
                un, vn, zn, okn = project_official(neigh[Wn], rig)
                au = np.concatenate([u[fit], un[okn].astype(int)])
                av = np.concatenate([v[fit], vn[okn].astype(int)])
                az = np.concatenate([z[fit], zn[okn]])
                keep = zbuffer_cull(au, av, az, W_img, H_img)
                ga, ma = patch_depth_grid(au[keep], av[keep], az[keep],
                                          W_img, H_img, MIN_PTS_ACCUM)
                score(f"accum+-{Wn}s", ga, ma)
                grids[Wn] = (au[keep], av[keep], az[keep], ga)

            Wv = VIS_WINDOW[rig_name]
            au, av, az, ga = grids[Wv]
            rgb = cv2.imread(img)
            row = np.concatenate([
                label(rgb.copy(), "RGB"),
                label(sparse_overlay(rgb, au, av, az),
                      f"accum +-{Wv}s ({len(az)//1000}k pts)"),
                label(cv2.resize(colorize(g1), (W_img, H_img),
                                 interpolation=cv2.INTER_NEAREST),
                      "patch depth: single scan"),
                label(cv2.resize(colorize(ga), (W_img, H_img),
                                 interpolation=cv2.INTER_NEAREST),
                      f"patch depth: accum +-{Wv}s"),
            ], axis=1)
            cv2.imwrite(str(OUT_VIS / f"{rig_name}_{vi:02d}.jpg"), row,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

        summary = {k: {"absrel": round(float(np.median(r["errs"])), 3),
                       "patch_coverage": round(float(np.mean(r["cov"])), 3)}
                   for k, r in res.items() if r["errs"]}
        report[rig_name] = summary
        print(f"\n=== {rig_name} (official calibration) ===")
        print(f"{'source':14s} {'absrel':>7s} {'coverage':>9s}")
        for k, r in summary.items():
            print(f"{k:14s} {r['absrel']:7.3f} {r['patch_coverage']:9.1%}")

    json.dump(report, open(CALIB / "accum_eval_official.json", "w"), indent=1)
    print(f"\n-> {CALIB}/accum_eval_official.json\n-> {OUT_VIS}/")


if __name__ == "__main__":
    main()
