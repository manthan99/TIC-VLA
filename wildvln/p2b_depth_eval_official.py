#!/usr/bin/env python3
"""P2b-3 rerun: DA3 vs LiDAR patch depth under the OFFICIAL calibration.

The first bake-off scored DA3 under fitted extrinsics that later proved
wrong (Velodyne cardinal flip, stale UMD bag intrinsics fx=265 vs true
357.8). DA3 takes intrinsics as input, so the UMD run was doubly corrupted:
wrong metric prior in, wrong LiDAR ground truth out. This rerun projects
through rigs.py T_cam_lidar + authoritative intrinsics only.

Scored on held-out LiDAR (50/50 fit/eval split per frame):
    lidar-patch   mode-cluster patch depth from the fit split (liftdepth.py)
    da3/scale     DA3-LARGE with official intrinsics, per-frame median scale
    da3/affine    same, robust affine in inverse depth
    hybrid        lidar-patch where covered, scaled DA3 elsewhere

Also renders QC rows: RGB | sparse LiDAR | DA3 scaled | hybrid patch grid.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m wildvln.p2b_depth_eval_official
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from wildvln.liftdepth import PATCH, patch_depth_grid
from wildvln.p2b_calibrate import load_pairs
from wildvln.p2b_depth_eval import affine_fit
from wildvln.rigs import UMD, ZED2

CALIB = Path("/data/patelm/ticvla/wildvln/p2b/_calib")
OUT_VIS = CALIB / "depth_visuals6"
RIGS = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD),
}
N_FRAMES = 40
N_VIS = 10
ZMIN, ZMAX = 0.7, 40.0


def project_official(pts, rig):
    T = np.asarray(rig.T_cam_lidar, np.float64)
    Xc = pts @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    ok = z > 0.5
    fx, fy, cx, cy = rig.intrinsics
    u = fx * Xc[:, 0] / np.where(ok, z, 1) + cx
    v = fy * Xc[:, 1] / np.where(ok, z, 1) + cy
    W, H = rig.image_size
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    return u, v, z, ok


def da3_depths(imgs, K):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to("cuda")
    fx, fy, cx, cy = K
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)[None]
    out = {}
    for f in imgs:
        img = np.asarray(Image.open(f).convert("RGB"))
        pred = m.inference([img], intrinsics=intr)
        d = np.asarray(pred.depth[0], np.float32)
        out[f] = cv2.resize(d, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    del m
    import torch
    torch.cuda.empty_cache()
    return out


def colorize(depth, mask=None):
    d = np.clip(depth, ZMIN, ZMAX)
    norm = (np.log(d) - np.log(ZMIN)) / (np.log(ZMAX) - np.log(ZMIN))
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    if mask is not None:
        img[~mask] = 30
    return img


def sparse_overlay(rgb, u, v, z):
    img = rgb.copy()
    d = np.clip(z, ZMIN, ZMAX)
    norm = (np.log(d) - np.log(ZMIN)) / (np.log(ZMAX) - np.log(ZMIN))
    colors = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8),
                               cv2.COLORMAP_TURBO)[:, 0]
    for x, y, c in zip(u.astype(int), v.astype(int), colors):
        cv2.circle(img, (x, y), 1, tuple(int(q) for q in c), -1)
    return img


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    OUT_VIS.mkdir(parents=True, exist_ok=True)
    report = {}
    for rig_name, (site, bag, rig) in RIGS.items():
        W, H = rig.image_size
        pairs = load_pairs(site, bag, rig, N_FRAMES)
        imgs = [f for f, _ in pairs]

        rng = np.random.default_rng(0)
        frames = []
        for f, pts in pairs:
            u, v, z, ok = project_official(pts, rig)
            u, v, z = u[ok], v[ok], z[ok]
            if len(z) < 400:
                continue
            fit = rng.random(len(z)) < 0.5
            frames.append((f, u, v, z, fit))

        dm = da3_depths(imgs, rig.intrinsics)

        res = {k: {"errs": [], "errs_hi": [], "cov": []}
               for k in ("lidar-patch", "da3/scale", "da3/affine", "hybrid")}
        vis_count = 0
        for f, u, v, z, fit in frames:
            uf, vf, zf = u[fit], v[fit], z[fit]
            ue, ve, ze = u[~fit], v[~fit], z[~fit]

            # --- lidar patch grid from fit split (official estimator) ---
            grid, gmask = patch_depth_grid(
                uf.astype(int), vf.astype(int), zf, W, H)
            pe = grid[(ve // PATCH).astype(int), (ue // PATCH).astype(int)]
            good = ~np.isnan(pe)
            e = np.abs(pe[good] - ze[good]) / ze[good]
            res["lidar-patch"]["errs"].append(float(np.mean(e)))
            hi = good & (ve < H // 2)
            if hi.sum() > 20:
                res["lidar-patch"]["errs_hi"].append(float(np.mean(
                    np.abs(pe[hi] - ze[hi]) / ze[hi])))
            res["lidar-patch"]["cov"].append(float(gmask.mean()))

            # --- DA3 ---
            d = dm[f]
            dvf = d[vf.astype(int), uf.astype(int)]
            dve = d[ve.astype(int), ue.astype(int)]
            gd = dvf > 1e-3
            if gd.sum() < 100:
                continue
            scale = float(np.median(zf[gd] / dvf[gd]))
            for mode in ("scale", "affine"):
                if mode == "scale":
                    pred = scale * dve
                else:
                    pred = affine_fit(zf[gd], dvf[gd], dve)
                ok2 = dve > 1e-3
                e = np.abs(pred[ok2] - ze[ok2]) / ze[ok2]
                res[f"da3/{mode}"]["errs"].append(float(np.mean(e)))
                hi = ok2 & (ve < H // 2)
                if hi.sum() > 20:
                    ph = scale * dve[hi] if mode == "scale" else \
                        affine_fit(zf[gd], dvf[gd], dve[hi])
                    res[f"da3/{mode}"]["errs_hi"].append(float(np.mean(
                        np.abs(ph - ze[hi]) / ze[hi])))
                res[f"da3/{mode}"]["cov"].append(1.0)

            # --- hybrid: lidar patch where covered, scaled DA3 elsewhere ---
            ph = np.where(np.isnan(pe), scale * dve, pe)
            ok2 = ~np.isnan(ph)
            e = np.abs(ph[ok2] - ze[ok2]) / ze[ok2]
            res["hybrid"]["errs"].append(float(np.mean(e)))
            hi = ok2 & (ve < H // 2)
            if hi.sum() > 20:
                res["hybrid"]["errs_hi"].append(float(np.mean(
                    np.abs(ph[hi] - ze[hi]) / ze[hi])))
            res["hybrid"]["cov"].append(1.0)

            # --- QC row ---
            if vis_count < N_VIS:
                rgb = cv2.imread(f)
                da3_scaled = colorize(scale * d)
                # hybrid patch grid rendered at patch resolution
                d_small = cv2.resize(scale * d, (grid.shape[1], grid.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                hyb = np.where(np.isnan(grid), d_small, grid)
                hyb_img = cv2.resize(colorize(hyb), (W, H),
                                     interpolation=cv2.INTER_NEAREST)
                row = np.concatenate([
                    label(rgb, "RGB"),
                    label(sparse_overlay(rgb, u, v, z), "LiDAR (official calib)"),
                    label(da3_scaled, "DA3 x per-frame scale"),
                    label(hyb_img, "hybrid patch (lidar>da3)"),
                ], axis=1)
                cv2.imwrite(str(OUT_VIS / f"{rig_name}_{vis_count:02d}.jpg"),
                            row, [cv2.IMWRITE_JPEG_QUALITY, 88])
                vis_count += 1

        summary = {}
        for k, r in res.items():
            summary[k] = {
                "absrel": round(float(np.median(r["errs"])), 3),
                "absrel_upper": round(float(np.median(r["errs_hi"])), 3)
                if r["errs_hi"] else None,
                "patch_coverage": round(float(np.mean(r["cov"])), 3),
            }
        report[rig_name] = summary
        print(f"\n=== {rig_name} (official calibration) ===")
        print(f"{'source':16s} {'absrel':>7s} {'upper-half':>10s} {'coverage':>9s}")
        for k, r in summary.items():
            print(f"{k:16s} {r['absrel']:7.3f} {str(r['absrel_upper']):>10s} "
                  f"{r['patch_coverage']:9.1%}")

    json.dump(report, open(CALIB / "depth_eval_official.json", "w"), indent=1)
    print(f"\n-> {CALIB}/depth_eval_official.json\n-> {OUT_VIS}/")


if __name__ == "__main__":
    main()
