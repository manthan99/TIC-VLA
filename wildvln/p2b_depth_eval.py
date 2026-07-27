#!/usr/bin/env python3
"""P2b-3: fair depth-source eval under the semantic extrinsic.

Candidates per rig, all scored on held-out LiDAR points (50/50 split per
frame; fit split used for scaling, eval split for error):

    lidar-patch     patch-median of fit-split LiDAR (the mono-free option)
    dav2 / depth-pro / da3   mono depth, per-frame scale-only AND affine fit
                             in inverse-depth space against the fit split

Reported: absrel on eval points, coverage of the 16-px patch grid, and
absrel restricted to the upper image half (where the 16-beam LiDAR is blind
and mono is the only option).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m wildvln.p2b_depth_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from wildvln.p2b_calibrate import BASE, load_pairs, project, rot_rpy
from wildvln.rigs import UMD, ZED2

CALIB = Path("/data/patelm/ticvla/wildvln/p2b/_calib")
RIGS = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD),
}
N_FRAMES = 40
PATCH = 16


def load_extrinsic(rig_name):
    ext = json.load(open(CALIB / "extrinsics_sem.json"))[rig_name]
    card = np.radians(ext["cardinal_yaw_deg"])
    cy, sy = np.cos(card), np.sin(card)
    Rcard = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    params = np.concatenate([np.radians(ext["rpy_deg"]), ext["t_m"]])
    return Rcard, params


def mono_models():
    from transformers import pipeline

    def dav2(imgs):
        p = pipeline("depth-estimation",
                     model="/data/patelm/ticvla/depth_models/dav2-metric-outdoor-large",
                     device=0, dtype=torch.float16)
        out = {f: np.array(p(Image.open(f))["predicted_depth"], np.float32)
               for f in imgs}
        del p
        torch.cuda.empty_cache()
        return out

    def dpro(imgs):
        p = pipeline("depth-estimation",
                     model="/data/patelm/ticvla/depth_models/depth-pro",
                     device=0, dtype=torch.float16)
        out = {f: np.array(p(Image.open(f))["predicted_depth"], np.float32)
               for f in imgs}
        del p
        torch.cuda.empty_cache()
        return out

    def da3(imgs, K=None, wh=None):
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
        torch.cuda.empty_cache()
        return out

    return {"dav2-metric": dav2, "depth-pro": dpro, "da3-large": da3}


def affine_fit(z_fit, d_fit, d_eval):
    """Robust affine in inverse-depth: 1/z ~ a*(1/d) + b."""
    x, y = 1.0 / np.maximum(d_fit, 1e-3), 1.0 / np.maximum(z_fit, 1e-3)
    a, b = 1.0, 0.0
    for _ in range(3):
        pred = a * x + b
        r = np.abs(pred - y)
        w = r < np.percentile(r, 80)
        A = np.stack([x[w], np.ones(w.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, y[w], rcond=None)
        a, b = sol
    inv = a / np.maximum(d_eval, 1e-3) + b
    return 1.0 / np.maximum(inv, 1e-4)


def main() -> None:
    report = {}
    for rig_name, (site, bag, rig) in RIGS.items():
        K, wh = rig.intrinsics, rig.image_size
        W, H = wh
        Rcard, params = load_extrinsic(rig_name)
        pairs = [(f, pts @ Rcard.T) for f, pts in load_pairs(site, bag, rig, N_FRAMES)]
        imgs = [f for f, _ in pairs]

        # project all clouds once
        frames = []
        rng = np.random.default_rng(0)
        for f, pts in pairs:
            u, v, z, ok = project(pts, params, K, wh)
            u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
            if len(z) < 400:
                continue
            fit = rng.random(len(z)) < 0.5
            frames.append((f, u, v, z, fit))

        res = {}
        # --- lidar patch-median baseline ---
        errs, errs_hi, cov = [], [], []
        for f, u, v, z, fit in frames:
            pu, pv = u // PATCH, v // PATCH
            key = pu * 1000 + pv
            med = {}
            for k in np.unique(key[fit]):
                med[k] = np.median(z[fit][key[fit] == k])
            pred = np.array([med.get(k, np.nan) for k in key[~fit]])
            zz = z[~fit]
            good = ~np.isnan(pred)
            e = np.abs(pred[good] - zz[good]) / zz[good]
            errs.append(float(np.mean(e)))
            hi = good & (v[~fit] < H // 2)
            if hi.sum() > 20:
                errs_hi.append(float(np.mean(
                    np.abs(pred[hi] - zz[hi]) / zz[hi])))
            cov.append(len(med) / ((W // PATCH) * (H // PATCH)))
        res["lidar-patch"] = {"absrel": round(float(np.median(errs)), 3),
                              "absrel_upper": round(float(np.median(errs_hi)), 3) if errs_hi else None,
                              "patch_coverage": round(float(np.mean(cov)), 3)}

        # --- mono models ---
        for name, fn in mono_models().items():
            dm = fn(imgs, K=K, wh=wh) if name == "da3-large" else fn(imgs)
            for mode in ("scale", "affine"):
                errs, errs_hi = [], []
                for f, u, v, z, fit in frames:
                    if f not in dm:
                        continue
                    d = dm[f]
                    if d.shape != (H, W):
                        d = cv2.resize(d, (W, H))
                    dvals = d[v, u]
                    good = dvals > 1e-3
                    zf, df = z[fit & good], dvals[fit & good]
                    ze, de = z[~fit & good], dvals[~fit & good]
                    if len(zf) < 100 or len(ze) < 100:
                        continue
                    if mode == "scale":
                        pred = float(np.median(zf / df)) * de
                    else:
                        pred = affine_fit(zf, df, de)
                    e = np.abs(pred - ze) / ze
                    errs.append(float(np.mean(e)))
                    hi = (v[~fit & good] < H // 2)
                    if hi.sum() > 20:
                        errs_hi.append(float(np.mean(e[hi])))
                res[f"{name}/{mode}"] = {
                    "absrel": round(float(np.median(errs)), 3),
                    "absrel_upper": round(float(np.median(errs_hi)), 3) if errs_hi else None,
                    "patch_coverage": 1.0}
        report[rig_name] = res
        print(f"\n=== {rig_name} ===")
        print(f"{'source':24s} {'absrel':>7s} {'upper-half':>10s} {'coverage':>9s}")
        for k, r in res.items():
            print(f"{k:24s} {r['absrel']:7.3f} {str(r['absrel_upper']):>10s} "
                  f"{r['patch_coverage']:9.1%}")

    json.dump(report, open(CALIB / "depth_eval.json", "w"), indent=1)
    print(f"\n-> {CALIB}/depth_eval.json")


if __name__ == "__main__":
    main()
