#!/usr/bin/env python3
"""P2b-2: fit camera<->LiDAR extrinsics per rig + metric-depth bake-off.

Extrinsic: 6 params (roll, pitch, yaw, tx, ty, tz) on top of the FLU->OpenCV
axis convention, fitted by making projected LiDAR depths agree with a dense
mono-depth map (per-frame scale absorbed, so the fit is driven by *where*
points land, not by the model's absolute scale). Clouds are motion-compensated
to the keyframe timestamp through the P1c poses before projection.

Bake-off: with the fitted extrinsic frozen, every candidate model is scored on
held-out frames: abs-rel after per-frame median scaling (what the pipeline
uses) and raw abs-rel without scaling (how metric the model really is).

Outputs under p2b/_calib/:
    <rig>.json            fitted extrinsic + residuals
    bakeoff.json          per-model, per-rig scores
    overlay_*.jpg         LiDAR-over-image renders before/after fit

Usage:
    python -m wildvln.p2b_calibrate
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

from wildvln.rigs import ZED2, UMD

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
CALIB = P2B_ROOT / "_calib"

RIG_SAMPLES = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD),
}
# LiDAR FLU -> camera OpenCV axis convention.
BASE = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)

MODELS = {
    "dav2-metric-outdoor-large": "/data/patelm/ticvla/depth_models/dav2-metric-outdoor-large",
    "depth-pro": "/data/patelm/ticvla/depth_models/depth-pro",
}
FIT_MODEL = "dav2-metric-outdoor-large"
N_FIT, N_EVAL = 35, 50


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_pairs(site, bag, rig, n, offset=0):
    """(image_path, cloud_in_kf_lidar_frame) pairs, motion compensated."""
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kt, kpose, kseg = idx["t"], idx["pose"], idx["seg_id"]
    cz = np.load(CALIB / f"{site}_clouds.npz")
    ct = cz["t"]
    z = np.load(P1_ROOT / site / bag / "poses_repaired.npz")
    pt, pp, pseg = z["t"], z["poses"], z["seg_id"]

    pairs = []
    step = max(1, len(ct) // (n + offset))
    for ci in range(offset * step, len(ct), step):
        t = ct[ci]
        ki = int(np.argmin(np.abs(kt - t)))
        if abs(kt[ki] - t) > 0.3:
            continue
        pi = int(np.argmin(np.abs(pt - t)))
        pj = int(np.argmin(np.abs(pt - kt[ki])))
        if pseg[pi] < 0 or pseg[pi] != pseg[pj] or kseg[ki] < 0:
            continue
        rel = np.linalg.inv(pp[pj]) @ pp[pi]     # cloud-time -> kf-time
        pts = cz[f"c{ci}"] @ rel[:3, :3].T + rel[:3, 3]
        img = P2B_ROOT / site / bag / "keyframes" / f"{int(kt[ki]*1e9)}.jpg"
        if img.exists():
            pairs.append((str(img), pts))
        if len(pairs) >= n:
            break
    return pairs


def project(pts, params, K, wh):
    r, p, y, tx, ty, tz = params
    Xc = (rot_rpy(r, p, y) @ BASE @ pts.T).T + np.array([tx, ty, tz])
    z = Xc[:, 2]
    ok = z > 0.5
    fx, fy, cx, cy = K
    u = fx * Xc[:, 0] / np.where(ok, z, 1) + cx
    v = fy * Xc[:, 1] / np.where(ok, z, 1) + cy
    W, H = wh
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    return u, v, z, ok


BOUND_ROT = np.radians(20.0)
BOUND_T = 0.6


def residual(params, samples, K, wh, ret_scales=False):
    """Robust log-depth residual with a coverage guard.

    One global scale across all frames (per-frame scale lets a degenerate
    collapse win), hard physical bounds, and a penalty for projecting fewer
    points into the image than the identity extrinsic does.
    """
    if np.any(np.abs(params[:3]) > BOUND_ROT) or np.any(np.abs(params[3:]) > BOUND_T):
        return (10.0, []) if ret_scales else 10.0
    ratios, per_frame = [], []
    n_in = n_ref = 0
    for depth_map, pts in samples:
        u, v, z, ok = project(pts, params, K, wh)
        n_in += int(ok.sum())
        n_ref += int(0.25 * len(pts))          # expect >=25% of cloud in a wide-FOV cam
        if ok.sum() < 200:
            per_frame.append(None)
            continue
        dm = depth_map[v[ok].astype(int), u[ok].astype(int)]
        good = dm > 0.1
        if good.sum() < 200:
            per_frame.append(None)
            continue
        per_frame.append((z[ok][good], dm[good]))
        ratios.append(np.median(z[ok][good] / dm[good]))
    if not ratios:
        return (10.0, []) if ret_scales else 10.0
    s_glob = float(np.median(ratios))
    errs = []
    for pf in per_frame:
        if pf is None:
            errs.append(0.5)
            continue
        zz, dd = pf
        e = np.abs(np.log(zz) - np.log(s_glob * dd))
        errs.append(float(np.minimum(e, 0.5).mean()))
    coverage_pen = max(0.0, 1.0 - n_in / n_ref)
    cost = float(np.mean(errs)) + 1.0 * coverage_pen
    return (cost, [s_glob]) if ret_scales else cost


def run_model(name, path, images):
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model=path, device=0,
                    torch_dtype=torch.float16)
    out = {}
    for f in images:
        pred = pipe(Image.open(f))
        d = np.array(pred["predicted_depth"], dtype=np.float32)
        if d.shape != (Image.open(f).height, Image.open(f).width):
            d = cv2.resize(d, Image.open(f).size, interpolation=cv2.INTER_LINEAR)
        out[f] = d
    del pipe
    torch.cuda.empty_cache()
    return out


def overlay(img_path, pts, params, K, wh, out_path):
    img = cv2.imread(img_path)
    u, v, z, ok = project(pts, params, K, wh)
    zz = np.clip(z[ok], 1, 25)
    colors = (cv2.applyColorMap((255 * (1 - (zz - 1) / 24)).astype(np.uint8),
                                cv2.COLORMAP_TURBO).reshape(-1, 3))
    for (uu, vv), c in zip(np.c_[u[ok], v[ok]].astype(int), colors):
        cv2.circle(img, (uu, vv), 1, tuple(int(x) for x in c), -1)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    CALIB.mkdir(parents=True, exist_ok=True)
    results = {}
    for rig_name, (site, bag, rig) in RIG_SAMPLES.items():
        K = rig.intrinsics
        wh = rig.image_size
        fit_pairs = load_pairs(site, bag, rig, N_FIT)
        eval_pairs = load_pairs(site, bag, rig, N_EVAL, offset=N_FIT)
        print(f"{rig_name}: {len(fit_pairs)} fit / {len(eval_pairs)} eval pairs")

        depths = run_model(FIT_MODEL, MODELS[FIT_MODEL],
                           [f for f, _ in fit_pairs])
        samples = [(depths[f], pts) for f, pts in fit_pairs]

        # The LiDAR frame axis convention is unknown per rig (AU's Velodyne
        # publishes in navsat_link, which is not x-forward), so search the four
        # cardinal yaw pre-rotations and fine-fit on top of each.
        best = None
        best_card = 0
        for card in (0, 90, 180, 270):
            cy, sy = np.cos(np.radians(card)), np.sin(np.radians(card))
            Rcard = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            csamples = [(d, pts @ Rcard.T) for d, pts in samples]
            for p0 in np.radians([-5, 0, 5, 10]):
                for tz0 in (-0.3, 0.0, 0.3):
                    xx = np.zeros(6)
                    xx[1], xx[5] = p0, tz0
                    r = minimize(residual, xx, args=(csamples, K, wh),
                                 method="Nelder-Mead",
                                 options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-5})
                    if best is None or r.fun < best.fun:
                        best, best_card = r, card
        params = best.x
        cy, sy = np.cos(np.radians(best_card)), np.sin(np.radians(best_card))
        Rcard = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        samples = [(d, pts @ Rcard.T) for d, pts in samples]
        fit_pairs = [(f, pts @ Rcard.T) for f, pts in fit_pairs]
        eval_pairs = [(f, pts @ Rcard.T) for f, pts in eval_pairs]
        res0 = residual(np.zeros(6), samples, K, wh)
        res1, scales = residual(params, samples, K, wh, ret_scales=True)
        rig_out = {
            "cardinal_yaw_deg": best_card,
            "rpy_deg": [round(float(np.degrees(a)), 2) for a in params[:3]],
            "t_m": [round(float(a), 3) for a in params[3:]],
            "residual_before": round(float(res0), 4),
            "residual_after": round(float(res1), 4),
            "fit_model": FIT_MODEL,
            "median_scale_fitmodel": round(float(np.median(scales)), 3),
        }
        print(f"  fitted rpy={rig_out['rpy_deg']} t={rig_out['t_m']}  "
              f"residual {res0:.3f} -> {res1:.3f}")

        f0, p0 = fit_pairs[len(fit_pairs) // 2]
        overlay(f0, p0, np.zeros(6), K, wh, CALIB / f"overlay_{rig_name}_before.jpg")
        overlay(f0, p0, params, K, wh, CALIB / f"overlay_{rig_name}_after.jpg")

        # bake-off on held-out pairs
        rig_out["bakeoff"] = {}
        for name, path in MODELS.items():
            dm = run_model(name, path, [f for f, _ in eval_pairs])
            absrel_s, absrel_raw, scs = [], [], []
            for f, pts in eval_pairs:
                u, v, z, ok = project(pts, params, K, wh)
                if ok.sum() < 200:
                    continue
                d = dm[f][v[ok].astype(int), u[ok].astype(int)]
                good = d > 0.1
                zz, dd = z[ok][good], d[good]
                s = np.median(zz / dd)
                absrel_s.append(float(np.mean(np.abs(s * dd - zz) / zz)))
                absrel_raw.append(float(np.mean(np.abs(dd - zz) / zz)))
                scs.append(float(s))
            rig_out["bakeoff"][name] = {
                "absrel_scaled": round(float(np.median(absrel_s)), 4),
                "absrel_raw": round(float(np.median(absrel_raw)), 4),
                "median_scale": round(float(np.median(scs)), 3),
                "n_frames": len(absrel_s),
            }
            print(f"  {name:28s} absrel(scaled)={rig_out['bakeoff'][name]['absrel_scaled']:.3f} "
                  f"absrel(raw)={rig_out['bakeoff'][name]['absrel_raw']:.3f} "
                  f"scale={rig_out['bakeoff'][name]['median_scale']:.2f}")
        results[rig_name] = rig_out
        json.dump(rig_out, open(CALIB / f"{rig_name}.json", "w"), indent=1)

    json.dump(results, open(CALIB / "bakeoff.json", "w"), indent=1)
    print(f"\n-> {CALIB}")


if __name__ == "__main__":
    main()
