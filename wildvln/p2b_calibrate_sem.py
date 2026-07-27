#!/usr/bin/env python3
"""P2b-2b: camera<->LiDAR extrinsics by LiDAR/semantic ground agreement.

The mono-depth objective inherited the depth models' biases (both rigs rode
the pitch bound). This one is model-bias-free: LiDAR points are classified
ground / obstacle geometrically (the sensor plate height above ground is
measured per rig), pixels are classified ground / not by Mask2Former-ADE, and
the extrinsic is fitted to make the two agree — ground points must project
onto ground pixels, obstacle points off them.

External validation: the ZED2 mount pitch is known from TF (2.86 deg down);
the fit never sees that number.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m wildvln.p2b_calibrate_sem
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

from wildvln.p2b_calibrate import BASE, load_pairs, project, rot_rpy
from wildvln.rigs import UMD, ZED2

CALIB = Path("/data/patelm/ticvla/wildvln/p2b/_calib")
SEG_MODEL = "/data/patelm/ticvla/depth_models/mask2former-ade"

RIGS = {
    "gnd-zed2": ("AU", "AU_chunk04", ZED2, 180),
    "gnd-umd": ("UMD_map1_2_lot9", "UMD_map1_2_lot9_chunk10", UMD, 0),
}
# ADE20K ids that count as traversable ground.
ADE_GROUND = {3, 6, 9, 11, 13, 29, 46, 52, 91, 94}
#  floor, road, grass, sidewalk, earth, field, sand, path, dirt track, land

GROUND_TOL = 0.15
OBST_MIN = 0.40
N_FIT = 30
BOUND_ROT = np.radians(12.0)
BOUND_T = 0.6


def ground_masks(images):
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        SEG_MODEL, dtype=torch.float16).to("cuda").eval()
    out = {}
    with torch.no_grad():
        for f in images:
            img = Image.open(f)
            inp = proc(images=img, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            res = model(**inp)
            sem = proc.post_process_semantic_segmentation(
                res, target_sizes=[img.size[::-1]])[0].cpu().numpy()
            out[f] = np.isin(sem, list(ADE_GROUND))
    del model
    torch.cuda.empty_cache()
    return out


def agreement(params, samples, K, wh):
    if np.any(np.abs(params[:3]) > BOUND_ROT) or np.any(np.abs(params[3:]) > BOUND_T):
        return 10.0
    scores = []
    for gmask, g_pts, o_pts in samples:
        got = 0.0
        for pts, want_ground, w in ((g_pts, True, 1.0), (o_pts, False, 1.0)):
            u, v, z, ok = project(pts, params, K, wh)
            if ok.sum() < 50:
                continue
            on_ground = gmask[v[ok].astype(int), u[ok].astype(int)]
            frac = on_ground.mean() if want_ground else (~on_ground).mean()
            got += w * float(frac)
        scores.append(got)
    return 2.0 - float(np.mean(scores)) if scores else 10.0


def main() -> None:
    results = {}
    for rig_name, (site, bag, rig, cardinal) in RIGS.items():
        K, wh = rig.intrinsics, rig.image_size
        pairs = load_pairs(site, bag, rig, N_FIT)
        cy, sy = np.cos(np.radians(cardinal)), np.sin(np.radians(cardinal))
        Rcard = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        gmasks = ground_masks([f for f, _ in pairs])

        samples = []
        plate = rig.lidar_height_m
        for f, pts in pairs:
            pts = pts @ Rcard.T
            gz = -plate
            g = pts[np.abs(pts[:, 2] - gz) < GROUND_TOL]
            o = pts[pts[:, 2] > gz + OBST_MIN]
            samples.append((gmasks[f], g, o))

        best = None
        for p0 in np.radians([-5, 0, 5, 10]):
            for tz0 in (-0.3, 0.0, 0.3):
                x0 = np.zeros(6)
                x0[1], x0[5] = p0, tz0
                r = minimize(agreement, x0, args=(samples, K, wh),
                             method="Nelder-Mead",
                             options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5})
                if best is None or r.fun < best.fun:
                    best = r
        params = best.x
        a0 = agreement(np.zeros(6), samples, K, wh)
        a1 = best.fun
        out = {"cardinal_yaw_deg": cardinal,
               "rpy_deg": [round(float(np.degrees(x)), 2) for x in params[:3]],
               "t_m": [round(float(x), 3) for x in params[3:]],
               "agreement_before": round(2 - a0, 3),
               "agreement_after": round(2 - a1, 3)}
        results[rig_name] = out
        print(f"{rig_name}: cardinal {cardinal}  rpy={out['rpy_deg']} "
              f"t={out['t_m']}  agreement {2-a0:.3f} -> {2-a1:.3f}  (max 2.0)")

        # overlay: ground pts green, obstacle red, on the middle frame
        f0, pts0 = pairs[len(pairs) // 2]
        pts0 = pts0 @ Rcard.T
        img = cv2.imread(f0)
        gz = -plate
        for sel, col in [(np.abs(pts0[:, 2] - gz) < GROUND_TOL, (80, 255, 80)),
                         (pts0[:, 2] > gz + OBST_MIN, (60, 60, 255))]:
            u, v, z, ok = project(pts0[sel], params, K, wh)
            for uu, vv in np.c_[u[ok], v[ok]].astype(int):
                cv2.circle(img, (uu, vv), 1, col, -1)
        gm = gmasks[f0]
        edge = cv2.Canny(gm.astype(np.uint8) * 255, 50, 150) > 0
        img[edge] = (255, 0, 255)
        cv2.imwrite(str(CALIB / f"overlay_sem_{rig_name}.jpg"), img)

    json.dump(results, open(CALIB / "extrinsics_sem.json", "w"), indent=1)
    print(f"-> {CALIB}/extrinsics_sem.json")


if __name__ == "__main__":
    main()
