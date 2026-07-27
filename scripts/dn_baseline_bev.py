#!/usr/bin/env python3
"""Base TIC-VLA ckpt on DynaNav eval samples, action head plotted on
MDE-derived BEV maps.

Per sample: run TICVLA (VLM CoT + action expert) -> 30x(dx,dy) 3 s
waypoints; run Depth-Anything-V2 metric MDE on the current frame;
backproject (pinhole for Spot head cam, f-theta polynomial for the
Nova Carter hawk); least-squares ground-plane fit on the lower image
-> metric rescale from the KNOWN camera height (trajectory z; sim
floor at z=0) -> level the cloud -> BEV obstacle scatter with GT
(green) vs predicted (red) waypoints. Saves per-sample png + a
summary json with the per-scene MDE scale factors (the ground-plane
scaling investigation).

Run (tic-vla env, one GPU):
  source .env.training
  CUDA_VISIBLE_DEVICES=2 python scripts/dn_baseline_bev.py --n 40
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

DATA = Path("/data/patelm/ticvla/dataset/DynaNav")
OUT = Path("/data/patelm/ticvla/outputs/dn_baseline_bev")
MDE = "/data/patelm/ticvla/depth_models/dav2-metric-outdoor-large"

HALF_M, RES = 8.0, 0.08          # BEV extent / cell for plotting
OB_Z = (0.15, 2.0)               # obstacle band above fitted ground
CROP_DY = 60                     # fisheye nominal 1920x1200 -> rendered 1080


def cam_rays(params, W, H):
    """Unit-less rays (x,y,1-style z=depth param for pinhole; unit sphere
    for fisheye) + a mask of valid pixels, subsampled on a stride grid."""
    st = 6
    u, v = np.meshgrid(np.arange(0, W, st) + 0.5, np.arange(0, H, st) + 0.5)
    u, v = u.ravel(), v.ravel()
    if params["cameraModel"] == "pinhole":
        fx = W * params["cameraFocalLength"] / params["cameraAperture"][0]
        fy = H * params["cameraFocalLength"] / params["cameraAperture"][1]
        rays = np.stack([(u - W / 2) / fx, (v - H / 2) / fy,
                         np.ones_like(u)], 1)
        return u, v, rays, np.ones(len(u), bool), True
    # f-theta fisheye: theta(r_px) polynomial, nominal frame is taller
    cx, cy = params["cameraFisheyeOpticalCentre"]
    cy = cy - CROP_DY
    A = params["cameraFisheyePolynomial"]
    dx, dy = u - cx, v - cy
    r = np.hypot(dx, dy)
    theta = sum(a * r ** i for i, a in enumerate(A))
    ok = theta < np.radians(params["cameraFisheyeMaxFOV"] / 2)
    phi = np.arctan2(dy, dx)
    s = np.sin(theta)
    rays = np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)], 1)
    return u, v, rays, ok, False


def ground_fit(P):
    """Iterative LSQ plane on lower-cloud points. Returns (n_unit, d):
    plane n.p = d with n pointing toward the camera (up)."""
    sel = P[P[:, 1] > np.percentile(P[:, 1], 55)]      # image-down = +y
    for _ in range(3):
        c = sel.mean(0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        n = vt[2]
        if n[1] > 0:
            n = -n                                     # up has -y_cam
        d = np.abs(sel @ n - (c @ n))
        sel = sel[d < max(np.percentile(d, 70), 0.05)]
    return n, c @ n


def bev_plot(ax, P_r, gt, pred, title):
    m = (np.abs(P_r[:, 0]) < HALF_M) & (np.abs(P_r[:, 1]) < HALF_M)
    ob = P_r[m & (P_r[:, 2] > OB_Z[0]) & (P_r[:, 2] < OB_Z[1])]
    gr = P_r[m & (np.abs(P_r[:, 2]) < 0.08)]
    ax.scatter(-gr[:, 1], gr[:, 0], s=1, c="#d8d4cc", lw=0)
    ax.scatter(-ob[:, 1], ob[:, 0], s=2, c="#5b6470", lw=0)
    ax.plot(-gt[:, 1], gt[:, 0], "-", c="#2e9e57", lw=2.2, label="GT 3 s")
    ax.plot(-pred[:, 1], pred[:, 0], "-", c="#d43d3d", lw=2.2,
            label="TIC-VLA")
    ax.plot(0, 0, marker="^", ms=9, c="k")
    ax.set_xlim(-HALF_M, HALF_M), ax.set_ylim(-2, HALF_M)
    ax.set_aspect("equal"), ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from transformers import (AutoImageProcessor,
                              AutoModelForDepthEstimation)

    from ticvla.training.evaluate import TestConfig, TICVLATester

    cfg = TestConfig(data_dir=str(DATA / "DynaNav_json_eval"),
                     num_test_samples=args.n, save_plots=False)
    tester = TICVLATester(cfg)
    n_data = len(tester.dataset.samples)
    random.seed(args.seed)
    idxs = random.sample(range(n_data), args.n)

    dp = AutoImageProcessor.from_pretrained(MDE)
    dm = AutoModelForDepthEstimation.from_pretrained(
        MDE, torch_dtype=torch.float16).to("cuda").eval()

    rows = []
    for k, idx in enumerate(idxs):
        try:
            r = tester.test_model_inference(idx)
        except Exception as e:
            print(f"[{k}] idx {idx} inference failed: {e}", flush=True)
            continue
        sp = Path(r["sample_path"])
        sj = json.loads(sp.read_text())
        img_p = (sp.parent / sj["current"]["img"]).resolve()
        rec = img_p.parents[2].name                     # recording dir
        scene = rec.split("_")[0]
        cam_dir = img_p.parents[1]
        pj = sorted((cam_dir / "camera_params").glob("*.json"))[0]
        params = json.loads(pj.read_text())
        # camera height = trajectory z (sim floor at z=0)
        import csv
        with open(img_p.parents[2] / "trajectory.csv") as f:
            row1 = list(csv.reader(f))[1]
        cam_h = float(row1[3])

        im = Image.open(img_p).convert("RGB")
        W, H = im.size
        with torch.no_grad():
            ins = dp(images=im, return_tensors="pt").to("cuda")
            ins["pixel_values"] = ins["pixel_values"].half()
            depth = dm(**ins).predicted_depth
            depth = torch.nn.functional.interpolate(
                depth[None], (H, W), mode="bilinear")[0, 0].float().cpu().numpy()

        u, v, rays, ok, pin = cam_rays(params, W, H)
        d = depth[v.astype(int), u.astype(int)]
        ok &= np.isfinite(d) & (d > 0.3) & (d < 40)
        P = rays[ok] * (d[ok, None] if pin else d[ok, None])   # z- or range-scaled
        n, dist = ground_fit(P)
        scale = cam_h / max(abs(dist), 1e-3)
        P *= scale
        # level: robot frame x fwd, y left, z up
        up = n
        fwd = np.array([0, 0, 1.0]) - up * up[2]
        fwd /= np.linalg.norm(fwd)
        left = np.cross(up, fwd)
        P_r = np.stack([P @ fwd, P @ left, P @ up + cam_h], 1)

        gt = np.asarray(r["gt_waypoints"], dtype=float)[:, :2]
        pred = np.asarray(r["pred_waypoints"], dtype=float)[:, :2]

        fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=110,
                                gridspec_kw={"width_ratios": [1.5, 1]})
        axs[0].imshow(im), axs[0].axis("off")
        axs[0].set_title(sj.get("instruction_file", ""), fontsize=7)
        bev_plot(axs[1], P_r, gt, pred,
                 f"{rec}  ade {r.get('ade', 0) if 'ade' in r else np.linalg.norm(pred[:len(gt)] - gt[:len(pred)], axis=1).mean():.2f}  scale {scale:.2f}")
        axs[1].legend(fontsize=6)
        fig.tight_layout()
        fp = OUT / f"{scene}_{k:02d}_{sp.stem}.png"
        fig.savefig(fp), plt.close(fig)
        rows.append({"scene": scene, "rec": rec, "sample": str(sp),
                     "png": fp.name, "scale": float(scale),
                     "cam_h": cam_h, "pinhole": bool(pin),
                     "ade": float(np.linalg.norm(
                         pred[:min(len(gt), len(pred))]
                         - gt[:min(len(gt), len(pred))], axis=1).mean()),
                     "response": r.get("response", "")[:800]})
        print(f"[{k}] {rec} scale {scale:.2f} ade {rows[-1]['ade']:.2f}",
              flush=True)

    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row["scale"])
    summ = {s: {"n": len(v), "scale_median": float(np.median(v)),
                "scale_iqr": [float(np.percentile(v, 25)),
                              float(np.percentile(v, 75))]}
            for s, v in by_scene.items()}
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows, "mde_scale_by_scene": summ}, indent=1))
    print(json.dumps(summ, indent=1))
    print("DN_BASELINE_BEV_DONE")


if __name__ == "__main__":
    main()
