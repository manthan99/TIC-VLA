#!/usr/bin/env python3
"""GrandTour BEV splat caches, part 1: wavemap-raycast patch depth.

Per mission: pick splat keyframes at KF_GAP_M spacing along the path
(matches GND's ~0.7 m kf spacing, so K_HIST=40 spans ~28 m of history),
then for each splat kf raycast the mission wavemap from the camera
center through every 32-px patch center -> z-depth grid (Hd, Wd).
NaN where the ray exits the map / exceeds range without hitting an
occupied cell (unknown-space rays stay NaN — honest holes, no floor
guessing).

Output: /data/patelm/ticvla/grandtour/p2c/depth/<bag>.npz
  grid (F, Hd, Wd) fp16 z-depth, kf_i (F,) indices into index.npz
  arrays, t (F,) stamps.

Usage: python -m wildvln.gt_splatcache [--bench] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
WM = Path("/data/patelm/ticvla/grandtour/wavemap")
OUT = Path("/data/patelm/ticvla/grandtour/p2c/depth")

KF_GAP_M = 0.7
PATCH = 32
T0, T1, STEP = 0.6, 25.0, 0.12
OCC_THR = 0.1


def splat_kfs(idx):
    """Distance-spaced subset of valid kf indices."""
    valid = np.flatnonzero(idx["valid"])
    xy = idx["pose"][valid][:, :2, 3]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    picks, nxt = [], 0.0
    for k, sk in enumerate(s):
        if sk >= nxt:
            picks.append(valid[k])
            nxt = sk + KF_GAP_M
    return np.asarray(picks)


def patch_rays(rig):
    fx, fy, cx, cy = rig["fx"], rig["fy"], rig["cx"], rig["cy"]
    W, H = rig["width"], rig["height"]
    wd, hd = (W + PATCH - 1) // PATCH, (H + PATCH - 1) // PATCH
    u = np.minimum((np.arange(wd) + 0.5) * PATCH, W - 1)
    v = np.minimum((np.arange(hd) + 0.5) * PATCH, H - 1)
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu.ravel() - cx) / fx,
                     (vv.ravel() - cy) / fy,
                     np.ones(hd * wd)], 1)          # z-depth param
    return rays, hd, wd


def raycast(m, T_world_cam, rays_cam):
    import pywavemap as wave
    R, tr = T_world_cam[:3, :3], T_world_cam[:3, 3]
    dirs = rays_cam @ R.T
    depth = np.full(len(dirs), np.nan, np.float32)
    active = np.arange(len(dirs))
    interp, near = m.interpolate, wave.InterpolationMode.NEAREST
    t = T0
    while t < T1 and len(active):
        pts = (tr + dirs[active] * t).astype(np.float64)
        vals = np.fromiter((interp(p, near) for p in pts),
                           np.float32, count=len(active))
        hit = vals > OCC_THR
        depth[active[hit]] = t
        active = active[~hit]
        t += STEP
    return depth


def do_mission(bag):
    import pywavemap as wave
    dst = OUT / f"{bag}.npz"
    if dst.exists():
        return f"{bag}: cached"
    idx = np.load(P2B / bag / "index.npz")
    rig = json.loads((P2B / bag / "rig.json").read_text())
    m = wave.Map.load(str(WM / f"{bag}.wvmp"))
    T_cb = np.asarray(rig["T_cam_base"])
    T_bc = np.linalg.inv(T_cb)
    rays, hd, wd = patch_rays(rig)
    picks = splat_kfs(idx)
    grids = np.full((len(picks), hd, wd), np.nan, np.float16)
    for n, ki in enumerate(picks):
        T_wc = idx["pose"][ki] @ T_bc
        grids[n] = raycast(m, T_wc, rays).reshape(hd, wd)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, grid=grids, kf_i=picks,
                        t=idx["t"][picks])
    return f"{bag}: {len(picks)} kfs grid {hd}x{wd}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    bags = sorted(p.name for p in P2B.iterdir()
                  if (p / "index.npz").exists()
                  and (WM / f"{p.name}.wvmp").exists())
    if args.bench:
        import pywavemap as wave
        bag = bags[0]
        idx = np.load(P2B / bag / "index.npz")
        rig = json.loads((P2B / bag / "rig.json").read_text())
        m = wave.Map.load(str(WM / f"{bag}.wvmp"))
        rays, hd, wd = patch_rays(rig)
        picks = splat_kfs(idx)
        T_wc = idx["pose"][picks[len(picks) // 2]] @ np.linalg.inv(
            np.asarray(rig["T_cam_base"]))
        t = time.time()
        d = raycast(m, T_wc, rays)
        dt = time.time() - t
        print(f"{bag}: {hd}x{wd} rays, {dt:.2f} s/kf, "
              f"{len(picks)} kfs/mission -> ~{dt * len(picks) / 60:.1f} "
              f"min/mission; hit {np.isfinite(d).mean():.0%}, "
              f"median depth {np.nanmedian(d):.1f} m")
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(do_mission, bags):
            print(r, flush=True)
    print("GT_SPLATCACHE_DONE")


if __name__ == "__main__":
    main()
