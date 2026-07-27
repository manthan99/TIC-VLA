#!/usr/bin/env python3
"""P2c-3: accumulated LiDAR patch-depth cache for every keyframe.

The adopted depth source (user decision 2026-07-26): scans within
+-depth_accum_window_s (1 s both rigs), ego-motion compensated through the
P1c poses, z-buffer culled, mode-cluster patch estimator — all under the
official calibration in rigs.py.

Dynamics de-smearing: each scan is masked BEFORE accumulation by the
semantic map (p2c_semantics) of its OWN nearest keyframe — a moving person
is removed at every scan time, not just where they stand at the target
frame, so no streak survives. Points outside the camera FOV are kept (they
never project into the image anyway). Requires the semantic pass to be done
for the bag; bags with missing sem dirs are skipped with a warning.

Keyframes on twist-bridged poses (valid=False) fall back to single-scan
(MIN_PTS_SINGLE) — their relative poses are exactly the untrusted ones.

Output per bag: p2c/depth/<site>/<bag>.npz
    t (F,), grid (F, gh, gw) f16 (NaN = no depth), n_scans (F,), single (F,)

Usage:
    python -m wildvln.p2c_depthcache [--workers 8]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import os
import numpy as np

from wildvln.liftdepth import (MIN_PTS_ACCUM, MIN_PTS_SINGLE, PATCH,
                               ouster_solar_mask, patch_depth_grid,
                               zbuffer_cull)
from wildvln.p1_odometry import _DTYPES
from wildvln.p2b_depth_eval_official import project_official
from wildvln.rigs import rig_for_site

P0_MANIFEST = Path("/data/patelm/ticvla/wildvln/p0/manifest.json")
P1_ROOT = Path(os.environ.get("WILDVLN_P1_ROOT", "/data/patelm/ticvla/wildvln/p1"))
P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT", "/data/patelm/ticvla/wildvln/p2b"))
SEM_ROOT = Path("/data/patelm/ticvla/wildvln/p2c/sem")
OUT_ROOT = Path(os.environ.get("WILDVLN_OUT_ROOT", "/data/patelm/ticvla/wildvln/p2c/depth"))

DYN_IDS = np.array([12, 20, 76, 80, 83, 90, 102, 103, 116, 127], np.uint8)
# person, car, boat, bus, truck, airplane, van, ship, minibike, bicycle
# (verified against mask2former-ade id2label in p2c_semantics at run time;
#  animal=126 excluded: ADE 'animal' fires on statues constantly)
DILATE_PX = 8
KF_ASSOC_MAX_S = 0.3


def cloud_xyz_arr(msg):
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype in _DTYPES:
            names.append(f.name)
            formats.append(_DTYPES[f.datatype])
            offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    fin = np.isfinite(xyz).all(axis=1)
    return xyz[fin], arr[fin]


def process_bag(rec):
    site, bag_path = rec["site"], rec["path"]
    bag = Path(bag_path).stem
    rig = rig_for_site(site)
    W_img, H_img = rig.image_size
    win = rig.depth_accum_window_s

    dst = OUT_ROOT / site / f"{bag}.npz"
    if dst.exists():
        return f"{site}/{bag}: cached"
    sem_dir = SEM_ROOT / site / bag
    if not sem_dir.is_dir():
        return f"{site}/{bag}: SKIP no semantics"

    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kt, kseg, kvalid = idx["t"], idx["seg_id"], idx["valid"]
    pz = np.load(P1_ROOT / site / bag / "poses_repaired.npz")
    pt, pp, pseg = pz["t"], pz["poses"], pz["seg_id"]

    sem_cache = {}

    def dyn_mask_of(ki):
        if ki not in sem_cache:
            p = sem_dir / f"{int(kt[ki]*1e9)}.png"
            if not p.exists():
                sem_cache[ki] = None
            else:
                lab = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                m = np.isin(lab, DYN_IDS).astype(np.uint8)
                sem_cache[ki] = cv2.dilate(
                    m, np.ones((DILATE_PX, DILATE_PX), np.uint8)).astype(bool)
            if len(sem_cache) > 8:
                sem_cache.pop(min(k for k in sem_cache if k != ki))
        return sem_cache[ki]

    def mask_scan(ts, xyz):
        """Drop points on dynamic pixels of the scan's nearest keyframe."""
        ki = int(np.argmin(np.abs(kt - ts)))
        if abs(kt[ki] - ts) > KF_ASSOC_MAX_S or kseg[ki] < 0:
            return xyz
        dyn = dyn_mask_of(ki)
        if dyn is None or not dyn.any():
            return xyz
        pi = int(np.argmin(np.abs(pt - ts)))
        pj = int(np.argmin(np.abs(pt - kt[ki])))
        if pseg[pi] < 0 or pseg[pi] != pseg[pj]:
            return xyz
        rel = np.linalg.inv(pp[pj]) @ pp[pi]
        u, v, z, ok = project_official(
            xyz @ rel[:3, :3].T + rel[:3, 3], rig)
        drop = np.zeros(len(xyz), bool)
        drop[ok] = dyn[v[ok].astype(int), u[ok].astype(int)]
        return xyz[~drop]

    from rosbags.highlevel import AnyReader

    gh = (H_img + PATCH - 1) // PATCH
    gw = (W_img + PATCH - 1) // PATCH
    grids = np.full((len(kt), gh, gw), np.nan, np.float16)
    n_scans = np.zeros(len(kt), np.int16)
    single = np.zeros(len(kt), bool)

    buf = []                      # (ts, masked_xyz, pose_idx)
    next_kf = 0

    def flush_ready(horizon):
        nonlocal next_kf
        while next_kf < len(kt) and kt[next_kf] + win < horizon:
            build(next_kf)
            next_kf += 1

    def build(ki):
        if kseg[ki] < 0 or not buf:
            return
        tk = kt[ki]
        pj = int(np.argmin(np.abs(pt - tk)))
        if pseg[pj] != kseg[ki]:
            return
        inv_kf = np.linalg.inv(pp[pj])
        if kvalid[ki]:
            sel = [b for b in buf if abs(b[0] - tk) <= win
                   and pseg[b[2]] == kseg[ki]]
            min_pts = MIN_PTS_ACCUM
        else:                     # bridged pose: nearest scan only
            cand = [b for b in buf if pseg[b[2]] == kseg[ki]]
            if not cand:
                return
            sel = [min(cand, key=lambda b: abs(b[0] - tk))]
            min_pts = MIN_PTS_SINGLE
            single[ki] = True
        if not sel:
            return
        pts = np.concatenate([
            xyz @ (inv_kf @ pp[pi])[:3, :3].T + (inv_kf @ pp[pi])[:3, 3]
            for ts, xyz, pi in sel])
        u, v, z, ok = project_official(pts, rig)
        u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
        if len(z) < 50:
            return
        keep = zbuffer_cull(u, v, z, W_img, H_img)
        g, _ = patch_depth_grid(u[keep], v[keep], z[keep],
                                W_img, H_img, min_pts)
        grids[ki] = g.astype(np.float16)
        n_scans[ki] = len(sel)

    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == rig.cloud_topic]
        for conn, bag_ns, raw in reader.messages(connections=conns):
            ts = bag_ns * 1e-9
            msg = reader.deserialize(raw, conn.msgtype)
            xyz, arr = cloud_xyz_arr(msg)
            if len(xyz) < 100:
                continue
            xyz = xyz[ouster_solar_mask(arr)]
            pi = int(np.argmin(np.abs(pt - ts)))
            buf.append((ts, mask_scan(ts, xyz), pi))
            while buf and buf[0][0] < ts - 2 * win - 1.0:
                buf.pop(0)
            flush_ready(ts)
    flush_ready(np.inf)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, t=kt, grid=grids, n_scans=n_scans, single=single)
    done = int((n_scans > 0).sum())
    cov = float(np.mean(~np.isnan(grids[n_scans > 0].astype(np.float32)))) \
        if done else 0.0
    return (f"{site}/{bag}: {done}/{len(kt)} frames, "
            f"median scans {int(np.median(n_scans[n_scans>0])) if done else 0}, "
            f"cov {cov:.1%}, bridged-single {int(single.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = json.load(open(P0_MANIFEST))
    # tolerate partially-imported pose roots (e.g. p2bf while FAST-LIO
    # reruns are still pending for some recordings)
    recs = [r for r in recs
            if (P2B_ROOT / r["site"] / Path(r["path"]).stem /
                "index.npz").exists()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_bag, r): r for r in recs}
        for fu in as_completed(futs):
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main()
