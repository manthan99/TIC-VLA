#!/usr/bin/env python3
"""P2d: semantic voxel painting + dynamics deletion for the privileged maps.

For every valid keyframe, the segment's voxels within PAINT_RANGE are
projected into the image (official calibration). A voxel only takes a label
vote when the accumulated depth cache confirms it is actually visible there
(its camera depth agrees with the patch depth) — no painting through walls.
Votes are ADE-150 labels from the P2c semantic maps; majority wins.

Dynamics deletion: voxels whose votes are majority-dynamic (person/vehicle,
same id set as the depth cache) are flagged `dynamic`. The privileged map
consumers (beyond-FOV raycasts, collision labels, RL height scans) must
drop them; voxels.npz itself stays immutable — this writes a sidecar.

Output per segment: p2d/<site>/<bag>/segNN_sem.npz
    label (uint8, 255 = unpainted), votes (uint16), dyn_frac (f16),
    dynamic (bool)  — aligned 1:1 with voxels.npz coords.

Usage:
    python -m wildvln.p2d_voxelpaint [--workers 8] [--sites ...]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import os
import numpy as np

from wildvln.liftdepth import PATCH
from wildvln.p2b_depth_eval_official import project_official
from wildvln.p2c_depthcache import DYN_IDS
from wildvln.rigs import rig_for_site

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
P2A_ROOT = Path(os.environ.get("WILDVLN_P2A_ROOT", "/data/patelm/ticvla/wildvln/p2a"))
P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT", "/data/patelm/ticvla/wildvln/p2b"))
SEM_ROOT = Path("/data/patelm/ticvla/wildvln/p2c/sem")
DEPTH_ROOT = Path(os.environ.get("WILDVLN_DEPTH_ROOT", "/data/patelm/ticvla/wildvln/p2c/depth"))
OUT_ROOT = Path(os.environ.get("WILDVLN_OUT_ROOT", "/data/patelm/ticvla/wildvln/p2d"))

PAINT_RANGE = 25.0
DEPTH_TOL_REL = 0.15         # visible if |z_vox - z_patch| < max(rel*z, abs)
DEPTH_TOL_ABS = 0.45         # ~2 voxels + f16 rounding
MIN_VOTES_DYN = 3
DYN_FRAC_MIN = 0.5


def process_bag(site, bag):
    rig = rig_for_site(site)
    W_img, H_img = rig.image_size
    bag_dir = P2A_ROOT / site / bag
    seg_dirs = sorted(bag_dir.glob("seg*"))
    if not seg_dirs:
        return f"{site}/{bag}: no segments"
    dcache_p = DEPTH_ROOT / site / f"{bag}.npz"
    if not dcache_p.exists():
        return f"{site}/{bag}: SKIP no depth cache"

    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kt, kpose, kseg, kvalid = idx["t"], idx["pose"], idx["seg_id"], idx["valid"]
    dc = np.load(dcache_p)
    dgrid = dc["grid"]
    assert len(dc["t"]) == len(kt)

    out_dir = OUT_ROOT / site / bag
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for seg_dir in seg_dirs:
        seg = int(seg_dir.name[3:])
        dst = out_dir / f"seg{seg:02d}_sem.npz"
        if dst.exists():
            continue
        vz = np.load(seg_dir / "voxels.npz")
        world = (vz["coords"].astype(np.float64) + vz["origin"] + 0.5) * vz["voxel"]
        n_vox = len(world)

        vote_vox, vote_lab = [], []
        for ki in np.where((kseg == seg) & kvalid)[0]:
            sem_p = SEM_ROOT / site / bag / f"{int(kt[ki]*1e9)}.png"
            if not sem_p.exists() or np.all(np.isnan(dgrid[ki].astype(np.float32))):
                continue
            lab_img = cv2.imread(str(sem_p), cv2.IMREAD_GRAYSCALE)
            P = kpose[ki]
            pos = P[:3, 3]
            box = (np.abs(world[:, 0] - pos[0]) < PAINT_RANGE) & \
                  (np.abs(world[:, 1] - pos[1]) < PAINT_RANGE)
            vidx = np.where(box)[0]
            if not len(vidx):
                continue
            local = (world[vidx] - pos) @ P[:3, :3]      # R^T (x - t)
            u, v, z, ok = project_official(local, rig)
            vidx, u, v, z = vidx[ok], u[ok].astype(int), v[ok].astype(int), z[ok]
            zp = dgrid[ki][v // PATCH, u // PATCH].astype(np.float32)
            vis = ~np.isnan(zp) & (np.abs(z - zp) <
                                   np.maximum(DEPTH_TOL_REL * zp, DEPTH_TOL_ABS))
            vote_vox.append(vidx[vis])
            vote_lab.append(lab_img[v[vis], u[vis]])

        label = np.full(n_vox, 255, np.uint8)
        votes = np.zeros(n_vox, np.uint16)
        dyn_frac = np.zeros(n_vox, np.float16)
        if vote_vox:
            vv = np.concatenate(vote_vox)
            vl = np.concatenate(vote_lab).astype(np.int64)
            pair = vv.astype(np.int64) * 256 + vl
            ps, counts = np.unique(pair, return_counts=True)
            pvox, plab = ps // 256, (ps % 256).astype(np.uint8)
            # per voxel: total votes, argmax label, dynamic fraction
            order = np.argsort(pvox, kind="stable")
            pvox, plab, counts = pvox[order], plab[order], counts[order]
            uvox, starts = np.unique(pvox, return_index=True)
            ends = np.append(starts[1:], len(pvox))
            tot = np.add.reduceat(counts, starts)
            votes[uvox] = np.minimum(tot, 65535).astype(np.uint16)
            isdyn = np.isin(plab, DYN_IDS)
            dynv = np.zeros(len(pvox) + 1, np.int64)
            np.cumsum(counts * isdyn, out=dynv[1:])
            dyn_frac[uvox] = ((dynv[ends] - dynv[starts]) / tot).astype(np.float16)
            for i, (a, b) in enumerate(zip(starts, ends)):
                label[uvox[i]] = plab[a + np.argmax(counts[a:b])]
        dynamic = (votes >= MIN_VOTES_DYN) & \
                  (dyn_frac.astype(np.float32) > DYN_FRAC_MIN)
        np.savez_compressed(dst, label=label, votes=votes,
                            dyn_frac=dyn_frac, dynamic=dynamic)
        painted = float((label != 255).mean())
        lines.append(f"seg{seg:02d} painted {painted:.0%} "
                     f"dyn {int(dynamic.sum())}/{n_vox}")
    return f"{site}/{bag}: " + ("; ".join(lines) if lines else "cached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sites", type=str, default="")
    args = ap.parse_args()

    jobs = []
    for site in sorted(P2A_ROOT.iterdir()):
        if not site.is_dir() or site.name.endswith(".json"):
            continue
        if args.sites and site.name not in args.sites.split(","):
            continue
        for bag in sorted(site.iterdir()):
            if bag.is_dir():
                jobs.append((site.name, bag.name))
    print(f"{len(jobs)} bags", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_bag, s, b) for s, b in jobs]
        for fu in as_completed(futs):
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main()
