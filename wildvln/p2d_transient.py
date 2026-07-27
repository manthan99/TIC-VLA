#!/usr/bin/env python3
"""P2d-2: transience filter — kills the operator-wall around the trajectory.

The semantic painter only sees the camera FOV; the operator walking BEHIND
the robot is never imaged, so their swept voxels stay unpainted and survive
as a phantom wall along the path (user-flagged in the 3D viewer).

The privileged map already stores the evidence: a static surface near the
path is re-hit by the 360-deg LiDAR for the whole time the robot is in
range, while a person moving with the robot occupies each voxel for only
~1-2 s. Flag as transient: dwell = last_seen - first_seen < DWELL_MAX_S,
AND within CORRIDOR_M of the segment trajectory (beyond it, brief sightings
can be legitimate — range edge, occlusion reveals), AND at person height
above the local ground raster (protects overhanging canopy briefly seen).

Updates the P2d sidecar in place: adds `transient` and `dynamic_all`
(= semantic dynamic | transient). Consumers should use dynamic_all.

Usage:
    python -m wildvln.p2d_transient
"""

from __future__ import annotations

from pathlib import Path

import os
import numpy as np
from scipy.spatial import cKDTree

P2A_ROOT = Path(os.environ.get("WILDVLN_P2A_ROOT", "/data/patelm/ticvla/wildvln/p2a"))
P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT", "/data/patelm/ticvla/wildvln/p2b"))
P2D_ROOT = Path(os.environ.get("WILDVLN_P2D_ROOT", "/data/patelm/ticvla/wildvln/p2d"))

DWELL_MAX_S = 2.5
CORRIDOR_M = 6.0
# vs local ground_z. Lower bound ABOVE the ground plane: ground cells are
# typically swept by a single beam-ring crossing (measured AU: median dwell
# 2.3 s, 52% under DWELL_MAX_S) so dwell alone cannot protect them
# (user-caught: "nothing survives on the ground plane").
HEIGHT_BAND = (0.3, 2.3)
# person evidence: short-dwell in-band voxels must form a vertical stack in
# their xy column (a walking person is a coherent column; isolated brief
# voxels are curb lips / foliage edges / range-edge ground).
COL_MIN_SPAN_M = 0.6
COL_MIN_VOX = 3


def process_segment(site, bag, seg_name):
    sem_p = P2D_ROOT / site / bag / f"{seg_name}_sem.npz"
    vz = np.load(P2A_ROOT / site / bag / seg_name / "voxels.npz")
    rz = np.load(P2A_ROOT / site / bag / seg_name / "rasters.npz")
    sm = dict(np.load(sem_p))

    coords, origin, voxel = vz["coords"], vz["origin"], float(vz["voxel"])
    dwell = vz["last_seen"] - vz["first_seen"]

    # world xy + z per voxel
    xy = (coords[:, :2].astype(np.float64) + origin[:2] + 0.5) * voxel
    z = (coords[:, 2].astype(np.float64) + origin[2] + 0.5) * voxel

    # corridor: distance to segment trajectory (keyframe poses of this seg)
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kmask = (idx["seg_id"] == int(seg_name[3:])) & idx["valid"]
    traj = idx["pose"][kmask][:, :2, 3]
    if len(traj) < 2:
        return None
    near = cKDTree(traj).query(xy, k=1)[0] < CORRIDOR_M

    # height above local ground raster
    gz = rz["ground_z"]
    hrel = z - gz[coords[:, 0], coords[:, 1]]
    person_band = (hrel > HEIGHT_BAND[0]) & (hrel < HEIGHT_BAND[1])

    cand = (dwell < DWELL_MAX_S) & near & person_band
    # column vertical-extent evidence
    transient = np.zeros_like(cand)
    ci = np.where(cand)[0]
    if len(ci):
        key = (coords[ci, 0].astype(np.int64) << 21) | coords[ci, 1]
        uniq, inv = np.unique(key, return_inverse=True)
        zc = coords[ci, 2].astype(np.int64)
        zmin = np.full(len(uniq), np.iinfo(np.int64).max)
        zmax = np.full(len(uniq), np.iinfo(np.int64).min)
        cnt = np.zeros(len(uniq), np.int64)
        np.minimum.at(zmin, inv, zc)
        np.maximum.at(zmax, inv, zc)
        np.add.at(cnt, inv, 1)
        col_ok = ((zmax - zmin) * voxel >= COL_MIN_SPAN_M) & \
                 (cnt >= COL_MIN_VOX)
        transient[ci] = col_ok[inv]
    sm["transient"] = transient
    sm["dynamic_all"] = sm["dynamic"] | transient
    np.savez_compressed(sem_p, **sm)
    return (int(transient.sum()), int(sm["dynamic"].sum()),
            int(sm["dynamic_all"].sum()), len(transient))


def main() -> None:
    for site in sorted(P2D_ROOT.iterdir()):
        if not site.is_dir() or site.name.startswith("_"):
            continue
        for bag in sorted(site.iterdir()):
            for f in sorted(bag.glob("seg*_sem.npz")):
                seg = f.stem.replace("_sem", "")
                r = process_segment(site.name, bag.name, seg)
                if r:
                    tr, dy, al, n = r
                    print(f"{site.name}/{bag.name}/{seg}: transient {tr} "
                          f"(+{tr - (al - dy) if al else 0} overlap) "
                          f"sem-dyn {dy} -> total {al}/{n} "
                          f"({al/n:.1%})", flush=True)


if __name__ == "__main__":
    main()
