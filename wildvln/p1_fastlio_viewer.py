#!/usr/bin/env python3
"""Export FAST-LIO trial maps into the 3D viewer payload format.

Voxelizes /cloud_registered at 0.2 m (hits >= 2, same as the eval) and
packs each run as a viewer "segment" (group = 0 everywhere: no semantics on
these trial maps — use height coloring). Trajectory from /Odometry.

Usage: python -m wildvln.p1_fastlio_viewer <out.json> <name:runbag> ...
"""

from __future__ import annotations

import base64
import gzip
import json
import sys

import numpy as np

from wildvln.p1_fastlio_eval import VOXEL, load_run


def one(name, run_bag):
    ts, xyz, quat, keys, zs = load_run(run_bag)
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    uniq, starts = np.unique(ks, return_index=True)
    hits = np.diff(np.append(starts, len(ks)))
    uniq = uniq[hits >= 2]
    ijk = np.stack([((uniq >> 42) & ((1 << 21) - 1)) - (1 << 20),
                    ((uniq >> 21) & ((1 << 21) - 1)) - (1 << 20),
                    (uniq & ((1 << 21) - 1)) - (1 << 20)], 1).astype(np.int32)
    origin = ijk.min(0)
    coords = (ijk - origin).astype(np.uint16)
    group = np.zeros(len(coords), np.uint8)
    traj = (xyz / VOXEL - origin).astype(np.float32)
    traj = traj[:: max(1, len(traj) // 2000)]
    return (coords.tobytes() + group.tobytes() + traj.tobytes(),
            {"name": name, "n": len(coords), "ntraj": len(traj)})


def main() -> None:
    out_json = sys.argv[1]
    segs, blob = [], b""
    for arg in sys.argv[2:]:
        name, run_bag = arg.split(":", 1)
        buf, meta = one(name, run_bag)
        segs.append(meta)
        blob += buf
        print(f"{name}: {meta['n']} voxels", flush=True)
    payload = {"site": "FAST-LIO trial", "voxel": VOXEL, "segments": segs,
               "blob": base64.b64encode(gzip.compress(blob, 6)).decode()}
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
