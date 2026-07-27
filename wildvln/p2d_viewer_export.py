#!/usr/bin/env python3
"""Export per-site payloads for the 3D voxel viewer artifact.

One JSON per site: a small manifest + one gzip'd base64 blob holding every
segment's buffers back to back (pos16 u16x3, group u8, traj f32x3). The
viewer decompresses with DecompressionStream and slices by the manifest.

Static voxels are subsampled to MAX_STATIC per segment; every voxel flagged
dynamic/transient survives (they are what the viewer exists to verify).
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

P2A_ROOT = Path(os.environ.get("WILDVLN_P2A_ROOT", "/data/patelm/ticvla/wildvln/p2a"))
P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT", "/data/patelm/ticvla/wildvln/p2b"))
P2D_ROOT = Path(os.environ.get("WILDVLN_P2D_ROOT", "/data/patelm/ticvla/wildvln/p2d"))

MAX_STATIC = 200_000
GROUND = {3, 6, 9, 11, 13, 29, 46, 52, 91, 94}
VEG = {4, 17, 66, 72}
STRUCT = {0, 1, 8, 14, 25, 32, 38, 42, 43, 48, 53, 87}
DYNC = {12, 20, 76, 80, 83, 90, 102, 103, 116, 127}


def seg_buffers(site, bag, seg_name):
    vz = np.load(P2A_ROOT / site / bag / seg_name / "voxels.npz")
    sm = np.load(P2D_ROOT / site / bag / f"{seg_name}_sem.npz")
    idx = np.load(P2B_ROOT / site / bag / "index.npz")

    coords = vz["coords"].astype(np.int32)
    origin, voxel = vz["origin"], float(vz["voxel"])
    label, dyn, trans = sm["label"], sm["dynamic"], sm["transient"]

    group = np.full(len(label), 5, np.uint8)
    group[label == 255] = 0
    for g, s in ((1, GROUND), (2, VEG), (3, STRUCT), (4, DYNC)):
        group[np.isin(label, list(s))] = g
    group[dyn] = 4
    group[trans & ~dyn] = 6

    flagged = dyn | trans
    if (~flagged).sum() > MAX_STATIC:
        rng = np.random.default_rng(0)
        keep = flagged.copy()
        keep[rng.choice(np.where(~flagged)[0], MAX_STATIC, replace=False)] = True
        coords, group = coords[keep], group[keep]

    kmask = (idx["seg_id"] == int(seg_name[3:])) & idx["valid"]
    traj = ((idx["pose"][kmask][:, :3, 3] / voxel) - origin).astype(np.float32)

    assert coords.max() < 65535
    return (coords.astype(np.uint16).tobytes() + group.tobytes()
            + traj.tobytes(), len(coords), len(traj), voxel)


def export_site(site, out_json):
    segs, blob = [], b""
    for bag_dir in sorted((P2D_ROOT / site).iterdir()):
        if not bag_dir.is_dir():
            continue
        for f in sorted(bag_dir.glob("seg*_sem.npz")):
            seg = f.stem.replace("_sem", "")
            buf, n, ntraj, voxel = seg_buffers(site, bag_dir.name, seg)
            short = bag_dir.name.replace(f"{site}_", "")
            segs.append({"name": f"{short}/{seg}", "n": n, "ntraj": ntraj})
            blob += buf
    payload = {
        "site": site, "voxel": voxel, "segments": segs,
        "blob": base64.b64encode(gzip.compress(blob, 6)).decode(),
    }
    Path(out_json).write_text(json.dumps(payload))
    mb = Path(out_json).stat().st_size / 1e6
    print(f"{site}: {len(segs)} segments, {mb:.1f} MB json", flush=True)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    for site in sorted(p.name for p in P2D_ROOT.iterdir()
                       if p.is_dir() and not p.name.startswith("_")):
        export_site(site, out_dir / f"viewer_{site}.json")
