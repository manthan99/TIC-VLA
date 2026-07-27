#!/usr/bin/env python3
"""P2d QC: before/after BEV renders of dynamics deletion.

Three panels per segment: obstacle height as built (P2a), the voxels flagged
dynamic in red, and the privileged map after deletion. Renders the N
segments with the most dynamic voxels for a site (dininghall is the
stress test: lunchtime foot traffic).

Usage:
    python -m wildvln.p2d_qc --site UMD_map2_1_dininghall --top 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

P2A_ROOT = Path("/data/patelm/ticvla/wildvln/p2a")
P2D_ROOT = Path("/data/patelm/ticvla/wildvln/p2d")


def render(site, bag, seg_name, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vz = np.load(P2A_ROOT / site / bag / seg_name / "voxels.npz")
    sm = np.load(P2D_ROOT / site / bag / f"{seg_name}_sem.npz")
    coords, voxel = vz["coords"], float(vz["voxel"])
    origin = vz["origin"]
    dyn = sm["dynamic"]

    nx = int(coords[:, 0].max()) + 1
    ny = int(coords[:, 1].max()) + 1

    def height_map(mask):
        col = coords[mask, 0].astype(np.int64) * ny + coords[mask, 1]
        zc = (coords[mask, 2].astype(np.float32) + origin[2] + 0.5) * voxel
        order = np.argsort(col, kind="stable")
        col, zc = col[order], zc[order]
        cu, starts = np.unique(col, return_index=True)
        g = np.full(nx * ny, np.nan, np.float32)
        t = np.full(nx * ny, np.nan, np.float32)
        g[cu] = np.minimum.reduceat(zc, starts)
        t[cu] = np.maximum.reduceat(zc, starts)
        return (t - g).reshape(nx, ny)

    all_mask = np.ones(len(coords), bool)
    ext = [origin[0] * voxel, (origin[0] + nx) * voxel,
           origin[1] * voxel, (origin[1] + ny) * voxel]
    panels = [
        ("privileged map (as built)", height_map(all_mask), None),
        (f"dynamic voxels flagged ({int(dyn.sum())})",
         height_map(all_mask), height_map(dyn) if dyn.any() else None),
        ("after dynamics deletion", height_map(~dyn), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(4.5, 6 * ny / nx)),
                             sharex=True, sharey=True)
    for ax, (title, hm, overlay) in zip(axes, panels):
        ax.imshow(np.clip(hm.T, 0, 3.0), origin="lower", cmap="viridis",
                  extent=ext, interpolation="nearest")
        if overlay is not None:
            red = np.zeros(overlay.T.shape + (4,))
            red[..., 0] = 1.0
            red[..., 3] = np.where(np.isnan(overlay.T), 0.0, 1.0)
            ax.imshow(red, origin="lower", extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{site}/{bag}/{seg_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="UMD_map2_1_dininghall")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()

    ranked = []
    for bag_dir in sorted((P2D_ROOT / args.site).iterdir()):
        for f in sorted(bag_dir.glob("seg*_sem.npz")):
            n = int(np.load(f)["dynamic"].sum())
            ranked.append((n, bag_dir.name, f.stem.replace("_sem", "")))
    ranked.sort(reverse=True)
    out = P2D_ROOT / "_qc"
    out.mkdir(exist_ok=True)
    for n, bag, seg in ranked[:args.top]:
        png = out / f"{args.site}_{bag}_{seg}.png"
        render(args.site, bag, seg, png)
        print(f"{bag}/{seg}: {n} dynamic voxels -> {png}", flush=True)


if __name__ == "__main__":
    main()
