#!/usr/bin/env python3
"""P2a: timestamped LiDAR voxel maps, one per trajectory segment (GT-only).

Second and last full read of each bag. Scans are transformed by the P1c
poses, quantized to VOXEL-m cells, and aggregated into a sparse voxel table
per segment (P1c chops trajectories at long KISS dropouts, so a bag can hold
several independent maps; frames with seg_id < 0 or valid=False contribute
nothing).

Per segment (out/<site>/<bag>/segNN/):
    voxels.npz   coords/origin/hits/first_seen/last_seen/z_mean/z_max
    rasters.npz  ground_z, top_z, density (per-column 2D)
    qc_bev.png   height-colored occupancy + segment trajectory

first_seen <= t is the causal filter; the full table is the privileged map.

Usage:
    python -m wildvln.p2a_voxelmap --workers 8 [--sites AU,...] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import os
import numpy as np

from wildvln.p1_odometry import cloud_to_xyz
from wildvln.rigs import rig_for_site

P1_ROOT = Path(os.environ.get("WILDVLN_P1_ROOT", "/data/patelm/ticvla/wildvln/p1"))
OUT_ROOT = Path(os.environ.get("WILDVLN_OUT_ROOT", "/data/patelm/ticvla/wildvln/p2a"))

VOXEL = 0.20
MAP_RANGE = 30.0
Z_MIN, Z_MAX = -2.0, 6.0
MIN_HITS = 2


def build_segment(seg_dir: Path, keys, ts, zs, traj_xy, title: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(keys, kind="stable")
    keys, ts, zs = keys[order], ts[order], zs[order]
    uniq, starts = np.unique(keys, return_index=True)
    hits = np.diff(np.append(starts, len(keys))).astype(np.int32)
    first_seen = np.minimum.reduceat(ts, starts)
    last_seen = np.maximum.reduceat(ts, starts)
    z_mean = (np.add.reduceat(zs.astype(np.float64), starts) / hits).astype(np.float32)
    z_max = np.maximum.reduceat(zs, starts).astype(np.float32)

    keep = hits >= MIN_HITS
    uniq, hits = uniq[keep], hits[keep]
    first_seen, last_seen = first_seen[keep], last_seen[keep]
    z_mean, z_max = z_mean[keep], z_max[keep]

    ix = ((uniq >> 42) & ((1 << 21) - 1)).astype(np.int32) - (1 << 20)
    iy = ((uniq >> 21) & ((1 << 21) - 1)).astype(np.int32) - (1 << 20)
    iz = (uniq & ((1 << 21) - 1)).astype(np.int32) - (1 << 20)
    origin = np.array([ix.min(), iy.min(), iz.min()])
    coords = np.stack([ix, iy, iz], 1) - origin

    seg_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seg_dir / "voxels.npz",
                        coords=coords.astype(np.int16), origin=origin, voxel=VOXEL,
                        hits=hits, first_seen=first_seen, last_seen=last_seen,
                        z_mean=z_mean, z_max=z_max)

    # 2D rasters
    nx = int(ix.max() - ix.min()) + 1
    ny = int(iy.max() - iy.min()) + 1
    col = coords[:, 0].astype(np.int64) * ny + coords[:, 1]
    corder = np.argsort(col, kind="stable")
    col_s = col[corder]
    zc = (iz[corder].astype(np.float32) + 0.5) * VOXEL
    cu, cstarts = np.unique(col_s, return_index=True)
    ground = np.full(nx * ny, np.nan, np.float32)
    top = np.full(nx * ny, np.nan, np.float32)
    dens = np.zeros(nx * ny, np.int32)
    ground[cu] = np.minimum.reduceat(zc, cstarts)
    top[cu] = np.maximum.reduceat(zc, cstarts)
    dens[cu] = np.add.reduceat(hits[corder], cstarts)
    ground, top = ground.reshape(nx, ny), top.reshape(nx, ny)
    np.savez_compressed(seg_dir / "rasters.npz", ground_z=ground, top_z=top,
                        density=dens.reshape(nx, ny), origin_xy=origin[:2],
                        voxel=VOXEL)

    # QC render
    fig, ax = plt.subplots(figsize=(11, max(4.0, 11 * ny / max(nx, 1))))
    ax.imshow(np.clip((top - ground).T, 0, 3.0), origin="lower", cmap="viridis",
              extent=[ix.min() * VOXEL, (ix.max() + 1) * VOXEL,
                      iy.min() * VOXEL, (iy.max() + 1) * VOXEL])
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="white", lw=1.2)
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="cyan", ms=8)
    ax.set_title(title, fontsize=11)
    fig.savefig(seg_dir / "qc_bev.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {"n_voxels": int(len(uniq)),
            "extent_m": [round(nx * VOXEL, 1), round(ny * VOXEL, 1)]}


def process_bag(site: str, bag_path: str, force: bool) -> dict:
    from rosbags.highlevel import AnyReader

    rig = rig_for_site(site)
    stem = Path(bag_path).stem
    p1_dir = P1_ROOT / site / stem
    out_dir = OUT_ROOT / site / stem
    if (out_dir / "meta.json").exists() and not force:
        return {"site": site, "bag": stem, "skipped": True}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(p1_dir / "poses_repaired.npz")
    poses, pose_t, valid = z["poses"], z["t"], z["valid"]
    seg_id = z["seg_id"] if "seg_id" in z.files else np.zeros(len(pose_t), np.int16)
    t0 = time.time()

    per_seg: dict = {}
    scan_i = 0
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == rig.cloud_topic]
        for conn, bag_ns, raw in reader.messages(connections=conns):
            t = bag_ns * 1e-9
            if scan_i >= len(pose_t):
                break
            if abs(t - pose_t[scan_i]) > 0.2 and t < pose_t[scan_i]:
                continue
            seg = int(seg_id[scan_i])
            if not valid[scan_i] or seg < 0:
                scan_i += 1
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            xyz = cloud_to_xyz(msg)
            rng = np.linalg.norm(xyz[:, :2], axis=1)
            keep = (rng > 0.5) & (rng < MAP_RANGE) & \
                   (xyz[:, 2] > Z_MIN) & (xyz[:, 2] < Z_MAX)
            pts = xyz[keep]
            P = poses[scan_i]
            world = pts @ P[:3, :3].T + P[:3, 3]
            ijk = np.floor(world / VOXEL).astype(np.int32)
            key = ((ijk[:, 0].astype(np.int64) + (1 << 20)) << 42) | \
                  ((ijk[:, 1].astype(np.int64) + (1 << 20)) << 21) | \
                  (ijk[:, 2].astype(np.int64) + (1 << 20))
            bucket = per_seg.setdefault(seg, ([], [], []))
            bucket[0].append(key)
            bucket[1].append(np.full(len(key), t))
            bucket[2].append(world[:, 2].astype(np.float32))
            scan_i += 1

    seg_metas = []
    for seg, (kl, tl, zl) in sorted(per_seg.items()):
        mask = seg_id == seg
        info = build_segment(
            out_dir / f"seg{seg:02d}",
            np.concatenate(kl), np.concatenate(tl), np.concatenate(zl),
            poses[mask][:, :2, 3],
            f"{site}/{stem} seg{seg:02d} — obstacle height over ground (m)")
        info["seg"] = seg
        seg_metas.append(info)

    meta = {"site": site, "bag": stem, "segments": seg_metas,
            "n_scans_used": int((valid & (seg_id >= 0)).sum()),
            "n_scans_skipped": int((~valid | (seg_id < 0)).sum()),
            "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sites", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open("/data/patelm/ticvla/wildvln/p0/manifest.json"))
    jobs = [(r["site"], r["path"]) for r in manifest if r["ok"]
            and (not args.sites or r["site"] in args.sites.split(","))]
    print(f"{len(jobs)} bags, {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_bag, s, p, args.force): (s, p)
                   for s, p in jobs}
        for fut in as_completed(futures):
            s, p = futures[fut]
            try:
                meta = fut.result()
            except Exception as exc:
                meta = {"site": s, "bag": Path(p).stem,
                        "error": f"{type(exc).__name__}: {exc}"}
            results.append(meta)
            if meta.get("skipped"):
                print(f"[skip] {s}/{meta['bag']}", flush=True)
            elif meta.get("error"):
                print(f"[ERR ] {s}/{meta['bag']}: {meta['error']}", flush=True)
            else:
                nv = sum(sm["n_voxels"] for sm in meta.get("segments", []))
                print(f"[ok ] {meta['site']:24s} {meta['bag']:44s} "
                      f"segs {len(meta.get('segments', []))}  {nv/1e6:5.2f}M vox  "
                      f"({meta.get('elapsed_s')}s)", flush=True)
    json.dump(results, open(OUT_ROOT / "p2a_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
