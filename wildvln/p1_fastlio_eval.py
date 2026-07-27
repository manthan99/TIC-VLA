#!/usr/bin/env python3
"""Evaluate a FAST-LIO trial run against the KISS-ICP baseline.

Reads the /Odometry (+ /cloud_registered) bag recorded by the docker
harness, then reports the same map-quality metrics that exposed the KISS
problems (gravity tilt of the ground plane near the trajectory, robust
plane RMS, ground roughness), side by side with the KISS numbers for the
same bag. Also renders a BEV height map from the registered clouds.

Usage:
    python -m wildvln.p1_fastlio_eval /path/to/out.bag --site SITE --bag BAGSTEM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VOXEL = 0.20
Z_MIN, Z_MAX = -3.0, 7.0


def load_run(bag_path):
    from rosbags.highlevel import AnyReader
    ts, xyz, quat = [], [], []
    keys_all, zs_all = [], []
    with AnyReader([Path(bag_path)]) as r:
        conns = [c for c in r.connections if c.topic in ("/Odometry", "/cloud_registered")]
        for conn, ns, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            if conn.topic == "/Odometry":
                p, q = m.pose.pose.position, m.pose.pose.orientation
                ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                xyz.append([p.x, p.y, p.z])
                quat.append([q.x, q.y, q.z, q.w])
            else:
                names = [f.name for f in m.fields]
                fmts = {1:"i1",2:"u1",3:"i2",4:"u2",5:"i4",6:"u4",7:"f4",8:"f8"}
                dt = np.dtype({"names": names,
                               "formats": [fmts[f.datatype] for f in m.fields],
                               "offsets": [f.offset for f in m.fields],
                               "itemsize": m.point_step})
                a = np.frombuffer(m.data, dt)
                pts = np.stack([a["x"], a["y"], a["z"]], 1)
                pts = pts[np.isfinite(pts).all(1)]
                pts = pts[(pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)]
                ijk = np.floor(pts / VOXEL).astype(np.int64)
                keys_all.append(((ijk[:, 0] + (1 << 20)) << 42)
                                | ((ijk[:, 1] + (1 << 20)) << 21)
                                | (ijk[:, 2] + (1 << 20)))
                zs_all.append(pts[:, 2].astype(np.float32))
    return (np.array(ts), np.array(xyz), np.array(quat),
            np.concatenate(keys_all) if keys_all else np.zeros(0, np.int64),
            np.concatenate(zs_all) if zs_all else np.zeros(0, np.float32))


def map_metrics(traj_xy, keys, zs):
    from scipy.spatial import cKDTree
    order = np.argsort(keys, kind="stable")
    ks, z = keys[order], zs[order]
    uniq, starts = np.unique(ks, return_index=True)
    hits = np.diff(np.append(starts, len(ks)))
    keep = hits >= 2
    uniq = uniq[keep]
    ix = ((uniq >> 42) & ((1 << 21) - 1)) - (1 << 20)
    iy = ((uniq >> 21) & ((1 << 21) - 1)) - (1 << 20)
    iz = (uniq & ((1 << 21) - 1)) - (1 << 20)

    # per-column ground
    col = ix * (1 << 21) + iy
    corder = np.argsort(col, kind="stable")
    cs = col[corder]
    zc = (iz[corder] + 0.5) * VOXEL
    cu, cstarts = np.unique(cs, return_index=True)
    gz = np.minimum.reduceat(zc, cstarts)
    gx = (((cu >> 21) & ((1 << 21) - 1)).astype(np.float64) - 0)
    cx = ((cu // (1 << 21)) + 0.5) * VOXEL
    cy = ((cu % (1 << 21)) - 0)
    cy = ((cu - (cu // (1 << 21)) * (1 << 21)) + 0.5) * VOXEL

    xy = np.stack([cx, cy], 1)
    near = cKDTree(traj_xy).query(xy, k=1)[0] < 6.0
    if near.sum() < 200:
        return None
    X = np.c_[xy[near], np.ones(int(near.sum()))]
    zn = gz[near]
    w = np.ones(len(zn), bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(X[w], zn[w], rcond=None)
        r = X @ c - zn
        w = np.abs(r) < max(3 * np.median(np.abs(r[w])) + 1e-6, 0.05)
    tilt = np.degrees(np.arctan(np.hypot(c[0], c[1])))
    rms = float(np.sqrt(np.mean((X[w] @ c - zn[w]) ** 2)))
    return {"tilt_deg": round(float(tilt), 2), "plane_rms_m": round(rms, 2),
            "n_ground_cols": int(near.sum()),
            "n_voxels": int(len(uniq))}


def render(traj_xy, keys, zs, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(keys, kind="stable")
    ks, z = keys[order], zs[order]
    uniq, starts = np.unique(ks, return_index=True)
    hits = np.diff(np.append(starts, len(ks)))
    keep = hits >= 2
    uniq = uniq[keep]
    ix = (((uniq >> 42) & ((1 << 21) - 1)) - (1 << 20)).astype(np.int32)
    iy = (((uniq >> 21) & ((1 << 21) - 1)) - (1 << 20)).astype(np.int32)
    iz = ((uniq & ((1 << 21) - 1)) - (1 << 20)).astype(np.int32)
    nx = ix.max() - ix.min() + 1
    ny = iy.max() - iy.min() + 1
    col = (ix - ix.min()).astype(np.int64) * ny + (iy - iy.min())
    corder = np.argsort(col, kind="stable")
    cs, zc = col[corder], (iz[corder] + 0.5) * VOXEL
    cu, cstarts = np.unique(cs, return_index=True)
    g = np.full(nx * ny, np.nan, np.float32)
    t = np.full(nx * ny, np.nan, np.float32)
    g[cu] = np.minimum.reduceat(zc, cstarts)
    t[cu] = np.maximum.reduceat(zc, cstarts)
    hm = (t - g).reshape(nx, ny)
    fig, ax = plt.subplots(figsize=(12, max(4.0, 12 * ny / nx)))
    ax.imshow(np.clip(hm.T, 0, 3.0), origin="lower", cmap="viridis",
              extent=[ix.min() * VOXEL, (ix.max() + 1) * VOXEL,
                      iy.min() * VOXEL, (iy.max() + 1) * VOXEL])
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="white", lw=1.2)
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="cyan", ms=8)
    ax.set_title(title, fontsize=11)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_bag")
    ap.add_argument("--out", default="/data/patelm/ticvla/wildvln/p1_fastlio")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    ts, xyz, quat, keys, zs = load_run(args.run_bag)
    print(f"{args.name}: {len(ts)} odom poses, {len(keys)/1e6:.1f}M map points")
    if len(ts) < 50:
        print("TOO FEW POSES — check /tmp logs in container")
        return
    L = float(np.sum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)))
    dz = float(xyz[-1, 2] - xyz[0, 2])
    print(f"traj length {L:.0f} m, net z drift {dz:+.2f} m "
          f"({100*abs(dz)/max(L,1):.2f}% of length)")
    m = map_metrics(xyz[:, :2], keys, zs)
    print("map metrics:", m)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.name}_odom.npz",
                        t=ts, xyz=xyz, quat=quat)
    render(xyz[:, :2], keys, zs, out / f"{args.name}_bev.png",
           f"FAST-LIO — {args.name} (obstacle height over ground)")
    print(f"-> {out}/{args.name}_bev.png")


if __name__ == "__main__":
    main()
