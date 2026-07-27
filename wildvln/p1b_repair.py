#!/usr/bin/env python3
"""P1b: bridge KISS-ICP dropouts with twist dead-reckoning.

KISS-ICP occasionally loses the map in degenerate scenes (the wooded trail:
one ~20 s stall-then-jump per bag) while tracking cleanly everywhere else, and
the wheel twist is reliable throughout on every rig. So: find intervals where
the KISS frame speed disagrees with the interpolated twist speed, replace the
poses inside them by SE(2) dead-reckoning from the last good pose, and rigidly
re-anchor everything after the gap (position + heading) so the downstream
trajectory continues smoothly. Frames inside a bridge are flagged; P2a must
not feed their scans into the voxel map, and windows overlapping a bridge are
tiered down.

Reads p1/<site>/<bag>/{poses,twist}.npz, writes poses_repaired.npz with:
    poses  (N,4,4)   repaired
    t      (N,)
    valid  (N,) bool  False inside bridged intervals

Usage:
    python -m wildvln.p1b_repair            # all bags under p1/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")

MISMATCH_MS = 1.0       # |v_kiss - v_twist| beyond this is suspect
MIN_RUN = 3             # consecutive suspect frames to open a bridge
MARGIN = 5              # frames of margin absorbed into the bridge each side


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def dead_reckon(t_grid, twist, t0, x0, y0, yaw0):
    """SE(2) integration of body twist from (x0, y0, yaw0) at t0 over t_grid."""
    tt, vx, vy, wz = twist[:, 0], twist[:, 1], twist[:, 2], twist[:, 3]
    out = []
    x, y, yaw, t_prev = x0, y0, yaw0, t0
    for t in t_grid:
        # integrate at twist rate between t_prev and t
        sel = (tt > t_prev) & (tt <= t)
        ts = np.concatenate([[t_prev], tt[sel], [t]])
        vs = np.interp(ts, tt, vx)
        us = np.interp(ts, tt, vy)
        ws = np.interp(ts, tt, wz)
        for i in range(len(ts) - 1):
            dt = ts[i + 1] - ts[i]
            yaw += ws[i] * dt
            x += (vs[i] * np.cos(yaw) - us[i] * np.sin(yaw)) * dt
            y += (vs[i] * np.sin(yaw) + us[i] * np.cos(yaw)) * dt
        out.append((x, y, yaw))
        t_prev = t
    return out


def repair_bag(bag_dir: Path) -> dict:
    z = np.load(bag_dir / "poses.npz")
    poses, t = z["poses"].copy(), z["t"]
    twist = np.load(bag_dir / "twist.npz")["data"]
    qc = {"bag": bag_dir.name, "site": bag_dir.parent.name,
          "n": len(poses), "bridges": [], "ok": True}
    if len(poses) < 10 or len(twist) < 10:
        np.savez_compressed(bag_dir / "poses_repaired.npz",
                            poses=poses, t=t, valid=np.ones(len(poses), bool))
        return qc

    xy = poses[:, :2, 3]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.clip(np.diff(t), 1e-6, None)
    v_kiss = d / dt
    v_tw = np.interp(t[1:], twist[:, 0], np.hypot(twist[:, 1], twist[:, 2]))
    suspect = np.abs(v_kiss - v_tw) > MISMATCH_MS

    # suspect runs -> bridged intervals with margin
    valid = np.ones(len(poses), bool)
    runs = []
    i = 0
    while i < len(suspect):
        if suspect[i]:
            j = i
            while j < len(suspect) and suspect[j]:
                j += 1
            if j - i >= MIN_RUN:
                runs.append((max(0, i - MARGIN), min(len(poses) - 1, j + MARGIN)))
            i = j
        else:
            i += 1
    # merge overlapping runs
    merged = []
    for a, b in runs:
        if merged and a <= merged[-1][1] + MARGIN:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    for a, b in merged:
        if a == 0:
            # bag starts broken: anchor at first frame after the gap instead
            valid[:b + 1] = False
            qc["bridges"].append({"frames": [int(a), int(b)], "mode": "head-cut"})
            continue
        anchor = poses[a - 1]
        x0, y0 = anchor[0, 3], anchor[1, 3]
        yaw0 = yaw_of(anchor[:3, :3])
        grid = t[a:b + 1]
        dr = dead_reckon(grid, twist, t[a - 1], x0, y0, yaw0)
        for k, (x, y, yaw) in zip(range(a, b + 1), dr):
            P = np.eye(4)
            P[:2, :2] = rot2(yaw)
            P[0, 3], P[1, 3] = x, y
            P[2, 3] = poses[a - 1][2, 3]
            poses[k] = P
        valid[a:b + 1] = False

        # re-anchor everything after the gap: KISS relative motion is good
        # again post-recovery, but its absolute frame jumped.
        if b + 1 < len(poses):
            old_end = z["poses"][b]
            new_end = poses[b]
            dyaw = yaw_of(new_end[:3, :3]) - yaw_of(old_end[:3, :3])
            R = np.eye(4)
            R[:2, :2] = rot2(dyaw)
            offset = new_end @ np.linalg.inv(R @ old_end)
            # apply planar-consistent correction to the remaining segment
            for k in range(b + 1, len(poses)):
                poses[k] = offset @ R @ z["poses"][k]
        qc["bridges"].append({"frames": [int(a), int(b)],
                              "seconds": round(float(t[b] - t[a]), 1),
                              "mode": "twist-bridge"})

    # post-repair QC
    xy = poses[:, :2, 3]
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    qc["path_m"] = round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1)
    qc["speed_max"] = round(float(v.max()), 2)
    qc["frac_over_limit"] = round(float((v > 3.0).mean()), 4)
    qc["frac_bridged"] = round(float((~valid).mean()), 4)
    qc["ok"] = qc["frac_over_limit"] < 0.02

    if not qc["ok"]:
        # Repair could not stabilize this bag (compounding gaps): degrade the
        # whole bag to twist dead-reckoning. Traces stay usable (tier C);
        # valid=False everywhere keeps every scan out of the voxel map.
        dr = dead_reckon(t, twist, float(twist[0, 0]), 0.0, 0.0, 0.0)
        for k, (x, y, yaw) in enumerate(dr):
            P = np.eye(4)
            P[:2, :2] = rot2(yaw)
            P[0, 3], P[1, 3] = x, y
            poses[k] = P
        valid = np.zeros(len(poses), bool)
        xy = poses[:, :2, 3]
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        qc["mode"] = "full-twist-fallback"
        qc["path_m"] = round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1)
        qc["speed_max"] = round(float(v.max()), 2)
        qc["frac_over_limit"] = round(float((v > 3.0).mean()), 4)
        qc["frac_bridged"] = 1.0
        qc["ok"] = qc["frac_over_limit"] < 0.02

    np.savez_compressed(bag_dir / "poses_repaired.npz", poses=poses, t=t, valid=valid)
    return qc


def main() -> None:
    results = []
    for site_dir in sorted(P1_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        for bag_dir in sorted(site_dir.iterdir()):
            if not (bag_dir / "poses.npz").exists():
                continue
            qc = repair_bag(bag_dir)
            results.append(qc)
            tag = "ok " if qc["ok"] else "BAD"
            nb = len(qc.get("bridges", []))
            print(f"[{tag}] {qc['site']:24s} {qc['bag']:44s} "
                  f"bridges {nb}  bridged {qc.get('frac_bridged', 0):.1%}  "
                  f"v_max {qc.get('speed_max', '-')}  over3 {qc.get('frac_over_limit', 0):.2%}",
                  flush=True)
    json.dump(results, open(P1_ROOT / "p1b_summary.json", "w"), indent=1)
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} bags ok after repair")


if __name__ == "__main__":
    main()
