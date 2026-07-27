#!/usr/bin/env python3
"""P3-1: cut windows and derive distance-parameterized traces + history.

Window anchors sit on the 2 Hz keyframe grid, every ANCHOR_STRIDE_S seconds,
strictly inside one trajectory segment (P1c) — a window never spans a chop.
Per anchor, all in the anchor's local frame (x forward, y left):

    trace_bev (10,2)  future path resampled at equal arc length over up to
                      TRACE_BUDGET_M; plus the realized length
    hist_bev  (10,2)  past path, same format, clamped at the segment start

Anchors with less than MIN_FUTURE_M of remaining in-segment travel are
dropped (stationary robot or segment tail).

Output: p3/windows.parquet (one row per window, trace/hist as flat columns)
        p3/qc_traces_<site>.png (traces over the segment raster maps)

Usage:
    python -m wildvln.p3_windows
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
P2A_ROOT = Path("/data/patelm/ticvla/wildvln/p2a")
P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p3")

ANCHOR_STRIDE_S = 5.0
TRACE_POINTS = 10
TRACE_BUDGET_M = 10.0
HIST_BUDGET_M = 10.0
MIN_FUTURE_M = 2.0


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def resample_by_arclength(path_xy: np.ndarray, n: int, budget: float):
    """First `budget` metres of a polyline -> n equally spaced points."""
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = min(float(s[-1]), budget)
    if total < 1e-6:
        return None, 0.0
    targets = np.linspace(total / n, total, n)
    out = np.stack([np.interp(targets, s, path_xy[:, 0]),
                    np.interp(targets, s, path_xy[:, 1])], 1)
    return out, total


def windows_for_bag(site: str, bag: str) -> list:
    z = np.load(P1_ROOT / site / bag / "poses_repaired.npz")
    poses, t, seg_id = z["poses"], z["t"], z["seg_id"]
    idx_path = P2B_ROOT / site / bag / "index.npz"
    if not idx_path.exists():
        return []
    kf = np.load(idx_path)
    kt, kseg = kf["t"], kf["seg_id"]

    rows = []
    xy = poses[:, :2, 3]
    last_anchor = -1e18
    for ki in range(len(kt)):
        ta = kt[ki]
        if ta - last_anchor < ANCHOR_STRIDE_S or kseg[ki] < 0:
            continue
        pi = int(np.argmin(np.abs(t - ta)))
        seg = seg_id[pi]
        if seg < 0 or seg != kseg[ki]:
            continue
        in_seg = np.where(seg_id == seg)[0]
        i0, i1 = in_seg[0], in_seg[-1]

        A = poses[pi]
        ca, sa = np.cos(-yaw_of(A[:3, :3])), np.sin(-yaw_of(A[:3, :3]))
        R = np.array([[ca, -sa], [sa, ca]])

        fut = (xy[pi:i1 + 1] - A[:2, 3]) @ R.T
        trace, tr_len = resample_by_arclength(fut, TRACE_POINTS, TRACE_BUDGET_M)
        if trace is None or tr_len < MIN_FUTURE_M:
            continue
        past = (xy[pi:i0:-1] - A[:2, 3]) @ R.T if pi > i0 else np.zeros((1, 2))
        hist, h_len = resample_by_arclength(past, TRACE_POINTS, HIST_BUDGET_M)
        if hist is None:
            hist, h_len = np.zeros((TRACE_POINTS, 2)), 0.0

        last_anchor = ta
        row = {"site": site, "bag": bag, "seg": int(seg), "t": float(ta),
               "kf": f"{int(ta*1e9)}.jpg",
               "trace_len_m": round(tr_len, 2), "hist_len_m": round(h_len, 2),
               "lateral_max_m": round(float(np.abs(trace[:, 1]).max()), 2)}
        for j in range(TRACE_POINTS):
            row[f"tx{j}"], row[f"ty{j}"] = round(float(trace[j, 0]), 3), round(float(trace[j, 1]), 3)
            row[f"hx{j}"], row[f"hy{j}"] = round(float(hist[j, 0]), 3), round(float(hist[j, 1]), 3)
        rows.append(row)
    return rows


def qc_render(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for site in df["site"].unique():
        sub = df[df["site"] == site]
        # largest segment of the first bag with most windows
        bag = sub.groupby("bag").size().idxmax()
        ss = sub[sub["bag"] == bag]
        seg = ss.groupby("seg").size().idxmax()
        ss = ss[ss["seg"] == seg]
        rast_p = P2A_ROOT / site / bag / f"seg{seg:02d}" / "rasters.npz"
        z = np.load(P1_ROOT / site / bag / "poses_repaired.npz")
        poses, pt, seg_id = z["poses"], z["t"], z["seg_id"]

        fig, ax = plt.subplots(figsize=(12, 9))
        if rast_p.exists():
            r = np.load(rast_p)
            h = np.clip((r["top_z"] - r["ground_z"]).T, 0, 3)
            ox, oy = r["origin_xy"]
            vox = float(r["voxel"])
            nx, ny = r["ground_z"].shape
            ax.imshow(h, origin="lower", cmap="Greys",
                      extent=[ox * vox, (ox + nx) * vox, oy * vox, (oy + ny) * vox],
                      alpha=0.8)
        for _, row in ss.iterrows():
            pi = int(np.argmin(np.abs(pt - row["t"])))
            A = poses[pi]
            yaw = np.arctan2(A[1, 0], A[0, 0])
            Rw = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            tr = np.array([[row[f"tx{j}"], row[f"ty{j}"]] for j in range(TRACE_POINTS)])
            w = tr @ Rw.T + A[:2, 3]
            ax.plot(w[:, 0], w[:, 1], "-", lw=1.4, alpha=0.85)
            ax.plot(w[-1, 0], w[-1, 1], ".", ms=4, color="red")
        ax.set_title(f"{site} / {bag} seg{seg:02d} — every window's 10 m trace GT "
                     f"over the LiDAR map ({len(ss)} windows)")
        ax.axis("equal")
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_ROOT / f"qc_traces_{site}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    manifest = json.load(open("/data/patelm/ticvla/wildvln/p0/manifest.json"))
    all_rows = []
    for rec in manifest:
        if not rec["ok"]:
            continue
        rows = windows_for_bag(rec["site"], Path(rec["path"]).stem)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_ROOT / "windows.parquet")

    print(f"{len(df)} windows across {df['site'].nunique()} sites")
    for site, sub in df.groupby("site"):
        print(f"  {site:24s} {len(sub):5d} windows  "
              f"trace len p50={sub['trace_len_m'].median():.1f} m  "
              f"full-budget={(sub['trace_len_m'] >= TRACE_BUDGET_M - 0.05).mean():.0%}  "
              f"lateral>2.5m={(sub['lateral_max_m'] > 2.5).mean():.0%}")
    qc_render(df)
    print(f"-> {OUT_ROOT}")


if __name__ == "__main__":
    main()
