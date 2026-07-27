#!/usr/bin/env python3
"""Top-down semantic BEV renders + collision analysis for eval dumps.

For each eval step: crop the privileged voxel map (p2af + p2df semantics,
dynamic_all removed) around the robot in the SAME motion-derived ego frame
as the trace GT, render ground semantics + obstacles top-down, overlay
history / GT / model predictions, and measure collisions (trace points
within ROBOT_R of an obstacle column at body height).

GT traces are scored too — if GT itself collides, the label pipeline (or
the map) has a problem, not the model.

Usage:
  python -m wildvln.p6_bev_sem --out DIR m0=dump.jsonl m3=... [--split ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from wildvln.p6_slice import net_heading, TURN_DEG

P2A = Path("/data/patelm/ticvla/wildvln/p2af")
P2B = Path("/data/patelm/ticvla/wildvln/p2bf")
P2D = Path("/data/patelm/ticvla/wildvln/p2df")

HALF_M = 12.5
RES = 0.1                      # render + collision grid
OBST_BAND = (0.35, 2.2)        # height above ground raster = body-level
ROBOT_R = 0.30
GROUND_MAX = 0.35

# ADE20K ids -> BGR ground colors (muted)
GROUND_COL = {
    6: (105, 105, 105),        # road
    11: (150, 150, 150),       # sidewalk
    3: (140, 140, 140),        # floor
    52: (120, 140, 150),       # path
    9: (90, 140, 90),          # grass
    13: (100, 125, 140),       # earth
    17: (85, 120, 85),         # plant
    46: (120, 150, 160),       # sand
    255: (70, 70, 78),         # unpainted
}
DEFAULT_GROUND = (95, 95, 105)
OBST_COL = (45, 45, 160)       # dark red-ish
COLORS = {"m0": (60, 60, 230), "m3": (200, 40, 200), "m4": (0, 165, 255),
          "grpo": (0, 220, 220), "kin": (150, 150, 150)}


class SemMap:
    """Static voxels of one bag: world xy, world z, label.

    Height-above-ground is computed LOCALLY per crop (anchored to the
    robot pose height) — the whole-bag ground raster is warped by slow
    FAST-LIO z-drift between passes (0.4-1.6 m at revisits), which is
    larger than the obstacle band."""

    def __init__(self, site, bag):
        from wildvln.rigs import rig_for_site
        self.rig = rig_for_site(site)
        idx = np.load(P2B / site / bag / "index.npz")
        self.kt, self.pose = idx["t"], idx["pose"]
        self.stamp2ki = {int(t * 1e9): i for i, t in enumerate(idx["t"])}
        xs, zs, ls, f0, f1 = [], [], [], [], []
        for segdir in sorted((P2A / site / bag).glob("seg*")):
            vz = np.load(segdir / "voxels.npz")
            sm = np.load(P2D / site / bag / f"{segdir.name}_sem.npz")
            coords, origin = vz["coords"], vz["origin"]
            voxel = float(vz["voxel"])
            keep = ~sm["dynamic_all"]
            xy = (coords[keep, :2].astype(np.float32)
                  + origin[:2] + 0.5) * voxel
            z = (coords[keep, 2].astype(np.float32) + origin[2] + 0.5) * voxel
            xs.append(xy)
            zs.append(z.astype(np.float32))
            ls.append(sm["label"][keep])
            f0.append(vz["first_seen"][keep])
            f1.append(vz["last_seen"][keep])
        self.xy = np.concatenate(xs)
        self.z = np.concatenate(zs)
        self.label = np.concatenate(ls)
        self.t0 = np.concatenate(f0)
        self.t1 = np.concatenate(f1)

        # carve the operator/self phantom tube: body-height voxels near
        # the driven trajectory with dwell too short for street furniture
        # (the p2d transient filter's 2.5 s cutoff misses an operator
        # moving WITH the robot; a bench/pole in a 2 m corridor is
        # re-hit by the 360-deg lidar for tens of seconds)
        from scipy.spatial import cKDTree
        traj = self.pose[:, :2, 3]
        d, j = cKDTree(traj).query(self.xy, k=1)
        zrel = self.z - (self.pose[j, 2, 3] - self.rig.lidar_height_m)
        phantom = ((self.t1 - self.t0 < 10.0) & (d < 2.0)
                   & (zrel > 0.25) & (zrel < 2.3))
        for a in ("xy", "z", "label", "t0", "t1"):
            setattr(self, a, getattr(self, a)[~phantom])

    def ego(self, ki):
        p = self.pose[ki][:3, 3]
        a = self.pose[max(ki - 1, 0)][:2, 3]
        b = self.pose[min(ki + 1, len(self.pose) - 1)][:2, 3]
        d = b - a
        ang = np.arctan2(d[1], d[0])
        R = np.array([[np.cos(ang), np.sin(ang)],
                      [-np.sin(ang), np.cos(ang)]])
        return R, p[:2]

    def crop(self, ki, t_win=45.0):
        """-> ego-frame xy, height-above-LOCAL-ground, label in the box.

        Time-local: only voxels observed within t_win of the current
        keyframe — revisit passes double the surfaces at 0.5-1.5 m z
        offset (slow FAST-LIO z-warp), which fakes wall-to-wall
        obstacles if the whole-bag map is used."""
        R, p = self.ego(ki)
        tki = self.kt[ki]
        rel = self.xy - p
        box = ((np.abs(rel[:, 0]) < HALF_M + 3)
               & (np.abs(rel[:, 1]) < HALF_M + 3)
               & (self.t0 < tki + t_win) & (self.t1 > tki - t_win))
        pe = rel[box] @ R.T
        keep = (np.abs(pe[:, 0]) < HALF_M) & (np.abs(pe[:, 1]) < HALF_M)
        pe, zw, lab = pe[keep], self.z[box][keep], self.label[box][keep]

        # local ground: per-0.5m-cell MEDIAN of plausible-ground voxels
        # (min latches onto sub-ground noise — ~10% of near-path voxels
        # sit up to 0.5 m below the road surface — which lifted the real
        # road into the obstacle band and faked wall-to-wall collisions)
        z0 = self.pose[ki][2, 3] - self.rig.lidar_height_m
        gcell = 0.5
        ng = int(2 * HALF_M / gcell) + 1
        ci = ((pe + HALF_M) / gcell).astype(np.int64)
        key = ci[:, 0] * ng + ci[:, 1]
        cand = np.abs(zw - z0) < 1.2           # plausible ground candidates
        gz = np.full(ng * ng, np.nan, np.float32)
        if cand.any():
            ck, cz = key[cand], zw[cand]
            order = np.lexsort((cz, ck))
            ck, cz = ck[order], cz[order]
            starts = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
            ends = np.r_[starts[1:], len(ck)]
            med = cz[(starts + ends - 1) // 2]      # per-cell median
            gz[ck[starts]] = np.clip(med, z0 - 1.2, z0 + 1.2)
        g = gz[key]
        g[np.isnan(g)] = z0
        return pe, zw - g, lab


class DepthMap:
    """Camera-verified local map: accumulated-depth backprojection over the
    last K keyframes (the SAME source the BEV model input uses) + per-
    keyframe semantic labels. No whole-bag ghosts: short window, camera
    FOV only, dynamics-masked at depth-cache build time."""

    K_HIST = 40
    PATCH = 32

    def __init__(self, site, bag):
        from wildvln.rigs import rig_for_site
        self.rig = rig_for_site(site)
        idx = np.load(P2B / site / bag / "index.npz")
        self.kt, self.pose, self.valid = idx["t"], idx["pose"], idx["valid"]
        self.stamp2ki = {int(t * 1e9): i for i, t in enumerate(idx["t"])}
        dz = np.load(Path("/data/patelm/ticvla/wildvln/p2cf/depth")
                     / site / f"{bag}.npz")
        self.depth = dz["grid"]
        self.semdir = Path("/data/patelm/ticvla/wildvln/p2c/sem") \
            / site / bag
        self.sems = sorted(self.semdir.glob("*.png"))
        assert len(self.sems) == len(self.kt)

        W, H = self.rig.image_size
        hd, wd = self.depth.shape[1:]
        u = (np.arange(wd) + 0.5) * self.PATCH
        v = (np.arange(hd) + 0.5) * self.PATCH
        uu, vv = np.meshgrid(np.minimum(u, W - 1), np.minimum(v, H - 1))
        self.uu, self.vv = uu.ravel(), vv.ravel()
        fx, fy, cx, cy = self.rig.intrinsics
        self.rays = np.stack([(self.uu - cx) / fx,
                              (self.vv - cy) / fy,
                              np.ones(hd * wd)], 1)
        T = np.asarray(self.rig.T_cam_lidar)
        self.R_lc, self.t_lc = T[:3, :3], T[:3, 3]
        self._semcache = {}

    def _sem(self, j):
        if j not in self._semcache:
            im = cv2.imread(str(self.sems[j]), cv2.IMREAD_UNCHANGED)
            sh = im.shape[0] / self.rig.image_size[1]
            sw = im.shape[1] / self.rig.image_size[0]
            self._semcache[j] = im[(self.vv * sh).astype(int),
                                   (self.uu * sw).astype(int)]
        return self._semcache[j]

    def ego(self, ki):
        p = self.pose[ki][:3, 3]
        a = self.pose[max(ki - 1, 0)][:2, 3]
        b = self.pose[min(ki + 1, len(self.pose) - 1)][:2, 3]
        d = b - a
        ang = np.arctan2(d[1], d[0])
        R = np.array([[np.cos(ang), np.sin(ang)],
                      [-np.sin(ang), np.cos(ang)]])
        return R, p[:2]

    def crop(self, ki):
        R, p = self.ego(ki)
        z0 = self.pose[ki][2, 3] - self.rig.lidar_height_m
        pts, labs = [], []
        for j in range(max(0, ki - self.K_HIST + 1), ki + 1):
            if not self.valid[j]:
                continue
            d = self.depth[j].ravel().astype(np.float32)
            ok = np.isfinite(d) & (d > 0.6) & (d < 30.0)
            if not ok.any():
                continue
            Xc = self.rays[ok] * d[ok, None]
            Xl = (Xc - self.t_lc) @ self.R_lc
            Pw = Xl @ self.pose[j][:3, :3].T + self.pose[j][:3, 3]
            pe = (Pw[:, :2] - p) @ R.T
            keep = (np.abs(pe) < HALF_M).all(1)
            pts.append(np.concatenate(
                [pe[keep], Pw[keep, 2:3]], 1))
            labs.append(self._sem(j)[ok][keep])
        if not pts:
            return (np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.uint8))
        P = np.concatenate(pts)
        lab = np.concatenate(labs)
        pe, zw = P[:, :2], P[:, 2]

        gcell = 0.5
        ng = int(2 * HALF_M / gcell) + 1
        ci = ((pe + HALF_M) / gcell).astype(np.int64)
        key = ci[:, 0] * ng + ci[:, 1]
        cand = np.abs(zw - z0) < 1.2
        gz = np.full(ng * ng, np.nan, np.float32)
        if cand.any():
            ck, cz = key[cand], zw[cand]
            order = np.lexsort((cz, ck))
            ck, cz = ck[order], cz[order]
            starts = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
            ends = np.r_[starts[1:], len(ck)]
            med = cz[(starts + ends - 1) // 2]
            gz[ck[starts]] = np.clip(med, z0 - 1.2, z0 + 1.2)
        g = gz[key]
        g[np.isnan(g)] = z0
        return pe, zw - g, lab


def to_px(pts, n):
    """ego (x fwd, y left) -> image px (x up, y left)."""
    c = (HALF_M - pts[:, 1]) / RES
    r = (HALF_M - pts[:, 0]) / RES
    return np.clip(np.stack([c, r], 1), 0, n - 1).astype(int)


def densify(pts, step=0.1):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.linalg.norm(b - a)
        for t in np.arange(step, d, step):
            out.append(a + (b - a) * t / d)
        out.append(b)
    return np.array(out)


def analyze(smap, ki, traces):
    """traces: {name: (K,2) pts}. -> render img, {name: collision stats}."""
    pe, hrel, label = smap.crop(ki)
    n = int(2 * HALF_M / RES)
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (40, 40, 46)

    gmask = hrel <= GROUND_MAX
    px = to_px(pe[gmask], n)
    cols = np.array([GROUND_COL.get(int(l), DEFAULT_GROUND)
                     for l in label[gmask]], np.uint8)
    img[px[:, 1], px[:, 0]] = cols
    # ground-class raster at 0.4 m (patch-depth points are too sparse
    # for 0.1 m; last write wins is fine at this density)
    gres = 0.4
    ngs = int(2 * HALF_M / gres)
    gs = np.clip(((np.stack(
        [(HALF_M - pe[gmask, 1]), (HALF_M - pe[gmask, 0])], 1)) / gres)
        .astype(int), 0, ngs - 1)
    gsem = np.full((ngs, ngs), 255, np.uint8)
    gsem[gs[:, 1], gs[:, 0]] = label[gmask]

    # obstacle test is SEMANTICS-FIRST: geometric height on these maps
    # is fragile (z-warp, hilly terrain, ghost layers). A voxel is an
    # obstacle if it sits in the body-height band AND
    #   - is painted a non-traversable class (building/tree/fence/...),
    #     with >=2-voxel column evidence, or
    #   - is unpainted (255) with >=3-voxel column evidence
    # traversable-labeled voxels are never obstacles.
    trav = np.isin(label, list(GROUND_COL.keys())[:-1]) | (label == 2)
    band = (hrel > OBST_BAND[0]) & (hrel < OBST_BAND[1]) & ~trav
    ck = (((pe[band] + HALF_M) / 0.2).astype(np.int64))
    key = ck[:, 0] * 4096 + ck[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    need = np.where(label[band] == 255, 3, 2)
    solid = cnt[inv] >= need
    occ = np.zeros((n, n), bool)
    opx = to_px(pe[band][solid], n)
    occ[opx[:, 1], opx[:, 0]] = True
    img[occ] = OBST_COL
    occ_d = binary_dilation(occ, iterations=int(ROBOT_R / RES))
    # the robot occupies the origin — its disc is self-contaminated
    yy, xx = np.mgrid[:n, :n]
    occ_d[(xx - n // 2) ** 2 + (yy - n // 2) ** 2
          < int(1.0 / RES) ** 2] = False

    stats = {}
    for name, pts in traces.items():
        if pts is None or len(pts) < 2:
            stats[name] = None
            continue
        dense = densify(pts)
        dense = dense[(np.abs(dense) < HALF_M - RES).all(1)]
        if not len(dense):
            stats[name] = None
            continue
        dp = to_px(dense, n)
        hit = occ_d[dp[:, 1], dp[:, 0]]
        s = np.arange(len(dense)) * 0.1
        # surface class under the trace
        gp = np.clip((np.stack([(HALF_M - dense[:, 1]),
                                (HALF_M - dense[:, 0])], 1) / gres)
                     .astype(int), 0, ngs - 1)
        under = gsem[gp[:, 1], gp[:, 0]]
        pav = np.isin(under, (6, 11, 3, 52, 46))
        soft = np.isin(under, (9, 13, 17))
        stats[name] = {
            "frac": float(hit.mean()),
            "collides": bool(hit.any()),
            "first_m": float(s[hit][0]) if hit.any() else None,
            "pave": float(pav.mean()), "soft": float(soft.mean()),
            "unk": float((under == 255).mean())}
    return img, stats


def draw_traces(img, traces, history=None):
    n = img.shape[0]
    if history is not None and len(history) >= 2:
        hp = to_px(np.asarray(history, float), n)
        cv2.polylines(img, [hp], False, (200, 200, 200), 1, cv2.LINE_AA)
    for name, pts in traces.items():
        if pts is None or len(pts) < 2:
            continue
        col = (60, 190, 60) if name == "gt" else COLORS.get(
            name, (255, 255, 255))
        tp = to_px(np.asarray(pts, float), n)
        cv2.polylines(img, [tp], False, col,
                      2 if name == "gt" else 1, cv2.LINE_AA)
    c = n // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 8)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test_site")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--src", choices=["voxel", "depth"], default="depth")
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("/data/patelm/ticvla/wildvln/p5/samples.parquet")
    sub = df[(df["mode"] == "chained") & (df.variant == 0)]
    idx = {(r["ep_id"], int(r["step"])): r for _, r in sub.iterrows()}

    preds = {}
    for spec in args.dumps:
        name, path = spec.split("=", 1)
        for line in open(path):
            r = json.loads(line)
            if r["split"] != args.split:
                continue
            preds.setdefault((r["ep_id"], r["step"]), {})[name] = r

    maps, meta, agg = {}, [], {}
    for (ep, step), by_model in sorted(preds.items()):
        row = idx[(ep, step)]
        key = (row["site"], row["bag"])
        if key not in maps:
            cls = DepthMap if args.src == "depth" else SemMap
            maps[key] = cls(*key)
        smap = maps[key]
        ki = smap.stamp2ki[int(Path(row["image"]).stem)]
        any_rec = next(iter(by_model.values()))
        traces = {"gt": np.asarray(any_rec["gt"], float)}
        for nm, rec in by_model.items():
            p = np.asarray(rec["pred"], float)
            traces[nm] = p if np.abs(p).max() < 30 else None
        img, stats = analyze(smap, ki, traces)
        hist = json.loads(row["history"]) if row["history"] else None
        img = draw_traces(img, traces, hist)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_NEAREST)
        nh = net_heading(any_rec["gt"])
        fname = f"{ep}_s{step:02d}.png"
        cv2.imwrite(str(out / fname), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        meta.append({"img": fname, "ep_id": ep, "step": step,
                     "nh_gt": None if nh is None else round(nh, 1),
                     "turn": bool(nh is not None and abs(nh) >= TURN_DEG),
                     "instruction": row["instruction"],
                     "collision": stats})
        for nm, s in stats.items():
            if s is not None:
                agg.setdefault(nm, []).append(s)

    print(f"{'model':8s} {'collides%':>9s} {'mean frac%':>10s} "
          f"{'first_m(med)':>12s}  n")
    summary = {}
    for nm, ss in sorted(agg.items()):
        firsts = [s["first_m"] for s in ss if s["first_m"] is not None]
        summary[nm] = {
            "collides": float(np.mean([s["collides"] for s in ss])),
            "frac": float(np.mean([s["frac"] for s in ss])),
            "first_med": float(np.median(firsts)) if firsts else None,
            "n": len(ss)}
        sm = summary[nm]
        print(f"{nm:8s} {sm['collides']*100:8.1f} {sm['frac']*100:9.1f} "
              f"{sm['first_med'] if sm['first_med'] else -1:12.1f} "
              f"{sm['n']:4d}")
    json.dump({"meta": meta, "summary": summary},
              open(out / "meta.json", "w"), indent=1)
    print("BEVSEM_DONE", len(meta), "->", out)


if __name__ == "__main__":
    main()
