#!/usr/bin/env python3
"""GrandTour causal BEV feature splat (port of wildvln.bev_splat).

Same GA-VLN recipe — backproject cached merged ViT tokens through
cached patch depth into the current motion-derived ego frame, mean-pool
per cell — with two platform adaptations:

  1. History is DISTANCE-windowed: cached splat kfs within HIST_M of
     arc-position behind the current kf (GrandTour raw kfs are 0.06 m
     apart; GND's K_HIST=40 at ~0.7 m spacing = ~28 m -> HIST_M 28).
  2. The z-band is SLOPE-AWARE: gated relative to the local path
     elevation (nearest path station), not the current pose height —
     on hilly missions ground 12 m ahead can sit far outside a
     pose-relative band and would be clipped.

Depth comes from wavemap raycasts (gt_splatcache), features from
gt_vitcache. Grid params imported from bev_splat so both platforms
stay comparable.

QC: python -m wildvln.gt_splat <bag> <ep_id> <step> -> png render.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from wildvln.bev_splat import CELL_M, GRID_HALF_M, D_KEEP

P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
DEPTH = Path("/data/patelm/ticvla/grandtour/p2c/depth")
VIT = Path("/data/patelm/ticvla/grandtour/p2c/vit")

HIST_M = 28.0
BASE_H = 0.55
Z_GROUND_KEEP = (-0.5, 3.0)       # vs local path ground
DEPTH_PATCH_PX = 32


class GtBevSplatter:
    def __init__(self, bag: str):
        idx = np.load(P2B / bag / "index.npz")
        self.pose, self.valid = idx["pose"], idx["valid"]
        self.stamp2i = {int(t * 1e9): i for i, t in enumerate(idx["t"])}
        rig = json.loads((P2B / bag / "rig.json").read_text())
        vz = np.load(VIT / f"{bag}.npz")
        dz = np.load(DEPTH / f"{bag}.npz")
        assert (vz["kf_i"] == dz["kf_i"]).all()
        self.kf_i = dz["kf_i"]
        self.feats, self.depth = vz["feats"], dz["grid"]
        hv, wv = (int(x) for x in vz["grid_hw"])

        # arc position of every valid kf + of each cached splat kf
        vidx = np.flatnonzero(self.valid)
        xyz = self.pose[vidx][:, :3, 3]
        seg = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.s_of = dict(zip(vidx.tolist(), s.tolist()))
        self.kf_s = np.array([self.s_of[k] for k in self.kf_i])
        self.ground = cKDTree(xyz[:, :2])
        self.ground_z = xyz[:, 2] - BASE_H

        W, H = rig["width"], rig["height"]
        u = (np.arange(wv) + 0.5) * W / wv
        v = (np.arange(hv) + 0.5) * H / hv
        uu, vv = np.meshgrid(u, v)
        self.px = np.minimum(uu.ravel().astype(int) // DEPTH_PATCH_PX,
                             self.depth.shape[2] - 1)
        self.py = np.minimum(vv.ravel().astype(int) // DEPTH_PATCH_PX,
                             self.depth.shape[1] - 1)
        fx, fy, cx, cy = rig["fx"], rig["fy"], rig["cx"], rig["cy"]
        rays = np.stack([(uu.ravel() - cx) / fx,
                         (vv.ravel() - cy) / fy,
                         np.ones(hv * wv)], 1)
        self.rays_base = rays @ np.linalg.inv(
            np.asarray(rig["T_cam_base"]))[:3, :3].T
        self.cam_in_base = np.linalg.inv(
            np.asarray(rig["T_cam_base"]))[:3, 3]

    def _ego(self, ki):
        p = self.pose[ki][:3, 3]
        a = self.pose[max(ki - 1, 0)][:2, 3]
        b = self.pose[min(ki + 1, len(self.pose) - 1)][:2, 3]
        d = b - a
        ang = np.arctan2(d[1], d[0])
        R = np.array([[np.cos(ang), np.sin(ang), 0],
                      [-np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])
        return R, p

    def splat(self, ki: int):
        """ki: index into index.npz arrays (use stamp2i for farm kfs).
        -> (cells (M,2) int, feats (M,2048) f32, counts (M,))."""
        R_ego, p_ego = self._ego(ki)
        s_i = self.s_of.get(int(ki))
        if s_i is None:
            s_i = self.kf_s[np.argmin(np.abs(self.kf_i - ki))]
        js = np.flatnonzero((self.kf_s <= s_i + 1e-6)
                            & (self.kf_s > s_i - HIST_M))
        n_cells = int(2 * GRID_HALF_M / CELL_M)
        all_keys, all_feats = [], []
        for j in js:
            d = self.depth[j][self.py, self.px].astype(np.float32)
            ok = np.isfinite(d) & (d > D_KEEP[0]) & (d < D_KEEP[1])
            if not ok.any():
                continue
            T = self.pose[self.kf_i[j]]
            Xb = self.rays_base[ok] * d[ok, None] + self.cam_in_base
            Pw = Xb @ T[:3, :3].T + T[:3, 3]
            Pe = (Pw - p_ego) @ R_ego.T
            _, gi = self.ground.query(Pw[:, :2], k=1)
            zg = Pw[:, 2] - self.ground_z[gi]
            keep = ((np.abs(Pe[:, 0]) < GRID_HALF_M)
                    & (np.abs(Pe[:, 1]) < GRID_HALF_M)
                    & (zg > Z_GROUND_KEEP[0]) & (zg < Z_GROUND_KEEP[1]))
            if not keep.any():
                continue
            ij = ((Pe[keep, :2] + GRID_HALF_M) / CELL_M).astype(np.int64)
            all_keys.append(ij[:, 0] * n_cells + ij[:, 1])
            all_feats.append(self.feats[j][ok][keep].astype(np.float32))
        if not all_keys:
            return (np.zeros((0, 2), np.int32),
                    np.zeros((0, self.feats.shape[2]), np.float32),
                    np.zeros(0, np.int32))
        keys = np.concatenate(all_keys)
        feats = np.concatenate(all_feats)
        uniq, inv, counts = np.unique(keys, return_inverse=True,
                                      return_counts=True)
        pooled = np.zeros((len(uniq), feats.shape[1]), np.float32)
        np.add.at(pooled, inv, feats)
        pooled /= counts[:, None]
        cells = np.stack([uniq // n_cells, uniq % n_cells], 1)
        return cells.astype(np.int32), pooled, counts.astype(np.int32)


@functools.lru_cache(maxsize=4)
def get_splatter(bag: str) -> GtBevSplatter:
    return GtBevSplatter(bag)


def main() -> None:
    import sys
    import cv2
    bag, ep_id, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    e = json.loads(Path(
        f"/data/patelm/ticvla/grandtour/p4/farm/{ep_id}.json").read_text())
    sp = get_splatter(bag)
    ki = sp.stamp2i[int(Path(e["steps"][step]["kf"]).stem)]
    cells, feats, counts = sp.splat(ki)
    n = int(2 * GRID_HALF_M / CELL_M)
    img = np.zeros((n, n, 3), np.uint8)
    if len(cells):
        heat = np.clip(counts / counts.max(), 0.15, 1.0)
        for (r, c), h in zip(cells, heat):
            img[n - 1 - r, n - 1 - c] = (int(40 + 60 * h),
                                         int(255 * h), int(60 * h))
    tr = np.asarray(e["steps"][step]["target"]["trace"], float)
    pr = np.clip(n - 1 - ((tr[:, 0] + GRID_HALF_M) / CELL_M), 0, n - 1)
    pc = np.clip(n - 1 - ((tr[:, 1] + GRID_HALF_M) / CELL_M), 0, n - 1)
    pts = np.stack([pc, pr], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, (60, 60, 255), 1, cv2.LINE_AA)
    img = cv2.resize(img, (n * 5, n * 5), interpolation=cv2.INTER_NEAREST)
    out = f"/tmp/gt_splat_{ep_id}_s{step:02d}.png"
    cv2.imwrite(out, img)
    print(f"{len(cells)} non-empty cells, mean count "
          f"{counts.mean():.1f} -> {out}")


if __name__ == "__main__":
    main()
