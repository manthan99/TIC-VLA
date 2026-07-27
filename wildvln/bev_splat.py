#!/usr/bin/env python3
"""Causal BEV feature splat for the Wild VLN dataloader (GA-VLN recipe).

For a keyframe ki: take the last K_HIST keyframes (causal), backproject each
frame's merged ViT tokens (p2c/vit, the LLM's own image tokens) at their
patch centers through the accumulated-LiDAR patch depth (p2cf/depth), move
them through the FAST-LIO poses (p2bf) into the CURRENT ego frame
(motion-derived: x = travel direction, y = left — same frame as the trace
GT), and mean-pool per BEV cell.

Grid: +-GRID_HALF_M egocentric, CELL_M cells -> (N,2) cell indices +
(N,2048) pooled features + (N,) hit counts. Sparse non-empty cells become
LLM tokens downstream (2D sinusoidal metric PE + modality embedding).

Nothing is precomputed — splat at load time so grid params stay tunable.

QC: python -m wildvln.bev_splat <site> <bag> <kf_index> -> png render.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wildvln.rigs import rig_for_site

P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT",
                               "/data/patelm/ticvla/wildvln/p2bf"))
VIT_ROOT = Path("/data/patelm/ticvla/wildvln/p2c/vit")
DEPTH_ROOT = Path(os.environ.get("WILDVLN_DEPTH_ROOT",
                                 "/data/patelm/ticvla/wildvln/p2cf/depth"))

K_HIST = 40
GRID_HALF_M = 12.0
CELL_M = 0.25
Z_KEEP = (-2.0, 3.0)      # vs current pose height: floor..head-ish, no canopy
D_KEEP = (0.6, 30.0)
DEPTH_PATCH_PX = 32


class BevSplatter:
    """Per-bag cache holder; splat() is cheap enough for the dataloader."""

    def __init__(self, site: str, bag: str):
        self.rig = rig_for_site(site)
        vz = np.load(VIT_ROOT / site / f"{bag}.npz")
        self.feats = vz["feats"]                  # (F, N, 2048) fp16
        self.grid_hw = tuple(int(x) for x in vz["grid_hw"])
        dz = np.load(DEPTH_ROOT / site / f"{bag}.npz")
        self.depth = dz["grid"]                   # (F, Hd, Wd) fp16
        idx = np.load(P2B_ROOT / site / bag / "index.npz")
        self.kt, self.pose, self.valid = idx["t"], idx["pose"], idx["valid"]
        assert len(self.kt) == len(self.feats) == len(self.depth)

        # patch-center pixels + camera rays (once per bag)
        W, H = self.rig.image_size
        hv, wv = self.grid_hw
        u = (np.arange(wv) + 0.5) * W / wv
        v = (np.arange(hv) + 0.5) * H / hv
        uu, vv = np.meshgrid(u, v)
        self.px = np.minimum(uu.ravel().astype(int) // DEPTH_PATCH_PX,
                             self.depth.shape[2] - 1)
        self.py = np.minimum(vv.ravel().astype(int) // DEPTH_PATCH_PX,
                             self.depth.shape[1] - 1)
        fx, fy, cx, cy = self.rig.intrinsics
        self.rays = np.stack([(uu.ravel() - cx) / fx,
                              (vv.ravel() - cy) / fy,
                              np.ones(hv * wv)], 1)
        T = np.asarray(self.rig.T_cam_lidar)      # lidar -> cam
        self.R_lc, self.t_lc = T[:3, :3], T[:3, 3]

    def _ego(self, ki: int):
        """Motion-derived ego frame at ki (matches trace GT frame)."""
        p = self.pose[ki][:3, 3]
        a = self.pose[max(ki - 1, 0)][:2, 3]
        b = self.pose[min(ki + 1, len(self.pose) - 1)][:2, 3]
        d = b - a
        ang = np.arctan2(d[1], d[0])
        R = np.array([[np.cos(ang), np.sin(ang), 0],
                      [-np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])
        return R, p

    def splat(self, ki: int, k_hist: int = K_HIST):
        """-> (cells (M,2) int in [0, 2*GRID_HALF/CELL), feats (M,2048) f32,
        counts (M,)). Cell (0,0) = back-left; x fwd rows, y left cols."""
        R_ego, p_ego = self._ego(ki)
        n_cells = int(2 * GRID_HALF_M / CELL_M)
        all_keys, all_feats = [], []
        js = [j for j in range(max(0, ki - k_hist + 1), ki + 1)
              if self.valid[j]]
        for j in js:
            d = self.depth[j][self.py, self.px].astype(np.float32)
            ok = np.isfinite(d) & (d > D_KEEP[0]) & (d < D_KEEP[1])
            if not ok.any():
                continue
            Xc = self.rays[ok] * d[ok, None]
            Xl = (Xc - self.t_lc) @ self.R_lc          # cam -> lidar frame
            Pw = Xl @ self.pose[j][:3, :3].T + self.pose[j][:3, 3]
            Pe = (Pw - p_ego) @ R_ego.T                # current ego frame
            keep = ((np.abs(Pe[:, 0]) < GRID_HALF_M)
                    & (np.abs(Pe[:, 1]) < GRID_HALF_M)
                    & (Pe[:, 2] > Z_KEEP[0]) & (Pe[:, 2] < Z_KEEP[1]))
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
        cells = np.stack([uniq // n_cells, uniq % n_cells], 1).astype(np.int32)
        return cells, pooled, counts.astype(np.int32)


def _qc(site, bag, ki):
    import cv2
    sp = BevSplatter(site, bag)
    cells, feats, counts = sp.splat(ki)
    n = int(2 * GRID_HALF_M / CELL_M)
    print(f"{site}/{bag} kf{ki}: {len(cells)} non-empty cells "
          f"of {n*n} ({100*len(cells)/(n*n):.0f}%), "
          f"counts p50 {np.percentile(counts, 50):.0f}")
    # feature PCA -> RGB
    X = feats - feats.mean(0)
    _, _, Vt = np.linalg.svd(X[:: max(1, len(X) // 2000)], full_matrices=False)
    rgb = X @ Vt[:3].T
    rgb = (rgb - rgb.min(0)) / (np.ptp(rgb, axis=0) + 1e-6)
    img = np.zeros((n, n, 3))
    img[cells[:, 0], cells[:, 1]] = rgb
    img = np.flipud(img)                    # x fwd = up
    occ = np.zeros((n, n))
    occ[cells[:, 0], cells[:, 1]] = np.log1p(counts)
    occ = np.flipud(occ / occ.max())
    out = np.concatenate([img, np.repeat(occ[..., None], 3, 2)], 1)
    p = f"/tmp/claude-1003/-home-nvidia-patelm-ws-rsl-TIC-VLA/7276802c-e3c3-471a-80e8-08a4e248fa6a/scratchpad/bev_{site}_{ki}.png"
    cv2.imwrite(p, (out * 255).astype(np.uint8)[:, :, ::-1],
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("->", p)


if __name__ == "__main__":
    import sys
    _qc(sys.argv[1], sys.argv[2], int(sys.argv[3]))
