"""Patch-level LiDAR depth for BEV lifting — the official estimator.

A bare per-patch median fails two ways once scans are accumulated:
  - stray returns (solar interference, dust) own patches that have no other
    points — the "sky blobs" flagged in review;
  - points legitimately observed from another viewpoint land behind the
    current view's occluders (see-through).

Both die with a mode-cluster estimate: histogram the patch's depths in log
space, take the densest cluster (with neighbors), and require a minimum
point count before the patch produces a depth at all.
"""

from __future__ import annotations

import numpy as np

PATCH = 16
LOG_BIN = 0.12          # ~12% depth resolution per bin
MIN_PTS_SINGLE = 3
MIN_PTS_ACCUM = 6


def patch_depth_grid(u, v, z, width, height, min_pts=MIN_PTS_SINGLE):
    """(u, v, z) points -> (grid, mask) at patch resolution.

    grid[gy, gx] = depth of the dominant surface in that 16-px patch, NaN
    where fewer than `min_pts` points support any cluster.
    """
    gw = (width + PATCH - 1) // PATCH
    gh = (height + PATCH - 1) // PATCH
    grid = np.full((gh, gw), np.nan, np.float32)
    if len(z) == 0:
        return grid, np.zeros_like(grid, bool)

    key = (v // PATCH).astype(np.int64) * gw + (u // PATCH).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    uniq, starts = np.unique(ks, return_index=True)
    ends = np.append(starts[1:], len(ks))

    for k, a, b in zip(uniq, starts, ends):
        pz = zs[a:b]
        if len(pz) < min_pts:
            continue
        lo = np.log(np.maximum(pz, 0.1))
        bins = np.floor(lo / LOG_BIN).astype(np.int64)
        vals, counts = np.unique(bins, return_counts=True)
        # densest bin plus immediate neighbors
        bi = int(np.argmax(counts))
        sel = np.abs(bins - vals[bi]) <= 1
        if sel.sum() < min_pts:
            continue
        grid.flat[k] = np.median(pz[sel])
    return grid, ~np.isnan(grid)


# Ouster solar-interference filter (measured on lot9, 2026-07-25): artifact
# points cluster densely toward the sun with reflectivity ~6 and ambient
# ~3500 vs ~15 / ~2700 for real returns. Applied to UMD depth lifting only
# (Velodyne publishes no reflectivity/ambient).
SOLAR_REFL_MAX = 12
SOLAR_AMBIENT_MIN = 3200


def ouster_solar_mask(arr) -> np.ndarray:
    """True = keep. `arr` is the structured cloud array with named fields."""
    names = arr.dtype.names
    if "reflectivity" not in names or "ambient" not in names:
        return np.ones(len(arr), bool)
    bad = (arr["reflectivity"].astype(float) < SOLAR_REFL_MAX) & \
          (arr["ambient"].astype(float) > SOLAR_AMBIENT_MIN)
    return ~bad


def zbuffer_cull(u, v, z, width, height, cell=8, tol=1.25):
    """Camera-side visibility culling for camera<->LiDAR parallax.

    The LiDAR sits ~0.5 m behind the camera, so at every occlusion edge it
    returns background points along rays where the camera sees foreground (a
    ~50 px band for a 2 m occluder over 10 m background). Keep only points
    within `tol` of the nearest depth in their (cell x cell) neighborhood;
    the parallax-shadow background loses to the true front surface.
    """
    gw = (width + cell - 1) // cell
    key = (v // cell).astype(np.int64) * gw + (u // cell).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    uniq, starts = np.unique(ks, return_index=True)
    ends = np.append(starts[1:], len(ks))
    zmin = np.full(int(uniq.max()) + 1, np.inf, np.float32)
    for k, a, b in zip(uniq, starts, ends):
        zmin[k] = zs[a:b].min()
    keep = z <= zmin[key] * tol
    return keep
