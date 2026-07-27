#!/usr/bin/env python3
"""Measure camera focal length from LiDAR/image bearing agreement.

Vertical structure (poles, trunks, building corners) shows up twice: as a range
discontinuity at some bearing in the LiDAR scan, and as a vertical edge at some
column in the image. A pinhole camera ties the two together:

    u = cx + fx * tan(theta - yaw0)

so sweeping (fx, yaw0) and correlating the two edge profiles recovers fx. The
camera-LiDAR lever arm is a few centimetres against object ranges of metres, so
it shifts bearings far too little to matter here.

Run it on a bag whose focal length is already known to see what the method's
accuracy actually is, then trust it on the unknown rig.

Usage:
    python scripts/lidar_fx_calibration.py --bag AU_chunk01.bag \
        --image-topic zed_node/rgb/image_rect_color/compressed \
        --cloud-topic /velodyne_points --cx 328.9 --truth 263.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def cloud_to_xyz(msg) -> np.ndarray:
    """PointCloud2 -> (N, 3) float array, dropping non-finite points."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in _DTYPES:
            continue
        names.append(f.name)
        formats.append(_DTYPES[f.datatype])
        offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
    return xyz[np.isfinite(xyz).all(axis=1)]


def lidar_edge_profile(xyz: np.ndarray, bearings: np.ndarray,
                       z_lo: float, z_hi: float, max_range: float) -> np.ndarray:
    """Range discontinuity strength per bearing bin, from a height slab."""
    rng = np.hypot(xyz[:, 0], xyz[:, 1])
    keep = (xyz[:, 2] > z_lo) & (xyz[:, 2] < z_hi) & (rng > 0.8) & (rng < max_range)
    if keep.sum() < 200:
        return np.zeros(len(bearings) - 1)
    pts, rng = xyz[keep], rng[keep]
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    idx = np.digitize(theta, bearings) - 1
    valid = (idx >= 0) & (idx < len(bearings) - 1)
    idx, rng = idx[valid], rng[valid]

    nearest = np.full(len(bearings) - 1, np.nan)
    order = np.argsort(rng)[::-1]
    nearest[idx[order]] = rng[order]          # closest point wins per bin
    filled = np.where(np.isnan(nearest), max_range, nearest)
    return np.abs(np.diff(filled, prepend=filled[0]))


def image_edge_profile(gray: np.ndarray, v_lo: float, v_hi: float) -> np.ndarray:
    """Vertical-edge energy per column over a row band."""
    band = gray[int(v_lo * gray.shape[0]):int(v_hi * gray.shape[0])]
    gx = np.abs(cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    return gx.sum(axis=0)


def normalize(sig: np.ndarray) -> np.ndarray:
    sig = sig - sig.mean()
    n = np.linalg.norm(sig)
    return sig / n if n > 1e-9 else sig


def decode(msg):
    if hasattr(msg, "format"):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2GRAY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--image-topic", required=True)
    ap.add_argument("--cloud-topic", required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--truth", type=float, default=None)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--z-lo", type=float, default=-0.3)
    ap.add_argument("--z-hi", type=float, default=1.5)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--v-lo", type=float, default=0.15)
    ap.add_argument("--v-hi", type=float, default=0.75)
    args = ap.parse_args()

    # Pair each cloud with the image closest in time.
    images, clouds = [], []
    with AnyReader([Path(args.bag)]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in (args.image_topic, args.cloud_topic)]
        if len(conns) < 2:
            raise SystemExit(f"topics not found; have {[c.topic for c in reader.connections]}")
        # Pair on bag receive time: the Ouster stamps its headers with sensor
        # uptime, so header stamps are not on a common clock across topics.
        for conn, bag_time, raw in reader.messages(connections=conns):
            stamp = bag_time * 1e-9
            if conn.topic == args.image_topic:
                images.append((stamp, raw, conn))
            elif len(clouds) < args.pairs * 3:
                clouds.append((stamp, raw, conn))

        img_times = np.array([t for t, _, _ in images])
        step = max(1, len(clouds) // args.pairs)
        width = None
        lidar_profiles, image_profiles, bearings = [], [], None

        for stamp, raw, conn in clouds[::step][:args.pairs]:
            j = int(np.argmin(np.abs(img_times - stamp)))
            if abs(img_times[j] - stamp) > 0.10:
                continue
            gray = decode(reader.deserialize(images[j][1], images[j][2].msgtype))
            if gray is None:
                continue
            if width is None:
                width = gray.shape[1]
                bearings = np.linspace(np.radians(-70), np.radians(70), 561)
            xyz = cloud_to_xyz(reader.deserialize(raw, conn.msgtype))
            lp = lidar_edge_profile(xyz, bearings, args.z_lo, args.z_hi, args.max_range)
            if not lp.any():
                continue
            lidar_profiles.append(lp)
            image_profiles.append(image_edge_profile(gray, args.v_lo, args.v_hi))

    print(f"{len(lidar_profiles)} synchronised LiDAR/image pairs, image width {width}")
    if len(lidar_profiles) < 5:
        raise SystemExit("not enough pairs")

    centers = 0.5 * (bearings[1:] + bearings[:-1])
    cols = np.arange(width)
    best = []
    fx_grid = np.arange(150.0, 500.0, 2.0)
    yaw_grid = np.radians(np.arange(-6.0, 6.01, 0.5))

    scores = np.zeros((len(fx_grid), len(yaw_grid)))
    for lp, ip in zip(lidar_profiles, image_profiles):
        ipn = normalize(ip)
        for a, fx in enumerate(fx_grid):
            for b, yaw in enumerate(yaw_grid):
                u = args.cx + fx * np.tan(centers - yaw)
                inside = (u > 0) & (u < width - 1)
                if inside.sum() < 50:
                    continue
                resampled = np.interp(cols, u[inside], lp[inside], left=0.0, right=0.0)
                scores[a, b] += float(normalize(resampled) @ ipn)

    a, b = np.unravel_index(np.argmax(scores), scores.shape)
    fx_hat, yaw_hat = fx_grid[a], np.degrees(yaw_grid[b])

    # Parabolic refinement along fx at the winning yaw.
    if 0 < a < len(fx_grid) - 1:
        y0, y1, y2 = scores[a - 1, b], scores[a, b], scores[a + 1, b]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            fx_hat += 2.0 * 0.5 * (y0 - y2) / denom

    print(f"  fx = {fx_hat:.1f} px,  yaw offset = {yaw_hat:+.1f} deg")
    if args.truth:
        print(f"  truth {args.truth:.1f} -> error {100 * (fx_hat - args.truth) / args.truth:+.1f}%")

    prof = scores[:, b]
    prof = (prof - prof.min()) / (prof.max() - prof.min() + 1e-12)
    for fx, s in zip(fx_grid[::10], prof[::10]):
        print(f"    fx {fx:5.0f} |{'#' * int(s * 50)}")


if __name__ == "__main__":
    main()
