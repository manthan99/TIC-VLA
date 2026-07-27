#!/usr/bin/env python3
"""Measure camera height and pitch from ground motion under known odometry.

Back-projecting a ground pixel needs a camera height, and the depth you get out
scales linearly with whatever height you assume. Drive a known distance and that
ambiguity resolves: too low a height puts the ground too close and predicts more
motion than the image shows, too high predicts less. Pitch falls out at the same
time, because it decides which pixel rows map to which distances.

So: track features on the ground between consecutive frames, and for each
candidate (height, pitch) back-project, move the point by the odometry, project
again, and count how many land within a couple of pixels of where they were
actually tracked to. Points on walls and cars never fit the ground model and
drop out as outliers, which is why this scores inliers rather than least squares.

Usage:
    python scripts/ground_flow_calibration.py \
        --bag /data/patelm/ticvla/GND_raw/AU/AU_chunk01.bag \
        --image-topic zed_node/rgb/image_rect_color/compressed \
        --fx 263.8 --fy 263.8 --cx 328.9 --cy 178.55
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

MIN_DEPTH_M = 0.15


def decode(msg):
    if hasattr(msg, "format"):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2GRAY)


def yaw_of(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                            1 - 2 * (q.y * q.y + q.z * q.z)))


def backproject(u, v, K, height, pitch_rad):
    """Pixels -> (forward, left) on the ground plane, matching projection.py."""
    fx, fy, cx, cy = K
    a = (u - cx) / fx
    b = (v - cy) / fy
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    denom = b * cos_p + sin_p
    with np.errstate(divide="ignore", invalid="ignore"):
        forward = height * (cos_p - b * sin_p) / denom
    z_cv = height * sin_p + forward * cos_p
    left = -a * z_cv
    valid = np.isfinite(forward) & (denom > 1e-6) & (z_cv > MIN_DEPTH_M) & (forward > 0.3)
    return forward, left, valid


def project(forward, left, K, height, pitch_rad):
    fx, fy, cx, cy = K
    x_cv = -left
    y_cv = np.full_like(forward, height)
    z_cv = forward
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    y_r = y_cv * cos_p - z_cv * sin_p
    z_r = y_cv * sin_p + z_cv * cos_p
    valid = z_r > MIN_DEPTH_M
    safe = np.where(valid, z_r, 1.0)
    return fx * x_cv / safe + cx, fy * y_r / safe + cy, valid


def collect_tracks(bag, image_topic, odom_topic, max_pairs, v_lo, min_speed, baseline_s):
    """Feature correspondences in the lower image, plus the odometry between them."""
    images, poses = [], []
    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in (image_topic, odom_topic)]
        if len({c.topic for c in conns}) < 2:
            raise SystemExit(f"topics missing; have {[c.topic for c in reader.connections]}")
        for conn, bag_time, raw in reader.messages(connections=conns):
            stamp = bag_time * 1e-9
            if conn.topic == image_topic:
                images.append((stamp, raw, conn))
            else:
                msg = reader.deserialize(raw, conn.msgtype)
                t = msg.twist.twist
                poses.append((stamp, t.linear.x, t.linear.y, t.angular.z))

        # Dead-reckon from the twist rather than differencing the pose. The AU
        # bag's filtered pose teleports (differencing it peaks at ~3e4 m/s where
        # the GPS fusion resets), which would corrupt every baseline that spans
        # a jump. Velocity is unaffected, and over the fraction of a second we
        # actually integrate, its drift is irrelevant.
        poses = np.array(poses)
        pt = poses[:, 0]
        dt = np.diff(pt, prepend=pt[0])
        pyaw = np.cumsum(poses[:, 3] * dt)
        px = np.cumsum((poses[:, 1] * np.cos(pyaw) - poses[:, 2] * np.sin(pyaw)) * dt)
        py = np.cumsum((poses[:, 1] * np.sin(pyaw) + poses[:, 2] * np.cos(pyaw)) * dt)

        # Consecutive frames are useless here: at walking pace the robot moves a
        # couple of centimetres between them, which is a pixel or two of ground
        # flow and cannot separate one height hypothesis from another. Skip
        # ahead far enough that the baseline actually carries information.
        out = []
        times = np.array([t for t, _, _ in images])
        gap = max(1, int(np.searchsorted(times - times[0], baseline_s)))
        step = max(1, (len(images) - gap) // (max_pairs * 4))
        for i in range(0, len(images) - gap, step):
            t0, raw0, c0 = images[i]
            t1, raw1, c1 = images[i + gap]
            if not (0.2 * baseline_s < t1 - t0 < 3.0 * baseline_s):
                continue
            x0, y0, a0 = (np.interp(t0, pt, px), np.interp(t0, pt, py), np.interp(t0, pt, pyaw))
            x1, y1, a1 = (np.interp(t1, pt, px), np.interp(t1, pt, py), np.interp(t1, pt, pyaw))
            dx_w, dy_w = x1 - x0, y1 - y0
            dist = float(np.hypot(dx_w, dy_w))
            if dist / (t1 - t0) < min_speed:
                continue
            # World delta -> forward/left of the frame-0 body frame.
            fwd = dx_w * np.cos(a0) + dy_w * np.sin(a0)
            lft = -dx_w * np.sin(a0) + dy_w * np.cos(a0)
            dpsi = float(a1 - a0)

            g0 = decode(reader.deserialize(raw0, c0.msgtype))
            g1 = decode(reader.deserialize(raw1, c1.msgtype))
            if g0 is None or g1 is None:
                continue
            h, w = g0.shape
            mask = np.zeros_like(g0)
            mask[int(v_lo * h):, :] = 255
            p0 = cv2.goodFeaturesToTrack(g0, 400, 0.005, 6, mask=mask)
            if p0 is None or len(p0) < 30:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
            if p1 is None:
                continue
            ok = st.ravel() == 1
            a_pts, b_pts = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
            if len(a_pts) < 30:
                continue
            out.append((a_pts, b_pts, fwd, lft, dpsi, (h, w)))
            if len(out) >= max_pairs:
                break
    return out


def score(tracks, K, height, pitch_deg, tol_px):
    """Fraction of tracked points explained by the ground model."""
    pitch = np.radians(pitch_deg)
    inliers = total = 0
    for a_pts, b_pts, fwd, lft, dpsi, _ in tracks:
        F, L, valid = backproject(a_pts[:, 0], a_pts[:, 1], K, height, pitch)
        if valid.sum() < 10:
            continue
        F, L = F[valid], L[valid]
        b_sel = b_pts[valid]
        # Same ground point expressed in the next frame.
        dF, dL = F - fwd, L - lft
        c, s = np.cos(dpsi), np.sin(dpsi)
        F2 = dF * c + dL * s
        L2 = -dF * s + dL * c
        u2, v2, vis = project(F2, L2, K, height, pitch)
        err = np.hypot(u2 - b_sel[:, 0], v2 - b_sel[:, 1])
        inliers += int(((err < tol_px) & vis).sum())
        total += int(vis.sum())
    return inliers / max(total, 1), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--image-topic", required=True)
    ap.add_argument("--odom-topic", default="/odometry/filtered")
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--pairs", type=int, default=90)
    ap.add_argument("--v-lo", type=float, default=0.55,
                    help="only track below this fraction of image height")
    ap.add_argument("--min-speed", type=float, default=0.3)
    ap.add_argument("--tol-px", type=float, default=3.0)
    ap.add_argument("--baseline", type=float, default=0.7,
                    help="seconds between the two frames of a pair")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    K = (args.fx, args.fy, args.cx, args.cy)
    tracks = collect_tracks(args.bag, args.image_topic, args.odom_topic,
                            args.pairs, args.v_lo, args.min_speed, args.baseline)
    print(f"{args.label or Path(args.bag).name}: {len(tracks)} usable frame pairs")
    if len(tracks) < 10:
        raise SystemExit("not enough motion in this bag")

    heights = np.arange(0.30, 1.55, 0.05)
    pitches = np.arange(-6.0, 18.1, 1.0)
    grid = np.zeros((len(heights), len(pitches)))
    for i, h in enumerate(heights):
        for j, p in enumerate(pitches):
            grid[i, j], _ = score(tracks, K, float(h), float(p), args.tol_px)

    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"  best: height = {heights[i]:.2f} m, pitch = {pitches[j]:+.1f} deg "
          f"({grid[i, j] * 100:.1f}% inliers)")

    print("\n  inlier % vs height (at best pitch):")
    for h, s in zip(heights, grid[:, j]):
        bar = "#" * int(s * 60)
        mark = "  <--" if abs(h - heights[i]) < 1e-9 else ""
        print(f"    {h:.2f} m |{bar:<60s}| {s * 100:5.1f}{mark}")
    print("\n  inlier % vs pitch (at best height):")
    for p, s in zip(pitches, grid[i, :]):
        bar = "#" * int(s * 60)
        mark = "  <--" if abs(p - pitches[j]) < 1e-9 else ""
        print(f"    {p:+5.1f}d |{bar:<60s}| {s * 100:5.1f}{mark}")


if __name__ == "__main__":
    main()
