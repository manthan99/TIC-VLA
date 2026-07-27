#!/usr/bin/env python3
"""Pull real camera calibration and mount geometry out of a ROS bag.

Reads camera_info for the true intrinsics and /tf_static for the camera pose
relative to the robot base, which is what the image-space trace projection needs
(focal length, principal point, mount height, mount pitch). No ROS install
required — uses the pure-python `rosbags` reader.

Usage:
    python scripts/extract_bag_calibration.py /data/patelm/ticvla/dataset/AU.bag
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def quat_to_rpy_deg(x: float, y: float, z: float, w: float) -> tuple:
    norm = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(float(v) for v in np.degrees([roll, pitch, yaw]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=str)
    parser.add_argument("--max-tf", type=int, default=4000,
                        help="Stop after this many /tf messages (static ones come first)")
    args = parser.parse_args()

    path = Path(args.bag)
    with AnyReader([path]) as reader:
        print(f"=== {path.name}  ({path.stat().st_size / 1e9:.2f} GB)\n")
        print("--- topics")
        for conn in sorted(reader.connections, key=lambda c: c.topic):
            print(f"  {conn.topic:55s} {conn.msgtype:42s} n={conn.msgcount}")

        # 1) Intrinsics from any camera_info topic.
        info_conns = [c for c in reader.connections if "CameraInfo" in c.msgtype]
        print("\n--- camera_info")
        if not info_conns:
            print("  (none found)")
        seen = set()
        for conn in info_conns:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic in seen:
                    break
                seen.add(conn.topic)
                K = np.asarray(msg.k).reshape(3, 3) if hasattr(msg, "k") else np.asarray(msg.K).reshape(3, 3)
                width = getattr(msg, "width", 0)
                height = getattr(msg, "height", 0)
                hfov = 2 * np.degrees(np.arctan((width / 2.0) / K[0, 0])) if K[0, 0] else float("nan")
                vfov = 2 * np.degrees(np.arctan((height / 2.0) / K[1, 1])) if K[1, 1] else float("nan")
                dist = getattr(msg, "d", None)
                if dist is None:
                    dist = getattr(msg, "D", [])
                print(f"  {conn.topic}")
                print(f"    frame_id : {getattr(msg.header, 'frame_id', '?')}")
                print(f"    size     : {width} x {height}")
                print(f"    fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
                print(f"    HFOV={hfov:.1f}°  VFOV={vfov:.1f}°")
                print(f"    model={getattr(msg, 'distortion_model', '?')}  D={list(dist)[:5]}")
                break

        # 2) Camera mount geometry from the transform tree.
        print("\n--- transforms (camera mount: height and pitch)")
        tf_conns = [c for c in reader.connections if c.topic in ("/tf_static", "/tf")]
        edges = {}
        count = 0
        for conn in tf_conns:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                for tr in msg.transforms:
                    parent = tr.header.frame_id.lstrip("/")
                    child = tr.child_frame_id.lstrip("/")
                    t, r = tr.transform.translation, tr.transform.rotation
                    edges.setdefault((parent, child), (
                        (t.x, t.y, t.z), quat_to_rpy_deg(r.x, r.y, r.z, r.w),
                        conn.topic,
                    ))
                count += 1
                if conn.topic == "/tf" and count > args.max_tf:
                    break
        if not edges:
            print("  (no transforms found)")
        for (parent, child), (xyz, rpy, topic) in sorted(edges.items()):
            flag = "  <-- camera" if any(k in child.lower() for k in ("cam", "rgb", "color", "optical")) else ""
            print(f"  {parent:28s} -> {child:32s} xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
                  f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})° [{topic}]{flag}")

        print("\nWhat to use: camera height = z of base_link -> camera chain; "
              "mount pitch = pitch of that same chain (positive = nose down).")


if __name__ == "__main__":
    main()
