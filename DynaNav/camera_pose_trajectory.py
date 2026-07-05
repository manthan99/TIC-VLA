#!/usr/bin/env python3
"""
camera_pose_to_trajectory.py

Usage:
  python camera_pose_to_trajectory.py /path/to/jsons/ \
      --out_csv trajectory.csv

Assumptions:
- Each file contains "cameraViewTransform": a 16-number flattened 4x4 matrix.
- That matrix is a *view* (world→camera). We invert to get camera→world (global pose).
- Flatten order is auto-detected; override with --flatten row|col if needed.
"""

import argparse
import glob
import json
import math
import os
from typing import List, Tuple, Optional

import numpy as np


def reshape_matrix_4x4(flat: List[float], order: str) -> np.ndarray:
    """Return 4x4 matrix from 16-length flat list with explicit order."""
    if len(flat) != 16:
        raise ValueError(f"Expected 16 numbers for a 4x4 matrix, got {len(flat)}")
    if order == "row":
        return np.array(flat, dtype=np.float64).reshape((4, 4), order="C")
    elif order == "col":
        return np.array(flat, dtype=np.float64).reshape((4, 4), order="F")
    else:
        raise ValueError("order must be 'row' or 'col'")


def score_as_affine(m: np.ndarray) -> float:
    """Heuristic score: how affine does this look? Prefer last row ~ [0,0,0,1]."""
    last_row = m[3, :]
    return float(np.linalg.norm(last_row[:3])) + abs(last_row[3] - 1.0)


def autodetect_order(flat: List[float]) -> str:
    """Pick 'col' vs 'row' by how affine the matrix looks after reshape."""
    m_col = reshape_matrix_4x4(flat, "col")
    m_row = reshape_matrix_4x4(flat, "row")
    s_col = score_as_affine(m_col)
    s_row = score_as_affine(m_row)

    def translation_strength(m: np.ndarray) -> float:
        return float(np.linalg.norm(m[:3, 3]))

    if abs(s_col - s_row) < 1e-6:
        return "col" if translation_strength(m_col) >= translation_strength(m_row) else "row"
    return "col" if s_col < s_row else "row"


def matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (x, y, z, w) quaternion."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def view_to_pose(view_flat: List[float], flatten_order: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flattened view matrix (world→camera) to pose (camera→world).
    Returns (position_xyz, quaternion_xyzw).
    """
    order = flatten_order or autodetect_order(view_flat)
    V = reshape_matrix_4x4(view_flat, "col" if order.startswith("col") else "row")
    T_cw = np.linalg.inv(V)

    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    qx, qy, qz, qw = matrix_to_quaternion(R)
    return t, np.array([qx, qy, qz, qw], dtype=np.float64)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", help="path to JSON file(s) containing cameraViewTransform.")
    ap.add_argument("--flatten", choices=["row", "col"], default=None,
                    help="Override flatten order. Default: auto-detect.")
    ap.add_argument("--out_csv", default="trajectory.csv",
                    help="Output CSV path (time,x,y,z,qx,qy,qz,qw).")
    args = ap.parse_args()

    # Expand globs and sort deterministically
    print(f"Searching for input files in: {args.inputs}")
    files = glob.glob(f"{args.inputs}/*.json")
    print(f"Found {len(files)} input files.")
    files = sorted(set(files))

    if not files:
        raise SystemExit("No input files found.")

    rows_csv = []

    for idx, path in enumerate(files):
        data = load_json(path)
        if "cameraViewTransform" not in data:
            raise ValueError(f"{path} missing 'cameraViewTransform'")

        view_flat = data["cameraViewTransform"]
        t_xyz, q_xyzw = view_to_pose(view_flat, args.flatten)

        # Use index as time
        rows_csv.append([idx,
                         t_xyz[0], t_xyz[1], t_xyz[2],
                         q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]])

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw"])
        w.writerows(rows_csv)

    print(f"Wrote {len(rows_csv)} poses to {args.out_csv}")


if __name__ == "__main__":
    main()
