#!/usr/bin/env python3
"""GrandTour -> GND-farm-compatible index tree (P2b equivalent).

Per mission:
  p2b/<mission>/index.npz    t (N,), pose (N,4,4) T_dlio_map->base,
                             valid (N,), seg_id (N,) zeros, path_z
  p2b/<mission>/keyframes/<stamp_ns>.jpg   symlinks to rectified frames
  p2b/<mission>/rig.json     rectified pinhole K + T_cam_base + sizes

Poses: dlio_map_odometry is T_dlio_map->hesai (devkit convention);
composed to base via tf statics (box_base indirection handled as in the
devkit's get_static_transform). Camera extrinsic likewise; the optical
convention is verified visually by projecting the driven path (QC mode).

Usage:
  python -m wildvln.gt_p2index --mission 2024-11-04-10-57-34 [--qc]
  python -m wildvln.gt_p2index --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

RAW = Path("/data/patelm/ticvla/grandtour/raw")
# farm-compatible layout: <root>/<site>/<bag> with site="grandtour"
P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
QC = Path("/data/patelm/ticvla/grandtour/qc/p2index")


def pq_to_se3(t):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([t["rotation"][k] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [t["translation"][k] for k in "xyz"]
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def static_to_base(tf_attrs, frame):
    """T_frame->base via the devkit's box_base indirection."""
    T_boxbase_base = pq_to_se3(tf_attrs["box_base"])
    if frame == "base":
        return np.eye(4)
    e = tf_attrs[frame]
    T = pq_to_se3(e)
    if e["base_frame_id"] == "box_base":
        T = T @ T_boxbase_base
    return T


def build(mission, qc=False):
    import zarr
    mdir = RAW / mission
    rect = mdir / "images" / "hdr_front_rect"
    if not rect.exists():
        return f"{mission}: no rect images"
    od = zarr.open_group(str(mdir / "data" / "dlio_map_odometry"),
                         mode="r", zarr_format=2)
    ots, opos, oq = (od["timestamp"][:], od["pose_pos"][:],
                     od["pose_orien"][:])
    hdr = zarr.open_group(str(mdir / "data" / "hdr_front"),
                          mode="r", zarr_format=2)
    hts = hdr["timestamp"][:]
    tf = zarr.open_group(str(mdir / "data" / "tf"), mode="r",
                         zarr_format=2).attrs["tf"]
    hes_attr = zarr.open_group(
        str(mdir / "data" / "hesai_points_undistorted"), mode="r",
        zarr_format=2).attrs["transform"]

    # T_map->base(t) = T_map->hesai(t) @ inv(T_boxbase->hesai)
    #                  @ T_boxbase->base
    T_boxbase_hesai = pq_to_se3(hes_attr)
    T_hesai_base = inv(T_boxbase_hesai) @ pq_to_se3(tf["box_base"])

    rot = R.from_quat(oq)
    slerp = Slerp(ots, rot)
    poses = np.zeros((len(hts), 4, 4), np.float64)
    valid = np.zeros(len(hts), bool)
    for i, t in enumerate(hts):
        if t < ots[0] or t > ots[-1]:
            continue
        Tm = np.eye(4)
        Tm[:3, :3] = slerp(t).as_matrix()
        for k in range(3):
            Tm[k, 3] = np.interp(t, ots, opos[:, k])
        poses[i] = Tm @ T_hesai_base
        valid[i] = True

    out = P2B / mission
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "index.npz", t=hts, pose=poses,
                        valid=valid, seg_id=np.zeros(len(hts), np.int32))
    frames = sorted(rect.glob("*.jpeg"))
    for i, t in enumerate(hts):
        if i >= len(frames):
            break
        link = out / "keyframes" / f"{int(t*1e9)}.jpg"
        if not link.exists():
            link.symlink_to(frames[i])

    intr = json.load(open(rect / "intrinsics.json"))
    # static_to_base() already maps base-frame points into the optical
    # camera frame (devkit attrs are parent/child-swapped) — verified
    # numerically: forward -> +z, ground -> +y
    T_cam_base = static_to_base(tf, "hdr_front")
    rig = {**intr, "T_cam_base": T_cam_base.tolist(), "mission": mission}
    json.dump(rig, open(out / "rig.json", "w"), indent=1)

    if qc:
        QC.mkdir(parents=True, exist_ok=True)
        # project the future driven path (true 3D) into sample frames
        ok = np.where(valid)[0]
        opt = R.from_euler("yzx", [0, 0, 0])  # placeholder no-op
        for i in ok[[len(ok) // 4, len(ok) // 2,
                     3 * len(ok) // 4]]:
            im = cv2.imread(str(frames[i]))
            Tw = poses[i]
            fut = [poses[j][:3, 3] for j in ok
                   if 0.3 < np.linalg.norm(poses[j][:3, 3] - Tw[:3, 3])
                   < 12 and hts[j] > hts[i]]
            if len(fut) < 3:
                continue
            Pb = (np.array(fut) - Tw[:3, 3]) @ Tw[:3, :3]
            Pc = Pb @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
            front = Pc[:, 2] > 0.3
            u = intr["fx"] * Pc[front, 0] / Pc[front, 2] + intr["cx"]
            v = intr["fy"] * Pc[front, 1] / Pc[front, 2] + intr["cy"]
            pts = [(int(a), int(b)) for a, b in zip(u, v)
                   if 0 <= a < intr["width"] and 0 <= b < intr["height"]]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(im, a, b, (60, 220, 60), 3, cv2.LINE_AA)
            p = QC / f"{mission}_{i:06d}.jpg"
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("qc ->", p)
    return f"{mission}: {int(valid.sum())}/{len(hts)} frames indexed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--qc", action="store_true")
    args = ap.parse_args()
    if args.all:
        missions = sorted(d.name for d in RAW.iterdir() if d.is_dir()
                          and (d / "images" / "hdr_front_rect").exists())
    else:
        missions = [args.mission]
    for m in missions:
        print(build(m, qc=args.qc), flush=True)
    print("GT_P2INDEX_DONE")


if __name__ == "__main__":
    main()
