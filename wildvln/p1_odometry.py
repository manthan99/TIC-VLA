#!/usr/bin/env python3
"""P1: one streaming pass per bag -> KISS-ICP poses, keyframes, GPS, twist.

rosbag1 chunks interleave topics, so reading any one topic costs most of the
bag; this stage therefore reads each bag exactly once and serves every
consumer that needs bulk data:

  poses.npz      KISS-ICP pose per cloud (bag time), 4x4 matrices
  keyframes.npz  2 Hz image keyframe bag-times (P2b decodes these frames)
  gps.npz        t, lat, lon, fix_ok
  twist.npz      t, vx, vy, wz     (EKF twist; pose field never used, see rigs)
  qc.json        speed stats, teleport check, GPS coverage, timing

Per-bag artifacts land in /data/patelm/ticvla/wildvln/p1/<site>/<bag_stem>/.
Bags process independently -> ProcessPoolExecutor over bags.

Usage:
    python -m wildvln.p1_odometry                    # everything in manifest
    python -m wildvln.p1_odometry --sites AU --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from wildvln.rigs import rig_for_site

P0_MANIFEST = Path("/data/patelm/ticvla/wildvln/p0/manifest.json")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p1")

KEYFRAME_HZ = 2.0
KISS_MAX_RANGE = 60.0
KISS_VOXEL = 0.6
SPEED_LIMIT = 3.0            # m/s; campus robots top out ~2

_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def cloud_to_xyz(msg) -> np.ndarray:
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype in _DTYPES:
            names.append(f.name)
            formats.append(_DTYPES[f.datatype])
            offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def process_bag(rec: dict) -> dict:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    site, bag_path = rec["site"], rec["path"]
    rig = rig_for_site(site)
    out_dir = OUT_ROOT / site / Path(bag_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "qc.json"
    if done_marker.exists():
        return {"bag": rec["bag"], "site": site, "skipped": True}

    cfg = KISSConfig()
    cfg.data.deskew = False
    cfg.data.max_range = KISS_MAX_RANGE
    cfg.mapping.voxel_size = KISS_VOXEL
    odom = KissICP(cfg)

    poses, pose_t = [], []
    keyframes = []
    gps_rows, twist_rows = [], []
    last_kf = -1e18
    t0 = time.time()

    with AnyReader([Path(bag_path)]) as reader:
        wanted = {rig.image_topic, rig.cloud_topic, rig.odom_topic, rig.gps_topic}
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, bag_ns, raw in reader.messages(connections=conns):
            t = bag_ns * 1e-9
            if conn.topic == rig.cloud_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                xyz = cloud_to_xyz(msg)
                if len(xyz) < 100:
                    continue
                odom.register_frame(xyz, np.zeros(len(xyz)))
                poses.append(odom.last_pose.copy())
                pose_t.append(t)
            elif conn.topic == rig.image_topic:
                if t - last_kf >= 1.0 / KEYFRAME_HZ:
                    keyframes.append(t)
                    last_kf = t
            elif conn.topic == rig.odom_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                tw = msg.twist.twist
                twist_rows.append((t, tw.linear.x, tw.linear.y, tw.angular.z))
            elif conn.topic == rig.gps_topic:
                msg = reader.deserialize(raw, conn.msgtype)
                if hasattr(msg, "fixType"):                    # ublox NavPVT
                    ok = int(msg.fixType) >= 3
                    lat, lon = msg.lat * 1e-7, msg.lon * 1e-7
                else:                                          # NavSatFix
                    ok = int(msg.status.status) >= 0
                    lat, lon = float(msg.latitude), float(msg.longitude)
                gps_rows.append((t, lat, lon, float(ok)))

    poses = np.array(poses) if poses else np.zeros((0, 4, 4))
    pose_t = np.array(pose_t)
    np.savez_compressed(out_dir / "poses.npz", poses=poses, t=pose_t)
    np.savez_compressed(out_dir / "keyframes.npz", t=np.array(keyframes))
    np.savez_compressed(out_dir / "gps.npz",
                        data=np.array(gps_rows) if gps_rows else np.zeros((0, 4)))
    np.savez_compressed(out_dir / "twist.npz",
                        data=np.array(twist_rows) if twist_rows else np.zeros((0, 4)))

    # QC: teleport check on the KISS trajectory itself.
    qc: dict = {"site": site, "bag": rec["bag"], "n_poses": len(poses),
                "n_keyframes": len(keyframes), "n_gps": len(gps_rows),
                "elapsed_s": round(time.time() - t0, 1), "problems": []}
    if len(poses) > 10:
        xy = poses[:, :2, 3]
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.clip(np.diff(pose_t), 1e-6, None)
        speed = d / dt
        qc["path_m"] = round(float(d.sum()), 1)
        qc["speed_median"] = round(float(np.median(speed)), 2)
        qc["speed_p99"] = round(float(np.percentile(speed, 99)), 2)
        qc["speed_max"] = round(float(speed.max()), 2)
        qc["frac_over_limit"] = round(float((speed > SPEED_LIMIT).mean()), 4)
        if qc["frac_over_limit"] > 0.02:
            qc["problems"].append(f"{qc['frac_over_limit']:.1%} of frames over "
                                  f"{SPEED_LIMIT} m/s")
    gps = np.array(gps_rows) if gps_rows else np.zeros((0, 4))
    qc["gps_fix_frac"] = round(float(gps[:, 3].mean()), 3) if len(gps) else 0.0
    qc["ok"] = not qc["problems"]
    json.dump(qc, open(done_marker, "w"), indent=1)
    return qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=str, default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    manifest = json.load(open(P0_MANIFEST))
    jobs = [r for r in manifest if r["ok"]
            and (not args.sites or r["site"] in args.sites.split(","))]
    print(f"{len(jobs)} bags, {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_bag, r): r for r in jobs}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                qc = fut.result()
            except Exception as exc:
                qc = {"site": r["site"], "bag": r["bag"], "ok": False,
                      "problems": [f"{type(exc).__name__}: {exc}"]}
            results.append(qc)
            if qc.get("skipped"):
                print(f"[skip] {qc['site']:24s} {qc['bag']}", flush=True)
            else:
                tag = "ok " if qc.get("ok") else "BAD"
                print(f"[{tag}] {qc['site']:24s} {qc['bag']:44s} "
                      f"path {qc.get('path_m', '?'):>7} m  "
                      f"v_med {qc.get('speed_median', '?')}  "
                      f"v_max {qc.get('speed_max', '?')}  "
                      f"gps {qc.get('gps_fix_frac', '?')}  "
                      f"({qc.get('elapsed_s', '?')}s)"
                      + ("" if qc.get("ok") else f"  <- {qc.get('problems')}"),
                      flush=True)

    bad = [q for q in results if not q.get("ok") and not q.get("skipped")]
    print(f"\n{len(results) - len(bad)}/{len(results)} bags ok")
    json.dump(results, open(OUT_ROOT / "p1_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
