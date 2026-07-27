#!/usr/bin/env python3
"""P2b-1: extract 2 Hz keyframe images + per-keyframe metadata.

Third and final bag read. Writes each keyframe as a JPEG (ZED2 topics are
already JPEG-compressed -> raw byte copy; UMD raw bgr8 -> encoded at q=92)
plus one index per bag with interpolated pose, segment id, and validity.
Everything downstream (depth, semantics, ViT features, voxel painting) reads
these files; bags are never opened again.

Output:
    p2b/<site>/<bag>/keyframes/<t_ns>.jpg
    p2b/<site>/<bag>/index.npz   t, pose (N,4,4), seg_id, valid

Usage:
    python -m wildvln.p2b_keyframes --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from wildvln.rigs import rig_for_site

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")


def interp_pose(poses, pose_t, t):
    """Nearest-pose lookup (10 Hz poses vs 2 Hz keyframes: <=50 ms off)."""
    i = int(np.clip(np.searchsorted(pose_t, t), 1, len(pose_t) - 1))
    if abs(pose_t[i - 1] - t) < abs(pose_t[i] - t):
        i -= 1
    return poses[i], i


def process_bag(site: str, bag_path: str) -> dict:
    from rosbags.highlevel import AnyReader

    rig = rig_for_site(site)
    stem = Path(bag_path).stem
    p1_dir = P1_ROOT / site / stem
    out_dir = OUT_ROOT / site / stem
    kf_dir = out_dir / "keyframes"
    if (out_dir / "index.npz").exists():
        return {"site": site, "bag": stem, "skipped": True}
    kf_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(p1_dir / "poses_repaired.npz")
    poses, pose_t, valid = z["poses"], z["t"], z["valid"]
    seg_id = z["seg_id"] if "seg_id" in z.files else np.zeros(len(pose_t), np.int16)
    want_t = np.load(p1_dir / "keyframes.npz")["t"]

    rows_t, rows_pose, rows_seg, rows_valid = [], [], [], []
    wi = 0
    t0 = time.time()
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == rig.image_topic]
        for conn, bag_ns, raw in reader.messages(connections=conns):
            t = bag_ns * 1e-9
            if wi >= len(want_t):
                break
            if t < want_t[wi] - 1e-4:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if hasattr(msg, "format"):                      # CompressedImage
                data = bytes(msg.data)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8) \
                        .reshape(msg.height, msg.width, -1)
                if msg.encoding == "rgb8":
                    arr = arr[:, :, ::-1]
                ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(arr),
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                data = enc.tobytes()
            (kf_dir / f"{int(t * 1e9)}.jpg").write_bytes(data)
            P, i = interp_pose(poses, pose_t, t)
            rows_t.append(t)
            rows_pose.append(P)
            rows_seg.append(seg_id[i])
            rows_valid.append(bool(valid[i]))
            wi += 1

    np.savez_compressed(out_dir / "index.npz",
                        t=np.array(rows_t), pose=np.array(rows_pose),
                        seg_id=np.array(rows_seg, np.int16),
                        valid=np.array(rows_valid, bool))
    return {"site": site, "bag": stem, "n_keyframes": len(rows_t),
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    manifest = json.load(open("/data/patelm/ticvla/wildvln/p0/manifest.json"))
    jobs = [(r["site"], r["path"]) for r in manifest if r["ok"]]
    print(f"{len(jobs)} bags", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_bag, s, p): (s, p) for s, p in jobs}
        for fut in as_completed(futures):
            s, p = futures[fut]
            try:
                m = fut.result()
                tag = "skip" if m.get("skipped") else "ok  "
                print(f"[{tag}] {m['site']:24s} {m['bag']:44s} "
                      f"{m.get('n_keyframes', '')} kf ({m.get('elapsed_s', '')}s)",
                      flush=True)
            except Exception as exc:
                print(f"[ERR ] {s}/{Path(p).stem}: {type(exc).__name__}: {exc}",
                      flush=True)


if __name__ == "__main__":
    main()
