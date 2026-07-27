#!/usr/bin/env python3
"""GrandTour first-look: extract mission tars, QC poses/images, mine turns.

Per mission:
  - extract data/*.tar + images/hdr_front.tar into zarr trees (idempotent)
  - pose QC on dlio_map_odometry + cpt7_ie_tc_odometry: length, duration,
    speed, z-range/drift (our GND scar tissue), timestamps
  - hdr_front: frame count, fps, sample frames dumped as jpg
  - straight-run RDP turn mining (same detector as the GND farm) on the
    trajectory -> episode candidate count
  - top-down trajectory render

Usage: python -m wildvln.gt_explore [--mission 2024-11-04-10-57-34]
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/patelm/ticvla/grandtour/raw")
OUT = Path("/data/patelm/ticvla/grandtour/qc")

MISSIONS = sorted(d.name for d in ROOT.iterdir()
                  if d.is_dir() and d.name[0].isdigit())


def extract(mission: str):
    mdir = ROOT / mission
    for tar in sorted(mdir.rglob("*.tar")):
        marker = tar.with_suffix(".extracted")
        if marker.exists():
            continue
        with tarfile.open(tar) as tf:
            tf.extractall(path=tar.parent)
        marker.touch()
        print(f"  extracted {tar.relative_to(ROOT)}")


def pose_qc(z, name):
    ts = z["timestamp"][:]
    p = z["pose_pos"][:]
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    length = float(d.sum())
    dur = float(ts[-1] - ts[0])
    dt = np.diff(ts)
    speed = d / np.maximum(np.diff(ts), 1e-6)
    print(f"  {name}: n={len(ts)} dur={dur/60:.1f}min len={length:.0f}m "
          f"rate={1/np.median(dt):.0f}Hz "
          f"speed p50 {np.median(speed):.2f}m/s "
          f"z range {p[:, 2].min():.1f}..{p[:, 2].max():.1f} "
          f"({p[:, 2].max()-p[:, 2].min():.1f} m)")
    return ts, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="")
    args = ap.parse_args()
    import zarr
    from wildvln.p4_fullann import detect_turns

    missions = [args.mission] if args.mission else MISSIONS
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for m in missions:
        mdir = ROOT / m
        if not (mdir / "data").exists():
            print(f"== {m}: not downloaded yet, skip")
            continue
        print(f"== {m}")
        extract(m)
        zg = mdir / "data" / ".zgroup"
        if not zg.exists():
            zg.write_text('{"zarr_format": 2}')
        root = zarr.open_group(str(mdir / "data"), mode="r", zarr_format=2)
        topics = sorted(d.name for d in (mdir / "data").iterdir()
                        if d.is_dir() and (d / ".zgroup").exists())
        print(f"  topics: {topics}")

        info = {"topics": topics}
        for src in ("dlio_map_odometry", "cpt7_ie_tc_odometry",
                    "anymal_state_odometry"):
            if src in topics:
                try:
                    ts, p = pose_qc(root[src], src)
                    info[src] = {"n": len(ts),
                                 "len_m": float(np.linalg.norm(
                                     np.diff(p[:, :2], 0), axis=1).sum())}
                except Exception as e:
                    print(f"  {src}: FAIL {e}")

        # turn mining on dlio trajectory (resampled to ~0.5 m spacing)
        if "dlio_map_odometry" in topics:
            p = root["dlio_map_odometry"]["pose_pos"][:]
            xy = p[:, :2]
            d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(d)])
            si = np.arange(0, s[-1], 0.5)
            xyr = np.stack([np.interp(si, s, xy[:, 0]),
                            np.interp(si, s, xy[:, 1])], 1)
            turns = detect_turns(xyr, si, si[0], si[-1])
            print(f"  turns >=30deg: {len(turns)} over {s[-1]:.0f} m")
            info["turns"] = len(turns)

            # top-down render
            n = 900
            img = np.full((n, n, 3), 30, np.uint8)
            mn, mx = xy.min(0), xy.max(0)
            sc = (n - 40) / max(float((mx - mn).max()), 1e-6)
            px = ((xy - mn) * sc + 20).astype(int)
            for a, b in zip(px[:-1], px[1:]):
                cv2.line(img, tuple(a), tuple(b), (80, 200, 80), 2)
            cv2.circle(img, tuple(px[0]), 8, (255, 255, 255), -1)
            cv2.imwrite(str(OUT / f"{m}_traj.png"), img[::-1])

        # hdr_front frames
        img_dir = mdir / "images" / "hdr_front"
        if not img_dir.exists():
            cands = list((mdir / "images").glob("*")) \
                if (mdir / "images").exists() else []
            img_dir = cands[0] if cands else None
        if img_dir and img_dir.is_dir():
            frames = sorted(img_dir.rglob("*.jpg")) + \
                sorted(img_dir.rglob("*.jpeg")) + sorted(img_dir.rglob("*.png"))
            print(f"  hdr_front files: {len(frames)} at {img_dir.name}")
            info["n_frames"] = len(frames)
            for k in range(0, 8):
                if not frames:
                    break
                f = frames[int(k * (len(frames) - 1) / 7)]
                im = cv2.imread(str(f))
                if im is None:
                    continue
                im = cv2.resize(im, (640, int(640 * im.shape[0]
                                              / im.shape[1])))
                cv2.imwrite(str(OUT / f"{m}_f{k}.jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        summary[m] = info
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print("GTEXPLORE_DONE", OUT)


if __name__ == "__main__":
    main()
