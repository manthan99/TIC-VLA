#!/usr/bin/env python3
"""P0: metadata-only inventory of the MVP bags, with rig-contract enforcement.

Deliberately reads NO bulk message data: rosbag1 chunks interleave topics, so
pulling even one full topic decompresses most of the bag. Everything here comes
from the connection records and index (counts, times), a single camera_info
message, and a GPS sample from the earliest chunks. The one full-bag streaming
read happens later, in P1, where it feeds KISS-ICP, the keyframe index, and the
GPS/twist tracks in a single pass.

Usage:
    python -m wildvln.p0_inventory                 # the 6 MVP sites
    python -m wildvln.p0_inventory --sites AU,UDC  # subset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from wildvln.rigs import check_camera_info, rig_for_site

GND_RAW = "/data/patelm/ticvla/GND_raw"
OUT_DIR = Path("/data/patelm/ticvla/wildvln/p0")

MVP_SITES = {
    "AU": f"{GND_RAW}/AU/AU_chunk*.bag",
    "GTown": f"{GND_RAW}/GTown/GTown_chunk*.bag",
    "UDC": f"{GND_RAW}/UDC/UDC_chunk*.bag",
    "UMD_map1_2_lot9": f"{GND_RAW}/UMD/map1/UMD_map1_2_lot9_chunk*.bag",
    "UMD_map2_1_dininghall": f"{GND_RAW}/UMD/map2/UMD_map2_1_dininghall_chunk*.bag",
    "UMD_map1_1_trail": f"{GND_RAW}/UMD/map1/UMD_map1_1_trail_chunk*.bag",
}

GPS_SAMPLE_MSGS = 300


def inspect_bag(bag_path: str, site: str) -> dict:
    rig = rig_for_site(site)
    rec: dict = {
        "site": site,
        "rig": rig.name,
        "bag": os.path.basename(bag_path),
        "path": bag_path,
        "size_gb": round(os.path.getsize(bag_path) / 1e9, 3),
        "ok": True,
        "problems": [],
    }
    try:
        with AnyReader([Path(bag_path)]) as reader:
            rec["duration_s"] = round((reader.end_time - reader.start_time) * 1e-9, 1)
            rec["start_time"] = reader.start_time * 1e-9
            topics = {c.topic: c.msgcount for c in reader.connections}
            rec["topics"] = topics

            # Required topics present, with sane counts.
            for role, topic in [("image", rig.image_topic), ("cloud", rig.cloud_topic),
                                ("odom", rig.odom_topic), ("gps", rig.gps_topic)]:
                count = topics.get(topic, 0)
                rec[f"n_{role}"] = count
                if count == 0:
                    rec["problems"].append(f"missing {role} topic {topic}")

            dur = max(rec["duration_s"], 1e-6)
            rec["image_hz"] = round(rec["n_image"] / dur, 2)
            rec["cloud_hz"] = round(rec["n_cloud"] / dur, 2)

            # Rig contract on the first camera_info message.
            ci = [c for c in reader.connections if c.topic == rig.camera_info_topic]
            if not ci:
                rec["problems"].append(f"missing camera_info {rig.camera_info_topic}")
            else:
                for conn, _, raw in reader.messages(connections=ci):
                    msg = reader.deserialize(raw, conn.msgtype)
                    err = check_camera_info(
                        rig,
                        np.array(msg.K).reshape(3, 3),
                        np.array(msg.P).reshape(3, 4),
                        np.array(msg.D),
                        msg.width, msg.height,
                    )
                    if err:
                        rec["problems"].append(f"camera contract: {err}")
                    break

            # GPS quality sample from the earliest chunks only (cheap).
            gps = [c for c in reader.connections if c.topic == rig.gps_topic]
            fixes = []
            if gps:
                for conn, _, raw in reader.messages(connections=gps):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if hasattr(msg, "fixType"):
                        fixes.append(int(msg.fixType) >= 3)
                    elif hasattr(msg, "status"):
                        fixes.append(int(msg.status.status) >= 0)
                    if len(fixes) >= GPS_SAMPLE_MSGS:
                        break
            rec["gps_fix_frac_sample"] = round(float(np.mean(fixes)), 3) if fixes else None
    except Exception as exc:  # damaged bag, unreadable index, ...
        rec["problems"].append(f"unreadable: {type(exc).__name__}: {exc}")

    rec["ok"] = not rec["problems"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=str, default="",
                    help="comma-separated subset of MVP sites")
    args = ap.parse_args()

    sites = {s: g for s, g in MVP_SITES.items()
             if not args.sites or s in args.sites.split(",")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for site, pattern in sites.items():
        bags = sorted(glob.glob(pattern),
                      key=lambda p: int(re.search(r"chunk(\d+)", p).group(1)))
        for bag in bags:
            rec = inspect_bag(bag, site)
            records.append(rec)
            flag = "ok " if rec["ok"] else "BAD"
            print(f"[{flag}] {site:24s} {rec['bag']:44s} "
                  f"{rec.get('duration_s', 0):7.1f}s  img {rec.get('n_image', 0):5d} "
                  f"({rec.get('image_hz', 0):.1f} Hz)  cloud {rec.get('n_cloud', 0):5d}  "
                  f"gps~{rec.get('gps_fix_frac_sample')}"
                  + ("" if rec["ok"] else f"  <- {rec['problems']}"), flush=True)

    out = OUT_DIR / "manifest.json"
    json.dump(records, open(out, "w"), indent=1)

    n_ok = sum(r["ok"] for r in records)
    total_min = sum(r.get("duration_s", 0) for r in records) / 60
    print(f"\n{n_ok}/{len(records)} bags pass contracts; "
          f"{total_min:.1f} min total -> {out}")


if __name__ == "__main__":
    main()
