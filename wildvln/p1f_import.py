#!/usr/bin/env python3
"""Import production FAST-LIO odometry into the pipeline pose format.

Reads p1_fastlio/<rec>.bag (/Odometry from the merged-recording runs) and
re-expresses the poses at every original chunk's cloud/keyframe timestamps:

    p1f/<site>/<chunk>/poses_repaired.npz   (t, poses, valid, seg_id)
    p2bf/<site>/<chunk>/index.npz           (t, pose, seg_id, valid)
                        keyframes -> symlink to the P2b jpg dir

All chunks of a recording share ONE gravity-consistent frame (seg_id = 0
everywhere covered): downstream stages run unchanged via env overrides
WILDVLN_P1_ROOT / WILDVLN_P2B_ROOT.

Clock nuance: FAST-LIO stamps /Odometry with the lidar header time. For UMD
the Ouster stamps headers with sensor UPTIME, while our indices use
bag-receive epoch; the importer extracts (bag_time, header_time) pairs for
the cloud topic from the merged bag and maps odometry into bag time. ZED2
headers are epoch already (velodyne transport delay ~10-50 ms, ignored).

Usage:
    python -m wildvln.p1f_import [--recs AU,GTown,...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

P0_MANIFEST = Path("/data/patelm/ticvla/wildvln/p0/manifest.json")
P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")
P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
RUN_ROOT = Path("/data/patelm/ticvla/wildvln/p1_fastlio")
MERGED = Path("/data/patelm/ticvla/GND_merged")
P1F_ROOT = Path("/data/patelm/ticvla/wildvln/p1f")
P2BF_ROOT = Path("/data/patelm/ticvla/wildvln/p2bf")

RECS = {
    "AU": "AU", "GTown": "GTown", "UDC": "UDC",
    "UMD_map1_2_lot9": "UMD_map1_2_lot9",
    "UMD_map1_1_trail": "UMD_map1_1_trail",
    "UMD_map2_1_dininghall": "UMD_map2_1_dininghall",
}
MAX_GAP_S = 0.3          # odometry must bracket a query time this tightly

# Trust only the first N seconds of a run's odometry (relative to its first
# message). dininghall: FAST-LIO diverges catastrophically at ~19.8 min
# (robot enters the dining hall building; z runs away km-scale) — both the
# r1.5 and r0.75 runs break at the same spot. Keep the clean outdoor part.
ODOM_TRUST_S = {"UMD_map2_1_dininghall": 1185.0}


def load_odom(rec):
    from rosbags.highlevel import AnyReader
    ts, xyz, quat = [], [], []
    with AnyReader([RUN_ROOT / f"{rec}.bag"]) as r:
        conns = [c for c in r.connections if c.topic == "/Odometry"]
        for conn, ns, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            xyz.append([p.x, p.y, p.z])
            quat.append([q.x, q.y, q.z, q.w])
    ts, xyz, quat = np.array(ts), np.array(xyz), np.array(quat)
    order = np.argsort(ts)
    return ts[order], xyz[order], quat[order]


def uptime_to_bagtime(rec, cloud_topic):
    """(header_t -> bag_t) mapping from the merged bag's cloud messages."""
    from rosbags.highlevel import AnyReader
    ht, bt = [], []
    with AnyReader([MERGED / f"{rec}.bag"]) as r:
        conns = [c for c in r.connections if c.topic == cloud_topic]
        for conn, ns, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            ht.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            bt.append(ns * 1e-9)
    ht, bt = np.array(ht), np.array(bt)
    order = np.argsort(ht)
    return ht[order], bt[order]


def make_sampler(ts, xyz, quat):
    slerp = Slerp(ts, Rotation.from_quat(quat))

    def sample(tq):
        tq = np.asarray(tq, float)
        idx = np.searchsorted(ts, tq).clip(1, len(ts) - 1)
        gap = np.maximum(np.abs(ts[idx] - tq), np.abs(ts[idx - 1] - tq))
        ok = (tq >= ts[0]) & (tq <= ts[-1]) & (gap < MAX_GAP_S)
        tc = tq.clip(ts[0], ts[-1])
        R = slerp(tc).as_matrix()
        p = np.stack([np.interp(tc, ts, xyz[:, k]) for k in range(3)], 1)
        P = np.tile(np.eye(4), (len(tq), 1, 1))
        P[:, :3, :3] = R
        P[:, :3, 3] = p
        return P, ok
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", default="")
    args = ap.parse_args()
    manifest = json.load(open(P0_MANIFEST))

    from wildvln.rigs import rig_for_site
    for rec, site in RECS.items():
        if args.recs and rec not in args.recs.split(","):
            continue
        run_bag = RUN_ROOT / f"{rec}.bag"
        if not run_bag.exists():
            print(f"{rec}: no production run yet, skipping", flush=True)
            continue
        ts, xyz, quat = load_odom(rec)
        if rec in ODOM_TRUST_S:
            keep = ts - ts[0] <= ODOM_TRUST_S[rec]
            ts, xyz, quat = ts[keep], xyz[keep], quat[keep]
            print(f"{rec}: trusting first {ODOM_TRUST_S[rec]:.0f}s only "
                  f"({keep.sum()}/{len(keep)} poses)", flush=True)
        rig = rig_for_site(site)
        if rig.time_base == "bag":     # ouster: odom is in sensor uptime
            ht, bt = uptime_to_bagtime(rec, rig.cloud_topic)
            ts = np.interp(ts, ht, bt)
            order = np.argsort(ts)
            ts, xyz, quat = ts[order], xyz[order], quat[order]
        keep = np.concatenate([[True], np.diff(ts) > 1e-4])
        ts, xyz, quat = ts[keep], xyz[keep], quat[keep]
        sample = make_sampler(ts, xyz, quat)
        print(f"{rec}: {len(ts)} odom poses "
              f"[{ts[0]:.1f}..{ts[-1]:.1f}]", flush=True)

        chunks = [r for r in manifest if r["site"] == site]
        for r in sorted(chunks, key=lambda x: x["bag"]):
            stem = Path(r["path"]).stem
            old = np.load(P1_ROOT / site / stem / "poses_repaired.npz")
            P, ok = sample(old["t"])
            d1 = P1F_ROOT / site / stem
            d1.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d1 / "poses_repaired.npz", t=old["t"],
                                poses=P, valid=ok,
                                seg_id=np.where(ok, 0, -1).astype(np.int16))

            idx = np.load(P2B_ROOT / site / stem / "index.npz")
            Pk, okk = sample(idx["t"])
            d2 = P2BF_ROOT / site / stem
            d2.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d2 / "index.npz", t=idx["t"], pose=Pk,
                                seg_id=np.where(okk, 0, -1).astype(np.int16),
                                valid=okk)
            link = d2 / "keyframes"
            if not link.exists():
                link.symlink_to(P2B_ROOT / site / stem / "keyframes")
            print(f"  {stem}: clouds {int(ok.sum())}/{len(ok)} "
                  f"keyframes {int(okk.sum())}/{len(okk)}", flush=True)


if __name__ == "__main__":
    main()
