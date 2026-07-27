#!/usr/bin/env python3
"""P1c: segment trajectories at long KISS-ICP dropouts (supersedes P1b).

P1b bridged every dropout with twist dead-reckoning and re-anchored the tail.
That is fine for sub-second glitches, but across a 15-20 s dropout the
dead-reckoned heading carries several degrees of error, so the re-anchored
tail is rotated relative to the head -- visually wrong maps (user-flagged).

New policy, from the original KISS poses:
  - gap <= SHORT_GAP_S : twist-bridge in place (valid=False for mapping)
  - gap  > SHORT_GAP_S : hard segment boundary; the gap's frames are dropped
    and each side becomes an independent sub-dataset. No re-anchoring ever.

Segments shorter than MIN_SEG_S or MIN_SEG_M are discarded. Output overwrites
poses_repaired.npz (adding a seg_id array; -1 = dropped frame) and writes
segments.json. Windows must never span a segment boundary; P2a builds one map
per segment.

Usage:
    python -m wildvln.p1c_segment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wildvln.p1b_repair import dead_reckon, rot2, yaw_of

P1_ROOT = Path("/data/patelm/ticvla/wildvln/p1")

MISMATCH_MS = 1.0
MIN_RUN = 3
MARGIN = 5
SHORT_GAP_S = 2.0
MIN_SEG_S = 30.0
MIN_SEG_M = 20.0


def detect_runs(v_kiss, v_tw):
    suspect = np.abs(v_kiss - v_tw) > MISMATCH_MS
    runs, i = [], 0
    while i < len(suspect):
        if suspect[i]:
            j = i
            while j < len(suspect) and suspect[j]:
                j += 1
            if j - i >= MIN_RUN:
                runs.append((max(0, i - MARGIN), min(len(suspect), j + MARGIN)))
            i = j
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a <= merged[-1][1] + MARGIN:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def process(bag_dir: Path) -> dict:
    z = np.load(bag_dir / "poses.npz")
    poses, t = z["poses"].copy(), z["t"]
    twist = np.load(bag_dir / "twist.npz")["data"]
    n = len(poses)
    qc = {"site": bag_dir.parent.name, "bag": bag_dir.name, "n": n,
          "short_bridges": 0, "long_gaps": 0, "segments": []}
    if n < 10 or len(twist) < 10:
        np.savez_compressed(bag_dir / "poses_repaired.npz", poses=poses, t=t,
                            valid=np.ones(n, bool), seg_id=np.zeros(n, np.int16))
        qc["segments"] = [{"seg": 0, "frames": [0, n - 1]}]
        return qc

    xy = poses[:, :2, 3]
    dt = np.clip(np.diff(t), 1e-6, None)
    v_kiss = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    v_tw = np.interp(t[1:], twist[:, 0], np.hypot(twist[:, 1], twist[:, 2]))
    runs = detect_runs(v_kiss, v_tw)

    valid = np.ones(n, bool)
    seg_id = np.zeros(n, np.int16)
    boundaries = []          # frame ranges that are hard cuts
    for a, b in runs:
        dur = float(t[min(b, n - 1)] - t[a])
        if dur <= SHORT_GAP_S and a > 0:
            # in-place twist bridge, no re-anchor (heading error negligible)
            anchor = poses[a - 1]
            dr = dead_reckon(t[a:b + 1], twist, t[a - 1],
                             anchor[0, 3], anchor[1, 3], yaw_of(anchor[:3, :3]))
            for k, (x, y, yaw) in zip(range(a, b + 1), dr):
                P = np.eye(4)
                P[:2, :2] = rot2(yaw)
                P[0, 3], P[1, 3] = x, y
                P[2, 3] = anchor[2, 3]
                poses[k] = P
            valid[a:b + 1] = False
            qc["short_bridges"] += 1
        else:
            boundaries.append((a, min(b, n - 1)))
            qc["long_gaps"] += 1

    # segments = spans between hard cuts
    cuts = [(-1, -1)] + boundaries + [(n, n)]
    seg = 0
    for (pa, pb), (na, nb) in zip(cuts[:-1], cuts[1:]):
        i0, i1 = pb + 1, na - 1
        if i1 <= i0:
            continue
        dur = float(t[i1] - t[i0])
        path = float(np.linalg.norm(
            np.diff(poses[i0:i1 + 1, :2, 3], axis=0), axis=1).sum())
        if dur < MIN_SEG_S or path < MIN_SEG_M:
            seg_id[i0:i1 + 1] = -1
            continue
        seg_id[i0:i1 + 1] = seg
        qc["segments"].append({"seg": seg, "frames": [int(i0), int(i1)],
                               "dur_s": round(dur, 1), "path_m": round(path, 1),
                               "t0": float(t[i0]), "t1": float(t[i1])})
        seg += 1
    for a, b in boundaries:
        seg_id[a:b + 1] = -1

    kept = float((seg_id >= 0).mean())
    qc["frac_kept"] = round(kept, 3)
    np.savez_compressed(bag_dir / "poses_repaired.npz",
                        poses=poses, t=t, valid=valid, seg_id=seg_id)
    json.dump(qc, open(bag_dir / "segments.json", "w"), indent=1)
    return qc


def main() -> None:
    results = []
    for site_dir in sorted(P1_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        for bag_dir in sorted(site_dir.iterdir()):
            if not (bag_dir / "poses.npz").exists():
                continue
            qc = process(bag_dir)
            results.append(qc)
            segs = qc["segments"]
            print(f"{qc['site']:24s} {qc['bag']:44s} "
                  f"segs {len(segs)}  short-bridges {qc['short_bridges']}  "
                  f"long-gaps {qc['long_gaps']}  kept {qc.get('frac_kept', 1.0):.0%}",
                  flush=True)
    json.dump(results, open(P1_ROOT / "p1c_summary.json", "w"), indent=1)
    multi = [r for r in results if len(r["segments"]) > 1]
    print(f"\n{len(results)} bags -> {sum(len(r['segments']) for r in results)} segments "
          f"({len(multi)} bags split)")


if __name__ == "__main__":
    main()
