#!/usr/bin/env python3
"""GrandTour language farm: run the GND P4 farm over the GrandTour
index tree (p2b/grandtour/<mission>/) built by gt_p2index.

Reuses p4_farm.mine + annotate_episode wholesale; only the rig lookup
is adapted (one shared rig — same physical sensor box across missions;
calibration spread is checked and logged at startup).

Usage: python -m wildvln.gt_farm [--workers 8] [--port 8118] [--limit N]
"""

from __future__ import annotations

import os

os.environ.setdefault("WILDVLN_P2B_ROOT",
                      "/data/patelm/ticvla/grandtour/p2b")

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import time                                                # noqa: E402
import traceback                                           # noqa: E402
from concurrent.futures import (ThreadPoolExecutor,        # noqa: E402
                                as_completed)
from pathlib import Path                                   # noqa: E402

import numpy as np                                         # noqa: E402

import wildvln.p4_farm as farm                             # noqa: E402

P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
OUT = Path("/data/patelm/ticvla/grandtour/p4/farm")
BASE_HEIGHT_M = 0.55        # ANYmal trunk height while walking


class GtRig:
    def __init__(self, rig_json):
        r = json.load(open(rig_json))
        self.intrinsics = (r["fx"], r["fy"], r["cx"], r["cy"])
        self.image_size = (r["width"], r["height"])
        self.T_cam_lidar = np.array(r["T_cam_base"])
        self.lidar_height_m = BASE_HEIGHT_M


def setup_rig():
    rigs = sorted(P2B.glob("*/rig.json"))
    fx = [json.load(open(p))["fx"] for p in rigs]
    print(f"rig spread across {len(rigs)} missions: "
          f"fx {min(fx):.1f}..{max(fx):.1f} "
          f"({100*(max(fx)-min(fx))/min(fx):.2f}%)")
    shared = GtRig(rigs[0])
    farm.rig_for_site = lambda site: shared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=8118)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--missions", default="",
                    help="comma-separated subset")
    args = ap.parse_args()

    farm.OUT = OUT
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    setup_rig()

    missions = (args.missions.split(",") if args.missions else
                sorted(d.name for d in P2B.iterdir() if d.is_dir()))
    eps = []
    for m in missions:
        try:
            eps += farm.mine("grandtour", m)
        except Exception as e:
            print(f"mine {m}: FAIL {e}")
    kinds = [e["kind"] for e in eps]
    print(f"{len(eps)} episodes ({kinds.count('turn')} turn / "
          f"{kinds.count('continue')} continue) "
          f"from {len(missions)} missions")
    if args.limit:
        eps = eps[:args.limit]

    def work(e):
        ep_id = (f"gt_{e['bag']}_{e['kind']}_{int(e['s0']):04d}")
        ck = OUT / f"{ep_id}.json"
        if ck.exists():
            return f"{ep_id}: cached"
        t0 = time.time()
        try:
            r = farm.annotate_episode(e, ep_id, args.port)
        except Exception:
            return f"{ep_id}: EXC {traceback.format_exc()[-200:]}"
        if r is None:
            return f"{ep_id}: skipped (detector/frames)"
        json.dump(r, open(ck, "w"))
        return (f"{ep_id}: {len(r['steps'])} steps "
                f"seq_ok={r['seq_ok']} {time.time()-t0:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, e) for e in eps]
        for f in as_completed(futs):
            print(f.result(), flush=True)
    n = len(list(OUT.glob("gt_*.json")))
    print(f"GT_FARM_DONE {n} episode files")


if __name__ == "__main__":
    main()
