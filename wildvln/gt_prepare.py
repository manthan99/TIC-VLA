#!/usr/bin/env python3
"""Batch-prepare all downloaded GrandTour missions:
extract tars -> undistort hdr_front -> wavemap occupancy map.

Parallel across missions (process pool); each worker is sequential
within its mission. Idempotent: .extracted markers, existing _rect
frames and .wvmp maps are skipped.

Usage: python -m wildvln.gt_prepare [--workers 6] [--skip-wavemap]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/data/patelm/ticvla/grandtour/raw")
PY = sys.executable


def prepare(mission: str, skip_wavemap: bool) -> str:
    from wildvln.gt_explore import extract
    steps = []
    try:
        extract(mission)
        steps.append("extract")
        # undistort (skip missions without hdr_front images)
        if (ROOT / mission / "images" / "hdr_front").exists():
            r = subprocess.run(
                [PY, "-m", "wildvln.gt_undistort", "--mission", mission,
                 "--batch"], capture_output=True, text=True, timeout=7200)
            steps.append("undistort" if r.returncode == 0
                         else f"undistort-FAIL:{r.stderr[-200:]}")
        else:
            steps.append("no-hdr-front")
        if not skip_wavemap:
            wm = Path("/data/patelm/ticvla/grandtour/wavemap") \
                / f"{mission}.wvmp"
            if wm.exists():
                steps.append("wavemap-cached")
            elif (ROOT / mission / "data"
                  / "hesai_points_undistorted").exists():
                r = subprocess.run(
                    [PY, "-m", "wildvln.gt_wavemap", "--mission", mission],
                    capture_output=True, text=True, timeout=7200)
                steps.append("wavemap" if r.returncode == 0
                             else f"wavemap-FAIL:{r.stderr[-200:]}")
            else:
                steps.append("no-hesai")
    except Exception:
        steps.append("EXC:" + traceback.format_exc()[-300:])
    return f"{mission}: {', '.join(steps)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-wavemap", action="store_true")
    args = ap.parse_args()
    missions = sorted(d.name for d in ROOT.iterdir()
                      if d.is_dir() and d.name[0].isdigit())
    print(f"{len(missions)} missions, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(prepare, m, args.skip_wavemap): m
                for m in missions}
        for f in as_completed(futs):
            print(f.result(), flush=True)
    print("GT_PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
