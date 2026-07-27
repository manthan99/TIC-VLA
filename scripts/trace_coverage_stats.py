#!/usr/bin/env python3
"""How many samples actually yield a usable 10-point image-space trace?

Not every frame can produce one. The robot may be nearly stationary, the path
may leave the field of view immediately, or the future may be too short. This
samples frames at random and reports the yield, plus why the rest were dropped,
so we know how much training data the trace formulation really gives us.

For GND it also sweeps camera height, since that is still uncalibrated and it
is worth knowing whether the choice moves the yield at all.

Usage:
    python scripts/trace_coverage_stats.py --per-dataset 2500
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ticvla.data.projection import (
    future_offsets_in_window,
    get_camera,
    image_trace_from_sample,
)

DATASET_ROOTS = {
    "DynaNav": "/data/patelm/ticvla/dataset/DynaNav/DynaNav_json",
    "SCAND": "/data/patelm/ticvla/dataset/SCAND/SCAND_json",
    "GND": "/data/patelm/ticvla/dataset/GND/GND_json",
}


def _resolve(rel: str, sample_path: Path) -> Path:
    return (sample_path.parent / rel).resolve()


def examine(job):
    """Classify one frame. Returns (dataset, outcome, path_length_m, height_variant)."""
    dataset, path_str, heights = job
    path = Path(path_str)
    try:
        data = json.load(open(path, "r"))
    except Exception:
        return [(dataset, "unreadable", 0.0, None)]

    offsets = future_offsets_in_window(data)
    if len(offsets) < 2:
        return [(dataset, "no future", 0.0, None)]
    length = float(np.linalg.norm(np.diff(offsets[:, :2], axis=0), axis=1).sum())
    if length < 1.0:
        return [(dataset, "stationary", length, None)]

    rel = data.get("current", {}).get("img", "")
    img_path = _resolve(rel, path)
    if not img_path.exists():
        return [(dataset, "image missing", length, None)]
    try:
        size = Image.open(img_path).size
    except Exception:
        return [(dataset, "image unreadable", length, None)]

    camera = get_camera(dataset, image_size=size, recording=path.parent.name)
    out = []
    for h in (heights or [None]):
        cam = camera if h is None else replace(camera, camera_height_m=h)
        trace = image_trace_from_sample(data, cam)
        outcome = "ok" if trace is not None and len(trace) == 10 else "not visible"
        out.append((dataset, outcome, length, h))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    jobs = []
    for dataset, root in DATASET_ROOTS.items():
        r = Path(root)
        if not r.exists():
            print(f"{dataset}: {root} missing, skipped")
            continue
        rng = random.Random(args.seed)
        folders = [d for d in r.iterdir() if d.is_dir()]
        rng.shuffle(folders)
        heights = [0.5, 0.7, 1.0] if dataset == "GND" else None
        picked = 0
        for folder in folders:
            if picked >= args.per_dataset:
                break
            files = [p for p in folder.glob("*.json") if not p.name.startswith(".")]
            if not files:
                continue
            for p in rng.sample(files, min(2, len(files))):
                jobs.append((dataset, str(p), heights))
                picked += 1
    print(f"examining {len(jobs)} frames across {len(DATASET_ROOTS)} datasets\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(examine, jobs, chunksize=32):
            results.extend(chunk)

    by_ds = {}
    for dataset, outcome, length, h in results:
        by_ds.setdefault(dataset, []).append((outcome, length, h))

    for dataset, rows in by_ds.items():
        variants = sorted({h for _, _, h in rows if h is not None}) or [None]
        print(f"=== {dataset} ===")
        for h in variants:
            sel = [(o, l) for o, l, hh in rows if hh == h]
            counts = Counter(o for o, _ in sel)
            n = len(sel)
            ok = counts.get("ok", 0)
            tag = "" if h is None else f"  [camera height {h:.2f} m]"
            print(f"  yield {ok}/{n} = {100 * ok / max(n, 1):.1f}%{tag}")
            for outcome, c in counts.most_common():
                if outcome == "ok":
                    continue
                print(f"      dropped: {outcome:18s} {c:5d}  ({100 * c / max(n, 1):.1f}%)")
        lens = np.array([l for o, l, hh in rows if o == "ok" and hh == variants[0]])
        if len(lens):
            print(f"  path length of kept traces: median {np.median(lens):.1f} m, "
                  f"p10 {np.percentile(lens, 10):.1f}, p90 {np.percentile(lens, 90):.1f}")
        print()


if __name__ == "__main__":
    main()
