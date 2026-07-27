#!/usr/bin/env python3
"""Create a recording-level held-out eval split for DynaNav.

DynaNav folders are overlapping 20s windows over ~105 underlying recordings
(e.g. hospital_10_0s_20s, hospital_10_10s_30s share frames), so the split must
hold out whole recordings. This script moves all window folders of the selected
eval recordings from DynaNav_json/ to DynaNav_json_eval/ (same parent, so the
relative ../../DynaNav_data image paths keep resolving) and writes a manifest.

Selection is deterministic (seeded), stratified by scene type and robot type:
hospital/office/warehouse: 2 spot + 2 wheeled each; outdoor: 1 spot + 2 wheeled.

Usage:
    python scripts/make_eval_split.py [--dry-run]
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

EVAL_PER_SCENE = {
    "hospital": {"spot": 2, "wheeled": 2},
    "office": {"spot": 2, "wheeled": 2},
    "warehouse": {"spot": 2, "wheeled": 2},
    "outdoor": {"spot": 1, "wheeled": 2},
}
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/data/patelm/ticvla/dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.data_root) / "DynaNav" / "DynaNav_json"
    dst = Path(args.data_root) / "DynaNav" / "DynaNav_json_eval"
    manifest_path = Path(args.data_root) / "DynaNav" / "eval_split_manifest.json"

    if manifest_path.exists():
        raise SystemExit(f"Manifest already exists at {manifest_path}; split was already made.")

    folders_by_rec = defaultdict(list)
    for d in sorted(src.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(.+?)_\d+s_\d+s$", d.name)
        key = m.group(1) if m else d.name
        folders_by_rec[key].append(d.name)

    rng = random.Random(SEED)
    eval_recs = []
    for scene, quota in EVAL_PER_SCENE.items():
        spot = sorted(k for k in folders_by_rec if k.startswith(scene) and "spot" in k)
        wheeled = sorted(k for k in folders_by_rec if k.startswith(scene) and "spot" not in k)
        eval_recs += rng.sample(spot, quota["spot"]) + rng.sample(wheeled, quota["wheeled"])

    eval_folders = sorted(f for rec in eval_recs for f in folders_by_rec[rec])
    n_total = sum(len(v) for v in folders_by_rec.values())
    print(f"eval recordings ({len(eval_recs)}/{len(folders_by_rec)}): {sorted(eval_recs)}")
    print(f"moving {len(eval_folders)}/{n_total} window folders -> {dst}")

    if args.dry_run:
        return

    dst.mkdir(exist_ok=True)
    for name in eval_folders:
        (src / name).rename(dst / name)

    manifest = {
        "seed": SEED,
        "eval_recordings": sorted(eval_recs),
        "eval_folders": eval_folders,
        "train_recordings": sorted(k for k in folders_by_rec if k not in eval_recs),
        "created": "2026-07-16",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
