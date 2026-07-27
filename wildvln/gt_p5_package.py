#!/usr/bin/env python3
"""P5 for GrandTour: package the gt_farm output into samples.parquet.

Same row contract as wildvln/p5_package.py. Splits are LOCATION-level
(mission -> location via the site's missions.json): held-out locations
= ARCHE (industrial, Wangen) + SBB (rail) — entire environments never
seen in training; val = 10% of remaining episodes.

Vehicle/person-anchored episodes (qc_anchor_bad) are excluded.

Usage: python -m wildvln.gt_p5_package
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FARM = Path("/data/patelm/ticvla/grandtour/p4/farm")
P2B = Path("/data/patelm/ticvla/grandtour/p2b/grandtour")
OUT = Path("/data/patelm/ticvla/grandtour/p5")
MISSIONS_JSON = Path("/tmp/claude-1003/-home-nvidia-patelm-ws-rsl-TIC-VLA"
                     "/7276802c-e3c3-471a-80e8-08a4e248fa6a/scratchpad"
                     "/gt_missions.json")
HELD_OUT = ("ARCHE", "SBB")
VAL_FRAC = 0.10


def location_map():
    ms = json.load(open(MISSIONS_JSON))
    return {m["date_time"]: m["name"].split(" - ")[0].strip()
            for m in ms}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loc = location_map()
    eps = [json.loads(p.read_text()) for p in sorted(FARM.glob("gt_*.json"))]
    kept = [e for e in eps if not e.get("qc_anchor_bad")]

    rng = np.random.default_rng(42)
    rows = []
    for e in kept:
        location = loc.get(e["bag"], "unknown")
        if location in HELD_OUT:
            split = "test_site"
        else:
            split = "val" if rng.random() < VAL_FRAC else "train"
        kf_dir = str(P2B / e["bag"] / "keyframes")
        for si, st in enumerate(e["steps"]):
            base = dict(
                ep_id=e["ep_id"], site=location, bag=e["bag"],
                kind=e["kind"], split=split, step=si,
                n_steps=len(e["steps"]),
                image=f"{kf_dir}/{st['kf']}",
                overlay=str(FARM / "img" / st["overlay"]),
                history=json.dumps(st["input"]["history"]),
                trace=json.dumps(st["target"]["trace"]),
                trace_len_m=st["target"]["trace_len_m"],
                turn_states=json.dumps(st["turn_states"]),
            )
            for vi, instr in enumerate(e["instruction_variants"]):
                rows.append(dict(base, mode="chained", variant=vi,
                                 instruction=instr,
                                 memory_in=json.dumps(st["input"]["memory"]),
                                 memory_out=json.dumps(
                                     st["target"]["memory"]),
                                 cot=st["target"]["cot"]))
            rows.append(dict(base, mode="t0", variant=0,
                             instruction=e["instruction_variants"][0],
                             memory_in=None, memory_out=None, cot=None))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "samples.parquet")
    stats = {
        "episodes_total": len(eps),
        "episodes_flagged": len(eps) - len(kept),
        "episodes_kept": len(kept), "samples": len(df),
        "by_mode": df["mode"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
        "by_kind": df["kind"].value_counts().to_dict(),
        "by_location": df["site"].value_counts().to_dict(),
        "episodes_by_split": {s: int(df[df.split == s]["ep_id"].nunique())
                              for s in df.split.unique()},
    }
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
