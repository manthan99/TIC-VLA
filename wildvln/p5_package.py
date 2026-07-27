#!/usr/bin/env python3
"""P5: package the annotation farm into a training-ready sample table.

One row per (chained step x goal-prompt variant) plus a memory-free T0
duplicate per step (mode="t0": no memory in/out, no CoT — plain
instruction->trace, trains the standalone mode and doubles as the
kinematic-shortcut breaker control).

Excludes vehicle-anchor-flagged episodes (p4/farm/_flagged.json).

Splits (episode-level, never step-level — chains must not straddle):
  test_site — ALL GTown episodes (held-out site)
  val       — 10% of remaining episodes (seeded)
  train     — rest

Output: p5/samples.parquet + p5/stats.json. Images stay as references
(keyframe path + overlay path); the SFT harness formats prompts/targets.

Usage: python -m wildvln.p5_package
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FARM = Path("/data/patelm/ticvla/wildvln/p4/farm")
P2B = Path("/data/patelm/ticvla/wildvln/p2bf")
OUT = Path("/data/patelm/ticvla/wildvln/p5")
HELD_OUT_SITE = "GTown"
VAL_FRAC = 0.10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(p.read_text()) for p in sorted(FARM.glob("*.json"))
           if not p.name.startswith("_")]
    eps = [e for e in eps if not e.get("skipped")]
    flagged = set(json.load(open(FARM / "_flagged.json")))
    kept = [e for e in eps if e["ep_id"] not in flagged]

    rng = np.random.default_rng(42)
    rows = []
    for e in kept:
        if e["site"] == HELD_OUT_SITE:
            split = "test_site"
        else:
            split = "val" if rng.random() < VAL_FRAC else "train"
        kf_dir = str(P2B / e["site"] / e["bag"] / "keyframes")
        for si, st in enumerate(e["steps"]):
            base = dict(
                ep_id=e["ep_id"], site=e["site"], bag=e["bag"],
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
                                 memory_out=json.dumps(st["target"]["memory"]),
                                 cot=st["target"]["cot"]))
            # memory-free T0 duplicate (canonical instruction only)
            rows.append(dict(base, mode="t0", variant=0,
                             instruction=e["instruction_variants"][0],
                             memory_in=None, memory_out=None, cot=None))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "samples.parquet")
    stats = {
        "episodes_total": len(eps), "episodes_flagged": len(flagged),
        "episodes_kept": len(kept),
        "samples": len(df),
        "by_mode": df["mode"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
        "by_kind": df["kind"].value_counts().to_dict(),
        "by_site": df["site"].value_counts().to_dict(),
        "episodes_by_split": {s: int(df[df.split == s]["ep_id"].nunique())
                              for s in df.split.unique()},
    }
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
