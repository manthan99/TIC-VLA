#!/usr/bin/env python3
"""DynaNav -> wildvln trace samples (instructions taken AS-IS).

Horizon decision (investigated 2026-07-27 over all 592 json windows,
23.5k samples): teleop speeds are ~0.45 m/s median (0.69 p90), so the
TIC-VLA 3 s action horizon covers only ~1.4 m; a 10 m trace would need
~22 s of future — geometrically available for 70% of samples but
stale in dynamic scenes. We cap DynaNav traces at DN_TRACE_M = 5 m
(88% of samples have that much path left; ~11 s at median speed) and
keep the "up to" phrasing so shorter tails still train.

Rows mirror the p5 schema (t0 mode: no farm memory/CoT — instruction
comes verbatim from the teleop annotation). Split: the recordings in
eval_split_manifest.json become split='test_site', the rest 'train'
(+10% val by episode).

Usage: python -m wildvln.dn_episodes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/data/patelm/ticvla/dataset/DynaNav")
OUT = Path("/data/patelm/ticvla/dnav/p5")

DN_TRACE_M = 5.0
TRACE_PTS = 10
STEP_EVERY = 10          # one sample per second (jsons are 10 Hz)
HIST_M = 5.0


def polyline(off_xy, budget, npts):
    """Resample cumulative future offsets to npts distance-spaced points."""
    p = np.vstack([[0.0, 0.0], off_xy])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = min(budget, s[-1])
    if L < 0.5:
        return None, 0.0
    t = np.linspace(L / npts, L, npts)
    out = np.stack([np.interp(t, s, p[:, k]) for k in (0, 1)], 1)
    return out, float(L)


def main() -> None:
    man = json.loads((DATA / "eval_split_manifest.json").read_text())
    eval_recs = set(man["eval_recordings"])
    rng = np.random.default_rng(0)
    rows = []
    windows = sorted((DATA / "DynaNav_json").iterdir()) + sorted(
        (DATA / "DynaNav_json_eval").iterdir())
    for wdir in windows:
        js = sorted(wdir.glob("*.json"))
        if not js:
            continue
        rec = "_".join(wdir.name.split("_")[:-2])
        scene = rec.split("_")[0]
        if rec in eval_recs:
            split = "test_site"
        else:
            split = "val" if rng.random() < 0.10 else "train"
        for si, jf in enumerate(js[::STEP_EVERY]):
            d = json.loads(jf.read_text())
            fu = d.get("future") or []
            if len(fu) < 5:
                continue
            off = np.array([f["offset"][:2] for f in fu])
            trace, tlen = polyline(off, DN_TRACE_M, TRACE_PTS)
            if trace is None:
                continue
            # history entries carry 'trajectory' relative to the WINDOW
            # start (not the current frame) — frame alignment is
            # ambiguous, so the prototype keeps history empty (t0-style)
            hxy = None
            instr_f = wdir / d.get("instruction_file", "missing")
            if not instr_f.exists():
                cand = list(wdir.glob("instruction*.txt"))
                instr_f = cand[0] if cand else None
            if instr_f is None:
                continue
            instr = instr_f.read_text().strip()
            img = str((jf.parent / d["current"]["img"]).resolve())
            rows.append(dict(
                ep_id=f"dn_{wdir.name}", site=scene, bag=rec,
                kind="dnav", split=split, step=si,
                n_steps=len(js[::STEP_EVERY]), image=img, overlay="",
                history=json.dumps(hxy.round(2).tolist()) if hxy is not None else "",
                trace=json.dumps(trace.round(2).tolist()),
                trace_len_m=round(tlen, 2), turn_states="[]",
                mode="t0", variant=0, instruction=instr,
                memory_in="", memory_out="", cot=""))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "samples.parquet")
    stats = {"samples": len(df),
             "by_split": df.split.value_counts().to_dict(),
             "by_scene": df.site.value_counts().to_dict(),
             "trace_len_median": float(df.trace_len_m.median()),
             "windows": len(windows)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("DN_EPISODES_DONE")


if __name__ == "__main__":
    main()
