#!/usr/bin/env python3
"""P6 sliced eval: straight vs turn steps, plus turn-execution metrics.

Reads per-row prediction dumps (p6_eval --dump / p6_eval_bev --dump) and
splits by the GT trace's net heading change:
  straight  |net| < 20 deg
  turn      |net| >= 20 deg   (sharp subset: >= 45 deg)

Turn metrics (on the turn slice):
  turn-exec   pred turns the right way with at least a third of the GT
              magnitude (the failure mode we saw: CoT says "mid-turn
              right", trace goes straight)
  head-err    |net_pred - net_gt| in degrees
False-turn rate (on the straight slice): |net_pred| >= 20 deg.

Usage: python -m wildvln.p6_slice dump1.jsonl [dump2.jsonl ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from wildvln.p6_eval import resample

TURN_DEG = 20.0
SHARP_DEG = 45.0


def net_heading(pts):
    pts = resample(np.asarray(pts, float))
    if pts is None:
        return None
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.degrees(ang[-1] - ang[0]))


def turn_center_s(pts):
    """Arc-length (m) where half the total heading change is done —
    'where along the trace the turn happens'."""
    pts = resample(np.asarray(pts, float))
    if pts is None:
        return None
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs(np.diff(ang))
    if turn.sum() < 1e-6:
        return None
    s = np.cumsum(np.linalg.norm(d, axis=1))[:-1]
    cum = np.cumsum(turn)
    return float(np.interp(cum[-1] / 2, cum, s))


def score(path):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for split in ("val", "test_site"):
        sub = [r for r in rows if r["split"] == split]
        for r in sub:
            r["nh_gt"] = net_heading(r["gt"])
            r["nh_pred"] = net_heading(r["pred"])
        n_degen = sum(1 for r in sub if r["ade"] > 50)
        if n_degen:
            print(f"  {split}: dropped {n_degen} degenerate rows (ADE>50)")
        sub = [r for r in sub if r["ade"] <= 50
               and r["nh_gt"] is not None and r["nh_pred"] is not None]
        straight = [r for r in sub if abs(r["nh_gt"]) < TURN_DEG]
        turn = [r for r in sub if abs(r["nh_gt"]) >= TURN_DEG]
        sharp = [r for r in turn if abs(r["nh_gt"]) >= SHARP_DEG]

        def af(rs):
            return (float(np.mean([r["ade"] for r in rs])),
                    float(np.mean([r["fde"] for r in rs]))) if rs else (
                float("nan"), float("nan"))

        def exec_rate(rs):
            ok = [np.sign(r["nh_pred"]) == np.sign(r["nh_gt"])
                  and abs(r["nh_pred"]) >= abs(r["nh_gt"]) / 3 for r in rs]
            return float(np.mean(ok)) if ok else float("nan")

        head_err = (float(np.mean([abs(r["nh_pred"] - r["nh_gt"])
                                   for r in turn]))
                    if turn else float("nan"))
        # placement: among executed turns, |arc-pos of turn center - GT|
        placed = []
        for r in turn:
            if not (np.sign(r["nh_pred"]) == np.sign(r["nh_gt"])
                    and abs(r["nh_pred"]) >= abs(r["nh_gt"]) / 3):
                continue
            cp, cg = turn_center_s(r["pred"]), turn_center_s(r["gt"])
            if cp is not None and cg is not None:
                placed.append(abs(cp - cg))
        place_err = float(np.mean(placed)) if placed else float("nan")
        false_turn = (float(np.mean([abs(r["nh_pred"]) >= TURN_DEG
                                     for r in straight]))
                      if straight else float("nan"))
        out[split] = {
            "straight": {"n": len(straight), "ade_fde": af(straight),
                         "false_turn": false_turn},
            "turn": {"n": len(turn), "ade_fde": af(turn),
                     "exec": exec_rate(turn), "head_err_deg": head_err,
                     "place_err_m": place_err, "n_placed": len(placed)},
            "sharp": {"n": len(sharp), "ade_fde": af(sharp),
                      "exec": exec_rate(sharp)},
        }
    return out


def main() -> None:
    for path in sys.argv[1:]:
        print(f"=== {path}")
        res = score(path)
        for split, s in res.items():
            st, tu, sh = s["straight"], s["turn"], s["sharp"]
            print(f"  {split:9s} straight n={st['n']:3d} "
                  f"ADE {st['ade_fde'][0]:.2f} FDE {st['ade_fde'][1]:.2f} "
                  f"false-turn {st['false_turn']*100:.0f}%")
            print(f"  {'':9s} turn     n={tu['n']:3d} "
                  f"ADE {tu['ade_fde'][0]:.2f} FDE {tu['ade_fde'][1]:.2f} "
                  f"exec {tu['exec']*100:.0f}% "
                  f"head-err {tu['head_err_deg']:.0f} deg "
                  f"place-err {tu['place_err_m']:.1f} m "
                  f"(n={tu['n_placed']})")
            print(f"  {'':9s} sharp    n={sh['n']:3d} "
                  f"ADE {sh['ade_fde'][0]:.2f} FDE {sh['ade_fde'][1]:.2f} "
                  f"exec {sh['exec']*100:.0f}%")
        print(json.dumps(res))


if __name__ == "__main__":
    main()
