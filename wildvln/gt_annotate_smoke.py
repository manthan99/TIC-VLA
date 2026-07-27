#!/usr/bin/env python3
"""GrandTour instruction-feasibility smoke: run the GND farm's stage-A
episode annotator (unchanged prompts, 122B server) on GrandTour turns.

Usage: python -m wildvln.gt_annotate_smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from wildvln.p4_farm import (EP_PROMPT, EP_SCHEMA, ROUTE_GT_TURNS,
                             ROUTE_GT_STRAIGHT, chat_retry, img_part_mem)
from wildvln.p4_fullann import build_bev, detect_turns

ROOT = Path("/data/patelm/ticvla/grandtour/raw")
OUT = Path("/data/patelm/ticvla/grandtour/qc/ann")
PORT = 8118
APPROACH_M, EXIT_M, FRAME_GAP_M = 22.0, 15.0, 3.5

MISSIONS = ["2024-11-04-10-57-34", "2024-10-01-11-29-55",
            "2024-10-01-12-00-49", "2024-11-14-15-22-43",
            "2024-11-18-12-05-01", "2024-12-03-13-26-40"]


def load(mission):
    import zarr
    mdir = ROOT / mission
    od = zarr.open_group(str(mdir / "data" / "dlio_map_odometry"),
                         mode="r", zarr_format=2)
    ts, pos = od["timestamp"][:], od["pose_pos"][:]
    hdr = zarr.open_group(str(mdir / "data" / "hdr_front"),
                          mode="r", zarr_format=2)
    hts = hdr["timestamp"][:]
    # resample trajectory at 0.5 m arc length, keep time mapping
    xy = pos[:, :2]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    si = np.arange(0, s[-1], 0.5)
    xyr = np.stack([np.interp(si, s, xy[:, 0]),
                    np.interp(si, s, xy[:, 1])], 1)
    tr = np.interp(si, s, ts)
    return xyr, si, tr, hts, mdir


def frame_at(mdir, hts, t):
    i = int(np.argmin(np.abs(hts - t)))
    return mdir / "images" / "hdr_front" / f"{i:06d}.jpeg"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mission in MISSIONS:
        xyr, si, tr, hts, mdir = load(mission)
        turns = detect_turns(xyr, si, si[0], si[-1])
        picked, last = [], -1e9
        for t in turns:
            if (t["s"] - APPROACH_M < 1 or t["s"] + EXIT_M > si[-1] - 1
                    or t["s"] - last < 30):
                continue
            picked.append(t)
            last = t["s"]
        for t in picked[:2]:
            s0, s1 = t["s"] - APPROACH_M, t["s"] + EXIT_M
            wturns = detect_turns(xyr, si, s0, s1)
            if not wturns:
                continue
            content = []
            targets = np.arange(s0, s1 + 1e-6, FRAME_GAP_M)
            n_f = 0
            for tg in targets:
                f = frame_at(mdir, hts, float(np.interp(tg, si, tr)))
                im = cv2.imread(str(f))
                if im is None:
                    continue
                im = cv2.resize(im, (768, int(768 * im.shape[0]
                                              / im.shape[1])))
                n_f += 1
                cv2.putText(im, f"frame {n_f}/{len(targets)}", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(im, f"frame {n_f}/{len(targets)}", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                content.append(img_part_mem(im))
                cv2.imwrite(str(OUT / f"{mission}_s{int(t['s'])}"
                                f"_f{n_f:02d}.jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            if n_f < 6:
                continue
            bev = build_bev(xyr, si, s0, s1, wturns)
            content.append(img_part_mem(bev))
            cv2.imwrite(str(OUT / f"{mission}_s{int(t['s'])}_bev.png"), bev)
            turn_list = "\n".join(
                f"  {ti+1}) {w['dir'].upper()} turn (~{w['deg']:.0f} deg), "
                f"about {w['s']-s0:.0f} m after the start"
                for ti, w in enumerate(wturns))
            route_gt = ROUTE_GT_TURNS.format(n=len(wturns),
                                             turn_list=turn_list)
            prompt = EP_PROMPT.format(
                bev_extra=", numbered X marks = the turns in driving order",
                route_gt=route_gt)
            content.append({"type": "text", "text": prompt})
            ann = chat_retry(PORT, content, EP_SCHEMA, 700)
            print(f"== {mission} @ s={t['s']:.0f} "
                  f"({len(wturns)} turns, {n_f} frames)")
            print("  instruction:", ann["instruction"])
            for m in ann["maneuvers"]:
                print(f"    {m['action']:12s} {m.get('cue','')}")
            results.append({"mission": mission, "s": t["s"], "ann": ann})
    json.dump(results, open(OUT / "smoke.json", "w"), indent=1)
    print("GT_ANN_SMOKE_DONE", len(results))


if __name__ == "__main__":
    main()
