#!/usr/bin/env python3
"""P4 prototype: decision-point episodes with multi-frame annotation.

Single-frame 10 m windows produce vacuous instructions ("walk straight...")
— on a corridor there is no decision to encode (user critique, 2026-07-27).
An instruction only carries information when the horizon contains a choice.

Episode = a stretch of trajectory centered on a real DECISION POINT (heading
change > TURN_DEG over TURN_ARC_M of arc), spanning APPROACH_M before to
EXIT_M after. The annotator (local VLM) sees, in hindsight:
  - a strip of keyframes at ~FRAME_GAP_M spacing across the episode
    (history AND future relative to the episode start), and
  - a BEV sketch of the executed path with the decision point marked;
and writes ONE instruction executable from the FIRST frame, plus a
structured maneuver list. The trained model will only ever see the first
frame (+ its own BEV memory) and the instruction.

Usage:
    python -m wildvln.p4_episodes --n 6 --port 8118
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import os

import cv2
import numpy as np
import requests

P2B_ROOT = Path(os.environ.get("WILDVLN_P2B_ROOT",
                               "/data/patelm/ticvla/wildvln/p2b"))
OUT = Path("/data/patelm/ticvla/wildvln/p4/_episodes")

TURN_DEG = 40.0
TURN_ARC_M = 8.0
APPROACH_M = 22.0
EXIT_M = 15.0
FRAME_GAP_M = 3.5
MIN_SPEED = 0.3

SITES = ["AU", "GTown", "UDC", "UMD_map1_2_lot9",
         "UMD_map2_1_dininghall", "UMD_map1_1_trail"]

PROMPT = """You are writing navigation instructions for a robot dataset. You see:
1. A sequence of frames from a robot's forward camera as it ACTUALLY drove a route (frame 1 = start, last frame = end). The route includes a turn.
2. The last image is a TOP-DOWN sketch of the driven path (start = circle, arrow = direction of travel, X = the main turn).

Write ONE instruction that a robot standing at FRAME 1, seeing only that view, could follow to reproduce the whole route. The robot cannot see the future frames or the sketch. The instruction must say WHERE to turn using permanent landmarks that become visible along the way (buildings, signs, fences, poles, path junctions) — never people or vehicles. Phrase like a human giving directions: what to follow, where to turn, what to head toward after the turn.

Return STRICT JSON:
  "instruction": one sentence, max 28 words,
  "maneuvers": ordered list, each {"action": "continue"|"turn_left"|"turn_right", "cue": short landmark phrase for when/where},
  "decision_visible_from_start": true if the turn location is already visible in frame 1, else false,
  "hindsight_notes": one sentence on anything in the future frames that was needed to write the instruction."""

SCHEMA = {"type": "object", "properties": {
    "instruction": {"type": "string"},
    "maneuvers": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
        "type": "object", "properties": {
            "action": {"enum": ["continue", "turn_left", "turn_right"]},
            "cue": {"type": "string"}},
        "required": ["action", "cue"]}},
    "decision_visible_from_start": {"type": "boolean"},
    "hindsight_notes": {"type": "string"}},
    "required": ["instruction", "maneuvers", "decision_visible_from_start",
                 "hindsight_notes"]}


def arc_length(xy):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(d)])


def find_episodes(site, bag):
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kt, kpose, kseg, kvalid = idx["t"], idx["pose"], idx["seg_id"], idx["valid"]
    eps = []
    for seg in np.unique(kseg[kseg >= 0]):
        m = (kseg == seg) & kvalid
        if m.sum() < 30:
            continue
        ii = np.where(m)[0]
        xy = kpose[ii][:, :2, 3]
        s = arc_length(xy)
        # skip standing-still stretches
        heading = np.degrees(np.unwrap(np.arctan2(*np.diff(xy, axis=0).T[::-1])))
        for j in range(len(ii) - 1):
            a, b = s[j], s[j] + TURN_ARC_M
            k = np.searchsorted(s, b)
            if k >= len(ii) - 1:
                break
            dh = heading[min(k, len(heading)-1)] - heading[max(j-1, 0)]
            if abs(dh) < TURN_DEG or abs(dh) > 150:
                continue
            if s[j] < APPROACH_M or s[-1] - s[k] < EXIT_M:
                continue
            eps.append({"site": site, "bag": bag, "seg": int(seg),
                        "idx": ii, "s": s, "j": j, "k": k,
                        "turn_deg": float(dh)})
    # dedup: keep episodes at least 25 m apart
    out, last_s = [], -1e9
    for e in sorted(eps, key=lambda e: e["s"][e["j"]]):
        if e["s"][e["j"]] - last_s > 25.0:
            out.append(e)
            last_s = e["s"][e["j"]]
    return out


def build_inputs(e):
    idx, s, j, k = e["idx"], e["s"], e["j"], e["k"]
    idx_np = np.load(P2B_ROOT / e["site"] / e["bag"] / "index.npz")
    kt, kpose = idx_np["t"], idx_np["pose"]
    s0, s1 = s[j] - APPROACH_M, s[k] + EXIT_M
    targets = np.arange(s0, s1, FRAME_GAP_M)
    picks = [int(np.argmin(np.abs(s - tg))) for tg in targets]
    picks = sorted(set(picks))
    frames = []
    for p in picks:
        f = P2B_ROOT / e["site"] / e["bag"] / "keyframes" / \
            f"{int(kt[idx[p]]*1e9)}.jpg"
        if f.exists():
            frames.append((p, f))
    # BEV sketch
    xy = kpose[idx][:, :2, 3]
    sel = (s >= s0) & (s <= s1)
    pts = xy[sel] - xy[sel][0]
    ang = -np.arctan2(*(np.diff(pts[:2], axis=0)[0])[::-1])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ R.T
    img = np.full((420, 420, 3), 250, np.uint8)
    scale = 380 / max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1)
    pix = ((pts - pts.min(0)) * scale + 20).astype(int)
    pix[:, 1] = 419 - pix[:, 1]
    for a, b in zip(pix[:-1], pix[1:]):
        cv2.line(img, tuple(a), tuple(b), (40, 40, 40), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(pix[0]), 9, (30, 130, 30), -1)
    tj = int(np.searchsorted(np.where(sel)[0], j))
    cv2.drawMarker(img, tuple(pix[min(tj, len(pix)-1)]), (30, 30, 220),
                   cv2.MARKER_TILTED_CROSS, 22, 4)
    d = pix[-1] - pix[-5 if len(pix) > 5 else -2]
    tip = tuple(pix[-1])
    cv2.arrowedLine(img, tuple(pix[-1] - d), tip, (40, 40, 40), 3,
                    tipLength=0.5)
    return frames, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--port", type=int, default=8118)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_eps = []
    for site in SITES:
        for bag_dir in sorted((P2B_ROOT / site).iterdir()):
            if (bag_dir / "index.npz").exists():
                all_eps += find_episodes(site, bag_dir.name)
    print(f"{len(all_eps)} decision-point episodes found")
    rng = np.random.default_rng(3)
    picks = [all_eps[i] for i in
             rng.choice(len(all_eps), min(args.n, len(all_eps)), replace=False)]

    results = []
    for ei, e in enumerate(picks):
        frames, bev = build_inputs(e)
        if len(frames) < 6:
            continue
        content = []
        fnames = []
        for fi, (p, f) in enumerate(frames):
            im = cv2.imread(str(f))
            cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fp = OUT / f"ep{ei}_f{fi:02d}.jpg"
            cv2.imwrite(str(fp), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            fnames.append(fp.name)
            b64 = base64.b64encode(fp.read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        bp = OUT / f"ep{ei}_bev.png"
        cv2.imwrite(str(bp), bev)
        b64 = base64.b64encode(bp.read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": PROMPT})
        r = requests.post(f"http://localhost:{args.port}/v1/chat/completions",
            json={"model": "qwen3.5-vl",
                  "messages": [{"role": "user", "content": content}],
                  "temperature": 0.2, "max_tokens": 700,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "response_format": {"type": "json_schema",
                      "json_schema": {"name": "ep", "schema": SCHEMA}}},
            timeout=300)
        ann = json.loads(r.json()["choices"][0]["message"]["content"])
        results.append({"site": e["site"], "bag": e["bag"],
                        "turn_deg": e["turn_deg"], "frames": fnames,
                        "bev": bp.name, "ann": ann})
        print(f"[{ei}] {e['site']} turn {e['turn_deg']:+.0f}deg "
              f"{len(frames)} frames -> {ann['instruction']!r}")
    json.dump(results, open(OUT / "episodes.json", "w"), indent=1)
    print(f"-> {OUT}/episodes.json")


if __name__ == "__main__":
    main()
