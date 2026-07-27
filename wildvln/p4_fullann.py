#!/usr/bin/env python3
"""P4 prototype: FULL chained annotations for decision-point AND straight
(continue) episodes.

Memory (MVP iteration 1, user decision 2026-07-28): the MINIMAL fixed-key
schema — landmarks + turns only, prove the concept first:
  {"passed": ["white columns", "crosswalk"],
   "turns": ["right at the T-junction"],
   "progress": 61, "active": "reach plaza, then left"}
The scalable typed event log with odometer stamps (passed/turn/crossed/
entered + odo, enabling return-to-X) is designed and shelved for a later
iteration — see the 2026-07-28 entries in the project memory.

Episode kinds:
  turn      — mined decision points (p4_episodes.find_episodes); the turn
              sequence inside the span is GEOMETRIC GT (straight-run RDP
              detector) and the instruction must cover every turn in order.
  continue  — straight stretches >= ~40 m with no detected turn: teaches
              the model NOT to hallucinate turns, gives memory copy-through
              GT, still gets passed-landmark updates.

Per step (~3.5 m): input {raw frame, instruction, memory_in, history
polyline}; target {cot, memory_out, 10-pt/10-m trace in motion-derived
robot frame}. turns + progress are GEOMETRIC overrides;
passed/crossed/entered + active from the annotator (sees GT-trace overlay
= privileged, never mentioned); no event -> memory copies through.

Usage: python -m wildvln.p4_fullann --n 4 --n-cont 3 --port 8118
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import requests

from wildvln.p4_episodes import (P2B_ROOT, SITES, FRAME_GAP_M,
                                 SCHEMA as EP_SCHEMA, arc_length,
                                 find_episodes)
from wildvln.rigs import rig_for_site

OUT = Path("/data/patelm/ticvla/wildvln/p4/_full")
TRACE_BUDGET_M = 10.0
TRACE_PTS = 10
TURN_WIN_M = 6.0        # "executing the turn" half-window in the step chain
MAX_EVENTS = 10
CONT_LEN_M = 40.0       # continue-episode length
CONT_MARGIN_M = 8.0     # keep-away from any detected turn
CONT_SPACING_M = 45.0

EP_PROMPT = """You are writing navigation instructions for a robot dataset. You see:
1. A sequence of frames from a robot's forward camera as it ACTUALLY drove a route (frame 1 = start, last frame = end).
2. The last image is a TOP-DOWN sketch of the driven path (start = circle, arrow = direction of travel{bev_extra}).

{route_gt}

Write ONE instruction that a robot standing at FRAME 1, seeing only that view, could follow to reproduce the whole route. The robot cannot see the future frames or the sketch. Use permanent landmarks that become visible along the way (buildings, signs, fences, poles, path junctions) — never people, animals, or vehicles. Parked cars/trucks/buses are NOT landmarks either (they move away); in vehicle-dominated scenes anchor on pavement markings, curbs, light poles, signs, trees, or building corners instead. Phrase like a human giving directions.

Return STRICT JSON:
  "instruction": one sentence, max 35 words,
  "maneuvers": ordered list, each {{"action": "continue"|"turn_left"|"turn_right", "cue": short landmark phrase for when/where}},
  "decision_visible_from_start": true if the first turn's location (or for a straight route, the far end) is already visible in frame 1, else false,
  "hindsight_notes": one sentence on anything in the future frames that was needed to write the instruction."""

ROUTE_GT_TURNS = """GROUND TRUTH from odometry — the route contains exactly {n} turn(s), in order (numbered X marks on the sketch):
{turn_list}
Your maneuver list MUST contain one turn entry for EACH of these, in the same order, with a landmark cue saying where it happens. The instruction must mention EVERY turn (a short clause per turn is fine). Do not invent extra turns."""

ROUTE_GT_STRAIGHT = """GROUND TRUTH from odometry — the route is STRAIGHT: it contains NO turns. Your maneuver list must contain ONLY "continue" entries (1-3), each cueing a landmark the robot passes. The instruction must NOT mention any turn: tell the robot what to follow and which landmarks it will pass so it can confirm it is on track."""

PARA_PROMPT = """Rewrite the following robot navigation instruction in {n} different phrasings. Rules:
- Keep EVERY landmark name exactly as written (e.g. "yellow tree", "blue emergency sign") — do not rename, drop, or add landmarks.
- Keep the number, order, and direction of turns identical.
- Only vary sentence structure, verbs, and connective wording; phrasings should sound like different people giving the same directions.
- Each max 35 words.

Instruction: "{instruction}"

Return STRICT JSON: {{"variants": [{n} strings]}}"""

PARA_SCHEMA = {"type": "object", "properties": {
    "variants": {"type": "array", "minItems": 4, "maxItems": 4,
                 "items": {"type": "string"}}},
    "required": ["variants"]}

STEP_PROMPT = """You are writing the inner monologue of a navigation robot for ONE moment of a route it is executing. The image is the robot's current forward camera view. The green line is a PRIVILEGED overlay showing the actual next 10 m of the route — you can use it to know where the route goes, but the robot cannot see it: NEVER mention the line, its color, or any overlay.

GROUND TRUTH about this moment — this is FACT, it overrides whatever the image suggests to you:
- Driven so far: {odo} m ({progress}% of the route)
- Turn status: {turn_status}
- {consistency}

The robot is following this instruction: "{instruction}"

Robot's MEMORY at this moment — landmarks already passed and turns already made (kept up to date, INCLUDING any turn that just happened):
{memory_in}

Return STRICT JSON:
  "cot": 1-3 short sentences of the robot's present-tense reasoning at THIS moment: what it recognizes in the current view, where it is relative to the instruction, and what it will do next. Ground it ONLY in what is visible in the image plus the memory — no future knowledge beyond the instruction, no people/vehicles as anchors.
  "passed_new": landmarks the robot has clearly just passed or is passing at THIS step (short names; empty list most steps). A landmark named in the instruction counts as passed once it is beside or behind the robot, not still ahead. Permanent things only, never people or vehicles; do not repeat anything already in memory.
  "active": the robot's current subgoal, max 8 words (e.g. "reach T-junction, then turn right"),
  "event": true ONLY if passed_new is non-empty or a turn just started/finished or the subgoal changed; false for plain mid-route steps."""

STEP_SCHEMA = {"type": "object", "properties": {
    "cot": {"type": "string"},
    "passed_new": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    "active": {"type": "string"},
    "event": {"type": "boolean"}},
    "required": ["cot", "passed_new", "active", "event"]}


def chat(port, content, schema, max_tokens=600):
    r = requests.post(f"http://localhost:{port}/v1/chat/completions",
        json={"model": "qwen3.5-vl",
              "messages": [{"role": "user", "content": content}],
              "temperature": 0.2, "max_tokens": max_tokens,
              "chat_template_kwargs": {"enable_thinking": False},
              "response_format": {"type": "json_schema",
                  "json_schema": {"name": "ann", "schema": schema}}},
        timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def img_part(path):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker; returns sorted indices of kept vertices."""
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = pts[b] - pts[a]
        L = np.linalg.norm(d)
        rel = pts[a + 1:b] - pts[a]
        if L < 1e-9:
            dist = np.linalg.norm(rel, axis=1)
        else:
            dist = np.abs(rel[:, 1] * d[0] - rel[:, 0] * d[1]) / L
        i = a + 1 + int(np.argmax(dist))
        if dist.max() > eps:
            keep.append(i)
            stack += [(a, i), (i, b)]
    return np.array(sorted(keep))


def detect_turns(xy, s, s0, s1, eps_m=1.0, min_deg=30.0, min_run_m=4.0):
    """Turns = heading change between consecutive STRAIGHT RUNS (>=min_run_m)
    of the RDP-simplified path in [s0, s1].

    Windowed heading change over-fires on stalls/wobble (heading is noise
    when the robot barely moves); per-vertex RDP angles miss rounded
    corners. Straight-run headings survive both. Positive angle = left.
    """
    sel = (s >= s0) & (s <= s1)
    pts, ss = xy[sel], s[sel]
    if len(pts) < 3:
        return []
    ki = _rdp(pts, eps_m)
    segs = [{"a": a, "b": b, "v": pts[b] - pts[a],
             "len": float(np.linalg.norm(pts[b] - pts[a]))}
            for a, b in zip(ki[:-1], ki[1:])]
    runs = [g for g in segs if g["len"] >= min_run_m]
    out = []
    for r1, r2 in zip(runs[:-1], runs[1:]):
        v1, v2 = r1["v"], r2["v"]
        ang = np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0],
                                    np.dot(v1, v2)))
        if abs(ang) >= min_deg:
            out.append({"s": float((ss[r1["b"]] + ss[r2["a"]]) / 2),
                        "dir": "left" if ang > 0 else "right",
                        "deg": float(abs(ang))})
    return out


def find_continue_episodes(site, bag):
    """Straight stretches with no detected turn: CONT_LEN_M windows tiled
    into inter-turn gaps (CONT_MARGIN_M keep-away). Also returns the count
    of segment-wide detected turns (for data-budget stats)."""
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kpose, kseg, kvalid = idx["pose"], idx["seg_id"], idx["valid"]
    eps, n_turns = [], 0
    for seg in np.unique(kseg[kseg >= 0]):
        m = (kseg == seg) & kvalid
        if m.sum() < 30:
            continue
        ii = np.where(m)[0]
        xy = kpose[ii][:, :2, 3]
        s = arc_length(xy)
        turns = detect_turns(xy, s, s[0], s[-1])
        n_turns += len(turns)
        bounds = [s[0]] + [t["s"] for t in turns] + [s[-1]]
        for a, b in zip(bounds[:-1], bounds[1:]):
            a2, b2 = a + CONT_MARGIN_M, b - CONT_MARGIN_M
            w = a2
            while w + CONT_LEN_M <= b2:
                eps.append({"kind": "continue", "site": site, "bag": bag,
                            "seg": int(seg), "idx": ii, "s": s,
                            "s0": float(w), "s1": float(w + CONT_LEN_M),
                            "turn_deg": 0.0})
                w += CONT_SPACING_M
    return eps, n_turns


def build_bev(xy, s, s0, s1, turns):
    """Path sketch with start circle, direction arrow, numbered turn marks."""
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
    s_sel = s[sel]
    for ti, t in enumerate(turns):
        q = pix[int(np.argmin(np.abs(s_sel - t["s"])))]
        cv2.drawMarker(img, tuple(q), (30, 30, 220),
                       cv2.MARKER_TILTED_CROSS, 22, 4)
        cv2.putText(img, str(ti + 1), (q[0] + 12, q[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 220), 2)
    d = pix[-1] - pix[-5 if len(pix) > 5 else -2]
    cv2.arrowedLine(img, tuple(pix[-1] - d), tuple(pix[-1]), (40, 40, 40), 3,
                    tipLength=0.5)
    return img


def local_polyline(xy, s, i0, budget, npts, forward=True):
    """Distance-parameterized polyline in the pose-frame at index i0.

    Frame: x = travel direction at i0 (motion-derived — mount-yaw free),
    y = left. Returns ((npts, 2), arc length) or (None, 0) if too short.
    """
    p0 = xy[i0]
    d = xy[min(i0 + 1, len(xy) - 1)] - xy[max(i0 - 1, 0)]
    ang = np.arctan2(d[1], d[0])
    R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    if forward:
        seg_s, seg_xy = s[i0:] - s[i0], xy[i0:]
    else:
        seg_s, seg_xy = s[i0] - s[:i0 + 1][::-1], xy[:i0 + 1][::-1]
    L = min(budget, seg_s[-1]) if len(seg_s) > 1 else 0.0
    if L < 0.5:
        return None, 0.0
    targets = np.linspace(L / npts, L, npts)
    out = np.stack([np.interp(targets, seg_s, seg_xy[:, k]) for k in (0, 1)], 1)
    return (out - p0) @ R.T, float(L)


def overlay_trace(img, trace, rig, z_rel=None, pts3d=None):
    """pts3d: optional (N,3) ground points already expressed in the TRUE
    sensor pose at this keyframe (full SE3 — carries the pitch/roll of a
    robot nosing up a staircase, which the yaw-only ego frame cannot).
    Without it, points are built from the ego trace: z_rel gives per-point
    ground elevation; flat ground otherwise."""
    if pts3d is not None:
        pts_l = np.asarray(pts3d, float)
    else:
        h = rig.lidar_height_m
        zcol = (np.asarray(z_rel).reshape(-1, 1) - h if z_rel is not None
                else np.full((len(trace), 1), -h))
        pts_l = np.concatenate([trace, zcol], 1)
    T = np.asarray(rig.T_cam_lidar)
    Xc = pts_l @ T[:3, :3].T + T[:3, 3]
    fx, fy, cx, cy = rig.intrinsics
    ok = Xc[:, 2] > 0.5
    u = (fx * Xc[:, 0] / Xc[:, 2] + cx)[ok]
    v = (fy * Xc[:, 1] / Xc[:, 2] + cy)[ok]
    W, H = rig.image_size
    pts = [(int(a), int(b)) for a, b in zip(u, v) if 0 <= a < W and 0 <= b < H]
    for p, q in zip(pts[:-1], pts[1:]):
        cv2.line(img, p, q, (60, 220, 60), 3, cv2.LINE_AA)
    return img


def turn_status_text(turns, s_i, s1):
    """Per-turn state at arc station s_i + human-readable summary."""
    states, parts = [], []
    for ti, t in enumerate(turns):
        half = TURN_WIN_M / 2
        if s_i < t["s"] - half:
            st = "upcoming"
            parts.append(f"turn {ti+1} ({t['dir']}) is ~{t['s']-s_i:.0f} m ahead")
        elif s_i <= t["s"] + half:
            st = "in_progress"
            parts.append(f"turn {ti+1} ({t['dir']}) is being executed NOW")
        else:
            st = "done"
            parts.append(f"turn {ti+1} ({t['dir']}) is completed")
        states.append(st)
    n_done = sum(st == "done" for st in states)
    body = "; ".join(parts) if parts else "the route is straight — NO turns"
    txt = body + f". {max(0, s1-s_i):.0f} m remain to the route end."
    if not turns:
        note = ("The CoT must NOT mention any turn — the robot just follows "
                "the path and confirms landmarks.")
    elif n_done == len(turns):
        note = ("ALL turns are already COMPLETED (they are in the memory "
                "log). The CoT must NOT search for a turn landmark or plan "
                "any turn — the robot has turned and is heading to the "
                "route end.")
    elif "in_progress" in states:
        note = ("The robot is MID-TURN right now — the CoT should describe "
                "executing this turn, not looking for it.")
    else:
        note = ("The next turn has NOT started yet — the CoT may look for "
                "its landmark but must not claim the robot is turning.")
    return states, n_done, txt, note


def pick_frames(e, kt, idx, s, s0, s1):
    """Keyframe picks at ~FRAME_GAP_M spacing across [s0, s1]."""
    targets = np.arange(s0, s1 + 1e-6, FRAME_GAP_M)
    picks = sorted(set(int(np.argmin(np.abs(s - tg))) for tg in targets))
    out = []
    for p in picks:
        f = (P2B_ROOT / e["site"] / e["bag"] / "keyframes" /
             f"{int(kt[idx[p]]*1e9)}.jpg")
        if f.exists():
            out.append((p, f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="turn episodes")
    ap.add_argument("--n-cont", type=int, default=3, help="continue episodes")
    ap.add_argument("--port", type=int, default=8118)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    turn_eps, cont_eps, n_turns_all = [], [], 0
    for site in SITES:
        for bag_dir in sorted((P2B_ROOT / site).iterdir()):
            if not (bag_dir / "index.npz").exists():
                continue
            turn_eps += find_episodes(site, bag_dir.name)
            ce, nt = find_continue_episodes(site, bag_dir.name)
            cont_eps += ce
            n_turns_all += nt
    print(f"mined: {len(turn_eps)} turn episodes (>=40deg/8m), "
          f"{len(cont_eps)} continue windows, "
          f"{n_turns_all} segment-wide turns >=30deg (episode potential)")

    rng = np.random.default_rng(3)
    picks = [dict(turn_eps[i], kind="turn") for i in
             rng.choice(len(turn_eps), min(args.n, len(turn_eps)),
                        replace=False)]
    picks += [cont_eps[i] for i in
              rng.choice(len(cont_eps), min(args.n_cont, len(cont_eps)),
                         replace=False)]

    episodes = []
    for ei, e in enumerate(picks):
        rig = rig_for_site(e["site"])
        idx_np = np.load(P2B_ROOT / e["site"] / e["bag"] / "index.npz")
        kt, kpose = idx_np["t"], idx_np["pose"]
        idx, s = e["idx"], e["s"]
        xy = kpose[idx][:, :2, 3]
        if e["kind"] == "turn":
            s0, s1 = s[e["j"]] - 22.0, s[e["k"]] + 15.0
            turns = detect_turns(xy, s, s0, s1)
            if not turns:   # long sweeping arc: fall back to the mined turn
                turns = [{"s": float(s[e["j"]]),
                          "dir": "left" if e["turn_deg"] > 0 else "right",
                          "deg": float(abs(e["turn_deg"]))}]
        else:
            s0, s1 = e["s0"], e["s1"]
            turns = []

        frames = pick_frames(e, kt, idx, s, s0, s1)
        if len(frames) < 6:
            continue

        if turns:
            turn_list = "\n".join(
                f"  {ti+1}) {t['dir'].upper()} turn (~{t['deg']:.0f} deg), "
                f"about {t['s']-s0:.0f} m after the start"
                for ti, t in enumerate(turns))
            route_gt = ROUTE_GT_TURNS.format(n=len(turns), turn_list=turn_list)
            bev_extra = ", numbered X marks = the turns in driving order"
        else:
            route_gt = ROUTE_GT_STRAIGHT
            bev_extra = ""

        # ---- stage A: episode instruction (frame strip + BEV)
        content = []
        for fi, (p, f) in enumerate(frames):
            im = cv2.imread(str(f))
            cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fp = OUT / f"ep{ei}_f{fi:02d}.jpg"
            cv2.imwrite(str(fp), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            content.append(img_part(fp))
        bp = OUT / f"ep{ei}_bev.png"
        cv2.imwrite(str(bp), build_bev(xy, s, s0, s1, turns))
        content.append(img_part(bp))
        content.append({"type": "text", "text": EP_PROMPT.format(
            bev_extra=bev_extra, route_gt=route_gt)})
        ep_ann = chat(args.port, content, EP_SCHEMA, 700)

        # 4 goal-prompt paraphrases (TIC-VLA style): same landmarks + turn
        # order, different wording; step targets are shared across variants
        para = chat(args.port, [{"type": "text", "text": PARA_PROMPT.format(
            n=4, instruction=ep_ann["instruction"])}], PARA_SCHEMA, 400)
        variants = [ep_ann["instruction"]] + para["variants"]

        turn_mans = [m for m in ep_ann["maneuvers"] if m["action"] != "continue"]
        ok_seq = ([m["action"] for m in turn_mans] ==
                  [f"turn_{t['dir']}" for t in turns])
        if not ok_seq:   # keep memory GT true to geometry
            turn_mans = [{"action": f"turn_{t['dir']}", "cue": ""}
                         for t in turns]
        print(f"[ep{ei}|{e['kind']}] {e['site']} {len(turns)} turn(s) "
              f"[{','.join(t['dir'] for t in turns)}] seq_ok={ok_seq} -> "
              f"{ep_ann['instruction']!r}")

        # ---- stage B: per-step chain, event-log memory
        memory = {"passed": [], "turns": [], "progress": 0,
                  "active": ep_ann["maneuvers"][0]["cue"][:60]}
        steps = []
        for si, (p, f) in enumerate(frames):
            s_i = s[p]
            odo = int(round(s_i - s0))
            progress = int(round(100 * np.clip((s_i - s0) / (s1 - s0), 0, 1)))
            states, n_done, tstat, note = turn_status_text(turns, s_i, s1)

            trace, tlen = local_polyline(xy, s, p, TRACE_BUDGET_M, TRACE_PTS)
            hist, _ = local_polyline(xy, s, p, TRACE_BUDGET_M, TRACE_PTS,
                                     forward=False)
            if trace is None:
                continue
            im = cv2.imread(str(f))
            op = OUT / f"ep{ei}_s{si:02d}_ov.jpg"
            cv2.imwrite(str(op), overlay_trace(im, trace, rig),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

            mem_in = json.loads(json.dumps(memory))   # deep copy
            # Merge geometric turn completions BEFORE the annotator call:
            # a post-turn view looks like a plain straight path, and without
            # the turn in its memory the annotator concludes the turn is
            # still ahead (user-caught). The SAMPLE keeps the pure chain
            # mem_in; the annotator gets the geometrically-updated log.
            mem_geo = json.loads(json.dumps(mem_in))
            mem_geo["progress"] = progress
            n_done_prev = len(mem_in["turns"])
            for ti in range(n_done_prev, n_done):   # turn(s) just completed
                cue = turn_mans[ti]["cue"] if ti < len(turn_mans) else ""
                cue = cue.removeprefix("at ").removeprefix("at the ")
                mem_geo["turns"].append(
                    turns[ti]["dir"] + (f" at {cue}" if cue else " turn"))

            prompt = STEP_PROMPT.format(
                instruction=ep_ann["instruction"],
                memory_in=json.dumps(mem_geo), odo=odo,
                progress=progress, turn_status=tstat, consistency=note)
            ann = chat(args.port, [img_part(op),
                       {"type": "text", "text": prompt}], STEP_SCHEMA, 500)

            # guard: all turns done but the CoT still hunts for a turn
            retried = False
            if turns and n_done == len(turns):
                bad = ("will turn", "until i", "look for", "looking for",
                       "locate", "don't see", "do not see", "not visible",
                       "when i reach", "prepare to turn", "approaching the turn")
                if any(b in ann["cot"].lower() for b in bad):
                    retried = True
                    ann = chat(args.port, [img_part(op), {"type": "text",
                        "text": prompt + "\n\nREMINDER: a previous draft "
                        "wrongly searched for the turn. ALL turns are "
                        "already DONE — write the CoT of a robot that has "
                        "finished turning and is completing the route."}],
                        STEP_SCHEMA, 500)

            # memory_out: geometric turns/progress + annotator passed/active
            mem_out = json.loads(json.dumps(mem_geo))
            known = {p.lower() for p in mem_out["passed"]}
            new_p = [p for p in ann["passed_new"] if p.lower() not in known]
            if ann["event"] or new_p or n_done > n_done_prev:
                mem_out["passed"] += new_p
                mem_out["active"] = ann["active"][:60]
            mem_out["passed"] = mem_out["passed"][-MAX_EVENTS:]
            steps.append({
                "kf": f"{int(kt[idx[p]]*1e9)}.jpg", "s_m": float(s_i),
                "overlay": op.name, "turn_states": states,
                "retried": retried,
                "input": {"instruction": ep_ann["instruction"],
                          "memory": mem_in,
                          "history": None if hist is None else
                              np.round(hist, 2).tolist()},
                "target": {"cot": ann["cot"], "memory": mem_out,
                           "trace": np.round(trace, 2).tolist(),
                           "trace_len_m": round(tlen, 1)}})
            memory = mem_out
            flag = "*" if steps[-1]["target"]["memory"] != mem_in else " "
            print(f"  s{si:02d}{flag} odo{odo:3d} {progress:3d}% "
                  f"t{len(mem_out['turns'])}p{len(mem_out['passed'])} {ann['cot'][:80]!r}")

        episodes.append({"site": e["site"], "bag": e["bag"],
                         "kind": e["kind"], "turn_deg": e["turn_deg"],
                         "instruction_variants": variants,
                         "turns_geo": turns, "seq_ok": bool(ok_seq),
                         "frames": [f"ep{ei}_f{fi:02d}.jpg"
                                    for fi in range(len(frames))],
                         "bev": bp.name, "episode_ann": ep_ann,
                         "steps": steps})

    json.dump(episodes, open(OUT / "episodes_full.json", "w"), indent=1)
    n_steps = sum(len(ep["steps"]) for ep in episodes)
    n_var = sum(len(ep["instruction_variants"]) for ep in episodes)
    print(f"-> {OUT}/episodes_full.json  ({len(episodes)} episodes, "
          f"{n_steps} chained steps x {n_var//max(len(episodes),1)} goal "
          f"variants = {sum(len(ep['steps'])*len(ep['instruction_variants']) for ep in episodes)} samples)")


if __name__ == "__main__":
    main()
