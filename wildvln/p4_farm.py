#!/usr/bin/env python3
"""P4 language farm: full chained annotations for ALL episodes.

Mining (on FAST-LIO poses via WILDVLN_P2B_ROOT=p2bf):
  turn episodes     — one per detected turn (straight-run RDP detector,
                      >=30 deg), 22 m approach + 15 m exit, 25 m dedup
  continue episodes — 40 m straight windows between turns (8 m margin)

Per episode: stage A instruction (+QC: transient-anchor rejection with one
retry BEFORE paraphrasing) -> 4 goal-prompt paraphrases (+dedup retry) ->
per-step chain (simple memory {passed, turns, progress, active}; geometric
turns/progress; active-churn damping: subgoal may only change on a turn
state transition or a new passed landmark).

Episodes are independent -> ThreadPool over episodes, chain sequential
inside each. Checkpointed per episode (p4/farm/<ep_id>.json); rerun skips
finished episodes.

Usage: WILDVLN_P2B_ROOT=/data/patelm/ticvla/wildvln/p2bf \
       python -m wildvln.p4_farm [--workers 8] [--port 8118] [--limit N]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from wildvln.p4_episodes import P2B_ROOT, SITES, FRAME_GAP_M, arc_length
from wildvln.p4_episodes import SCHEMA as EP_SCHEMA
from wildvln.p4_fullann import (EP_PROMPT, ROUTE_GT_TURNS, ROUTE_GT_STRAIGHT,
                                PARA_PROMPT, PARA_SCHEMA, STEP_PROMPT,
                                STEP_SCHEMA, MAX_EVENTS, TRACE_BUDGET_M,
                                TRACE_PTS, chat, detect_turns, build_bev,
                                local_polyline, overlay_trace,
                                turn_status_text)
from wildvln.rigs import rig_for_site

OUT = Path("/data/patelm/ticvla/wildvln/p4/farm")
APPROACH_M, EXIT_M = 22.0, 15.0
DEDUP_M = 25.0
CONT_LEN_M, CONT_MARGIN_M, CONT_SPACING_M = 40.0, 8.0, 45.0

BANNED = re.compile(
    r"\b(person|people|group|pedestrian|man|woman|child|kid|crowd|walker|"
    r"dog|animal|car|cars|vehicle|truck|bus|van|bike|bicycle|cyclist|"
    r"scooter|stroller)\b", re.I)


def chat_retry(port, content, schema, max_tokens=600, tries=3):
    for a in range(tries):
        try:
            return chat(port, content, schema, max_tokens)
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))


def img_part_mem(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}


def valid_runs(kseg, kvalid):
    """Contiguous index runs of usable keyframes (valid & seg>=0)."""
    good = (kseg >= 0) & kvalid
    runs, start = [], None
    for i, g in enumerate(good):
        if g and start is None:
            start = i
        elif not g and start is not None:
            runs.append(np.arange(start, i)); start = None
    if start is not None:
        runs.append(np.arange(start, len(good)))
    return [r for r in runs if len(r) >= 30]


def mine(site, bag):
    idx = np.load(P2B_ROOT / site / bag / "index.npz")
    kpose, kseg, kvalid = idx["pose"], idx["seg_id"], idx["valid"]
    eps = []
    for ii in valid_runs(kseg, kvalid):
        xy = kpose[ii][:, :2, 3]
        s = arc_length(xy)
        turns = detect_turns(xy, s, s[0], s[-1])
        # turn episodes, 25 m dedup on turn station
        last = -1e9
        for t in turns:
            if (t["s"] - APPROACH_M < s[0] + 1 or t["s"] + EXIT_M > s[-1] - 1
                    or t["s"] - last < DEDUP_M):
                continue
            last = t["s"]
            eps.append({"kind": "turn", "site": site, "bag": bag,
                        "idx": ii, "s": s,
                        "s0": float(t["s"] - APPROACH_M),
                        "s1": float(t["s"] + EXIT_M)})
        # continue episodes in inter-turn gaps
        bounds = [s[0]] + [t["s"] for t in turns] + [s[-1]]
        for a, b in zip(bounds[:-1], bounds[1:]):
            w = a + CONT_MARGIN_M
            while w + CONT_LEN_M <= b - CONT_MARGIN_M:
                eps.append({"kind": "continue", "site": site, "bag": bag,
                            "idx": ii, "s": s,
                            "s0": float(w), "s1": float(w + CONT_LEN_M)})
                w += CONT_SPACING_M
    return eps


def pick_frames(e, kt, idx, s, s0, s1):
    targets = np.arange(s0, s1 + 1e-6, FRAME_GAP_M)
    picks = sorted(set(int(np.argmin(np.abs(s - tg))) for tg in targets))
    out = []
    for p in picks:
        f = (P2B_ROOT / e["site"] / e["bag"] / "keyframes" /
             f"{int(kt[idx[p]]*1e9)}.jpg")
        if f.exists():
            out.append((p, f))
    return out


def annotate_episode(e, ep_id, port):
    rig = rig_for_site(e["site"])
    idx_np = np.load(P2B_ROOT / e["site"] / e["bag"] / "index.npz")
    kt, kpose = idx_np["t"], idx_np["pose"]
    idx, s = e["idx"], e["s"]
    xy = kpose[idx][:, :2, 3]
    s0, s1 = e["s0"], e["s1"]
    turns = detect_turns(xy, s, s0, s1) if e["kind"] == "turn" else []
    if e["kind"] == "turn" and not turns:
        return None                       # detector disagreement, skip
    frames = pick_frames(e, kt, idx, s, s0, s1)
    if len(frames) < 6:
        return None

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

    # ---- stage A (+ transient-anchor QC BEFORE paraphrasing)
    content = []
    for fi, (p, f) in enumerate(frames):
        im = cv2.imread(str(f))
        cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, f"frame {fi+1}/{len(frames)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        content.append(img_part_mem(im))
    bev = build_bev(xy, s, s0, s1, turns)
    content.append(img_part_mem(bev))
    prompt_a = EP_PROMPT.format(bev_extra=bev_extra, route_gt=route_gt)
    content.append({"type": "text", "text": prompt_a})
    ep_ann = chat_retry(port, content, EP_SCHEMA, 700)

    def anchors_text(a):
        return " ".join([a["instruction"]] + [m["cue"] for m in a["maneuvers"]])

    qc_anchor_retry = False
    if BANNED.search(anchors_text(ep_ann)):
        qc_anchor_retry = True
        content[-1] = {"type": "text", "text": prompt_a +
                       "\n\nREMINDER: your previous draft anchored on a "
                       "person, animal or vehicle — STRICTLY FORBIDDEN. "
                       "Use only permanent static landmarks."}
        ep_ann = chat_retry(port, content, EP_SCHEMA, 700)
    qc_anchor_bad = bool(BANNED.search(anchors_text(ep_ann)))

    # ---- paraphrases (+ dedup retry)
    def get_variants():
        para = chat_retry(port, [{"type": "text", "text": PARA_PROMPT.format(
            n=4, instruction=ep_ann["instruction"])}], PARA_SCHEMA, 400)
        return para["variants"]

    variants = get_variants()
    canon = ep_ann["instruction"].strip().lower()
    if len({v.strip().lower() for v in variants} - {canon}) < 4:
        variants = get_variants()
    variants = [ep_ann["instruction"]] + [
        v for v in variants if v.strip().lower() != canon][:4]

    turn_mans = [m for m in ep_ann["maneuvers"] if m["action"] != "continue"]
    ok_seq = ([m["action"] for m in turn_mans] ==
              [f"turn_{t['dir']}" for t in turns])
    if not ok_seq:
        turn_mans = [{"action": f"turn_{t['dir']}", "cue": ""} for t in turns]

    # ---- stage B chain
    memory = {"passed": [], "turns": [], "progress": 0,
              "active": ep_ann["maneuvers"][0]["cue"][:60]}
    prev_states = None
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
        # 3D path stations expressed in the TRUE kf pose (full SE3 —
        # carries pitch/roll on stairs/slopes), dropped to ground height
        pw = kpose[idx][:, :3, 3]
        targets = s_i + np.linspace(tlen / TRACE_PTS, tlen, TRACE_PTS)
        st3 = np.stack([np.interp(targets, s, pw[:, k])
                        for k in range(3)], 1)
        st3[:, 2] -= rig.lidar_height_m
        Ti = np.linalg.inv(kpose[idx][p])
        pts3d = st3 @ Ti[:3, :3].T + Ti[:3, 3]
        ov = overlay_trace(im, trace, rig, pts3d=pts3d)
        op = OUT / "img" / f"{ep_id}_s{si:02d}.jpg"
        cv2.imwrite(str(op), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])

        mem_in = json.loads(json.dumps(memory))
        mem_geo = json.loads(json.dumps(mem_in))
        mem_geo["progress"] = progress
        n_done_prev = len(mem_in["turns"])
        for ti in range(n_done_prev, n_done):
            cue = turn_mans[ti]["cue"] if ti < len(turn_mans) else ""
            cue = cue.removeprefix("at ").removeprefix("at the ")
            mem_geo["turns"].append(
                turns[ti]["dir"] + (f" at {cue}" if cue else " turn"))

        ann = chat_retry(port, [img_part_mem(ov), {"type": "text",
                         "text": STEP_PROMPT.format(
                             instruction=ep_ann["instruction"],
                             memory_in=json.dumps(mem_geo), odo=odo,
                             progress=progress, turn_status=tstat,
                             consistency=note)}],
                         STEP_SCHEMA, 500)
        retried = False
        if turns and n_done == len(turns):
            bad = ("will turn", "until i", "look for", "looking for",
                   "locate", "when i reach", "prepare to turn",
                   "approaching the turn")
            if any(b in ann["cot"].lower() for b in bad):
                retried = True
                ann = chat_retry(port, [img_part_mem(ov), {"type": "text",
                    "text": STEP_PROMPT.format(
                        instruction=ep_ann["instruction"],
                        memory_in=json.dumps(mem_geo), odo=odo,
                        progress=progress, turn_status=tstat,
                        consistency=note) +
                    "\n\nREMINDER: ALL turns are already DONE — write the "
                    "CoT of a robot that has finished turning and is "
                    "completing the route."}], STEP_SCHEMA, 500)

        mem_out = json.loads(json.dumps(mem_geo))
        known = {x.lower() for x in mem_out["passed"]}
        new_p = [x for x in ann["passed_new"]
                 if x.lower() not in known and not BANNED.search(x)]
        state_changed = prev_states is not None and states != prev_states
        if new_p or n_done > n_done_prev or state_changed:
            mem_out["passed"] += new_p
            mem_out["active"] = ann["active"][:60]
        mem_out["passed"] = mem_out["passed"][-MAX_EVENTS:]
        steps.append({
            "kf": f"{int(kt[idx[p]]*1e9)}.jpg", "s_m": float(s_i),
            "overlay": op.name, "turn_states": states, "retried": retried,
            "input": {"instruction": ep_ann["instruction"],
                      "memory": mem_in,
                      "history": None if hist is None else
                          np.round(hist, 2).tolist()},
            "target": {"cot": ann["cot"], "memory": mem_out,
                       "trace": np.round(trace, 2).tolist(),
                       "trace_len_m": round(tlen, 1)}})
        memory = mem_out
        prev_states = states

    bp = OUT / "img" / f"{ep_id}_bev.png"
    cv2.imwrite(str(bp), bev)
    return {"ep_id": ep_id, "site": e["site"], "bag": e["bag"],
            "kind": e["kind"], "s0": s0, "s1": s1,
            "turns_geo": turns, "seq_ok": bool(ok_seq),
            "qc_anchor_retry": qc_anchor_retry,
            "qc_anchor_bad": qc_anchor_bad,
            "instruction_variants": variants, "bev": bp.name,
            "episode_ann": ep_ann, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=8118)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    (OUT / "img").mkdir(parents=True, exist_ok=True)

    eps = []
    for site in SITES:
        for bag_dir in sorted((P2B_ROOT / site).iterdir()):
            if (bag_dir / "index.npz").exists():
                eps += mine(site, bag_dir.name)
    for i, e in enumerate(eps):
        e["ep_id"] = (f"{e['site']}_{e['bag']}_{e['kind']}"
                      f"_{int(e['s0']):04d}").replace("/", "_")
    n_turn = sum(1 for e in eps if e["kind"] == "turn")
    print(f"mined {len(eps)} episodes ({n_turn} turn, "
          f"{len(eps)-n_turn} continue)", flush=True)
    if args.limit:
        eps = eps[:args.limit]

    todo = [e for e in eps if not (OUT / f"{e['ep_id']}.json").exists()]
    print(f"{len(todo)} to annotate ({len(eps)-len(todo)} cached)", flush=True)

    def work(e):
        try:
            r = annotate_episode(e, e["ep_id"], args.port)
            if r is None:
                (OUT / f"{e['ep_id']}.json").write_text(json.dumps(
                    {"ep_id": e["ep_id"], "skipped": True}))
                return f"{e['ep_id']}: skipped"
            (OUT / f"{e['ep_id']}.json").write_text(json.dumps(r))
            flag = " ANCHOR-BAD" if r["qc_anchor_bad"] else ""
            return (f"{e['ep_id']}: {len(r['steps'])} steps "
                    f"seq_ok={r['seq_ok']}{flag}")
        except Exception:
            return f"{e['ep_id']}: ERROR\n{traceback.format_exc()}"

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, e) for e in todo]
        for n, fu in enumerate(as_completed(futs)):
            print(f"[{n+1}/{len(todo)} {time.time()-t0:.0f}s] {fu.result()}",
                  flush=True)

    done = [json.loads(p.read_text()) for p in OUT.glob("*.json")
            if not p.name.startswith("_")]
    ok = [d for d in done if not d.get("skipped")]
    n_steps = sum(len(d["steps"]) for d in ok)
    n_samples = sum(len(d["steps"]) * len(d["instruction_variants"])
                    for d in ok)
    n_bad = sum(1 for d in ok if d.get("qc_anchor_bad"))
    print(f"FARM_DONE {len(ok)} episodes, {n_steps} steps, "
          f"{n_samples} samples (x variants), {n_bad} anchor-flagged",
          flush=True)


if __name__ == "__main__":
    main()
