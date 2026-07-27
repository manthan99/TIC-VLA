#!/usr/bin/env python3
"""Qualitative M-eval: run a checkpoint over held-out GTown episode chains
and dump image + GT-vs-predicted trace overlays + CoT/memory texts.

Usage: python -m wildvln.p6_qual --ckpt p6/m0/final [--model ...] [--n 3]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from wildvln.p4_fullann import overlay_trace
from wildvln.p6_eval import TRACE_RE
from wildvln.p6_sft import MODEL, MAX_PIXELS, build_text
from wildvln.rigs import rig_for_site

OUT = Path("/data/patelm/ticvla/wildvln/p6/_qual")


def overlay2(img, gt, pred, rig):
    img = overlay_trace(img, np.asarray(gt), rig)              # green GT
    if pred is not None:
        h = rig.lidar_height_m
        pts_l = np.concatenate([np.asarray(pred),
                                np.full((len(pred), 1), -h)], 1)
        T = np.asarray(rig.T_cam_lidar)
        Xc = pts_l @ T[:3, :3].T + T[:3, 3]
        fx, fy, cx, cy = rig.intrinsics
        ok = Xc[:, 2] > 0.5
        u = (fx * Xc[:, 0] / Xc[:, 2] + cx)[ok]
        v = (fy * Xc[:, 1] / Xc[:, 2] + cy)[ok]
        W, H = rig.image_size
        pts = [(int(a), int(b)) for a, b in zip(u, v)
               if 0 <= a < W and 0 <= b < H]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(img, p, q, (60, 60, 230), 3, cv2.LINE_AA)   # red pred
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=MAX_PIXELS)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, args.ckpt).eval()

    df = pd.read_parquet(
        "/data/patelm/ticvla/wildvln/p5/samples.parquet")
    sub = df[(df.split == "test_site") & (df["mode"] == "chained")
             & (df.variant == 0)]
    eps = [g for _, g in sub.groupby("ep_id") if g.kind.iloc[0] == "turn"]
    rng = np.random.default_rng(5)
    picks = [eps[i] for i in rng.choice(len(eps), args.n, replace=False)]

    results = []
    for g in picks:
        g = g.sort_values("step")
        rig = rig_for_site(g.site.iloc[0])
        steps = []
        for _, row in g.iterrows():
            user, _ = build_text(row)
            img = Image.open(row["image"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user}]}]
            prompt = proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=prompt, images=[img],
                       return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=400,
                                   do_sample=False)
            txt = proc.tokenizer.decode(o[0, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            m = re.search(r"<trace_bev>(.*?)</trace_bev>", txt, re.S)
            pred = ([(float(a), float(b)) for a, b in
                     TRACE_RE.findall(m.group(1))] if m else None)
            im = cv2.imread(row["image"])
            ov = overlay2(im, json.loads(row["trace"]), pred, rig)
            name = f"{row['ep_id']}_s{row['step']:02d}.jpg"
            cv2.imwrite(str(OUT / name), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            steps.append({"img": name, "gen": txt,
                          "gt_cot": row["cot"],
                          "gt_mem": row["memory_out"],
                          "instruction": row["instruction"]})
            print(f"{row['ep_id']} s{row['step']}: {txt[:100]!r}")
        results.append({"ep_id": g.ep_id.iloc[0], "steps": steps})
    json.dump(results, open(OUT / "qual.json", "w"), indent=1)
    print("QUAL_DONE", len(results))


if __name__ == "__main__":
    main()
