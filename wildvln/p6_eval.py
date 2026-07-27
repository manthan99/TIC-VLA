#!/usr/bin/env python3
"""P6 eval: trace ADE/FDE on val / test_site splits.

Rows: kinematic baseline (constant-curvature extrapolation of the history
polyline — the shortcut-learning control), and any SFT checkpoint
(parse <trace_bev>(x, y), ...</trace_bev> from generations).

Usage:
  python -m wildvln.p6_eval --baseline            # kinematic rows only
  python -m wildvln.p6_eval --ckpt p6/m0/final    # + model rows
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES = Path("/data/patelm/ticvla/wildvln/p5/samples.parquet")
TRACE_RE = re.compile(r"\(([-\d.]+),\s*([-\d.]+)\)")


def resample(pts, n=10):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return None
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 0.3:
        return None
    t = np.linspace(s[-1] / n, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, k]) for k in (0, 1)], 1)


def ade_fde(pred, gt):
    pred, gt = resample(pred), resample(gt)
    if pred is None or gt is None:
        return None
    e = np.linalg.norm(pred - gt, axis=1)
    return float(e.mean()), float(e[-1])


def kinematic(history):
    """Constant-curvature extrapolation from the last 10 m of history."""
    h = np.asarray(history, float)[::-1]     # history stored future->past
    if len(h) < 3:
        return None
    # history is in the CURRENT frame, ending at origin; fit turn rate
    v1 = -h[-1] + h[-2] if len(h) >= 2 else None
    # heading change per meter over the recent past
    d = np.diff(h, axis=0)
    seg = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dang = np.diff(np.unwrap(ang))
    good = seg[1:] > 0.05
    kappa = (dang[good] / seg[1:][good]).mean() if good.any() else 0.0
    kappa = float(np.clip(kappa, -0.5, 0.5))
    # roll forward from origin heading 0 (robot frame)
    pts, th, p = [], 0.0, np.zeros(2)
    for _ in range(10):
        th += kappa * 1.0
        p = p + np.array([np.cos(th), np.sin(th)])
        pts.append(p.copy())
    return np.array(pts)


def eval_rows(df, pred_fn, name, dump=None):
    out = {}
    recs = []
    for split in ("val", "test_site"):
        sub = df[(df.split == split) & (df["mode"] == "chained")
                 & (df.variant == 0)]
        ades, fdes, n_bad = [], [], 0
        for _, row in sub.iterrows():
            pred = pred_fn(row)
            if pred is None:
                n_bad += 1
                continue
            r = ade_fde(pred, json.loads(row["trace"]))
            if r is None:
                n_bad += 1
                continue
            ades.append(r[0])
            fdes.append(r[1])
            if dump:
                recs.append({"split": split, "ep_id": row["ep_id"],
                             "site": row["site"], "bag": row["bag"],
                             "step": int(row["step"]),
                             "pred": np.asarray(pred).tolist(),
                             "gt": json.loads(row["trace"]),
                             "ade": r[0], "fde": r[1]})
        out[split] = {"ade": float(np.mean(ades)), "fde": float(np.mean(fdes)),
                      "n": len(ades), "failed": n_bad}
        print(f"{name:12s} {split:9s} ADE {out[split]['ade']:.3f} "
              f"FDE {out[split]['fde']:.3f} (n={len(ades)}, fail {n_bad})")
    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"dumped {len(recs)} rows -> {dump}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--maneuver", action="store_true",
                    help="prompt format of --maneuver-trained checkpoints")
    args = ap.parse_args()
    df = pd.read_parquet(SAMPLES)
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)

    results = {"kinematic": eval_rows(
        df, lambda r: kinematic(json.loads(r["history"]))
        if r["history"] and r["history"] != "null" else None, "kinematic",
        dump=args.dump and args.dump.replace(".jsonl", "_kin.jsonl"))}

    if args.ckpt:
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from wildvln.p6_sft import MODEL, MAX_PIXELS, build_text
        mpath = args.model or MODEL
        proc = AutoProcessor.from_pretrained(mpath, max_pixels=MAX_PIXELS)
        base = AutoModelForImageTextToText.from_pretrained(
            mpath, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model = PeftModel.from_pretrained(base, args.ckpt).eval()

        @torch.no_grad()
        def model_pred(row):
            user, _ = build_text(row, maneuver=args.maneuver)
            img = Image.open(row["image"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user}]}]
            prompt = proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=prompt, images=[img],
                       return_tensors="pt").to("cuda:0")
            out = model.generate(**enc, max_new_tokens=400, do_sample=False)
            txt = proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            m = re.search(r"<trace_bev>(.*?)</trace_bev>", txt, re.S)
            if not m:
                return None
            try:
                pts = [(float(a), float(b))
                       for a, b in TRACE_RE.findall(m.group(1))]
            except ValueError:
                return None
            return np.array(pts) if len(pts) >= 2 else None

        results[args.ckpt] = eval_rows(df, model_pred, "sft",
                                       dump=args.dump)

    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
