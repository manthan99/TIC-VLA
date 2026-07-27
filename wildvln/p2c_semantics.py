#!/usr/bin/env python3
"""P2c-1: Mask2Former-ADE semantic label maps for every keyframe.

One uint8 ADE-150 label PNG per keyframe. Downstream consumers:
  - dynamics masking in the depth cache (drop LiDAR points on person/vehicle
    pixels of each scan's nearest keyframe BEFORE accumulation — kills the
    smear streaks seen in depth_visuals7);
  - semantic voxel painting + dynamics deletion in the privileged map;
  - ground masks for BEV supervision (ADE_GROUND, see p2b_calibrate_sem).

Resumable: existing outputs are skipped. Shard by bag:

    CUDA_VISIBLE_DEVICES=g python -m wildvln.p2c_semantics --shard i --nshards N
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p2c/sem")
SEG_MODEL = "/data/patelm/ticvla/depth_models/mask2former-ade"
BATCH = 8

# ADE-150 ids of things that move (matched by name from the model's id2label
# at load time and asserted against this set — never trust hardcoded ids).
DYNAMIC_NAMES = ("person", "car", "bus", "truck", "van", "boat", "airplane",
                 "bicycle", "minibike", "animal", "ship")


def dynamic_ids(id2label) -> set:
    ids = {int(i) for i, n in id2label.items()
           if any(w in n.lower().split(", ")[0] for w in DYNAMIC_NAMES)}
    assert ids, "no dynamic classes matched id2label"
    return ids


def bags():
    out = []
    for site in sorted(P2B_ROOT.iterdir()):
        if site.name.startswith("_") or not site.is_dir():
            continue
        for bag in sorted(site.iterdir()):
            if (bag / "keyframes").is_dir():
                out.append((site.name, bag.name, bag / "keyframes"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        SEG_MODEL, dtype=torch.float16).to("cuda").eval()
    dyn = dynamic_ids(model.config.id2label)
    print(f"dynamic ids: {sorted(dyn)}")

    todo = bags()[args.shard::args.nshards]
    for site, bag, kdir in todo:
        out = OUT_ROOT / site / bag
        out.mkdir(parents=True, exist_ok=True)
        frames = sorted(kdir.glob("*.jpg"))
        frames = [f for f in frames
                  if not (out / f.name.replace(".jpg", ".png")).exists()]
        done = 0
        for i in range(0, len(frames), BATCH):
            chunk = frames[i:i + BATCH]
            imgs = [Image.open(f) for f in chunk]
            with torch.no_grad():
                inp = proc(images=imgs, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].half()
                res = model(**inp)
                sems = proc.post_process_semantic_segmentation(
                    res, target_sizes=[im.size[::-1] for im in imgs])
            for f, sem in zip(chunk, sems):
                lab = sem.cpu().numpy().astype(np.uint8)
                Image.fromarray(lab, mode="L").save(
                    out / f.name.replace(".jpg", ".png"), optimize=True)
            done += len(chunk)
        print(f"[shard {args.shard}] {site}/{bag}: +{done} "
              f"({len(list(out.glob('*.png')))} total)", flush=True)


if __name__ == "__main__":
    main()
