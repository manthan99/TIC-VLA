#!/usr/bin/env python3
"""P2c-2: RynnBrain ViT feature cache for every keyframe.

Caches the MERGED vision tokens (pooler_output: 2x2-merged 32-px patches,
2048-d — exactly the image tokens the LLM consumes). BEV lifting splats
these into metric cells through the accumulated patch depth; caching the
post-merger features keeps the splatted tokens in the LLM's input space
(GA-VLN recipe).

One npz per bag: feats (F, N, 2048) fp16, t (F,), grid_hw — resumable at
bag granularity.

    CUDA_VISIBLE_DEVICES=g python -m wildvln.p2c_vitfeat --shard i --nshards N
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

P2B_ROOT = Path("/data/patelm/ticvla/wildvln/p2b")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p2c/vit")
MODEL = "/data/patelm/ticvla/RynnBrain1.1-2B"
BATCH = 16


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

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    vis = getattr(model, "visual", None) or getattr(model.model, "visual")

    for site, bag, kdir in bags()[args.shard::args.nshards]:
        out = OUT_ROOT / site
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{bag}.npz"
        if dst.exists():
            continue
        frames = sorted(kdir.glob("*.jpg"))
        feats, ts, grid_hw = [], [], None
        for i in range(0, len(frames), BATCH):
            chunk = frames[i:i + BATCH]
            imgs = [Image.open(f) for f in chunk]
            ip = proc.image_processor(images=imgs, return_tensors="pt")
            pv = ip["pixel_values"].to("cuda", torch.bfloat16)
            thw = ip["image_grid_thw"].to("cuda")
            with torch.no_grad():
                o = vis(pv, grid_thw=thw)
            n_tok = (thw[:, 1] * thw[:, 2] // 4).tolist()
            if grid_hw is None:
                grid_hw = (int(thw[0, 1]) // 2, int(thw[0, 2]) // 2)
            for f, fe in zip(chunk, o.pooler_output.split(n_tok)):
                feats.append(fe.to(torch.float16).cpu().numpy())
                ts.append(int(f.stem) / 1e9)
        np.savez_compressed(
            dst, feats=np.stack(feats), t=np.array(ts),
            grid_hw=np.array(grid_hw))
        print(f"[shard {args.shard}] {site}/{bag}: {len(feats)} frames "
              f"grid {grid_hw}", flush=True)


if __name__ == "__main__":
    main()
