#!/usr/bin/env python3
"""GrandTour BEV splat caches, part 2: RynnBrain ViT features.

Same recipe as p2c_vitfeat (merged vision tokens = the LLM's image-token
space, 2048-d fp16) but only at the distance-spaced splat keyframes from
gt_splatcache.splat_kfs — GrandTour kfs are 0.06 m apart, caching all of
them would be ~40x waste.

Output: /data/patelm/ticvla/grandtour/p2c/vit/<bag>.npz
  feats (F, N, 2048) fp16, kf_i (F,), t (F,), grid_hw (2,)

Usage: CUDA_VISIBLE_DEVICES=1 python -m wildvln.gt_vitcache
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wildvln.gt_splatcache import P2B, splat_kfs

OUT = Path("/data/patelm/ticvla/grandtour/p2c/vit")
MODEL = "/data/patelm/ticvla/RynnBrain1.1-2B"
BATCH = 16


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    vis = getattr(model, "visual", None) or getattr(model.model, "visual")

    bags = sorted(p.name for p in P2B.iterdir()
                  if (p / "index.npz").exists())
    OUT.mkdir(parents=True, exist_ok=True)
    for bag in bags:
        dst = OUT / f"{bag}.npz"
        if dst.exists():
            continue
        idx = np.load(P2B / bag / "index.npz")
        picks = splat_kfs(idx)
        # match files by nearest stamp — ns stamps exceed float64
        # integer precision, so int(t*1e9) reconstruction is unsafe
        kdir = P2B / bag / "keyframes"
        files = sorted(kdir.glob("*.jpg"))
        fns = np.array([int(f.stem) for f in files], np.int64)
        want = (idx["t"][picks] * 1e9).astype(np.int64)
        near = np.searchsorted(fns, want)
        frames = []
        for w, j in zip(want, near):
            cand = [c for c in (j - 1, j, j + 1) if 0 <= c < len(fns)]
            best = min(cand, key=lambda c: abs(int(fns[c]) - int(w)))
            assert abs(int(fns[best]) - int(w)) < 1_000_000, bag
            frames.append(files[best])
        feats, grid_hw = [], None
        for i in range(0, len(frames), BATCH):
            imgs = [Image.open(f) for f in frames[i:i + BATCH]]
            ip = proc.image_processor(images=imgs, return_tensors="pt")
            pv = ip["pixel_values"].to("cuda", torch.bfloat16)
            thw = ip["image_grid_thw"].to("cuda")
            with torch.no_grad():
                o = vis(pv, grid_thw=thw)
            n_tok = (thw[:, 1] * thw[:, 2] // 4).tolist()
            if grid_hw is None:
                grid_hw = (int(thw[0, 1]) // 2, int(thw[0, 2]) // 2)
            for fe in o.pooler_output.split(n_tok):
                feats.append(fe.to(torch.float16).cpu().numpy())
        np.savez_compressed(dst, feats=np.stack(feats), kf_i=picks,
                            t=idx["t"][picks], grid_hw=np.array(grid_hw))
        print(f"{bag}: {len(feats)} kfs grid {grid_hw}", flush=True)
    print("GT_VITCACHE_DONE")


if __name__ == "__main__":
    main()
