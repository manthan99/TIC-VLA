#!/usr/bin/env python3
"""Eval the M1 BEV-token checkpoint: ADE/FDE on val / test_site.

Rebuilds BevModel (base + LoRA adapter + bev_head.pt), splats per sample,
generates with the embedding-swap hook.

Usage: python -m wildvln.p6_eval_bev --ckpt p6/m1/final
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wildvln.bev_splat import CELL_M, GRID_HALF_M
from wildvln.p6_eval import TRACE_RE, eval_rows, kinematic
from wildvln.p6_sft import MODEL, MAX_PIXELS, build_text
from wildvln.p6_sft_bev import (BEV_MAX, BEV_TOKEN, BevModel, get_splatter,
                                sin_pe, stamp_index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--maneuver", action="store_true")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    proc.tokenizer.add_special_tokens(
        {"additional_special_tokens": [BEV_TOKEN]})
    bev_id = proc.tokenizer.convert_tokens_to_ids(BEV_TOKEN)

    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base.resize_token_embeddings(len(proc.tokenizer))
    vlm = PeftModel.from_pretrained(base, args.ckpt).eval()
    hidden = vlm.config.text_config.hidden_size
    model = BevModel(vlm, bev_id, hidden).to(torch.bfloat16).to("cuda:0")
    head = torch.load(Path(args.ckpt) / "bev_head.pt", weights_only=True)
    model.proj.load_state_dict(head["proj"])
    model.pe_proj.load_state_dict(head["pe_proj"])
    model.modality.data = head["modality"].to("cuda:0")
    model.eval()

    df = pd.read_parquet("/data/patelm/ticvla/wildvln/p5/samples.parquet")

    @torch.no_grad()
    def pred(row):
        stamp = int(Path(row["image"]).stem)
        ki = stamp_index(row["site"], row["bag"])[stamp]
        cells, feats, counts = get_splatter(row["site"], row["bag"]).splat(ki)
        if len(cells) > BEV_MAX:
            keep = np.argsort(-counts)[:BEV_MAX]
            cells, feats = cells[keep], feats[keep]
        xy_m = cells.astype(np.float32) * CELL_M - GRID_HALF_M + CELL_M / 2
        pe = sin_pe(xy_m, feats.shape[1])

        user, _ = build_text(row, maneuver=args.maneuver)
        user += (f"\nBEV memory map ({len(cells)} occupied cells, "
                 f"+-12 m around you): " + BEV_TOKEN * len(cells))
        img = Image.open(row["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user}]}]
        prompt = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=prompt, images=[img],
                   return_tensors="pt").to("cuda:0")
        bf = torch.from_numpy(feats)[None].to("cuda:0", torch.bfloat16)
        bp = torch.from_numpy(pe)[None].to("cuda:0", torch.bfloat16)
        out = model.generate(bev_feats=bf, bev_pe=bp, **enc,
                             max_new_tokens=400, do_sample=False)
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

    results = {args.ckpt: eval_rows(df, pred, "sft-bev", dump=args.dump)}
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
