#!/usr/bin/env python3
"""Diagnose the M1 loss gap (~1.2 vs M0's ~0.3 train loss).

Per-sample teacher-forced loss, decomposed by target region
(<think> / <memory> / <trace_bev>), under three conditions:
  m0        : M0 checkpoint, M0 prompt                  (control)
  m1        : M1 checkpoint, full BEV prompt + injection
  m1-nobev  : M1 checkpoint, M0 prompt (no BEV segment) — isolates
              whether the BEV tokens or the surgery (resize/wrapper)
              carries the gap.

Usage: CUDA_VISIBLE_DEVICES=0 python -m wildvln.p6_diag_bev --n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wildvln.bev_splat import CELL_M, GRID_HALF_M
from wildvln.p6_sft import MODEL, MAX_PIXELS, build_text
from wildvln.p6_sft_bev import (BEV_MAX, BEV_TOKEN, BevModel, get_splatter,
                                sin_pe, stamp_index)

P6 = Path("/data/patelm/ticvla/wildvln/p6")


def region_masks(tok, tgt_ids, tgt):
    """Map target token spans to think/memory/trace regions by char offset."""
    spans = {}
    for tag in ("think", "memory", "trace_bev"):
        a = tgt.find(f"<{tag}>")
        if a < 0:
            continue
        b = tgt.find(f"</{tag}>") + len(f"</{tag}>")
        spans[tag] = (a, b)
    # cumulative char position per token via re-decode
    offs, pos = [], 0
    for t in tgt_ids:
        s = tok.decode([t])
        offs.append((pos, pos + len(s)))
        pos += len(s)
    masks = {k: np.array([o[0] >= a and o[0] < b for o in offs])
             for k, (a, b) in spans.items()}
    return masks


@torch.no_grad()
def sample_loss(model, proc, row, user, tgt, bev=None):
    img = Image.open(row["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": user}]}]
    prompt = proc.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=prompt, images=[img], return_tensors="pt")
    tgt_ids = proc.tokenizer(
        tgt + proc.tokenizer.eos_token,
        return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = torch.cat([enc["input_ids"], tgt_ids], 1).cuda()
    labels = torch.cat(
        [torch.full_like(enc["input_ids"], -100), tgt_ids], 1).cuda()
    kw = {"input_ids": ids, "labels": labels,
          "attention_mask": torch.ones_like(ids),
          "pixel_values": enc["pixel_values"].cuda(),
          "image_grid_thw": enc["image_grid_thw"].cuda()}
    if "mm_token_type_ids" in enc:
        kw["mm_token_type_ids"] = torch.cat(
            [enc["mm_token_type_ids"],
             torch.zeros_like(tgt_ids)], 1).cuda()
    if bev is not None:
        out = model(bev_feats=bev[0], bev_pe=bev[1], **kw)
    else:
        out = model(**kw)
    logits = out.logits[0, :-1]
    lab = labels[0, 1:]
    keep = lab != -100
    ce = torch.nn.functional.cross_entropy(
        logits[keep].float(), lab[keep], reduction="none").cpu().numpy()
    # region decomposition over the target tokens (order preserved)
    masks = region_masks(proc.tokenizer, tgt_ids[0].tolist(), tgt)
    n = len(tgt_ids[0])
    ce_tgt = ce[-n:] if len(ce) >= n else ce
    rd = {k: float(ce_tgt[:len(m)][m[:len(ce_tgt)]].mean())
          for k, m in masks.items() if m[:len(ce_tgt)].any()}
    rd["all"] = float(ce.mean())
    return rd


def bev_inputs(row):
    stamp = int(Path(row["image"]).stem)
    ki = stamp_index(row["site"], row["bag"])[stamp]
    cells, feats, counts = get_splatter(row["site"], row["bag"]).splat(ki)
    if len(cells) > BEV_MAX:
        keep = np.argsort(-counts)[:BEV_MAX]
        cells, feats = cells[keep], feats[keep]
    xy = cells.astype(np.float32) * CELL_M - GRID_HALF_M + CELL_M / 2
    pe = sin_pe(xy, feats.shape[1])
    bf = torch.from_numpy(feats)[None].cuda().to(torch.bfloat16)
    bp = torch.from_numpy(pe)[None].cuda().to(torch.bfloat16)
    return len(cells), bf, bp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--m1", default=str(P6 / "m1" / "final"))
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    df = pd.read_parquet("/data/patelm/ticvla/wildvln/p5/samples.parquet")
    rows = df[(df.split == "val") & (df["mode"] == "chained")
              & (df.variant == 0)].head(args.n)

    agg = {}

    def add(name, rd):
        agg.setdefault(name, []).append(rd)

    # ---- M0 control ----
    proc0 = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    base0 = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    m0 = PeftModel.from_pretrained(base0, str(P6 / "m0" / "final")).eval()
    for _, row in rows.iterrows():
        user, tgt = build_text(row)
        add("m0", sample_loss(m0, proc0, row, user, tgt))
    del m0, base0
    torch.cuda.empty_cache()

    # ---- M1 with and without BEV segment ----
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    proc.tokenizer.add_special_tokens(
        {"additional_special_tokens": [BEV_TOKEN]})
    bev_id = proc.tokenizer.convert_tokens_to_ids(BEV_TOKEN)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base.resize_token_embeddings(len(proc.tokenizer))
    vlm = PeftModel.from_pretrained(base, args.m1).eval()
    model = BevModel(vlm, bev_id,
                     vlm.config.text_config.hidden_size).to(
                         torch.bfloat16).cuda()
    head = torch.load(Path(args.m1) / "bev_head.pt", weights_only=True)
    model.proj.load_state_dict(head["proj"])
    model.pe_proj.load_state_dict(head["pe_proj"])
    model.modality.data = head["modality"].cuda()
    model.eval()

    for _, row in rows.iterrows():
        user, tgt = build_text(row)
        ncells, bf, bp = bev_inputs(row)
        user_bev = user + (f"\nBEV memory map ({ncells} occupied cells, "
                           f"+-12 m around you): " + BEV_TOKEN * ncells)
        add("m1", sample_loss(model, proc, row, user_bev, tgt, bev=(bf, bp)))
        add("m1-nobev", sample_loss(model, proc, row, user, tgt))

    for name, rds in agg.items():
        keys = sorted({k for rd in rds for k in rd})
        line = " ".join(
            f"{k} {np.mean([rd[k] for rd in rds if k in rd]):.3f}"
            for k in keys)
        print(f"DIAG {name:9s} {line}")


if __name__ == "__main__":
    main()
