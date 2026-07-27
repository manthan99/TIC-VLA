#!/usr/bin/env python3
"""P6-M1: SFT with BEV tokens — the GA-VLN-style variant.

Adds the causal BEV splat (wildvln.bev_splat) as extra LLM tokens:
  - user text contains a run of <|bev|> placeholder tokens (one per
    non-empty splat cell, capped at BEV_MAX);
  - a forward hook on the language embedding layer replaces those
    positions with projector(cell features) + 2D sinusoidal metric PE
    (cell centers in meters, ego frame) + a learned modality embedding.
    The multimodal image merge is untouched (it runs after embedding).
  - trainable: LoRA adapters (LLM) + projector + modality embedding.

Usage:
  python -m wildvln.p6_sft_bev --steps 30 --run smoke_bev
  python -m wildvln.p6_sft_bev --epochs 2 --run m1
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from wildvln.bev_splat import BevSplatter, CELL_M, GRID_HALF_M
from wildvln.p6_sft import (MODEL, SAMPLES, OUT_ROOT, MAX_PIXELS,
                            build_text, collate)

BEV_TOKEN = "<|bev|>"
BEV_MAX = 256
FEAT_DIM = 2048


def sin_pe(xy_m, dim):
    """2D sinusoidal PE over metric coords in [-GRID_HALF_M, GRID_HALF_M]."""
    half = dim // 2
    pe = np.zeros((len(xy_m), dim), np.float32)
    for k, coord in enumerate(xy_m.T):          # x then y
        c = (coord + GRID_HALF_M) / (2 * GRID_HALF_M)   # [0, 1]
        div = np.exp(np.arange(0, half, 2) * -(np.log(200.0) / half))
        ang = c[:, None] * div[None, :] * 2 * np.pi * 20
        pe[:, k * half:k * half + len(div) * 2:2] = np.sin(ang)
        pe[:, k * half + 1:k * half + len(div) * 2:2] = np.cos(ang)
    return pe


@functools.lru_cache(maxsize=4)
def get_splatter(site, bag):
    return BevSplatter(site, bag)


@functools.lru_cache(maxsize=64)
def stamp_index(site, bag):
    idx = np.load(Path("/data/patelm/ticvla/wildvln/p2bf")
                  / site / bag / "index.npz")
    return {int(t * 1e9): i for i, t in enumerate(idx["t"])}


class BevVlnDataset(Dataset):
    def __init__(self, df, processor, bev_id, maneuver=False):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.bev_id = bev_id
        self.maneuver = maneuver

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        stamp = int(Path(row["image"]).stem)
        ki = stamp_index(row["site"], row["bag"])[stamp]
        cells, feats, counts = get_splatter(row["site"], row["bag"]).splat(ki)
        if len(cells) > BEV_MAX:
            keep = np.argsort(-counts)[:BEV_MAX]
            cells, feats = cells[keep], feats[keep]
        xy_m = cells.astype(np.float32) * CELL_M - GRID_HALF_M + CELL_M / 2
        pe = sin_pe(xy_m, FEAT_DIM)

        user, tgt = build_text(row, maneuver=self.maneuver)
        user += (f"\nBEV memory map ({len(cells)} occupied cells, "
                 f"+-12 m around you): " + BEV_TOKEN * len(cells))
        img = Image.open(row["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user}]}]
        prompt = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=prompt, images=[img], return_tensors="pt")
        tgt_ids = self.proc.tokenizer(
            tgt + self.proc.tokenizer.eos_token,
            return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([enc["input_ids"], tgt_ids], 1)[0]
        labels = torch.cat(
            [torch.full_like(enc["input_ids"], -100), tgt_ids], 1)[0]
        out = {"input_ids": input_ids, "labels": labels,
               "pixel_values": enc["pixel_values"],
               "image_grid_thw": enc["image_grid_thw"],
               "bev_feats": torch.from_numpy(feats),
               "bev_pe": torch.from_numpy(pe)}
        if "mm_token_type_ids" in enc:
            out["mm_token_type_ids"] = torch.cat(
                [enc["mm_token_type_ids"], torch.zeros_like(tgt_ids)], 1)[0]
        return out


def bev_collate(batch, pad_id):
    bev = [(b.pop("bev_feats"), b.pop("bev_pe")) for b in batch]
    out = collate(batch, pad_id)
    M = max(len(f) for f, _ in bev)
    feats = torch.zeros(len(batch), max(M, 1), FEAT_DIM)
    pes = torch.zeros(len(batch), max(M, 1), FEAT_DIM)
    for i, (f, p) in enumerate(bev):
        feats[i, :len(f)] = f
        pes[i, :len(p)] = p
    out["bev_feats"], out["bev_pe"] = feats, pes
    return out


class BevModel(nn.Module):
    """Wraps the (PEFT) VLM; swaps <|bev|> embeddings via a hook."""

    def __init__(self, vlm, bev_id, hidden):
        super().__init__()
        self.vlm = vlm
        self.bev_id = bev_id
        self.proj = nn.Sequential(nn.LayerNorm(FEAT_DIM),
                                  nn.Linear(FEAT_DIM, hidden))
        self.pe_proj = nn.Linear(FEAT_DIM, hidden, bias=False)
        self.modality = nn.Parameter(torch.zeros(hidden))
        # Qwen token embeddings are tiny-norm; a unit-scale projector makes
        # BEV tokens ~50x louder than text and stalls training (v1 plateaued
        # at loss ~1.0 vs M0's 0.3). Start as near-zero perturbations.
        nn.init.normal_(self.proj[1].weight, std=1e-3)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.normal_(self.pe_proj.weight, std=1e-4)
        self._ctx = None
        emb = self.vlm.get_input_embeddings()
        emb.register_forward_hook(self._swap)

    def _swap(self, module, inp, out):
        if self._ctx is None:
            return out
        ids, feats, pe = self._ctx
        if ids.shape[1] != out.shape[1]:     # generation steps past prefill
            return out
        mask = ids == self.bev_id
        if not mask.any():
            return out
        emb = (self.proj(feats) + self.pe_proj(pe)
               + self.modality).to(out.dtype)
        out = out.clone()
        for i in range(ids.shape[0]):
            m = mask[i]
            out[i, m] = emb[i, :int(m.sum())]
        return out

    def gradient_checkpointing_enable(self, **kw):
        self.vlm.gradient_checkpointing_enable(**kw)

    def forward(self, bev_feats=None, bev_pe=None, **kw):
        self._ctx = (kw["input_ids"], bev_feats, bev_pe)
        try:
            return self.vlm(**kw)
        finally:
            self._ctx = None

    @torch.no_grad()
    def generate(self, bev_feats=None, bev_pe=None, **kw):
        self._ctx = (kw["input_ids"], bev_feats, bev_pe)
        try:
            return self.vlm.generate(**kw)
        finally:
            self._ctx = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m1")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--turn-boost", type=int, default=1)
    ap.add_argument("--maneuver", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              Trainer, TrainingArguments)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    n_added = proc.tokenizer.add_special_tokens(
        {"additional_special_tokens": [BEV_TOKEN]})
    bev_id = proc.tokenizer.convert_tokens_to_ids(BEV_TOKEN)

    vlm = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if n_added:
        vlm.resize_token_embeddings(len(proc.tokenizer))
    vlm.config.use_cache = False
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*(visual|vision|merger).*")
    vlm = get_peft_model(vlm, lcfg)
    hidden = vlm.config.text_config.hidden_size
    model = BevModel(vlm, bev_id, hidden).to(torch.bfloat16)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.1f}M (incl. projector)")

    df = pd.read_parquet(SAMPLES)
    tdf = df[df.split == "train"]
    if args.turn_boost > 1:
        def _net_deg(t):
            p = np.array(json.loads(t))
            d = np.diff(p, axis=0)
            a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            return abs(np.degrees(a[-1] - a[0]))
        turn = tdf[tdf["trace"].map(_net_deg) >= 20.0]
        tdf = pd.concat([tdf] + [turn] * (args.turn_boost - 1))
        print(f"turn-boost {args.turn_boost}: {len(turn)} turn rows -> "
              f"{len(tdf)} total")
    tr = BevVlnDataset(tdf.sample(frac=1, random_state=0), proc, bev_id,
                       maneuver=args.maneuver)
    va = BevVlnDataset(df[df.split == "val"], proc, bev_id,
                       maneuver=args.maneuver)

    out = OUT_ROOT / args.run
    targs = TrainingArguments(
        output_dir=str(out), run_name=f"wildvln-{args.run}",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps else -1,
        learning_rate=args.lr, warmup_ratio=0.03,
        lr_scheduler_type="cosine", bf16=True,
        logging_steps=10, eval_strategy="steps", eval_steps=200,
        save_strategy="no",
        dataloader_num_workers=4, remove_unused_columns=False,
        gradient_checkpointing=True, report_to=[])
    pad = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    trainer = Trainer(model=model, args=targs,
                      train_dataset=tr, eval_dataset=va,
                      data_collator=lambda b: bev_collate(b, pad))
    trainer.train()
    (out / "final").mkdir(parents=True, exist_ok=True)
    model.vlm.save_pretrained(str(out / "final"))
    torch.save({"proj": model.proj.state_dict(),
                "pe_proj": model.pe_proj.state_dict(),
                "modality": model.modality.data,
                "bev_id": bev_id},
               out / "final" / "bev_head.pt")
    proc.save_pretrained(str(out / "final"))
    print("SFT_DONE", out / "final")


if __name__ == "__main__":
    main()
