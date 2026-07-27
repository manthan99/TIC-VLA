#!/usr/bin/env python3
"""P6-M0: SFT RynnBrain1.1-2B on Wild VLN chained samples — image +
language ONLY (no BEV tokens). Establishes whether the instruction /
memory / CoT scaffolding trains before any model surgery.

Sample formatting (raw keyframe image — never the overlay):
  user   : instruction + memory_in (chained) + history polyline + task spec
  target : <think>cot</think>\n<memory>{json}</memory>\n
           <trace_bev>(x, y), ...</trace_bev>          (chained)
           <trace_bev>(x, y), ...</trace_bev>          (t0)
Trace in metric meters, robot frame (x fwd, y left), 2 decimals — the
frame the BEV-token variant will share. Loss on assistant tokens only.
LoRA PEFT (user decision): adapters on the LLM attention+MLP projections,
vision tower + merger frozen; only adapters are saved.

Usage:
  python -m wildvln.p6_sft --steps 100 --run smoke     # sanity
  python -m wildvln.p6_sft --epochs 2 --run m0         # full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MODEL = "/data/patelm/ticvla/RynnBrain1.1-2B"          # default; --model overrides
SAMPLES = Path("/data/patelm/ticvla/wildvln/p5/samples.parquet")
OUT_ROOT = Path("/data/patelm/ticvla/wildvln/p6")
MAX_PIXELS = 640 * 400          # cap ViT tokens (~250) for training speed


def fmt_pts(pts):
    return ", ".join(f"({x:.2f}, {y:.2f})" for x, y in pts)


def trace_maneuver(trace):
    """Discrete turn label from the GT trace (x fwd, y left)."""
    p = np.asarray(trace, float)
    d = np.diff(p, axis=0)
    a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    deg = np.degrees(a[-1] - a[0])
    return "left" if deg >= 20 else "right" if deg <= -20 else "straight"


def build_text(row, maneuver=False):
    instr = row["instruction"]
    parts = ["You are a ground robot navigating outdoors. The image is "
             "your current forward camera view.",
             f"Instruction: {instr}"]
    chained = row["mode"] == "chained"
    if chained:
        parts.append(f"Memory (landmarks passed, turns made, progress, "
                     f"current subgoal): {row['memory_in']}")
    hist = json.loads(row["history"]) if row["history"] else None
    if hist:
        parts.append("Path just driven (robot frame, meters, x forward / "
                     f"y left, oldest first): {fmt_pts(hist[::-1])}")
    if chained and maneuver:
        parts.append("Think step by step about where you are relative to "
                     "the instruction, update your memory, commit to your "
                     "next maneuver as <maneuver>straight|left|right"
                     "</maneuver>, then predict your next trace of up to "
                     "10 meters as <trace_bev>(x1, y1), (x2, y2), ..."
                     "</trace_bev> in meters. The trace must execute the "
                     "committed maneuver.")
    elif chained:
        parts.append("Think step by step about where you are relative to "
                     "the instruction, update your memory, then predict "
                     "your next trace of up to 10 meters as <trace_bev>"
                     "(x1, y1), (x2, y2), ...</trace_bev> in meters.")
    else:
        parts.append("Predict your next trace of up to 10 meters as "
                     "<trace_bev>(x1, y1), (x2, y2), ...</trace_bev> "
                     "in meters, x forward, y left.")
    user = "\n".join(parts)

    trace = json.loads(row["trace"])
    tgt = f"<trace_bev>{fmt_pts(trace)}</trace_bev>"
    if chained and maneuver:
        tgt = f"<maneuver>{trace_maneuver(trace)}</maneuver>\n" + tgt
    if chained:
        mem = row["memory_out"]
        tgt = f"<think>{row['cot']}</think>\n<memory>{mem}</memory>\n" + tgt
    return user, tgt


class VlnDataset(Dataset):
    def __init__(self, df, processor, maneuver=False):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.maneuver = maneuver

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        user, tgt = build_text(row, maneuver=self.maneuver)
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
               "image_grid_thw": enc["image_grid_thw"]}
        if "mm_token_type_ids" in enc:      # Qwen3.5 M-RoPE; absent on Qwen3-VL
            out["mm_token_type_ids"] = torch.cat(
                [enc["mm_token_type_ids"], torch.zeros_like(tgt_ids)], 1)[0]
        return out


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    has_mm = "mm_token_type_ids" in batch[0]
    mm = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
        if has_mm:
            mm[i, :n] = b["mm_token_type_ids"]
    out = {"input_ids": ids, "labels": lab, "attention_mask": att,
           "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
           "image_grid_thw": torch.cat(
               [b["image_grid_thw"] for b in batch])}
    if has_mm:
        out["mm_token_type_ids"] = mm
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m0")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--turn-boost", type=int, default=1,
                    help="duplicate train rows whose GT trace turns >=20deg "
                         "this many times (counters the 75%% straight prior)")
    ap.add_argument("--maneuver", action="store_true",
                    help="insert a <maneuver>straight|left|right</maneuver> "
                         "commitment token between memory and trace")
    ap.add_argument("--samples", nargs="+", default=[str(SAMPLES)],
                    help="one or more p5 parquets (rows share one schema); "
                         "train/val splits are concatenated across them")
    args = ap.parse_args()
    mpath = args.model

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              Trainer, TrainingArguments)
    proc = AutoProcessor.from_pretrained(mpath, max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        mpath, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # adapters on the LLM only — leave the ViT + merger frozen
        exclude_modules=r".*(visual|vision|merger).*")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    df = pd.concat([pd.read_parquet(p) for p in args.samples],
                   ignore_index=True)
    for p in args.samples:
        print("samples:", p)
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
    tr = VlnDataset(tdf.sample(frac=1, random_state=0), proc,
                    maneuver=args.maneuver)
    va = VlnDataset(df[df.split == "val"], proc, maneuver=args.maneuver)
    print(f"train {len(tr)}, val {len(va)}")

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
        save_strategy="steps", save_steps=400, save_total_limit=2,
        dataloader_num_workers=8, remove_unused_columns=False,
        gradient_checkpointing=True, report_to=[])
    pad = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    trainer = Trainer(model=model, args=targs,
                      train_dataset=tr, eval_dataset=va,
                      data_collator=lambda b: collate(b, pad))
    trainer.train()
    trainer.save_model(str(out / "final"))
    proc.save_pretrained(str(out / "final"))
    print("SFT_DONE", out / "final")


if __name__ == "__main__":
    main()
