#!/usr/bin/env python3
"""P6 GRPO: RL round on an SFT checkpoint with a geometric trace reward.

Per prompt: sample G completions (temperature), reward each by the
negative ADE of its parsed trace vs GT (format failures / insane coords
get the floor reward), advantage = group-normalized reward (GRPO),
loss = -advantage * mean token logprob + beta * k3-KL to the frozen
SFT policy (LoRA adapters disabled = reference — no second model).

The reward never sees tokens — only geometry. The model is free to move
turn placement, curvature, everything, as long as the trace improves.

Usage:
  python -m wildvln.p6_grpo --ckpt p6/m3/final --run grpo1 \
      --prompts 1500 --G 8
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

from wildvln.p6_eval import TRACE_RE, resample
from wildvln.p6_sft import MODEL, MAX_PIXELS, OUT_ROOT, SAMPLES, build_text

R_FLOOR = -4.0


def trace_reward(txt, gt):
    m = re.search(r"<trace_bev>(.*?)</trace_bev>", txt, re.S)
    if not m:
        return R_FLOOR
    try:
        pts = [(float(a), float(b)) for a, b in TRACE_RE.findall(m.group(1))]
    except ValueError:
        return R_FLOOR
    if len(pts) < 2:
        return R_FLOOR
    p = np.asarray(pts, float)
    if np.abs(p).max() > 25:
        return R_FLOOR
    pr, gr = resample(p), resample(np.asarray(gt, float))
    if pr is None or gr is None:
        return R_FLOOR
    ade = float(np.linalg.norm(pr - gr, axis=1).mean())
    return -min(ade, -R_FLOOR)


def pick_prompts(df, n, seed=0):
    """Train chained rows; turn steps first-class, straights to fill."""
    tr = df[(df.split == "train") & (df["mode"] == "chained")].copy()

    def net_deg(t):
        p = np.array(json.loads(t))
        d = np.diff(p, axis=0)
        a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        return abs(np.degrees(a[-1] - a[0]))
    tr["nh"] = tr["trace"].map(net_deg)
    rng = np.random.default_rng(seed)
    turn = tr[tr.nh >= 20]
    straight = tr[tr.nh < 20]
    n_turn = min(len(turn), int(n * 0.6))
    picks = pd.concat([
        turn.sample(n=n_turn, random_state=seed),
        straight.sample(n=min(len(straight), n - n_turn),
                        random_state=seed)])
    return picks.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", default="grpo1")
    ap.add_argument("--prompts", type=int, default=1500)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=300)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    model = PeftModel.from_pretrained(base, args.ckpt, is_trainable=True)
    # frozen copy of the SFT adapter = KL reference (disable_adapter()
    # would reference the PRE-SFT base and pull training away from SFT)
    model.load_adapter(args.ckpt, adapter_name="ref")
    model.set_adapter("default")
    model.eval()          # no dropout: sampled and scored policy match
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)

    df = pd.read_parquet(SAMPLES)
    prompts = pick_prompts(df, args.prompts)
    out = OUT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)

    eos = proc.tokenizer.eos_token_id
    pad = proc.tokenizer.pad_token_id or eos
    r_hist, kl_hist = [], []

    for step, (_, row) in enumerate(prompts.iterrows()):
        user, _ = build_text(row, maneuver=True)
        img = Image.open(row["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user}]}]
        ptxt = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=ptxt, images=[img],
                   return_tensors="pt").to("cuda:0")
        plen = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temp, top_p=0.95,
                num_return_sequences=args.G, max_new_tokens=350,
                pad_token_id=pad)
        comps = gen[:, plen:]
        gt = json.loads(row["trace"])
        texts = proc.tokenizer.batch_decode(comps, skip_special_tokens=True)
        rewards = np.array([trace_reward(t, gt) for t in texts], np.float32)
        r_hist.append(float(rewards.mean()))
        if rewards.std() < 1e-4:
            continue                       # no learning signal in this group
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        # token logprobs of the sampled completions (policy + reference)
        ids = gen
        att = (ids != pad).long()
        att[:, :plen] = enc["attention_mask"].repeat(args.G, 1)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i in range(args.G):
            n = int((comps[i] != pad).sum())
            mask[i, plen:plen + n] = True
        kw = {"input_ids": ids, "attention_mask": att,
              "pixel_values": enc["pixel_values"].repeat(args.G, 1),
              "image_grid_thw": enc["image_grid_thw"].repeat(args.G, 1)}
        if "mm_token_type_ids" in enc:
            mm = torch.zeros_like(ids)
            mm[:, :plen] = enc["mm_token_type_ids"].repeat(args.G, 1)
            kw["mm_token_type_ids"] = mm

        def token_logp(logits):
            # CE keeps memory sane vs a full-vocab float log_softmax
            return -torch.nn.functional.cross_entropy(
                logits[:, :-1].transpose(1, 2), ids[:, 1:],
                reduction="none")

        logp = token_logp(model(**kw).logits)
        with torch.no_grad():
            model.set_adapter("ref")
            logp_ref = token_logp(model(**kw).logits)
            model.set_adapter("default")

        m = mask[:, 1:]
        per_tok = logp * m
        n_tok = m.sum(1).clamp(min=1)
        pg = -(torch.tensor(adv, device=logp.device)
               * (per_tok.sum(1) / n_tok)).mean()
        d = (logp_ref - logp) * m
        kl = (torch.exp(d) - d - 1).sum(1) / n_tok       # k3 estimator
        loss = pg + args.beta * kl.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        kl_hist.append(float(kl.mean()))

        if (step + 1) % 10 == 0:
            print(f"GRPO step {step+1}/{len(prompts)} "
                  f"r(mean last50) {np.mean(r_hist[-50:]):.3f} "
                  f"kl {np.mean(kl_hist[-50:]):.4f}", flush=True)
        if (step + 1) % args.save_every == 0:
            model.save_pretrained(str(out / f"step{step+1}"))
    model.save_pretrained(str(out / "final"))
    proc.save_pretrained(str(out / "final"))
    json.dump({"r_hist": r_hist, "kl_hist": kl_hist},
              open(out / "history.json", "w"))
    print("GRPO_DONE", out / "final")


if __name__ == "__main__":
    main()
