#!/usr/bin/env python3
"""Wild VLN trace-policy HTTP server (DynaNav sim prototype).

Wraps a wildvln SFT/GRPO adapter checkpoint behind a tiny JSON API so
the DynaNav Isaac Sim benchmark on another machine can query it
(DynaNav/wildvln_remote.py is the matching client shim).

POST /act  {"image_b64": <jpeg/png b64>, "instruction": str,
            "memory": optional dict, "history": optional [[x,y],...]}
->         {"trace": [[x_fwd,y_left],...10], "cot": str,
            "memory_out": dict|null, "raw": str, "ms": float}
GET /health -> {"ok": true, "ckpt": ...}

Run:  CUDA_VISIBLE_DEVICES=3 python -m wildvln.dn_server \
          --ckpt /data/patelm/ticvla/wildvln/p6/grpo1/final --port 8121
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from wildvln.p6_sft import MAX_PIXELS, build_text

STATE = {}


def load(ckpt, model_path):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path, max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload().eval()      # dense weights: ~2-3x decode
    STATE.update(proc=proc, model=model, ckpt=ckpt)


def parse_trace(text):
    m = re.search(r"<trace_bev>(.*?)</trace_bev>", text, re.S)
    if not m:
        return None
    try:
        pts = [(float(a), float(b)) for a, b in
               re.findall(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", m.group(1))]
    except ValueError:
        return None
    pts = [(x, y) for x, y in pts if abs(x) < 25 and abs(y) < 25]
    return pts or None


def act(req):
    t0 = time.time()
    img = Image.open(io.BytesIO(
        base64.b64decode(req["image_b64"]))).convert("RGB")
    mem = req.get("memory")
    row = dict(instruction=req["instruction"],
               mode="chained" if mem else "t0",
               memory_in=json.dumps(mem) if mem else "",
               history=json.dumps(req["history"]) if req.get("history") else "",
               trace="[]", memory_out="", cot="")
    user, _ = build_text(row)
    proc, model = STATE["proc"], STATE["model"]
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": user}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    enc = proc(text=prompt, images=[img], return_tensors="pt").to("cuda")
    t_pre = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False)
    t_gen = time.time()
    n_new = int(out.shape[1] - enc["input_ids"].shape[1])
    text = proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    trace = parse_trace(text)
    cot = (re.search(r"<think>(.*?)</think>", text, re.S) or [None, ""])[1]
    mo = re.search(r"<memory>(.*?)</memory>", text, re.S)
    try:
        memory_out = json.loads(mo.group(1)) if mo else None
    except json.JSONDecodeError:
        memory_out = None
    return {"trace": trace, "cot": cot.strip() if cot else "",
            "memory_out": memory_out, "raw": text,
            "ms": round(1e3 * (time.time() - t0), 1),
            "ms_pre": round(1e3 * (t_pre - t0), 1),
            "ms_gen": round(1e3 * (t_gen - t_pre), 1),
            "n_new_tokens": n_new}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": True, "ckpt": STATE.get("ckpt")})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            self._send(200, act(req))
        except Exception as e:            # noqa: BLE001 — surface to client
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/data/patelm/ticvla/wildvln/p6/grpo1/final")
    ap.add_argument("--model", default="/data/patelm/ticvla/RynnBrain1.1-2B")
    ap.add_argument("--port", type=int, default=8121)
    args = ap.parse_args()
    load(args.ckpt, args.model)
    print(f"DN_SERVER_READY port {args.port} ckpt {args.ckpt}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
