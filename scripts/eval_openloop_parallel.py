#!/usr/bin/env python3
"""Parallel open-loop eval: N samples split across all visible GPUs.

Each worker process pins one GPU, evaluates its shard of a fixed-seed sample
list, and writes per-sample metrics to JSON; the parent aggregates ADE/FDE.

Usage:
    source .env.training
    python scripts/eval_openloop_parallel.py --num-samples 400 --output-dir "$TICVLA_OUTPUT_DIR/open_loop_eval_400"
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def worker(gpu_id: int, sample_ids: list, args, out_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import random

    import numpy as np
    import torch

    random.seed(args.seed + gpu_id)
    np.random.seed(args.seed + gpu_id)
    torch.manual_seed(args.seed + gpu_id)

    from ticvla.training.evaluate import TestConfig, TICVLATester

    config = TestConfig(
        data_dir=args.data_dir,
        num_test_samples=len(sample_ids),
        save_plots=False,
    )
    tester = TICVLATester(config)

    results = []
    for n, idx in enumerate(sample_ids):
        try:
            r = tester.test_model_inference(idx)
            gt = r["gt_waypoints"]
            if torch.is_tensor(gt):
                gt = gt.float().cpu().numpy()
            pred = r["pred_waypoints"]
            m = min(len(gt), len(pred))
            dists = np.linalg.norm(pred[:m, :2] - gt[:m, :2], axis=1)
            results.append({
                "sample_idx": int(idx),
                "sample_path": str(r["sample_path"]),
                "ade": float(dists.mean()),
                "fde": float(np.linalg.norm(pred[m - 1, :2] - gt[m - 1, :2])),
                "time_delay": float(r["time_delay"]),
                "inference_time": float(r["inference_time"]),
                "response_len": len(r["response"]),
            })
        except Exception as e:
            results.append({"sample_idx": int(idx), "error": str(e)})
        if (n + 1) % 10 == 0:
            print(f"[gpu{gpu_id}] {n + 1}/{len(sample_ids)} done", flush=True)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.environ.get("TICVLA_DATA_ROOT", ""), "DynaNav", "DynaNav_json"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.environ.get("TICVLA_OUTPUT_DIR", "outputs"), "open_loop_eval_parallel"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed-seed sample selection over the full dataset (dataset size probed once here).
    import numpy as np

    from ticvla.data.vlm_data import TICVLADataset_VLM
    n_total = len(TICVLADataset_VLM([args.data_dir], 90).samples)
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(n_total, size=min(args.num_samples, n_total), replace=False).tolist()
    shards = [ids[i::args.num_gpus] for i in range(args.num_gpus)]
    print(f"dataset size {n_total}; evaluating {len(ids)} samples on {args.num_gpus} GPUs")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    procs = []
    for g, shard in enumerate(shards):
        p = ctx.Process(target=worker, args=(g, shard, args, str(out_dir / f"shard_{g}.json")))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    results = []
    for g in range(args.num_gpus):
        f = out_dir / f"shard_{g}.json"
        if f.exists():
            results.extend(json.load(open(f)))
    ok = [r for r in results if "ade" in r]
    failed = [r for r in results if "error" in r]
    ades = [r["ade"] for r in ok]
    fdes = [r["fde"] for r in ok]
    summary = {
        "num_requested": len(ids),
        "num_ok": len(ok),
        "num_failed": len(failed),
        "ade_mean": float(np.mean(ades)),
        "ade_std": float(np.std(ades)),
        "ade_median": float(np.median(ades)),
        "fde_mean": float(np.mean(fdes)),
        "fde_std": float(np.std(fdes)),
        "fde_median": float(np.median(fdes)),
        "inference_time_mean": float(np.mean([r["inference_time"] for r in ok])),
        "wall_time_s": time.time() - t0,
        "checkpoint": os.environ.get("TICVLA_CHECKPOINT_PATH", ""),
        "data_dir": args.data_dir,
        "seed": args.seed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
