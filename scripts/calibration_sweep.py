#!/usr/bin/env python3
"""Sweep camera pitch/height/FOV on one frame to find the calibration by eye.

Each tile overlays a metric ground grid for a different parameter combination.
The correct combination is the one where the grid lies flat on the real ground:
rails parallel to the path, range bars on the right spots, horizon on the true
horizon.

Usage:
    python scripts/calibration_sweep.py --dataset GND --sweep pitch_height
    python scripts/calibration_sweep.py --sample <json> --sweep hfov
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ticvla.data.projection import future_offsets_in_window, get_camera, image_trace_from_sample

import sys

sys.path.insert(0, str(Path(__file__).parent))
from visualize_trace_sample import _resolve, detect_dataset, draw_ground_grid  # noqa: E402

DATASET_ROOTS = {
    "DynaNav": "/data/patelm/ticvla/dataset/DynaNav/DynaNav_json",
    "SCAND": "/data/patelm/ticvla/dataset/SCAND/SCAND_json",
    "GND": "/data/patelm/ticvla/dataset/GND/GND_json",
}


def pick_frame(dataset: str, seed: int, min_path_m: float = 8.0) -> Path:
    """Find a frame with enough forward travel for the grid to be informative."""
    rng = random.Random(seed)
    root = Path(DATASET_ROOTS[dataset])
    folders = [d for d in root.iterdir() if d.is_dir()]
    for _ in range(300):
        folder = rng.choice(folders)
        candidates = [p for p in folder.glob("*.json") if not p.name.startswith(".")]
        if not candidates:
            continue
        candidate = rng.choice(candidates)
        offsets = future_offsets_in_window(json.load(open(candidate, "r")))
        if len(offsets) > 1:
            length = float(np.linalg.norm(np.diff(offsets[:, :2], axis=0), axis=1).sum())
            if length >= min_path_m:
                return candidate
    raise RuntimeError(f"No suitable frame found in {root}")


def build_variants(base, sweep: str) -> List[Tuple[str, object]]:
    """Parameter combinations to render, as (label, camera) pairs."""
    variants: List[Tuple[str, object]] = []
    if sweep == "pitch_height":
        for pitch in (0.0, 5.0, 10.0, 15.0):
            for height in (0.6, 1.0):
                variants.append((
                    f"pitch {pitch:.0f}°, h {height:.2f} m",
                    replace(base, pitch_deg=pitch, camera_height_m=height),
                ))
    elif sweep == "hfov":
        for hfov in (60.0, 70.0, 90.0, 110.0):
            focal = (base.width / 2.0) / np.tan(np.radians(hfov) / 2.0)
            for pitch in (0.0, 10.0):
                variants.append((
                    f"HFOV {hfov:.0f}° (fx {focal:.0f}), pitch {pitch:.0f}°",
                    replace(base, fx=focal, fy=focal, pitch_deg=pitch),
                ))
    elif sweep == "fine":
        # Fine grid around the coarse optimum (pitch 5 deg, height 1.0 m).
        for pitch in (3.0, 5.0, 7.0, 9.0):
            for height in (0.85, 1.00, 1.15):
                variants.append((
                    f"pitch {pitch:.0f}°, h {height:.2f} m",
                    replace(base, pitch_deg=pitch, camera_height_m=height),
                ))
    elif sweep == "recalib":
        # Range reopened after the focal lengths were corrected: the earlier
        # "1.0 m" pick was made against an fx that was 73% too long, so height
        # was absorbing a focal error. Both LiDARs put the sensor plate ~0.40 m
        # above ground, which makes the low end plausible again.
        for pitch in (0.0, 3.0, 6.0, 9.0):
            for height in (0.50, 0.70, 1.00):
                variants.append((
                    f"pitch {pitch:.0f}°, h {height:.2f} m",
                    replace(base, pitch_deg=pitch, camera_height_m=height),
                ))
    elif sweep == "height":
        for height in (1.00, 1.15, 1.30, 1.45, 1.60, 1.75):
            variants.append((f"height {height:.2f} m", replace(base, camera_height_m=height)))
    elif sweep == "pitch":
        for pitch in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0):
            variants.append((f"pitch {pitch:.0f}°", replace(base, pitch_deg=pitch)))
    else:
        raise ValueError(f"Unknown sweep '{sweep}'")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GND")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--sweep", type=str, default="pitch_height",
                        choices=["pitch_height", "hfov", "pitch", "fine", "height", "recalib"])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    sample_path = Path(args.sample) if args.sample else pick_frame(args.dataset, args.seed)
    dataset = detect_dataset(sample_path)
    data = json.load(open(sample_path, "r"))
    image_path = _resolve(data.get("current", {}).get("img", ""), sample_path)
    image = Image.open(image_path).convert("RGB")
    base = get_camera(dataset, image_size=image.size, recording=sample_path.parent.name)

    variants = build_variants(base, args.sweep)
    n_cols = 2 if len(variants) <= 8 else 3
    n_rows = int(np.ceil(len(variants) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.4 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, camera) in zip(axes, variants):
        ax.imshow(image)
        ax.axis("off")
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        draw_ground_grid(ax, camera)
        trace = image_trace_from_sample(data, camera, normalize=False)
        if trace is not None:
            ax.plot(trace[:, 0], trace[:, 1], "-", color="#00FF66", lw=3, zorder=2)
            ax.scatter(trace[:, 0], trace[:, 1], c=np.arange(len(trace)), cmap="autumn",
                       s=60, edgecolors="black", linewidths=0.8, zorder=3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(label, fontsize=11, fontweight="bold")

    for ax in axes[len(variants):]:
        ax.axis("off")

    fig.suptitle(
        f"{dataset} calibration sweep ({args.sweep}) — {sample_path.parent.name}/{sample_path.name}\n"
        f"pick the tile where the blue grid lies flat on the ground and the pink horizon "
        f"matches the real horizon",
        fontsize=13,
    )
    out_path = Path(args.out or f"/data/patelm/ticvla/outputs/calibration/{dataset}_{args.sweep}_{sample_path.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"{sample_path}\n-> {out_path}")


if __name__ == "__main__":
    main()
