#!/usr/bin/env python3
"""Load and visualize one image-space trace training sample.

A sample is: the language goal, the current frame with the 10-point ground-truth
trace drawn on it, the historical frames at 3/6/9 s back, and the chain-of-thought
for the current frame.

Usage:
    python scripts/visualize_trace_sample.py --num-samples 4
    python scripts/visualize_trace_sample.py --sample /path/to/rgb_00660.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ticvla.data.projection import (
    TRACE_POINTS,
    HORIZON_SECONDS,
    WINDOW_SECONDS,
    future_offsets_in_window,
    camera_for_frame,
    get_camera,
    history_frames_at,
    history_path_in_current_frame,
    image_trace_from_sample,
)

DEFAULT_JSON_ROOT = "/data/patelm/ticvla/dataset/DynaNav/DynaNav_json"
LOOKBACKS_S = (9.0, 6.0, 3.0)


def _resolve(path: str, sample_path: Path) -> str:
    """Resolve a dataset-relative path against the sample's folder."""
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    return os.path.normpath(os.path.join(sample_path.parent, path))


def _read_text(path: str, sample_path: Path) -> str:
    resolved = _resolve(path, sample_path)
    try:
        return open(resolved, "r").read().strip()
    except OSError:
        return ""


def detect_dataset(sample_path: Path) -> str:
    """Infer the dataset from the sample path."""
    text = str(sample_path)
    for name in ("DynaNav", "SCAND", "GND"):
        if f"/{name}_json/" in text or f"/{name}/" in text:
            return name
    raise ValueError(f"Cannot infer dataset from path: {sample_path}")


def load_trace_sample(
    sample_path: str | Path,
    dataset: Optional[str] = None,
    n_points: int = TRACE_POINTS,
    per_frame_pitch: bool = False,
) -> Dict:
    """Assemble one training sample: goal, frames, CoT, and the image-space trace."""
    sample_path = Path(sample_path)
    data = json.load(open(sample_path, "r"))
    dataset = dataset or detect_dataset(sample_path)

    current_image = _resolve(data.get("current", {}).get("img", ""), sample_path)
    try:
        image_size = Image.open(current_image).size
    except OSError:
        image_size = None
    camera = get_camera(dataset, image_size=image_size, recording=sample_path.parent.name)
    if per_frame_pitch:
        camera = camera_for_frame(camera, data)

    trace_norm = image_trace_from_sample(data, camera, n_points=n_points)
    history = history_frames_at(data, LOOKBACKS_S)
    for frame in history:
        frame["img"] = _resolve(frame["img"], sample_path)

    offsets = future_offsets_in_window(data)
    return {
        "sample_path": str(sample_path),
        "dataset": dataset,
        "timestamp_s": float(data.get("timestamp", 0.0) or 0.0),
        "instruction": _read_text(data.get("instruction_file", ""), sample_path),
        "cot": _read_text(data.get("cot", ""), sample_path),
        "current_image": current_image,
        "history_frames": history,
        "trace_norm": trace_norm,
        "trace_px": None if trace_norm is None else trace_norm * np.array([camera.width, camera.height]),
        "future_offsets": offsets,
        "history_path": history_path_in_current_frame(data),
        "path_length_m": float(np.linalg.norm(np.diff(offsets[:, :2], axis=0), axis=1).sum()) if len(offsets) > 1 else 0.0,
        "camera": camera,
    }


def draw_ground_grid(ax, camera, max_range_m: float = 20.0) -> None:
    """Overlay a metric ground-plane grid to expose calibration errors.

    If the camera model is right, the grid lies flat on the floor: rails run
    parallel to the path and range bars sit at the stated distances. A grid that
    floats or sinks means the height/focal is off; a horizon in the wrong place
    means the pitch is non-zero.
    """
    lateral_offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
    ranges = (2.0, 5.0, 10.0, 20.0)

    # Longitudinal rails at fixed lateral offsets.
    forward = np.linspace(0.5, max_range_m, 120)
    for lateral in lateral_offsets:
        pts = np.column_stack([forward, np.full_like(forward, lateral), np.zeros_like(forward)])
        uv, ok = camera.project_ground(pts)
        uv = uv[ok]
        style = dict(color="#00BFFF", lw=1.6, alpha=0.75) if lateral == 0.0 else dict(color="#00BFFF", lw=0.9, alpha=0.45)
        ax.plot(uv[:, 0], uv[:, 1], "--", **style, zorder=1)

    # Lateral bars at fixed ranges, labelled with their distance.
    lateral = np.linspace(-3.0, 3.0, 60)
    for rng in ranges:
        pts = np.column_stack([np.full_like(lateral, rng), lateral, np.zeros_like(lateral)])
        uv, ok = camera.project_ground(pts)
        uv = uv[ok]
        if len(uv) == 0:
            continue
        ax.plot(uv[:, 0], uv[:, 1], "-", color="#00BFFF", lw=1.1, alpha=0.6, zorder=1)
        ax.annotate(f"{rng:.0f} m", (uv[-1, 0], uv[-1, 1]), fontsize=8, color="#00BFFF",
                    xytext=(4, -2), textcoords="offset points", zorder=1)

    # Horizon: where the ground plane vanishes, accounting for camera pitch.
    horizon = camera.horizon_v
    ax.axhline(horizon, color="#FF00AA", lw=1.0, ls=":", alpha=0.8, zorder=1)
    ax.annotate("horizon", (6, horizon - 4), fontsize=8, color="#FF00AA", zorder=1)


def draw_bev(ax, sample: Dict) -> None:
    """Bird's-eye view: where the robot came from and where it is going.

    Both paths are in the current frame's axes — x forward, y left — with the
    robot at the origin looking up the plot.
    """
    past = sample["history_path"]
    future = sample["future_offsets"]

    for radius in (5, 10, 15, 20):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#B9BCC4",
                                lw=0.7, ls=":", zorder=0))
        ax.annotate(f"{radius} m", (0.06 * radius, radius), fontsize=7,
                    color="#8A8F98", zorder=0)

    if len(past) > 1:
        ax.plot(-past[:, 1], past[:, 0], "-", color="#3B82F6", lw=2.4,
                label=f"travelled ({sample['timestamp_s']:.0f} s)", zorder=2)
    if len(future) > 1:
        ax.plot(-future[:, 1], future[:, 0], "-", color="#12B76A", lw=2.4,
                label="future (trace window)", zorder=2)
        ax.plot(-future[-1, 1], future[-1, 0], "*", color="#12B76A", ms=13, zorder=3)

    # Robot at the origin, heading up.
    ax.plot(0, 0, "o", color="#111827", ms=8, zorder=4)
    ax.annotate("", xy=(0, 1.6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.6), zorder=4)

    span = 6.0
    for path in (past, future):
        if len(path):
            span = max(span, float(np.abs(path[:, :2]).max()) * 1.15)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span * 0.35, span)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("left ← y (m) → right", fontsize=8)
    ax.set_ylabel("x forward (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_title("BEV — travelled vs. future", fontsize=10, fontweight="bold")


def _wrap(text: str, width: int = 110, max_lines: int = 14) -> str:
    import textwrap

    if not text:
        return "(none)"
    lines: List[str] = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... [truncated]"]
    return "\n".join(lines)


def visualize_sample(sample: Dict, out_path: str | Path, show_grid: bool = False) -> None:
    """Render the sample to a single annotated figure."""
    history = sample["history_frames"]
    n_hist = len(history)

    n_cols = max(n_hist, 1)
    fig = plt.figure(figsize=(21, 13))
    grid = fig.add_gridspec(
        3, n_cols + 1,
        width_ratios=[1] * n_cols + [0.85],
        height_ratios=[1.05, 2.7, 1.25],
        hspace=0.16, wspace=0.1,
    )

    # Row 1: historical context frames.
    for i, frame in enumerate(history):
        ax = fig.add_subplot(grid[0, i])
        ax.axis("off")
        try:
            ax.imshow(Image.open(frame["img"]).convert("RGB"))
        except OSError as exc:
            ax.text(0.5, 0.5, f"missing\n{exc}", ha="center", va="center", fontsize=7)
        clamped = " (clamped to window start)" if abs(
            sample["timestamp_s"] - frame["actual_s"] - frame["requested_s"]
        ) > 0.15 else ""
        ax.set_title(
            f"history −{frame['requested_s']:.0f}s → t={frame['actual_s']:.1f}s{clamped}",
            fontsize=10,
        )

    # Row 2: current frame with the ground-truth trace, plus the BEV alongside.
    draw_bev(fig.add_subplot(grid[1, n_cols]), sample)
    ax = fig.add_subplot(grid[1, :n_cols])
    ax.axis("off")
    try:
        image = Image.open(sample["current_image"]).convert("RGB")
        ax.imshow(image)
    except OSError as exc:
        ax.text(0.5, 0.5, f"missing current frame\n{exc}", ha="center", va="center")

    if show_grid:
        # Grid lines run off-image; pin the axes to the frame so the photo keeps its size.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        draw_ground_grid(ax, sample["camera"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    trace_px = sample["trace_px"]
    if trace_px is not None:
        ax.plot(trace_px[:, 0], trace_px[:, 1], "-", color="#00FF66", linewidth=4, alpha=0.9, zorder=2)
        ax.scatter(
            trace_px[:, 0], trace_px[:, 1],
            c=np.arange(len(trace_px)), cmap="autumn", s=150,
            edgecolors="black", linewidths=1.2, zorder=3,
        )
        for i, (x, y) in enumerate(trace_px):
            ax.annotate(
                str(i + 1), (x, y), fontsize=9, fontweight="bold", color="white",
                xytext=(9, 6), textcoords="offset points",
                path_effects=None, zorder=4,
            )
        title = f"current frame t={sample['timestamp_s']:.1f}s — GT trace, {len(trace_px)} points"
    else:
        title = f"current frame t={sample['timestamp_s']:.1f}s — NO TRACE (filtered out)"
    ax.set_title(
        f"{title}   |   trace covers {sample['timestamp_s']:.1f}s → "
        f"{sample['timestamp_s'] + HORIZON_SECONDS:.1f}s "
        f"(fixed {HORIZON_SECONDS:.0f}s horizon, {sample['path_length_m']:.1f} m of travel)",
        fontsize=12, fontweight="bold",
    )

    # Row 3: language goal, chain of thought, and the numeric target.
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    if trace_px is None:
        trace_str = "(no trace)"
    else:
        trace_str = "[" + ", ".join(
            f"[{x:.3f}, {y:.3f}]" for x, y in sample["trace_norm"]
        ) + "]"
    text = (
        f"[CAMERA] {sample['camera'].describe()}\n\n"
        f"[LANGUAGE GOAL]\n{_wrap(sample['instruction'], max_lines=3)}\n\n"
        f"[CHAIN OF THOUGHT — current frame]\n{_wrap(sample['cot'], max_lines=7)}\n\n"
        f"[GT TRACE — {len(trace_px) if trace_px is not None else 0} normalized points]\n{_wrap(trace_str, max_lines=3)}"
    )
    ax.text(
        0.008, 0.97, text, transform=ax.transAxes, fontsize=9.5,
        va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F3F4EE", edgecolor="#C9C9C0"),
    )

    fig.suptitle(
        f"{sample['dataset']} · {Path(sample['sample_path']).parent.name} / {Path(sample['sample_path']).name}",
        fontsize=13, y=0.995,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default=None, help="Path to one rgb_*.json")
    parser.add_argument("--json-root", type=str, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-path-m", type=float, default=2.0,
                        help="Skip near-stationary samples when picking at random")
    parser.add_argument("--min-lateral", type=float, default=0.0,
                        help="Require at least this much lateral (left/right) travel in metres, "
                             "so only turning samples are picked")
    parser.add_argument("--min-time", type=float, default=0.0,
                        help="Only sample frames at least this far into the window, so the "
                             "3/6/9 s look-backs are real frames rather than clamped to the start")
    parser.add_argument("--grid", action="store_true",
                        help="Overlay a metric ground-plane grid for calibration diagnosis")
    parser.add_argument("--stratify", action="store_true",
                        help="Spread samples across distinct recording prefixes (campuses/robots)")
    parser.add_argument("--any-frame", action="store_true",
                        help="Allow frames without a CoT annotation (only ~1 in 10 frames has one)")
    parser.add_argument("--out-dir", type=str,
                        default="/data/patelm/ticvla/outputs/trace_samples")
    args = parser.parse_args()

    if args.sample:
        paths = [Path(args.sample)]
    else:
        rng = random.Random(args.seed)
        folders = [d for d in Path(args.json_root).iterdir() if d.is_dir()]
        if args.stratify:
            # One folder per site prefix (e.g. GND campus, SCAND robot+location).
            by_prefix = {}
            for d in folders:
                by_prefix.setdefault(d.name.split("_chunk")[0].split("_1")[0], []).append(d)
            folders = [rng.choice(v) for v in by_prefix.values()]
            rng.shuffle(folders)
        paths = []
        attempts = 0
        while len(paths) < args.num_samples and folders and attempts < 400:
            attempts += 1
            folder = folders[len(paths) % len(folders)] if args.stratify else rng.choice(folders)
            # DynaNav uses rgb_*.json; SCAND and GND use img_<timestamp>.json.
            candidates = [p for p in folder.glob("*.json") if not p.name.startswith(".")]
            if not candidates:
                continue
            candidate = rng.choice(candidates)
            data = json.load(open(candidate, "r"))
            if float(data.get("timestamp", 0.0) or 0.0) < args.min_time:
                continue
            if not args.any_frame:
                cot_path = _resolve(data.get("cot", ""), candidate)
                if not cot_path or not os.path.exists(cot_path):
                    continue
            offsets = future_offsets_in_window(data)
            if len(offsets) > 1:
                length = float(np.linalg.norm(np.diff(offsets[:, :2], axis=0), axis=1).sum())
                lateral = float(np.abs(offsets[:, 1]).max())
                if length >= args.min_path_m and lateral >= args.min_lateral:
                    paths.append(candidate)

    for path in paths:
        sample = load_trace_sample(path)
        name = f"{path.parent.name}__{path.stem}.png"
        out_path = Path(args.out_dir) / sample["dataset"] / name
        visualize_sample(sample, out_path, show_grid=args.grid)
        n = 0 if sample["trace_norm"] is None else len(sample["trace_norm"])
        off = sample["future_offsets"]
        lat = float(off[np.abs(off[:, 1]).argmax(), 1]) if len(off) else 0.0
        turn = "LEFT " if lat > 0.5 else ("RIGHT" if lat < -0.5 else "straight")
        print(f"{path.parent.name}/{path.name}: t={sample['timestamp_s']:.1f}s "
              f"path={sample['path_length_m']:.1f}m lat={lat:+.1f}m {turn} trace_pts={n} -> {out_path}")


if __name__ == "__main__":
    main()
