"""Export an RL PPO checkpoint for DynaNav benchmark testing.

DynaNav behavior scripts load Lightning-style checkpoints with a top-level
``state_dict`` and ``model.*`` keys. RL checkpoints store weights under
``ticvla_model`` (full wrapper) including a ``value_head`` that DynaNav does
not use.

This script copies a supervised base checkpoint and overlays RL-finetuned
``action_expert`` weights so both Spot and Nova Carter behavior scripts can
load the result (including ``strict=True`` on Nova).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_rl_state(rl_checkpoint: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(rl_checkpoint, map_location="cpu", weights_only=False)
    if "ticvla_model" in ckpt:
        return ckpt["ticvla_model"]
    if "actor_critic" in ckpt:
        return ckpt["actor_critic"]
    raise KeyError(
        f"Expected 'ticvla_model' or 'actor_critic' in {rl_checkpoint}, "
        f"found keys: {sorted(ckpt.keys())}"
    )


def export_for_dynanav(
    rl_checkpoint: Path,
    base_checkpoint: Path,
    output_path: Path,
) -> int:
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in base:
        raise KeyError(f"Base checkpoint missing 'state_dict': {base_checkpoint}")

    out_state = dict(base["state_dict"])
    rl_state = _load_rl_state(rl_checkpoint)

    updated = 0
    for key, value in rl_state.items():
        if not key.startswith("ticvla_model.action_expert."):
            continue
        target_key = "model." + key[len("ticvla_model.") :]
        if target_key in out_state and out_state[target_key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {target_key}: "
                f"base {tuple(out_state[target_key].shape)} vs rl {tuple(value.shape)}"
            )
        out_state[target_key] = value
        updated += 1

    if updated == 0:
        raise RuntimeError(
            f"No action_expert tensors found in RL checkpoint {rl_checkpoint}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out_state}, output_path)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        required=True,
        help="RL PPO checkpoint, e.g. rl/logs/ticvla_ppo/.../model_500.pth",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Supervised action checkpoint used to start RL, e.g. action/last.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .ckpt path for DynaNav (set TICVLA_CHECKPOINT_PATH to this file)",
    )
    args = parser.parse_args()

    count = export_for_dynanav(args.rl_checkpoint, args.base_checkpoint, args.output)
    print(
        f"Exported {count} action_expert tensors to {args.output}\n"
        f"Set TICVLA_CHECKPOINT_PATH={args.output.resolve()} before running DynaNav."
    )


if __name__ == "__main__":
    main()
