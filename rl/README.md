# TIC-VLA RL Fine-Tuning

This folder contains PPO scripts for fine-tuning TIC-VLA policies in the IsaacLab navigation environment.

## IsaacLab Setup

Use IsaacLab `v2.2.1` with the Isaac Sim 5.0.0 installation configured in the top-level README.

```bash
export TICVLA_ROOT="$(pwd)"
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAAC_SIM_ROOT=~/isaacsim
export ISAAC_SIM_PYTHON="${ISAAC_SIM_ROOT}/python.sh"

git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_ROOT}"
cd "${ISAACLAB_ROOT}"
ln -sfn "${ISAAC_SIM_ROOT}" _isaac_sim
./isaaclab.sh --conda tic-vla
conda deactivate
conda activate tic-vla

python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install -e source/isaaclab --no-deps
python -m pip install -e source/isaaclab_assets --no-deps
python -m pip install -e source/isaaclab_tasks --no-deps
python -m pip install -e source/isaaclab_rl --no-deps
python -m pip install -e source/isaaclab_mimic --no-deps
```

Verify registration:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p - <<'PY'
import gymnasium as gym
import rl.env

gym.spec("Isaac-Navigation-TICVLA-COCO")
print("TIC-VLA IsaacLab environment is registered.")
PY
```

## Training

Set paths from your `.env.training` (or export manually):

```bash
source .env.training
export TICVLA_ROOT="$(pwd)"
export TICVLA_CHECKPOINT_PATH="${TICVLA_OUTPUT_DIR}/checkpoints/ticvla/action/last.ckpt"
```

TIC-VLA policy trainer. This is the only model entrypoint in `rl/fine_tuning/`; it uses the same KV-cache action model path as the main TIC-VLA stack.

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p "${TICVLA_ROOT}/rl/fine_tuning/train_rl.py" \
  --task Isaac-Navigation-TICVLA-COCO \
  --enable_cameras \
  --model_path "${TICVLA_BASE_MODEL_PATH}" \
  --checkpoint "${TICVLA_CHECKPOINT_PATH}" \
  --max_iterations 500 \
  --rollout_steps 512 \
  --save_interval 50
```

Logs and checkpoints are written under `TIC-VLA/rl/logs/ticvla_ppo/` (override with `TICVLA_RL_LOG_DIR` if needed).

`--reward_scale` rescales rewards and value targets inside the trainer only (default `0.01`). Environment reward definitions in `rl/env/rewards.py` are unchanged; TensorBoard `Reward/*` logs still report raw env rewards.

## DynaNav Testing With an RL Checkpoint

### 1. Export RL weights for DynaNav

Use the same supervised checkpoint you passed to RL training as the base template (VLM + action-expert skeleton). Only `action_expert` weights are replaced with the RL-finetuned tensors; the frozen VLM stays from the base file.

```bash
export TICVLA_ROOT="$(pwd)"
source .env.training
export TICVLA_CHECKPOINT_PATH="${TICVLA_OUTPUT_DIR}/checkpoints/ticvla/action/last.ckpt"
export RL_RUN_DIR="${TICVLA_ROOT}/rl/logs/ticvla_ppo/<run_timestamp>"
export RL_CHECKPOINT="${RL_RUN_DIR}/checkpoints/model_N.pth"

python "${TICVLA_ROOT}/rl/fine_tuning/export_rl_checkpoint.py" \
  --rl-checkpoint "${RL_CHECKPOINT}" \
  --base-checkpoint "${TICVLA_CHECKPOINT_PATH}" \
  --output "${RL_RUN_DIR}/checkpoints/model_N_dynanav.ckpt"
```

### 2. Point DynaNav at the exported checkpoint

```bash
source .env.testing
export TICVLA_CHECKPOINT_PATH="${RL_RUN_DIR}/checkpoints/model_N_dynanav.ckpt"
```

Keep `TICVLA_BASE_MODEL_PATH` set to the same InternVL base used during RL training.

### 3. Run the benchmark

From the repository root:

```bash
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml
# Full benchmark set:
# DynaNav/run_benchmark.sh DynaNav/configs/benchmark_full.yaml
```

See `DynaNav/README.md` for Isaac Sim setup and `.env.testing` variables.

## Acknowledgement

The RL navigation environment builds on assets and environment design from [Urban-Sim](https://github.com/metadriverse/urban-sim).
