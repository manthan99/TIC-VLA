"""
Main training loop for TIC-VLA with PPO.
"""

import argparse
import sys
import os
import math
import time
from datetime import datetime
from pathlib import Path
import numpy as np  
from PIL import Image
import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = RL_ROOT / "logs" / "ticvla_ppo"


def get_base_env(env):
    """Return the underlying Isaac Lab environment (unwrap gym wrappers)."""
    return env.unwrapped if hasattr(env, "unwrapped") else env


def get_robot_pose(base_env, device):
    """Return robot pose (x, y, yaw) for each environment."""
    try:
        robot = base_env.scene["robot"]
    except Exception:
        robot = base_env.scene.articulations["robot"]

    data = robot.data
    pos = data.root_pos_w[:, :2].detach().clone()
    if hasattr(data, "root_quat_w"):
        quat = data.root_quat_w.detach().clone()
    else:
        quat = data.root_state_w[:, 3:7].detach().clone()

    pos = pos.to(device=device, dtype=torch.float32)
    quat = quat.to(device=device, dtype=torch.float32)

    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pose = torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)
    return pose


# add argparse arguments
parser = argparse.ArgumentParser(description="Train TIC-VLA with PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=400, help="RL Policy training iterations.")
parser.add_argument("--rollout_steps", type=int, default=512, help="Number of steps per rollout.")
parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N iterations.")
parser.add_argument(
    "--model_path",
    type=str,
    default=os.getenv("TICVLA_BASE_MODEL_PATH", "OpenGVLab/InternVL3-1B"),
    help="Path to TIC-VLA base model (VLM weights).",
)
parser.add_argument("--initial_wait_time", type=float, default=5.0, help="Simulation seconds to wait after reset for VLM response.")
parser.add_argument(
    "--reward_scale",
    type=float,
    default=0.01,
    help="Multiply env rewards (and matching value targets) before GAE/PPO update for stable value learning.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# TIC-VLA observations include an IsaacLab CameraCfg, so rendering must be enabled
# even when video recording is disabled.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

try:
    from .model import ValueNetwork
    from .ticvla import TICVLA
    from .ppo import PPO
except ImportError:
    from model import ValueNetwork
    from ticvla import TICVLA
    from ppo import PPO


def main():
    """Train TIC-VLA with PPO."""
    
    # Load environment config
    from rl.env.config import TICVLANavEnvCfg
    env_cfg = TICVLANavEnvCfg()
    
    # Override configurations with CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        # Force single-environment training in this script
        env_cfg.scene.num_envs = 1

    device = os.getenv(
        "TICVLA_SIM_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    env_cfg.sim.device = device
    print(f"[INFO] Using device={device} for simulation and PPO")

    # Training hyperparameters
    max_iterations = args_cli.max_iterations
    rollout_steps = args_cli.rollout_steps
    num_envs = env_cfg.scene.num_envs
    reward_scale = float(args_cli.reward_scale)
    if reward_scale <= 0.0:
        raise ValueError(f"reward_scale must be > 0, got {reward_scale}")
    print(f"[INFO] PPO reward_scale={reward_scale} (env rewards unchanged; scaling applied in training only)")

    # Random seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # Create log directory under TIC-VLA/rl/logs/
    log_root = Path(os.getenv("TICVLA_RL_LOG_DIR", str(DEFAULT_LOG_ROOT)))
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(log_dir / "checkpoints", exist_ok=True)
    frame_save_dir = log_dir / "frames"
    frame_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    
    # Save config
    config_dict = {
        "max_iterations": max_iterations,
        "rollout_steps": rollout_steps,
        "num_envs": num_envs,
        "seed": args_cli.seed,
        "device": device,
        "base_model_path": args_cli.model_path,
        "model_path": args_cli.checkpoint,
        "initial_wait_time": args_cli.initial_wait_time,
        "reward_scale": reward_scale,
    }
    dump_yaml(str(log_dir / "config.yaml"), config_dict)
    
    # Tensorboard writer
    writer = SummaryWriter(str(log_dir))

    # Create environment
    print(f"[INFO] Creating environment...")
    task_name = args_cli.task if args_cli.task else "Isaac-Navigation-TICVLA-COCO"
    env = gym.make(task_name, cfg=env_cfg)
    print(f"[INFO] Environment created with {num_envs} environments")
    base_env = get_base_env(env)
    step_dt = float(getattr(base_env, "step_dt", getattr(env, "step_dt", 0.1)))

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"[INFO] Observation space: {obs_space}")
    print(f"[INFO] Action space: {action_space}")
    
    # Determine observation structure
    if isinstance(obs_space, gym.spaces.Dict):
        obs_keys = list(obs_space.spaces.keys())
        has_rgb = 'rgb' in obs_keys
        has_vector = any(k != 'rgb' for k in obs_keys)
    else:
        raise ValueError("Expected Dict observation space for TICVLA")
    
    if hasattr(action_space, 'shape') and action_space.shape is not None:
        action_dim = int(action_space.shape[-1])
    else:
        action_dim = 3  # Default action dimension

    # Create TIC-VLA model
    print(f"[INFO] Loading TIC-VLA model from {args_cli.model_path}...")
    ticvla_model = TICVLA(model_path=args_cli.model_path, device=device)
    
    # Load checkpoint if provided
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        print(f"[INFO] Loading checkpoint from {args_cli.checkpoint}...")
        ticvla_model.load_checkpoint(args_cli.checkpoint)
    
    # Create wrapper to make TIC-VLA compatible with PPO
    class TICVLAWrapper(nn.Module):
        """
        Wrapper to make TIC-VLA compatible with general PPO interface.
        
        Key design decisions for KV cache handling:
        1. During rollout: Uses wrapper's forward() which extracts image embeddings from RGB tensors
           and uses the latest KV cache from ticvla_model._latest_kv_cache
        2. During PPO updates: Also uses latest KV cache (from end of rollout).
        3. KV cache represents delayed frame context (from a few seconds ago).
        """
        def __init__(self, ticvla_model, dt=0.1):
            super().__init__()
            ticvla_model.train()  # Set to training mode
            self.ticvla_model = ticvla_model
            self.dt = dt
            
            hidden_dim = 512
            robot_state_dim = 5 
            self.value_head = ValueNetwork(robot_state_dim=robot_state_dim, action_dim=2, hidden_dim=hidden_dim, 
                        visual_pool_output=4, device=ticvla_model.device)
        
        def _extract_image_embeds_from_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
            """
            Extract image embeddings from RGB tensor using VLM's extract_feature.
            
            Args:
                rgb: (B, H, W, 3) or (B, C, H, W) RGB tensor in range [0, 1]
            
            Returns:
                current_image_embeds: (B, L_img, H) image embeddings
            """
            B = rgb.shape[0]
            device = self.ticvla_model.device
            
            # Convert RGB tensor to PIL Image format for VLM processing
            # RGB is expected to be in [0, 1] range, convert to [0, 255]
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
            
            # Handle different tensor formats: (B, H, W, 3) or (B, C, H, W)
            if rgb.shape[-1] == 3:
                # (B, H, W, 3) -> (B, C, H, W)
                rgb = rgb.permute(0, 3, 1, 2)
            
            # Convert to uint8 and then to PIL Images
            rgb_uint8 = rgb.clamp(0, 255).to(torch.uint8)
            current_image_embeds_list = []
            
            # Import image processing functions
            try:
                from .ticvla import build_transform, dynamic_preprocess
            except ImportError:
                from ticvla import build_transform, dynamic_preprocess
            
            for i in range(B):
                # Convert single image tensor to PIL Image
                img_tensor = rgb_uint8[i]  # (C, H, W)
                img_array = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                img_pil = Image.fromarray(img_array)
                
                # Process image using the same method as ticvla.load_image
                transform = build_transform(input_size=448)
                images = dynamic_preprocess(img_pil, image_size=448, use_thumbnail=True, max_num=1)
                pixel_values = torch.stack([transform(image) for image in images]).to(torch.bfloat16).to(device)
                
                # Extract embeddings using VLM
                with torch.no_grad():
                    img_embeds = self.ticvla_model.vlm.extract_feature(pixel_values)  # (num_tiles, num_img_tokens, H)
                    img_embeds = img_embeds.reshape(-1, img_embeds.shape[-1])  # (total_tokens, H)
                
                current_image_embeds_list.append(img_embeds)
            
            # Pad to same length
            max_tokens = max(e.shape[0] for e in current_image_embeds_list) if current_image_embeds_list else 0
            if max_tokens > 0:
                padded_embeds = []
                for e in current_image_embeds_list:
                    if e.shape[0] < max_tokens:
                        pad = torch.zeros(max_tokens - e.shape[0], e.shape[1], device=device, dtype=e.dtype)
                        e = torch.cat([e, pad], dim=0)
                    padded_embeds.append(e)
                current_image_embeds = torch.stack(padded_embeds, dim=0)  # (B, L_img, H)
            else:
                # Fallback: create empty embeddings
                hidden_size = self.ticvla_model.vlm.config.llm_config.hidden_size
                current_image_embeds = torch.zeros(B, 0, hidden_size, device=device, dtype=torch.bfloat16)
            
            return current_image_embeds
        
        def waypoints_to_action(self, waypoints: torch.Tensor) -> torch.Tensor:
            """
            Convert a short-horizon waypoint trajectory into current actions.
            
            Args:
                waypoints: (B, T, 2) waypoints [x, y] in current ego frame (T≈30 over ~3s)
            
            Returns:
                action_mean: (B, 2) mean actions [speed, yaw_speed]
            """
            B = waypoints.shape[0]
            device = waypoints.device
            
            # If waypoints is empty or all zeros, return zero actions
            if waypoints.numel() == 0 or waypoints.shape[1] == 0:
                return torch.zeros(B, 2, device=device, dtype=waypoints.dtype)

            T = waypoints.shape[1]
            if T < 2:
                return torch.zeros(B, 2, device=device, dtype=waypoints.dtype)

            # Nominal time step assuming ~3s horizon over T waypoints
            dt = torch.tensor(3.0, device=device, dtype=waypoints.dtype) / float(T)

            # Use ~1s lookahead waypoint as the target point (T≈30 ⇒ idx≈10)
            L_idx = 9
            target = waypoints[:, L_idx, :]                             
            time_to_target = dt * float(L_idx + 1)

            # Forward speed needed to reach the target in that horizon
            v_x = target[:, 0] / time_to_target
            v_y = target[:, 1] / time_to_target
            v =  torch.sqrt(v_x**2 + v_y**2) 
            v_yaw = torch.atan2(v_y, v_x+1e-3) 

            action_mean = torch.stack([v, v_yaw], dim=-1) 

            return action_mean
        
        def forward(self, obs: dict, kv_cache=None):
            """
            Forward pass for RL inference using KV cache.
            
            Args:
                obs: Dictionary with keys: 'rgb', 'pose_command', 'robot_state', 'time_delay'
                kv_cache: Optional KV cache tuple from rollout. If None, uses model's latest cache.
            
            Returns:
                action_mean: (B, 3) mean actions
                action_std: (B, 3) action standard deviations
                value: (B, 1) value estimates
            """
            rgb = obs['rgb']  # (B, H, W, 3) or (B, C, H, W)
            B = rgb.shape[0] if isinstance(rgb, torch.Tensor) else len(rgb)
            device = self.ticvla_model.device
            if isinstance(rgb, torch.Tensor) and rgb.device != device:
                rgb = rgb.to(device, non_blocking=True)
            
            # Extract image embeddings from RGB
            current_image_embeds = self._extract_image_embeds_from_rgb(rgb)
            
            # Get robot state and time delay
            robot_state = obs.get('robot_state', torch.zeros(B, 5, device=device))
            if isinstance(robot_state, torch.Tensor):
                if robot_state.device != device:
                    robot_state = robot_state.to(device, non_blocking=True)
            else:
                robot_state = torch.as_tensor(robot_state, device=device, dtype=torch.float32)

            time_delay = obs.get('time_delay', torch.zeros(B, device=device))
            if isinstance(time_delay, torch.Tensor):
                if time_delay.device != device:
                    time_delay = time_delay.to(device, non_blocking=True)
            else:
                time_delay = torch.as_tensor(time_delay, device=device, dtype=torch.float32)

            pose_command = obs.get('pose_command', None)
            if pose_command is None:
                raise ValueError("pose_command is required")
            else:
                pose_command = torch.as_tensor(pose_command, device=device, dtype=torch.float32)
                if pose_command.dim() == 1:
                    pose_command = pose_command.unsqueeze(0)
                if pose_command.shape[0] == 1 and B > 1:
                    pose_command = pose_command.expand(B, -1)
                if pose_command.shape[1] != 2:
                    pose_command = pose_command.reshape(pose_command.shape[0], -1)
                    if pose_command.shape[1] != 2:
                        pose_command = pose_command[:, :2] if pose_command.shape[1] > 2 else torch.nn.functional.pad(
                            pose_command, (0, max(0, 2 - pose_command.shape[1]))
                        )
            
            # Ensure correct shapes
            if robot_state.dim() == 1:
                robot_state = robot_state.unsqueeze(0)
            if time_delay.dim() == 0:
                time_delay = time_delay.unsqueeze(0)
            
            # Prepare state: [vx, vy, yaw_speed, dx, dy] + time_delay = (B, 6)
            # If robot_state is (B, 3), pad with zeros for dx, dy
            if robot_state.shape[1] == 3:
                # Assume it's [vx, vy, yaw_speed], pad with dx=0, dy=0
                robot_state = torch.cat([robot_state, torch.zeros(B, 2, device=device)], dim=-1)
            elif robot_state.shape[1] != 5:
                # Fallback: create default robot state
                robot_state = torch.zeros(B, 5, device=device)
            
            # Concat robot state and time delay → (B, 6)
            state = torch.cat([robot_state, time_delay.unsqueeze(-1)], dim=-1).unsqueeze(-1)  # (B, 6, 1)
            state = state.to(torch.bfloat16)
            
            # Get KV cache (use provided or model's latest)
            if kv_cache is None:
                kv_cache = self.ticvla_model._latest_kv_cache
            
            # If no KV cache available, create empty one
            if kv_cache is None or len(kv_cache) == 0:
                # Create empty KV cache (will result in zero waypoints, but won't crash)
                # The action_expert will handle empty KV cache
                kv_cache = tuple()
            
            # Predict waypoints using action_expert
            waypoints = self.ticvla_model.action_expert(current_image_embeds, state, kv_cache)  # (B, T, 2)

            # Convert waypoints to action mean
            action_mean = self.waypoints_to_action(waypoints)  # (B, 3)

            # Get value and action_std from value_head (expects batch dict with 'rgb' and 'robot_state')
            value_batch = {
                'image_embeds': current_image_embeds,
                # Use pose_command as value-network robot state input
                'robot_state': torch.cat([robot_state[:, :3], pose_command], dim=-1), # (vx, vy, yaw_speed, goal_x, goal_y)
            }
            value, action_std = self.value_head(value_batch)  # value: (B,), action_std: (B, 3)
            
            # Ensure correct shapes
            if value.dim() == 1:
                value = value.unsqueeze(-1)  # (B, 1)
            if action_std.dim() == 1:
                action_std = action_std.unsqueeze(0)  # (1, 3) -> expand to (B, 3)
            if action_std.shape[0] == 1 and B > 1:
                action_std = action_std.expand(B, -1)  # (B, 3)
            
            return action_mean, action_std, value
        
        def evaluate_actions(self, obs: dict, actions: torch.Tensor):
            """
            Evaluate actions - returns (log_probs, values, entropy).
            
            Args:
                obs: Observation dictionary, may contain '_kv_cache' key with list of KV caches
                actions: Actions to evaluate
            """
            # Extract KV cache from obs if present (one per sample in batch)
            kv_cache_list = obs.pop('_kv_cache', None)
            
            # If KV cache list is provided, process each sample with its own KV cache
            if kv_cache_list is not None:
                B = actions.shape[0]
                action_log_probs_list = []
                values_list = []
                entropy_list = []
                
                # Process each sample with its own KV cache
                for i in range(B):
                    # Create single-sample obs dict
                    single_obs = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
                    single_action = actions[i:i+1]
                    # Get KV cache for this sample (may be empty tuple if None was stored)
                    if i < len(kv_cache_list):
                        single_kv_cache = kv_cache_list[i]
                        # Convert empty tuple to None for forward() to handle
                        if single_kv_cache == tuple() or single_kv_cache is None:
                            single_kv_cache = None
                    else:
                        single_kv_cache = None
                    
                    # Forward pass with this sample's KV cache
                    action_mean, action_std, value = self.forward(single_obs, kv_cache=single_kv_cache)
                    
                    # Compute log prob and entropy
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action_log_prob = dist.log_prob(single_action).sum(dim=-1, keepdim=True)
                    entropy = dist.entropy().sum(dim=-1, keepdim=True)
                    
                    action_log_probs_list.append(action_log_prob)
                    values_list.append(value)
                    entropy_list.append(entropy)
                
                # Stack results
                action_log_probs = torch.cat(action_log_probs_list, dim=0)
                values = torch.cat(values_list, dim=0)
                entropy = torch.cat(entropy_list, dim=0)
            else:
                # Fallback: use latest KV cache for all samples
                action_mean, action_std, values = self.forward(obs, kv_cache=None)
                dist = torch.distributions.Normal(action_mean, action_std)
                action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            return action_log_probs, values, entropy
    
    # Wrap TIC-VLA for PPO compatibility
    ticvla_wrapper = TICVLAWrapper(ticvla_model, dt=step_dt)
    
    # Create PPO agent
    ppo = PPO(
        actor_critic=ticvla_wrapper,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_loss_coef=0.1,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        num_epochs=5,
        num_minibatches=8,
        device=device,
    )
    
    # Override optimizer to only train trainable parameters
    base_lr = 1e-5
    value_lr = 5e-5

    value_params = [p for p in ticvla_wrapper.value_head.parameters() if p.requires_grad]
    action_params = [p for p in ticvla_wrapper.ticvla_model.action_expert.parameters() if p.requires_grad]
    trainable_count = len(value_params) + len(action_params)
    print(
        f"[INFO] RL trainable params: value_head={sum(p.numel() for p in value_params)}, "
        f"action_expert_last_layer={sum(p.numel() for p in action_params)} "
        f"(total tensors={trainable_count})"
    )

    param_groups = [
        {"params": value_params, "lr": value_lr},
        {"params": action_params, "lr": base_lr},
    ]

    ppo.optimizer = optim.AdamW(param_groups)

    def _extract_instruction_from_info(info):
        """Extract instruction string from env info/extras."""
        if isinstance(info, (list, tuple)) and len(info) > 0:
            # For vectorized envs, info can be list-like; use the first entry (single env training).
            return _extract_instruction_from_info(info[0])
        if not isinstance(info, dict):
            return None

        def _normalize(inst):
            if inst is None:
                return None
            if isinstance(inst, str):
                return inst
            if isinstance(inst, (list, tuple)) and len(inst) > 0:
                return inst[0]
            if isinstance(inst, dict) and len(inst) > 0:
                # Try common keys first, then fallback to first value
                for k in (0, "0"):
                    if k in inst:
                        return inst[k]
                return next(iter(inst.values()))
            return None

        return _normalize(info.get("instruction")) or _normalize(
            info.get("extras", {}).get("instruction")
        )

    # Helper to get per-episode instruction from env (falls back to CLI)
    def get_env_instruction(info=None):
        inst = _extract_instruction_from_info(info)
        if isinstance(inst, str) and inst:
            return inst
        try:
            getter = getattr(base_env, "get_current_instruction", None)
            if callable(getter):
                inst = getter(0)
                if inst:
                    return inst
        except Exception:
            pass
        cli_inst = getattr(args_cli, "instruction", None)
        if isinstance(cli_inst, str) and cli_inst:
            return cli_inst
        # Safe fallback so downstream VLM prompt is never empty
        return "Navigate safely and efficiently forward."

    # Reset environment to get observation shapes
    start_iteration = 0
    obs, reset_info = env.reset()
        
    obs = obs["policy"]
    current_instruction = get_env_instruction(reset_info)
    
    # Initialize state tracking
    current_robot_pose = get_robot_pose(base_env, device)
    sim_time = 0.0
    total_sim_time = 0.0
    # Monotonic simulation step counter used for VLM latency bookkeeping
    current_step = 0
    vlm_generation_start_step = None
    last_inference_duration = 0.0
    last_response_sim_time = None
    last_processed_completion_step = None
    initial_wait_time = max(args_cli.initial_wait_time, 0.0)
    waiting_for_vlm_response = initial_wait_time > 0.0
    wait_deadline = total_sim_time + initial_wait_time if waiting_for_vlm_response else None
    
    # Maintain RGB history for delayed frame context
    rgb_history: list = []
    # Keep at least ~12s of frames so 3s interval sampling always has coverage
    history_len_seconds = 12.0
    history_len = int(history_len_seconds / step_dt)
    frame_counter = 0

    def save_rgb_frame_to_disk(rgb_array, sim_time: float, pose_xyz: np.ndarray | None = None) -> str:
        """Convert RGB array to PNG on disk and track in history."""
        nonlocal frame_counter, rgb_history
        arr = torch.as_tensor(rgb_array).cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
            if arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape[0] == 1:
                arr = np.repeat(np.transpose(arr, (1, 2, 0)), 3, axis=2)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        frame_path = str(frame_save_dir / f"frame_{frame_counter:06d}.jpg")
        Image.fromarray(arr_uint8).save(frame_path)
        rgb_history.append({"time": sim_time, "path": frame_path, "pose": pose_xyz})
        frame_counter += 1
        if len(rgb_history) > history_len:
            removed = rgb_history.pop(0)
            old_path = removed.get("path")
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        return frame_path

    def sample_delayed_image_paths(history, interval_sec: float = 3.0, max_frames: int = 4):
        """Sample frames spaced by interval_sec (roughly 3s) from newest to oldest.
        
        Returns:
            paths: list of sampled frame paths ordered from oldest -> newest
            anchor_pose: pose (x, y, yaw) of the oldest sampled frame (used for dx/dy)
        """
        if not history:
            return [], None
        selected = []
        last_time = None
        for item in reversed(history):
            t = item["time"]
            if last_time is None or (last_time - t) >= interval_sec - 1e-6:
                selected.append(item)
                last_time = t
            if len(selected) >= max_frames:
                break            
        selected = list(reversed(selected))
        if not selected:
            return [], None
        anchor_pose = selected[0].get("pose", None)
        return [it["path"] for it in selected], anchor_pose

    def clear_rgb_history():
        """Remove cached frame files and reset history."""
        nonlocal rgb_history
        for item in rgb_history:
            p = item.get("path")
            if p and os.path.exists(p):                                                                                         
                try:                        
                    os.remove(p)                                                                                                                                                                                                                                                                                                                                
                except OSError:
                    pass
        rgb_history = []

    def format_previous_waypoints_text(history: list) -> str:
        """Build previous-waypoints text (1s displacements) for the VLM prompt.

        Uses poses stored in ``rgb_history`` (x, y, yaw) and computes 1s
        displacements expressed in the older frame's FLU coordinates.
        """
        if not history:
            return "From 0.0s to current timestamp time is 0.0s. No waypoints available."

        # Keep only entries that have pose information
        valid = [h for h in history if h.get("pose") is not None]
        if len(valid) < 2:
            elapsed = max(0.0, history[-1]["time"] - history[0]["time"])
            return f"From 0.0s to current timestamp time is {elapsed:.1f}s. No waypoints available."

        start_time = float(valid[0]["time"])
        current_time = float(valid[-1]["time"])
        elapsed_time = max(0.0, current_time - start_time)

        waypoints = []
        prev_idx = 0
        t_target = start_time + 1.0  # first 1s interval

        while t_target <= current_time + 1e-6 and prev_idx < len(valid) - 1:
            next_idx = None
            for j in range(prev_idx + 1, len(valid)):
                if float(valid[j]["time"]) >= t_target:
                    next_idx = j
                    break
            if next_idx is None:
                break

            prev_pose = valid[prev_idx]["pose"]
            curr_pose = valid[next_idx]["pose"]
            if prev_pose is None or curr_pose is None or len(prev_pose) < 3 or len(curr_pose) < 3:
                prev_idx = next_idx
                t_target += 1.0
                continue

            delta_xy = np.array(curr_pose[:2], dtype=np.float32) - np.array(prev_pose[:2], dtype=np.float32)
            yaw_prev = float(prev_pose[2])
            c, s = math.cos(yaw_prev), math.sin(yaw_prev)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)  # world -> prev FLU
            delta_body = delta_xy @ rot
            waypoints.append((float(delta_body[0]), float(delta_body[1]), 0.0))

            prev_idx = next_idx
            t_target += 1.0

        if not waypoints:
            return f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. No waypoints available."

        waypoint_list = ", ".join(f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in waypoints)
        lines = [
            f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. (a list of waypoints 1s in between): {waypoint_list}",
            "Each waypoint (x, y, z) is the displacement over the previous 1.0s. x is forward, y is left, z is up.",
        ]
        return "\n".join(lines)
    
    # Get actual observation shapes dynamically
    rgb_shape = obs['rgb'].shape[1:] if len(obs['rgb'].shape) == 4 else obs['rgb'].shape
    pose_shape = obs['pose_command'].shape[1:] if len(obs['pose_command'].shape) > 1 else (2,)
    vel_shape = obs['base_lin_vel'].shape[1:] if len(obs['base_lin_vel'].shape) > 1 else (3,)
    ang_vel_shape = obs['base_ang_vel'].shape[1:] if len(obs['base_ang_vel'].shape) > 1 else (3,)
    
    # Initialize storage for rollout data (single environment)
    obs_rgb_buffer = np.zeros((rollout_steps, *rgb_shape), dtype=np.float32)
    obs_pose_buffer = np.zeros((rollout_steps, *pose_shape), dtype=np.float32)
    obs_vel_buffer = np.zeros((rollout_steps, *vel_shape), dtype=np.float32)
    obs_ang_vel_buffer = np.zeros((rollout_steps, *ang_vel_shape), dtype=np.float32)
    action_buffer = np.zeros((rollout_steps, action_dim), dtype=np.float32)
    reward_buffer = np.zeros((rollout_steps,), dtype=np.float32)
    done_buffer = np.zeros((rollout_steps,), dtype=np.float32)
    log_prob_buffer = np.zeros((rollout_steps, 1), dtype=np.float32)
    value_buffer = np.zeros((rollout_steps, 1), dtype=np.float32)
    time_delay_buffer = np.zeros((rollout_steps,), dtype=np.float32)
    displacement_buffer = np.zeros((rollout_steps, 2), dtype=np.float32)
    # Store KV cache references for each step (for PPO updates)
    kv_cache_buffer = [None] * rollout_steps  # List to store KV cache tuples
    
    # Training progress bar
    pbar = tqdm(range(start_iteration, max_iterations), desc="Training")

    for iteration in pbar:
        # Collect rollout data
        with torch.no_grad():
            iteration_episode_rewards: list[float] = []
            episode_reward: float = 0.0
            step = 0
            current_image_paths = []
            delayed_image_paths = []
            
            while step < rollout_steps:
                # Cache observations locally
                if isinstance(obs['rgb'], torch.Tensor):
                    current_rgb_np = obs['rgb'][0].cpu().numpy()
                else:
                    current_rgb_np = np.array(obs['rgb'])[0]

                if isinstance(obs['pose_command'], torch.Tensor):
                    current_pose_np = obs['pose_command'][0].cpu().numpy()
                else:
                    current_pose_np = np.array(obs['pose_command'])[0]

                if isinstance(obs['base_lin_vel'], torch.Tensor):
                    current_lin_vel_np = obs['base_lin_vel'][0].cpu().numpy()
                else:
                    current_lin_vel_np = np.array(obs['base_lin_vel'])[0]

                if isinstance(obs['base_ang_vel'], torch.Tensor):
                    current_ang_vel_np = obs['base_ang_vel'][0].cpu().numpy()
                else:
                    current_ang_vel_np = np.array(obs['base_ang_vel'])[0]

                # Update RGB history buffer on disk and sample delayed frames (~3s apart)
                current_robot_pose = get_robot_pose(base_env, device)
                current_image_path = save_rgb_frame_to_disk(
                    current_rgb_np, total_sim_time, current_robot_pose[0].cpu().numpy()
                )
                current_image_paths = [current_image_path] if current_image_path else []
                delayed_image_paths, anchor_pose_hist = sample_delayed_image_paths(
                    rgb_history, interval_sec=3.0, max_frames=4
                )
                
                # Poll for KV cache completion before computing delay
                ticvla_model._poll_kv_cache_future(current_step)

                # Update latency tracking when a fresh VLM response arrives
                completion_step = ticvla_model._kv_cache_completion_step
                if completion_step is not None and completion_step != last_processed_completion_step:
                    generation_step = ticvla_model._kv_cache_generation_step
                    if generation_step is not None:
                        step_delay = max(0, completion_step - generation_step)
                        last_inference_duration = step_delay * step_dt
                    else:
                        last_inference_duration = 0.0
                    last_response_sim_time = total_sim_time
                    last_processed_completion_step = completion_step
                    waiting_for_vlm_response = False
                    wait_deadline = None

                # Compute time delay (generation latency + time since response)
                elapsed_since_response = 0.0
                if last_response_sim_time is not None:
                    elapsed_since_response = max(0.0, total_sim_time - last_response_sim_time)
                latency_seconds = float(last_inference_duration + elapsed_since_response)
                time_delay = torch.tensor([latency_seconds], device=device, dtype=torch.bfloat16)
                
                # Prepare robot state: [vx, vy, yaw_speed, dx, dy]
                # dx, dy is displacement expressed in the anchor frame (previous frame).
                anchor_xy = None
                anchor_yaw = None
                gen_pose = ticvla_model._kv_cache_generation_pose
                if gen_pose is not None:
                    pos = gen_pose.get('position', None)
                    quat = gen_pose.get('quaternion', None)
                    if pos is not None and len(pos) >= 2:
                        anchor_xy = torch.as_tensor(pos[:2], device=device, dtype=torch.float32)
                    if quat is not None and len(quat) >= 4:
                        w, x, y, z = quat[:4]
                        anchor_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                # Fallback to delayed history anchor if no generation pose is available yet
                if anchor_xy is None and anchor_pose_hist is not None:
                    anchor_xy = torch.as_tensor(anchor_pose_hist[:2], device=device, dtype=torch.float32)
                    if len(anchor_pose_hist) >= 3:
                        anchor_yaw = float(anchor_pose_hist[2])

                if anchor_xy is not None:
                    current_xy = current_robot_pose[0, :2].to(device=device, dtype=torch.float32)
                    delta_xy = current_xy - anchor_xy
                    if anchor_yaw is not None:
                        c = math.cos(anchor_yaw); s = math.sin(anchor_yaw)
                        rot = torch.tensor([[c, s * -1], [s, c]], device=device, dtype=torch.float32)
                        delta_xy = delta_xy @ rot  # world -> anchor frame
                    displacement = delta_xy.unsqueeze(0)  # (1, 2)
                else:
                    displacement = torch.zeros(1, 2, device=device)

                base_lin_vel = obs['base_lin_vel'].to(device, non_blocking=True)
                base_ang_vel = obs['base_ang_vel'].to(device, non_blocking=True)
                robot_state = torch.cat([
                    base_lin_vel[:, :2],  # vx, vy
                    base_ang_vel[:, 2:3],  # yaw_speed
                    displacement  # dx, dy
                ], dim=-1)  # (1, 5)

                instruction_text = current_instruction if isinstance(current_instruction, str) else str(current_instruction)
                # Capture actual robot quaternion for prompt context
                try:
                    robot = base_env.scene["robot"]
                except Exception:
                    robot = base_env.scene.articulations["robot"]
                data = robot.data
                if hasattr(data, "root_quat_w"):
                    current_quat = data.root_quat_w[0].detach().clone()
                else:
                    current_quat = data.root_state_w[0, 3:7].detach().clone()
                current_quat_list = current_quat.float().cpu().tolist()
                
                # Start generation (non-blocking)
                prev_wp_text = format_previous_waypoints_text(rgb_history)
                ticvla_model._start_kv_cache_generation(
                    delayed_image_paths if delayed_image_paths else current_image_paths,
                    instruction_text,
                    dict(max_new_tokens=200, do_sample=False),
                    current_step,
                    previous_waypoints_text=prev_wp_text,
                    robot_type="wheeled robot",
                    current_robot_pose={
                        'position': current_robot_pose[0, :2].cpu().numpy().tolist(),
                        'quaternion': current_quat_list
                    }
                )
                
                # Use wrapper's forward() method for RL inference (handles RGB tensors directly)
                # This will use the latest KV cache from async generation
                action_start_time = time.perf_counter()
                obs_dict = {
                    'rgb': obs['rgb'],
                    'pose_command': torch.as_tensor(current_pose_np, device=device, dtype=torch.float32).reshape(1, -1),
                    'robot_state': robot_state,
                    'time_delay': time_delay,
                }
                action_mean, action_std, values = ticvla_wrapper.forward(obs_dict)
                action_runtime_sec = time.perf_counter() - action_start_time
                
                # Poll for async KV cache generation completion (for next steps)
                ticvla_model._poll_kv_cache_future(current_step)

                # Sample actions
                if waiting_for_vlm_response:
                    actions = torch.zeros_like(action_mean)
                    action_log_probs = torch.zeros((actions.shape[0], 1), device=actions.device, dtype=actions.dtype)
                else:
                    dist = torch.distributions.Normal(action_mean, action_std)
                    actions = dist.sample()
                    action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                
                # Step environment with actions
                env_actions = actions.to(device)
                next_obs, reward, terminated, truncated, info = env.step(env_actions)
                
                if isinstance(reward, torch.Tensor):
                    reward_scalar = float(reward.flatten()[0].item())
                else:
                    reward_scalar = float(np.array(reward).flatten()[0])
                if np.isnan(reward_scalar):
                    print("[WARN] NaN reward detected, setting to 0.0")
                    reward_scalar = 0.0

                done = terminated | truncated
                sim_time += step_dt
                total_sim_time += step_dt
                current_step += 1

                if waiting_for_vlm_response and wait_deadline is not None and total_sim_time >= wait_deadline:
                    waiting_for_vlm_response = False
                    wait_deadline = None

                # Determine done flag
                if isinstance(done, torch.Tensor):
                    done_flag = bool(done.flatten()[0].item())
                    done_scalar = float(done.flatten()[0].item())
                else:
                    done_flag = bool(np.array(done).flatten()[0])
                    done_scalar = float(np.array(done).flatten()[0])

                if done_flag:
                    ticvla_model.cancel_generation(reason="episode boundary reset")
                    clear_rgb_history()
                    sim_time = 0.0
                    vlm_generation_start_step = None
                    last_response_sim_time = None
                    last_inference_duration = 0.0
                    last_processed_completion_step = None
                    waiting_for_vlm_response = initial_wait_time > 0.0
                    wait_deadline = total_sim_time + initial_wait_time if waiting_for_vlm_response else None
                    current_instruction = get_env_instruction(info) or current_instruction
                    iteration_episode_rewards.append(episode_reward + reward_scalar)
                    episode_reward = 0.0
                else:
                    episode_reward += reward_scalar

                if not waiting_for_vlm_response:
                    obs_rgb_buffer[step] = current_rgb_np
                    obs_pose_buffer[step] = current_pose_np
                    obs_vel_buffer[step] = current_lin_vel_np
                    obs_ang_vel_buffer[step] = current_ang_vel_np
                    time_delay_buffer[step] = float(latency_seconds)
                    displacement_buffer[step] = displacement[0].detach().to(torch.float32).cpu().numpy()
                    action_buffer[step] = actions[0].detach().to(torch.float32).cpu().numpy()
                    done_buffer[step] = done_scalar
                    reward_buffer[step] = reward_scalar
                    log_prob_buffer[step] = action_log_probs[0].detach().to(torch.float32).cpu().numpy()
                    value_buffer[step] = values[0].detach().to(torch.float32).cpu().numpy()
                    # Store KV cache for this collected step (each step has its own KV cache)
                    current_kv_cache = ticvla_model._latest_kv_cache
                    kv_cache_buffer[step] = current_kv_cache if current_kv_cache is not None else tuple()
                    step += 1

                # Update observation for next loop iteration
                obs = next_obs["policy"]

            # Get value for last observation (for GAE)
            base_lin_vel_final = obs['base_lin_vel'].to(device, non_blocking=True)
            base_ang_vel_final = obs['base_ang_vel'].to(device, non_blocking=True)
            robot_state_final = torch.cat([
                base_lin_vel_final[:, :2],
                base_ang_vel_final[:, 2:3],
                torch.zeros(1, 2, device=device)
            ], dim=-1)

            obs_dict = {
                'rgb': obs['rgb'],
                'pose_command': torch.as_tensor(current_pose_np, device=device, dtype=torch.float32).reshape(1, -1),
                'robot_state': robot_state_final,
                'time_delay': torch.tensor([float(latency_seconds)], device=device),
            }

            _, _, next_values = ticvla_wrapper.forward(obs_dict)
            next_values_np = next_values.squeeze().flatten().cpu().numpy()

            # Scale rewards/values for GAE so return targets match critic training scale.
            scaled_rewards = reward_buffer * reward_scale
            scaled_values = value_buffer * reward_scale
            scaled_next_values = np.atleast_1d(next_values_np) * reward_scale

            advantages, returns = ppo.compute_gae(
                rewards=torch.from_numpy(scaled_rewards).unsqueeze(1),
                values=torch.from_numpy(scaled_values),
                next_values=torch.from_numpy(scaled_next_values),
                dones=torch.from_numpy(done_buffer).unsqueeze(1),
            )
            
            advantages_np = advantages.squeeze(-1).cpu().numpy()
            returns_np = returns.squeeze(-1).cpu().numpy()

        # Prepare batch for PPO update
        total_samples = rollout_steps
        
        # Flatten observations into batch dictionary
        rgb_flat = torch.from_numpy(obs_rgb_buffer).reshape(total_samples, *obs_rgb_buffer.shape[1:]).to(device)
        pose_flat = torch.from_numpy(obs_pose_buffer).reshape(total_samples, obs_pose_buffer.shape[-1]).to(device)
        vel_flat = torch.from_numpy(obs_vel_buffer).reshape(total_samples, obs_vel_buffer.shape[-1]).to(device)
        ang_vel_flat = torch.from_numpy(obs_ang_vel_buffer).reshape(total_samples, obs_ang_vel_buffer.shape[-1]).to(device)
        time_delay_flat = torch.from_numpy(time_delay_buffer).reshape(total_samples).to(device)
        displacement_flat = torch.from_numpy(displacement_buffer).reshape(total_samples, 2).to(device)
        
        # Prepare robot state: [vx, vy, yaw_speed, dx, dy]
        robot_state_flat = torch.cat([
            vel_flat[:, :2],  # vx, vy
            ang_vel_flat[:, 2:3],  # yaw_speed
            displacement_flat  # dx, dy from delayed anchor to current
        ], dim=-1)
        
        combined_batch = {
            'rgb': rgb_flat,
            'pose_command': pose_flat,
            'robot_state': robot_state_flat,
            'time_delay': time_delay_flat,
            '_kv_cache': kv_cache_buffer[:rollout_steps],  # Pass KV cache list (one per step)
        }
        
        # Note: Each step in the rollout has its own KV cache stored in kv_cache_buffer.
        # The wrapper's evaluate_actions will extract and use the appropriate KV cache
        # for each sample during PPO updates. PPO.update() handles '_kv_cache' specially
        # since it's a list (not a tensor) and passes it correctly to minibatches.
        
        # Flatten actions, log_probs, advantages, returns, values
        action_flat = torch.from_numpy(action_buffer).reshape(total_samples, action_dim).to(device)
        log_prob_flat = torch.from_numpy(log_prob_buffer).reshape(total_samples, 1).to(device)
        advantages_flat = torch.from_numpy(advantages_np).reshape(total_samples).to(device)
        returns_flat = torch.from_numpy(returns_np).reshape(total_samples).to(device)
        value_flat = torch.from_numpy(value_buffer * reward_scale).reshape(total_samples, 1).to(device)

        # Update policy
        stats = ppo.update(
            obs=combined_batch,
            actions=action_flat,
            old_log_probs=log_prob_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            old_values=value_flat,
        )

        # Log statistics
        writer.add_scalar("Loss/Policy", stats["policy_loss"], iteration)
        writer.add_scalar("Loss/Value", stats["value_loss"], iteration)
        writer.add_scalar("Loss/Entropy", stats["entropy_loss"], iteration)
        writer.add_scalar("Stats/ApproxKL", stats["approx_kl"], iteration)
        writer.add_scalar("Stats/ClipFraction", stats["clip_fraction"], iteration)
        writer.add_scalar("Reward/Mean", reward_buffer.mean(), iteration)
        writer.add_scalar("Reward/Std", reward_buffer.std(), iteration)
        if iteration_episode_rewards:
            episodic_mean = float(np.mean(iteration_episode_rewards))
            episodic_std = float(np.std(iteration_episode_rewards))
            writer.add_scalar("Reward/EpisodicMean", episodic_mean, iteration)
            writer.add_scalar("Reward/EpisodicStd", episodic_std, iteration)
        
        if 'log' in info:
            for key, value in info['log'].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Info/{key}", value, iteration)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    writer.add_scalar(f"Info/{key}", value.item(), iteration)

        # Update progress bar
        pbar.set_postfix({
            "policy_loss": f"{stats['policy_loss']:.4f}",
            "value_loss": f"{stats['value_loss']:.4f}",
            "mean_reward": f"{reward_buffer.mean():.4f}",
        })

        # Save checkpoint
        if (iteration + 1) % args_cli.save_interval == 0:
            checkpoint_path = str(log_dir / "checkpoints" / f"model_{iteration + 1}.pth")
            ppo.save(checkpoint_path)
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            if "actor_critic" in checkpoint_data:
                checkpoint_data["ticvla_model"] = checkpoint_data.pop("actor_critic")
            checkpoint_data["iteration"] = iteration + 1
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n[INFO] Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_checkpoint_path = str(log_dir / "checkpoints" / "final_model.pth")
    ppo.save(final_checkpoint_path)
    checkpoint_data = torch.load(final_checkpoint_path, weights_only=False)
    if "actor_critic" in checkpoint_data:
        checkpoint_data["ticvla_model"] = checkpoint_data.pop("actor_critic")
    checkpoint_data["iteration"] = max_iterations
    torch.save(checkpoint_data, final_checkpoint_path)
    print(f"\n[INFO] Training completed! Final model saved to {final_checkpoint_path}")

    # Close the simulator
    env.close()
    writer.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
