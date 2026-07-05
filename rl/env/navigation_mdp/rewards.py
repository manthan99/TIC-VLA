# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return (1 - torch.tanh(distance / std)).float()

def position_command_error_tanh_adaptive(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    k: float = 0.3,
    std_min: float = 0.5,
    std_max: float = 3.0,
) -> torch.Tensor:
    """
    Like position_command_error_tanh, but std adapts to the initial goal distance D0.
    std = clip(k * D0, std_min, std_max)
    """
    # Current goal (in BODY frame) from your command term: [x_b, y_b, ...]
    cmd = env.command_manager.get_command(command_name)   # [E, >=2]
    pos_b = cmd[:, :2]
    dist_now = torch.norm(pos_b, dim=1)                   # [E]

    # Prefer a stable per-episode initial error if available
    term = env.command_manager.get_term(command_name)
    D0 = getattr(term, "init_err2d", dist_now)  # [E], set on reset/resample only

    # Compute adaptive std per environment
    std = torch.clamp(k * D0, min=std_min, max=std_max)
    std = torch.where(std > 1e-6, std, torch.full_like(std, std_min))

    # Shaped proximity reward: 1 - tanh(distance / std) in [0, 1)
    reward = (1.0 - torch.tanh(dist_now / std)).float()

    return reward


def hit_pedestrian(env, env_ids=None, robot_radius: float = 0.35, person_radius: float | None = None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robots_xyz = env._robots_xyz(env_ids)
    robots_xy = robots_xyz[:, :2]  # Get xy part
    people_xy = env.people_xy.to(env.device)  # Ensure same device
    if people_xy.numel() == 0:
        return torch.zeros(len(env_ids), device=env.device)
    diff = robots_xy.unsqueeze(1) - people_xy.unsqueeze(0)   # [E,N,2]
    dist2 = (diff * diff).sum(-1)                            # [E,N]
    pr = float(getattr(env.cfg, "person_radius", 0.6)) if person_radius is None else float(person_radius)
    thr2 = (robot_radius + pr) ** 2
    return (dist2.min(dim=1).values <= thr2).to(env.device, dtype=torch.float32)


def facing_direction_reward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward forward movement and turning, penalize lateral slip and backward motion.
    
    Args:
        env: The environment instance.
        command_name: Name of the command (e.g., "pose_command").
        asset_cfg: The robot asset configuration.
    
    Returns:
        A reward tensor: positive for forward movement, negative for lateral slip or backward motion.
    """
    # Get robot's current velocity in body frame
    asset = env.scene[asset_cfg.name]
    vel_body = asset.data.root_lin_vel_b[:, :2]  # [num_envs, 2]

    # Reward weights
    forward_reward_weight = 1.0    # reward for forward movement
    lateral_penalty_weight = 1.0   # penalty for lateral slip
    backward_penalty_weight = 1.5  # penalty for backward motion

    vx = vel_body[:, 0]  # forward in body frame
    vy = vel_body[:, 1]  # lateral in body frame

    # Forward movement reward (positive vx)
    forward_reward = forward_reward_weight * torch.clamp(vx, min=0.0)
    
    # Lateral slip penalty (any non-zero vy)
    lateral_penalty = lateral_penalty_weight * torch.abs(vy)
    
    # Backward motion penalty (negative vx)
    backward_penalty = backward_penalty_weight * torch.clamp(-vx, min=0.0)

    # Total reward: positive for forward movement, negative for lateral slip or backward motion
    reward = forward_reward - lateral_penalty - backward_penalty

    return reward

def goal_directed_movement_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Reward movement that actually gets closer to the goal, discouraging circling.
    
    This function rewards the robot for reducing distance to the goal rather than
    just moving forward, which prevents circling behavior.
    """
    # Get current distance to goal
    cmd = env.command_manager.get_command(command_name)
    current_distance = torch.norm(cmd[:, :2], dim=1)  # Distance in body frame
    
    # Initialize previous distance on first call
    if not hasattr(env, "_prev_goal_distance"):
        env._prev_goal_distance = current_distance.clone()
    
    # Calculate progress (negative means getting closer)
    progress = env._prev_goal_distance - current_distance
    
    # Update previous distance for next step
    env._prev_goal_distance = current_distance.clone()
    
    # Reward positive progress (getting closer), penalize negative progress (getting farther)
    # Scale the reward to be proportional to the progress made
    reward = progress
    
    # Optional: Add small penalty for excessive lateral movement to encourage efficiency
    vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
    speed = torch.norm(vel_b, dim=1)
    
    # Only penalize lateral movement if we're not making progress toward goal
    goal_vec_b = torch.nn.functional.normalize(cmd[:, :2], dim=1, eps=1e-6)
    proj = (vel_b * goal_vec_b).sum(-1)
    lateral = vel_b - (proj.unsqueeze(1) * goal_vec_b)
    lateral_penalty = lateral.norm(dim=1)
    
    # Small penalty for lateral movement when not progressing toward goal
    efficiency_penalty = torch.where(
        progress <= 0,  # Only penalize when not making progress
        lateral_penalty * 0.1,  # Small penalty
        torch.zeros_like(lateral_penalty)
    )
    
    total_reward = reward - efficiency_penalty
    
    return total_reward

def moving_towards_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Reward only when achieving a new shortest distance to goal.
    Tracks the best distance achieved so far and only rewards when improving on it.
    This prevents rewards for going away and then coming back.
    """
    command = env.command_manager.get_command(command_name)
    current_distance = torch.norm(command[:, :2], dim=1)  # Current distance to goal
    
    # Initialize or reset best distance tracker at the start of each episode
    if not hasattr(env, "_best_goal_distance") or not hasattr(env, "_episode_ids"):
        env._best_goal_distance = current_distance.clone()
        env._episode_ids = env.episode_length_buf.clone()
    else:
        # Reset best distance for new episodes (where episode_length_buf is 0 or different from stored episode_ids)
        new_episode_mask = (env.episode_length_buf == 0) | (env.episode_length_buf < env._episode_ids)
        env._best_goal_distance[new_episode_mask] = current_distance[new_episode_mask]
        env._episode_ids = env.episode_length_buf.clone()
    
    # Only reward if current distance is better than best distance seen so far
    improvement = env._best_goal_distance - current_distance
    reward = torch.clamp(improvement, min=0.0)  # Only positive rewards for improvement
    
    # Update best distance if we've achieved a new record
    env._best_goal_distance = torch.minimum(env._best_goal_distance, current_distance)
    
    # Apply warmup gate (only start rewarding after 10 steps)
    reward = reward * (env.episode_length_buf >= 10).float()
    
    return reward 



def target_vel_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]
    distance_to_target_pos = torch.linalg.norm(target_pos, dim=1, keepdim=True)
    
    asset = env.scene['robot']
    vel = asset.data.root_lin_vel_b[:, 0:2]
    
    vel_direction = target_pos / distance_to_target_pos.clamp_min(1e-6)
    reward_vel = (vel * vel_direction).sum(-1)
    return reward_vel

def upside_down_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.5) -> torch.Tensor:
    """
    Penalty for being upside down.
    
    Args:
        env: The environment instance.
        asset_cfg: The robot asset configuration.
        threshold: The threshold for considering the robot upside down.
    
    Returns:
        A tensor of penalties for each environment.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get the quaternion (w, x, y, z) - Isaac Lab uses wxyz format
    quat = asset.data.root_quat_w  # [num_envs, 4]
    
    # The z-component of the quaternion indicates orientation
    # When the robot is upright, quat[:, 2] (z-component) should be close to 0
    # When upside down, quat[:, 2] should be close to ±1
    z_component = torch.abs(quat[:, 2])
    
    # Penalty if upside down (1.0 if upside down, 0.0 if upright)
    penalty = (z_component > threshold).float()
    
    return penalty


# --- PROGRESS REWARD ---
def progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    normalize_by_init: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Signed shaping: + when closer, - when farther.
    Uses the command's per-step progress field (prev_d - curr_d), already reset-safe.
    """
    cmd = env.command_manager.get_command(command_name)     # [..., progress]
    prog = cmd[:, -1]                             # meters / step (prev_d - curr_d)

    if normalize_by_init:
        term = env.command_manager.get_term(command_name)
        D0 = getattr(term, "init_err2d", torch.abs(prog))   # [E]
        prog = prog / D0.clamp_min(1e-6)                    # unitless per step

    # Optional warmup so first few steps don't dominate
    gate = (env.episode_length_buf >= 5).float()
    asym_factor = 0.5  # penalize going away less harshly
    prog = torch.where(prog > 0, prog, asym_factor * prog).clamp_min(0.0)
    reward = scale * prog * gate
    
    return reward


def no_progress_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    window: int = 20,
    tol: float = 0.05,
    normalize_by_init: bool = True,
) -> torch.Tensor:
    """
    Penalize dithering: if cumulative progress over the last `window` steps
    is < tol, emit a penalty (weight this term NEGATIVE in the cfg).

    Progress is prev_distance - curr_distance (>0 means getting closer).
    """
    cmd = env.command_manager.get_command(command_name)
    step_prog = cmd[:, -1]  # meters/step

    if normalize_by_init:
        term = env.command_manager.get_term(command_name)
        D0 = getattr(term, "init_err2d", torch.ones_like(step_prog))
        step_prog = step_prog / D0.clamp_min(1e-6)  # unitless

    E = step_prog.shape[0]
    dev = step_prog.device
    dtype = step_prog.dtype

    # Init per-env ring buffer + write pointer + episode id shadow
    if not hasattr(env, "_np_prog_buf"):
        env._np_prog_buf = torch.zeros(E, window, device=dev, dtype=dtype)
        env._np_fill = torch.zeros(E, device=dev, dtype=torch.long)
        env._np_episode_ids = env.episode_length_buf.clone()

    # Reset buffers on per-env episode reset
    new_ep = (env.episode_length_buf == 0) | (env.episode_length_buf < env._np_episode_ids)
    if new_ep.any():
        env._np_prog_buf[new_ep] = 0.0
        env._np_fill[new_ep] = 0
        env._np_episode_ids = env.episode_length_buf.clone()

    # Write current step progress into ring buffer (skip very first step)
    valid = env.episode_length_buf > 0
    if valid.any():
        rows = torch.where(valid)[0]
        fill_now = env._np_fill[valid]                # [V]
        cols = (fill_now % window)                    # [V]
        env._np_prog_buf[rows, cols] = step_prog[valid]
        env._np_fill[valid] = fill_now + 1            # DO NOT CAP

    # Sum over the ring buffer
    cum_prog = env._np_prog_buf.sum(dim=1)            # [E]

    # Only start penalizing once we've actually filled a full window
    have_window = (env._np_fill >= window).float()

    # Deficit: 0 if enough progress, up to 1 if insufficient positive progress
    # Only penalize when making positive progress but not enough (ignore negative progress)
    # If cumulative progress is negative, no penalty (let progress_reward handle it)
    is_positive_progress = cum_prog > 0.0
    positive_progress = torch.clamp(cum_prog, min=0.0)  # Ignore negative progress
    deficit = torch.where(is_positive_progress, 
                         torch.clamp((tol - positive_progress) / tol, min=0.0, max=1.0),
                         torch.zeros_like(cum_prog))  # No penalty for negative progress
    penalty = have_window * deficit

    return penalty

def instant_idle_penalty(env, command_name="pose_command",
                         prog_thresh=0.002,  # normalized units if normalize_by_init
                         normalize_by_init=True) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    prog = cmd[:, -1]  # prev_d - curr_d (m/step)

    if normalize_by_init:
        term = env.command_manager.get_term(command_name)
        D0 = getattr(term, "init_err2d", torch.ones_like(prog))
        prog = prog / D0.clamp_min(1e-6)

    # penalty = 1 when progress tiny or negative; 0 when decent progress
    penalty = (prog <= prog_thresh).float()
    return penalty

def time_based_penalty(env: ManagerBasedRLEnv, command_name: str = "pose_command", base_time_threshold: float = 50.0) -> torch.Tensor:
    """
    Time-based penalty that scales with curriculum difficulty.
    Penalty increases as time passes, with threshold based on curriculum stage.
    
    Args:
        env: The environment instance.
        command_name: Name of the command for curriculum access.
        base_time_threshold: Base time threshold in steps.
    
    Returns:
        A tensor of penalties for each environment.
    """
    # Get current episode length
    episode_length = env.episode_length_buf  # [num_envs]
    
    # Get curriculum progress (0.0 to 1.0)
    # We'll estimate curriculum progress from command ranges
    command = env.command_manager.get_term(command_name)
    x_range = command.cfg.ranges.pos_x
    y_range = command.cfg.ranges.pos_y
    
    # Estimate curriculum progress based on range sizes
    # Start ranges: X(-5,5)=10, Y(-5,5)=10
    # End ranges: X(-20,30)=50, Y(-20,30)=50
    start_x_size = 10.0  # (-5 to 5)
    end_x_size = 50.0    # (-20 to 30)
    current_x_size = x_range[1] - x_range[0]
    
    # Calculate curriculum progress (0.0 to 1.0)
    curriculum_progress = min((current_x_size - start_x_size) / (end_x_size - start_x_size), 1.0)
    curriculum_progress = max(curriculum_progress, 0.0)
    
    time_threshold = base_time_threshold * (1.0 + curriculum_progress * 2.0)  # 50 to 150 steps
    
    # Calculate penalty based on how much time has passed
    time_excess = torch.clamp(episode_length - time_threshold, min=0.0)
    penalty = time_excess / 100.0  # Scale penalty (0.01 per step over threshold)
    
    return penalty

def vlm_waypoint_tracking_error(
    env: "ManagerBasedRLEnv",
    std: float = 1.5,
    clamp_min_std: float = 0.5,
) -> torch.Tensor:
    if not hasattr(env, "num_envs"):
        return torch.tensor(0.0, device=getattr(env, "device", "cpu")).repeat(1)

    E = env.num_envs
    dev = env.device

    has = getattr(env, "_vlm_has_guidance", None)
    if has is None:
        return torch.zeros(E, device=dev, dtype=torch.float32)
    if has.dtype != torch.bool:
        has = has > 0.5

    # Current world pose and velocity
    robot = env.scene["robot"]
    pos_w = robot.data.root_state_w[:, 0:3]

    # Waypoint already in world frame (xy)
    wpt_w_buf = getattr(env, "_vlm_waypoint_w", None)
    if wpt_w_buf is None:
        return torch.zeros(E, device=dev, dtype=torch.float32)
    wpt_w = torch.zeros_like(wpt_w_buf)
    wpt_w[has] = wpt_w_buf[has]

    # Planar tracking error in world
    err_xy = (wpt_w[:, :2] - pos_w[:, :2]).norm(dim=1)

    use_std = torch.clamp(torch.as_tensor(std, device=dev, dtype=err_xy.dtype), min=clamp_min_std)
    reward = (1.0 - torch.tanh(err_xy / use_std)).float()
    reward = torch.where(has, reward, torch.zeros_like(reward))
    return reward


def speed_matching_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_speed: float = 1.0) -> torch.Tensor:
    """Penalize deviation from target speed (XY plane)."""
    asset = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    speed = torch.norm(vel_xy, dim=1)
    return torch.square(speed - target_speed)
