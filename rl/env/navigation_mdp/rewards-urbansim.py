# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return (1 - torch.tanh(distance / std)).float()

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def moving_towards_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    movement_xy = command[:, -1:]
    reward = movement_xy[:, 0]
    return reward * (env.episode_length_buf >= 10).float() 

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


def not_moving_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_velocity: float = 0.1) -> torch.Tensor:
    """
    Penalty for not moving (velocity too low).
    
    Args:
        env: The environment instance.
        asset_cfg: The robot asset configuration.
        min_velocity: Minimum velocity threshold in m/s.
    
    Returns:
        A tensor of penalties for each environment.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get linear velocity in body frame
    lin_vel = asset.data.root_lin_vel_b  # [num_envs, 3]
    
    # Calculate speed (magnitude of velocity)
    speed = torch.norm(lin_vel, dim=1)  # [num_envs]
    
    # Penalty if moving too slowly (1.0 if not moving, 0.0 if moving fast enough)
    penalty = (speed < min_velocity).float()
    
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