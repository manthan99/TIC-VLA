from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""

def arrive(env: ManagerBasedRLEnv, threshold: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return (distance <= threshold).bool()
# --------------------------
# Upside-down with grace/sustain
# --------------------------
def upside_down(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    grace_steps: int = 50,
    sustain_steps: int = 3,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    device = asset.data.root_quat_w.device
    num_envs = env.num_envs

    # lazy-init counter
    cnt = getattr(env, "_upside_down_cnt", None)
    if (cnt is None) or (cnt.numel() != num_envs):
        cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
        setattr(env, "_upside_down_cnt", cnt)

    steps = getattr(env, "episode_length_buf", torch.zeros(num_envs, dtype=torch.long, device=device))

    quat = asset.data.root_quat_w  # [E,4] (wxyz)
    z_component = torch.abs(quat[:, 2])
    bad = z_component > float(threshold)

    past_grace = steps >= int(grace_steps)

    # increment / reset consecutive-counter
    inc_mask = bad & past_grace
    cnt[inc_mask] += 1
    cnt[~inc_mask] = 0

    # terminate only if sustained
    return cnt >= int(sustain_steps)

def upside_down_reset(env, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
    if env_ids is None:
        env_ids = slice(None)
    if hasattr(env, "_upside_down_cnt"):
        env._upside_down_cnt[env_ids] = 0
    return {}
upside_down.reset = upside_down_reset


# --------------------------
# Collision with grace/sustain
# --------------------------
def illegal_contact(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    grace_steps: int = 20,
    sustain_steps: int = 1,
) -> torch.Tensor:
    """
    Terminate when the contact force exceeds threshold, but:
      - ignore during grace_steps
      - require sustain_steps consecutive violations
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: [E, T_hist, B(=body_ids or all), 3]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    device = net_contact_forces.device
    num_envs = env.num_envs

    # lazy-init counter
    cnt = getattr(env, "_collision_cnt", None)
    if (cnt is None) or (cnt.numel() != num_envs):
        cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
        setattr(env, "_collision_cnt", cnt)

    steps = getattr(env, "episode_length_buf", torch.zeros(num_envs, dtype=torch.long, device=device))

    # reduce over time/history and bodies in sensor_cfg.body_ids
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is None:
        # if your ContactSensor is configured to provide only the monitored bodies,
        # we can just max over the last dimension
        forces_mag = torch.norm(net_contact_forces, dim=-1)             # [E, T_hist, B]
        peak = forces_mag.amax(dim=(1, 2))                               # [E]
    else:
        forces_mag = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)  # [E, T_hist, |body_ids|]
        peak = forces_mag.amax(dim=(1, 2))                                    # [E]

    bad = peak > float(threshold)
    past_grace = steps >= int(grace_steps)

    inc_mask = bad & past_grace
    cnt[inc_mask] += 1
    cnt[~inc_mask] = 0

    return cnt >= int(sustain_steps)

def illegal_contact_reset(env, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
    if env_ids is None:
        env_ids = slice(None)
    if hasattr(env, "_collision_cnt"):
        env._collision_cnt[env_ids] = 0
    return {}
illegal_contact.reset = illegal_contact_reset


def stuck(
    env,
    min_velocity: float = 0.1,
    window_steps: int = 50,
    grace_period_steps: int = 100,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # device from existing tensors
    asset = env.scene[asset_cfg.name]
    device = asset.data.root_lin_vel_b.device

    # lazy-init the per-env counter that spans steps within an episode
    counter = getattr(env, "_stuck_counter", None)
    if (counter is None) or (counter.numel() != env.num_envs):
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        setattr(env, "_stuck_counter", counter)

    # steps since the last reset (grace window gate)
    steps = getattr(env, "episode_length_buf", None)
    if steps is None:
        # fallback if your env uses a different name
        steps = getattr(env, "episode_step", None)
    if steps is None:
        # last-resort fallback: treat as step 0 so we never trigger at reset
        steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    # planar speed in BODY frame
    lin_vel_b = asset.data.root_lin_vel_b[:, :2]
    speed = torch.linalg.norm(lin_vel_b, dim=1)
    speed = torch.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    low = speed < float(min_velocity)
    past_grace = steps >= int(grace_period_steps)

    # increment when both: below speed AND outside grace period
    should_inc = low & past_grace
    counter[should_inc] += 1
    counter[~should_inc] = 0

    return counter >= int(window_steps)

def stuck_reset(env, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
    if env_ids is None:
        env_ids = slice(None)
    if hasattr(env, "_stuck_counter"):
        env._stuck_counter[env_ids] = 0
    return {}
stuck.reset = stuck_reset
