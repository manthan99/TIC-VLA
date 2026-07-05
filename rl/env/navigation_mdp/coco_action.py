from __future__ import annotations

import torch
from dataclasses import MISSING
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionAction, JointVelocityAction, JointPositionActionCfg, JointVelocityActionCfg, JointEffortActionCfg, JointEffortAction
from isaaclab.envs import ManagerBasedRLEnv

DT = 0.1
MAX_W = 2.0
MAX_V = 2.0
ACTION_INTERVAL = 4

# Support two types of action space
# 2. velocity and angular velocity
# 1. waypoint

# ===========================
# Action space: Velocity and angular velocity
# ===========================
class ClassicalCarAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: "ClassicalCarActionCfg"
    """The configuration of the action term."""

    def __init__(self, cfg: "ClassicalCarActionCfg", env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._counter = 0
        self.last_wheel_angle = torch.zeros(self.num_envs, 1, device=self.device)

        self.axle_names = ["base_to_front_axle_joint"]
        self.wheel_names = ["front_left_wheel_joint","front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"]
        self.shock_names = [".*shock_joint"]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

        # prepare low level actions
        self.acceleration_action: JointVelocityAction = JointVelocityAction(JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0, use_default_offset=False), env)
        self.steering_action: JointPositionAction = JointPositionAction(JointPositionActionCfg(asset_name="robot", joint_names=self.axle_names, scale=1., use_default_offset=True), env)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions


    def apply_actions(self):
        if self._counter % ACTION_INTERVAL == 0:
            max_wheel_v = 4.
            wheel_base = 1.5
            radius_rear = 0.3
            max_ang = 40 * torch.pi / 180
            # Incoming actions are interpreted as body-frame linear velocity (m/s) and yaw rate (rad/s).
            v_body = self.raw_actions[..., :1]
            w_body = self.raw_actions[..., 1:2]

            # Map (v, w) to steering angle using a bicycle model: delta = atan(L * w / v).
            # When |v| is tiny, fall back to straight steering (delta = 0) to avoid blow-up.
            eps = 1e-3
            steer_cmd = torch.where(
                v_body.abs() > eps, torch.atan(wheel_base * w_body / v_body), torch.zeros_like(w_body)
            )
            steer_cmd = steer_cmd.clamp(-max_ang, max_ang)
            steer_cmd[steer_cmd.abs() < 0.05] = 0.0  # deadband

            # Wheel speed target: allow forward/backward within limits, then convert to wheel angular vel.
            v_clamped = v_body.clamp(-max_wheel_v, max_wheel_v)
            wheel_ang_vel = v_clamped / radius_rear

            # Ackermann-ish mapping to left/right wheel angles from the steering angle.
            R = wheel_base / torch.tan(torch.clamp(steer_cmd, min=-max_ang + 1e-6, max=max_ang - 1e-6))
            left_wheel_angle = torch.arctan(wheel_base / (R - 0.5 * 1.8))
            right_wheel_angle = torch.arctan(wheel_base / (R + 0.5 * 1.8))

        
            self.steering_action.process_actions(((right_wheel_angle + left_wheel_angle) / 2.))
            self.acceleration_action.process_actions(
                torch.cat([wheel_ang_vel, wheel_ang_vel, wheel_ang_vel, wheel_ang_vel], dim=1)
            )
        
        self.steering_action.apply_actions()
        self.acceleration_action.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor):
        pass
    

@configclass
class ClassicalCarActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = ClassicalCarAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """Whether to visualize debug information. Defaults to False."""