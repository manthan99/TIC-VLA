# isaaclab/ticvla/navigation_mdp.py
import torch
import math
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from collections.abc import Sequence
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, yaw_quat, quat_from_euler_xyz
from typing import Iterable
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedEnv

class NamedWaypointsPose2dCommand(CommandTerm):
    """Pose command generator identical in *behavior and API* to UniformPose2dCommand,
    except targets are drawn from user-provided world-frame waypoints.

    Buffers and outputs (parity with UniformPose2dCommand):
      - pos_command_w, heading_command_w
      - pos_command_b, heading_command_b
      - .command -> [E, 8] = [ x_b, y_b, z_b, yaw_err,  rel_move_x, rel_move_y, rel_move_z,  progress ]
      - metrics["error_pos_2d"], metrics["error_heading"]
      - optional debug visualization identical to UniformPose2dCommand
    """
    cfg: "NamedWaypointsPose2dCommandCfg"
    
    def __init__(self, cfg: "NamedWaypointsPose2dCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # robot handle (same as Uniform)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # ---- parse waypoints ----
        # Accepts dict[str, Iterable[float]] with (x, y[, z])
        if not cfg.named_points:
            raise ValueError("NamedWaypointsPose2dCommand: cfg.named_points is empty.")

        names: list[str] = []
        xy: list[tuple[float, float]] = []
        z_given: list[float | None] = []

        for name, p in cfg.named_points.items():
            if not isinstance(p, Iterable):
                raise ValueError(f"Waypoint '{name}' must be an iterable of (x,y[,z]).")
            p = list(p)
            if len(p) < 2:
                raise ValueError(f"Waypoint '{name}' must have at least (x, y).")
            x, y = float(p[0]), float(p[1])
            z = float(p[2]) if len(p) >= 3 else None
            names.append(name)
            xy.append((x, y))
            z_given.append(z)

        self._names = names
        self._wpts_xy = torch.tensor(xy, device=self.device, dtype=torch.float32)  # [K,2]
        self._wpts_z = torch.full((len(z_given),), float("nan"), device=self.device)
        for i, z in enumerate(z_given):
            if z is not None:
                self._wpts_z[i] = float(z)

        # selected waypoint index per env
        self._target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ---- buffers (exact names & shapes as UniformPose2dCommand) ----
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.relative_movement = torch.zeros_like(self.pos_command_w)
        self.distance_between_frame = torch.zeros_like(self.pos_command_w)
        self._prev_robot_pos_w = None 

        # progress memory
        self._have_prev = False  # first-frame gate

        # metrics (match keys used by upstream code)
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

        # initial sample + compute derived buffers
        self._resample_command(list(range(self.num_envs)))
        self._update_command()
        self._update_metrics()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command for specified environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        # Reset the progress tracking for reset environments
        if hasattr(self, 'temp_target_vec'):
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids, :3]
            self.temp_target_vec[env_ids] = target_vec
            self.distance_between_frame[env_ids, 0] = 0.0
            self.relative_movement[env_ids].zero_()
            if self._prev_robot_pos_w is not None:
                self._prev_robot_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids, :3]
        
        # Call parent reset method
        return super().reset(env_ids)

    # === required by CommandTerm ===
    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.pos_command_b,
            self.heading_command_b.unsqueeze(1),
            self.relative_movement, #movement since last frame
            self.distance_between_frame[:, 0:1]], #progress
            dim=1
        )

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )
    
    def _resample_command(self, env_ids: Sequence[int]):
        # Select new waypoint indices. We keep the selection logic simple & robust:
        #   - Avoid waypoints that are too close to current robot position (cfg.min_goal_sep)
        #   - Optionally avoid reusing the previous waypoint (cfg.avoid_prev)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        E = len(env_ids)
        ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        K = self._wpts_xy.shape[0]
        rxy = self.robot.data.root_pos_w[ids_t, :2]                     # [E,2]
        dists = torch.cdist(rxy, self._wpts_xy)                         # [E,K]

        min_sep = float(getattr(self.cfg, "min_goal_sep", 0.8))
        mask = dists > min_sep                                          # [E,K]

        avoid_prev = bool(getattr(self.cfg, "avoid_prev", True))
        if avoid_prev:
            prev = self._target_idx[ids_t]
            mask[torch.arange(E, device=self.device), prev] = False

        new_idx = torch.empty(E, dtype=torch.long, device=self.device)
        for i in range(E):
            valid = torch.nonzero(mask[i], as_tuple=False).flatten()
            if valid.numel() > 0:
                new_idx[i] = valid[torch.randint(0, valid.numel(), (1,), device=self.device)]
            else:
                # fallback: choose farthest (and try not to reuse prev)
                dd = dists[i]
                if avoid_prev and K > 1:
                    dd = dd.clone()
                    dd[prev[i]] = -1.0
                new_idx[i] = torch.argmax(dd)

        self._target_idx[ids_t] = new_idx

        # Set pos_command_w: XY from waypoint; Z from waypoint if given else robot’s default root height
        gxy = self._wpts_xy[new_idx, :]                                  # [E,2]
        self.pos_command_w[ids_t, 0:2] = gxy
        z_way = self._wpts_z[new_idx]                                     # [E]
        use_way_z = torch.isfinite(z_way)
        self.pos_command_w[ids_t, 2] = torch.where(
            use_way_z,
            z_way,
            self.robot.data.default_root_state[ids_t, 2],
        )

        # Set heading in world (same logic as UniformPose2dCommand)
        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[ids_t] - self.robot.data.root_pos_w[ids_t]
            target_dir = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped = wrap_to_pi(target_dir + torch.pi)

            curr_h = self.robot.data.heading_w[ids_t]
            to_target = wrap_to_pi(target_dir - curr_h).abs()
            to_flip = wrap_to_pi(flipped - curr_h).abs()
            self.heading_command_w[ids_t] = torch.where(to_target < to_flip, target_dir, flipped)
        else:
            low, high = self.cfg.ranges.heading  # same contract as UniformPose2dCommandCfg
            r = torch.empty(E, device=self.device)
            self.heading_command_w[ids_t] = r.uniform_(float(low), float(high))
        
        if getattr(self, "_have_prev", False):
            # align prev vec with the new target so progress = 0 right after resample
            target_vec = self.pos_command_w[ids_t] - self.robot.data.root_pos_w[ids_t, :3]
            self.temp_target_vec[ids_t] = target_vec
            self.distance_between_frame[ids_t, 0] = 0.0
            self.relative_movement[ids_t].zero_()


    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]  # [E,3] world delta
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

        if not getattr(self, "_have_prev", False):
            self.temp_target_vec = target_vec.clone()
            self.distance_between_frame.zero_()
            self.relative_movement.zero_()
            # store robot pose for relative movement
            self._prev_robot_pos_w = self.robot.data.root_pos_w[:, :3].clone()
            self._have_prev = True
            return

        # distance change (prev - curr); positive means we got closer
        prev_d = torch.norm(self.temp_target_vec[:, :2], dim=1)
        curr_d = torch.norm(target_vec[:, :2], dim=1)
        self.distance_between_frame[:, 0] = prev_d - curr_d

        r_now = self.robot.data.root_pos_w[:, :3]
        dr_w = r_now - self._prev_robot_pos_w                # world delta of robot
        self.relative_movement[:] = quat_apply_inverse(       # body-frame motion
            yaw_quat(self.robot.data.root_quat_w), dr_w
        )

        # advance prevs
        self.temp_target_vec = target_vec.clone()
        self._prev_robot_pos_w = r_now.clone()

        
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    # ---- _debug_vis_callback: mirror Uniform’s z=2.5 placement ----
    def _debug_vis_callback(self, event):
        self.goal_pose_visualizer.visualize(
            translations=torch.cat(
                [self.pos_command_w[:, :2],
                torch.ones_like(self.pos_command_w[:, 2:3]) * 2.5],
                dim=1
            ),
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )

@configclass
class NamedWaypointsPose2dCommandCfg(CommandTermCfg):
    """Same interface as UniformPose2dCommandCfg, but picks from user waypoints."""
    class_type: type = NamedWaypointsPose2dCommand

    asset_name: str = MISSING
    simple_heading: bool = True
    min_goal_sep: float = 0.8
    avoid_prev: bool = True

    @configclass
    class Ranges:
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()
    named_points: dict = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)