# isaaclab/ticvla/navigation_mdp/walkable_command.py
import torch
import math
import random
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

class WalkablePose2dCommand(CommandTerm):
    """Command generator that uses pre-computed walkable positions from the environment.
    
    This ensures all goals are actually reachable by the robot, avoiding unreachable
    positions that might be inside walls or obstacles.
    
    Buffers and outputs (same as UniformPose2dCommand):
      - pos_command_w, heading_command_w
      - pos_command_b, heading_command_b
      - .command -> [E, 8] = [ x_b, y_b, z_b, yaw_err,  rel_move_x, rel_move_y, rel_move_z,  progress ]
      - metrics["error_pos_2d"], metrics["error_heading"]
    """
    cfg: "WalkablePose2dCommandCfg"
    
    def __init__(self, cfg: "WalkablePose2dCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # robot handle
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Get walkable positions per environment
        self.walkable_positions_per_env = self._get_walkable_positions_per_env(env)

        # selected waypoint index per env
        self._target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ---- buffers (same as UniformPose2dCommand) ----
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.relative_movement = torch.zeros_like(self.pos_command_w)
        self.distance_between_frame = torch.zeros_like(self.pos_command_w)

        # progress memory
        self._have_prev = False  # first-frame gate
        self.temp_target_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_robot_pos_w = self.robot.data.root_pos_w[:, :3].clone()

        # metrics (match keys used by upstream code)
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        
        # progress tracking (match pose_2d_command.py)
        self.prev_err2d = torch.zeros(self.num_envs, device=self.device)
        # Stable initial error per episode/resample for adaptive rewards
        self.init_err2d = torch.zeros(self.num_envs, device=self.device)
        self.progress2d = torch.zeros(self.num_envs, 1, device=self.device)

        # initial sample + compute derived buffers
        self._resample_command(list(range(self.num_envs)))
        self._update_command()
        self._update_metrics()
        
        # Initialize prev_err2d to current error to avoid negative progress on first frame
        self.prev_err2d = self.metrics["error_pos_2d"].clone()
        # Initialize init_err2d once at startup to the current error baseline
        self.init_err2d = self.metrics["error_pos_2d"].clone()
        self._have_prev = True

    def _get_walkable_positions_per_env(self, env) -> torch.Tensor:
        """Get walkable positions for each environment around their respective origins.
        
        Each environment gets its own unique set of positions sampled around its own origin.
        This ensures each environment has different walkable areas and goals.
        """
        num_envs = self.num_envs
        K = self.cfg.num_walkable_positions
        
        # Read curriculum ranges from cucciculum.py to determine sampling bounds
        # These match the x_range_end and y_range_end from increase_moving_distance()
        x_min, x_max = -20.0, 30.0  # From curriculum: x_range_end=(-20.0, 30.0)
        y_min, y_max = -20.0, 30.0  # From curriculum: y_range_end=(-20.0, 30.0)
        
        # Initialize tensor for all environments: [num_envs, num_positions, 2]
        all_positions = torch.zeros((num_envs, K, 2), device=self.device, dtype=torch.float32)
        
        # Get environment origins
        env_origins = env.scene.env_origins[:, :2]  # [E,2]
        
        # Sample positions for each environment individually
        for env_idx in range(num_envs):
            env_origin = env_origins[env_idx, :2]  # [2] - x, y coordinates
            
            # Sample positions around this environment's origin
            positions = env.get_walkable_positions_in_rect_cached(
                K, self.cfg.robot_radius, (x_min, x_max), (y_min, y_max)
            )  # [K,2] around (0,0)
            
            # Translate to this environment's origin
            all_positions[env_idx] = positions + env_origin.unsqueeze(0)  # [K,2]
        
        return all_positions.contiguous()


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command for specified environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        # Reset the progress tracking for reset environments
        if hasattr(self, 'temp_target_vec'):
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids, :3]
            self.temp_target_vec[env_ids] = target_vec
            self.relative_movement[env_ids].zero_()
            # Reset progress tracking (match pose_2d_command.py)
            self.progress2d[env_ids, 0] = 0.0
            # Initialize prev_err2d to current error to avoid negative progress
            curr_err = torch.norm(self.pos_command_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=1)
            self.prev_err2d[env_ids] = curr_err
            # Also reset the stable initial error baseline for these envs
            self.init_err2d[env_ids] = curr_err
        
        # Call parent reset method
        return super().reset(env_ids)

    # === required by CommandTerm ===
    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.pos_command_b, #[3] Goal position in robot frame
            self.heading_command_b.unsqueeze(1), #[1] Goal heading in robot frame
            self.relative_movement, #[3] Relative movement since last frame
            self.progress2d], #[1] Progress since last frame
            dim=1
        )

    def _update_metrics(self):
        # logs data (match pose_2d_command.py)
        err2d = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_pos_2d"] = err2d

        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )

        # Initialize prev_err2d on first call
        if not hasattr(self, "prev_err2d"):
            self.prev_err2d = err2d.clone()

        # Compute progress from previous frame (positive if closer)
        new_progress = self.prev_err2d - err2d                      # [E]
        self.progress2d[:, 0] = new_progress                        # write metric

        # NOW update prev_err2d for the next step
        self.prev_err2d = err2d.clone()
    
    def _resample_command(self, env_ids: Sequence[int]):
        # Select new waypoint indices from walkable positions
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        E = len(env_ids)
        
        ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Get walkable positions for each environment: [E, K, 2]
        env_walkable_positions = self.walkable_positions_per_env[ids_t]  # [E, K, 2]
        K = env_walkable_positions.shape[1]  # Number of positions per environment
        
        # Get robot positions: [E, 2]
        rxy = self.robot.data.root_pos_w[ids_t, :2]  # [E, 2]
        
        # Calculate distances from each robot to its environment's walkable positions
        dists = torch.zeros((E, K), device=self.device)
        for i in range(E):
            dists[i] = torch.norm(env_walkable_positions[i] - rxy[i], dim=1)  # [K]

        # Apply curriculum range constraints (set by cucciculum.py)
        min_sep = float(getattr(self.cfg, "min_goal_sep", 0.8))

        if hasattr(self._env, 'command_manager') and self._env.command_manager is not None:
            command_term = self._env.command_manager.get_term("pose_command")
            x_min, x_max = command_term.cfg.ranges.pos_x
            y_min, y_max = command_term.cfg.ranges.pos_y
        else:
            # Fallback to self.cfg during initialization
            x_min, x_max = self.cfg.ranges.pos_x
            y_min, y_max = self.cfg.ranges.pos_y
        
        # Filter walkable positions by curriculum ranges (relative to each environment's origin)
        mask = torch.zeros((E, K), dtype=torch.bool, device=self.device)
        for i in range(E):
            env_origin = self._env.scene.env_origins[ids_t[i], :2]  # [2]
            wp_xy = env_walkable_positions[i]  # [K, 2]
            
            # Convert to relative coordinates (relative to environment origin)
            wp_relative = wp_xy - env_origin.unsqueeze(0)  # [K, 2]
            
            # Check if positions are within curriculum range
            in_x_range = (wp_relative[:, 0] >= x_min) & (wp_relative[:, 0] <= x_max)
            in_y_range = (wp_relative[:, 1] >= y_min) & (wp_relative[:, 1] <= y_max)
            in_curriculum_range = in_x_range & in_y_range
            
            # EXCLUSIVE NEW AREA SELECTION: Only select goals from newly added curriculum areas
            # This ensures that when curriculum progresses, only new areas are explored
            exclude_previous_curriculum = torch.ones(K, dtype=torch.bool, device=self.device)
            
            # Apply curriculum exclusion to force exploration of only NEW areas
            if hasattr(self, '_prev_curriculum_ranges') and self._prev_curriculum_ranges is not None:
                prev_x_min, prev_x_max = self._prev_curriculum_ranges['x']
                prev_y_min, prev_y_max = self._prev_curriculum_ranges['y']
                
                # Check if current curriculum range is actually different from previous
                ranges_changed = (prev_x_min != x_min or prev_x_max != x_max or 
                                prev_y_min != y_min or prev_y_max != y_max)
                
                if ranges_changed:
                    # Define NEW areas only (exclude previous curriculum areas)
                    # Example: prev=(-5,5), curr=(-8,8) → new areas = (-8,-5) ∪ (5,8)
                    
                    # X-axis new areas
                    new_x_areas = []
                    if x_min < prev_x_min:  # Extended left
                        new_x_areas.append((x_min, prev_x_min))
                    if x_max > prev_x_max:  # Extended right
                        new_x_areas.append((prev_x_max, x_max))
                    
                    # Y-axis new areas  
                    new_y_areas = []
                    if y_min < prev_y_min:  # Extended down
                        new_y_areas.append((y_min, prev_y_min))
                    if y_max > prev_y_max:  # Extended up
                        new_y_areas.append((prev_y_max, y_max))
                    
                    # If we have new areas, only allow goals in those areas
                    if new_x_areas or new_y_areas:
                        in_new_area = torch.zeros(K, dtype=torch.bool, device=self.device)
                        
                        # Check if position is in any new X area
                        for new_x_min, new_x_max in new_x_areas:
                            x_in_new_area = (wp_relative[:, 0] >= new_x_min) & (wp_relative[:, 0] <= new_x_max)
                            # Must also be within current Y range
                            y_in_range = (wp_relative[:, 1] >= y_min) & (wp_relative[:, 1] <= y_max)
                            in_new_area |= (x_in_new_area & y_in_range)
                        
                        # Check if position is in any new Y area
                        for new_y_min, new_y_max in new_y_areas:
                            y_in_new_area = (wp_relative[:, 1] >= new_y_min) & (wp_relative[:, 1] <= new_y_max)
                            # Must also be within current X range
                            x_in_range = (wp_relative[:, 0] >= x_min) & (wp_relative[:, 0] <= x_max)
                            in_new_area |= (x_in_new_area & y_in_range)
                        
                        # Only allow goals in new areas
                        exclude_previous_curriculum = in_new_area
                    else:
                        # No new areas, fall back to current range
                        exclude_previous_curriculum = torch.ones(K, dtype=torch.bool, device=self.device)
            
            # Mask: goals must be far enough AND within curriculum ranges AND in new areas only
            mask[i] = (dists[i] > min_sep) & in_curriculum_range & exclude_previous_curriculum
            
            # Store current curriculum ranges for next time (for env 0 only)
            if i == 0:
                self._prev_curriculum_ranges = {
                    'x': (x_min, x_max),
                    'y': (y_min, y_max)
                }
            
            # If no positions in new areas, fall back to current curriculum range
            if env_ids[i] == 0 and mask[i].sum().item() == 0:
                new_area_only = exclude_previous_curriculum.sum().item()
                in_range = in_curriculum_range.sum().item()
                if new_area_only == 0 and in_range > 0:
                    mask[i] = (dists[i] > min_sep) & in_curriculum_range

        avoid_prev = bool(getattr(self.cfg, "avoid_prev", True))
        if avoid_prev:
            prev = self._target_idx[ids_t]
            mask[torch.arange(E, device=self.device), prev] = False

        new_idx = torch.empty(E, dtype=torch.long, device=self.device)
        for i in range(E):
            valid = torch.nonzero(mask[i], as_tuple=False).flatten()
            if valid.numel() > 0:
                selected_idx = valid[torch.randint(0, valid.numel(), (1,), device=self.device)]
                new_idx[i] = selected_idx
            else:
                # fallback: choose farthest within curriculum range (and try not to reuse prev)
                dd = dists[i].clone()
                # Get curriculum range for this environment
                env_origin = self._env.scene.env_origins[ids_t[i], :2]
                wp_xy = env_walkable_positions[i]
                wp_relative = wp_xy - env_origin.unsqueeze(0)
                in_x_range = (wp_relative[:, 0] >= x_min) & (wp_relative[:, 0] <= x_max)
                in_y_range = (wp_relative[:, 1] >= y_min) & (wp_relative[:, 1] <= y_max)
                in_curriculum_range = in_x_range & in_y_range
                dd[~in_curriculum_range] = -1.0  # Exclude positions outside curriculum range
                if avoid_prev and K > 1:
                    dd[prev[i]] = -1.0
                new_idx[i] = torch.argmax(dd)

        self._target_idx[ids_t] = new_idx

        # Set pos_command_w: XY from walkable positions; Z from robot's default root height
        gxy = torch.zeros((E, 2), device=self.device)
        for i in range(E):
            gxy[i] = env_walkable_positions[i, new_idx[i]]  # [2]
        
        self.pos_command_w[ids_t, 0:2] = gxy
        self.pos_command_w[ids_t, 2] = self.robot.data.default_root_state[ids_t, 2]

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
            low, high = self.cfg.ranges.heading
            r = torch.empty(E, device=self.device)
            self.heading_command_w[ids_t] = r.uniform_(float(low), float(high))
        
        if getattr(self, "_have_prev", False):
            # align prev vec with the new target so progress = 0 right after resample
            target_vec = self.pos_command_w[ids_t] - self.robot.data.root_pos_w[ids_t, :3]
            self.temp_target_vec[ids_t] = target_vec
            self.relative_movement[ids_t].zero_()
            # Reset progress tracking (match pose_2d_command.py)
            self.progress2d[ids_t, 0] = 0.0
            # Initialize prev_err2d to current error to avoid negative progress
            curr_err = torch.norm(self.pos_command_w[ids_t, :2] - self.robot.data.root_pos_w[ids_t, :2], dim=1)
            self.prev_err2d[ids_t] = curr_err
            # Update initial error baseline when a new goal is sampled
            self.init_err2d[ids_t] = curr_err

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]  # [E,3] world delta
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

        curr_robot_pos_w = self.robot.data.root_pos_w[:, :3]
        delta_w = curr_robot_pos_w - self._prev_robot_pos_w  # [E,3] world delta since last frame
        self.relative_movement[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), delta_w
        )

        self._prev_robot_pos_w = curr_robot_pos_w.clone()
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

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
class WalkablePose2dCommandCfg(CommandTermCfg):
    """Configuration for walkable-based pose command generator."""
    class_type: type = WalkablePose2dCommand

    asset_name: str = MISSING
    simple_heading: bool = True
    min_goal_sep: float = 0.8
    avoid_prev: bool = True
    num_walkable_positions: int = 125000
    robot_radius: float = 0.5
    
    # Note: Curriculum is handled by the existing curriculum system in cucciculum.py

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (3.0, 5.0)  # Will be updated by curriculum
        pos_y: tuple[float, float] = (0.0, 10.0)  # Will be updated by curriculum
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/walkable_pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
