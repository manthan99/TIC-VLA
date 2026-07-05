from isaaclab.utils import configclass
import rl.env.navigation_mdp as mdp
from rl.env.navigation_mdp.walkable_command import WalkablePose2dCommandCfg
from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
import math

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pose_command = WalkablePose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(1e12, 1e12),  # Never resample during episodes
        debug_vis=False,
        num_walkable_positions=25000,
        robot_radius=0.5,
        min_goal_sep=1.5,  # Force goals to be at least 1.5m away (was 0.8m default)
        # Note: Curriculum ranges will be updated by cucciculum.py
        ranges=WalkablePose2dCommandCfg.Ranges(
            pos_x=(-1.0, 10.0), 
            pos_y=(-10.0, 10.0), 
            heading=(-math.pi, math.pi)
        ),
    )

    # pose_command = UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=True,
    #     resampling_time_range=(1e12, 1e12),  # Never resample during episodes
    #     debug_vis=False,
    #     ranges=UniformPose2dCommandCfg.Ranges(
    #         pos_x=(15.0, 15.1),
    #         pos_y=(-5.0, -4.9),
    #         heading=(0.0, 0.0)
    #     ),
    # )