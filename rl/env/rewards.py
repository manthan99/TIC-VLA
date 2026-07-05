"""
Reward and termination terms for mobile navigation environment.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
import rl.env.navigation_mdp as nav_mdp
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    arrived_reward = RewTerm(
        mdp.is_terminated_term, weight=400.0, params={"term_keys": "arrive"}
    )
    collision_penalty = RewTerm(func=mdp.is_terminated_term, weight=-100, params={"term_keys": "collision"})
    hit_pedestrian_penalty = RewTerm(
        func=nav_mdp.hit_pedestrian,
        weight=-100.0,
        params={"robot_radius": 0.3, "person_radius": 0.5},
    )
    # upside_down_penalty = RewTerm(
    #     func=nav_mdp.upside_down_penalty,
    #     weight=-70, 
    #     params={"threshold": 0.5},
    # )
    #time_out_penalty = RewTerm(mdp.is_terminated_term, weight=-100, params={"term_keys": "time_out", "ignore_time_out": False})
    # stuck_penalty = RewTerm(mdp.is_terminated_term, weight=-40, params={"term_keys": "stuck"})
    # time_based_penalty = RewTerm(
    #     func=nav_mdp.time_based_penalty,
    #     weight=-0.1,
    #     params={"command_name": "pose_command"}
    # )
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh_adaptive,
        weight=1.0, 
        params={"command_name": "pose_command", "k": 0.3, "std_min": 1.0, "std_max": 9.0},  
    )
    speed_matching = RewTerm(
        func=nav_mdp.speed_matching_penalty,
        weight=-0.1,
        params={"target_speed": 1.0}
    )
    progress = RewTerm(
        func=nav_mdp.progress_reward,
        weight=5.0,
        params={"command_name": "pose_command", "normalize_by_init": True, "scale": 100.0},
    )
    
    # # Penalize dithering/standing still (weight is NEGATIVE)
    # no_progress = RewTerm(
    #     func=nav_mdp.no_progress_penalty,
    #     weight=-1,  # tune: -0.2 to -1.0; stronger pushes against idling
    #     params={"command_name": "pose_command", "window": 10, "tol": 0.02, "normalize_by_init": True},
    # )
    # instant_no_move = RewTerm(
    #     func=nav_mdp.instant_idle_penalty,
    #     weight=-0.4,  # small but immediate; tune -0.2..-0.6
    #     params={"command_name": "pose_command", "prog_thresh": 0.002, "normalize_by_init": True},
    # )

    # Encourage tracking of VLM-provided waypoints with latency compensation
    # vlm_waypoint_tracking = RewTerm(
    #     func=nav_mdp.vlm_waypoint_tracking_error,
    #     weight=10.0,
    #     params={"std": 1.5, "clamp_min_std": 0.5},
    # )