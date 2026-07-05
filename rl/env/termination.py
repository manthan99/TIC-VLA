import rl.env.navigation_mdp as nav_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loc_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import torch
from typing import Sequence


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=loc_mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=nav_mdp.illegal_contact, time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body_link"), # change path to your robot
            "threshold": 1.0,
            "grace_steps": 20,       # <-- ignore first 20 steps
            "sustain_steps": 2,   
            },
    )

    # hit_pedestrian = DoneTerm(
    #     func=nav_mdp.hit_pedestrian,
    #     time_out=False,
    #     params={"robot_radius": 0.3, "person_radius": 0.5},
    # )

    # collision = DoneTerm(
    #     func=nav_mdp.illegal_contact, time_out=False,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), # change path to your robot
    #         "threshold": 1.0,
    #         "grace_steps": 20,       # <-- ignore first 20 steps
    #         "sustain_steps": 2, 
    #     },
    # )


    arrive = DoneTerm(
        func=nav_mdp.arrive, time_out=False,
        params={"threshold": 1.5, "command_name": "pose_command"},
    )
    # upside_down = DoneTerm(
    #     func=nav_mdp.upside_down, time_out=False,
    #     params={"threshold": 0.5, 
    #             "asset_cfg": SceneEntityCfg("robot"),
    #             "grace_steps": 50,
    #             "sustain_steps": 3,
    #         },
    # )
    # stuck = DoneTerm(
    #     func=nav_mdp.stuck, time_out=False,
    #     params={"min_velocity": 0.1, "window_steps": 50, "grace_period_steps": 100, "asset_cfg": SceneEntityCfg("robot")},
    # )