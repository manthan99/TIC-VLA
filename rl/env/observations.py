from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
import rl.env.navigation_mdp as nav_mdp
import isaaclab.envs.mdp as mdp

@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        pose_command = ObsTerm(func=nav_mdp.advanced_generated_commands,
                               params={"command_name": "pose_command", "max_dim": 2, "normalize": True})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        #projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        rgb = ObsTerm(func=nav_mdp.image_processed, params={"sensor_cfg": SceneEntityCfg("camera")})

        def __post_init__(self):
            # IMPORTANT: returns a dict with the four terms, not a concatenated tensor
            self.concatenate_terms = False

    policy: Policy = Policy()
