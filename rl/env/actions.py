"""
Action terms for mobile navigation environment.
"""
import torch
from dataclasses import dataclass
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import rl.env.navigation_mdp as nmdp
from rl.env.navigation_mdp.coco_action import ClassicalCarActionCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()


class CrowdStepper(ActionTerm):
    """Steps pedestrians & authoring every physics substep."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._throttle = getattr(cfg, "author_every_n_substeps", 3)

    @property
    def action_dim(self): 
        return 0

    @property
    def raw_actions(self):
        return torch.zeros((self._env.num_envs, 0), device=self._env.device)

    @property
    def processed_actions(self):
        return self.raw_actions

    def process_actions(self, actions): 
        pass

    def apply_actions(self):
        dt = self._env.physics_dt 
        self._env._step_people(dt)

        # Authoring while _prev_people_xy still refers to the previous frame
        if (self._env._sim_step_counter % max(1, self._throttle)) == 0:
            self._env._apply_people_to_stage()


@dataclass
class CrowdStepperCfg(ActionTermCfg):
    class_type: type = CrowdStepper
    asset_name: str = "terrain"    
    author_every_n_substeps: int = 3

#========================================== END OF CROWD STEPPER ===========================================

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # coco one robot action configuration
    joint_pos = ClassicalCarActionCfg(asset_name="robot")

    # unitree go2 robot action configuration
    # pretrained_policy: nmdp.PreTrainedPolicyActionCfg = nmdp.PreTrainedPolicyActionCfg(
    #     asset_name="robot",
    #     policy_path="/path/to/exported/policy.pt",
    #     low_level_decimation=4,
    #     low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
    #     low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    #     debug_vis=False,
    # )

    # people's action configuration
    #crowd_stepper = CrowdStepperCfg(asset_name="terrain") # disable crowd stepper for faster training