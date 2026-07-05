# rl/env/register_envs.py
import gymnasium as gym
from gymnasium.error import Error as GymError

def _register_once():
    try:
        gym.spec("Isaac-Navigation-TICVLA-COCO")
    except GymError:
        gym.register(
            id="Isaac-Navigation-TICVLA-COCO",
            entry_point="rl.env.env:TICVLANavEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "rl.env.config:TICVLANavEnvCfg",
                "rl_games_cfg_entry_point": "rl.env.agent:rlg_cnn_config.yaml",
                "rsl_rl_cfg_entry_point": "rl.env.agent.train_nav_rsl_rl:TICVLANavEnvPPORunnerCfg",
                "skrl_cfg_entry_point": "rl.env.agent:skrl_nav.yaml",
            },
        )

_register_once()