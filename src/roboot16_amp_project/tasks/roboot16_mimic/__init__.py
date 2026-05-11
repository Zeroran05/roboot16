import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Roboot16-Mimic-Project-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:Roboot16MimicEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:Roboot16MimicPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16MimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Roboot16-Mimic-Project-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:Roboot16MimicPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16MimicPPORunnerCfg",
    },
)
