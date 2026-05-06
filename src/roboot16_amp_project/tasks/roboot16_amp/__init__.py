import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Roboot16-AMP-Flat-Project-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Roboot16AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16AmpFlatPPORunnerCfg",
        "amp_rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:Roboot16FlatAMPRunnerCfg",
    },
)

gym.register(
    id="Isaac-Roboot16-AMP-Flat-Project-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Roboot16AmpFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16AmpFlatPPORunnerCfg",
        "amp_rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:Roboot16FlatAMPRunnerCfg",
    },
)

gym.register(
    id="Isaac-Roboot16-AMP-Flat-HighSpeed-Project-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.high_speed_env_cfg:Roboot16AmpHighSpeedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16AmpFlatPPORunnerCfg",
        "amp_rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:Roboot16FlatAMPRunnerCfg",
    },
)

gym.register(
    id="Isaac-Roboot16-AMP-Flat-HighSpeed-Project-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.high_speed_env_cfg:Roboot16AmpHighSpeedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Roboot16AmpFlatPPORunnerCfg",
        "amp_rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:Roboot16FlatAMPRunnerCfg",
    },
)
