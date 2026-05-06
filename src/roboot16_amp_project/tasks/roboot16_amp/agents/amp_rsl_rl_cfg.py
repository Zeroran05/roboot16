from __future__ import annotations

from roboot16_amp_project.paths import AMP_EXPERT_DIR


AMP_TASK_REWARD_LERP = 0.78 #0.7
AMP_HIGH_SPEED_DIR = AMP_EXPERT_DIR / "run_high_speed"


Roboot16FlatAMPRunnerCfg = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 5000,
    "save_interval": 50,
    "experiment_name": "roboot16_amp_flat_project",
    "run_name": "run_high_speed",
    "empirical_normalization": False,
    "logger": "tensorboard",
    "amp_task_reward_lerp": AMP_TASK_REWARD_LERP,
    "obs_groups": {
        "policy": ["policy"],
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "AMP_PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "amp_replay_buffer_size": 100000,
        "use_smooth_ratio_clipping": False,
    },
    "discriminator": {
        "hidden_dims": [1024, 512, 256],
        "reward_scale": 0.3,
        "loss_type": "LSGAN",
        "empirical_normalization": True,
    },
    "dataset": {
        "amp_data_path": str(AMP_HIGH_SPEED_DIR),
        "datasets": {
            "run1_5": 0.125,
            "run1_5_mirrored": 0.125,
            "run1_7": 0.125,
            "run1_7_mirrored": 0.125,
            "run2_7": 0.125,
            "run2_7_mirrored": 0.125,
            "run2_10": 0.125,
            "run2_10_mirrored": 0.125,
        },
        "slow_down_factor": 1,
    },
}
