from __future__ import annotations

from roboot16_amp_project.paths import AMP_EXPERT_DIR


AMP_TASK_REWARD_LERP = 0.5 #0.7
AMP_DATASET_ROOT = AMP_EXPERT_DIR

AMP_HIGH_SPEED_DATASETS = [
    "run_high_speed/run1_5",
    "run_high_speed/run1_5_mirrored",
    "run_high_speed/run1_7",
    "run_high_speed/run1_7_mirrored",
    "run_high_speed/run2_7",
    "run_high_speed/run2_7_mirrored",
    "run_high_speed/run2_10",
    "run_high_speed/run2_10_mirrored",
]

AMP_LOW_SPEED_DATASETS = [
    "run_low_speed/run1_1",
    "run_low_speed/run1_1_mirrored",
    "run_low_speed/run1_3",
    "run_low_speed/run1_3_mirrored",
    "run_low_speed/run1_4",
    "run_low_speed/run1_4_mirrored",
    "run_low_speed/run1_6",
    "run_low_speed/run1_6_mirrored",
    "run_low_speed/run2_1",
    "run_low_speed/run2_1_mirrored",
    "run_low_speed/run2_2",
    "run_low_speed/run2_2_mirrored",
    "run_low_speed/run2_3",
    "run_low_speed/run2_3_mirrored",
    "run_low_speed/run2_4",
    "run_low_speed/run2_4_mirrored",
    "run_low_speed/run2_5",
    "run_low_speed/run2_5_mirrored",
    "run_low_speed/run2_6",
    "run_low_speed/run2_6_mirrored",
    "run_low_speed/run2_8",
    "run_low_speed/run2_8_mirrored",
    "run_low_speed/run2_9",
    "run_low_speed/run2_9_mirrored",
    "run_low_speed/stand",
    "run_low_speed/stand_mirrored",
]

AMP_DATASETS = AMP_HIGH_SPEED_DATASETS + AMP_LOW_SPEED_DATASETS
AMP_DATASET_WEIGHT = 1.0 / len(AMP_DATASETS)


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
        "reward_scale": 1,# 0.3，奖励太小，提高奖励scale
        "loss_type": "LSGAN",
        "empirical_normalization": True,
    },
    "dataset": {
        "amp_data_path": str(AMP_DATASET_ROOT),
        "datasets": {name: AMP_DATASET_WEIGHT for name in AMP_DATASETS},
        "slow_down_factor": 1,
    },
}
