from __future__ import annotations

from pathlib import Path

from roboot16_amp_project.paths import AMP_EXPERT_DIR


AMP_TASK_REWARD_LERP = 0.7 #0.7
AMP_DATASET_ROOT = AMP_EXPERT_DIR
AMP_SPEED_CONDITIONING_TAU = 0.20
# Treat only exact zero-speed commands as stand sampling, with a tiny tolerance
# to avoid floating-point edge cases in the command tensor.
AMP_STAND_ONLY_SPEED_THRESHOLD = 1.0e-6


def _collect_amp_txt_datasets(root: Path) -> list[str]:
    datasets: list[str] = []
    for group in ["stand", "walk", "run", "sprint"]:
        group_dir = root / group
        datasets.extend(
            f"{group}/{path.stem}"
            for path in sorted(group_dir.glob("*.txt"))
        )
    return datasets


AMP_DATASETS = _collect_amp_txt_datasets(AMP_DATASET_ROOT)
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
        "speed_conditioning_tau": AMP_SPEED_CONDITIONING_TAU,
        "stand_only_speed_threshold": AMP_STAND_ONLY_SPEED_THRESHOLD,
    },
}
