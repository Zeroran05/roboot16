# Deploy

This folder mirrors the `unitree_rl_gym/deploy` layout, but keeps the Roboot16 sim2sim flow local to `roboot16_amp_project`.

## Layout

```text
deploy/
├── deploy_mujoco/
│   ├── configs/
│   │   └── roboot16.yaml
│   └── deploy_mujoco_roboot16.py
├── scripts/
│   ├── check_model.py
│   ├── common.py
│   └── export_roboot16_policy_jit.py
└── pre_train/
    └── roboot16/
        └── README.md
```

## 1. Export a deployable policy

The MuJoCo deploy script expects a TorchScript / JIT policy file.

You can export one from an RSL-RL checkpoint with the local deploy helper:

```bash
cd /Users/zhuran/IsaacLab
python roboot16_amp_project/deploy/scripts/export_roboot16_policy_jit.py
```

Or specify paths explicitly:

```bash
python roboot16_amp_project/deploy/scripts/export_roboot16_policy_jit.py \
  --checkpoint /Users/zhuran/IsaacLab/roboot16_amp_project/logs/rsl_rl/roboot16_flat_project/2026-05-04_00-28-57/model_50.pt \
  --output /Users/zhuran/IsaacLab/roboot16_amp_project/deploy/pre_train/roboot16/policy.pt
```

## 2. Check the MuJoCo model

```bash
python roboot16_amp_project/deploy/scripts/check_model.py --no-viewer
```

To open the viewer on macOS:

```bash
mjpython roboot16_amp_project/deploy/scripts/check_model.py
```

## 3. Run MuJoCo sim2sim

```bash
cd /Users/zhuran/IsaacLab
python roboot16_amp_project/deploy/deploy_mujoco/deploy_mujoco_roboot16.py roboot16.yaml
```
