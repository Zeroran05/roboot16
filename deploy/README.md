# Deploy

This folder mirrors the `unitree_rl_gym/deploy` layout, but keeps the Roboot16 sim2sim flow local to `roboot16`.

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
cd /path/to/IsaacLab
python roboot16/deploy/scripts/export_roboot16_policy_jit.py
```

Or specify paths explicitly:

```bash
python roboot16/deploy/scripts/export_roboot16_policy_jit.py \
  --checkpoint /path/to/IsaacLab/roboot16/logs/rsl_rl/<run_dir>/model_<iter>.pt \
  --output /path/to/IsaacLab/roboot16/deploy/pre_train/roboot16/policy.pt
```

## 2. Check the MuJoCo model

```bash
python roboot16/deploy/scripts/check_model.py --no-viewer
```

To open the viewer on macOS:

```bash
mjpython roboot16/deploy/scripts/check_model.py
```

## 3. Run MuJoCo sim2sim

```bash
cd /path/to/IsaacLab
python roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16.py roboot16.yaml
```
