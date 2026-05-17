# Deploy

This directory keeps the local Roboot16 deployment flow inside `roboot16/`.

## Layout

```text
deploy/
├── deploy_mujoco/
│   ├── configs/
│   │   └── roboot16.yaml
│   ├── deploy_mujoco_roboot16.py
│   ├── deploy_mujoco_roboot16_mimic.py
│   └── sim2sim.py
├── pre_train/
│   ├── roboot16/                  # TorchScript policies for classic deploy flow
│   └── roboot16_mimic/            # ONNX policies for mimic deploy flow
└── scripts/
    ├── check_model.py
    ├── common.py
    └── export_roboot16_policy_jit.py
```

## 1. Export a Deployable Policy

Classic MuJoCo deploy expects a TorchScript policy:

```bash
cd /path/to/IsaacLab
python roboot16/deploy/scripts/export_roboot16_policy_jit.py \
  --checkpoint /path/to/IsaacLab/roboot16/logs/rsl_rl/<run>/model_<iter>.pt \
  --output /path/to/IsaacLab/roboot16/deploy/pre_train/roboot16/policy.pt
```

For mimic policies, the easier path is usually to use the Isaac Lab `play.py` entry point, which exports
`policy.onnx` with motion metadata into the run's `exported/` folder.

## 2. Check the MuJoCo Model

Headless:

```bash
python roboot16/deploy/scripts/check_model.py --no-viewer
```

With viewer on macOS:

```bash
mjpython roboot16/deploy/scripts/check_model.py
```

## 3. Run Classic sim2sim

```bash
cd /path/to/IsaacLab
python roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16.py roboot16.yaml
```

The positional argument is resolved from `deploy/deploy_mujoco/configs/`.

## 4. Run Mimic sim2sim

```bash
cd /path/to/IsaacLab
mjpython roboot16/deploy/deploy_mujoco/deploy_mujoco_roboot16_mimic.py \
  --onnx roboot16/deploy/pre_train/roboot16_mimic/policy.onnx \
  --motion-npz roboot16/data/mimic/walk1_subject1_test1_30s.npz \
  --model roboot16/assets/Roboot1.6/xml/scene_1.xml
```

Useful optional flags:

- `--camera-follow`
- `--show-reference-frames`
- `--show-reference-robot`
- `--reference-robot-offset-y 1.0`
- `--reference-robot-alpha 0.35`
- `--time-step-start <index>`

## 5. Current Artifact Convention

- Classic deploy policy: `deploy/pre_train/roboot16/policy.pt`
- Mimic deploy policy: `deploy/pre_train/roboot16_mimic/policy.onnx`
- Default MuJoCo scene: `assets/Roboot1.6/xml/scene_1.xml`
